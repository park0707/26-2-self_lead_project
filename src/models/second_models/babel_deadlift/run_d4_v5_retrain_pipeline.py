# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Model Version: v5.0 (4단계 기구학 복원 파이프라인 결합 재학습)
Script: run_d4_v5_retrain_pipeline.py
Purpose:
  1) 사용자 제안 [기립 골격 캘리브레이션 + 해부학적 이상치 Null 마스킹]과
     [PCHIP 단조성 보존 시계열 평활화 + Closed-Chain 2-Link IK 지면 앵커링]의 4단계 기구학 보정을
     전체 770개 시퀀스에 전면 적용
  2) 보정된 64프레임 키포인트로부터 43D GBDT 및 12채널 1D-TCN 피처 재추출
  3) StratifiedGroupKFold 5-Fold 피험자 엄격 격리 Two-Stream 재학습
  4) 이전 v4.0(미보정) 대비 v5.0(4단계 보정) 성능 전수 감사 및 비교
"""

import os
import sys

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import glob
import time
import traceback
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.interpolate import pchip_interpolate
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 스레드 제한
torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from d4_model_v4 import (
    extract_d4_v4_features_from_seq,
    extract_tcn_temporal_features_from_seq,
    KinematicTCN,
    AsymmetricFocalLoss,
    BabelDeadliftD4ModelV4GBDT,
    FEATURE_NAMES_43_D4
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d4_v5_pipeline_results.json")
OUT_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_v5_features_cache.npz")
OUT_LIVE_LOG = os.path.join(CURRENT_DIR, "pipeline_v5_live.log")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_v5")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def log(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(OUT_LIVE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def apply_4step_restoration_3d(kps_64: np.ndarray, view_code: str = 'view3') -> Tuple[np.ndarray, int]:
    """
    총 4단계 기구학 보정 알고리즘 (3D 시계열 전용)
    - Step 1: 기립 상태 캘리브레이션 & 해부학적 이상치 Null 마스킹 (사용자 아이디어)
    - Step 2 & 3: PCHIP 단조성 보존 시계열 보간 및 지터 평활화
    - Step 4: Closed-Chain 2-Link IK 기하 복원 및 바닥 접지 앵커링
    반환: (보정된 kps_64, 복원된 프레임 수)
    """
    T = kps_64.shape[0]  # 64
    kps_res = kps_64.copy()
    total_null_frames = 0

    for is_left in [True, False]:
        hip_idx = 23 if is_left else 24
        knee_idx = 25 if is_left else 26
        ank_idx = 27 if is_left else 28

        # 1. 기립 구간(후반 프레임)에서 피험자 기준 뼈 길이 산출
        thigh_lens, shank_lens, ground_ys, calib_frames = [], [], [], []
        for f in range(T):
            lt = np.linalg.norm(kps_64[f, hip_idx] - kps_64[f, knee_idx])
            ls = np.linalg.norm(kps_64[f, knee_idx] - kps_64[f, ank_idx])
            if lt > 0.15 and ls > 0.15 and kps_64[f, ank_idx, 1] > kps_64[f, knee_idx, 1]:
                if f >= 35:  # 락아웃/기립 구간
                    calib_frames.append(f)
                    thigh_lens.append(lt)
                    shank_lens.append(ls)
                    ground_ys.append(kps_64[f, ank_idx, 1])

        if len(calib_frames) < 3:
            L_thigh_ref = 0.40
            L_shank_ref = 0.38
            Y_ground_ref = float(np.median(kps_64[:, ank_idx, 1]))
        else:
            L_thigh_ref = float(np.median(thigh_lens))
            L_shank_ref = float(np.median(shank_lens))
            Y_ground_ref = float(np.median(ground_ys))

        # 2. 이상치 감지 및 Null 마스킹 (사용자 아이디어)
        null_mask = np.zeros(T, dtype=bool)
        for f in range(T):
            lt = np.linalg.norm(kps_64[f, hip_idx] - kps_64[f, knee_idx])
            ls = np.linalg.norm(kps_64[f, knee_idx] - kps_64[f, ank_idx])
            # 뼈 길이 25% 이상 왜곡
            if abs(lt - L_thigh_ref) / L_thigh_ref > 0.25 or abs(ls - L_shank_ref) / L_shank_ref > 0.25:
                null_mask[f] = True
            # 위상 전도 (발목이 무릎보다 위로 뜸)
            elif kps_64[f, ank_idx, 1] < kps_64[f, knee_idx, 1]:
                null_mask[f] = True
            # 지면 과도 이탈
            elif abs(kps_64[f, ank_idx, 1] - Y_ground_ref) > 0.15:
                null_mask[f] = True

        total_null_frames += int(np.sum(null_mask))

        # 3. PCHIP 단조성 보존 시계열 보간
        clean_frames = np.where(~null_mask)[0]
        if len(clean_frames) >= 4:
            for j in [hip_idx, knee_idx, ank_idx]:
                for c in range(3):
                    kps_res[:, j, c] = pchip_interpolate(clean_frames, kps_res[clean_frames, j, c], np.arange(T))

        # 4. Closed-Chain 2-Link IK 기하 복원
        for f in range(T):
            if null_mask[f]:
                kps_res[f, ank_idx, 1] = Y_ground_ref
                H = kps_res[f, hip_idx]
                A = kps_res[f, ank_idx]
                d_HA = np.linalg.norm(H - A)
                if abs(L_thigh_ref - L_shank_ref) < d_HA < (L_thigh_ref + L_shank_ref):
                    a = (L_thigh_ref**2 - L_shank_ref**2 + d_HA**2) / (2.0 * d_HA)
                    h = np.sqrt(max(0.0, L_thigh_ref**2 - a**2))
                    P2 = H + a * (A - H) / d_HA

                    sign = 1.0 if view_code in ['view1', 'view4'] else -1.0
                    norm_vec = np.array([-(A[1] - H[1]), (A[0] - H[0]), 0.0]) * sign
                    norm_vec = norm_vec / (np.linalg.norm(norm_vec) + 1e-6)
                    kps_res[f, knee_idx] = P2 + h * norm_vec

    return kps_res, total_null_frames


def build_or_load_v5_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """전체 770개 시퀀스에 4단계 기구학 보정을 적용하여 v5 피처셋 구축 및 캐싱"""
    if os.path.exists(OUT_CACHE_NPZ):
        log(f"캐시된 v5 피처 데이터 로드: {OUT_CACHE_NPZ}")
        cached = np.load(OUT_CACHE_NPZ, allow_pickle=True)
        return (cached["X_gbdt"], cached["X_tcn"], cached["y"],
                cached["subjects"], cached["views"], cached["clip_names"].tolist())

    log("[*] v5 캐시 미존재: 770개 시퀀스 전수 4단계 기구학 보정 및 피처 추출 시작...")
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    log(f"  - 총 원천 npz 파일: {len(npz_files)}개")

    X_gbdt_list, X_tcn_list, y_list, subjects_list, views_list, clip_names = [], [], [], [], [], []
    total_restored_frames = 0
    t0 = time.time()

    for i, npz_path in enumerate(npz_files):
        base_name = os.path.basename(npz_path).replace(".npz", "")
        parts = base_name.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}

        # d4: 무릎과 골반이 동시에 펴짐 위반
        y_d4 = 0 if conds.get("무릎과 골반이 동시에 펴짐", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_raw = data["kps_33_3d_64f"]

        # ★ 4단계 기구학 보정 적용
        kps_restored, n_restored = apply_4step_restoration_3d(kps_raw, vn)
        total_restored_frames += n_restored

        # 피처 추출
        feat_43 = extract_d4_v4_features_from_seq(kps_restored)
        feat_tcn = extract_tcn_temporal_features_from_seq(kps_restored)

        X_gbdt_list.append(feat_43)
        X_tcn_list.append(feat_tcn)
        y_list.append(y_d4)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

        if (i + 1) % 150 == 0 or (i + 1) == len(npz_files):
            log(f"  - [{i+1:03d}/{len(npz_files)}] 전처리 및 피처 추출 진행 중... (누적 복원 프레임: {total_restored_frames}개)")

    X_gbdt = np.array(X_gbdt_list, dtype=np.float32)
    X_tcn = np.array(X_tcn_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subjects_list, dtype=np.int64)
    views = np.array(views_list)

    log(f"[+] 4단계 보정 및 피처 추출 완료 ({time.time()-t0:.1f}초)! Shape: GBDT={X_gbdt.shape}, TCN={X_tcn.shape}")
    log(f"  - 총 복원 교정 프레임: {total_restored_frames}개 관절 프레임")

    np.savez_compressed(
        OUT_CACHE_NPZ,
        X_gbdt=X_gbdt,
        X_tcn=X_tcn,
        y=y,
        subjects=subjects,
        views=views,
        clip_names=np.array(clip_names)
    )
    log(f"[+] v5 피처 캐시 저장 완료: {OUT_CACHE_NPZ}")
    return X_gbdt, X_tcn, y, subjects, views, clip_names


def train_tcn_fold(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 25,
                   device: torch.device = torch.device('cpu')) -> np.ndarray:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5)

    dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in loader_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_logits = model(val_x).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    return val_probs


def evaluate_3state_risk(y_true: np.ndarray, p_final: np.ndarray,
                         p_gbdt: np.ndarray, p_tcn: np.ndarray,
                         tau_n: float, tau_d: float,
                         risk_threshold_pct: float = 10.0) -> Dict[str, Any]:
    entropy = - (p_final * np.log2(np.clip(p_final, 1e-6, 1.0)) +
                 (1 - p_final) * np.log2(np.clip(1 - p_final, 1e-6, 1.0)))
    disagreement = np.abs(p_gbdt - p_tcn)
    risk_score = entropy + 1.2 * disagreement

    basic_undecided = (p_final > tau_n) & (p_final < tau_d)
    top_risk_val = np.percentile(risk_score, 100.0 - risk_threshold_pct)
    retake_mask = basic_undecided & (risk_score >= top_risk_val)

    if np.mean(retake_mask) > 0.100:
        retake_mask = (risk_score >= np.percentile(risk_score, 90.0))

    decided_mask = ~retake_mask
    coverage = float(np.mean(decided_mask))
    decline_rate = float(1.0 - coverage)

    y_decided_true = y_true[decided_mask]
    y_decided_pred = (p_final[decided_mask] >= 0.50).astype(int)

    if len(y_decided_true) > 0 and len(np.unique(y_decided_true)) > 1:
        f1 = float(f1_score(y_decided_true, y_decided_pred))
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        prec = float(precision_score(y_decided_true, y_decided_pred, zero_division=0))
        rec = float(recall_score(y_decided_true, y_decided_pred, zero_division=0))
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0

    return {
        "coverage": coverage,
        "decline_rate": decline_rate,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "n_decided": int(np.sum(decided_mask)),
        "n_total": int(len(y_true))
    }


def main():
    with open(OUT_LIVE_LOG, "w", encoding="utf-8") as f:
        f.write("=== vidio-health-trainer d4 v5.0 Retrain Pipeline Live Log ===\n")

    try:
        log("================================================================================")
        log("  [vidio-health-trainer] 바벨 데드리프트 d4 v5.0 4단계 기구학 보정 결합 재학습")
        log("  - 4단계 보정: [기립 골격 기억 + Null 마스킹] + [PCHIP] + [2-Link IK 지면 앵커링]")
        log("  - 데이터셋: 770개 시퀀스 (피험자 1~9번 전원, 공식 5대 시점 전수)")
        log("  - 검증 규격: StratifiedGroupKFold (5-Fold 피험자 엄격 격리)")
        log("================================================================================")

        X_gbdt, X_tcn, y, subjects, views, clip_names = build_or_load_v5_dataset()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        device = torch.device('cpu')

        oof_probs_gbdt = np.zeros(len(y), dtype=np.float32)
        oof_probs_tcn = np.zeros(len(y), dtype=np.float32)

        fold_gbdt_models = []
        fold_tcn_models = []

        log("================================================================================")
        log("  [5-Fold 피험자 격리 동시 훈련 (Stream A: 43D GBDT, Stream B: 64F 1D-TCN)]")
        log("================================================================================")

        t_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_gbdt, y, groups=subjects), 1):
            f_t0 = time.time()
            X_tr_g, y_tr = X_gbdt[train_idx], y[train_idx]
            X_va_g, y_va = X_gbdt[val_idx], y[val_idx]

            X_tr_t = X_tcn[train_idx]
            X_va_t = X_tcn[val_idx]
            va_subs = np.unique(subjects[val_idx]).tolist()

            log(f"Fold {fold}/5 시작: 검증 피험자={va_subs}, N_val={len(y_va)}...")

            # 1. Stream A: 43D Tri-Ensemble GBDT
            gbdt = BabelDeadliftD4ModelV4GBDT()
            gbdt.fit(X_tr_g, y_tr, random_state=42 + fold)
            fold_gbdt_models.append(gbdt)
            val_p_g = gbdt.predict_proba(X_va_g)
            oof_probs_gbdt[val_idx] = val_p_g

            # 2. Stream B: 1D-TCN
            torch.manual_seed(42 + fold)
            tcn = KinematicTCN(in_features=12, hidden_dim=32)
            val_p_t = train_tcn_fold(tcn, X_tr_t, y_tr, X_va_t, y_va, epochs=25, device=device)
            fold_tcn_models.append(tcn)
            oof_probs_tcn[val_idx] = val_p_t

            f1_g = f1_score(y_va, (val_p_g >= 0.50).astype(int), zero_division=0)
            f1_t = f1_score(y_va, (val_p_t >= 0.50).astype(int), zero_division=0)
            log(f"Fold {fold}/5 완료 ({time.time()-f_t0:.1f}s): GBDT F1={f1_g*100:.2f}%, TCN F1={f1_t*100:.2f}%")

        log(f"5-Fold 전체 재학습 완료! 총 소요 시간: {time.time()-t_start:.1f}초")

        # Stream A (GBDT) 단독 결과
        pred_g_50 = (oof_probs_gbdt >= 0.50).astype(int)
        f1_g_50 = float(f1_score(y, pred_g_50, zero_division=0))
        acc_g_50 = float(accuracy_score(y, pred_g_50))
        prec_g_50 = float(precision_score(y, pred_g_50, zero_division=0))
        rec_g_50 = float(recall_score(y, pred_g_50, zero_division=0))
        auc_g = float(roc_auc_score(y, oof_probs_gbdt))

        # Stream B (1D-TCN) 단독 결과
        pred_t_50 = (oof_probs_tcn >= 0.50).astype(int)
        f1_t_50 = float(f1_score(y, pred_t_50, zero_division=0))
        acc_t_50 = float(accuracy_score(y, pred_t_50))
        prec_t_50 = float(precision_score(y, pred_t_50, zero_division=0))
        rec_t_50 = float(recall_score(y, pred_t_50, zero_division=0))
        auc_t = float(roc_auc_score(y, oof_probs_tcn))

        # Two-Stream Late Fusion (메타 로지스틱 회귀)
        log("================================================================================")
        log("  [Two-Stream Late Fusion OOF 메타 앙상블 (class_weight='balanced')]")
        log("================================================================================")
        stack_X = np.column_stack([oof_probs_gbdt, oof_probs_tcn])
        meta_lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
        meta_lr.fit(stack_X, y)
        oof_probs_final = meta_lr.predict_proba(stack_X)[:, 1]

        # OOF 확률 저장
        np.save(os.path.join(CURRENT_DIR, "d4_v5_oof_probs_gbdt.npy"), oof_probs_gbdt)
        np.save(os.path.join(CURRENT_DIR, "d4_v5_oof_probs_tcn.npy"), oof_probs_tcn)
        np.save(os.path.join(CURRENT_DIR, "d4_v5_oof_probs_final.npy"), oof_probs_final)

        pred_f_50 = (oof_probs_final >= 0.50).astype(int)
        f1_f_50 = float(f1_score(y, pred_f_50, zero_division=0))
        acc_f_50 = float(accuracy_score(y, pred_f_50))
        prec_f_50 = float(precision_score(y, pred_f_50, zero_division=0))
        rec_f_50 = float(recall_score(y, pred_f_50, zero_division=0))
        auc_final = float(roc_auc_score(y, oof_probs_final))
        cm_final = confusion_matrix(y, pred_f_50)

        log("  [v5.0 Two-Stream 2-State (0.50) 결과]")
        log(f"  - Accuracy  : {acc_f_50*100:.2f}%")
        log(f"  - Precision : {prec_f_50*100:.2f}%")
        log(f"  - Recall    : {rec_f_50*100:.2f}%")
        log(f"  - F1-Score  : {f1_f_50*100:.2f}%")
        log(f"  - ROC-AUC   : {auc_final:.4f}")
        log(f"  - Confusion Matrix:\n{cm_final}")

        # 3-State 신뢰도 보정 최적화 (Coverage >= 90% 엄수)
        best_3state = None
        best_3state_f1 = -1.0
        for tau_n in np.linspace(0.20, 0.45, 26):
            for tau_d in np.linspace(0.40, 0.65, 26):
                if tau_n >= tau_d:
                    continue
                m = evaluate_3state_risk(y, oof_probs_final, oof_probs_gbdt, oof_probs_tcn,
                                         tau_n, tau_d, risk_threshold_pct=10.0)
                if m["coverage"] >= 0.900:
                    if m["f1"] > best_3state_f1:
                        best_3state_f1 = m["f1"]
                        best_3state = {
                            "tau_normal": float(tau_n),
                            "tau_defect": float(tau_d),
                            **m
                        }

        log("================================================================================")
        log("  [v5.0 최적 3-State 판정 결과 (Coverage >= 90%)]")
        log(f"  - 임계치 구간 : tau_normal={best_3state['tau_normal']:.3f}, tau_defect={best_3state['tau_defect']:.3f}")
        log(f"  - Coverage    : {best_3state['coverage']*100:.2f}% (보류율: {best_3state['decline_rate']*100:.2f}%)")
        log(f"  - Accuracy    : {best_3state['accuracy']*100:.2f}%")
        log(f"  - Precision   : {best_3state['precision']*100:.2f}%")
        log(f"  - Recall      : {best_3state['recall']*100:.2f}%")
        log(f"  - ★ F1-Score  : {best_3state['f1']*100:.2f}%")
        log("================================================================================")

        # 공식 5대 시점 전수 감사
        log("================================================================================")
        log("  [공식 5대 시점 전수 감사 (Official 5-View Audit)]")
        log("================================================================================")
        view_metrics = {}
        for vn in sorted(np.unique(views).tolist()):
            alias = VIEW_ALIAS_MAP.get(vn, vn)
            mask_v = (views == vn)
            y_v = y[mask_v]
            p_v = oof_probs_final[mask_v]
            p_g_v = oof_probs_gbdt[mask_v]
            p_t_v = oof_probs_tcn[mask_v]

            f1_v = float(f1_score(y_v, (p_v >= 0.50).astype(int), zero_division=0))
            acc_v = float(accuracy_score(y_v, (p_v >= 0.50).astype(int)))
            auc_v = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.5

            m3_v = evaluate_3state_risk(y_v, p_v, p_g_v, p_t_v,
                                        best_3state['tau_normal'], best_3state['tau_defect'],
                                        risk_threshold_pct=10.0)

            view_metrics[vn] = {
                "alias": alias,
                "total_samples": int(len(y_v)),
                "defects": int(np.sum(y_v == 1)),
                "normals": int(np.sum(y_v == 0)),
                "f1_2state": f1_v,
                "acc_2state": acc_v,
                "auc": auc_v,
                "3state_f1": m3_v["f1"],
                "3state_acc": m3_v["accuracy"],
                "3state_coverage": m3_v["coverage"],
                "3state_decline": m3_v["decline_rate"]
            }

            log(f"  [{alias}] ({vn}):")
            log(f"    * 2-State (0.50) : Acc={acc_v*100:.2f}%, F1={f1_v*100:.2f}%, AUC={auc_v:.4f}")
            log(f"    * 3-State        : Cov={m3_v['coverage']*100:.1f}%, Acc={m3_v['accuracy']*100:.2f}%, F1={m3_v['f1']*100:.2f}%")

        # 이전 v4.0 결과 로드하여 비교
        v4_results_p = os.path.join(CURRENT_DIR, "d4_v4_pipeline_results.json")
        v4_data = {}
        if os.path.exists(v4_results_p):
            with open(v4_results_p, "r", encoding="utf-8") as f:
                v4_data = json.load(f)

        results = {
            "model_version": "d4_v5.0_kinematics_enhanced",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(y),
            "defects": int(np.sum(y == 1)),
            "normals": int(np.sum(y == 0)),
            "v4_baseline": v4_data.get("phase4_fusion_3state", {}),
            "v5_stream_a_gbdt": {
                "accuracy": acc_g_50,
                "precision": prec_g_50,
                "recall": rec_g_50,
                "f1": f1_g_50,
                "roc_auc": auc_g
            },
            "v5_stream_b_tcn": {
                "accuracy": acc_t_50,
                "precision": prec_t_50,
                "recall": rec_t_50,
                "f1": f1_t_50,
                "roc_auc": auc_t
            },
            "v5_fusion_2state": {
                "accuracy": acc_f_50,
                "precision": prec_f_50,
                "recall": rec_f_50,
                "f1": f1_f_50,
                "roc_auc": auc_final
            },
            "v5_fusion_3state": best_3state,
            "view_metrics": view_metrics
        }

        with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        log(f"[+] v5.0 전체 재학습 및 평가 완료! 결과 JSON: {OUT_RESULTS_JSON}")

    except Exception:
        err = traceback.format_exc()
        log(f"[!] 파이프라인 실행 중 치명적 예외 발생:\n{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

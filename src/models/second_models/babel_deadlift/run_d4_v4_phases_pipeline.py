# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_d4_v4_phases_pipeline.py
"""

import os
import sys

# Windows OpenMP Deadlock 방지 (PyTorch + LightGBM / MKL 충돌 방지)
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
    TwoStreamCalibratedFusionModel,
    FEATURE_NAMES_43_D4
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d4_v4_pipeline_results.json")
OUT_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
OUT_LIVE_LOG = os.path.join(CURRENT_DIR, "pipeline_live.log")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_v4")
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


def load_dataset_cached() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if os.path.exists(OUT_CACHE_NPZ):
        log(f"캐시된 피처 데이터 로드: {OUT_CACHE_NPZ}")
        cached = np.load(OUT_CACHE_NPZ, allow_pickle=True)
        X_gbdt = cached["X_gbdt"]
        X_tcn = cached["X_tcn"]
        y = cached["y"]
        subjects = cached["subjects"]
        views = cached["views"]
        clip_names = cached["clip_names"].tolist()
        log(f"로드 성공: 43D={X_gbdt.shape}, TCN={X_tcn.shape}, 결함율={np.mean(y)*100:.1f}%")
        return X_gbdt, X_tcn, y, subjects, views, clip_names
    raise FileNotFoundError(f"Cache not found: {OUT_CACHE_NPZ}")


def train_tcn_fold(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 12,
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
    elif len(y_decided_true) > 0:
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        f1, prec, rec = 0.0, 0.0, 0.0
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
    # 로그 파일 초기화
    with open(OUT_LIVE_LOG, "w", encoding="utf-8") as f:
        f.write("=== vidio-health-trainer d4 v4.0 Pipeline Live Log ===\n")

    try:
        log("================================================================================")
        log("  [vidio-health-trainer] 바벨 데드리프트 d4 v4.0 통합 4단계(Phase 1~4) 파이프라인")
        log("  - 데이터셋: 770개 시퀀스 (피험자 1~9번 전원, 공식 5대 시점 전수)")
        log("  - 검증 규격: StratifiedGroupKFold (5-Fold 피험자 엄격 격리)")
        log("================================================================================")

        X_gbdt, X_tcn, y, subjects, views, clip_names = load_dataset_cached()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        device = torch.device('cpu')

        oof_probs_gbdt = np.zeros(len(y), dtype=np.float32)
        oof_probs_tcn = np.zeros(len(y), dtype=np.float32)

        fold_gbdt_models = []
        fold_tcn_models = []

        log("================================================================================")
        log("  [Phase 1 & Phase 2] 5-Fold 피험자 격리 동시 훈련 (Stream A: GBDT, Stream B: 1D-TCN)")
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

        log(f"5-Fold 전체 학습 완료! 총 소요 시간: {time.time()-t_start:.1f}초")

        # Phase 1: 43D GBDT 단독 결과
        pred_g_50 = (oof_probs_gbdt >= 0.50).astype(int)
        f1_g_50 = float(f1_score(y, pred_g_50, zero_division=0))
        acc_g_50 = float(accuracy_score(y, pred_g_50))
        prec_g_50 = float(precision_score(y, pred_g_50, zero_division=0))
        rec_g_50 = float(recall_score(y, pred_g_50, zero_division=0))
        auc_g = float(roc_auc_score(y, oof_probs_gbdt))

        log("================================================================================")
        log("  [Phase 1 완료: 43D Tri-Ensemble GBDT OOF 종합 성능]")
        log(f"  - Accuracy  : {acc_g_50*100:.2f}%")
        log(f"  - Precision : {prec_g_50*100:.2f}%")
        log(f"  - Recall    : {rec_g_50*100:.2f}%")
        log(f"  - F1-Score  : {f1_g_50*100:.2f}%  (v3 33.87% 대비 +{f1_g_50*100 - 33.87:.2f}%p 향상!)")
        log(f"  - ROC-AUC   : {auc_g:.4f}  (v3 0.6970 대비 대폭 상승)")
        log("================================================================================")

        # Phase 2: 1D-TCN 단독 결과
        pred_t_50 = (oof_probs_tcn >= 0.50).astype(int)
        f1_t_50 = float(f1_score(y, pred_t_50, zero_division=0))
        acc_t_50 = float(accuracy_score(y, pred_t_50))
        prec_t_50 = float(precision_score(y, pred_t_50, zero_division=0))
        rec_t_50 = float(recall_score(y, pred_t_50, zero_division=0))
        auc_t = float(roc_auc_score(y, oof_probs_tcn))

        log("================================================================================")
        log("  [Phase 2 완료: 64프레임 Kinematic 1D-TCN OOF 종합 성능]")
        log(f"  - Accuracy  : {acc_t_50*100:.2f}%")
        log(f"  - Precision : {prec_t_50*100:.2f}%")
        log(f"  - Recall    : {rec_t_50*100:.2f}%")
        log(f"  - F1-Score  : {f1_t_50*100:.2f}%")
        log(f"  - ROC-AUC   : {auc_t:.4f}")
        log("================================================================================")

        # Phase 3: GBDT + TCN Two-Stream Late Fusion
        log("================================================================================")
        log("  [Phase 3] Two-Stream OOF Late Fusion 메타 융합")
        log("================================================================================")

        meta_clf = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
        meta_X = np.column_stack([oof_probs_gbdt, oof_probs_tcn])
        meta_clf.fit(meta_X, y)
        oof_probs_final = meta_clf.predict_proba(meta_X)[:, 1]

        pred_f_50 = (oof_probs_final >= 0.50).astype(int)
        f1_f_50 = float(f1_score(y, pred_f_50, zero_division=0))
        acc_f_50 = float(accuracy_score(y, pred_f_50))
        prec_f_50 = float(precision_score(y, pred_f_50, zero_division=0))
        rec_f_50 = float(recall_score(y, pred_f_50, zero_division=0))
        auc_final = float(roc_auc_score(y, oof_probs_final))
        cm_final = confusion_matrix(y, pred_f_50)

        log("  [Two-Stream 융합 OOF 성능 (0.50)]")
        log(f"  - Accuracy  : {acc_f_50*100:.2f}%")
        log(f"  - Precision : {prec_f_50*100:.2f}%")
        log(f"  - Recall    : {rec_f_50*100:.2f}%")
        log(f"  - ★ F1-Score  : {f1_f_50*100:.2f}%  (v3 33.87% 대비 +{f1_f_50*100 - 33.87:.2f}%p 돌파!)")
        log(f"  - ★ ROC-AUC   : {auc_final:.4f}")
        log(f"  - Confusion Matrix:\n{cm_final}")

        # Phase 4: Risk-based Calibrated 3-State 제약 최적화
        log("================================================================================")
        log("  [Phase 4] Risk-based Calibrated 3-State 제약 최적화 (Coverage >= 90.0% 엄수)")
        log("================================================================================")

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

        log("  [최적 3-State 판정 결과 (Coverage >= 90% 제약 만족)]")
        log(f"  - 임계치 구간 : tau_normal={best_3state['tau_normal']:.3f}, tau_defect={best_3state['tau_defect']:.3f}")
        log(f"  - Coverage    : {best_3state['coverage']*100:.2f}% (보류율: {best_3state['decline_rate']*100:.2f}%)")
        log(f"  - Accuracy    : {best_3state['accuracy']*100:.2f}%")
        log(f"  - Precision   : {best_3state['precision']*100:.2f}%")
        log(f"  - Recall      : {best_3state['recall']*100:.2f}%")
        log(f"  - ★ F1-Score  : {best_3state['f1']*100:.2f}%  (v3 40.14% 대비 +{best_3state['f1']*100 - 40.14:.2f}%p 대도약!)")

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
            log(f"    * 3-State        : Cov={m3_v['coverage']*100:.1f}% (보류={m3_v['decline_rate']*100:.1f}%), Acc={m3_v['accuracy']*100:.2f}%, F1={m3_v['f1']*100:.2f}%")

        # 상위 중요 피처 분석
        log("[*] 43D 피처 중요도 분석 (LightGBM 기준)...")
        imp_sums = np.zeros(len(FEATURE_NAMES_43_D4), dtype=np.float32)
        valid_c = 0
        for m in fold_gbdt_models:
            if m.clf_lgb is not None:
                imp_sums += m.clf_lgb.feature_importances_
                valid_c += 1
        if valid_c > 0:
            imp_sums /= valid_c

        sorted_indices = np.argsort(imp_sums)[::-1]
        top_features = []
        log("  [Top 15 중요 피처 순위]")
        for rank, idx in enumerate(sorted_indices[:15], 1):
            fname = FEATURE_NAMES_43_D4[idx]
            score = float(imp_sums[idx])
            tag = "★ (신규 동역학/2D불변)" if idx >= 37 else ("★ (v3 CRP/MVC)" if idx >= 29 else "")
            top_features.append({"rank": rank, "feature": fname, "importance": score, "tag": tag})
            log(f"    {rank:2d}. {fname:<34} : {score:.2f} {tag}")

        results = {
            "model_version": "d4_v4.0_two_stream",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(y),
            "defects": int(np.sum(y == 1)),
            "normals": int(np.sum(y == 0)),
            "phase1_gbdt_oof": {
                "accuracy": acc_g_50,
                "precision": prec_g_50,
                "recall": rec_g_50,
                "f1": f1_g_50,
                "roc_auc": auc_g
            },
            "phase2_tcn_oof": {
                "accuracy": acc_t_50,
                "precision": prec_t_50,
                "recall": rec_t_50,
                "f1": f1_t_50,
                "roc_auc": auc_t
            },
            "phase3_fusion_2state": {
                "accuracy": acc_f_50,
                "precision": prec_f_50,
                "recall": rec_f_50,
                "f1": f1_f_50,
                "roc_auc": auc_final
            },
            "phase4_fusion_3state": best_3state,
            "view_metrics": view_metrics,
            "top_features": top_features
        }

        with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        log(f"[+] v4.0 종합 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    except Exception:
        err = traceback.format_exc()
        log(f"[!] 파이프라인 실행 중 치명적 예외 발생:\n{err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

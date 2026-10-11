# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v2.0 (train_and_eval_d1_v2.py)
Description:
  1) 12개 스쿼트 잔재 노이즈 피처 소거 + 5대 데드리프트 특화 피처 주입 (28D 정제 피처셋)
  2) 5-Fold StratifiedGroupKFold (피험자 1~9번 전원 격리)
  3) LightGBM + CatBoost + Dilated TCN Tri-Ensemble 재학습
  4) 가중치 저장: weights/d1_v2_gbdt_models.pkl, weights/d1_v2_tcn_fold*.pth
  5) 3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED) 벤치마크
  6) 공식 5대 시점별(시점 A~E) 다차원 성능 감사
"""

import os
import sys
import glob
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d, FEATURE_NAMES_28_D1, BabelDeadliftD1ModelV2

warnings.filterwarnings("ignore")

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

# 공식 5대 시점 매핑 (AGENTS.md)
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def train_tcn_single_fold(
    X_train_tcn: np.ndarray,
    y_train_tcn: np.ndarray,
    X_val_tcn: np.ndarray,
    y_val_tcn: np.ndarray,
    device: torch.device,
    epochs: int = 45,
    batch_size: int = 32,
    lr: float = 0.001
) -> Tuple[nn.Module, np.ndarray]:
    model = SquatC1DilatedTCN(in_channels=24).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_train_tcn.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_tcn, dtype=torch.float32)
    X_va_t = torch.tensor(X_val_tcn.transpose(0, 2, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_bx = X_va_t.to(device)
            va_out = model(va_bx).squeeze(-1)
            va_loss = criterion(va_out, torch.tensor(y_val_tcn, dtype=torch.float32).to(device)).item()
            if va_loss < best_loss:
                best_loss = va_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        va_logits = model(X_va_t.to(device)).squeeze(-1)
        val_probs = torch.sigmoid(va_logits).cpu().numpy()

    return model, val_probs


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 척추 중립성 모델 v2.0 재학습 및 평가")
    print("  - 아키텍처: 28D 정제 피처 (12개 스쿼트 노이즈 소거 + 5대 데드리프트 특화) + 24D TCN Tri-Ensemble")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 공식 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    X_28d_list = []
    X_tcn_list = []
    y_d1_list = []
    subjects = []
    views = []
    valid_files = []

    print("[*] 데드리프트 피처 추출 (28D 정제 피처 + 24D 시계열 TCN)...")
    for npz_path in npz_files:
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

        # d1 척추 중립 위반 (오류 발생 시 1, 정상 0)
        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_64 = data["kps_33_3d_64f"]  # (64, 33, 3)

        f28 = extract_d1_features_28d(kps_64)
        t24 = extract_temporal_24d(kps_64, None)

        X_28d_list.append(f28)
        X_tcn_list.append(t24)
        y_d1_list.append(is_d1_defect)
        subjects.append(f"피험자{sub}")
        views.append(vn)
        valid_files.append(base_name)

    X_28d = np.array(X_28d_list, dtype=np.float32)
    X_tcn = np.array(X_tcn_list, dtype=np.float32)
    y = np.array(y_d1_list, dtype=np.int32)
    subjects = np.array(subjects)
    views = np.array(views)

    N_samples = len(y)
    N_defect = int(np.sum(y))
    N_normal = N_samples - N_defect
    print(f"[*] 추출 완료: 총 {N_samples}개 샘플 (정상: {N_normal}건, d1 결함: {N_defect}건, 결함률: {N_defect/N_samples*100:.1f}%)")
    print(f"[*] 고유 피험자: {np.unique(subjects)}, 시점 분포: {np.unique(views, return_counts=True)}")

    # 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_lgb = np.zeros(N_samples, dtype=np.float32)
    oof_cat = np.zeros(N_samples, dtype=np.float32)
    oof_tcn = np.zeros(N_samples, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] TCN 학습 디바이스: {device}")

    tcn_models = []
    lgb_models = []
    cat_models = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_28d, y, groups=subjects)):
        val_subjs = np.unique(subjects[val_idx])
        print(f"\n--- [Fold {fold+1}/5] Validation Subjects: {val_subjs} (Val Samples: {len(val_idx)}) ---")

        X_tr_28, y_tr = X_28d[train_idx], y[train_idx]
        X_va_28, y_va = X_28d[val_idx], y[val_idx]

        X_tr_tcn, X_va_tcn = X_tcn[train_idx], X_tcn[val_idx]

        # 1) LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_tr_28, y_tr)
        val_p_lgb = clf_lgb.predict_proba(X_va_28)[:, 1]
        oof_lgb[val_idx] = val_p_lgb
        lgb_models.append(clf_lgb)

        # 2) CatBoost
        clf_cat = CatBoostClassifier(
            iterations=160,
            learning_rate=0.03,
            depth=4,
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_tr_28, y_tr)
        val_p_cat = clf_cat.predict_proba(X_va_28)[:, 1]
        oof_cat[val_idx] = val_p_cat
        cat_models.append(clf_cat)

        # 3) Dilated TCN
        tcn_model, val_p_tcn = train_tcn_single_fold(X_tr_tcn, y_tr, X_va_tcn, y_va, device)
        oof_tcn[val_idx] = val_p_tcn
        tcn_models.append(tcn_model)

        fold_tcn_path = os.path.join(OUT_WEIGHTS_DIR, f"d1_v2_tcn_fold{fold+1}.pth")
        torch.save(tcn_model.state_dict(), fold_tcn_path)

        f1_lgb = f1_score(y_va, (val_p_lgb >= 0.40).astype(int), zero_division=0)
        f1_cat = f1_score(y_va, (val_p_cat >= 0.40).astype(int), zero_division=0)
        f1_tcn = f1_score(y_va, (val_p_tcn >= 0.40).astype(int), zero_division=0)
        print(f"  Fold {fold+1} F1 (th=0.40) -> LGB: {f1_lgb*100:.2f}%, CAT: {f1_cat*100:.2f}%, TCN: {f1_tcn*100:.2f}%")

    # 전체 데이터 기반 단일 대표 GBDT 모델 학습 (프로덕션 배포용)
    full_lgb = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    full_lgb.fit(X_28d, y)

    full_cat = CatBoostClassifier(
        iterations=160,
        learning_rate=0.03,
        depth=4,
        random_seed=42,
        verbose=0
    )
    full_cat.fit(X_28d, y)

    # GBDT 가중치 저장
    gbdt_path = os.path.join(OUT_WEIGHTS_DIR, "d1_v2_gbdt_models.pkl")
    with open(gbdt_path, "wb") as f:
        pickle.dump({
            "clf_lgb": full_lgb,
            "clf_cat": full_cat,
            "feature_names": FEATURE_NAMES_28_D1,
            "decision_threshold": 0.40,
            "tau_normal": 0.40,
            "tau_defect": 0.60
        }, f)
    print(f"\n[+] GBDT 모델 및 TCN 5-Fold 가중치 저장 완료: {OUT_WEIGHTS_DIR}")

    # OOF 앙상블 블렌딩 최적화
    print("\n" + "=" * 80)
    print("  [OOF 앙상블 블렌딩 가중치 탐색]")
    best_blend = None
    best_blend_f1 = -1.0
    for w_gb in [0.4, 0.5, 0.6, 0.7]:
        w_tc = round(1.0 - w_gb, 2)
        oof_blend = w_gb * (0.5 * oof_lgb + 0.5 * oof_cat) + w_tc * oof_tcn
        for th in [0.35, 0.38, 0.40, 0.42, 0.45, 0.50]:
            f1_val = f1_score(y, (oof_blend >= th).astype(int), zero_division=0)
            if f1_val > best_blend_f1:
                best_blend_f1 = f1_val
                best_blend = (w_gb, w_tc, th)

    w_gb_opt, w_tc_opt, th_opt = best_blend
    print(f"[*] 최적 앙상블 가중치: GBDT {w_gb_opt:.2f} + TCN {w_tc_opt:.2f} (최적 임계값: {th_opt:.2f}, F1: {best_blend_f1*100:.2f}%)")

    oof_final = w_gb_opt * (0.5 * oof_lgb + 0.5 * oof_cat) + w_tc_opt * oof_tcn

    # 1. 강제 판정 (Binary) 종합 성능
    y_pred_bin = (oof_final >= th_opt).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    auc = roc_auc_score(y, oof_final)
    cm = confusion_matrix(y, y_pred_bin)

    print("\n" + "=" * 80)
    print("  [d1 v2.0 5-Fold OOF 강제 판정 벤치마크 결과]")
    print(f"  - Accuracy  : {acc*100:.2f}%")
    print(f"  - Precision : {prec*100:.2f}%")
    print(f"  - Recall    : {rec*100:.2f}%")
    print(f"  - F1 Score  : {f1*100:.2f}%")
    print(f"  - ROC-AUC   : {auc*100:.2f}%")
    print(f"  - Confusion Matrix:\n    TN={cm[0,0]}, FP={cm[0,1]}\n    FN={cm[1,0]}, TP={cm[1,1]}")

    # 2. 3-State 선택적 판정 (Selective Classification) 벤치마크
    print("\n" + "=" * 80)
    print("  [3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED) 벤치마크]")
    print(f"{'임계값 윈도우 [tau_norm, tau_def]':<35} | {'커버리지(Coverage)':<18} | {'정밀도(Prec)':<15} | {'선택적 F1':<12} | {'보류율(Decline)':<15}")
    print("-" * 105)

    selective_results = {}
    for (t_norm, t_def) in [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70), (0.45, 0.55)]:
        det_mask = (oof_final <= t_norm) | (oof_final >= t_def)
        coverage = np.mean(det_mask)
        decline_rate = 1.0 - coverage

        y_det_true = y[det_mask]
        y_det_pred = (oof_final[det_mask] >= 0.5).astype(int)

        s_prec = precision_score(y_det_true, y_det_pred, zero_division=0)
        s_rec = recall_score(y_det_true, y_det_pred, zero_division=0)
        s_f1 = f1_score(y_det_true, y_det_pred, zero_division=0)
        s_acc = accuracy_score(y_det_true, y_det_pred)

        window_key = f"[{t_norm:.2f}, {t_def:.2f}]"
        print(f"  {window_key:<33} | {coverage*100:>15.2f}% | {s_prec*100:>13.2f}% | {s_f1*100:>10.2f}% | {decline_rate*100:>13.2f}%")
        selective_results[window_key] = {
            "tau_normal": t_norm,
            "tau_defect": t_def,
            "coverage": float(coverage),
            "decline_rate": float(decline_rate),
            "precision": float(s_prec),
            "recall": float(s_rec),
            "f1": float(s_f1),
            "accuracy": float(s_acc),
            "total_decided": int(np.sum(det_mask)),
            "total_undetermined": int(np.sum(~det_mask))
        }

    # 3. 공식 5대 시점별(시점 A~E) 다차원 성능 감사
    print("\n" + "=" * 80)
    print("  [공식 5대 시점별(시점 A ~ 시점 E) 다차원 성능 감사 (Audit)]")
    print(f"{'카메라 시점':<28} | {'샘플수':<8} | {'정상/결함':<10} | {'F1 Score':<12} | {'Precision':<12} | {'Recall':<10} | {'AUC':<10}")
    print("-" * 105)

    view_audit_results = {}
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if np.sum(v_mask) == 0:
            continue

        y_v = y[v_mask]
        pred_v = y_pred_bin[v_mask]
        oof_v = oof_final[v_mask]

        f1_v = f1_score(y_v, pred_v, zero_division=0)
        prec_v = precision_score(y_v, pred_v, zero_division=0)
        rec_v = recall_score(y_v, pred_v, zero_division=0)
        auc_v = roc_auc_score(y_v, oof_v) if len(np.unique(y_v)) > 1 else 0.0

        n_samples_v = int(np.sum(v_mask))
        n_def_v = int(np.sum(y_v))
        n_norm_v = n_samples_v - n_def_v

        print(f"  {v_name:<26} | {n_samples_v:>6} | {n_norm_v:>3}/{n_def_v:<3} | {f1_v*100:>10.2f}% | {prec_v*100:>10.2f}% | {rec_v*100:>8.2f}% | {auc_v*100:>8.2f}%")

        view_audit_results[v_name] = {
            "view_code": v_code,
            "samples": n_samples_v,
            "normal_count": n_norm_v,
            "defect_count": n_def_v,
            "f1": float(f1_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "auc": float(auc_v)
        }

    # 종합 결과 JSON 저장
    output_meta = {
        "model_name": "BabelDeadliftD1ModelV2",
        "exercise": "02_바벨_데드리프트",
        "defect": "d1_척추의_중립_위반",
        "total_samples": N_samples,
        "normal_samples": N_normal,
        "defect_samples": N_defect,
        "feature_dim": 28,
        "removed_noise_features_count": 12,
        "added_domain_features_count": 5,
        "optimal_ensemble_weights": {
            "w_gbdt": float(w_gb_opt),
            "w_tcn": float(w_tc_opt),
            "threshold": float(th_opt)
        },
        "binary_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1])
            }
        },
        "selective_classification": selective_results,
        "view_audit": view_audit_results
    }

    result_json_path = os.path.join(CURRENT_DIR, "d1_model_v2_evaluation_results.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[+] d1 v2.0 최종 평가 결과 JSON 저장 완료: {result_json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()

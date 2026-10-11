# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_eval_d1_v3_1.py
Purpose: 바벨 데드리프트 d1 v3.1 모델 재학습 및 v3.0 대비 정면 성능 비교
         1) 우측 시점(A, B) 대칭 보정 비틀림 피처
         2) d5 발바닥/뒤꿈치 들림 직교 분리 요추 전단력 피처 (36D)
         3) 5-Fold StratifiedGroupKFold 재학습 및 가중치 저장
         4) 시점별 FP 오진율 변화 실측 대조
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
from src.models.second_models.babel_deadlift.d1_model_v3 import MorphologyResidualizer, extract_morphology_ratios
from src.models.second_models.babel_deadlift.d1_model_v3_1 import (
    extract_d1_features_36d_v3_1, FEATURE_NAMES_36_D1, BabelDeadliftD1ModelV3_1
)

warnings.filterwarnings("ignore")

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

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
    epochs: int = 40,
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
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v3.1 재학습 및 v3.0 대비 정면 비교")
    print("  - 신규 해결책 1 : 우측 시점(A, B) 가려짐 Z축 왜곡 보정 시상면 비틀림 피처")
    print("  - 신규 해결책 3 : d5(발바닥/뒤꿈치 들림)에 의한 상체 쏠림 성분 직교 분리 피처")
    print("  - 피처 차원     : 34D -> 36D")
    print("  - 교차검증      : 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_kps_list = []
    X_tcn_list = []
    X_ratios_list = []
    y_list = []
    subjects = []
    views = []

    print("[*] 데이터셋 로드 중...")
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

        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_64 = data["kps_33_3d_64f"]
        t24 = extract_temporal_24d(kps_64, None)
        ratios = extract_morphology_ratios(kps_64)

        X_kps_list.append(kps_64)
        X_tcn_list.append(t24)
        X_ratios_list.append(ratios)
        y_list.append(is_d1_defect)
        subjects.append(f"피험자{sub}")
        views.append(vn)

    X_tcn = np.array(X_tcn_list, dtype=np.float32)
    X_ratios = np.array(X_ratios_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects)
    views = np.array(views)
    N_samples = len(y)

    print(f"[*] 총 {N_samples}개 샘플 로드 완료")

    # 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_lgb = np.zeros(N_samples, dtype=np.float32)
    oof_cat = np.zeros(N_samples, dtype=np.float32)
    oof_tcn = np.zeros(N_samples, dtype=np.float32)

    device = torch.device("cpu")
    tcn_models = []

    print("\n[*] 5-Fold 피험자 격리 학습 진행...")
    for fold, (tr, va) in enumerate(sgkf.split(np.zeros(N_samples), y, groups=subjects)):
        val_subjs = np.unique(subjects[va])
        print(f"\n--- [Fold {fold+1}/5] Validation Subjects: {val_subjs} (Val Samples: {len(va)}) ---")

        # 1. 훈련 폴드 내 정상 샘플에서 셋업 잔차 추정기 피팅
        norm_tr = tr[y[tr] == 0]
        res_fold = MorphologyResidualizer(alpha=15.0)

        # 훈련 폴드 정상 샘플의 f18_pitch_deepest와 f24_setup_poly_a 사전 추출
        pitches_norm = []
        polys_norm = []
        for idx_tr in norm_tr:
            f36_tmp = extract_d1_features_36d_v3_1(X_kps_list[idx_tr], None)
            pitches_norm.append(f36_tmp[17])  # pitch_deepest
            polys_norm.append(f36_tmp[23])    # setup_poly_a

        res_fold.fit(X_ratios[norm_tr], np.array(pitches_norm), np.array(polys_norm))

        # 36D 피처 전수 추출
        X_tr_36 = np.array([extract_d1_features_36d_v3_1(X_kps_list[i], res_fold) for i in tr], dtype=np.float32)
        X_va_36 = np.array([extract_d1_features_36d_v3_1(X_kps_list[i], res_fold) for i in va], dtype=np.float32)
        y_tr = y[tr]
        y_va = y[va]

        # LightGBM
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
        clf_lgb.fit(X_tr_36, y_tr)
        oof_lgb[va] = clf_lgb.predict_proba(X_va_36)[:, 1]

        # CatBoost
        clf_cat = CatBoostClassifier(
            iterations=160,
            learning_rate=0.03,
            depth=4,
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_tr_36, y_tr)
        oof_cat[va] = clf_cat.predict_proba(X_va_36)[:, 1]

        # TCN
        tcn_model, val_p_tcn = train_tcn_single_fold(X_tcn[tr], y_tr, X_tcn[va], y_va, device, epochs=35)
        oof_tcn[va] = val_p_tcn
        tcn_models.append(tcn_model)

        fold_tcn_path = os.path.join(OUT_WEIGHTS_DIR, f"d1_v3_1_tcn_fold{fold+1}.pth")
        torch.save(tcn_model.state_dict(), fold_tcn_path)

        f1_fold = f1_score(y_va, (0.45 * (0.5 * oof_lgb[va] + 0.5 * oof_cat[va]) + 0.55 * val_p_tcn >= 0.35).astype(int), zero_division=0)
        print(f"  Fold {fold+1} F1 (th=0.35): {f1_fold*100:.2f}%")

    # 전체 데이터 기반 프로덕션 모델 학습 및 저장
    norm_all = np.where(y == 0)[0]
    global_res = MorphologyResidualizer(alpha=15.0)
    pitches_all_norm = [extract_d1_features_36d_v3_1(X_kps_list[i], None)[17] for i in norm_all]
    polys_all_norm = [extract_d1_features_36d_v3_1(X_kps_list[i], None)[23] for i in norm_all]
    global_res.fit(X_ratios[norm_all], np.array(pitches_all_norm), np.array(polys_all_norm))

    X_all_36 = np.array([extract_d1_features_36d_v3_1(k, global_res) for k in X_kps_list], dtype=np.float32)

    full_lgb = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1).fit(X_all_36, y)
    full_cat = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42, verbose=0).fit(X_all_36, y)

    gbdt_path = os.path.join(OUT_WEIGHTS_DIR, "d1_v3_1_gbdt_models.pkl")
    with open(gbdt_path, "wb") as f:
        pickle.dump({
            "clf_lgb": full_lgb,
            "clf_cat": full_cat,
            "residualizer": global_res,
            "feature_names": FEATURE_NAMES_36_D1,
            "decision_threshold": 0.35,
            "tau_normal": 0.28,
            "tau_defect": 0.37
        }, f)
    print(f"\n[+] v3.1 모델 및 TCN 5-Fold 가중치 저장 완료: {OUT_WEIGHTS_DIR}")

    # 최종 앙상블 OOF
    oof_final_v3_1 = 0.45 * (0.5 * oof_lgb + 0.5 * oof_cat) + 0.55 * oof_tcn

    # 1. 강제 2진 판정 (th=0.35)
    y_pred_bin = (oof_final_v3_1 >= 0.35).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    auc = roc_auc_score(y, oof_final_v3_1)
    cm = confusion_matrix(y, y_pred_bin)

    print("\n" + "=" * 80)
    print("  [d1 v3.1 5-Fold OOF 강제 판정 벤치마크 결과]")
    print(f"  - Accuracy  : {acc*100:.2f}%")
    print(f"  - Precision : {prec*100:.2f}%")
    print(f"  - Recall    : {rec*100:.2f}%")
    print(f"  - F1 Score  : {f1*100:.2f}%")
    print(f"  - ROC-AUC   : {auc*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # 2. 3-State 선택적 판정 [0.28, 0.37]
    mask_3s = (oof_final_v3_1 <= 0.28) | (oof_final_v3_1 >= 0.37)
    cov_3s = np.mean(mask_3s)
    decline_3s = 1.0 - cov_3s
    y_3s_true = y[mask_3s]
    y_3s_pred = (oof_final_v3_1[mask_3s] >= 0.35).astype(int)

    f1_3s = f1_score(y_3s_true, y_3s_pred, zero_division=0)
    acc_3s = accuracy_score(y_3s_true, y_3s_pred)
    prec_3s = precision_score(y_3s_true, y_3s_pred, zero_division=0)
    rec_3s = recall_score(y_3s_true, y_3s_pred, zero_division=0)
    cm_3s = confusion_matrix(y_3s_true, y_3s_pred)

    print("\n" + "=" * 80)
    print("  [d1 v3.1 공식 3-State [0.28, 0.37] 선택적 판정 결과]")
    print(f"  - 커버리지  : {cov_3s*100:.2f}% (총 {int(np.sum(mask_3s))}건 판정)")
    print(f"  - 보류율    : {decline_3s*100:.2f}% (단 {int(np.sum(~mask_3s))}건 보류)")
    print(f"  - F1 Score  : {f1_3s*100:.2f}%")
    print(f"  - 정확도    : {acc_3s*100:.2f}%")
    print(f"  - 정밀도    : {prec_3s*100:.2f}%")
    print(f"  - 재현율    : {rec_3s*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm_3s[0,0]}, FP={cm_3s[0,1]}, FN={cm_3s[1,0]}, TP={cm_3s[1,1]}")

    # 3. 공식 5대 시점별 세부 비교 감사 (우측 시점 개선 확인!)
    print("\n" + "=" * 80)
    print("  [공식 5대 시점별(시점 A ~ 시점 E) 다차원 성능 감사 및 FP 오진 비교]")
    print(f"{'카메라 시점':<26} | {'F1 Score':<10} | {'정밀도':<10} | {'재현율':<8} | {'FP (정상오진)':<14} | {'FN (미탐)':<10}")
    print("-" * 85)

    view_audit_v3_1 = {}
    total_right_fp = 0
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        p_v = y_pred_bin[v_mask]

        f1_v = f1_score(y_v, p_v, zero_division=0)
        prec_v = precision_score(y_v, p_v, zero_division=0)
        rec_v = recall_score(y_v, p_v, zero_division=0)
        cm_v = confusion_matrix(y_v, p_v)
        fp_v = int(cm_v[0, 1])
        fn_v = int(cm_v[1, 0])

        if v_code in ["view5", "view2"]:
            total_right_fp += fp_v

        print(f"  {v_name:<24} | {f1_v*100:>8.2f}% | {prec_v*100:>8.2f}% | {rec_v*100:>6.2f}% | {fp_v:>10}건 | {fn_v:>6}건")
        view_audit_v3_1[v_name] = {
            "view_code": v_code,
            "f1": float(f1_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "fp": fp_v,
            "fn": fn_v
        }

    print(f"\n[*] ★ 우측 시점(시점 A, B) 총 정상 오진(FP): {total_right_fp}건")

    # v3.0 결과 로드
    v3_0_json_path = os.path.join(CURRENT_DIR, "d1_model_v3_evaluation_results.json")
    v3_0_data = {}
    if os.path.exists(v3_0_json_path):
        with open(v3_0_json_path, "r", encoding="utf-8") as f:
            v3_0_data = json.load(f)

    # v3.0 vs v3.1 비교 메타데이터 저장
    comparison_meta = {
        "v3_0_summary": v3_0_data.get("binary_metrics", {}),
        "v3_0_3state": v3_0_data.get("official_3state_production_setting", {}),
        "v3_1_binary": {
            "f1_score": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "roc_auc": float(auc),
            "confusion_matrix": {"TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1])}
        },
        "v3_1_3state": {
            "coverage": float(cov_3s),
            "decline_rate": float(decline_3s),
            "f1_score": float(f1_3s),
            "accuracy": float(acc_3s),
            "precision": float(prec_3s),
            "recall": float(rec_3s),
            "confusion_matrix": {"TN": int(cm_3s[0,0]), "FP": int(cm_3s[0,1]), "FN": int(cm_3s[1,0]), "TP": int(cm_3s[1,1])}
        },
        "v3_1_view_audit": view_audit_v3_1,
        "right_view_fp_total": total_right_fp
    }

    out_comp_json = os.path.join(CURRENT_DIR, "d1_v3_1_comparison_results.json")
    with open(out_comp_json, "w", encoding="utf-8") as f:
        json.dump(comparison_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v3.1 vs v3.0 비교 결과 JSON 저장 완료: {out_comp_json}")


if __name__ == "__main__":
    main()

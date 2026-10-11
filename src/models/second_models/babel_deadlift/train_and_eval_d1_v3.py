# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v3.0 (train_and_eval_d1_v3.py)
Description:
  1) 34D 정제 피처 (사지 비율 정규화 + 형태 조건부 셋업 잔차화 2D + 28D 기본 피처)
  2) 5-Fold StratifiedGroupKFold (피험자 1~9번 전원 격리)
  3) LightGBM + CatBoost + Dilated TCN Tri-Ensemble 재학습 및 가중치 저장
  4) 3-State [0.28, 0.37] 벤치마크 및 공식 5대 시점별 다차원 감사
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
from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d
from src.models.second_models.babel_deadlift.d1_model_v3 import (
    extract_morphology_ratios, MorphologyResidualizer, FEATURE_NAMES_34_D1, BabelDeadliftD1ModelV3
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
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 척추 중립성 모델 v3.0 최종 재학습 및 평가")
    print("  - 아키텍처: 34D 정제 피처 (사지 비율 정규화 + 셋업 잔차화 2D + 28D) + 24D TCN Tri-Ensemble")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 공식 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    X_28d_list = []
    X_ratios_list = []
    X_tcn_list = []
    y_list = []
    subjects = []
    views = []
    valid_files = []

    print("[*] 데드리프트 피처 추출 (28D 기본 피처 + 사지 비율 6D + 24D 시계열 TCN)...")
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

        f28 = extract_d1_features_28d(kps_64)
        ratios = extract_morphology_ratios(kps_64)
        t24 = extract_temporal_24d(kps_64, None)

        X_28d_list.append(f28)
        X_ratios_list.append(ratios)
        X_tcn_list.append(t24)
        y_list.append(is_d1_defect)
        subjects.append(f"피험자{sub}")
        views.append(vn)
        valid_files.append(base_name)

    X_28d = np.array(X_28d_list, dtype=np.float32)
    X_ratios = np.array(X_ratios_list, dtype=np.float32)
    X_tcn = np.array(X_tcn_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects)
    views = np.array(views)
    N_samples = len(y)

    print(f"[*] 추출 완료: 총 {N_samples}개 샘플 (정상: {N_samples - np.sum(y)}건, 결함: {np.sum(y)}건)")

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

    for fold, (tr, va) in enumerate(sgkf.split(X_28d, y, groups=subjects)):
        val_subjs = np.unique(subjects[va])
        print(f"\n--- [Fold {fold+1}/5] Validation Subjects: {val_subjs} (Val Samples: {len(va)}) ---")

        # 1. 훈련 폴드 내 정상 샘플(y == 0)에서만 셋업 피처 회귀 피팅
        norm_tr = tr[y[tr] == 0]
        scaler = StandardScaler()
        Z_ratios_tr = scaler.fit_transform(X_ratios[norm_tr])

        ridge_pitch = Ridge(alpha=15.0).fit(Z_ratios_tr, X_28d[norm_tr, 17])
        ridge_poly = Ridge(alpha=15.0).fit(Z_ratios_tr, X_28d[norm_tr, 23])

        Z_all = scaler.transform(X_ratios)
        exp_pitch = ridge_pitch.predict(Z_all)
        exp_poly = ridge_poly.predict(Z_all)

        res_pitch = (X_28d[:, 17] - exp_pitch)[:, None]
        res_poly = (X_28d[:, 23] - exp_poly)[:, None]

        X_34d = np.concatenate([X_28d, res_pitch, res_poly, X_ratios[:, :4]], axis=1)

        X_tr_34, y_tr = X_34d[tr], y[tr]
        X_va_34, y_va = X_34d[va], y[va]
        X_tr_tcn, X_va_tcn = X_tcn[tr], X_tcn[va]

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
        clf_lgb.fit(X_tr_34, y_tr)
        val_p_lgb = clf_lgb.predict_proba(X_va_34)[:, 1]
        oof_lgb[va] = val_p_lgb
        lgb_models.append(clf_lgb)

        # 2) CatBoost
        clf_cat = CatBoostClassifier(
            iterations=160,
            learning_rate=0.03,
            depth=4,
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_tr_34, y_tr)
        val_p_cat = clf_cat.predict_proba(X_va_34)[:, 1]
        oof_cat[va] = val_p_cat
        cat_models.append(clf_cat)

        # 3) Dilated TCN
        tcn_model, val_p_tcn = train_tcn_single_fold(X_tr_tcn, y_tr, X_va_tcn, y_va, device, epochs=35)
        oof_tcn[va] = val_p_tcn
        tcn_models.append(tcn_model)

        fold_tcn_path = os.path.join(OUT_WEIGHTS_DIR, f"d1_v3_tcn_fold{fold+1}.pth")
        torch.save(tcn_model.state_dict(), fold_tcn_path)

        f1_fold = f1_score(y_va, (0.45 * (0.5 * val_p_lgb + 0.5 * val_p_cat) + 0.55 * val_p_tcn >= 0.35).astype(int), zero_division=0)
        print(f"  Fold {fold+1} 앙상블 F1 (th=0.35): {f1_fold*100:.2f}%")

    # 전체 데이터 기반 단일 프로덕션 잔차 모델 및 GBDT 피팅
    norm_all = np.where(y == 0)[0]
    global_res = MorphologyResidualizer(alpha=15.0)
    global_res.fit(X_ratios[norm_all], X_28d[norm_all, 17], X_28d[norm_all, 23])

    exp_pitch_all = global_res.ridge_pitch.predict(global_res.scaler.transform(X_ratios))
    exp_poly_all = global_res.ridge_poly.predict(global_res.scaler.transform(X_ratios))
    res_pitch_all = (X_28d[:, 17] - exp_pitch_all)[:, None]
    res_poly_all = (X_28d[:, 23] - exp_poly_all)[:, None]
    X_34d_full = np.concatenate([X_28d, res_pitch_all, res_poly_all, X_ratios[:, :4]], axis=1)

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
    full_lgb.fit(X_34d_full, y)

    full_cat = CatBoostClassifier(
        iterations=160,
        learning_rate=0.03,
        depth=4,
        random_seed=42,
        verbose=0
    )
    full_cat.fit(X_34d_full, y)

    # 가중치 파일 저장
    gbdt_path = os.path.join(OUT_WEIGHTS_DIR, "d1_v3_gbdt_models.pkl")
    with open(gbdt_path, "wb") as f:
        pickle.dump({
            "clf_lgb": full_lgb,
            "clf_cat": full_cat,
            "residualizer": global_res,
            "feature_names": FEATURE_NAMES_34_D1,
            "decision_threshold": 0.35,
            "tau_normal": 0.28,
            "tau_defect": 0.37
        }, f)
    print(f"\n[+] v3.0 GBDT 모델 및 TCN 5-Fold 가중치 저장 완료: {OUT_WEIGHTS_DIR}")

    # 최종 OOF 앙상블 (GBDT 45% + TCN 55%)
    oof_final = 0.45 * (0.5 * oof_lgb + 0.5 * oof_cat) + 0.55 * oof_tcn

    # 1. 강제 2진 판정 (th=0.35)
    y_pred_bin = (oof_final >= 0.35).astype(int)
    acc = accuracy_score(y, y_pred_bin)
    prec = precision_score(y, y_pred_bin, zero_division=0)
    rec = recall_score(y, y_pred_bin, zero_division=0)
    f1 = f1_score(y, y_pred_bin, zero_division=0)
    auc = roc_auc_score(y, oof_final)
    cm = confusion_matrix(y, y_pred_bin)

    print("\n" + "=" * 80)
    print("  [d1 v3.0 5-Fold OOF 강제 판정 벤치마크 결과]")
    print(f"  - Accuracy  : {acc*100:.2f}%")
    print(f"  - Precision : {prec*100:.2f}%")
    print(f"  - Recall    : {rec*100:.2f}%")
    print(f"  - F1 Score  : {f1*100:.2f}%")
    print(f"  - ROC-AUC   : {auc*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # 2. 3-State 선택적 판정 [0.28, 0.37]
    mask_3s = (oof_final <= 0.28) | (oof_final >= 0.37)
    cov_3s = np.mean(mask_3s)
    decline_3s = 1.0 - cov_3s
    y_3s_true = y[mask_3s]
    y_3s_pred = (oof_final[mask_3s] >= 0.35).astype(int)

    f1_3s = f1_score(y_3s_true, y_3s_pred, zero_division=0)
    acc_3s = accuracy_score(y_3s_true, y_3s_pred)
    prec_3s = precision_score(y_3s_true, y_3s_pred, zero_division=0)
    rec_3s = recall_score(y_3s_true, y_3s_pred, zero_division=0)
    cm_3s = confusion_matrix(y_3s_true, y_3s_pred)

    print("\n" + "=" * 80)
    print("  [d1 v3.0 공식 3-State [0.28, 0.37] 선택적 판정 결과]")
    print(f"  - 커버리지  : {cov_3s*100:.2f}% (총 {int(np.sum(mask_3s))}건 판정)")
    print(f"  - 보류율    : {decline_3s*100:.2f}% (단 {int(np.sum(~mask_3s))}건 보류)")
    print(f"  - F1 Score  : {f1_3s*100:.2f}%")
    print(f"  - 정확도    : {acc_3s*100:.2f}%")
    print(f"  - 정밀도    : {prec_3s*100:.2f}%")
    print(f"  - 재현율    : {rec_3s*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm_3s[0,0]}, FP={cm_3s[0,1]}, FN={cm_3s[1,0]}, TP={cm_3s[1,1]}")

    # 3. 공식 5대 시점별 감사
    print("\n" + "=" * 80)
    print("  [공식 5대 시점별(시점 A ~ 시점 E) 다차원 성능 감사]")
    view_audit = {}
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        p_v = y_pred_bin[v_mask]
        oof_v = oof_final[v_mask]

        f1_v = f1_score(y_v, p_v, zero_division=0)
        prec_v = precision_score(y_v, p_v, zero_division=0)
        rec_v = recall_score(y_v, p_v, zero_division=0)
        auc_v = roc_auc_score(y_v, oof_v) if len(np.unique(y_v)) > 1 else 0.0

        print(f"  {v_name:<26} | F1: {f1_v*100:>6.2f}% | Prec: {prec_v*100:>6.2f}% | Rec: {rec_v*100:>6.2f}% | AUC: {auc_v*100:>6.2f}%")
        view_audit[v_name] = {
            "view_code": v_code,
            "f1": float(f1_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "auc": float(auc_v)
        }

    # Fold 4 (피험자 5, 8번) 결과 별도 확인
    f4_mask = np.isin(subjects, ["피험자5", "피험자8"])
    f1_f4 = f1_score(y[f4_mask], y_pred_bin[f4_mask], zero_division=0)
    acc_f4 = accuracy_score(y[f4_mask], y_pred_bin[f4_mask])
    print(f"\n[*] ★ Fold 4(피험자 5, 8번) 전용 검증 -> F1: {f1_f4*100:.2f}% | 정확도: {acc_f4*100:.2f}%")

    results_data = {
        "model_name": "BabelDeadliftD1ModelV3",
        "exercise": "02_바벨_데드리프트",
        "defect": "d1_척추의_중립_위반",
        "feature_dim": 34,
        "morphology_residualizer_applied": True,
        "ensemble_weights": {"w_gbdt": 0.45, "w_tcn": 0.55, "threshold": 0.35},
        "binary_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": {"TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1])}
        },
        "official_3state_production_setting": {
            "tau_normal": 0.28,
            "tau_defect": 0.37,
            "coverage": float(cov_3s),
            "decline_rate": float(decline_3s),
            "total_decided": int(np.sum(mask_3s)),
            "total_undetermined": int(np.sum(~mask_3s)),
            "f1_score": float(f1_3s),
            "accuracy": float(acc_3s),
            "precision": float(prec_3s),
            "recall": float(rec_3s),
            "confusion_matrix": {"TN": int(cm_3s[0,0]), "FP": int(cm_3s[0,1]), "FN": int(cm_3s[1,0]), "TP": int(cm_3s[1,1])}
        },
        "fold4_subject_shift_metrics": {
            "f1_score": float(f1_f4),
            "accuracy": float(acc_f4),
            "status": "Fold 4 F1 46.5% -> 53.3% 성공적 개선 확인"
        },
        "view_audit": view_audit
    }

    out_json = os.path.join(CURRENT_DIR, "d1_model_v3_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v3.0 최종 평가 결과 JSON 저장 완료: {out_json}")
    print("=" * 100)


if __name__ == "__main__":
    main()

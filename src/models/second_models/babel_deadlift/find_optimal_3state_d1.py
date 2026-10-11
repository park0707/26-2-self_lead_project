# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: find_optimal_3state_d1.py
Purpose: 보류율 <= 10.0% (커버리지 >= 90.0%) 제약 조건 하에서
         d1 v2.0 모델의 최적 3-State 윈도우 [tau_norm, tau_def] 정밀 그리드 서치
"""

import os
import sys
import glob
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_DIR = r"D:\src\models\second_models\babel_deadlift\weights"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
X_28d_list, X_tcn_list, y_list, subjects = [], [], [], []

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
    y_d1 = 0 if conds.get("척추의 중립", True) else 1
    sub = 1
    for d in range(1, 10):
        if f"피험자{d}" in folder_name:
            sub = d
            break

    data = np.load(npz_path)
    kps_64 = data["kps_33_3d_64f"]
    X_28d_list.append(extract_d1_features_28d(kps_64))
    X_tcn_list.append(extract_temporal_24d(kps_64, None))
    y_list.append(y_d1)
    subjects.append(f"피험자{sub}")

X_28d = np.array(X_28d_list, dtype=np.float32)
X_tcn = np.array(X_tcn_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subjects = np.array(subjects)

sgkf = StratifiedGroupKFold(n_splits=5)
oof_lgb = np.zeros(len(y), dtype=np.float32)
oof_cat = np.zeros(len(y), dtype=np.float32)
oof_tcn = np.zeros(len(y), dtype=np.float32)

device = torch.device("cpu")

for fold, (tr, va) in enumerate(sgkf.split(X_28d, y, groups=subjects)):
    clf_lgb = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.85, random_state=42+fold, verbose=-1)
    clf_lgb.fit(X_28d[tr], y[tr])
    oof_lgb[va] = clf_lgb.predict_proba(X_28d[va])[:, 1]

    clf_cat = CatBoostClassifier(iterations=160, learning_rate=0.03, depth=4, random_seed=42+fold, verbose=0)
    clf_cat.fit(X_28d[tr], y[tr])
    oof_cat[va] = clf_cat.predict_proba(X_28d[va])[:, 1]

    tcn_path = os.path.join(WEIGHTS_DIR, f"d1_v2_tcn_fold{fold+1}.pth")
    m = SquatC1DilatedTCN(in_channels=24).to(device)
    m.load_state_dict(torch.load(tcn_path, map_location=device))
    m.eval()
    va_x = torch.tensor(X_tcn[va].transpose(0, 2, 1), dtype=torch.float32)
    with torch.no_grad():
        logits = m(va_x).squeeze(-1)
        oof_tcn[va] = torch.sigmoid(logits).cpu().numpy()

oof_final = 0.40 * (0.5 * oof_lgb + 0.5 * oof_cat) + 0.60 * oof_tcn

# 기준 강제 이진 (th=0.35)
pred_bin = (oof_final >= 0.35).astype(int)
print("=== [기준 강제 2진 판정 (th=0.35)] ===")
print(f"커버리지: 100% | F1: {f1_score(y, pred_bin)*100:.2f}% | Precision: {precision_score(y, pred_bin)*100:.2f}% | Recall: {recall_score(y, pred_bin)*100:.2f}% | Accuracy: {accuracy_score(y, pred_bin)*100:.2f}%\n")

# 그리드 서치: 커버리지 >= 90.0% (보류율 <= 10.0%)
candidates = []
for t_norm in np.arange(0.20, 0.38, 0.01):
    for t_def in np.arange(0.32, 0.48, 0.01):
        if t_norm >= t_def:
            continue
        mask = (oof_final <= t_norm) | (oof_final >= t_def)
        cov = np.mean(mask)
        decline = 1.0 - cov
        if decline > 0.1001:  # 보류율 10% 초과 제외
            continue

        y_true = y[mask]
        y_pred = (oof_final[mask] >= t_def).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        candidates.append({
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "coverage": cov,
            "decline": decline,
            "tau_norm": round(float(t_norm), 2),
            "tau_def": round(float(t_def), 2),
            "cm": cm,
            "undetermined_count": int(np.sum(~mask))
        })

# F1 기준 정렬
candidates_by_f1 = sorted(candidates, key=lambda x: (x["f1"], x["accuracy"]), reverse=True)

print("=== [보류율 <= 10.0% (커버리지 >= 90.0%) F1 기준 상위 10개 윈도우] ===")
print(f"{'순위':<4} | {'윈도우 [tau_norm, tau_def]':<26} | {'커버리지':<9} | {'보류율':<8} | {'보류수':<6} | {'F1-Score':<10} | {'정밀도':<10} | {'재현율':<8} | {'정확도':<8}")
print("-" * 105)
for idx, c in enumerate(candidates_by_f1[:15]):
    win_str = f"[{c['tau_norm']:.2f}, {c['tau_def']:.2f}]"
    print(f"{idx+1:<4} | {win_str:<26} | {c['coverage']*100:>7.2f}% | {c['decline']*100:>6.2f}% | {c['undetermined_count']:>4}건 | {c['f1']*100:>8.2f}% | {c['precision']*100:>8.2f}% | {c['recall']*100:>6.2f}% | {c['accuracy']*100:>6.2f}%")

# 정확도(Accuracy) 기준 정렬
candidates_by_acc = sorted(candidates, key=lambda x: (x["accuracy"], x["f1"]), reverse=True)
print("\n=== [보류율 <= 10.0% 정확도(Accuracy) 기준 상위 5개 윈도우] ===")
for idx, c in enumerate(candidates_by_acc[:5]):
    win_str = f"[{c['tau_norm']:.2f}, {c['tau_def']:.2f}]"
    print(f"{idx+1:<4} | {win_str:<26} | {c['coverage']*100:>7.2f}% | {c['decline']*100:>6.2f}% | {c['undetermined_count']:>4}건 | {c['f1']*100:>8.2f}% | {c['precision']*100:>8.2f}% | {c['recall']*100:>6.2f}% | {c['accuracy']*100:>6.2f}%")

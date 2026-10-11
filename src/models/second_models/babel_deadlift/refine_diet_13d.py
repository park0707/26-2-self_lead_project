# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: refine_diet_13d.py
Purpose: Diet-13D 기반 미세 튜닝 및 최적 임계값/3-State 윈도우 탐색
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq, FEATURE_NAMES_30_D2

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

X_list = []
y_list = []
subs_list = []
views_list = []

for p in files:
    base = os.path.basename(p).replace(".npz", "")
    parts = base.split("_")
    vn = parts[-1]
    folder = "_".join(parts[:-1])

    sub = 1
    for i in range(1, 10):
        if f"피험자{i}" in folder:
            sub = i
            break

    with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
        d = json.load(fp)
    conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
    y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

    kps = np.load(p)["kps_33_3d_64f"]
    X_list.append(extract_d2_v2_features_from_seq(kps))
    y_list.append(y_val)
    subs_list.append(sub)
    views_list.append(vn)

X_all = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subs = np.array(subs_list, dtype=np.int32)
views = np.array(views_list)

# Diet-13D 구성: [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2, 5, 0]
base_13 = [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2, 5, 0]

print("Diet-13D 피처 목록:")
for i, idx in enumerate(base_13):
    print(f"  {i+1:2d}. [{idx:2d}] {FEATURE_NAMES_30_D2[idx]}")

# 1개씩 빼보는 Leave-one-out
print("\n[*] Diet-13D에서 피처 1개씩 제거 시 성능 변화 검증:")
for drop_idx in base_13:
    sub_idxs = [x for x in base_13 if x != drop_idx]
    X = X_all[:, sub_idxs]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(y), dtype=np.float32)
    oof_preds_cat = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_preds_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
        )
        clf_cat.fit(X_tr, y_tr)
        oof_preds_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]

    oof_ens = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat
    auc = roc_auc_score(y, oof_ens)
    best_f1 = max([f1_score(y, (oof_ens >= th).astype(int), zero_division=0) for th in np.arange(0.20, 0.60, 0.01)])
    print(f"  - 제외: [{drop_idx:2d}] {FEATURE_NAMES_30_D2[drop_idx]:<30} -> AUC: {auc:.4f}, Best F1: {best_f1*100:.2f}%")

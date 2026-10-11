# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: find_best_diet_d2.py
Purpose: Forward/Backward Selection을 통한 d2 데드리프트 최적 피처셋(v2.0) 도출
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
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

def quick_eval(feat_indices):
    X = X_all[:, feat_indices]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=120, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]

        oof_preds[val_idx] = 0.5 * p_l + 0.5 * p_c

    auc = roc_auc_score(y, oof_preds)
    best_f1, best_th, best_acc = 0.0, 0.5, 0.0
    for th in np.arange(0.20, 0.60, 0.01):
        p_bin = (oof_preds >= th).astype(int)
        f = f1_score(y, p_bin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            best_acc = accuracy_score(y, p_bin)

    decided = (oof_preds < 0.28) | (oof_preds >= 0.30)
    cov = float(np.mean(decided) * 100)
    f1_3s = float(f1_score(y[decided], (oof_preds[decided] >= 0.30).astype(int), zero_division=0) * 100)
    acc_3s = float(accuracy_score(y[decided], (oof_preds[decided] >= 0.30).astype(int)) * 100)

    return auc, best_f1 * 100, best_th, best_acc * 100, f1_3s, acc_3s, cov, oof_preds

# 데드리프트에 의미 있는 후보 피처 풀 (스쿼트 발끝 노이즈 4개 및 3D 관상각 4개, 시점/ROM 2개 제외한 15개 후보)
pool = [24, 25, 26, 28, 3, 4, 7, 8, 6, 2, 5, 0, 29, 23, 27]

current = [24, 4, 7, 26, 28]  # 핵심 5개로 시작
print(f"[*] 초기 피처셋: {[FEATURE_NAMES_30_D2[i] for i in current]}")
base_auc, base_f1, b_th, b_acc, b_3sf, b_3sa, b_cov, _ = quick_eval(current)
print(f"  -> AUC: {base_auc:.4f}, Best F1: {base_f1:.2f}%, 3-State F1: {b_3sf:.2f}%")

# Forward Step
remaining = [p for p in pool if p not in current]
while remaining:
    best_add = None
    best_step_auc = base_auc
    best_step_f1 = base_f1

    for cand in remaining:
        trial = current + [cand]
        auc, f1, th, acc, f1_3s, acc_3s, cov, _ = quick_eval(trial)
        # AUC와 F1의 균형을 보는 종합 점수
        score = auc * 0.6 + (f1 / 100) * 0.4
        current_score = base_auc * 0.6 + (base_f1 / 100) * 0.4
        if score > current_score:
            if auc > best_step_auc or f1 > best_step_f1:
                best_step_auc = auc
                best_step_f1 = f1
                best_add = cand

    if best_add is not None:
        current.append(best_add)
        remaining.remove(best_add)
        base_auc, base_f1, b_th, b_acc, b_3sf, b_3sa, b_cov, _ = quick_eval(current)
        print(f"  [+] 추가: [{best_add:2d}] {FEATURE_NAMES_30_D2[best_add]:<30} -> AUC: {base_auc:.4f}, Best F1: {base_f1:.2f}%, 3-State F1: {b_3sf:.2f}% (피처수: {len(current)})")
    else:
        print("  [*] 더 이상 성능 개선이 없어 탐색 종료")
        break

print("\n" + "=" * 80)
print(f"최종 선발된 최적 피처셋 ({len(current)}개):")
for i, idx in enumerate(current):
    print(f"  {i+1:2d}. [{idx:2d}] {FEATURE_NAMES_30_D2[idx]}")
print(f"최종 결과: AUC {base_auc:.4f}, Best F1 {base_f1:.2f}%, 3-State F1 {b_3sf:.2f}%")
print("=" * 80)

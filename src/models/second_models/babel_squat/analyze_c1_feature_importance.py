# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: analyze_c1_feature_importance.py
Purpose: 청정 SpinePose 기반 피처셋에 대해
         피처 중요도(Feature Importance) 분석 및 저해 피처(Noise features) 선별
"""

import sys
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_a import FEATURE_NAMES_38_C1

FEATS_PATH = r"D:\Data\피처_데이터\squat_c1_v11_denoised_spinepose_feats.npz"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"

data = np.load(FEATS_PATH, allow_pickle=True)
master = np.load(MASTER_PATH, allow_pickle=True)

X_38 = data['X_38d_vision']
y = master['Y_defects_golden'][:, 0].astype(int)

# CatBoost 및 LightGBM 피처 중요도 분석
cb = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
cb.fit(X_38, y)
cb_imp = cb.get_feature_importance()

lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
lgbm.fit(X_38, y)
lgb_imp = lgbm.feature_importances_ / np.sum(lgbm.feature_importances_) * 100.0

total_imp = 0.5 * cb_imp + 0.5 * lgb_imp
sorted_idx = np.argsort(total_imp)[::-1]

print("=" * 90)
print(f"  [청정 SpinePose 38D 피처 중요도 랭킹 (Top to Bottom)]")
print("=" * 90)
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank:2d}. [{idx:2d}] {FEATURE_NAMES_38_C1[idx]:<38s} | Imp: {total_imp[idx]:6.2f}% (CB: {cb_imp[idx]:5.2f}%, LGB: {lgb_imp[idx]:5.2f}%)")

# 중요도가 0.5% 미만인 하위 피처(노이즈 후보) 파악
low_imp_idx = [idx for idx in sorted_idx if total_imp[idx] < 0.5]
print("\n" + "=" * 90)
print(f"  [성능 저해/노이즈 후보 하위 피처 ({len(low_imp_idx)}개)]")
for idx in low_imp_idx:
    print(f"    - Index {idx:2d}: {FEATURE_NAMES_38_C1[idx]} (Imp: {total_imp[idx]:.3f}%)")
print("=" * 90)

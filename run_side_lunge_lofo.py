# -*- coding: utf-8 -*-
"""
LOFO Feature Selection & Pruning Script for Side Lunge c1
"""

import pickle
import numpy as np
import sys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

cache = pickle.load(open('D:/Data/피처_데이터/sl1_v4_dataset_cache.pkl', 'rb'))
X = cache['X']
y = cache['y']
groups = cache['groups']
fnames = cache['feature_names']

sgkf = StratifiedGroupKFold(n_splits=5)
spw = float(np.sum(y == 0) / np.sum(y == 1))

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.04,
    'num_leaves': 31,
    'max_depth': 6,
    'scale_pos_weight': spw,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

def eval_subset(feat_names_sub):
    sub_indices = [fnames.index(f) for f in feat_names_sub]
    X_sub = X[:, sub_indices]
    oof_pred = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X_sub, y, groups):
        trn_d = lgb.Dataset(X_sub[trn_idx], label=y[trn_idx])
        val_d = lgb.Dataset(X_sub[val_idx], label=y[val_idx], reference=trn_d)
        m = lgb.train(lgb_params, trn_d, num_boost_round=400, valid_sets=[val_d], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof_pred[val_idx] = m.predict(X_sub[val_idx])
    auc = roc_auc_score(y, oof_pred)
    f1 = f1_score(y, (oof_pred >= 0.5).astype(int))
    return auc, f1

# 베이스라인 36D
base_auc, base_f1 = eval_subset(fnames)
print(f"=== 베이스라인 36D 성능: AUC={base_auc:.4f}, 2-State F1={base_f1:.4f} ===")

# 1. 완전 중복 및 극심한 다중공선성/노이즈 피처 9개 1차 제거
redundant_to_remove = [
    'femur_horizontal_level_ratio',       # drop_ratio와 r=1.000 동일
    'lead_knee_delta_from_standing',      # rom_ratio와 r=1.000 동일
    'front_view_femur_drop_gated',        # drop_ratio와 r=0.965 동일
    'lead_rear_knee_ratio',               # angle_diff와 r=0.954 동일
    'lead_knee_visibility_mean',          # gain=0.00 (무가치)
    'lead_knee_angle_mean',               # angle_min과 r=0.940 동일
    'lead_knee_90_error_abs',             # 단순 90도 임계값 오차 (노이즈)
    'lead_knee_under_90_ratio',           # 단순 90도 체류비 (노이즈)
    'bottom_phase_dwell_ratio',           # gain 하위 2위
]

cleaned_features = [f for f in fnames if f not in redundant_to_remove]
print(f"1단계 중복/노이즈 제거 후 피처 수: 36D -> {len(cleaned_features)}D")
auc1, f1_1 = eval_subset(cleaned_features)
print(f"1단계 제거 후 성능: AUC={auc1:.4f} (베이스라인 대비 {auc1-base_auc:+.4f}), F1={f1_1:.4f} (베이스라인 대비 {f1_1-base_f1:+.4f})")

# 2. LOFO (Leave-One-Feature-Out)
print(f"\n=== [2] {len(cleaned_features)}개 피처 대상 LOFO 분석 (하나씩 뺐을 때 성능 변화) ===")
lofo_results = []
for f in cleaned_features:
    cand_subset = [x for x in cleaned_features if x != f]
    auc, f1 = eval_subset(cand_subset)
    diff_f1 = f1 - f1_1
    diff_auc = auc - auc1
    lofo_results.append((f, auc, f1, diff_f1, diff_auc))

lofo_results.sort(key=lambda x: x[3], reverse=True)

print(f"{'피처명':<36s} | {'제거시 F1':<10s} | {'F1 변화량':<10s} | {'제거시 AUC':<10s} | {'판정'}")
print("-" * 85)
for f, auc, f1, df1, dauc in lofo_results:
    verdict = '제거 추천 (노이즈)' if df1 > 0 else '유지 (핵심 피처)'
    print(f"{f:<36s} | {f1:.4f}     | {df1:>+8.4f}   | {auc:.4f}     | {verdict}")

# 3. LOFO에서 F1을 떨어뜨리는(제거하면 오히려 좋은) 노이즈 피처들 추가 배제한 최종 정예 피처셋 도출
noise_features = [f for f, auc, f1, df1, dauc in lofo_results if df1 > 0]
elite_features = [f for f in cleaned_features if f not in noise_features]
print(f"\n최종 정예 피처 수: {len(elite_features)}D")
print(f"제거된 노이즈 피처 목록 ({len(noise_features)}개): {noise_features}")

auc_final, f1_final = eval_subset(elite_features)
print(f"\n=== 최종 정예 피처셋({len(elite_features)}D) 성능 ===")
print(f"  - ROC-AUC: {auc_final:.4f} (36D 베이스라인 대비 {auc_final-base_auc:+.4f})")
print(f"  - 2-State F1: {f1_final:.4f} (36D 베이스라인 대비 {f1_final-base_f1:+.4f})")

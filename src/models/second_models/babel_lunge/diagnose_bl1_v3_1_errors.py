# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: diagnose_bl1_v3_1_errors.py
Purpose: bl1_v3.1 오분류 샘플(FP, FN) 및 취약 도메인(피험자2, view3) 정밀 진단
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

V3_1_CACHE = r"D:\Data\피처_데이터\bl1_v3_1_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v3_1_evaluation_results.json"
MODEL_BUNDLE = r"D:\src\models\second_models\babel_lunge\bl1_v3_1_bundle.pkl"

with open(V3_1_CACHE, "rb") as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
groups = data['groups']
views = data['views']
clip_names = data['clip_names']
feature_names = data['feature_names']

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    import json
    res = json.load(f)

tau_clean = res['best_3state']['tau_clean']
tau_defect = res['best_3state']['tau_defect']

# 5-fold OOF 예측값 재현 (bundle 또는 kfold)
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

sgkf = StratifiedGroupKFold(n_splits=5)
n_samples = len(y)
oof_preds = np.zeros(n_samples, dtype=np.float32)

spw = float(np.sum(y == 0) / np.sum(y == 1))

lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'learning_rate': 0.04, 'num_leaves': 23, 'max_depth': 5, 'scale_pos_weight': spw,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 3.0, 'reg_alpha': 0.5,
    'min_child_samples': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
}
cat_params = {
    'iterations': 400, 'learning_rate': 0.04, 'depth': 5, 'scale_pos_weight': spw,
    'l2_leaf_reg': 3.0, 'rsm': 0.7, 'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 40
}
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'learning_rate': 0.04,
    'max_depth': 5, 'scale_pos_weight': spw, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_lambda': 3.0, 'reg_alpha': 0.5, 'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
}

w = [0.343, 0.450, 0.207]

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    m_lgb = lgb.train(lgb_params, train_data, num_boost_round=350, valid_sets=[val_data], callbacks=[lgb.early_stopping(35, verbose=False)])
    p_lgb = m_lgb.predict(X_val, num_iteration=m_lgb.best_iteration)

    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    p_cat = m_cat.predict_proba(X_val)[:, 1]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
    p_xgb = m_xgb.predict(dval)

    oof_preds[val_idx] = w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb

# 상태 결정
states = np.zeros(n_samples, dtype=int)
states[oof_preds < tau_clean] = 0
states[oof_preds >= tau_defect] = 1
states[(oof_preds >= tau_clean) & (oof_preds < tau_defect)] = 2

decided = (states != 2)
y_dec = y[decided]
pred_dec = states[decided]
probs_dec = oof_preds[decided]
groups_dec = groups[decided]
views_dec = views[decided]
clips_dec = clip_names[decided]
X_dec = X[decided]

fp_mask = (y_dec == 0) & (pred_dec == 1)
fn_mask = (y_dec == 1) & (pred_dec == 0)

print(f"[*] 총 확정 샘플 수: {len(y_dec)}개")
print(f"[*] False Positive (FP, 정상인데 결함 판정): {np.sum(fp_mask)}개")
print(f"[*] False Negative (FN, 결함인데 정상 판정): {np.sum(fn_mask)}개")

df_err = pd.DataFrame({
    'clip': clips_dec[fp_mask | fn_mask],
    'subj': groups_dec[fp_mask | fn_mask],
    'view': views_dec[fp_mask | fn_mask],
    'y_true': y_dec[fp_mask | fn_mask],
    'pred': pred_dec[fp_mask | fn_mask],
    'prob': probs_dec[fp_mask | fn_mask],
    'err_type': np.where(y_dec[fp_mask | fn_mask] == 0, 'FP', 'FN')
})

print("\n=== 오분류(FP/FN) 피험자별 분포 ===")
print(df_err['subj'].value_counts())

print("\n=== 오분류(FP/FN) 시점별 분포 ===")
print(df_err['view'].value_counts())

print("\n=== 피험자2의 특성 분석 ===")
p2_idx = (groups == '피험자2')
p_other_idx = (groups != '피험자2')

print(f"피험자2 샘플 수: {np.sum(p2_idx)}개 (정상: {np.sum(y[p2_idx]==0)}, 결함: {np.sum(y[p2_idx]==1)})")

# 주요 피처 평균 비교
print("\n피처별 평균 비교 (피험자2 vs 타 피험자):")
for f_idx, fname in enumerate(feature_names[:10]):
    mean_p2 = np.mean(X[p2_idx, f_idx])
    mean_oth = np.mean(X[p_other_idx, f_idx])
    print(f"  {fname:<30}: 피험자2={mean_p2:.2f} vs 타피험자={mean_oth:.2f}")

for f_idx in [15, 16]:  # 신규 수직거리 피처
    fname = feature_names[f_idx]
    mean_p2 = np.mean(X[p2_idx, f_idx])
    mean_oth = np.mean(X[p_other_idx, f_idx])
    print(f"  {fname:<30}: 피험자2={mean_p2:.2f} vs 타피험자={mean_oth:.2f}")

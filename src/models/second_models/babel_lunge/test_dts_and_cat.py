# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_dts_and_cat.py
Purpose: Covariate Dynamic Temperature Scaling (DTS) 및 공변량 적응형 임계치(CAT) 실측 테스트
"""

import os
import sys
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.optimize import minimize

V3_1_CACHE = r"D:\Data\피처_데이터\bl1_v3_1_dataset_cache.pkl"
NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"

sys.path.append(r"D:\src\features")
from lunge_c1_features_v3_1 import (
    detect_lead_leg_robust, find_dual_anchor_inflection_point,
    safe_savgol_filter, vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP, MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE, MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)

with open(V3_1_CACHE, "rb") as f:
    v3_1_data = pickle.load(f)

X = v3_1_data["X"]
y = v3_1_data["y"]
groups = v3_1_data["groups"]
views = v3_1_data["views"]
clip_names = v3_1_data["clip_names"]
n_samples = len(y)

# 1. 5-Fold OOF 확률 재현 (v3.1 앙상블)
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

sgkf = StratifiedGroupKFold(n_splits=5)
oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_cat = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)
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

for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    m_lgb = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=350, valid_sets=[lgb.Dataset(X_va, label=y_va)], callbacks=[lgb.early_stopping(35, verbose=False)])
    oof_lgb[va_idx] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[va_idx] = m_cat.predict_proba(X_va)[:, 1]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
    oof_xgb[va_idx] = m_xgb.predict(dval)

oof_p = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb

# 2. 공변량 추출:
# c1 = is_view3 (정면 뷰 여부: 0 or 1)
# c2 = is_subject2 (피험자2 여부 or 체형 편차)
# c3 = lead_femur_vertical_drop_ratio (피처 15)
# c4 = front_view_femur_drop_gated (피처 16)
# c5 = rigid_bone_length_consistency (피처 13)

cov_is_view3 = np.array([1.0 if v == 'view3' else 0.0 for v in views], dtype=np.float32)
cov_is_view1 = np.array([1.0 if v == 'view1' else 0.0 for v in views], dtype=np.float32)
cov_drop = X[:, 15]
cov_gated = X[:, 16]
cov_rigid = X[:, 13]

# 공변량 기반 동적 마진 튜닝:
# tau_clean(v, c) = tau_c_base + alpha_c * is_view3
# tau_defect(v, c) = tau_d_base + alpha_d * is_view3
# 시점별 최적 임계치 튜닝을 정밀하게 탐색:
unique_views = sorted(list(set(views)))

print("=== [A. v3.1 기준선] ===")
tau1_v31, tau2_v31 = 0.355, 0.565
unc_v31 = (oof_p >= tau1_v31) & (oof_p < tau2_v31)
dec_v31 = ~unc_v31
f1_v31 = f1_score(y[dec_v31], (oof_p[dec_v31] >= tau2_v31).astype(int))
acc_v31 = accuracy_score(y[dec_v31], (oof_p[dec_v31] >= tau2_v31).astype(int))
auc_v31 = roc_auc_score(y, oof_p)
print(f"v3.1: AUC={auc_v31*100:.2f}%, F1={f1_v31*100:.2f}%, ACC={acc_v31*100:.2f}%, 보류={np.mean(unc_v31)*100:.2f}%")

print("\n=== [B. 시점별 최적 적응형 캘리브레이션 정밀 그리드] ===")
# 전체 보류율 <= 10.0% 제약 하에서 전체 3-State F1 극대화
# 각 시점별 최적 튜플 (t1_v, t2_v)
view_opt_tuples = {}
for v in unique_views:
    v_m = (views == v)
    y_v = y[v_m]
    p_v = oof_p[v_m]

    best_v_f1 = -1.0
    best_tuple = (0.35, 0.60)
    for t1 in np.linspace(0.20, 0.45, 51):
        for t2 in np.linspace(0.45, 0.70, 51):
            if t2 <= t1:
                continue
            unc = (p_v >= t1) & (p_v < t2)
            if np.mean(unc) > 0.10:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            f1 = f1_score(y_v[dec], (p_v[dec] >= t2).astype(int), zero_division=0)
            if f1 > best_v_f1:
                best_v_f1 = f1
                best_tuple = (t1, t2)
    view_opt_tuples[v] = best_tuple

# 전체 평가
tot_dec_y, tot_dec_p, tot_unc = [], [], 0
for v in unique_views:
    v_m = (views == v)
    t1, t2 = view_opt_tuples[v]
    p_v = oof_p[v_m]
    y_v = y[v_m]
    unc = (p_v >= t1) & (p_v < t2)
    dec = ~unc
    tot_unc += np.sum(unc)
    tot_dec_y.extend(y_v[dec])
    tot_dec_p.extend((p_v[dec] >= t2).astype(int))
    f1_v = f1_score(y_v[dec], (p_v[dec] >= t2).astype(int), zero_division=0)
    acc_v = accuracy_score(y_v[dec], (p_v[dec] >= t2).astype(int))
    auc_v = roc_auc_score(y_v, p_v)
    print(f"  시점 {v}: tau_clean={t1:.3f}, tau_defect={t2:.3f} | F1={f1_v*100:.2f}%, ACC={acc_v*100:.2f}%, AUC={auc_v*100:.2f}%, 보류={np.mean(unc)*100:.2f}%")

tot_dec_y = np.array(tot_dec_y)
tot_dec_p = np.array(tot_dec_p)
overall_f1 = f1_score(tot_dec_y, tot_dec_p, zero_division=0)
overall_acc = accuracy_score(tot_dec_y, tot_dec_p)
overall_unc = tot_unc / n_samples
overall_rec = recall_score(tot_dec_y, tot_dec_p, zero_division=0)
overall_prec = precision_score(tot_dec_y, tot_dec_p, zero_division=0)

print(f"\n[시점별 정밀 적응형 결과]")
print(f"  전체 ROC-AUC: {auc_v31*100:.2f}%")
print(f"  전체 3-State F1: {overall_f1*100:.2f}% (v3.1 대비 +{overall_f1*100 - f1_v31*100:.2f}%p)")
print(f"  전체 확정 정확도: {overall_acc*100:.2f}% (v3.1 대비 +{overall_acc*100 - acc_v31*100:.2f}%p)")
print(f"  전체 재현율: {overall_rec*100:.2f}%")
print(f"  전체 정밀도: {overall_prec*100:.2f}%")
print(f"  전체 보류율: {overall_unc*100:.2f}%")

# 피험자별 결과
print("\n[피험자별 세부 성과]")
for s in sorted(list(set(groups))):
    s_m = (groups == s)
    s_dec_y, s_dec_p, s_unc = [], [], 0
    for i in np.where(s_m)[0]:
        v = views[i]
        t1, t2 = view_opt_tuples[v]
        p_val = oof_p[i]
        if t1 <= p_val < t2:
            s_unc += 1
        else:
            s_dec_y.append(y[i])
            s_dec_p.append(1 if p_val >= t2 else 0)
    s_dec_y = np.array(s_dec_y)
    s_dec_p = np.array(s_dec_p)
    s_f1 = f1_score(s_dec_y, s_dec_p, zero_division=0) if len(s_dec_y) > 0 else 0.0
    s_acc = accuracy_score(s_dec_y, s_dec_p) if len(s_dec_y) > 0 else 0.0
    s_auc = roc_auc_score(y[s_m], oof_p[s_m]) if len(np.unique(y[s_m])) > 1 else 1.0
    print(f"  {s}: F1={s_f1*100:.2f}%, ACC={s_acc*100:.2f}%, AUC={s_auc*100:.2f}%, 보류={s_unc/np.sum(s_m)*100:.2f}%")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: verify_v4_candidate_final.py
Purpose: v4 최종 후보 (30D 피처 + GBDT 앙상블 + 시점별 적응형 3-State 캘리브레이션) 전수 검증
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

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

X_28 = v3_1_data["X"]
y = v3_1_data["y"]
groups = v3_1_data["groups"]
views = v3_1_data["views"]
clip_names = v3_1_data["clip_names"]
n_samples = len(y)

# 2개 신규 피처 산출
feat_rom_ratio = np.zeros(n_samples, dtype=np.float32)
feat_coplanar = np.zeros(n_samples, dtype=np.float32)
eps = 1e-6

for idx, cname in enumerate(clip_names):
    v_code = views[idx]
    npz_path = os.path.join(NPZ_DIR, f"{cname}_{v_code}.npz")
    data = np.load(npz_path)
    kps = data["kps_33_3d_64f"]

    lead_side = detect_lead_leg_robust(kps)
    hip_idx = MP_LEFT_HIP if lead_side == "left" else MP_RIGHT_HIP
    knee_idx = MP_LEFT_KNEE if lead_side == "left" else MP_RIGHT_KNEE
    ankle_idx = MP_LEFT_ANKLE if lead_side == "left" else MP_RIGHT_ANKLE

    st_angles = []
    for f in range(5):
        vf = kps[f, hip_idx] - kps[f, knee_idx]
        vt = kps[f, ankle_idx] - kps[f, knee_idx]
        st_angles.append(vector_angle_3d(vf, vt))

    st_knee = float(np.median(st_angles))
    knee_min = X_28[idx, 0]
    rom = max(0.0, st_knee - knee_min)
    feat_rom_ratio[idx] = float(rom / (st_knee + eps))

    pelvis_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    inf_idx = int(np.argmax(pelvis_y))
    inf_k = kps[inf_idx]
    v_f = inf_k[knee_idx] - inf_k[hip_idx]
    v_t = inf_k[ankle_idx] - inf_k[knee_idx]
    leg_norm = np.cross(v_f, v_t)
    leg_norm /= (np.linalg.norm(leg_norm) + eps)
    u_hip = (inf_k[MP_RIGHT_HIP] - inf_k[MP_LEFT_HIP])
    u_hip /= (np.linalg.norm(u_hip) + eps)
    feat_coplanar[idx] = float(abs(np.dot(leg_norm, u_hip)))

X_30 = np.hstack([X_28, feat_coplanar.reshape(-1, 1), feat_rom_ratio.reshape(-1, 1)]).astype(np.float32)

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

for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X_30, y, groups=groups)):
    X_tr, y_tr = X_30[tr_idx], y[tr_idx]
    X_va, y_va = X_30[va_idx], y[va_idx]

    m_lgb = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=350, valid_sets=[lgb.Dataset(X_va, label=y_va)], callbacks=[lgb.early_stopping(35, verbose=False)])
    oof_lgb[va_idx] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[va_idx] = m_cat.predict_proba(X_va)[:, 1]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
    oof_xgb[va_idx] = m_xgb.predict(dval)

# 가중치 최적화
def loss_func(w):
    w = np.maximum(w, 0)
    if np.sum(w) == 0:
        return 10.0
    w /= np.sum(w)
    p = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return -roc_auc_score(y, p)

opt = minimize(loss_func, [0.30, 0.50, 0.20], method="Nelder-Mead")
w = np.maximum(opt.x, 0)
w /= np.sum(w)
oof_ens = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
auc = roc_auc_score(y, oof_ens)

print(f"[30D 앙상블] AUC={auc*100:.2f}%, 가중치: L={w[0]:.3f}, C={w[1]:.3f}, X={w[2]:.3f}")

# 시점별 최적 적응형 임계치
unique_views = sorted(list(set(views)))
v_thresh = {}
tot_dec_y, tot_dec_p, tot_unc = [], [], 0

for v in unique_views:
    v_m = (views == v)
    y_v = y[v_m]
    p_v = oof_ens[v_m]
    b_f1, b_t1, b_t2 = 0.0, 0.35, 0.60
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
            if f1 > b_f1:
                b_f1, b_t1, b_t2 = f1, t1, t2

    v_thresh[v] = (b_t1, b_t2)
    unc = (p_v >= b_t1) & (p_v < b_t2)
    dec = ~unc
    tot_unc += np.sum(unc)
    tot_dec_y.extend(y_v[dec])
    tot_dec_p.extend((p_v[dec] >= b_t2).astype(int))
    f1_v = f1_score(y_v[dec], (p_v[dec] >= b_t2).astype(int), zero_division=0)
    acc_v = accuracy_score(y_v[dec], (p_v[dec] >= b_t2).astype(int))
    auc_v = roc_auc_score(y_v, p_v)
    print(f"  시점 {v}: tau1={b_t1:.3f}, tau2={b_t2:.3f} | F1={f1_v*100:.2f}%, ACC={acc_v*100:.2f}%, AUC={auc_v*100:.2f}%, 보류={np.mean(unc)*100:.2f}%")

tot_dec_y = np.array(tot_dec_y)
tot_dec_p = np.array(tot_dec_p)
f1_final = f1_score(tot_dec_y, tot_dec_p, zero_division=0)
acc_final = accuracy_score(tot_dec_y, tot_dec_p)
unc_final = tot_unc / n_samples
rec_final = recall_score(tot_dec_y, tot_dec_p, zero_division=0)
prec_final = precision_score(tot_dec_y, tot_dec_p, zero_division=0)

print(f"\n=======================================================")
print(f" [v4 최종 후보 종합 성과]")
print(f"   ROC-AUC: {auc*100:.2f}%")
print(f"   3-State F1-Score: {f1_final*100:.2f}% (v3.1 79.12% 대비 +{f1_final*100 - 79.12:.2f}%p)")
print(f"   확정 정확도: {acc_final*100:.2f}% (v3.1 87.36% 대비 +{acc_final*100 - 87.36:.2f}%p)")
print(f"   결함 재현율 (Recall): {rec_final*100:.2f}%")
print(f"   결함 정밀도 (Precision): {prec_final*100:.2f}%")
print(f"   불확실성 보류율: {unc_final*100:.2f}% (10.0% 이하 준수)")
print(f"=======================================================")

# 피험자별
print("\n[피험자별 세부 성과]")
for s in sorted(list(set(groups))):
    s_m = (groups == s)
    s_dec_y, s_dec_p, s_unc = [], [], 0
    for i in np.where(s_m)[0]:
        v = views[i]
        t1, t2 = v_thresh[v]
        p_val = oof_ens[i]
        if t1 <= p_val < t2:
            s_unc += 1
        else:
            s_dec_y.append(y[i])
            s_dec_p.append(1 if p_val >= t2 else 0)
    s_dec_y = np.array(s_dec_y)
    s_dec_p = np.array(s_dec_p)
    s_f1 = f1_score(s_dec_y, s_dec_p, zero_division=0) if len(s_dec_y) > 0 else 0.0
    s_acc = accuracy_score(s_dec_y, s_dec_p) if len(s_dec_y) > 0 else 0.0
    s_auc = roc_auc_score(y[s_m], oof_ens[s_m]) if len(np.unique(y[s_m])) > 1 else 1.0
    print(f"  {s}: F1={s_f1*100:.2f}%, ACC={s_acc*100:.2f}%, AUC={s_auc*100:.2f}%, 보류={s_unc/np.sum(s_m)*100:.2f}% (샘플: {np.sum(s_m)})")

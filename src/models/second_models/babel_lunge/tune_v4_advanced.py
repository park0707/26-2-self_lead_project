# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tune_v4_advanced.py
Purpose: v4 고도화 최적화 (피처 정예화 + 앙상블 최적화 + 시점별/체형별 적응형 3-State 캘리브레이션)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from scipy.optimize import minimize

NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
V3_1_CACHE = r"D:\Data\피처_데이터\bl1_v3_1_dataset_cache.pkl"

sys.path.append(r"D:\src\features")
from lunge_c1_features_v3_1 import (
    detect_lead_leg_robust,
    find_dual_anchor_inflection_point,
    safe_savgol_filter,
    vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)

with open(V3_1_CACHE, "rb") as f:
    v3_1_data = pickle.load(f)

X_v3_1 = v3_1_data["X"]
y = v3_1_data["y"]
groups = v3_1_data["groups"]
views = v3_1_data["views"]
clip_names = v3_1_data["clip_names"]
feature_names = list(v3_1_data["feature_names"])
n_samples = len(y)

# 1. ROM 피처 및 PACS 피처 정밀 산출
feat_rom = np.zeros(n_samples, dtype=np.float32)
feat_coplanar = np.zeros(n_samples, dtype=np.float32)
feat_morph = np.zeros(n_samples, dtype=np.float32)

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

    # 직립 무릎 각도 (초기 5프레임)
    st_angles = []
    f_lens, t_lens = [], []
    for f in range(5):
        vf = kps[f, hip_idx] - kps[f, knee_idx]
        vt = kps[f, ankle_idx] - kps[f, knee_idx]
        st_angles.append(vector_angle_3d(vf, vt))
        f_lens.append(np.linalg.norm(vf))
        t_lens.append(np.linalg.norm(vt))

    st_knee = float(np.median(st_angles))
    knee_min = X_v3_1[idx, 0]
    feat_rom[idx] = float(st_knee - knee_min)
    feat_morph[idx] = float(np.median(f_lens) / max(0.1, np.median(t_lens)))

    # 최저점 coplanarity
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

# 정규화된 ROM (직립 대비 굴곡 비율: rom / st_knee)
feat_rom_ratio = (feat_rom / (feat_standing_knee_arr := (X_v3_1[:, 0] + feat_rom) + eps)).astype(np.float32)

print("[*] 피처 엔지니어링 완료")

# 피처셋 구성 테스트:
# Option 1: v3.1 28D 그대로 사용 (시점별 적응형 캘리브레이션 고도화)
# Option 2: 29D (v3.1 + feat_coplanar)
# Option 3: 29D (v3.1 + feat_rom_ratio)
# Option 4: 30D (v3.1 + feat_coplanar + feat_rom_ratio)

options = {
    "v3.1 (28D)": X_v3_1,
    "v4-Opt1 (29D: +coplanar)": np.hstack([X_v3_1, feat_coplanar.reshape(-1, 1)]),
    "v4-Opt2 (29D: +rom_ratio)": np.hstack([X_v3_1, feat_rom_ratio.reshape(-1, 1)]),
    "v4-Opt3 (30D: +coplanar+rom_ratio)": np.hstack([X_v3_1, feat_coplanar.reshape(-1, 1), feat_rom_ratio.reshape(-1, 1)])
}

def train_and_eval_option(X_in, opt_name):
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

    for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X_in, y, groups=groups)):
        X_tr, y_tr = X_in[tr_idx], y[tr_idx]
        X_va, y_va = X_in[va_idx], y[va_idx]

        m_lgb = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=350, valid_sets=[lgb.Dataset(X_va, label=y_va)], callbacks=[lgb.early_stopping(35, verbose=False)])
        oof_lgb[va_idx] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[va_idx] = m_cat.predict_proba(X_va)[:, 1]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
        oof_xgb[va_idx] = m_xgb.predict(dval)

    # 앙상블 가중치 Nelder-Mead
    def loss_func(w):
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        p = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -roc_auc_score(y, p)

    opt = minimize(loss_func, [0.34, 0.45, 0.21], method="Nelder-Mead")
    w = np.maximum(opt.x, 0)
    w /= np.sum(w)
    oof_ens = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    auc = roc_auc_score(y, oof_ens)

    # A. 전역 단일 임계치
    best_f1, best_acc, best_tau1, best_tau2, best_unc = 0.0, 0.0, 0.4, 0.6, 0.0
    for t1 in np.linspace(0.25, 0.45, 41):
        for t2 in np.linspace(0.45, 0.65, 41):
            if t2 <= t1:
                continue
            unc = (oof_ens >= t1) & (oof_ens < t2)
            if np.mean(unc) > 0.10:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            f1 = f1_score(y[dec], (oof_ens[dec] >= t2).astype(int), zero_division=0)
            acc = accuracy_score(y[dec], (oof_ens[dec] >= t2).astype(int))
            if f1 > best_f1:
                best_f1, best_acc, best_tau1, best_tau2, best_unc = f1, acc, t1, t2, np.mean(unc)

    # B. 시점별 적응형 임계치
    unique_views = sorted(list(set(views)))
    v_thresh = {}
    tot_dec_y, tot_dec_p, tot_unc = [], [], 0
    for v_code in unique_views:
        v_mask = (views == v_code)
        y_v = y[v_mask]
        oof_v = oof_ens[v_mask]
        b_f1, b_t1, b_t2 = 0.0, 0.35, 0.60
        for t1 in np.linspace(0.20, 0.45, 26):
            for t2 in np.linspace(0.45, 0.70, 26):
                if t2 <= t1:
                    continue
                unc = (oof_v >= t1) & (oof_v < t2)
                if np.mean(unc) > 0.10:
                    continue
                dec = ~unc
                if np.sum(dec) == 0:
                    continue
                f1 = f1_score(y_v[dec], (oof_v[dec] >= t2).astype(int), zero_division=0)
                if f1 > b_f1:
                    b_f1, b_t1, b_t2 = f1, t1, t2
        v_thresh[v_code] = (b_t1, b_t2)
        unc = (oof_v >= b_t1) & (oof_v < b_t2)
        dec = ~unc
        tot_unc += np.sum(unc)
        tot_dec_y.extend(y_v[dec])
        tot_dec_p.extend((oof_v[dec] >= b_t2).astype(int))

    tot_dec_y = np.array(tot_dec_y)
    tot_dec_p = np.array(tot_dec_p)
    f1_adp = f1_score(tot_dec_y, tot_dec_p, zero_division=0)
    acc_adp = accuracy_score(tot_dec_y, tot_dec_p)
    unc_adp = tot_unc / n_samples

    print(f"=== {opt_name} ===")
    print(f"  ROC-AUC: {auc*100:.2f}% | 가중치: L={w[0]:.3f}, C={w[1]:.3f}, X={w[2]:.3f}")
    print(f"  [전역 임계치] 3-State F1: {best_f1*100:.2f}% | ACC: {best_acc*100:.2f}% | 보류: {best_unc*100:.2f}%")
    print(f"  [시점 적응형] 3-State F1: {f1_adp*100:.2f}% | ACC: {acc_adp*100:.2f}% | 보류: {unc_adp*100:.2f}%")
    return {
        "opt_name": opt_name, "auc": auc, "w": w,
        "f1_global": best_f1, "acc_global": best_acc, "unc_global": best_unc,
        "f1_adp": f1_adp, "acc_adp": acc_adp, "unc_adp": unc_adp,
        "v_thresh": v_thresh, "oof_ens": oof_ens
    }

results = []
for name, X_mat in options.items():
    res = train_and_eval_option(X_mat, name)
    results.append(res)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_ensemble_and_xgboost.py
Purpose: dl2_rtmp_1 앙상블 다양성(CatBoost + LightGBM + XGBoost) 및 온도 스케일링 검증
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8')

CACHE_PATH = r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz"
data = np.load(CACHE_PATH, allow_pickle=True)
X_all = data["X"]
y = data["y"]
subjects = data["subjects"]
views = data["views"]
feature_names = list(data["feature_names"])

SELECTED_12D = [
    "RTMP_lead_hip_adduction_bottom",
    "RTMP_lead_knee_valgus_peak",
    "RTMP_lead_heel_toe_skew",
    "RTMP_lead_foot_axis_stability",
    "RTMP_fppa_eccentric_to_bottom_delta",
    "RTMP_npkd_bottom",
    "MP_trunk_pelvis_yaw_torsion",
    "MP_lead_foot_inward_ratio",
    "RTMP_lead_foot_inward_ratio",
    "RTMP_dkcd_medial_deviation_bottom",
    "RTMP_mkto_velocity",
    "RTMP_pelvic_rotation_bottom"
]
idx_12d = [feature_names.index(fn) for fn in SELECTED_12D]
X = X_all[:, idx_12d]
N, D = X.shape

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(N, dtype=np.float32)
oof_cat = np.zeros(N, dtype=np.float32)
oof_xgb = np.zeros(N, dtype=np.float32)

n_pos = np.sum(y == 1)
n_neg = np.sum(y == 0)
cost_weight = (float(n_neg) / float(n_pos)) * 1.35

for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    
    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=180, max_depth=4, num_leaves=13,
        learning_rate=0.033, min_child_samples=15,
        colsample_bytree=0.75, subsample=0.85,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
        random_state=42 + fold, verbose=-1, n_jobs=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]
    
    # 2. CatBoost
    clf_cat = CatBoostClassifier(
        iterations=250, depth=5, learning_rate=0.030,
        l2_leaf_reg=4.0, auto_class_weights='Balanced',
        random_seed=42 + fold, verbose=0, thread_count=-1
    )
    clf_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]
    
    # 3. XGBoost
    clf_xgb = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.030,
        subsample=0.85, colsample_bytree=0.75,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
        random_state=42 + fold, verbosity=0, n_jobs=-1
    )
    clf_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_va)[:, 1]

auc_lgb = roc_auc_score(y, oof_lgb)
auc_cat = roc_auc_score(y, oof_cat)
auc_xgb = roc_auc_score(y, oof_xgb)

print(f"[*] LightGBM OOF AUC : {auc_lgb:.4f}")
print(f"[*] CatBoost OOF AUC : {auc_cat:.4f}")
print(f"[*] XGBoost OOF AUC  : {auc_xgb:.4f}")

# Blending grid search
best_blend_auc = 0.0
best_w = None
for w_l in np.arange(0.1, 0.9, 0.1):
    for w_c in np.arange(0.1, 0.9 - w_l, 0.1):
        w_x = round(1.0 - w_l - w_c, 2)
        if w_x < 0.05:
            continue
        p = w_l * oof_lgb + w_c * oof_cat + w_x * oof_xgb
        a = roc_auc_score(y, p)
        if a > best_blend_auc:
            best_blend_auc = a
            best_w = (round(w_l, 2), round(w_c, 2), round(w_x, 2))

print(f"[+] 최적 앙상블 가중치: LightGBM={best_w[0]}, CatBoost={best_w[1]}, XGBoost={best_w[2]}")
print(f"[+] 앙상블 블렌딩 OOF AUC: {best_blend_auc:.4f} (단일 최고 대비 {best_blend_auc - max(auc_lgb, auc_cat, auc_xgb):+.4f})")

oof_blend = best_w[0] * oof_lgb + best_w[1] * oof_cat + best_w[2] * oof_xgb

# Per-view calibration check with blended probabilities
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

for max_retake in [0.10, 0.15]:
    print(f"\n[+] 보류율 제한 <= {max_retake*100:.0f}% 결과:")
    per_view_res = {}
    mask_all = np.zeros(N, dtype=bool)
    preds_all = np.zeros(N, dtype=int)
    for v in VIEW_ORDER:
        v_mask = (views == v)
        y_v = y[v_mask]
        p_v = oof_blend[v_mask]
        auc_v = roc_auc_score(y_v, p_v)
        best_v_f1 = f1_score(y_v, (p_v >= 0.5).astype(int), zero_division=0)
        best_cfg_v = (0.499, 0.501, 0.0)
        for tc in np.arange(0.15, 0.65, 0.005):
            for td in np.arange(tc + 0.005, 0.85, 0.005):
                m = (p_v <= tc) | (p_v >= td)
                retake = 1.0 - np.mean(m)
                if retake <= max_retake + 0.001:
                    s = f1_score(y_v[m], (p_v[m] >= td).astype(int), zero_division=0)
                    c_f1 = f1_score(1 - y_v[m], (p_v[m] < tc).astype(int), zero_division=0)
                    if c_f1 >= 0.50 and s > best_v_f1:
                        best_v_f1 = s
                        best_cfg_v = (float(tc), float(td), float(retake))
        per_view_res[v] = (auc_v, best_v_f1, best_cfg_v)
        m_v = v_mask & ((oof_blend <= best_cfg_v[0]) | (oof_blend >= best_cfg_v[1]))
        mask_all |= m_v
        preds_all[m_v] = (oof_blend[m_v] >= best_cfg_v[1]).astype(int)

    total_f1 = f1_score(y[mask_all], preds_all[mask_all], zero_division=0)
    total_acc = accuracy_score(y[mask_all], preds_all[mask_all])
    print(f"  - 전체 3-State F1: {total_f1*100:.2f}%, 정확도: {total_acc*100:.2f}%, 커버리지: {np.mean(mask_all)*100:.2f}% (보류율: {(1-np.mean(mask_all))*100:.2f}%)")
    for v in VIEW_ORDER:
        auc_v, f1_v, cfg_v = per_view_res[v]
        star = " ★ [70% 돌파!]" if f1_v >= 0.70 else ""
        print(f"    {v} ({VIEW_ALIAS_MAP[v]}): F1={f1_v*100:.2f}%{star:<12} | AUC={auc_v:.4f} | tau={cfg_v[0]:.3f}/{cfg_v[1]:.3f} | Retake={cfg_v[2]*100:.1f}%")

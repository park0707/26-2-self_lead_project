# -*- coding: utf-8 -*-
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

d_fair = np.load(r"D:\rtmp_fair_dataset.npz", allow_pickle=True)
d_2d = np.load(r"D:\Data\피처_데이터\lunge_c5_rtmp_3_features.npz", allow_pickle=True)

y = d_fair['y_c5'].astype(int)
groups = d_fair['subjects'].astype(int)
views = d_fair['views'].astype(str)
X_base = d_fair['X_c5'].astype(np.float32)
X_2d = d_2d['X'].astype(np.float32)

from src.features.lunge_c5_features_v5 import LUNGE_C5_V5_FEATURE_NAMES
from src.features.lunge_c5_features_rtmp_3 import LUNGE_C5_RTMP_3_FEATURE_NAMES

inter_names = [
    'trunk_to_thigh_compression_ratio',
    'shoulder_to_hip_velocity_ratio',
    'cervico_angle_x_trunk_comp',
    'torso_angle_minus_tibia_parallel',
    'shoulder_hip_x_disp_diff',
    'vspa_peak_x_trunk_comp'
]

extra_feats = np.column_stack([
    X_2d[:, 8] / (X_2d[:, 9] + 1e-4),
    X_2d[:, 35] / (X_2d[:, 36] + 1e-4),
    X_2d[:, 18] * X_2d[:, 8],
    X_2d[:, 14] - X_2d[:, 26],
    np.abs(X_2d[:, 33] - X_2d[:, 34]),
    X_base[:, 1] * X_2d[:, 8]
])

X_all = np.hstack([X_base, X_2d, extra_feats])
names = list(LUNGE_C5_V5_FEATURE_NAMES) + list(LUNGE_C5_RTMP_3_FEATURE_NAMES) + inter_names

dead_names = {
    'lunge_pelvic_tilt_sagittal',
    'lunge_trunk_knee_moment_arm',
    'lunge_virtual_sagittal_pitch_3d',
    'lunge_head_torso_hyperext_sync',
    'lunge_pelvis_spine_decoupling_angle',
    'v_neck_pelvis_y'
}
keep_idx = [i for i, n in enumerate(names) if n not in dead_names]
X = X_all[:, keep_idx]
names_opt = [names[i] for i in keep_idx]
print(f"Clean optimized features: {len(names_opt)}D (dropped {len(dead_names)} dead/duplicate features)")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Grid of hyperparameters to evaluate
configs = [
    {'name': 'CB_d6_lr03_l2_4_spw1.2', 'depth': 6, 'lr': 0.03, 'l2': 4.0, 'spw': 1.2, 'iter': 400},
    {'name': 'CB_d7_lr03_l2_6_spw1.3', 'depth': 7, 'lr': 0.03, 'l2': 6.0, 'spw': 1.3, 'iter': 400},
    {'name': 'CB_d5_lr04_l2_3_spw1.2', 'depth': 5, 'lr': 0.04, 'l2': 3.0, 'spw': 1.2, 'iter': 450},
    {'name': 'CB_d6_lr025_l2_8_spw1.3', 'depth': 6, 'lr': 0.025, 'l2': 8.0, 'spw': 1.3, 'iter': 500},
]

for cfg in configs:
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        cb = CatBoostClassifier(iterations=cfg['iter'], depth=cfg['depth'], learning_rate=cfg['lr'], l2_leaf_reg=cfg['l2'], scale_pos_weight=cfg['spw'], verbose=0, random_seed=42+fold, thread_count=4)
        cb.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=35)
        oof[va] = cb.predict_proba(X[va])[:, 1]
    
    y_pred = np.full(len(y), 2, dtype=int)
    for v in np.unique(views):
        idx = (views == v)
        y_v, oof_v = y[idx], oof[idx]
        best_f1_v, best_tau_v = 0, (0.5, 0.5)
        for t_norm in np.arange(0.15, 0.85, 0.005):
            for t_def in np.arange(t_norm, 0.90, 0.005):
                p = np.where(oof_v >= t_def, 1, np.where(oof_v < t_norm, 0, 2))
                cov = (p != 2)
                ret = 1.0 - np.mean(cov)
                if ret > 0.120: continue
                if np.sum(cov) > 0:
                    f1_v = f1_score(y_v[cov], p[cov], zero_division=0)
                    if f1_v > best_f1_v:
                        best_f1_v, best_tau_v = f1_v, (round(t_norm, 3), round(t_def, 3))
        y_pred[idx] = np.where(oof_v >= best_tau_v[1], 1, np.where(oof_v < best_tau_v[0], 0, 2))
        
    cov_all = (y_pred != 2)
    f1_3s = f1_score(y[cov_all], y_pred[cov_all])
    ret_3s = 1.0 - np.mean(cov_all)
    f1_2s = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    print(f"{cfg['name']} -> 2S: {f1_2s*100:.2f}% | AUC: {auc:.4f} | 3S F1: {f1_3s*100:.2f}% (Retake: {ret_3s*100:.2f}%)")

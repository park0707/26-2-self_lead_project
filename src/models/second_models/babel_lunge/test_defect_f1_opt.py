# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from lunge_c2_features_v4 import detect_lead_leg_v4

prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
c = np.load("lunge_c2_features_v4_cache.npz", allow_pickle=True)
clip_names = c["clip_names"]
y = c["y"]
subjects = c["subjects"]
views = c["views"]
from lunge_c2_features_v5 import FEATURE_INDICES_16D
X_16 = c["X"][:, FEATURE_INDICES_16D]

# 1. delta_q_angle 추출
delta_q_list = []
delta_knee_disp_list = []

for name in clip_names:
    p = os.path.join(prep_dir, name + ".npz")
    d = np.load(p)
    kps = d["kps_33_3d_64f"][:, :, :3]
    is_left, t_bot = detect_lead_leg_v4(kps)
    hip_idx = 23 if is_left else 24
    knee_idx = 25 if is_left else 26
    ank_idx = 27 if is_left else 28
    toe_idx = 31 if is_left else 32
    heel_idx = 29 if is_left else 30
    lat_sign = 1.0 if is_left else -1.0

    # delta_q_angle
    v_thigh_std = np.mean(kps[:3, knee_idx, :3] - kps[:3, hip_idx, :3], axis=0)
    v_shank_std = np.mean(kps[:3, ank_idx, :3] - kps[:3, knee_idx, :3], axis=0)
    q_std = np.degrees(np.arccos(np.clip(np.dot(v_thigh_std, v_shank_std) / (np.linalg.norm(v_thigh_std)*np.linalg.norm(v_shank_std) + 1e-6), -1.0, 1.0)))

    v_thigh_bot = kps[t_bot, knee_idx, :3] - kps[t_bot, hip_idx, :3]
    v_shank_bot = kps[t_bot, ank_idx, :3] - kps[t_bot, knee_idx, :3]
    q_bot = np.degrees(np.arccos(np.clip(np.dot(v_thigh_bot, v_shank_bot) / (np.linalg.norm(v_thigh_bot)*np.linalg.norm(v_shank_bot) + 1e-6), -1.0, 1.0)))
    delta_q = float(q_bot - q_std)
    delta_q_list.append(delta_q)

    # knee displacement vs foot angle
    v_foot_xz = np.array([kps[t_bot, toe_idx, 0] - kps[t_bot, heel_idx, 0], kps[t_bot, toe_idx, 2] - kps[t_bot, heel_idx, 2]], dtype=np.float32)
    u_f = v_foot_xz / (np.linalg.norm(v_foot_xz) + 1e-6)
    v_disp_xz = np.array([kps[t_bot, knee_idx, 0] - kps[0, knee_idx, 0], kps[t_bot, knee_idx, 2] - kps[0, knee_idx, 2]], dtype=np.float32)
    norm_d = np.linalg.norm(v_disp_xz)
    if norm_d > 0.02:
        u_d = v_disp_xz / norm_d
        ang_kf = float(np.arccos(np.clip(np.dot(u_f, u_d), -1.0, 1.0)) * 180.0 / np.pi)
    else:
        ang_kf = 0.0
    delta_knee_disp_list.append(ang_kf)

X_18 = np.column_stack([X_16, delta_q_list, delta_knee_disp_list]).astype(np.float32)
print("X_18 shape:", X_18.shape)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cost_sensitive_weight = (float(np.sum(y==0)) / float(np.sum(y==1))) * 1.35

oof_lgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_18, y, groups=subjects), 1):
    X_tr, y_tr = X_18[train_idx], y[train_idx]
    X_va, y_va = X_18[val_idx], y[val_idx]

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=130, max_depth=3, num_leaves=7, learning_rate=0.040,
        min_child_samples=18, colsample_bytree=0.75, subsample=0.85,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_sensitive_weight,
        random_state=42+fold, verbose=-1, n_jobs=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

    clf_cat = CatBoostClassifier(
        iterations=160, depth=4, learning_rate=0.038, l2_leaf_reg=3.5,
        auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=-1
    )
    clf_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]

p_blend = 0.5 * oof_lgb + 0.5 * oof_cat
print("AUC:", roc_auc_score(y, p_blend))

# 서브에이전트 제안: Defect F1 직접 최적화 그리드 탐색 (보류율 <= 10.0%)
best_f1 = 0
best_stats = {}
thresholds = np.linspace(0.46, 0.58, 61)

for tn in thresholds:
    for td in thresholds:
        if td <= tn:
            continue
        dec = (p_blend <= tn) | (p_blend >= td)
        retake_rate = 1.0 - np.mean(dec)
        if retake_rate <= 0.1001:
            y_d = y[dec]
            p_d = (p_blend[dec] >= td).astype(int)
            d_f1 = f1_score(y_d, p_d, zero_division=0)
            c_f1 = f1_score(1 - y_d, 1 - p_d, zero_division=0)
            acc = accuracy_score(y_d, p_d)
            prec = precision_score(y_d, p_d, zero_division=0)
            rec = recall_score(y_d, p_d, zero_division=0)
            cm = confusion_matrix(y_d, p_d)
            if d_f1 > best_f1:
                best_f1 = d_f1
                best_stats = {
                    "tn": tn, "td": td, "retake": retake_rate, "cov": 1.0 - retake_rate,
                    "defect_f1": d_f1, "clean_f1": c_f1, "macro_f1": (d_f1+c_f1)/2.0,
                    "acc": acc, "prec": prec, "rec": rec, "cm": cm
                }

print("\n=== Defect F1 Direct Maximized Result ===")
print(f"Defect F1: {best_stats['defect_f1']*100:.2f}% | Clean F1: {best_stats['clean_f1']*100:.2f}% | Macro F1: {best_stats['macro_f1']*100:.2f}%")
print(f"Accuracy: {best_stats['acc']*100:.2f}% | Precision: {best_stats['prec']*100:.2f}% | Recall: {best_stats['rec']*100:.2f}%")
print(f"Retake: {best_stats['retake']*100:.2f}% (Coverage: {best_stats['cov']*100:.2f}%) | tn={best_stats['tn']:.4f}, td={best_stats['td']:.4f}")
print(f"Confusion Matrix: TN={best_stats['cm'][0,0]}, FP={best_stats['cm'][0,1]}, FN={best_stats['cm'][1,0]}, TP={best_stats['cm'][1,1]}")

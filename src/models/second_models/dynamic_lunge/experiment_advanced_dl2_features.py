# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: experiment_advanced_dl2_features.py
Purpose: 다이나믹 런지 c2 (dl2_rtmp_1) 성능 극대화를 위한
         고도화 2D/3D 생체역학 피처 및 시점/동작 인터랙션 피처 검증
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BABEL_LUNGE_DIR = r"D:\src\models\second_models\babel_lunge"
if BABEL_LUNGE_DIR not in sys.path:
    sys.path.insert(0, BABEL_LUNGE_DIR)

CACHE_PATH = r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz"
data = np.load(CACHE_PATH, allow_pickle=True)
X_46 = data["X"]
y = data["y"]
subjects = data["subjects"]
views = data["views"]
ex_types = data["ex_types"]
clip_names = data["clip_names"]

N = len(y)
print(f"[*] Base 46D 캐시 로드 완료: {N}개 시퀀스")

# 1. View One-Hot (5D)
view_codes = ['view5', 'view2', 'view3', 'view4', 'view1']
X_view = np.zeros((N, 5), dtype=np.float32)
for j, v in enumerate(view_codes):
    X_view[:, j] = (views == v).astype(np.float32)

# 2. Exercise Type One-Hot (1D: is_forward)
X_ex = (ex_types == '05_스텝_포워드').astype(np.float32).reshape(-1, 1)

# 3. 추가 RTMPose 고유 2D 기하학 피처 8D 추출
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]

print("[*] 추가 RTMPose 2D 정밀 기하 피처 추출 중...")
X_extra_rtmp = np.zeros((N, 8), dtype=np.float32)
# Features:
# 1. rtmp_ankle_knee_x_offset_bot (무릎-발목 횡이탈)
# 2. rtmp_knee_toe_x_offset_bot (무릎-발끝 횡이탈)
# 3. rtmp_toe_heel_x_diff_bot (발 내회전 2D 직접 측정)
# 4. rtmp_foot_length_bot
# 5. rtmp_knee_valgus_2d_angle_bot (2D 고관절-무릎-발목 각도)
# 6. rtmp_torso_tilt_bot (어깨-골반 수직축 기울기)
# 7. rtmp_knee_travel_x (하강 시 무릎의 순수 횡방향 총 이동거리)
# 8. rtmp_knee_ankle_x_ratio (골반 너비 대비 무릎-발목 횡편차 비)

from lunge_c2_features_rtmp_4 import detect_lead_leg_2d
from lunge_c2_features_hybrid import map_halpe26_to_mp33_2d

for idx in range(N):
    fname = f"{clip_names[idx]}_{views[idx]}.npz"
    rtmp_path = None
    for rd in RTMP_DIRS:
        c = os.path.join(rd, fname)
        if os.path.exists(c):
            rtmp_path = c
            break
    if not rtmp_path:
        continue
    
    rtmp_kps_26 = np.load(rtmp_path)['kps_26_64f'] # (64, 26, 3)
    kps_2d_mp = map_halpe26_to_mp33_2d(rtmp_kps_26) # (64, 33, 2)
    is_left_lead, t_bot = detect_lead_leg_2d(kps_2d_mp)
    
    side_mult = -1.0 if is_left_lead else 1.0
    hip_idx = 23 if is_left_lead else 24
    knee_idx = 25 if is_left_lead else 26
    ank_idx = 27 if is_left_lead else 28
    heel_idx = 29 if is_left_lead else 30
    toe_idx = 31 if is_left_lead else 32
    
    pelvis_w = np.linalg.norm(kps_2d_mp[t_bot, 23] - kps_2d_mp[t_bot, 24]) + 1e-5
    
    # 1. ankle-knee x offset normalized
    ak_x = (kps_2d_mp[t_bot, knee_idx, 0] - kps_2d_mp[t_bot, ank_idx, 0]) * side_mult / pelvis_w
    # 2. knee-toe x offset normalized
    kt_x = (kps_2d_mp[t_bot, knee_idx, 0] - kps_2d_mp[t_bot, toe_idx, 0]) * side_mult / pelvis_w
    # 3. toe-heel x diff normalized
    th_x = (kps_2d_mp[t_bot, toe_idx, 0] - kps_2d_mp[t_bot, heel_idx, 0]) * side_mult / pelvis_w
    # 4. foot length normalized
    foot_len = np.linalg.norm(kps_2d_mp[t_bot, toe_idx] - kps_2d_mp[t_bot, heel_idx]) / pelvis_w
    
    # 5. 2D FPPA / Knee valgus angle at bot
    v_thigh = kps_2d_mp[t_bot, hip_idx] - kps_2d_mp[t_bot, knee_idx]
    v_shank = kps_2d_mp[t_bot, ank_idx] - kps_2d_mp[t_bot, knee_idx]
    cos_ang = np.dot(v_thigh, v_shank) / ((np.linalg.norm(v_thigh) + 1e-5) * (np.linalg.norm(v_shank) + 1e-5))
    knee_ang_2d = float(np.arccos(np.clip(cos_ang, -1.0, 1.0)) * 180.0 / np.pi)
    
    # 6. Torso tilt
    mid_sh = 0.5 * (kps_2d_mp[t_bot, 11] + kps_2d_mp[t_bot, 12])
    mid_hip = 0.5 * (kps_2d_mp[t_bot, 23] + kps_2d_mp[t_bot, 24])
    torso_tilt = (mid_sh[0] - mid_hip[0]) * side_mult / pelvis_w
    
    # 7. Knee travel X during eccentric
    knee_travel_x = np.std(kps_2d_mp[:max(t_bot, 5), knee_idx, 0]) / pelvis_w
    
    # 8. Ankle-knee vertical ratio
    ak_dist = np.linalg.norm(kps_2d_mp[t_bot, knee_idx] - kps_2d_mp[t_bot, ank_idx]) / pelvis_w
    
    X_extra_rtmp[idx] = [ak_x, kt_x, th_x, foot_len, knee_ang_2d, torso_tilt, knee_travel_x, ak_dist]

X_extra_rtmp = np.nan_to_num(X_extra_rtmp, nan=0.0, posinf=0.0, neginf=0.0)
print("[+] 추가 RTMPose 8D 추출 완료!")

# Concatenate full feature bank: 46D + 8D + 5D(view) + 1D(ex) = 60D
X_all_60 = np.concatenate([X_46, X_extra_rtmp, X_view, X_ex], axis=1)
print(f"[+] 종합 피처 뱅크 Shape: {X_all_60.shape}")

# Feature names list
extra_names = [
    "RTMP_extra_ankle_knee_x_offset",
    "RTMP_extra_knee_toe_x_offset",
    "RTMP_extra_toe_heel_x_diff",
    "RTMP_extra_foot_length_ratio",
    "RTMP_extra_knee_2d_fppa_angle",
    "RTMP_extra_torso_lateral_tilt",
    "RTMP_extra_knee_travel_x_std",
    "RTMP_extra_shank_length_ratio",
    "VIEW_view5", "VIEW_view2", "VIEW_view3", "VIEW_view4", "VIEW_view1",
    "EX_is_forward"
]
all_60_names = list(data["feature_names"]) + extra_names

# Evaluate different combinations
def evaluate_cv(X_mat, name):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    cost_weight = (float(n_neg) / float(n_pos)) * 1.35
    
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_mat, y, groups=subjects), 1):
        X_tr, y_tr = X_mat[tr_idx], y[tr_idx]
        X_va, y_va = X_mat[val_idx], y[val_idx]
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=180, max_depth=4, num_leaves=13,
            learning_rate=0.033, min_child_samples=15,
            colsample_bytree=0.75, subsample=0.85,
            reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        cat_clf = CatBoostClassifier(
            iterations=250, depth=5, learning_rate=0.030,
            l2_leaf_reg=4.0, auto_class_weights='Balanced',
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        lgb_clf.fit(X_tr, y_tr)
        cat_clf.fit(X_tr, y_tr)
        
        p_lgb = lgb_clf.predict_proba(X_va)[:, 1]
        p_cat = cat_clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = 0.50 * p_lgb + 0.50 * p_cat
        
    auc = roc_auc_score(y, oof_probs)
    f1_2s = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
    
    # 3-State search (retake <= 10%)
    best_3s_10 = f1_2s
    best_retake_10 = 0.0
    for tc in np.arange(0.25, 0.55, 0.01):
        for td in np.arange(tc + 0.01, 0.75, 0.01):
            m = (oof_probs <= tc) | (oof_probs >= td)
            r = 1.0 - np.mean(m)
            if r <= 0.1001:
                s = f1_score(y[m], (oof_probs[m] >= td).astype(int), zero_division=0)
                if s > best_3s_10:
                    best_3s_10 = s
                    best_retake_10 = r
                    
    # Per-view performance check
    per_view_3s = {}
    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    for v in view_order:
        mask_v = (views == v)
        y_v = y[mask_v]
        p_v = oof_probs[mask_v]
        auc_v = roc_auc_score(y_v, p_v)
        # 3-state per view <= 10%
        best_v_f1 = f1_score(y_v, (p_v >= 0.5).astype(int), zero_division=0)
        for tc in np.arange(0.20, 0.60, 0.01):
            for td in np.arange(tc + 0.01, 0.80, 0.01):
                m_v = (p_v <= tc) | (p_v >= td)
                r_v = 1.0 - np.mean(m_v)
                if r_v <= 0.1001:
                    s_v = f1_score(y_v[m_v], (p_v[m_v] >= td).astype(int), zero_division=0)
                    if s_v > best_v_f1:
                        best_v_f1 = s_v
        per_view_3s[v] = (auc_v, best_v_f1)
        
    print(f"[{name}] AUC={auc:.4f} | 2S-F1={f1_2s*100:.2f}% | 3S-F1(<=10%)={best_3s_10*100:.2f}% (Retake={best_retake_10*100:.1f}%)")
    v_strs = " | ".join([f"{v}: {per_view_3s[v][1]*100:.1f}% (AUC {per_view_3s[v][0]:.3f})" for v in view_order])
    print(f"   Views: {v_strs}")
    return oof_probs, auc, best_3s_10

print("\n" + "=" * 90)
print("  [실험 1: 46D Base vs 46D + RTMP Extra 8D vs 60D All]")
print("=" * 90)
oof_46, auc_46, f1_46 = evaluate_cv(X_46, "1. 46D Base")
oof_54, auc_54, f1_54 = evaluate_cv(np.concatenate([X_46, X_extra_rtmp], axis=1), "2. 46D + RTMP Extra 8D (54D)")
oof_60, auc_60, f1_60 = evaluate_cv(X_all_60, "3. 60D Full (Base + RTMP Extra + View + Ex)")

# Feature selection from 60D
print("\n[*] 60D 전체 피처 단변량 AUC 상위 20개:")
univariate = []
for i, fn in enumerate(all_60_names):
    a = roc_auc_score(y, X_all_60[:, i])
    univariate.append((max(a, 1 - a), a, i, fn))
univariate.sort(reverse=True)
for r, (eff, raw, i, fn) in enumerate(univariate[:20], 1):
    print(f"  {r:2d}. {fn:40s} : AUC={eff:.4f}")

# Test top 20 from 60D
top20_from_60 = [x[2] for x in univariate[:20]]
oof_top20_60, auc_top20_60, f1_top20_60 = evaluate_cv(X_all_60[:, top20_from_60], "4. 60D 중 단변량 상위 20D")

# Forward selection from top 30
top30_idx = [x[2] for x in univariate[:30]]
print("\n[*] 60D 기반 Forward Selection 진행 중...")
selected_60 = [univariate[0][2]]
best_score = 0.0
for step in range(20):
    best_cand = None
    best_cand_score = best_score
    for cand in top30_idx:
        if cand in selected_60:
            continue
        curr_test = selected_60 + [cand]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(N, dtype=np.float32)
        X_sub = X_all_60[:, curr_test]
        for tr, va in sgkf.split(X_sub, y, groups=subjects):
            clf = lgb.LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.04,
                                     scale_pos_weight=1.8, random_state=42, verbose=-1, n_jobs=-1)
            clf.fit(X_sub[tr], y[tr])
            oof[va] = clf.predict_proba(X_sub[va])[:, 1]
        score = roc_auc_score(y, oof)
        if score > best_cand_score:
            best_cand_score = score
            best_cand = cand
    if best_cand is not None:
        selected_60.append(best_cand)
        best_score = best_cand_score
    else:
        break

print(f"[*] Forward Selected 피처 개수: {len(selected_60)}, Quick AUC={best_score:.4f}")
print("선택된 피처:")
for idx in selected_60:
    print(f"  - {all_60_names[idx]}")

oof_fs_60, auc_fs_60, f1_fs_60 = evaluate_cv(X_all_60[:, selected_60], f"5. 60D Forward Selected {len(selected_60)}D")

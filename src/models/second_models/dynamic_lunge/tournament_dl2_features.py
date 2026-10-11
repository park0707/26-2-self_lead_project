# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tournament_dl2_features.py
Purpose: 다이나믹 런지 c2 (dl2_rtmp_1) 피처셋 및 모델 아키텍처 토너먼트
         - 1. bl2_final 계승 20D 정예 피처셋
         - 2. 순수 RTMPose 23D 피처셋 (RTMP 18D + RTMP Frontal 5D)
         - 3. 전체 46D 하이브리드 피처셋
         - 4. 다이나믹 런지 최적화 정예 피처셋 탐색
         - 5-Fold StratifiedGroupKFold (피험자 1~9번 완전 격리)
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

CACHE_PATH = r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz"
data = np.load(CACHE_PATH, allow_pickle=True)
X_all = data["X"]
y = data["y"]
subjects = data["subjects"]
views = data["views"]
ex_types = data["ex_types"]
feature_names = list(data["feature_names"])

# bl2_final selected 20 features
BL2_FINAL_20D = [
    "RTMP_lead_foot_axis_stability",
    "RTMP_fppa_eccentric_to_bottom_delta",
    "RTMP_lead_hip_adduction_bottom",
    "MP_lead_foot_axis_stability",
    "RTMP_knee_displacement_foot_heading_angle",
    "RTMP_dkcd_medial_deviation_bottom",
    "MP_knee_displacement_foot_heading_angle",
    "RTMP_lead_knee_ankle_alignment",
    "RTMP_ntmd_standing",
    "MP_pelvic_rotation_bottom",
    "MP_lead_knee_valgus_peak",
    "MP_ntmd_standing",
    "MP_trunk_pelvis_yaw_torsion",
    "RTMP_lead_foot_inward_angle_bottom",
    "RTMP_npkd_bottom",
    "RTMP_ntmd_bottom",
    "MP_lead_knee_velocity_ml",
    "MP_mkto_velocity",
    "RTMP_lead_heel_toe_skew",
    "RTMP_mkto_bottom"
]

def eval_feature_subset(feat_indices, name, cost_mult=1.35):
    X = X_all[:, feat_indices]
    N, D = X.shape
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    cost_weight = (float(n_neg) / float(n_pos)) * cost_mult

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=160, max_depth=4, num_leaves=13,
            learning_rate=0.033, min_child_samples=15,
            colsample_bytree=0.75, subsample=0.85,
            reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        cat_clf = CatBoostClassifier(
            iterations=220, depth=5, learning_rate=0.030,
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
    
    # 3-State sweep (retake <= 10%)
    best_3s_f1_10 = f1_2s
    best_cfg_10 = (0.5, 0.5, 0.0)
    for tc in np.arange(0.20, 0.55, 0.01):
        for td in np.arange(tc + 0.01, 0.80, 0.01):
            mask = (oof_probs <= tc) | (oof_probs >= td)
            retake = 1.0 - np.mean(mask)
            if retake <= 0.1001:
                s = f1_score(y[mask], (oof_probs[mask] >= td).astype(int), zero_division=0)
                if s > best_3s_f1_10:
                    best_3s_f1_10 = s
                    best_cfg_10 = (float(tc), float(td), float(retake))
                    
    # 3-State sweep (retake <= 15%)
    best_3s_f1_15 = f1_2s
    best_cfg_15 = (0.5, 0.5, 0.0)
    for tc in np.arange(0.20, 0.55, 0.01):
        for td in np.arange(tc + 0.01, 0.80, 0.01):
            mask = (oof_probs <= tc) | (oof_probs >= td)
            retake = 1.0 - np.mean(mask)
            if retake <= 0.1501:
                s = f1_score(y[mask], (oof_probs[mask] >= td).astype(int), zero_division=0)
                if s > best_3s_f1_15:
                    best_3s_f1_15 = s
                    best_cfg_15 = (float(tc), float(td), float(retake))

    return {
        'name': name,
        'dim': D,
        'auc': float(auc),
        'f1_2s': float(f1_2s),
        'f1_3s_10': float(best_3s_f1_10),
        'retake_10': best_cfg_10[2],
        'cfg_10': best_cfg_10,
        'f1_3s_15': float(best_3s_f1_15),
        'retake_15': best_cfg_15[2],
        'cfg_15': best_cfg_15,
        'oof_probs': oof_probs
    }

print("=" * 90)
print("  [다이나믹 런지 c2 토너먼트: 피처 구성별 OOF 성능 비교]")
print("=" * 90)

# Candidate 1: bl2_final 20D
idx_bl2_20 = [feature_names.index(fn) for fn in BL2_FINAL_20D if fn in feature_names]
res_bl2 = eval_feature_subset(idx_bl2_20, "bl2_final 20D 계승")

# Candidate 2: All 46D
res_46 = eval_feature_subset(list(range(46)), "46D 전체 하이브리드")

# Candidate 3: Pure RTMPose 23D (RTMP_로 시작하는 피처만)
idx_rtmp_23 = [i for i, fn in enumerate(feature_names) if fn.startswith("RTMP_")]
res_rtmp = eval_feature_subset(idx_rtmp_23, "순수 RTMPose 23D")

# Candidate 4: Univariate AUC 기반 상위 20D / 25D 탐색
univariate = []
for i, fn in enumerate(feature_names):
    a = roc_auc_score(y, X_all[:, i])
    univariate.append((max(a, 1 - a), a, i, fn))
univariate.sort(reverse=True)

print("\n[*] 46D 피처 단변량 ROC-AUC 상위 15개:")
for rank, (eff_a, raw_a, i, fn) in enumerate(univariate[:15], 1):
    print(f"  {rank:2d}. {fn:42s} : Effective AUC={eff_a:.4f} (Raw={raw_a:.4f})")

top_20_idx = [x[2] for x in univariate[:20]]
res_top20 = eval_feature_subset(top_20_idx, "단변량 상위 20D")

top_25_idx = [x[2] for x in univariate[:25]]
res_top25 = eval_feature_subset(top_25_idx, "단변량 상위 25D")

# Candidate 5: Greedy Forward Selection from top candidates
print("\n[*] Greedy Forward Selection 진행 중...")
selected = [univariate[0][2]]
best_score = 0.0
for step in range(25):
    best_cand = None
    best_cand_score = best_score
    for cand in range(46):
        if cand in selected:
            continue
        curr_test = selected + [cand]
        # Quick evaluation with LightGBM
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=np.float32)
        X_sub = X_all[:, curr_test]
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
        selected.append(best_cand)
        best_score = best_cand_score
    else:
        break

print(f"[*] Forward Selected 피처 개수: {len(selected)}, Quick AUC={best_score:.4f}")
res_forward = eval_feature_subset(selected, f"전진선택 정예 {len(selected)}D")

# Print summary table
candidates = [res_bl2, res_46, res_rtmp, res_top20, res_top25, res_forward]
print("\n" + "=" * 90)
print(f"{'후보 피처셋 명칭':<25} | {'차원':<4} | {'ROC-AUC':<8} | {'2-State F1':<10} | {'3-State F1 (<=10%)':<18} | {'3-State F1 (<=15%)':<18}")
print("-" * 90)
for c in candidates:
    print(f"{c['name']:<25} | {c['dim']:<4} | {c['auc']:.4f}   | {c['f1_2s']*100:6.2f}%   | {c['f1_3s_10']*100:6.2f}% (R:{c['retake_10']*100:4.1f}%) | {c['f1_3s_15']*100:6.2f}% (R:{c['retake_15']*100:4.1f}%)")
print("=" * 90)

# Check View C and other views for champion
champ = max(candidates, key=lambda x: (x['auc'], x['f1_3s_10']))
print(f"\n[★] 챔피언 피처셋 선정: {champ['name']} (AUC: {champ['auc']:.4f})")
if 'selected' in locals() and champ['name'].startswith("전진선택"):
    print("선택된 피처 목록:")
    for idx in selected:
        print(f"  - {feature_names[idx]}")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: verify_feature_recovery.py
Purpose: c2, c3, c4 피처 엔지니어링 결함 진단 및 완전한 도메인 정예 피처셋 복원 후
         5-Fold StratifiedGroupKFold 성능 회복 실측 검증
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"

def evaluate_submodel(X, y, groups, model_name=""):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        cb = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, verbose=False, random_seed=42 + fold)
        cb.fit(X[trn_idx], y[trn_idx])
        lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.04, num_leaves=25, verbose=-1, random_state=42 + fold)
        lgbm.fit(X[trn_idx], y[trn_idx])
        p = 0.60 * cb.predict_proba(X[val_idx])[:, 1] + 0.40 * lgbm.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = p
    
    preds = (oof >= 0.50).astype(int)
    f1 = f1_score(y, preds, zero_division=0) * 100.0
    prec = precision_score(y, preds, zero_division=0) * 100.0
    rec = recall_score(y, preds, zero_division=0) * 100.0
    try:
        auc = roc_auc_score(y, oof) * 100.0
    except Exception:
        auc = 50.0
    print(f"  [{model_name}] F1: {f1:.2f}% | AUC: {auc:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}%")
    return f1, auc

def main():
    print("=" * 90)
    print("  [vidio-health-trainer] c2, c3, c4 피처 결함 진단 및 성능 회복 실험")
    print("=" * 90)
    
    d = np.load(MASTER_PATH, allow_pickle=True)
    X_fa = d['X_format_a']
    X_c1_raw = d['X_c1']
    X_c2_raw = d['X_c2']
    X_c3_raw = d['X_c3']
    X_c4_raw = d['X_c4']
    Y_def = d['Y_defects_golden']
    y_norm = d['y_normal_golden']
    subjects = d['subjects']
    views = d['views']
    N = len(subjects)

    print(f"[*] 마스터 데이터 로드 완료: N={N}")

    # ==========================================
    # 1. 이전 train_and_evaluate_v6_true_64f.py 기준 (문제 재현)
    # ==========================================
    print("\n--- 1. 기존 슬라이싱 버그 상태의 성능 (재현) ---")
    c2_fa_idx_old = list(range(90, 95)) + list(range(40, 50))
    X_c2_old = np.concatenate([X_c2_raw, X_fa[:, c2_fa_idx_old]], axis=1)
    
    c3_fa_idx_old = list(range(50, 60)) + list(range(65, 70)) + list(range(75, 80)) + list(range(0, 10))
    X_c3_old = np.concatenate([X_c3_raw, X_fa[:, c3_fa_idx_old]], axis=1)

    c4_fa_idx_old = list(range(20, 30)) + list(range(70, 75)) + list(range(120, 130)) + list(range(110, 120))
    X_c4_old = np.concatenate([X_c4_raw, X_fa[:, c4_fa_idx_old]], axis=1)

    evaluate_submodel(X_c2_old, Y_def[:, 1], subjects, "c2 OLD (버그 상태)")
    evaluate_submodel(X_c3_old, Y_def[:, 2], subjects, "c3 OLD (버그 상태)")
    evaluate_submodel(X_c4_old, Y_def[:, 3], subjects, "c4 OLD (버그 상태)")

    # ==========================================
    # 2. Format A 올바른 인덱스 맵핑
    # ==========================================
    # FEATURE_NAMES_28_3D:
    # 0: knee_angle_l (0-4), 1: knee_angle_r (5-9), 2: hip_angle_l (10-14), 3: hip_angle_r (15-19)
    # 4: ankle_angle_l (20-24), 5: ankle_angle_r (25-29), 6: elbow_angle_l (30-34), 7: elbow_angle_r (35-39)
    # 8: torso_pitch (40-44), 9: torso_roll (45-49), 10: knee_valgus_l (50-54), 11: knee_valgus_r (55-59)
    # 12: sh_hip_twist (60-64), 13: knee_distance (65-69), 14: ankle_distance (70-74), 15: ankle_z_spread (75-79)
    # 16: knee_asymmetry (80-84), 17: hip_asymmetry (85-89), 18: neck_spine (90-94), 19: barbell_dist (95-99)
    # 20: knee_hip_synch (100-104), 21: signed_ankle_z_l (105-109), 22: signed_ankle_z_r (110-114)
    # 23: step_dx (115-119), 24: step_dz (120-124), 25: shank_vert_l (125-129), 26: shank_vert_r (130-134)
    # 27: hip_knee_rom (135-139)
    # Snapshots: 140~160 (21개)

    print("\n--- 2. Format A 정밀 인덱스 매핑 수정 후 성능 ---")
    # c3 올바른 인덱스:
    # knee_valgus_l/r(50-59), knee_dist(65-69), knee_asym(80-84), knee_angle_l/r(0-9), hip_asym(85-89), bottom_snapshots(140, 141, 150, 151)
    c3_fa_correct = list(range(50, 60)) + list(range(65, 70)) + list(range(80, 85)) + list(range(0, 10)) + list(range(85, 90)) + [140, 141, 150, 151]
    X_c3_corr = np.concatenate([X_c3_raw, X_fa[:, c3_fa_correct]], axis=1)
    evaluate_submodel(X_c3_corr, Y_def[:, 2], subjects, "c3 인덱스 수정 (48D)")

    # c4 올바른 인덱스:
    # ankle_angle_l/r(20-29), ankle_dist(70-74), ankle_z_spread(75-79), shank_vert_l/r(125-135), signed_ankle_z_l/r(105-115), bottom_snapshots(144, 145, 154, 158, 159)
    c4_fa_correct = list(range(20, 30)) + list(range(70, 80)) + list(range(125, 135)) + list(range(105, 115)) + [144, 145, 154, 158, 159]
    X_c4_corr = np.concatenate([X_c4_raw, X_fa[:, c4_fa_correct]], axis=1)
    evaluate_submodel(X_c4_corr, Y_def[:, 3], subjects, "c4 인덱스 수정 (54D)")

if __name__ == '__main__':
    main()

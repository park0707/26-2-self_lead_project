# -*- coding: utf-8 -*-
"""
Compare c3 models:
1. v3.0 Baseline (dynamic ankle width in ka_drop)
2. v4-A (fixed standing ankle width in ka_drop)
3. v4-B (dual ka_drop: dynamic + fixed standing)
"""
import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    # 1. Base v3 features (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    c3_24d = np.concatenate([v1_data_c3['feats'], foot_data_c3['foot_feats']], axis=1).astype(np.float32)
    X_v3 = c3_24d[:, C3_INDICES].copy()

    feat_names_c3 = list(v1_data_c3['feature_names']) + list(foot_data_c3['feature_names'])
    v3_names = [feat_names_c3[i] for i in C3_INDICES]
    ka_drop_idx = v3_names.index('c3_knee_ankle_ratio_drop')

    # 2. Extract fixed ka_drop
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    N, T, C = X_b_64.shape
    fixed_ka_drop = np.zeros(N, dtype=np.float32)

    for i in range(N):
        seq = X_b_64[i]
        knee_angle_mean = (seq[:, 56] + seq[:, 57]) / 2.0
        b = int(np.clip(np.argmax(gaussian_filter1d(180.0 - knee_angle_mean, 0.8)), 1, T - 2))

        knee_dist = seq[:, 69]
        ankle_dist = np.maximum(seq[:, 70], 0.05)
        stand_ankle = max(float(np.mean(ankle_dist[:max(1, T // 10)])), 0.05)

        ka_fixed = knee_dist / stand_ankle
        ka_fixed_s = gaussian_filter1d(ka_fixed, sigma=0.8)
        ka_standing = float(np.mean(ka_fixed_s[:max(1, T // 10)]))
        ka_nadir = float(ka_fixed_s[b])
        fixed_ka_drop[i] = max(0.0, ka_standing - ka_nadir)

    # Variant A: Replace ka_drop with fixed_ka_drop
    X_v3_fixed = X_v3.copy()
    X_v3_fixed[:, ka_drop_idx] = fixed_ka_drop

    # Variant B: Add fixed_ka_drop alongside original ka_drop
    X_v3_both = np.hstack([X_v3, fixed_ka_drop[:, None]])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_feats(X):
        oof = np.zeros(len(y_c3))
        for trn, val in sgkf.split(X, y_c3, groups=subjects):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(X[trn], y_c3[trn])
            clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf.fit(X[trn], y_c3[trn])
            oof[val] = 0.60 * cb.predict_proba(X[val])[:, 1] + 0.40 * clf.predict_proba(X[val])[:, 1]
        
        pred = (oof >= 0.50).astype(int)
        p = precision_score(y_c3, pred) * 100
        r = recall_score(y_c3, pred) * 100
        f = f1_score(y_c3, pred) * 100
        auc = roc_auc_score(y_c3, oof) * 100
        missed = int(np.sum((y_c3 == 1) & (pred == 0)))
        return {'p': p, 'r': r, 'f': f, 'auc': auc, 'missed': missed, 'oof': oof}

    print("[1/3] v3.0 기준선 평가 중...")
    res_orig = eval_feats(X_v3)
    print("[2/3] v4-A (기립 고정 발목 너비 1:1 대체) 평가 중...")
    res_fixed = eval_feats(X_v3_fixed)
    print("[3/3] v4-B (동적 비율 + 기립 고정 2중 융합) 평가 중...")
    res_both = eval_feats(X_v3_both)

    print("\n" + "=" * 100)
    print("  [c3 발목 너비 고정 실험 5-Fold 실측 성능 비교표 (N=1,906)]")
    print("=" * 100)
    print(f"{'모델 피처 구성':<36} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'ROC-AUC':<10} | {'놓친 결함'}")
    print("-" * 100)
    print(f"{'v3.0 기준선 (동적 발목 너비 분모)':<36} | {res_orig['p']:6.2f}%     | {res_orig['r']:6.2f}%     | {res_orig['f']:6.2f}%     | {res_orig['auc']:6.2f}%   | {res_orig['missed']}건")
    print(f"{'v4-A (기립 고정 발목 너비 1:1 대체)':<36} | {res_fixed['p']:6.2f}%     | {res_fixed['r']:6.2f}%     | {res_fixed['f']:6.2f}%     | {res_fixed['auc']:6.2f}%   | {res_fixed['missed']}건")
    print(f"{'v4-B (동적 비율 + 기립 고정 2중 융합)':<36} | {res_both['p']:6.2f}%     | {res_both['r']:6.2f}%     | {res_both['f']:6.2f}%     | {res_both['auc']:6.2f}%   | {res_both['missed']}건")
    print("=" * 100)

if __name__ == '__main__':
    main()

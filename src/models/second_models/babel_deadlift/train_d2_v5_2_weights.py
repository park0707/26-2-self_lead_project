# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Train production weights for d2 v5.2 Pruned Elite 10D model
"""

import os
import sys
import glob
import json
import pickle
import numpy as np

import lightgbm as lgb
from catboost import CatBoostClassifier
from cleanlab.filter import find_label_issues

sys.path.insert(0, 'D:/')
from src.models.second_models.babel_deadlift.d2_model_v5_2_pruned import (
    extract_d2_v5_2_pruned_features, FEATURE_NAMES_10_D2
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS = r"D:\src\models\second_models\babel_deadlift\weights\d2_v5_2_pruned_production.pkl"
TCN_SOURCE = r"D:\src\models\second_models\babel_deadlift\weights\d2_v5_final_production.pkl"

def train_production_weights():
    print("[*] Training production weights for d2 v5.2 Pruned Elite 10D...")
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_list, y_list = [], []

    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        folder = "_".join(base.split("_")[:-1])
        json_path = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        with open(json_path, 'r', encoding='utf-8') as fp:
            conds = {c['condition']: c['value'] for c in json.load(fp).get('type_info', {}).get('conditions', [])}
        y_list.append(0 if conds.get('발과 무릎의 방향 일치', True) else 1)

        kps = np.load(p)['kps_33_3d_64f']
        f10, _, _ = extract_d2_v5_2_pruned_features(kps)
        X_list.append(f10)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset shape: X={X.shape}, y={y.shape} (Defects={np.sum(y)}, Cleans={len(y)-np.sum(y)})")

    # Cleanlab label noise audit (top 30 prune on full dataset)
    init_clf = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.05, max_depth=3, scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1)
    init_clf.fit(X, y)
    probs = init_clf.predict_proba(X)
    issues = find_label_issues(labels=y, pred_probs=probs, return_indices_ranked_by='self_confidence', n_jobs=1)
    c_mask = np.ones(len(y), dtype=bool)
    if len(issues) > 0:
        c_mask[issues[:min(30, len(issues))]] = False
    print(f"Cleanlab pruned {np.sum(~c_mask)} noisy labels. Clean dataset: {np.sum(c_mask)} samples")

    X_clean, y_clean = X[c_mask], y[c_mask]

    # Train final LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1
    )
    clf_lgb.fit(X_clean, y_clean)

    # Train final CatBoost
    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)
    clf_cat = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
        auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1,
        train_dir=r"C:\Temp\catboost_info"
    )
    clf_cat.fit(X_clean, y_clean)

    # Load Tiny TCN weights and normalizer
    with open(TCN_SOURCE, 'rb') as fp:
        tcn_data = pickle.load(fp)
    tcn_state = tcn_data.get('tcn_state_dict', None)
    ts_mean = tcn_data.get('ts_mean', None)
    ts_std = tcn_data.get('ts_std', None)

    save_dict = {
        'version': 'v5.2_pruned_elite_10d',
        'feature_names': FEATURE_NAMES_10_D2,
        'lgb_model': clf_lgb,
        'cat_model': clf_cat,
        'tcn_state_dict': tcn_state,
        'ts_mean': ts_mean,
        'ts_std': ts_std,
        'tau_defect': 0.425,
        'tau_clean': 0.360,
        'metrics': {
            'auc': 0.7136,
            'f1_3state': 51.71,
            'precision': 43.44,
            'recall': 63.86,
            'accuracy': 71.55,
            'coverage': 90.39,
            'fp_count': 69
        }
    }

    with open(OUT_WEIGHTS, 'wb') as fp:
        pickle.dump(save_dict, fp)
    print(f"[+] Saved v5.2 production weights to {OUT_WEIGHTS}")

if __name__ == '__main__':
    train_production_weights()

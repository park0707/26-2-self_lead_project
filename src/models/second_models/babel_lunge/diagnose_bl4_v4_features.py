# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: diagnose_bl4_v4_features.py
Purpose: v4 30차원 피처들의 개별 기여도 및 노이즈 진단
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

CACHE_V4_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v4.npz"

def diagnose():
    data = np.load(CACHE_V4_PATH, allow_pickle=True)
    X = data['X_30d']
    y = data['y']
    fnames = list(data['feature_names'])
    subjects = data['subjects']

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    imps = np.zeros(len(fnames))

    for tr, va in sgkf.split(X, y, groups=subjects):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.04, max_depth=4, verbose=-1, random_state=42)
        clf.fit(X[tr], y[tr])
        imps += clf.feature_importances_ / 5.0

    df = pd.DataFrame({'feature': fnames, 'importance': imps}).sort_values(by='importance', ascending=False)
    print("=" * 60)
    print("  [v4 30D 피처별 Feature Importance 순위]")
    print("=" * 60)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        print(f" {rank:2d} | {row['feature']:30s} | {row['importance']:.2f}")

if __name__ == "__main__":
    diagnose()

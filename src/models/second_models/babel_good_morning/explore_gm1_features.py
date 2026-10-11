import os
import sys
import numpy as np
import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path: sys.path.insert(0, 'd:/')

from src.models.second_models.babel_stiff_deadlift.stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    FEATURE_NAMES_19D
)

CACHE_PATH = os.path.join(CURRENT_DIR, "gm1_dataset_cache.npz")

def main():
    print("[*] GM1 전체 19D 피처 탐색 및 중요도 분석")
    data = np.load(CACHE_PATH, allow_pickle=True)
    labels = data['labels']
    subject_ids = data['subject_ids']
    X_seqs = data['X']
    N = len(labels)

    # 1. 19D 피처 전수 추출
    feats_19d = np.zeros((N, 19), dtype=np.float32)
    for i in range(N):
        feats_19d[i] = extract_knee_invariance_features_v6(X_seqs[i], p_sd3=0.0)

    # 2. CatBoost를 사용한 피처 중요도 분석 (All 19D)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    importances = np.zeros(19, dtype=np.float32)
    
    for tr_idx, val_idx in sgkf.split(np.arange(N), labels, subject_ids):
        X_tr, y_tr = feats_19d[tr_idx], labels[tr_idx]
        cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=4, auto_class_weights='Balanced', verbose=0, random_seed=42)
        cb.fit(X_tr, y_tr)
        importances += cb.get_feature_importance()
        
    importances /= 5.0
    
    # 3. 피처 중요도 출력
    sorted_idx = np.argsort(importances)[::-1]
    print("\n[★ 19D 전체 피처 중요도 순위]")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"{rank:2d}. [{idx:2d}] {FEATURE_NAMES_19D[idx]:28s}: {importances[idx]:.2f}%")

    # 4. 피처 간 상관계수 분석 (다중공선성 제거 목적)
    df = pd.DataFrame(feats_19d, columns=FEATURE_NAMES_19D)
    corr_matrix = df.corr().abs()
    
    print("\n[★ 상관계수 (r >= 0.85) 쌍]")
    high_corr_pairs = []
    for i in range(19):
        for j in range(i+1, 19):
            if corr_matrix.iloc[i, j] >= 0.85:
                high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
                print(f"[{i:2d}] {FEATURE_NAMES_19D[i]} <-> [{j:2d}] {FEATURE_NAMES_19D[j]}: {corr_matrix.iloc[i, j]:.3f}")

if __name__ == '__main__':
    main()

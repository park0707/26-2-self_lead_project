# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: analyze_c3_feature_redundancy.py
Purpose: 바벨 런지 c3 18D 피처셋 전수 조사
         - 상관관계 분석 (|r| > 0.85 고상관 다중공선성 쌍 탐색)
         - CatBoost / LightGBM Feature Importance (Gain & Permutation)
         - 성능 저해(노이즈/과적합 유발) 피처 선별 및 v3 최적 피처 서브셋 도출
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v2 import FEATURE_NAMES_18D

CACHE_18D_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_18d.npz"

def main():
    print("=" * 80)
    print(" [1단계] 바벨 런지 c3 18D 피처셋 전수 조사 및 상관관계/기여도 정밀 분석")
    print("=" * 80)
    
    data = np.load(CACHE_18D_PATH, allow_pickle=True)
    X = data['X'] # (1085, 18)
    y = data['y'] # (1085,)
    groups = data['groups'] # (1085,)
    
    n_samples, n_feats = X.shape
    df_feat = pd.DataFrame(X, columns=FEATURE_NAMES_18D)
    
    # 1. 상관관계 매트릭스 계산 및 고상관 피처 쌍 (|r| >= 0.85) 추출
    corr_matrix = df_feat.corr().abs()
    high_corr_pairs = []
    
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            r_val = corr_matrix.iloc[i, j]
            if r_val >= 0.85:
                high_corr_pairs.append((FEATURE_NAMES_18D[i], FEATURE_NAMES_18D[j], r_val))
                
    print(f"\n[*] 1. 고상관(다중공선성, |r| >= 0.85) 피처 쌍 ({len(high_corr_pairs)}개 발견):")
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  - {f1} <--> {f2}: r = {r:.4f}")
        
    # 2. CatBoost 및 LightGBM 5-Fold Permutation Importance 및 Feature Importance 산출
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos = np.sum(y == 0) / np.sum(y == 1)
    
    cb_imp_gain = np.zeros(n_feats)
    lgb_imp_gain = np.zeros(n_feats)
    perm_imp_mean = np.zeros(n_feats)
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        # CatBoost
        cb = CatBoostClassifier(iterations=160, learning_rate=0.04, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        cb.fit(X_tr, y_tr)
        cb_imp_gain += cb.get_feature_importance()
        
        # LightGBM
        lgb_m = lgb.LGBMClassifier(n_estimators=140, learning_rate=0.035, max_depth=5, num_leaves=18, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=4)
        lgb_m.fit(X_tr, y_tr)
        lgb_imp_gain += lgb_m.feature_importances_
        
        # Permutation Importance (CatBoost 기준 검증셋에 대해 F1 점수 하락도 측정)
        r_perm = permutation_importance(cb, X_va, y_va, scoring='f1', n_repeats=5, random_state=42)
        perm_imp_mean += r_perm.importances_mean
        
    cb_imp_gain /= 5.0
    lgb_imp_gain /= 5.0
    perm_imp_mean /= 5.0
    
    df_eval = pd.DataFrame({
        'Feature': FEATURE_NAMES_18D,
        'CatBoost_Gain': cb_imp_gain,
        'LightGBM_Split': lgb_imp_gain,
        'Permutation_F1_Impact': perm_imp_mean
    }).sort_values(by='Permutation_F1_Impact', ascending=False)
    
    print("\n[*] 2. 18D 피처별 5-Fold 중요도 및 성능 기여도 (Permutation F1 Impact):")
    print(df_eval.to_string(index=False))
    
    # 3. 부정적 기여(Permutation F1 < 0 또는 Importance 최하위) 및 고상관 중복 피처 식별
    harmful_or_redundant = []
    for _, row in df_eval.iterrows():
        f = row['Feature']
        p_imp = row['Permutation_F1_Impact']
        cb_imp = row['CatBoost_Gain']
        if p_imp <= 0.0 or cb_imp < 1.0:
            harmful_or_redundant.append((f, p_imp, cb_imp))
            
    print(f"\n[*] 3. 성능 저해(노이즈/과적합) 또는 기여도 극저(중복) 피처 목록 ({len(harmful_or_redundant)}개):")
    for f, p, c in harmful_or_redundant:
        print(f"  - {f}: Permutation ΔF1 = {p:+.5f} | CatBoost Gain = {c:.2f}")

if __name__ == "__main__":
    main()

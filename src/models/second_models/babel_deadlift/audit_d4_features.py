# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반
Script: audit_d4_features.py
Description: 43D 피처셋 상관관계 및 5-Fold 피험자 격리 순열 중요도(Permutation Importance) 정밀 분석
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, 'd4_v4_features_cache.npz')

from d4_model_v5 import FEATURE_NAMES_43_D4

def run_feature_audit():
    print("=" * 80)
    print(" [1단계] 43D 피처 상관관계 및 다중공선성 정밀 감사")
    print("=" * 80)
    
    if not os.path.exists(CACHE_PATH):
        print(f"Error: Cache not found at {CACHE_PATH}")
        return
        
    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X_gbdt']  # (770, 43)
    y = data['y']       # (770,)
    subjects = data['subjects'] # (770,)
    views = data['views']       # (770,)
    
    N, D = X.shape
    print(f"Dataset: N={N}, Features={D}, Positives={np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
    
    df_features = pd.DataFrame(X, columns=FEATURE_NAMES_43_D4)
    corr_matrix = df_features.corr().abs()
    
    # 1. 고상관 피처 쌍 (|r| >= 0.85) 추출
    high_corr_pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            r_val = corr_matrix.iloc[i, j]
            if r_val >= 0.85:
                high_corr_pairs.append((FEATURE_NAMES_43_D4[i], FEATURE_NAMES_43_D4[j], r_val))
                
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n[고상관 피처 쌍 (|r| >= 0.85)] 총 {len(high_corr_pairs)}개 발견:")
    for feat_a, feat_b, r in high_corr_pairs:
        print(f"  - {feat_a} <--> {feat_b} : r = {r:.4f}")

    print("\n" + "=" * 80)
    print(" [2단계] 5-Fold StratifiedGroupKFold 기준 Tri-GBDT Permutation Importance 산출")
    print("=" * 80)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 베이스라인 모델 학습 및 OOF 예측 생성 함수
    def train_and_eval(X_curr, y_curr, subjects_curr):
        oof_preds = np.zeros(N)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_curr, y_curr, groups=subjects_curr)):
            X_tr, y_tr = X_curr[train_idx], y_curr[train_idx]
            X_va, y_va = X_curr[val_idx], y_curr[val_idx]
            
            # GBDT Tri-Ensemble
            model_lgb = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
                class_weight='balanced', random_state=42 + fold, verbose=-1
            )
            model_cat = CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
            )
            model_rf = RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced',
                random_state=42 + fold, n_jobs=-1
            )
            
            model_lgb.fit(X_tr, y_tr)
            model_cat.fit(X_tr, y_tr)
            model_rf.fit(X_tr, y_tr)
            
            p_lgb = model_lgb.predict_proba(X_va)[:, 1]
            p_cat = model_cat.predict_proba(X_va)[:, 1]
            p_rf = model_rf.predict_proba(X_va)[:, 1]
            
            oof_preds[val_idx] = 0.45 * p_lgb + 0.35 * p_cat + 0.20 * p_rf
            
        auc = roc_auc_score(y_curr, oof_preds)
        best_f1 = 0
        best_tau = 0.5
        for tau in np.arange(0.30, 0.85, 0.02):
            f1 = f1_score(y_curr, (oof_preds >= tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        return auc, best_f1, best_tau, oof_preds

    base_auc, base_f1, base_tau, oof_base = train_and_eval(X, y, subjects)
    print(f"Base 43D Tri-GBDT OOF: AUC={base_auc:.4f}, Best F1={base_f1*100:.2f}% (tau={base_tau:.2f})")
    
    # Permutation Importance: 각 피처를 셔플했을 때 OOF F1 저하량 (F1_base - F1_perm)
    # 저하량이 양수이면 중요한 피처(셔플 시 성능 하락), 음수이면 노이즈/독성 피처(셔플 시 오히려 성능 향상!)
    print("\nPermutation Feature Importance 계산 중 (43개 피처 셔플)...")
    perm_importances = []
    
    # 재현성을 위해 fold별 사전 학습된 모델들을 저장
    trained_models = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        model_lgb = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
            class_weight='balanced', random_state=42 + fold, verbose=-1
        )
        model_cat = CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.05,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
        )
        model_rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight='balanced',
            random_state=42 + fold, n_jobs=-1
        )
        model_lgb.fit(X_tr, y_tr)
        model_cat.fit(X_tr, y_tr)
        model_rf.fit(X_tr, y_tr)
        trained_models.append((train_idx, val_idx, model_lgb, model_cat, model_rf))
        
    for feat_idx in range(D):
        feat_name = FEATURE_NAMES_43_D4[feat_idx]
        oof_perm_preds = np.zeros(N)
        
        # 5회 반복 셔플 평균
        delta_f1_trials = []
        delta_auc_trials = []
        
        for trial in range(5):
            np.random.seed(100 + trial * 43 + feat_idx)
            for train_idx, val_idx, mlgb, mcat, mrf in trained_models:
                X_va_perm = X[val_idx].copy()
                # 해당 피처만 셔플
                shuffled_col = np.random.permutation(X_va_perm[:, feat_idx])
                X_va_perm[:, feat_idx] = shuffled_col
                
                p_lgb = mlgb.predict_proba(X_va_perm)[:, 1]
                p_cat = mcat.predict_proba(X_va_perm)[:, 1]
                p_rf = mrf.predict_proba(X_va_perm)[:, 1]
                oof_perm_preds[val_idx] = 0.45 * p_lgb + 0.35 * p_cat + 0.20 * p_rf
                
            trial_auc = roc_auc_score(y, oof_perm_preds)
            trial_f1 = f1_score(y, (oof_perm_preds >= base_tau).astype(int), zero_division=0)
            
            delta_f1 = base_f1 - trial_f1 # 양수면 중요, 음수면 셔플시 더 좋아짐(노이즈)
            delta_auc = base_auc - trial_auc
            delta_f1_trials.append(delta_f1)
            delta_auc_trials.append(delta_auc)
            
        mean_delta_f1 = np.mean(delta_f1_trials)
        mean_delta_auc = np.mean(delta_auc_trials)
        
        # 모델별 내장 중요도 (LGB Gain)
        lgb_gains = [mlgb.booster_.feature_importance(importance_type='gain')[feat_idx] for _, _, mlgb, _, _ in trained_models]
        mean_gain = np.mean(lgb_gains)
        
        perm_importances.append({
            'index': feat_idx,
            'name': feat_name,
            'mean_delta_f1': mean_delta_f1,
            'mean_delta_auc': mean_delta_auc,
            'lgb_gain': mean_gain
        })
        
    df_importance = pd.DataFrame(perm_importances)
    df_importance = df_importance.sort_values(by='mean_delta_f1', ascending=False)
    
    print("\n" + "=" * 80)
    print(" [Permutation Importance 감사 결과 (순열 중요도 순위)]")
    print("=" * 80)
    for idx, row in df_importance.iterrows():
        status = "★ 핵심" if row['mean_delta_f1'] > 0.005 else ("● 보통" if row['mean_delta_f1'] > 0.000 else "▼ 저해/노이즈(음수/0)")
        print(f"[{row['index']:2d}] {row['name']:<35} | ΔF1: {row['mean_delta_f1']*100:+6.2f}% | ΔAUC: {row['mean_delta_auc']:+7.4f} | LGB Gain: {row['lgb_gain']:8.1f} | {status}")
        
    # 저장
    df_importance.to_csv(os.path.join(CURRENT_DIR, 'd4_feature_importance_audit.csv'), index=False, encoding='utf-8-sig')
    print("\n결과 저장 완료: d4_feature_importance_audit.csv")

if __name__ == '__main__':
    run_feature_audit()

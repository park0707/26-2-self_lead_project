# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / Knee Valgus)
Version: v6.0 Two-Tier (2계층 생체역학 모호성 정제 모델)
Script: train_d2_v6_twotier_weights.py

- Level 4 (완전 역전 및 라벨 오염 27건) 전수 폐기
- Level 1~3 (358개 실무 정제 데이터셋) 기반 최종 프로덕션 가중치 학습
- 10D 초정예 생체역학 피처 + Dual GBDT (LightGBM + CatBoost) + Tiny TCN Meta-Fusion
"""

import os
import sys
import glob
import json
import pickle
import numpy as np

import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d2_model_v5_2_pruned import (
    extract_d2_v5_2_pruned_features, FEATURE_NAMES_10_D2
)

CLASSIFIED_CSV = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_ambiguity_classified.csv"
BENCHMARK_JSON = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_two_tier_benchmark.json"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
OUT_WEIGHTS = os.path.join(WEIGHTS_DIR, "d2_v6_twotier_production.pkl")
TCN_SOURCE = os.path.join(WEIGHTS_DIR, "d2_v5_final_production.pkl")

def train_production_weights():
    print("[*] Training production weights for d2 v6.0 Two-Tier Model...")
    import pandas as pd
    df = pd.read_csv(CLASSIFIED_CSV)
    
    # Prune Level 4 (27 samples)
    df_clean = df[df['ambiguity_level'] != 4].copy()
    print(f"Dataset after Level 4 Pruning: {len(df_clean)} samples (Defects: {df_clean['y_d2'].sum()}, Cleans: {len(df_clean)-df_clean['y_d2'].sum()})")
    
    X_clean = df_clean[FEATURE_NAMES_10_D2].values
    y_clean = df_clean['y_d2'].values
    
    # Train final LightGBM on Level 1~3
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1
    )
    clf_lgb.fit(X_clean, y_clean)
    
    # Train final CatBoost on Level 1~3
    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)
    clf_cat = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
        auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1,
        train_dir=r"C:\Temp\catboost_info"
    )
    clf_cat.fit(X_clean, y_clean)
    
    # Load Tiny TCN weights
    with open(TCN_SOURCE, 'rb') as fp:
        tcn_data = pickle.load(fp)
    tcn_state = tcn_data.get('tcn_state_dict', None)
    ts_mean = tcn_data.get('ts_mean', None)
    ts_std = tcn_data.get('ts_std', None)
    
    with open(BENCHMARK_JSON, 'r', encoding='utf-8') as fp:
        bench_data = json.load(fp)
        
    save_dict = {
        'version': 'v6.0_twotier_ambiguity_pruned',
        'feature_names': FEATURE_NAMES_10_D2,
        'lgb_model': clf_lgb,
        'cat_model': clf_cat,
        'tcn_state_dict': tcn_state,
        'ts_mean': ts_mean,
        'ts_std': ts_std,
        'tau_defect': 0.425,
        'tau_clean': 0.360,
        'tier1_metrics': bench_data['tier1_level_1_only'],
        'tier2_metrics': bench_data['tier2_level_1_to_3'],
        'baseline_metrics': bench_data['baseline_v5_2'],
        'view_breakdown': bench_data['view_breakdown']
    }
    
    with open(OUT_WEIGHTS, 'wb') as fp:
        pickle.dump(save_dict, fp)
    print(f"[+] Successfully trained and saved production weights to {OUT_WEIGHTS}")

if __name__ == '__main__':
    train_production_weights()

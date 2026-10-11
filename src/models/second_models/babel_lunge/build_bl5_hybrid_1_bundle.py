# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_bl5_hybrid_1_bundle.py
Purpose: bl5_hybrid_1 최종 프로덕션 앙상블 번들(CatBoost + LightGBM + XGBoost) 패키징 및 검증
"""

import os
import sys
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")
import pickle
import json
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

HYBRID_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_hybrid_1_79d_features.npz"
JSON_PATH = r"D:\src\models\second_models\babel_lunge\bl5_hybrid_1_benchmark_results.json"
BUNDLE_DIR = r"D:\src\models\second_models\babel_lunge\weights"
BUNDLE_PATH = os.path.join(BUNDLE_DIR, "bl5_hybrid_1_bundle.pkl")
FINAL_BUNDLE_PATH = os.path.join(BUNDLE_DIR, "bl5_final_bundle.pkl")

def build_bundle():
    print("[*] 하이브리드 피처 캐시 로드 중...")
    c = np.load(HYBRID_CACHE_PATH, allow_pickle=True)
    X = c['X'].astype(np.float32)
    y = c['y'].astype(int)
    feature_names = list(c['feature_names'])

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    print(f"[*] 전체 {len(y)}개 샘플 대상 3대 모델 적합 중 (CatBoost + LightGBM + XGBoost)...")
    # 1. CatBoost
    cb = CatBoostClassifier(
        iterations=450, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
        scale_pos_weight=1.2, verbose=0, random_seed=42, thread_count=4
    )
    cb.fit(X, y)

    # 2. LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=350, max_depth=5, num_leaves=25, learning_rate=0.03,
        scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4, verbose=-1
    )
    lgbm.fit(X, y)

    # 3. XGBoost
    xg = xgb.XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.03,
        scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4, eval_metric='logloss'
    )
    xg.fit(X, y)

    view_thresholds = {
        'view5': (0.200, 0.290),
        'view2': (0.270, 0.490),
        'view3': (0.180, 0.225),
        'view4': (0.245, 0.370),
        'view1': (0.475, 0.670),
    }

    bundle = {
        'model_name': 'bl5_hybrid_1',
        'status': '[PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)',
        'description': 'MediaPipe 3D (36D) + RTMPose 2D (37D) + Cross-Modal (6D) = 79D Feature Fusion',
        'feature_names': feature_names,
        'ensemble_weights': [0.5, 0.3, 0.2],
        'model_cat': cb,
        'model_lgb': lgbm,
        'model_xgb': xg,
        'tau_clean': 0.230,
        'tau_defect': 0.340,
        'view_thresholds': view_thresholds,
        'overall_performance': benchmark_data['overall_performance'],
        'view_audit': benchmark_data['view_audit']
    }

    os.makedirs(BUNDLE_DIR, exist_ok=True)
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 번들 저장 완료: {BUNDLE_PATH} ({os.path.getsize(BUNDLE_PATH):,} bytes)")

    with open(FINAL_BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 최종 공식 배포 번들 저장 완료: {FINAL_BUNDLE_PATH} ({os.path.getsize(FINAL_BUNDLE_PATH):,} bytes)")

if __name__ == '__main__':
    build_bundle()

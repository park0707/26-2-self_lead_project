# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Module: train_dl5_rtmp_1.py
Purpose: dl5_rtmp_1 최종 프로덕션 모델 훈련 및 번들(weights/dl5_rtmp_1_bundle.pkl) 패키징
         - bl5_final (MediaPipe 3D 36D + RTMPose 2D 37D + Cross-Modal 6D = 79D) 아키텍처 계승
         - 1,468개 다이나믹 런지 전체 시퀀스 기반 3-Way 앙상블 학습 (CatBoost + LightGBM + XGBoost)
         - 공식 5대 시점(시점 A~E) 적응형 3-State 캘리브레이션 임계값 내장
"""

import os
import sys
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")
import json
import pickle
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

DATASET_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_rtmp_dataset.npz"
BENCHMARK_JSON_PATH = r"D:\tests\dl5_rtmp_comparison_results.json"
WEIGHTS_DIR = r"D:\src\models\second_models\dynamic_lunge\weights"
BUNDLE_PATH = os.path.join(WEIGHTS_DIR, "dl5_rtmp_1_bundle.pkl")

VIEW_THRESHOLDS_10 = {
    'view5': (0.2066, 0.3236), # 시점 A (우측 후방 45°) - F1 83.48%
    'view2': (0.2847, 0.4734), # 시점 B (우측 전방 45°) - F1 81.58%
    'view3': (0.1394, 0.1752), # 시점 C (정면 0°) - F1 65.96%
    'view4': (0.3273, 0.4228), # 시점 D (좌측 전방 45°) - F1 73.87%
    'view1': (0.4274, 0.6058), # 시점 E (좌측 후방 45°) - F1 84.96%
}

VIEW_THRESHOLDS_15 = {
    'view5': (0.1676, 0.3821), # 시점 A (우측 후방 45°) - F1 84.79%
    'view2': (0.2658, 0.5111), # 시점 B (우측 전방 45°) - F1 82.41%
    'view3': (0.1215, 0.1752), # 시점 C (정면 0°) - F1 66.67%
    'view4': (0.3083, 0.4609), # 시점 D (좌측 전방 45°) - F1 75.12%
    'view1': (0.4472, 0.7445), # 시점 E (좌측 후방 45°) - F1 87.18%
}

def train_production_bundle():
    print("=" * 80)
    print("  [vidio-health-trainer] dl5_rtmp_1 프로덕션 번들 모델 훈련 및 패키징 (79D Hybrid)")
    print("=" * 80)

    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data['X_hybrid_79d'].astype(np.float32)
    y = data['y'].astype(int)
    feature_names = list(data['hybrid_feature_names'])

    with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    hyb_bench = bench_data['dl5_rtmp_hybrid']

    N = len(y)
    print(f"[*] 다이나믹 런지 전체 {N}개 샘플 대상 3-Way 프로덕션 앙상블 학습 시작...")
    print(f"[*] 피처 차원: {X.shape[1]}D (MediaPipe 36D + RTMPose 37D + Cross-Modal 6D)")

    # 1. CatBoost (iterations=450, depth=6, l2_leaf_reg=5.0)
    print("[1/3] CatBoostClassifier 학습 중...")
    cb = CatBoostClassifier(
        iterations=450, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
        scale_pos_weight=1.2, verbose=0, random_seed=42, thread_count=4
    )
    cb.fit(X, y)

    # 2. LightGBM (n_estimators=350, max_depth=5, num_leaves=25)
    print("[2/3] LGBMClassifier 학습 중...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=350, max_depth=5, num_leaves=25, learning_rate=0.03,
        scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4, verbose=-1
    )
    lgbm.fit(X, y)

    # 3. XGBoost (n_estimators=350, max_depth=4)
    print("[3/3] XGBClassifier 학습 중...")
    xg = xgb.XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.03,
        scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4, eval_metric='logloss'
    )
    xg.fit(X, y)

    bundle = {
        'model_name': 'dl5_rtmp_1',
        'exercise': '다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)',
        'condition': 'c5 (상체의 과도한 숙임/젖힘 여부)',
        'status': '[PRODUCTION CONFIRMED] 다이나믹 런지 공식 확정 모델 (2026-10-11)',
        'architecture': 'bl5_final 계승 (MediaPipe 3D 36D + RTMPose 2D 37D + Cross-Modal 6D = 79D Feature Fusion)',
        'feature_names': feature_names,
        'ensemble_weights': [0.5, 0.3, 0.2],
        'model_cat': cb,
        'model_lgb': lgbm,
        'model_xgb': xg,
        'tau_clean_default': 0.250,
        'tau_defect_default': 0.400,
        'view_thresholds_10': VIEW_THRESHOLDS_10,
        'view_thresholds_15': VIEW_THRESHOLDS_15,
        'view_thresholds': VIEW_THRESHOLDS_10, # 기본값: 보류율 <= 10.0%
        'benchmark_results': {
            '2state_f1': hyb_bench['2state']['f1'],
            'roc_auc': hyb_bench['2state']['auc'],
            '2state_accuracy': hyb_bench['2state']['accuracy'],
            '3state_adaptive_10_f1': hyb_bench['3state_adaptive_10']['f1'],
            '3state_adaptive_10_rejection': hyb_bench['3state_adaptive_10']['retake_rate'],
            '3state_adaptive_10_accuracy': hyb_bench['3state_adaptive_10']['accuracy'],
            '3state_adaptive_10_precision': hyb_bench['3state_adaptive_10']['precision'],
            '3state_adaptive_10_recall': hyb_bench['3state_adaptive_10']['recall'],
            '3state_adaptive_15_f1': hyb_bench['3state_adaptive_15']['f1'],
            '3state_adaptive_15_rejection': hyb_bench['3state_adaptive_15']['retake_rate'],
            'per_view_10': hyb_bench['3state_adaptive_10']['per_view'],
            'per_view_15': hyb_bench['3state_adaptive_15']['per_view'],
            'per_exercise': hyb_bench['per_exercise']
        }
    }

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n[+] 번들 저장 완료: {BUNDLE_PATH} ({os.path.getsize(BUNDLE_PATH):,} bytes)")
    print("[+] 5-Fold OOF 실측 3-State F1: 77.41% (보류율 9.13%) / 78.36% (보류율 13.96%)")
    print(f"[+] ROC-AUC: {hyb_bench['2state']['auc']:.4f}")

if __name__ == '__main__':
    train_production_bundle()

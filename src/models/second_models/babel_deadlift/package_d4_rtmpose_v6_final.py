# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎-골반 동시 신전 불일치 (Knee-Hip Synchrony Violation, 힙 슈트업)
Script: package_d4_rtmpose_v6_final.py
Purpose: d4 v6_rtmp 차세대 모델을 공식 프로덕션 최종 모델로 패키징 및 배포
         - 770건 전수 27D 피처 추출 및 검증
         - 5-Fold 앙상블 모델 보존 + Full-Dataset(770건) 최종 앙상블 모델 학습
         - Level 1 온도 스케일링 & Level 2 시점 적응형 임계치 메타데이터 결합
         - weights_d4_rtmpose/d4_rtmpose_v6_bundle.pkl 및 d4_rtmpose_v6_final_metadata.json 생성
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from d4_model_v6_rtmp import (
    extract_d4_v6_rtmp_features,
    FEATURE_NAMES_27_D4,
    DEFAULT_OPTIMIZATION_PARAMS,
    VIEW_ALIAS_MAP
)

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_d4_rtmpose")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

OUT_BUNDLE_PATH = os.path.join(WEIGHTS_DIR, "d4_rtmpose_v6_bundle.pkl")
OUT_META_JSON = os.path.join(CURRENT_DIR, "d4_rtmpose_v6_final_metadata.json")
FOLD_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights_d4_v6_rtmp.pkl")


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] Packaging Deadlift d4 v6_rtmp Final Production Model")
    print("=" * 100)

    # 1. 메타 데이터 및 770건 피처 로드
    cache_meta = np.load(r'D:\src\models\second_models\babel_deadlift\deadlift_5models_oof_cache.npz', allow_pickle=True)
    y_all = cache_meta['y']
    y_d4 = y_all[:, 3].astype(int)
    views = cache_meta['views']
    subjects = cache_meta['subjects']
    clip_names = cache_meta['clip_names']
    N = len(y_d4)

    print(f"[+] Total samples: {N}, Defect samples: {int(np.sum(y_d4))} ({np.mean(y_d4)*100:.2f}%)")

    # 2. 27D 피처 추출/로드
    feat_cache_p = r"C:\Users\phj20\.gemini\antigravity\brain\a4874ac0-231d-41cf-8107-cbdefc551991\scratch\features_d4_breakthrough.npy"
    if os.path.exists(feat_cache_p):
        print(f"[+] Loading pre-extracted 27D features from cache: {feat_cache_p}")
        X_27d = np.load(feat_cache_p)
    else:
        print("[*] Extracting 27D features for all 770 samples...")
        RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"
        X_list = []
        for i, cname in enumerate(clip_names):
            p = os.path.join(RTM_DIR, cname + ".npz")
            d = np.load(p, allow_pickle=True)
            kps_26 = d['kps_26_64f']
            # halpe26 to mp33 mapping
            mp33 = np.zeros((64, 33, 3), dtype=np.float32)
            mapping = {0:0, 1:2, 2:5, 3:7, 4:8, 5:11, 6:12, 7:13, 8:14, 9:15, 10:16, 11:23, 12:24, 13:25, 14:26, 15:27, 16:28, 20:31, 21:32, 24:29, 25:30}
            for h_i, m_i in mapping.items():
                mp33[:, m_i] = kps_26[:, h_i]
            v = views[i]
            feat = extract_d4_v6_rtmp_features(mp33, v)
            X_list.append(feat)
        X_27d = np.array(X_list, dtype=np.float32)

    print(f"[+] Feature matrix shape: {X_27d.shape}")

    # 3. 기존 5-Fold 모델 번들 로드
    with open(FOLD_WEIGHTS_PATH, "rb") as f:
        fold_models = pickle.load(f)
    print(f"[+] Successfully loaded 5-Fold ensemble models ({len(fold_models)} folds).")

    # 4. 전체 770건 전수 학습 (Full-Dataset Fit) 모델 구성
    print("[*] Training Full-Dataset Production Ensemble (CatBoost, LightGBM, RandomForest)...")
    pos_weight = (N - np.sum(y_d4)) / max(1, np.sum(y_d4))

    # Full LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.035,
        max_depth=4,
        num_leaves=14,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    clf_lgb.fit(X_27d, y_d4)

    # Full CatBoost
    clf_cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.035,
        depth=4,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    clf_cat.fit(X_27d, y_d4)

    # Full RandomForest
    clf_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf_rf.fit(X_27d, y_d4)

    print("[+] Full-Dataset models successfully trained.")

    # 5. 최종 프로덕션 번들(Bundle) 직렬화
    production_bundle = {
        'model_name': 'vidio-health-trainer - Barbell Deadlift d4 (Knee-Hip Synchrony)',
        'version': 'v6_rtmp',
        'status': 'FINAL_PRODUCTION_CONFIRMED',
        'release_date': '2026-10-09',
        'feature_dim': 27,
        'feature_names': FEATURE_NAMES_27_D4,
        'fold_ensemble_models': fold_models,
        'full_models': {
            'lightgbm': clf_lgb,
            'catboost': clf_cat,
            'random_forest': clf_rf
        },
        'optimization_params': DEFAULT_OPTIMIZATION_PARAMS,
        'performance_metrics': {
            'auc_roc': 90.43,
            'strict_baseline_3state_f1': 64.84,
            'calibrated_view_adaptive_3state_f1': 71.64,
            'calibrated_coverage': 93.90,
            'per_view_3state_f1': {
                'view1 (시점 E: 좌측 후방 45도)': 78.05,
                'view2 (시점 B: 우측 전방 45도)': 70.97,
                'view3 (시점 C: 정면 0도)': 65.00,
                'view4 (시점 D: 좌측 전방 45도)': 72.73,
                'view5 (시점 A: 우측 후방 45도)': 78.26
            }
        }
    }

    with open(OUT_BUNDLE_PATH, 'wb') as f:
        pickle.dump(production_bundle, f)
    print(f"[+] Saved final production model bundle to: {OUT_BUNDLE_PATH}")

    # 6. 메타데이터 JSON 생성
    meta_json = {
        "model_name": "vidio-health-trainer - Barbell Deadlift d4",
        "version": "v6_rtmp",
        "defect_code": "d4",
        "defect_description": "무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Extension Synchrony, 힙 슈트업)",
        "status": "FINAL_OFFICIAL_PRODUCTION",
        "date": "2026-10-09",
        "weights_bundle_path": OUT_BUNDLE_PATH,
        "feature_dim": 27,
        "features": FEATURE_NAMES_27_D4,
        "performance": {
            "roc_auc": 90.43,
            "baseline_fixed_window_3state_f1": 64.84,
            "calibrated_view_adaptive_3state_f1": 71.64,
            "coverage": 93.90,
            "per_view_summary": {
                "view1_E": {"f1": 78.05, "cov": 85.1, "view_name": "좌측 후방 45도"},
                "view2_B": {"f1": 70.97, "cov": 82.5, "view_name": "우측 전방 45도"},
                "view3_C": {"f1": 65.00, "cov": 82.5, "view_name": "정면 0도"},
                "view4_D": {"f1": 72.73, "cov": 82.5, "view_name": "좌측 전방 45도"},
                "view5_A": {"f1": 78.26, "cov": 87.7, "view_name": "우측 후방 45도"}
            }
        },
        "optimization": {
            "level1_temperature_scaling": DEFAULT_OPTIMIZATION_PARAMS["view_temperatures"],
            "level2_view_thresholds": DEFAULT_OPTIMIZATION_PARAMS["view_thresholds"],
            "global_thresholds": DEFAULT_OPTIMIZATION_PARAMS["global_thresholds"]
        },
        "view_taxonomy": VIEW_ALIAS_MAP
    }

    with open(OUT_META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta_json, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved final production metadata JSON to: {OUT_META_JSON}")
    print("[*] Packaging completed successfully!")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v1.0 (sd2_v1)
Script: package_sd2_v1_bundle.py
Purpose: 학습된 sd2_v1 가중치(TCN, LightGBM, CatBoost 5-Fold 모델)와 3-State 파라미터를 단일 번들 파일로 패키징
"""

import os
import sys
import glob
import pickle
import json
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_WEIGHTS_DIR = r"D:\src\models\second_models\babel_deadlift\weights_stiff_spine"
SUMMARY_JSON = r"D:\src\models\second_models\babel_deadlift\stiff_spine_model_evaluation_summary.json"
TARGET_BUNDLE_DIR = os.path.join(CURRENT_DIR, "weights")
TARGET_BUNDLE_PATH = os.path.join(TARGET_BUNDLE_DIR, "sd2_v1_bundle.pkl")


def package_bundle():
    os.makedirs(TARGET_BUNDLE_DIR, exist_ok=True)
    
    with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # 1. 5-Fold TCN 모델 state_dict 로드
    tcn_weights = []
    for fold in range(5):
        pt_path = os.path.join(SRC_WEIGHTS_DIR, f"stiff_spine_tcn_fold{fold}.pt")
        state = torch.load(pt_path, map_location='cpu')
        tcn_weights.append(state)

    # 2. 5-Fold LightGBM 로드
    lgb_models = []
    for fold in range(5):
        pkl_path = os.path.join(SRC_WEIGHTS_DIR, f"stiff_spine_lgb_fold{fold}.pkl")
        with open(pkl_path, 'rb') as f:
            lgb_models.append(pickle.load(f))

    # 3. 5-Fold CatBoost 로드
    cat_models = []
    for fold in range(5):
        pkl_path = os.path.join(SRC_WEIGHTS_DIR, f"stiff_spine_cat_fold{fold}.pkl")
        with open(pkl_path, 'rb') as f:
            cat_models.append(pickle.load(f))

    # 번들 객체 생성
    bundle = {
        'model_name': 'sd2_v1',
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'sd2 (척추의 중립 위반 / Spine Neutrality)',
        'version': 'v1.0',
        'tcn_weights': tcn_weights,
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'ensemble_weights': {'tcn': 0.25, 'lgb': 0.40, 'cat': 0.35},
        'thresholds': {
            'tau_clean': summary['three_state_metrics']['tau_clean'],
            'tau_defect': summary['three_state_metrics']['tau_def'],
            'binary_threshold': 0.50
        },
        'metrics': {
            'defect_f1': summary['binary_ensemble_metrics']['f1'],
            'roc_auc': summary['binary_ensemble_metrics']['auc'],
            'accuracy': summary['binary_ensemble_metrics']['accuracy'],
            'three_state_f1': summary['three_state_metrics']['f1'],
            'coverage': summary['three_state_metrics']['coverage'],
            'abstain_rate': summary['three_state_metrics']['abstain_rate']
        }
    }

    with open(TARGET_BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    
    # babel_stiff_deadlift 루트에도 편의용 복사본 저장
    root_bundle = os.path.join(CURRENT_DIR, "sd2_model_v1_weights.pkl")
    with open(root_bundle, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"[OK] sd2_v1 번들 패키징 완료: {TARGET_BUNDLE_PATH}")
    print(f"[OK] sd2_model_v1_weights.pkl 저장 완료: {root_bundle}")


if __name__ == '__main__':
    package_bundle()

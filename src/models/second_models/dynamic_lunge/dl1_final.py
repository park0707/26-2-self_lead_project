# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Module: dl1_final.py
Purpose: 다이나믹 런지 c1 최종 공식 프로덕션 모델 배포 인터페이스 (dl1_v4 기반)
Architecture:
  - Feature Extractor: lunge_c1_features_v4.py (30D 정예 도메인 피처)
  - Classifier: GBDT 3종(LightGBM + CatBoost + XGBoost) 최적 앙상블
  - Decision Logic: 공식 5대 시점별 적응형 3-State 선택적 분류 (보류율 <= 10.0% 제약 준수)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

import numpy as np
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"D:\src\features")
from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    extract_lunge_c1_features_v4,
    LUNGE_C1_FEATURE_NAMES_V4
)

BUNDLE_PATH = r"D:\src\models\second_models\dynamic_lunge\dl1_v4_bundle.pkl"
ALT_BUNDLE_PATH = r"D:\src\models\second_models\dynamic_lunge\weights\dl1_v4_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


class DL1Classifier:
    """다이나믹 런지 c1(앞다리 무릎 각도 90도) 최종 공식 프로덕션 분류기 (dl1_v4 기반)"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            if os.path.exists(ALT_BUNDLE_PATH):
                bundle_path = ALT_BUNDLE_PATH
            else:
                raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.model_name = self.bundle.get('model_name', 'dl1_v4')
        self.feature_names = self.bundle['feature_names']
        self.weights = self.bundle['ensemble_weights']
        self.models_lgb = self.bundle['models_lgb']
        self.models_cat = self.bundle['models_cat']
        self.models_xgb = self.bundle['models_xgb']
        self.view_thresholds = self.bundle.get('view_thresholds', {})

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        vis_64f: Optional[np.ndarray] = None,
        view_code: str = 'view3'
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c1 세부 결함 추론
        - Input: kps_64f (64, 33, 3) 3D 정규화 키포인트 배열
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 상세 피처 딕셔너리
        """
        feat_dict = extract_lunge_c1_features_v4(kps_64f, vis_64f)
        feat_vec = np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)

        # 5-Fold 모델 앙상블 평균
        p_lgb = float(np.mean([m.predict(feat_vec)[0] for m in self.models_lgb]))
        p_cat = float(np.mean([m.predict_proba(feat_vec)[0, 1] for m in self.models_cat]))
        dmat = xgb.DMatrix(feat_vec)
        p_xgb = float(np.mean([m.predict(dmat)[0] for m in self.models_xgb]))

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        # 시점별 임계치 로드 (기본값: 정면 view3)
        v_th = self.view_thresholds.get(view_code, self.view_thresholds.get('view3', {'tau_clean': 0.35, 'tau_defect': 0.60}))
        tau_clean = float(v_th['tau_clean'])
        tau_defect = float(v_th['tau_defect'])

        if prob < tau_clean:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "앞다리 무릎 각도가 이상적인 90도 범위에 부합합니다."
        elif prob >= tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "앞다리 무릎이 90도까지 충분히 굽혀지지 않았습니다 (최저점에서 더 깊게 앉으세요)."
        else:
            state = 2
            state_str = "UNCERTAIN (판단 보류 / 재촬영)"
            feedback = "판단 신뢰도 경계 영역입니다. 카메라 앵글을 확인하고 재시도하세요."

        return {
            'model_name': self.model_name,
            'exercise': '다이나믹 런지 (Dynamic Lunge)',
            'defect_code': 'c1',
            'defect_name': '앞다리 무릎 각도 90도',
            'view_code': view_code,
            'view_name': VIEW_ALIAS_MAP.get(view_code, view_code),
            'state': state,
            'state_str': state_str,
            'defect_probability': prob,
            'feedback': feedback,
            'metrics': {
                'lead_knee_angle_min': feat_dict['lead_knee_angle_min'],
                'lead_knee_90_error': feat_dict['lead_knee_90_error_abs'],
                'lead_femur_vertical_drop_ratio': feat_dict['lead_femur_vertical_drop_ratio'],
                'lead_leg_coplanarity': feat_dict['lead_leg_coplanarity'],
                'lead_knee_rom_ratio': feat_dict['lead_knee_rom_ratio'],
                'sagittal_knee_90_error': feat_dict['sagittal_knee_90_error'],
            },
            'thresholds': {
                'tau_clean': tau_clean,
                'tau_defect': tau_defect
            }
        }


_DEFAULT_CLASSIFIER: Optional[DL1Classifier] = None


def get_dl1_classifier() -> DL1Classifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = DL1Classifier()
    return _DEFAULT_CLASSIFIER


def predict_dynamic_lunge_c1(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None,
    view_code: str = 'view3'
) -> Dict[str, Any]:
    """공식 표준 1-함수 추론 인터페이스"""
    clf = get_dl1_classifier()
    return clf.predict_sequence(kps_64f, vis_64f, view_code=view_code)

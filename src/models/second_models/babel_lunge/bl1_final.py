# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Module: bl1_final.py
Purpose: 바벨 런지 c1 최종 공식 프로덕션 모델 배포 인터페이스 (bl1_v3.1 기반)
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-03)
Architecture:
  - Feature Extractor: lunge_c1_features_v3_1.py (28D 도메인 기하 피처)
    * 사용자 제안 피처: lead_femur_vertical_drop_ratio (|Δy| / L_femur)
    * 관상면 정면 지향도 게이팅: front_view_femur_drop_gated
  - Classifier: GBDT 3종(LightGBM + CatBoost + XGBoost) 최적 앙상블
  - Decision Logic: 3-State 선택적 분류 (보류율 9.99% <= 10.0% 준수)
    * tau_clean = 0.355, tau_defect = 0.565
  - Performance:
    * ROC-AUC: 90.64% (90% 돌파)
    * 3-State F1-Score: 79.12%
    * 확정 정확도 (ACC): 87.36%
    * 결함 검출 재현율 (Recall): 82.33%
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

import numpy as np
import xgboost as xgb

# 도메인 피처 추출 모듈 v3.1 임포트
sys.path.append(r"D:\src\features")
from lunge_c1_features_v3_1 import (
    extract_lunge_c1_feature_vector_v3_1,
    extract_lunge_c1_features_v3_1,
    LUNGE_C1_FEATURE_NAMES_V3_1
)

BUNDLE_PATH = r"D:\src\models\second_models\babel_lunge\bl1_v3_1_bundle.pkl"

# AGENTS.md 5대 공식 시점 명칭 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


class BL1Classifier:
    """바벨 런지 c1(앞다리 무릎 각도 90도) 최종 공식 프로덕션 분류기"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.model_name = self.bundle.get('model_name', 'bl1_v3_1')
        self.feature_names = self.bundle['feature_names']
        self.tau_clean = float(self.bundle['tau_clean'])
        self.tau_defect = float(self.bundle['tau_defect'])
        self.weights = self.bundle['ensemble_weights']
        self.model_lgb = self.bundle['model_lgb']
        self.model_cat = self.bundle['model_cat']
        self.model_xgb = self.bundle['model_xgb']

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        vis_64f: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c1 세부 결함 추론
        - Input: kps_64f (64, 33, 3) 3D 정규화 키포인트 배열
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 상세 피처 딕셔너리
        """
        feat_dict = extract_lunge_c1_features_v3_1(kps_64f, vis_64f)
        feat_vec = np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)

        p_lgb = float(self.model_lgb.predict(feat_vec)[0])
        p_cat = float(self.model_cat.predict_proba(feat_vec)[0, 1])
        p_xgb = float(self.model_xgb.predict(xgb.DMatrix(feat_vec))[0])

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        if prob < self.tau_clean:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "앞다리 무릎 각도가 이상적인 90도 범위에 부합합니다."
        elif prob >= self.tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "앞다리 무릎이 90도까지 충분히 굽혀지지 않았습니다 (최저점에서 더 깊게 앉으세요)."
        else:
            state = 2
            state_str = "UNCERTAIN (판단 보류 / 재촬영)"
            feedback = "판단 신뢰도 경계 영역입니다. 카메라 앵글을 확인하고 재시도하세요."

        return {
            'model_name': self.model_name,
            'defect_code': 'c1',
            'defect_name': '앞다리 무릎 각도 90도',
            'state': state,
            'state_str': state_str,
            'defect_probability': prob,
            'feedback': feedback,
            'metrics': {
                'lead_knee_angle_min': feat_dict['lead_knee_angle_min'],
                'lead_knee_90_error': feat_dict['lead_knee_90_error_abs'],
                'lead_femur_vertical_drop_ratio': feat_dict['lead_femur_vertical_drop_ratio'],
                'front_view_femur_drop_gated': feat_dict['front_view_femur_drop_gated'],
                'sagittal_knee_90_error': feat_dict['sagittal_knee_90_error'],
            },
            'thresholds': {
                'tau_clean': self.tau_clean,
                'tau_defect': self.tau_defect
            }
        }


# 전역 싱글톤 인스턴스 (고속 서빙용)
_DEFAULT_CLASSIFIER: Optional[BL1Classifier] = None


def get_bl1_classifier() -> BL1Classifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = BL1Classifier()
    return _DEFAULT_CLASSIFIER


def predict_lunge_c1(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """공식 표준 1-함수 추론 인터페이스"""
    clf = get_bl1_classifier()
    return clf.predict_sequence(kps_64f, vis_64f)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Defect: c5 (상체의 과도한 숙임/젖힘 여부)
Module: bl5_final.py
Purpose: 바벨 런지 c5 최종 공식 프로덕션 모델 배포 인터페이스 (bl5_hybrid_1 기반)
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)
Architecture:
  - Feature Extractor: lunge_c5_features_hybrid.py (79D 하이브리드 피처)
    * MediaPipe 36D: 글로벌 3D 배향, VSPA 시상면 피치, 3D 모멘트암, 위상 적분
    * RTMPose 37D: 초정밀 2D 키포인트 궤적, 수직 저크, 동적 압축비, 스펙트럼 에너지 비
    * Cross-Modal 6D: 3D-2D 피치각차, 원근 단축비 보정, 동적 불일치도, 어깨/골반 속도비
  - Classifier: 3-Way 최적 앙상블 (CatBoost 50% + LightGBM 30% + XGBoost 20%)
  - Decision Logic: 공식 5대 시점(시점 A~E) 적응형 3-State 선택적 분류 (보류율 9.62% <= 10.0% 준수)
  - Performance:
    * 3-State F1-Score: 75.86% (MediaPipe 전용 73.11% 대비 +2.75%p, RTMP 전용 67.86% 대비 +8.00%p)
    * ROC-AUC: 0.8555 (역대 최고치)
    * 선택적 정확도 (ACC): 80.66%
    * 결함 검출 재현율 (Recall): 82.50%
    * 선택적 정밀도 (Precision): 70.21%
"""

import os
import sys
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")
import pickle
from typing import Dict, Any, Optional
import numpy as np

from src.models.second_models.babel_lunge.bl5_hybrid_1 import (
    BabelLungeCondition5ModelHybrid,
    VIEW_ALIAS_MAP,
    VIEW_THRESHOLDS
)

BUNDLE_PATH = r"D:\src\models\second_models\babel_lunge\weights\bl5_final_bundle.pkl"


class BL5Classifier:
    """바벨 런지 c5(상체의 과도한 숙임/젖힘) 최종 공식 프로덕션 분류기"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            # Fallback to hybrid_1 bundle
            fallback = os.path.join(os.path.dirname(bundle_path), "bl5_hybrid_1_bundle.pkl")
            if os.path.exists(fallback):
                bundle_path = fallback
            else:
                raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")

        self.engine = BabelLungeCondition5ModelHybrid(bundle_path=bundle_path)
        self.model_name = self.engine.model_name
        self.feature_names = self.engine.feature_names
        self.tau_clean = self.engine.tau_clean
        self.tau_defect = self.engine.tau_defect
        self.weights = self.engine.weights

    def predict_sequence(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray,
        view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c5 세부 결함 추론
        - Input:
            * mp_kps_33_3d: (64, 33, 3) 3D 정규화 키포인트 배열
            * rtmp_kps_26_2d: (64, 26, 2) 또는 (64, 26, 3) RTMPose 2D 키포인트 배열
            * view: 공식 시점 코드 ('view1'~'view5' 또는 '시점 A'~'시점 E')
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 상세 피처 및 코칭 피드백
        """
        return self.engine.predict_sequence(
            mp_kps_33_3d=mp_kps_33_3d,
            rtmp_kps_26_2d=rtmp_kps_26_2d,
            view=view
        )

    def extract_features(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray
    ) -> np.ndarray:
        return self.engine.extract_features(mp_kps_33_3d, rtmp_kps_26_2d)


# 최종 공식 인스턴스 인터페이스 제공
def get_bl5_final_classifier() -> BL5Classifier:
    return BL5Classifier()

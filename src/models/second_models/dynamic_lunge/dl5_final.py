# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Defect: c5 (상체의 과도한 숙임/젖힘 여부, Torso Sagittal Neutrality)
Module: dl5_final.py
Purpose: 다이나믹 런지 c5 최종 공식 프로덕션 모델 배포 인터페이스 (dl5_rtmp_1 기반)
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)
Criterion: 3-State 적응형 F1 보류율 10.0% 이하 (F1: 77.41%, 실제 보류율: 9.13%, ROC-AUC: 0.8565)
Architecture:
  - Feature Extractor: lunge_c5_features_hybrid.py (79D 하이브리드 피처)
    * MediaPipe 3D 36D: 글로벌 3D 배향, VSPA 시상면 피치, 3D 모멘트암, 위상 적분
    * RTMPose 2D 37D: 초정밀 2D 키포인트 궤적, 수직 저크, 동적 압축비, 스펙트럼 에너지 비
    * Cross-Modal 6D: 3D-2D 피치각차, 원근 단축비 보정, 동적 불일치도, 어깨/골반 속도비
  - Classifier: 3-Way 최적 앙상블 (CatBoost 50% + LightGBM 30% + XGBoost 20%)
  - Decision Logic: 공식 5대 시점(시점 A~E) 적응형 3-State 선택적 분류 (보류율 9.13% <= 10.0% 엄격 준수)
  - Performance:
    * 3-State F1-Score: 77.41% (바벨런지 제로샷 69.82% 대비 +7.59%p 대폭 향상, 70% 완벽 돌파)
    * ROC-AUC: 0.8565 (역대 최고치)
    * 선택적 확정 정확도 (Accuracy): 79.84%
    * 결함 검출 재현율 (Recall): 86.65%
    * 선택적 정밀도 (Precision): 69.95%
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pickle
from typing import Dict, Any, Optional
import numpy as np

from src.models.second_models.dynamic_lunge.dl5_rtmp_1 import (
    DL5RTMP1Classifier,
    VIEW_CANONICAL_MAP,
    VIEW_DISPLAY_MAP,
    VIEW_THRESHOLDS_10
)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl5_final_bundle.pkl")
FALLBACK_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl5_rtmp_1_bundle.pkl")


class DL5Classifier:
    """다이나믹 런지 c5(상체의 과도한 숙임/젖힘) 최종 공식 프로덕션 분류기"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            if os.path.exists(FALLBACK_BUNDLE_PATH):
                bundle_path = FALLBACK_BUNDLE_PATH
            else:
                raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")

        self.engine = DL5RTMP1Classifier(bundle_path=bundle_path)
        self.model_name = self.engine.model_name
        self.feature_names = self.engine.feature_names
        self.tau_clean = self.engine.tau_clean_default
        self.tau_defect = self.engine.tau_defect_default
        self.weights = self.engine.weights
        self.view_thresholds = self.engine.view_thresholds_10

    def predict_sequence(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray,
        view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c5 세부 결함 추론 (보류율 <= 10.0% 기준)
        - Input:
            * mp_kps_33_3d: (64, 33, 3) MediaPipe 3D 정규화 키포인트 배열
            * rtmp_kps_26_2d: (64, 26, 2) 또는 (64, 26, 3) RTMPose 2D 키포인트 배열
            * view: 공식 시점 코드 ('view1'~'view5' 또는 '시점 A'~'시점 E')
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 상세 피처 및 코칭 피드백
        """
        return self.engine.predict_sequence(
            mp_kps_33_3d=mp_kps_33_3d,
            rtmp_kps_26_2d=rtmp_kps_26_2d,
            view=view,
            max_rejection=0.10
        )

    def predict_proba(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray,
        view: Optional[str] = None
    ) -> float:
        return self.engine.predict_proba(mp_kps_33_3d, rtmp_kps_26_2d, view=view)

    @classmethod
    def extract_features(
        cls,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray
    ) -> np.ndarray:
        return DL5RTMP1Classifier.extract_features(mp_kps_33_3d, rtmp_kps_26_2d)


# 최종 공식 인스턴스 인터페이스 제공
def get_dl5_final_classifier() -> DL5Classifier:
    return DL5Classifier()


def predict_dynamic_lunge_c5_final(
    mp_kps_33_3d: np.ndarray,
    rtmp_kps_26_2d: np.ndarray,
    view: Optional[str] = None
) -> Dict[str, Any]:
    clf = get_dl5_final_classifier()
    return clf.predict_sequence(mp_kps_33_3d, rtmp_kps_26_2d, view=view)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 1 - 앞다리 무릎 각도 90도 오류 (Lead Knee 90 Degree Angle Defect)
Module: cl1_final.py
Purpose: 크로스 런지 Condition 1 공식 최종 확정 챔피언 모델 (Final Champion Production Model)
         - MediaPipe 33관절 3D 포즈(55D) + RTMPose Halpe26 2D 포즈(74D) + 교차 상호작용(5D) 134D 하이브리드 융합
         - CatBoost + LightGBM + XGBoost 3-way 앙상블 (피험자 9명 StratifiedGroupKFold 완전 격리)
         - Guo et al. (ICML 2017) 온도 스케일링 ECE 캘리브레이션 (ECE 2.21%)
         - 공식 5대 시점(시점 A~시점 E)별 적응형 3-State 신뢰도 게이팅:
           * 3-State F1-Score: 93.62% (역대 최고 챔피언 확정)
           * 3-State 정확도: 95.18%, 정밀도: 92.44%
           * 불확실성 보류율(Retake Rate): 8.88% (진단 커버리지 91.12%)
           * 강제 2-State F1: 89.03%, ROC-AUC: 97.31%
           * 정면 뷰(시점 C) 3-State F1: 92.96% 달성 (RTMPose 단독 88.31% 대비 +4.65%p)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union, List

import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.cross_lunge.cl1_hybrid_1 import (
    extract_cl1_hybrid_1_features,
    compute_cross_interaction_features,
    CL1_HYBRID_1_FEATURE_NAMES,
    CL1Hybrid1Classifier,
    VIEW_NAME_TO_CODE,
    VIEW_CODE_TO_NAME
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl1_final_bundle.pkl")
CL1_FINAL_FEATURE_NAMES = CL1_HYBRID_1_FEATURE_NAMES


def extract_cl1_final_features(
    kps_26_64f: np.ndarray,
    kps_33_3d_64f: np.ndarray,
    view_code: str = 'view3',
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    크로스 런지 c1 공식 최종 134D 하이브리드 피처 벡터 추출
    """
    return extract_cl1_hybrid_1_features(
        kps_26_64f=kps_26_64f,
        kps_33_3d_64f=kps_33_3d_64f,
        view_code=view_code,
        vis_64f=vis_64f
    )


class CL1FinalClassifier(CL1Hybrid1Classifier):
    """크로스 런지 c1 공식 최종 챔피언 추론 엔진"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        super().__init__(bundle_path=bundle_path)
        self.model_name = "cl1_final"
        self.defect_name = "앞다리 무릎 각도 90도 오류 (MediaPipe 3D + RTMPose 2D 융합 최종 확정 챔피언)"


# 싱글톤 캐시
_CACHED_FINAL_CLASSIFIER: Optional[CL1FinalClassifier] = None


def predict_cross_lunge_c1(
    kps_26_or_features: np.ndarray,
    kps_33_3d_64f: Optional[np.ndarray] = None,
    camera_view: str = '시점 C',
    bundle_path: Optional[str] = None,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    크로스 런지 Condition 1 (앞다리 무릎 각도 90도 오류) 공식 최종 판정 함수
    
    Args:
        kps_26_or_features: (64, 26, 3) RTMPose 텐서 또는 134D 피처 벡터
        kps_33_3d_64f: (64, 33, 3) MediaPipe 3D 관절 텐서 (텐서 입력 시 필수)
        camera_view: 공식 5대 시점 명칭('시점 A'~'시점 E') 또는 내부 코드('view1'~'view5')
        bundle_path: 가중치 번들 경로 (기본값: cl1_final_bundle.pkl)
        vis_64f: MediaPipe 관절 신뢰도 배열 (선택)
        
    Returns:
        결함 확률, 2-State 판정(NORMAL/DEFECT), 3-State 판정(NORMAL/DEFECT/UNCERTAIN), 시점별 적용 임계값 등
    """
    global _CACHED_FINAL_CLASSIFIER
    if _CACHED_FINAL_CLASSIFIER is None or bundle_path is not None:
        _CACHED_FINAL_CLASSIFIER = CL1FinalClassifier(bundle_path=bundle_path)

    arr = np.asarray(kps_26_or_features)
    if arr.ndim == 3 and kps_33_3d_64f is not None:
        return _CACHED_FINAL_CLASSIFIER.predict(
            kps_26_64f=arr,
            kps_33_3d_64f=kps_33_3d_64f,
            camera_view=camera_view,
            vis_64f=vis_64f
        )
    elif arr.ndim in (1, 2):
        vec = arr.flatten()
        return _CACHED_FINAL_CLASSIFIER.predict_features(vec, camera_view=camera_view)
    else:
        raise ValueError("유효하지 않은 입력 인자: (64, 26, 3) 및 (64, 33, 3) 텐서 쌍 또는 134D 피처 벡터가 필요합니다.")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/barbell_lunge/barbell_lunge_features.py
Purpose: [Tier 3] 바벨 런지 5대 세부 결함(bl1~bl5) 표준 피처 추출 모듈
         - bl1: 상체 기울임/척추 중립 (MediaPipe 33)
         - bl2: 앞무릎 굴곡각/돌출 (MediaPipe 33 + RTMPose 26)
         - bl3: 뒷무릎 지면 접촉/하강 (MediaPipe 33 2D)
         - bl4: 골반 불균형/비대칭 (MediaPipe 33)
         - bl5: 보폭 및 중심 안정성 (MediaPipe 33 + RTMPose 26)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.features.lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
from src.models.second_models.babel_lunge.bl2_final import extract_hybrid_2_features
from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_14d_features_v4
from src.features.lunge_c4_features_v4 import extract_lunge_c4_features_v4
from src.models.second_models.babel_lunge.bl5_final import BL5Classifier


def _get_rtm26_or_fallback(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        return skeletons.rtm26_64f
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    return SkeletalTopologyAdapter.convert_mediapipe_to_halpe26(skeletons.mp33_64f)


def extract_barbell_lunge_bl1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_lunge_c1_feature_vector_v4(skeletons.mp33_64f)


def extract_barbell_lunge_bl2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_hybrid_2_features(skeletons.mp33_64f, rtm)


def extract_barbell_lunge_bl3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_lunge_bl3_14d_features_v4(skeletons.mp33_64f[:, :, :2])


def extract_barbell_lunge_bl4(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_lunge_c4_features_v4(skeletons.mp33_64f)


def extract_barbell_lunge_bl5(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return BL5Classifier.extract_features(skeletons.mp33_64f, rtm)

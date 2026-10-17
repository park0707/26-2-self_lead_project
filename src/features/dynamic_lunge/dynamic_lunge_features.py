# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/dynamic_lunge/dynamic_lunge_features.py
Purpose: [Tier 3] 다이나믹 런지 5대 세부 결함(dl1~dl5) 표준 피처 추출 모듈
         - dl1: 척추 중립 (MediaPipe 33)
         - dl2: 앞무릎 안정성 (MediaPipe 33 + RTMPose 26)
         - dl3: 뒷무릎 동적 착지 (RTMPose 26)
         - dl4: 골반 밸런스 (RTMPose 26)
         - dl5: 동적 착지 충격 흡수 (MediaPipe 33 + RTMPose 26)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.features.lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
from src.models.second_models.babel_lunge.bl2_final import extract_hybrid_2_features
from src.features.lunge_dl3_rtmp_features import extract_lunge_dl3_rtmp_features_all
from src.models.second_models.dynamic_lunge.dl4_rtmp_1 import DL4RTMP1Classifier
from src.models.second_models.dynamic_lunge.dl5_final import DL5Classifier


def _get_rtm26_or_fallback(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        return skeletons.rtm26_64f
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    return SkeletalTopologyAdapter.convert_mediapipe_to_halpe26(skeletons.mp33_64f)


def extract_dynamic_lunge_dl1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_lunge_c1_feature_vector_v4(skeletons.mp33_64f)


def extract_dynamic_lunge_dl2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_hybrid_2_features(skeletons.mp33_64f, rtm)


def extract_dynamic_lunge_dl3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_lunge_dl3_rtmp_features_all(rtm)


def extract_dynamic_lunge_dl4(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return DL4RTMP1Classifier.extract_features(rtm)


def extract_dynamic_lunge_dl5(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return DL5Classifier.extract_features(skeletons.mp33_64f, rtm)

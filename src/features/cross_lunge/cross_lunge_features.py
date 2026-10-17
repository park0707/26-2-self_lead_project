# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/cross_lunge/cross_lunge_features.py
Purpose: [Tier 3] 크로스 런지 3대 세부 결함(cl1~cl3) 표준 피처 추출 모듈
         - cl1: 교차 보폭 및 척추 회전 (RTMPose 26)
         - cl2: 지지다리 무릎 안쪽 무너짐 (RTMPose 26)
         - cl3: 골반 과회전/경사 (RTMPose 26)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.models.second_models.cross_lunge.cl1_rtmp_1 import extract_cl1_rtmp_1_features
from src.models.second_models.cross_lunge.cl2_rtmp_1 import extract_cl2_rtmp_1_features
from src.models.second_models.cross_lunge.cl3_rtmp_1 import extract_cl3_rtmp_1_features


def _get_rtm26_or_fallback(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        return skeletons.rtm26_64f
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    return SkeletalTopologyAdapter.convert_mediapipe_to_halpe26(skeletons.mp33_64f)


def extract_cross_lunge_cl1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_cl1_rtmp_1_features(rtm)


def extract_cross_lunge_cl2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_cl2_rtmp_1_features(rtm)


def extract_cross_lunge_cl3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return extract_cl3_rtmp_1_features(rtm)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/stiff_deadlift/stiff_deadlift_features.py
Purpose: [Tier 3] 바벨 스티프 데드리프트 3대 세부 결함(sd1~sd3) 표준 피처 추출 모듈
         - sd1: 척추 중립 위반 (RTMPose 26 / MediaPipe 33)
         - sd2: 무릎 굴곡/과신전 (RTMPose 26)
         - sd3: 골반 힌지/바벨 이탈 (RTMPose 26)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons
from src.models.second_models.babel_stiff_deadlift.sd1_rtmp_1_features import extract_sd1_rtmp_1_features
from src.models.second_models.babel_stiff_deadlift.sd2_model_v3_rtmp import extract_sd2_rtmp_features
from src.models.second_models.babel_stiff_deadlift.sd3_model_v4_rtmp import extract_sd3_rtmp_features


def _get_rtm26_or_mp_tensor(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        return skeletons.rtm26_64f
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    return SkeletalTopologyAdapter.convert_mediapipe_to_halpe26(skeletons.mp33_64f)


def extract_stiff_deadlift_sd1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """sd1: 척추 중립 위반"""
    rtm = _get_rtm26_or_mp_tensor(skeletons)
    return extract_sd1_rtmp_1_features(rtm)


def extract_stiff_deadlift_sd2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """sd2: 무릎 굴곡/과신전"""
    rtm = _get_rtm26_or_mp_tensor(skeletons)
    return extract_sd2_rtmp_features(rtm)


def extract_stiff_deadlift_sd3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """sd3: 골반 힌지/바벨 이탈"""
    rtm = _get_rtm26_or_mp_tensor(skeletons)
    return extract_sd3_rtmp_features(rtm)

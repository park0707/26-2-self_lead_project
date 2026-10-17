# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/good_morning/good_morning_features.py
Purpose: [Tier 3] 바벨 굿모닝 3대 세부 결함(gm1~gm3) 표준 피처 추출 모듈
         - gm1: 척추 중립 위반 (RTMPose 26)
         - gm2: 무릎 과도한 굴곡/스쿼트화 (RTMPose 26)
         - gm3: 힙 힌지 미흡/가동범위 부족 (RTMPose 26)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.models.second_models.babel_good_morning.gm1_rtmp_1 import BabelGoodMorningGM1RTMP1
from src.models.second_models.babel_good_morning.gm2_rtmp_1 import BabelGoodMorningGM2RTMP1
from src.models.second_models.babel_good_morning.gm3_rtmp_1 import BabelGoodMorningGM3RTMP1


def _get_rtm26_or_fallback(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        return skeletons.rtm26_64f
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    return SkeletalTopologyAdapter.convert_mediapipe_to_halpe26(skeletons.mp33_64f)


def extract_good_morning_gm1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return BabelGoodMorningGM1RTMP1.extract_features(rtm)


def extract_good_morning_gm2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return BabelGoodMorningGM2RTMP1.extract_features(rtm)


def extract_good_morning_gm3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    rtm = _get_rtm26_or_fallback(skeletons)
    return BabelGoodMorningGM3RTMP1.extract_features(rtm)

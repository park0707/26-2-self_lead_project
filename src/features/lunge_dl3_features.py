# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 스텝 포워드 & 백워드)
Module: lunge_dl3_features.py
Purpose: 스포츠 생체역학(IEEE TBME, Journal of Biomechanics) 선행 연구 기반
         바벨 런지 bl3_v4의 14D/18D 고급 기구학 피처 엔진을 100% 계승하여
         다이나믹 런지 세 번째 결함 '뒤다리 무릎 각도 90도 (dl3)' 전담 피처 추출기 구축
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, 'd:/')
from src.features.lunge_bl3_features_v4 import (
    extract_lunge_bl3_14d_features_v4 as _extract_14d,
    extract_lunge_bl3_18d_features_v4 as _extract_18d,
    FEATURE_NAMES_14D,
    FEATURE_NAMES_18D,
    FEATURE_INDICES_14D
)

def extract_lunge_dl3_14d_features(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 관절 배열로부터 dl3 14D 정예 고급 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (14,) 또는 (N, 14)
    """
    return _extract_14d(kps)

def extract_lunge_dl3_18d_features(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 관절 배열로부터 dl3 18D 고급 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (18,) 또는 (N, 18)
    """
    return _extract_18d(kps)

__all__ = [
    'extract_lunge_dl3_14d_features',
    'extract_lunge_dl3_18d_features',
    'FEATURE_NAMES_14D',
    'FEATURE_NAMES_18D',
    'FEATURE_INDICES_14D'
]

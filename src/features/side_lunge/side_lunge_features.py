# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/side_lunge/side_lunge_features.py
Purpose: [Tier 3] 사이드 런지 5대 세부 결함(sl1~sl5) 표준 피처 추출 모듈
         - sl1: 척추 전경각/중립 (MediaPipe 33)
         - sl2: 굴곡측 무릎-발목 정렬 (MediaPipe 33)
         - sl3: 신전측 다리 완전 신전 (MediaPipe 33)
         - sl4: 골반 높이 비대칭/회전 (MediaPipe 33)
         - sl5: 중심 이동 및 보폭 안정성 (MediaPipe 33)
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.features.side_lunge_c1_features_v11 import extract_side_lunge_c1_feature_vector_v11
from src.models.second_models.side_lunge.side_lunge_c2_v7_features import extract_side_lunge_c2_v7_features_51d
from src.features.side_lunge_c3_features_v3 import extract_side_lunge_c3_features_v3_28d
from src.features.side_lunge_c4_features_v4 import extract_side_lunge_c4_features_v4
from src.features.side_lunge_c5_features_v7 import extract_side_lunge_c5_v7_feature_vector


def extract_side_lunge_sl1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_side_lunge_c1_feature_vector_v11(skeletons.mp33_64f)


def extract_side_lunge_sl2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_side_lunge_c2_v7_features_51d(skeletons.mp33_64f, view_code=skeletons.view_code)


def extract_side_lunge_sl3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_side_lunge_c3_features_v3_28d(skeletons.mp33_64f)


def extract_side_lunge_sl4(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_side_lunge_c4_features_v4(skeletons.mp33_64f)


def extract_side_lunge_sl5(skeletons: PreprocessedSkeletons) -> np.ndarray:
    return extract_side_lunge_c5_v7_feature_vector(skeletons.mp33_64f)

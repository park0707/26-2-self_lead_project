# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/deadlift/deadlift_features.py
Purpose: [Tier 3] 바벨 데드리프트 5대 세부 결함(d1~d5) 표준 피처 추출 모듈
         - d1: 척추 중립 위반 (25D)
         - d2: 무릎 모임/외반 (18D)
         - d3: 발바닥 접지 및 힙 슈트업 (20D)
         - d4: 무릎-골반 동시 신전 불일치 (15D)
         - d5: 바벨 궤적 신체 밀착 실패 (18D)
         - 루트 스크립트(run_deadlift_...benchmark.py) 의존성 100% 영구 제거
"""

import numpy as np
from src.web.core.contracts import PreprocessedSkeletons

from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import extract_d1_v4_3_diet_features_25d
from src.models.second_models.babel_deadlift.d2_model_v5_final import extract_d2_v5_final_features
from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq
from src.models.second_models.babel_deadlift.d4_model_v5 import extract_d4_v5_features_from_seq, DIET_CORE_INDICES_15_D4
from src.models.second_models.babel_deadlift.d5_model_v5_1 import extract_d5_v4_gated_features_from_seq, KEPT_FEATURE_INDICES_D5


def _get_mp33_tensor(skeletons: PreprocessedSkeletons) -> np.ndarray:
    if skeletons.mp33_64f is not None and skeletons.mp33_64f.shape[1] == 33:
        return skeletons.mp33_64f
    if skeletons.rtm26_64f is not None:
        from src.web.core.topology_adapter import SkeletalTopologyAdapter
        return SkeletalTopologyAdapter.rtmpose_26_to_mediapipe_33(skeletons.rtm26_64f)
    raise ValueError("데드리프트 피처 추출에 필요한 골격 텐서가 누락되었습니다.")


def extract_deadlift_d1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """d1: 척추 중립 위반 (25D)"""
    mp = _get_mp33_tensor(skeletons)
    return extract_d1_v4_3_diet_features_25d(mp)


def extract_deadlift_d2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """d2: 무릎 모임/외반 (18D)"""
    mp = _get_mp33_tensor(skeletons)
    res = extract_d2_v5_final_features(mp)
    return res[0] if isinstance(res, tuple) else res


def extract_deadlift_d3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """d3: 발바닥 접지 및 힙 슈트업 (20D)"""
    mp = _get_mp33_tensor(skeletons)
    return extract_d3_v5_3_opt_features_from_seq(mp)


def extract_deadlift_d4(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """d4: 무릎-골반 동시 신전 불일치 (15D)"""
    mp = _get_mp33_tensor(skeletons)
    feats = extract_d4_v5_features_from_seq(mp)
    return feats[DIET_CORE_INDICES_15_D4]


def extract_deadlift_d5(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """d5: 바벨 궤적 신체 밀착 실패 (18D)"""
    mp = _get_mp33_tensor(skeletons)
    feats = extract_d5_v4_gated_features_from_seq(mp)
    return feats[KEPT_FEATURE_INDICES_D5]

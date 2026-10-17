# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/registry.py
Purpose: [Tier 3] 8대 운동 33종 세부 결함 표준 피처 레지스트리 허브 (FeatureRegistry)
         - Contract-First 아키텍처: PreprocessedSkeletons 입력 수신
         - 모든 결함에 대해 (D,) 1차원 float32 순수 피처 벡터 반환
         - 루트 스크립트 의존성 100% 영구 제거
"""

import os
import sys
import logging
from typing import Dict, Callable, Optional, Union, List
import numpy as np

PROJECT_ROOT = "D:/"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [
    PROJECT_ROOT, SRC_ROOT,
    os.path.join(SRC_ROOT, "features"),
    os.path.join(SRC_ROOT, "models", "second_models", "babel_squat"),
    os.path.join(SRC_ROOT, "models", "second_models", "babel_deadlift"),
    os.path.join(SRC_ROOT, "models", "second_models", "babel_stiff_deadlift"),
    os.path.join(SRC_ROOT, "models", "second_models", "babel_lunge"),
    os.path.join(SRC_ROOT, "models", "second_models", "dynamic_lunge"),
    os.path.join(SRC_ROOT, "models", "second_models", "side_lunge"),
    os.path.join(SRC_ROOT, "models", "second_models", "cross_lunge"),
    os.path.join(SRC_ROOT, "models", "second_models", "babel_good_morning")
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.web.core.contracts import PreprocessedSkeletons

# 01. 바벨 스쿼트
from src.features.squat.squat_features import (
    extract_squat_c1, extract_squat_c2, extract_squat_c3, extract_squat_c4
)
# 02. 바벨 데드리프트
from src.features.deadlift.deadlift_features import (
    extract_deadlift_d1, extract_deadlift_d2, extract_deadlift_d3,
    extract_deadlift_d4, extract_deadlift_d5
)
# 03. 바벨 스티프 데드리프트
from src.features.stiff_deadlift.stiff_deadlift_features import (
    extract_stiff_deadlift_sd1, extract_stiff_deadlift_sd2, extract_stiff_deadlift_sd3
)
# 04. 바벨 런지
from src.features.barbell_lunge.barbell_lunge_features import (
    extract_barbell_lunge_bl1, extract_barbell_lunge_bl2, extract_barbell_lunge_bl3,
    extract_barbell_lunge_bl4, extract_barbell_lunge_bl5
)
# 05. 다이나믹 런지
from src.features.dynamic_lunge.dynamic_lunge_features import (
    extract_dynamic_lunge_dl1, extract_dynamic_lunge_dl2, extract_dynamic_lunge_dl3,
    extract_dynamic_lunge_dl4, extract_dynamic_lunge_dl5
)
# 06. 사이드 런지
from src.features.side_lunge.side_lunge_features import (
    extract_side_lunge_sl1, extract_side_lunge_sl2, extract_side_lunge_sl3,
    extract_side_lunge_sl4, extract_side_lunge_sl5
)
# 07. 크로스 런지
from src.features.cross_lunge.cross_lunge_features import (
    extract_cross_lunge_cl1, extract_cross_lunge_cl2, extract_cross_lunge_cl3
)
# 08. 바벨 굿모닝
from src.features.good_morning.good_morning_features import (
    extract_good_morning_gm1, extract_good_morning_gm2, extract_good_morning_gm3
)

logger = logging.getLogger("feature_registry")


class FeatureRegistry:
    """8대 운동 33종 전 결함 도메인 피처 추출 레지스트리"""

    # 33종 세부 결함 코드 매핑
    REGISTRY: Dict[str, Callable[[PreprocessedSkeletons], np.ndarray]] = {
        # 01. 바벨 스쿼트
        'c1': extract_squat_c1,
        'c2': extract_squat_c2,
        'c3': extract_squat_c3,
        'c4': extract_squat_c4,

        # 02. 바벨 데드리프트
        'd1': extract_deadlift_d1,
        'd2': extract_deadlift_d2,
        'd3': extract_deadlift_d3,
        'd4': extract_deadlift_d4,
        'd5': extract_deadlift_d5,

        # 03. 바벨 스티프 데드리프트
        'sd1': extract_stiff_deadlift_sd1,
        'sd2': extract_stiff_deadlift_sd2,
        'sd3': extract_stiff_deadlift_sd3,

        # 04. 바벨 런지
        'bl1': extract_barbell_lunge_bl1,
        'bl2': extract_barbell_lunge_bl2,
        'bl3': extract_barbell_lunge_bl3,
        'bl4': extract_barbell_lunge_bl4,
        'bl5': extract_barbell_lunge_bl5,

        # 05. 다이나믹 런지
        'dl1': extract_dynamic_lunge_dl1,
        'dl2': extract_dynamic_lunge_dl2,
        'dl3': extract_dynamic_lunge_dl3,
        'dl4': extract_dynamic_lunge_dl4,
        'dl5': extract_dynamic_lunge_dl5,

        # 06. 사이드 런지
        'sl1': extract_side_lunge_sl1,
        'sl2': extract_side_lunge_sl2,
        'sl3': extract_side_lunge_sl3,
        'sl4': extract_side_lunge_sl4,
        'sl5': extract_side_lunge_sl5,

        # 07. 크로스 런지
        'cl1': extract_cross_lunge_cl1,
        'cl2': extract_cross_lunge_cl2,
        'cl3': extract_cross_lunge_cl3,

        # 08. 바벨 굿모닝
        'gm1': extract_good_morning_gm1,
        'gm2': extract_good_morning_gm2,
        'gm3': extract_good_morning_gm3,
    }

    @classmethod
    def get_features(cls, defect_code: str, skeletons: PreprocessedSkeletons) -> np.ndarray:
        """
        단일 결함 코드에 대한 온디맨드 피처 추출 실행
        - 반환: (D,) 1차원 float32 넘파이 배열
        """
        extractor = cls.REGISTRY.get(defect_code)
        if extractor is None:
            raise KeyError(f"등록되지 않은 결함 코드입니다: '{defect_code}'")

        feat = extractor(skeletons)
        feat_arr = np.asarray(feat, dtype=np.float32).ravel()
        return feat_arr

    @classmethod
    def extract_all_for_exercise(
        cls,
        defect_codes: List[str],
        skeletons: PreprocessedSkeletons
    ) -> Dict[str, np.ndarray]:
        """주어진 결함 코드 목록에 대한 일괄 피처 추출"""
        results = {}
        for code in defect_codes:
            results[code] = cls.get_features(code, skeletons)
        return results

    @classmethod
    def has_defect(cls, defect_code: str) -> bool:
        return defect_code in cls.REGISTRY

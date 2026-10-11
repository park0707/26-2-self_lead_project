# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 척추 중립(c1) 최종 25D 피처 통합 추출기 v4
- 기존 16D 베이스라인 중 유효 피처
- 척추 곡률 심화 지표 (3D)
- 선행 연구/논문 기반 신규 생체역학 지표 (6D)
전체를 결합하여 최종 F1 85.55%, ROC-AUC 92.93%를 달성한 공식 25D 피처 추출기.
"""

import numpy as np
from typing import List, Tuple
from src.models.second_models.extract_squat_spine_trend import extract_squat_spine_trend_features, SPINE_TREND_FEATURE_NAMES_9
from src.models.second_models.extract_squat_spine_literature import extract_squat_spine_literature_features, SPINE_LIT_FEATURE_NAMES_6
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics

SPINE_FINAL_V4_FEATURE_NAMES_25: List[str] = (
    SPINE_TREND_FEATURE_NAMES_9
    + ['torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min']
    + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']
    + ['fine_neck_back_waist_angle_min', 'fine_neck_back_waist_angle_drop', 'fine_head_forward_protrusion_max']
    + SPINE_LIT_FEATURE_NAMES_6
)


def extract_squat_spine_features_25d(
    X_b: np.ndarray,
    curv_feats_6d: np.ndarray,
    fine_feats_12d: np.ndarray
) -> np.ndarray:
    """
    Format B (N, 16, 92) 및 기구축 세밀/곡률 피처로부터
    척추 중립 전담 최종 25D 피처셋을 결합 생성한다.

    Returns:
        X_spine_25d: (N, 25) float32 배열
    """
    N = len(X_b)
    # 1. 9D 동적 경향성
    feats_9d = extract_squat_spine_trend_features(X_b)
    
    # 2. 최하사점 상체-정강이 평행도 (1D)
    X_bot = extract_squat_bottom_biomechanics(X_b)
    torso_shank_err = X_bot[:, 1:2]
    
    # 3. 고개/시선 공변성 피처 (3D)
    gaze_pitch_mean = fine_feats_12d[:, 0:1]
    gaze_pitch_min  = fine_feats_12d[:, 1:2]
    cervical_min    = fine_feats_12d[:, 3:4]
    
    # 4. 척추 곡률 지표 (6D: 귀-어깨-고관절 3D + 목-등-허리/머리돌출 3D)
    curv_6d = curv_feats_6d[:, :6]
    
    # 5. 선행 연구 문헌 기반 신규 생체역학 지표 (6D)
    lit_6d = extract_squat_spine_literature_features(X_b)
    
    # 결합 (9 + 1 + 3 + 6 + 6 = 25D)
    X_25d = np.concatenate([
        feats_9d,
        torso_shank_err,
        cervical_min,
        gaze_pitch_mean,
        gaze_pitch_min,
        curv_6d,
        lit_6d
    ], axis=1).astype(np.float32)
    
    return np.nan_to_num(X_25d, nan=0.0, posinf=10.0, neginf=-10.0)

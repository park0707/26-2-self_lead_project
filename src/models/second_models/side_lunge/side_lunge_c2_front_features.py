# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: side_lunge_c2_front_features.py
Purpose: 시점 B(우측 전방 45°) 및 시점 D(좌측 전방 45°) 전용 SOTA 불변 피처 추출기 (32D)
         - 45도 카메라 광축(Optical Axis) 투영 발 각도 및 대칭 정규화 도입
         - Savitzky-Golay 스무딩 필터 적용
         - c3 공간 거리 다리 분리 엔진 100% 통합
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.signal import savgol_filter

import sys
import os
PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_spatial_leg_separator import detect_side_lunge_lead_leg_spatial
from src.models.second_models.side_lunge.side_lunge_c2_features_v3 import (
    FEATURE_NAMES_30D,
    safe_savgol_filter_kps,
    build_pacs_basis,
    vector_angle_3d,
    extract_side_lunge_c2_features_29d
)

FEATURE_NAMES_32D = FEATURE_NAMES_30D + [
    "front_optical_axis_foot_yaw",          # 31. 45도 전방 카메라 광축 평면 상의 발 지향 정사영 각도
    "front_optical_axis_knee_foot_alignment"# 32. 45도 관측 평면 상의 무릎 하강축 vs 발끝축 사잇각
]


def extract_side_lunge_c2_front_features_32d(kps_seq_3d: np.ndarray, view_code: str = "view2") -> np.ndarray:
    """(64, 33, 3) MediaPipe 3D 포즈 시퀀스로부터 시점 B/D 전용 32D 피처 추출"""
    base_30d = extract_side_lunge_c2_features_29d(kps_seq_3d, view_code=view_code)
    
    if kps_seq_3d.shape[-1] == 4:
        kps_seq_3d = kps_seq_3d[:, :, :3]
    kps_clean = safe_savgol_filter_kps(kps_seq_3d, window=7, poly=2)
    
    T, V, C = kps_clean.shape
    is_left_lead, t_bot = detect_side_lunge_lead_leg_spatial(kps_clean)
    t_bot = max(5, min(T - 5, t_bot))
    
    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        lat_sign = -1.0

    # 45도 전방 카메라 광축 벡터
    # view2 (우측 전방 45°): [+1/√2, 0, 1/√2]
    # view4 (좌측 전방 45°): [-1/√2, 0, 1/√2]
    view_sign = 1.0 if view_code == 'view2' else -1.0
    cam_axis = np.array([view_sign * 0.7071, 0.0, 0.7071], dtype=np.float32)
    cam_normal = np.array([-0.7071, 0.0, view_sign * 0.7071], dtype=np.float32)
    
    # 최하사점 발 벡터
    v_foot = kps_clean[t_bot, toe_idx] - kps_clean[t_bot, heel_idx]
    v_foot_proj = np.array([np.dot(v_foot, cam_normal), v_foot[1], np.dot(v_foot, cam_axis)])
    foot_proj_norm = np.linalg.norm(v_foot_proj) + 1e-6
    u_foot_proj = v_foot_proj / foot_proj_norm
    front_optical_axis_foot_yaw = float(np.degrees(np.arctan2(u_foot_proj[0] * lat_sign, u_foot_proj[2] + 1e-6)))
    
    # 45도 관측 평면 상의 무릎 하강축 vs 발끝축 사잇각
    v_knee_disp = kps_clean[t_bot, knee_idx] - kps_clean[0, knee_idx]
    front_optical_axis_knee_foot_alignment = vector_angle_3d(v_knee_disp, v_foot)
    
    return np.concatenate([base_30d, np.array([front_optical_axis_foot_yaw, front_optical_axis_knee_foot_alignment], dtype=np.float32)])

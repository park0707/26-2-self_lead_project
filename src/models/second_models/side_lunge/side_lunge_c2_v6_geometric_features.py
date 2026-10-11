# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: side_lunge_c2_v6_geometric_features.py
Purpose: 사용자 제안 3D 상체-대퇴-발 기하 직교/평행 정렬 알고리즘 통합 SOTA 피처 추출기 (39D)
         - 1. 고관절 중점-어깨 중점 척추축 x 좌우 어깨축 외적 기반 상체 정면 법선 벡터 (v_torso)
         - 2. 고관절-앞다리 무릎 대퇴골 벡터 (v_thigh), 뒤꿈치-앞꿈치 발끝 벡터 (v_foot)
         - 3. Y축(중력) 제외 XZ 수평 바닥 평면 투영 및 L2 단위벡터 정규화 (체격 편향 0.00%)
         - 4. 3대 내적 및 완충 대역(Tolerance Band) 편차:
              - 몸통-대퇴 직교 내적 및 90도 편차각 (|v_torso · v_thigh| ≈ 0)
              - 대퇴-발 직교 내적 및 90도 편차각 (|v_thigh · v_foot| ≈ 0)
              - 몸통-발 평행 내적 및 정렬 오차각 (v_torso · v_foot ≈ 1, Δθ ≈ 0)
              - 기립(t=0) 대비 최하사점(t=t_bot) 상대 기하 변화량 (ΔDot_torso_foot, ΔDot_torso_thigh)
         - 5. 기존 sl2_v5_front의 32D 불변 피처셋 완전 계승 (총 39D)
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
from src.models.second_models.side_lunge.side_lunge_c2_front_features import (
    FEATURE_NAMES_32D,
    extract_side_lunge_c2_front_features_32d
)

USER_GEOMETRIC_FEATURE_NAMES_7D = [
    "user_geom_torso_thigh_ortho_dot",      # 33. [사용자 기하] XZ 투영 상체-대퇴 단위벡터 내적 (직교 목표: 0.0)
    "user_geom_thigh_foot_ortho_dot",       # 34. [사용자 기하] XZ 투영 대퇴-발 단위벡터 내적 (직교 목표: 0.0)
    "user_geom_torso_foot_parallel_dot",    # 35. [사용자 기하] XZ 투영 상체-발 단위벡터 내적 (평행 목표: 1.0)
    "user_geom_torso_foot_angle_deg",       # 36. [사용자 기하] 상체 정면 vs 발끝 사잇각 (degree, 0도 목표)
    "user_geom_torso_thigh_ortho_dev_deg",  # 37. [사용자 기하] 상체-대퇴 90도 직교 대비 완충 편차각 (degree)
    "user_geom_thigh_foot_ortho_dev_deg",   # 38. [사용자 기하] 대퇴-발 90도 직교 대비 완충 편차각 (degree)
    "user_geom_delta_torso_foot_alignment", # 39. [사용자 기하] 기립 대비 최하사점 상체-발 내적 상대 변화량
]

FEATURE_NAMES_39D = FEATURE_NAMES_32D + USER_GEOMETRIC_FEATURE_NAMES_7D


def compute_user_geometric_alignment(kps_frame: np.ndarray, lead_hip_idx: int, lead_knee_idx: int,
                                     lead_heel_idx: int, lead_toe_idx: int, lat_sign: float) -> Tuple[float, float, float, float, float, float]:
    """
    단일 프레임에서 사용자 제안 기하 직교/평행 정렬 지표 계산
    1) 상체 정면 벡터 (고관절-어깨 척추 x 좌우 어깨)
    2) 대퇴 벡터 (고관절 -> 무릎)
    3) 발 벡터 (뒤꿈치 -> 앞꿈치)
    4) Y축 제외 XZ 평면 투영 및 L2 단위벡터 정규화
    5) 내적 3종 및 사잇각/편차각 계산
    """
    lh = kps_frame[23, :3]
    rh = kps_frame[24, :3]
    ls = kps_frame[11, :3]
    rs = kps_frame[12, :3]

    mid_hip = (lh + rh) * 0.5
    mid_sh = (ls + rs) * 0.5

    # 좌우 어깨 횡축 벡터
    v_shoulder_lat = ls - rs
    norm_sh = np.linalg.norm(v_shoulder_lat) + 1e-6
    u_shoulder_lat = v_shoulder_lat / norm_sh

    # 척추 상향 벡터 (골반 중심 -> 어깨 중심)
    v_spine_up = mid_sh - mid_hip
    norm_up = np.linalg.norm(v_spine_up) + 1e-6
    u_spine_up = v_spine_up / norm_up

    # 1. 상체 정면 법선 벡터 (척추 x 좌우 어깨 외적)
    v_torso_fwd = np.cross(u_shoulder_lat, u_spine_up)
    norm_tf = np.linalg.norm(v_torso_fwd) + 1e-6
    v_torso_fwd /= norm_tf

    # 2. 대퇴골 벡터 (고관절 -> 무릎)
    v_thigh = kps_frame[lead_knee_idx, :3] - kps_frame[lead_hip_idx, :3]

    # 3. 발 장축 벡터 (뒤꿈치 -> 앞꿈치)
    v_foot = kps_frame[lead_toe_idx, :3] - kps_frame[lead_heel_idx, :3]

    # 4. Y축 제외 XZ 수평 평면 투영 및 L2 단위벡터 정규화
    # MediaPipe: X는 가로, Y는 수직(상하), Z는 깊이
    v_torso_xz = np.array([v_torso_fwd[0], v_torso_fwd[2]], dtype=np.float32)
    v_thigh_xz = np.array([v_thigh[0], v_thigh[2]], dtype=np.float32)
    v_foot_xz = np.array([v_foot[0], v_foot[2]], dtype=np.float32)

    norm_t_xz = np.linalg.norm(v_torso_xz) + 1e-6
    norm_th_xz = np.linalg.norm(v_thigh_xz) + 1e-6
    norm_f_xz = np.linalg.norm(v_foot_xz) + 1e-6

    u_torso_xz = v_torso_xz / norm_t_xz
    u_thigh_xz = v_thigh_xz / norm_th_xz
    u_foot_xz = v_foot_xz / norm_f_xz

    # 좌/우 방향성 부호 대칭 정규화 (left lead: lat_sign=1, right lead: lat_sign=-1)
    # 대퇴는 옆으로 벌어지므로 lateral 성분에 부호 적용
    u_thigh_xz_sym = np.array([u_thigh_xz[0] * lat_sign, u_thigh_xz[1]], dtype=np.float32)

    # 5. 내적 계산
    # 몸통-대퇴 직교 내적 (이상적: 0.0)
    dot_torso_thigh = float(np.clip(np.dot(u_torso_xz, u_thigh_xz_sym), -1.0, 1.0))

    # 대퇴-발 직교 내적 (이상적: 0.0)
    dot_thigh_foot = float(np.clip(np.dot(u_thigh_xz_sym, u_foot_xz), -1.0, 1.0))

    # 몸통-발 평행 내적 (이상적: 1.0)
    dot_torso_foot = float(np.clip(np.dot(u_torso_xz, u_foot_xz), -1.0, 1.0))

    # 사잇각 및 직교 편차각 계산
    torso_foot_angle_deg = float(np.degrees(np.arccos(dot_torso_foot)))
    
    # 90도 직교 대비 완충 편차각
    # |dot|이 0이면 90도, 완충 편차 0도
    thigh_torso_angle = float(np.degrees(np.arccos(np.clip(abs(dot_torso_thigh), 0.0, 1.0))))
    torso_thigh_ortho_dev_deg = abs(thigh_torso_angle - 90.0)

    thigh_foot_angle = float(np.degrees(np.arccos(np.clip(abs(dot_thigh_foot), 0.0, 1.0))))
    thigh_foot_ortho_dev_deg = abs(thigh_foot_angle - 90.0)

    return (dot_torso_thigh, dot_thigh_foot, dot_torso_foot,
            torso_foot_angle_deg, torso_thigh_ortho_dev_deg, thigh_foot_ortho_dev_deg)


def extract_side_lunge_c2_v6_geometric_features_39d(kps_seq_3d: np.ndarray, view_code: str = "view2") -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 포즈 시퀀스로부터 사용자 기하 직교/평행 알고리즘 결합 39D 피처 추출
    """
    # 1. sl2_v5_front 베이스 32D 피처 추출
    base_32d = extract_side_lunge_c2_front_features_32d(kps_seq_3d, view_code=view_code)

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

    # 기립(t=0) 프레임 기하 지표
    (stand_dot_t_th, stand_dot_th_f, stand_dot_t_f,
     stand_ang_t_f, stand_dev_t_th, stand_dev_th_f) = compute_user_geometric_alignment(
        kps_clean[0], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign
    )

    # 최하사점(t=t_bot) 프레임 기하 지표
    (bot_dot_t_th, bot_dot_th_f, bot_dot_t_f,
     bot_ang_t_f, bot_dev_t_th, bot_dev_th_f) = compute_user_geometric_alignment(
        kps_clean[t_bot], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign
    )

    # 기립 대비 최하사점 상대 기하 변화량
    delta_torso_foot_alignment = bot_dot_t_f - stand_dot_t_f

    # 7D 사용자 기하 피처 벡터
    user_geom_7d = np.array([
        bot_dot_t_th,               # 33. XZ 상체-대퇴 직교 내적
        bot_dot_th_f,               # 34. XZ 대퇴-발 직교 내적
        bot_dot_t_f,                # 35. XZ 상체-발 평행 내적
        bot_ang_t_f,                # 36. 상체-발 사잇각 (deg)
        bot_dev_t_th,               # 37. 상체-대퇴 90도 직교 편차각 (deg)
        bot_dev_th_f,               # 38. 대퇴-발 90도 직교 편차각 (deg)
        delta_torso_foot_alignment  # 39. 기립 대비 최하사점 상체-발 내적 상대 변화량
    ], dtype=np.float32)

    return np.concatenate([base_32d, user_geom_7d])

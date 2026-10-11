# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: side_lunge_c2_v7_features.py
Purpose: 3대 혁신 기하 알고리즘 통합 SOTA 51D 피처 추출기 (sl2_v7)
         1. [아이디어 1: 카메라 시선 광축 폐색 역투영 피처] (3D)
            - 카메라 광축 시선(Ray) 상 뒤꿈치-앞꿈치 일직선성(Collinearity) 및 가로 편위도
            - 후방 가림 현상 자체를 45도 외회전 결함 증거로 역이용하는 폐색 스코어
         2. [아이디어 2: 하강 전 구간 다단계 시계열 벡터 추출] (5D)
            - 가림이 없는 하강 30%, 60%, 100%(최하사점) 단계별 상체-대퇴-발 내적
            - 하강 전 구간 상체-발 평행 내적 중앙값(Median) 및 분산 지표
         3. [아이디어 3: 정강이 비틀림 기반 족부 역학적 복원] (4D)
            - 무릎-발목 3D 경골 비틀림(Tibial Torsion) 해부학적 족부 지향 방향 복원
            - 상체-정강이 정합도 및 관측 발끝 잔차(Residual)
         4. [sl2_v6의 39D 피처 전수 계승] (39D + 12D = 총 51D)
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
from side_lunge_c2_features_v3 import (
    FEATURE_NAMES_30D,
    safe_savgol_filter_kps,
    build_pacs_basis,
    vector_angle_3d,
    extract_side_lunge_c2_features_29d
)
from side_lunge_c2_v6_geometric_features import (
    FEATURE_NAMES_39D,
    compute_user_geometric_alignment
)

V7_NEW_FEATURE_NAMES_12D = [
    # 1. 하강 전 구간 다단계 시계열 벡터 (5D)
    "temporal_geom_t30_torso_foot_parallel",    # 40. [시계열] 하강 30% 지점(가림 0% 초기) 상체-발 평행 내적
    "temporal_geom_t30_torso_thigh_ortho",      # 41. [시계열] 하강 30% 지점 상체-대퇴 직교 내적
    "temporal_geom_t60_torso_foot_parallel",    # 42. [시계열] 하강 60% 지점(중간 하강) 상체-발 평행 내적
    "temporal_geom_descent_median_torso_foot",  # 43. [시계열] 하강 전 구간 상체-발 평행 내적 중앙값 (Median)
    "temporal_geom_descent_stability_std",      # 44. [시계열] 하강 전 구간 상체-발 내적 표준편차 (흔들림 분산)

    # 2. 카메라 시선 광축 폐색 역투영 피처 (3D)
    "ray_proj_heel_toe_collinearity",           # 45. [역투영] 카메라 시선 광축 vs 발 벡터 평행 사잇각 (일직선성)
    "ray_proj_transverse_deviation",            # 46. [역투영] 카메라 시선 수직 평면 발끝 횡방향 이탈 거리
    "ray_proj_occlusion_likelihood_score",      # 47. [역투영] 정강이-뒤꿈치 발끝 가림 발생 개연성 점수

    # 3. 정강이 비틀림 기반 족부 역학적 복원 피처 (4D)
    "kinematic_shank_torsion_angle",            # 48. [역학복원] 3차원 정강이뼈(경골) 외염 비틀림 각도
    "kinematic_reconstructed_foot_yaw",         # 49. [역학복원] 정강이 비틀림으로부터 해부학적 유추한 발끝 각도
    "kinematic_torso_shank_alignment",          # 50. [역학복원] 상체 정면 법선축 vs 정강이 진행축 정합 내적
    "kinematic_foot_vs_observed_deviation",     # 51. [역학복원] 역학적 유추 각도 vs 관측 발끝 각도 잔차
]

FEATURE_NAMES_51D = FEATURE_NAMES_39D + V7_NEW_FEATURE_NAMES_12D


def compute_v7_breakthrough_features_12d(kps_clean: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 스무딩된 3D 포즈 텐서로부터 3대 혁신 12D 피처 계산
    """
    T, V, C = kps_clean.shape
    is_left_lead, t_bot = detect_side_lunge_lead_leg_spatial(kps_clean)
    t_bot = max(5, min(T - 5, t_bot))

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        lat_sign = -1.0

    # -------------------------------------------------------------------------
    # 1. 하강 전 구간 다단계 시계열 벡터 (5D)
    # -------------------------------------------------------------------------
    t_start = 0
    t_30 = max(0, int(t_start + 0.30 * (t_bot - t_start)))
    t_60 = max(0, int(t_start + 0.60 * (t_bot - t_start)))

    # t=30% 지점 기하 정합
    res_30 = compute_user_geometric_alignment(kps_clean[t_30], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign)
    dot_t30_thigh = res_30[0]
    dot_t30_foot = res_30[2]

    # t=60% 지점 기하 정합
    res_60 = compute_user_geometric_alignment(kps_clean[t_60], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign)
    dot_t60_foot = res_60[2]

    # 하강 전 구간(t_start ~ t_bot) 상체-발 평행 내적 시계열 추적
    descent_dots = []
    for t_step in range(t_start, t_bot + 1):
        r_step = compute_user_geometric_alignment(kps_clean[t_step], hip_idx, knee_idx, heel_idx, toe_idx, lat_sign)
        descent_dots.append(r_step[2]) # dot_torso_foot

    descent_dots_arr = np.array(descent_dots, dtype=np.float32)
    descent_median_torso_foot = float(np.median(descent_dots_arr))
    descent_stability_std = float(np.std(descent_dots_arr))

    # -------------------------------------------------------------------------
    # 2. 카메라 시선 광축 폐색 역투영 피처 (3D)
    # -------------------------------------------------------------------------
    # 카메라 위치별 광축 벡터 (공식 5대 시점 정의)
    cam_axis_map = {
        'view5': np.array([0.7071, 0.0, 0.7071], dtype=np.float32),   # 시점 A: 우측 후방 45°
        'view2': np.array([0.7071, 0.0, -0.7071], dtype=np.float32),  # 시점 B: 우측 전방 45°
        'view3': np.array([0.0, 0.0, -1.0], dtype=np.float32),        # 시점 C: 정면 0°
        'view4': np.array([-0.7071, 0.0, -0.7071], dtype=np.float32), # 시점 D: 좌측 전방 45°
        'view1': np.array([-0.7071, 0.0, 0.7071], dtype=np.float32),  # 시점 E: 좌측 후방 45°
    }
    r_cam = cam_axis_map.get(view_code, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # 최하사점 뒤꿈치 -> 앞꿈치 벡터
    v_foot_bot = kps_clean[t_bot, toe_idx, :3] - kps_clean[t_bot, heel_idx, :3]
    norm_fb = np.linalg.norm(v_foot_bot) + 1e-6
    u_foot_bot = v_foot_bot / norm_fb

    # 발 벡터와 카메라 시선 광축의 사잇각 (Collinearity: 0도 또는 180도에 가까울수록 광축과 일직선)
    ray_dot = np.clip(np.dot(u_foot_bot, r_cam), -1.0, 1.0)
    ray_collinearity_deg = float(np.degrees(np.arccos(abs(ray_dot)))) # 0도에 가까울수록 가림 일직선

    # 카메라 시선 수직 평면에서의 횡방향 편위도
    r_cam_transverse = np.array([-r_cam[2], 0.0, r_cam[0]], dtype=np.float32)
    transverse_dev = float(abs(np.dot(u_foot_bot, r_cam_transverse)))

    # 후방 시점 가림 개연성 점수 (후방 뷰에서 광축과 일직선일수록 점수 증가)
    is_rear = 1.0 if view_code in ['view1', 'view5'] else 0.0
    occlusion_likelihood = float(is_rear * (1.0 - transverse_dev))

    # -------------------------------------------------------------------------
    # 3. 정강이 비틀림 기반 족부 역학적 복원 피처 (4D)
    # -------------------------------------------------------------------------
    # 무릎 횡축 벡터 (좌우 무릎을 잇는 축 또는 골반 기저 기준 대퇴과 횡축)
    p_knee = kps_clean[t_bot, knee_idx, :3]
    p_ank = kps_clean[t_bot, ank_idx, :3]
    p_hip = kps_clean[t_bot, hip_idx, :3]

    v_shank = p_ank - p_knee
    norm_shank = np.linalg.norm(v_shank) + 1e-6
    u_shank = v_shank / norm_shank

    # 대퇴 굴곡 평면 법선 (고관절-무릎 x 무릎-발목)
    v_thigh = p_knee - p_hip
    norm_thigh = np.linalg.norm(v_thigh) + 1e-6
    u_thigh = v_thigh / norm_thigh

    plane_norm = np.cross(u_thigh, u_shank)
    norm_pn = np.linalg.norm(plane_norm) + 1e-6
    u_plane_norm = plane_norm / norm_pn

    # 발목-뒤꿈치 축 벡터
    p_heel = kps_clean[t_bot, heel_idx, :3]
    v_ank_heel = p_heel - p_ank
    norm_ah = np.linalg.norm(v_ank_heel) + 1e-6
    u_ank_heel = v_ank_heel / norm_ah

    # 3차원 정강이뼈 비틀림 각도 (대퇴-무릎 평면 법선 vs 발목 축 사잇각)
    shank_torsion_dot = np.clip(np.dot(u_plane_norm, u_ank_heel), -1.0, 1.0)
    kinematic_shank_torsion_angle = float(np.degrees(np.arccos(shank_torsion_dot)))

    # 정강이 비틀림으로부터 해부학적으로 유추한 족부 요(Yaw) 각도
    # 발은 정강이의 외회전과 일정한 해부학적 결합비를 가짐
    kinematic_reconstructed_foot_yaw = float((kinematic_shank_torsion_angle - 90.0) * lat_sign)

    # 상체 전방축 vs 정강이 진행축 정합도
    R_pelvis, _ = build_pacs_basis(kps_clean[t_bot])
    torso_fwd_3d = R_pelvis[2] # 체간 전방축
    torso_shank_alignment = float(np.dot(torso_fwd_3d, u_shank))

    # 관측 발끝 각도 vs 역학적 유추 각도 간의 잔차
    obs_foot_yaw = float(np.degrees(np.arctan2(u_foot_bot[0] * lat_sign, u_foot_bot[2] + 1e-6)))
    foot_vs_observed_deviation = float(abs(obs_foot_yaw - kinematic_reconstructed_foot_yaw))

    v7_features_12d = np.array([
        dot_t30_foot,
        dot_t30_thigh,
        dot_t60_foot,
        descent_median_torso_foot,
        descent_stability_std,
        ray_collinearity_deg,
        transverse_dev,
        occlusion_likelihood,
        kinematic_shank_torsion_angle,
        kinematic_reconstructed_foot_yaw,
        torso_shank_alignment,
        foot_vs_observed_deviation
    ], dtype=np.float32)

    return v7_features_12d


def extract_side_lunge_c2_v7_features_51d(kps_seq_3d: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 텐서로부터 51D 피처 추출 (sl2_v6 39D + v7 12D)
    """
    if kps_seq_3d.shape[-1] == 4:
        kps_seq_3d = kps_seq_3d[:, :, :3]
    kps_clean = safe_savgol_filter_kps(kps_seq_3d, window=7, poly=2)

    # 1. sl2_v6 39D 피처 추출
    from side_lunge_c2_v6_geometric_features import extract_side_lunge_c2_v6_geometric_features_39d
    base_39d = extract_side_lunge_c2_v6_geometric_features_39d(kps_clean, view_code=view_code)

    # 2. v7 3대 혁신 12D 피처 계산
    v7_12d = compute_v7_breakthrough_features_12d(kps_clean, view_code=view_code)

    return np.concatenate([base_39d, v7_12d])

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cross_lunge_c2_v7_features.py
Purpose: 사이드 런지 sl2_v7 SOTA 모델을 크로스 런지의 동작 특성에 맞추어 정밀 가공 계승한 51D 피처 추출기
         1. [Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)]
         2. [Group B: 후방 뷰 가림 극복을 위한 정강이 비틀림 & 궤적 불변 피처 (8D)]
         3. [Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)]
         4. [Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)]
         5. [Group E: 크로스 런지 삼방향(몸통-앞무릎-앞발) 일치 기하 정합 피처 (7D)]
         6. [Group F: 카메라 광축 폐색 및 단축 피처 (2D)]
         7. [Group G: sl2_v7 혁신 시계열, 광축 역투영, 역학적 복원 피처 (12D)]
         총 51차원(51D) 무누수(0.00% Leakage) 순수 비전 MediaPipe 33관절 3D 텐서 기반
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial

FEATURE_NAMES_51D = [
    # Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)
    "pacs_trunk_knee_azimuth_error",        # 1. 체간 전방축 대비 앞무릎 수평 방위 편차각
    "pacs_thigh_heading_to_trunk_angle",    # 2. 대퇴골 진행 벡터와 체간 정면축 사잇각
    "pacs_knee_ankle_coronal_tilt",         # 3. 골반 관상면 상 무릎-발목 수직 대비 기울기각
    "pacs_lead_hip_rotation_proxy",         # 4. 대퇴골 3D 벡터와 체간 시상면 내외전 각도
    "pacs_dkcd_medial_deviation",           # 5. 고관절-발목 기계축 대비 무릎 중심 내측 편위 비율
    "pacs_fppa_3d_collapse",                # 6. 하지 3관절 3차원 입체 FPPA 각도 붕괴량
    "pacs_lead_foot_yaw_angle",             # 7. 체간 전방축 대비 발 장축 벡터의 횡단면 편차각
    "pacs_heel_toe_skew_ratio",             # 8. 발뒤꿈치 대비 발끝의 내외측 편차 비율

    # Group B: 후방 뷰 가림 극복 정강이 비틀림 & 궤적 불변 피처 (8D)
    "shank_torsion_plane_angle",            # 9. 대퇴-슬관절 평면 법선 vs 발목-뒤꿈치 축 비틀림 사잇각
    "knee_trajectory_to_ankle_vector_angle",# 10. 기립->최하사점 무릎 이동 변위 vs 발목 축 사잇각
    "knee_trajectory_linearity_rmse",       # 11. 하강 구간 무릎 이동 궤적 선형 이탈도 (RMSE)
    "rear_heel_ankle_knee_alignment",       # 12. 뒤꿈치-발목-무릎 정렬각 (족부 회외/회내)
    "lead_knee_mediolateral_sway_variance", # 13. 하강 구간 앞무릎 내외측 흔들림 시간 분산
    "inter_knee_to_ankle_span_ratio",       # 14. 양 무릎 간 거리 대비 양 발목 간 거리 비율
    "lead_shank_sagittal_progression_ratio",# 15. 정강이뼈 전후 진행 변위 대비 좌우 변위 비율
    "lead_foot_heel_contact_stability",     # 16. 하강~최하사점 앞발 뒤꿈치 수직 위치 변동 계수

    # Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
    "delta_foot_yaw_standing_to_bottom",    # 17. 기립 대비 최하사점 발 외회전 상대 변화량 (Δθ_foot)
    "eccentric_knee_inward_velocity_peak",  # 18. 하강기 무릎 내측 내전 최대 속도
    "delta_q_angle_standing_to_bottom",     # 19. 기립 대비 최하사점 3D Q-angle 변화량
    "bottom_phase_alignment_variance",      # 20. 최하사점 윈도우 내 무릎-발 사잇각 시간 분산
    "concentric_knee_rebound_jerk",         # 21. 상승기 무릎 궤적 저크 평활도
    "trunk_pelvis_yaw_torsion_mean",        # 22. 하강 구간 상체-골반 비틀림 각도 평균
    "pelvic_rotation_dynamic_range",        # 23. 골반 Yaw 각도의 Max - Min 동적 범위
    "fppa_pre_bottom_slope",                # 24. 최하사점 직전 5프레임 FPPA 변화 기울기
    "foot_axis_temporal_stability",         # 25. 전 구간 발끝 방위각의 시간적 분산

    # Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
    "view_is_view1",                        # 26. 시점 E (좌측 후방 45°)
    "view_is_view2",                        # 27. 시점 B (우측 전방 45°)
    "view_is_view3",                        # 28. 시점 C (정면 0°)
    "view_is_view4",                        # 29. 시점 D (좌측 전방 45°)
    "view_is_view5",                        # 30. 시점 A (우측 후방 45°)

    # Group E: 크로스 런지 삼방향(몸통-앞무릎-앞발) 일치 기하 정합 피처 (7D)
    "user_geom_torso_knee_parallel_dot",    # 31. XZ 평면 투영 상체 정면 vs 앞무릎 진행 평행 내적 (목표: 1.0)
    "user_geom_knee_foot_parallel_dot",     # 32. XZ 평면 투영 앞무릎 진행 vs 앞발 장축 평행 내적 (목표: 1.0)
    "user_geom_torso_foot_parallel_dot",    # 33. XZ 평면 투영 상체 정면 vs 앞발 장축 평행 내적 (목표: 1.0)
    "user_geom_torso_foot_angle_deg",       # 34. 상체 정면 vs 앞발 장축 사잇각 (목표: 0도)
    "user_geom_torso_knee_angle_deg",       # 35. 상체 정면 vs 앞무릎 진행 사잇각 (목표: 0도)
    "user_geom_knee_foot_angle_deg",        # 36. 앞무릎 진행 vs 앞발 장축 사잇각 (목표: 0도)
    "user_geom_delta_torso_foot_alignment", # 37. 기립 대비 최하사점 상체-앞발 평행 내적 상대 변화량

    # Group F: 카메라 광축 폐색 및 단축 피처 (2D)
    "optical_axis_foreshortening",           # 38. 카메라 광축 대비 대퇴 단축 비율
    "shank_optical_collinearity",            # 39. 정강이축과 광축 간 일직선성 사잇각

    # Group G: sl2_v7 혁신 피처 (12D)
    "temporal_geom_t30_torso_foot_parallel",    # 40. [시계열] 하강 30% 지점 상체-앞발 평행 내적
    "temporal_geom_t30_torso_knee_parallel",    # 41. [시계열] 하강 30% 지점 상체-앞무릎 평행 내적
    "temporal_geom_t60_torso_foot_parallel",    # 42. [시계열] 하강 60% 지점 상체-앞발 평행 내적
    "temporal_geom_descent_median_torso_foot",  # 43. [시계열] 하강 전 구간 상체-앞발 평행 내적 중앙값
    "temporal_geom_descent_stability_std",      # 44. [시계열] 하강 전 구간 상체-앞발 내적 표준편차
    "ray_proj_heel_toe_collinearity",           # 45. [역투영] 카메라 시선 광축 vs 발 벡터 평행 사잇각
    "ray_proj_transverse_deviation",            # 46. [역투영] 카메라 시선 수직 평면 발끝 횡방향 이탈 거리
    "ray_proj_occlusion_likelihood_score",      # 47. [역투영] 정강이-뒤꿈치 발끝 가림 발생 개연성 점수
    "kinematic_shank_torsion_angle",            # 48. [역학복원] 3차원 정강이뼈 외염 비틀림 각도
    "kinematic_reconstructed_foot_yaw",         # 49. [역학복원] 정강이 비틀림으로부터 해부학적 유추한 발끝 각도
    "kinematic_torso_shank_alignment",          # 50. [역학복원] 상체 정면 법선축 vs 정강이 진행축 정합 내적
    "kinematic_foot_vs_observed_deviation",     # 51. [역학복원] 역학적 유추 각도 vs 관측 발끝 각도 잔차
]


def safe_savgol_filter_kps(kps_seq: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """MediaPipe 관절 텐서 (T, 33, 3) 대상 Savitzky-Golay 스무딩 필터 적용"""
    T = kps_seq.shape[0]
    if T < window:
        return kps_seq.copy()
    out = np.zeros_like(kps_seq)
    for v in range(kps_seq.shape[1]):
        for c in range(kps_seq.shape[2]):
            out[:, v, c] = savgol_filter(kps_seq[:, v, c], window_length=window, polyorder=poly, mode='interp')
    return out


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 간의 사잇각 (degree, 0~180)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def build_pacs_basis(kps_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """단일 프레임에서 Gram-Schmidt 정규 골반 좌표계(PACS) 직교 기저 구축"""
    lh = kps_frame[23, :3]
    rh = kps_frame[24, :3]
    ls = kps_frame[11, :3]
    rs = kps_frame[12, :3]

    mid_hip = (lh + rh) * 0.5
    mid_sh = (ls + rs) * 0.5

    v_lat = lh - rh
    norm_lat = np.linalg.norm(v_lat)
    u_x = np.array([1.0, 0.0, 0.0], dtype=np.float32) if norm_lat < 1e-6 else v_lat / norm_lat

    v_up = mid_sh - mid_hip
    norm_up = np.linalg.norm(v_up)
    u_up_temp = np.array([0.0, -1.0, 0.0], dtype=np.float32) if norm_up < 1e-6 else v_up / norm_up

    v_fwd = np.cross(u_x, u_up_temp)
    norm_fwd = np.linalg.norm(v_fwd)
    u_z = np.array([0.0, 0.0, 1.0], dtype=np.float32) if norm_fwd < 1e-6 else v_fwd / norm_fwd

    u_y = np.cross(u_z, u_x)
    norm_y = np.linalg.norm(u_y)
    u_y = np.array([0.0, -1.0, 0.0], dtype=np.float32) if norm_y < 1e-6 else u_y / norm_y

    R_pacs = np.vstack([u_x, u_y, u_z])
    return R_pacs, mid_hip


def compute_cross_lunge_geometric_alignment(kps_frame: np.ndarray, lead_hip_idx: int, lead_knee_idx: int,
                                            lead_heel_idx: int, lead_toe_idx: int) -> Tuple[float, float, float, float, float, float]:
    """
    크로스 런지 단일 프레임 삼방향(몸통-앞무릎-앞발) 일치 기하 지표 계산
    1) 상체 정면 벡터 (고관절-어깨 척추 x 좌우 어깨)
    2) 앞무릎 진행 벡터 (앞고관절 -> 앞무릎)
    3) 앞발 장축 벡터 (뒤꿈치 -> 앞꿈치)
    4) Y축 제외 XZ 수평면 투영 후 단위벡터 정규화
    5) 3방향 상호 평행 내적 및 사잇각 3종 산출
    """
    lh = kps_frame[23, :3]
    rh = kps_frame[24, :3]
    ls = kps_frame[11, :3]
    rs = kps_frame[12, :3]

    mid_hip = (lh + rh) * 0.5
    mid_sh = (ls + rs) * 0.5

    v_shoulder_lat = ls - rs
    norm_sh = np.linalg.norm(v_shoulder_lat) + 1e-6
    u_shoulder_lat = v_shoulder_lat / norm_sh

    v_spine_up = mid_sh - mid_hip
    norm_up = np.linalg.norm(v_spine_up) + 1e-6
    u_spine_up = v_spine_up / norm_up

    # 상체 정면 법선 벡터
    v_torso_fwd = np.cross(u_shoulder_lat, u_spine_up)
    v_torso_fwd /= (np.linalg.norm(v_torso_fwd) + 1e-6)

    # 앞무릎 진행 벡터 (고관절 -> 무릎)
    v_knee = kps_frame[lead_knee_idx, :3] - kps_frame[lead_hip_idx, :3]

    # 앞발 장축 벡터 (뒤꿈치 -> 앞꿈치)
    v_foot = kps_frame[lead_toe_idx, :3] - kps_frame[lead_heel_idx, :3]

    # XZ 수평 평면 투영 (MediaPipe: Y축이 수직 아래)
    v_torso_xz = np.array([v_torso_fwd[0], v_torso_fwd[2]], dtype=np.float32)
    v_knee_xz = np.array([v_knee[0], v_knee[2]], dtype=np.float32)
    v_foot_xz = np.array([v_foot[0], v_foot[2]], dtype=np.float32)

    u_torso_xz = v_torso_xz / (np.linalg.norm(v_torso_xz) + 1e-6)
    u_knee_xz = v_knee_xz / (np.linalg.norm(v_knee_xz) + 1e-6)
    u_foot_xz = v_foot_xz / (np.linalg.norm(v_foot_xz) + 1e-6)

    # 내적 3종 (모두 평행 목표: 1.0)
    dot_torso_knee = float(np.clip(np.dot(u_torso_xz, u_knee_xz), -1.0, 1.0))
    dot_knee_foot = float(np.clip(np.dot(u_knee_xz, u_foot_xz), -1.0, 1.0))
    dot_torso_foot = float(np.clip(np.dot(u_torso_xz, u_foot_xz), -1.0, 1.0))

    # 사잇각 3종 (모두 0도 목표)
    ang_torso_foot = float(np.degrees(np.arccos(dot_torso_foot)))
    ang_torso_knee = float(np.degrees(np.arccos(dot_torso_knee)))
    ang_knee_foot = float(np.degrees(np.arccos(dot_knee_foot)))

    return dot_torso_knee, dot_knee_foot, dot_torso_foot, ang_torso_foot, ang_torso_knee, ang_knee_foot


def extract_cross_lunge_c2_v7_features_51d(kps_seq_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 관절 텐서로부터 크로스 런지 c2 51D 피처 추출
    """
    kps_clean = safe_savgol_filter_kps(kps_seq_64f[:, :, :3], window=7, poly=2)
    T, V, C = kps_clean.shape

    # 1. 앞다리(Lead Leg) 및 최하사점(t_bot) 식별
    is_left_lead, t_bot = detect_cross_lunge_lead_leg_spatial(kps_clean)
    t_bot = max(5, min(T - 5, t_bot))

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        opp_hip, opp_knee, opp_ank = 24, 26, 28
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        opp_hip, opp_knee, opp_ank = 23, 25, 27

    # 기립 상태 (t=0~3 평균)
    kps_stand = np.mean(kps_clean[:4], axis=0)
    kps_bot = kps_clean[t_bot]

    # PACS 직교계
    R_pacs_bot, mid_hip_bot = build_pacs_basis(kps_bot)
    R_pacs_stand, _ = build_pacs_basis(kps_stand)

    # -------------------------------------------------------------------------
    # Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)
    # -------------------------------------------------------------------------
    v_thigh_bot = kps_bot[knee_idx] - kps_bot[hip_idx]
    v_thigh_pacs = R_pacs_bot @ v_thigh_bot
    v_shank_bot = kps_bot[ank_idx] - kps_bot[knee_idx]
    v_shank_pacs = R_pacs_bot @ v_shank_bot
    v_foot_bot = kps_bot[toe_idx] - kps_bot[heel_idx]
    v_foot_pacs = R_pacs_bot @ v_foot_bot

    # 1. 체간 전방축 대비 앞무릎 수평 방위 편차각 (X vs Z)
    pacs_trunk_knee_azimuth_error = float(np.degrees(np.arctan2(abs(v_thigh_pacs[0]), max(abs(v_thigh_pacs[2]), 1e-4))))

    # 2. 대퇴골 진행 벡터와 체간 정면축 사잇각
    u_fwd_pacs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pacs_thigh_heading_to_trunk_angle = vector_angle_3d(v_thigh_pacs, u_fwd_pacs)

    # 3. 골반 관상면 상 무릎-발목 수직 대비 기울기각
    pacs_knee_ankle_coronal_tilt = float(np.degrees(np.arctan2(abs(v_shank_pacs[0]), max(abs(v_shank_pacs[1]), 1e-4))))

    # 4. 대퇴골 3D 벡터와 체간 시상면 내외전 각도
    pacs_lead_hip_rotation_proxy = float(np.degrees(np.arctan2(v_thigh_pacs[0], max(abs(v_thigh_pacs[1]), 1e-4))))

    # 5. DKCD: 고관절-발목 축 대비 무릎 내측 편위 비율
    v_ha = kps_bot[ank_idx] - kps_bot[hip_idx]
    norm_ha = np.linalg.norm(v_ha) + 1e-6
    u_ha = v_ha / norm_ha
    knee_proj = kps_bot[hip_idx] + np.dot(kps_bot[knee_idx] - kps_bot[hip_idx], u_ha) * u_ha
    pacs_dkcd_medial_deviation = float(np.linalg.norm(kps_bot[knee_idx] - knee_proj) / (norm_ha + 1e-6))

    # 6. FPPA 3D 붕괴량
    ang_thigh_shank = vector_angle_3d(v_thigh_bot, v_shank_bot)
    pacs_fppa_3d_collapse = float(180.0 - ang_thigh_shank)

    # 7. 발 장축 횡단면 편차각
    pacs_lead_foot_yaw_angle = float(np.degrees(np.arctan2(abs(v_foot_pacs[0]), max(abs(v_foot_pacs[2]), 1e-4))))

    # 8. 발뒤꿈치 대비 발끝 편차 비율
    heel_pacs = R_pacs_bot @ (kps_bot[heel_idx] - mid_hip_bot)
    toe_pacs = R_pacs_bot @ (kps_bot[toe_idx] - mid_hip_bot)
    pacs_heel_toe_skew_ratio = float(abs(toe_pacs[0] - heel_pacs[0]) / (abs(toe_pacs[2] - heel_pacs[2]) + 1e-4))

    # -------------------------------------------------------------------------
    # Group B: 후방 뷰 극복 정강이 비틀림 & 궤적 불변 피처 (8D)
    # -------------------------------------------------------------------------
    # 9. 대퇴-슬관절 법선 vs 발목-뒤꿈치 축 비틀림
    n_femur_knee = np.cross(v_thigh_bot, v_shank_bot)
    v_ank_heel = kps_bot[heel_idx] - kps_bot[ank_idx]
    shank_torsion_plane_angle = vector_angle_3d(n_femur_knee, v_ank_heel)

    # 10. 기립->최하사점 무릎 이동 변위 vs 발목 축 사잇각
    v_knee_disp = kps_bot[knee_idx] - kps_stand[knee_idx]
    knee_trajectory_to_ankle_vector_angle = vector_angle_3d(v_knee_disp, v_shank_bot)

    # 11. 무릎 선형 이탈도 (하강 구간 RMSE)
    knee_pts = kps_clean[:t_bot + 1, knee_idx, :]
    if len(knee_pts) > 2:
        t_arr = np.linspace(0, 1, len(knee_pts))[:, None]
        linear_interp = knee_pts[0:1] + t_arr * (knee_pts[-1:] - knee_pts[0:1])
        knee_trajectory_linearity_rmse = float(np.sqrt(np.mean((knee_pts - linear_interp) ** 2)))
    else:
        knee_trajectory_linearity_rmse = 0.0

    # 12. 뒤꿈치-발목-무릎 정렬각
    rear_heel_ankle_knee_alignment = vector_angle_3d(kps_bot[ank_idx] - kps_bot[heel_idx], kps_bot[knee_idx] - kps_bot[ank_idx])

    # 13. 앞무릎 내외측 흔들림 분산
    pacs_xs = [float((R_pacs_bot @ (kps_clean[t, knee_idx] - mid_hip_bot))[0]) for t in range(t_bot + 1)]
    lead_knee_mediolateral_sway_variance = float(np.var(pacs_xs))

    # 14. 양 무릎 간 거리 대비 양 발목 간 거리 비율
    dist_knees = float(np.linalg.norm(kps_bot[knee_idx] - kps_bot[opp_knee]))
    dist_ankles = float(np.linalg.norm(kps_bot[ank_idx] - kps_bot[opp_ank]))
    inter_knee_to_ankle_span_ratio = float(dist_knees / (dist_ankles + 1e-4))

    # 15. 정강이뼈 전후 진행 변위 대비 좌우 변위
    lead_shank_sagittal_progression_ratio = float(abs(v_shank_pacs[2]) / (abs(v_shank_pacs[0]) + 1e-4))

    # 16. 뒤꿈치 수직 위치 변동 계수
    heel_ys = kps_clean[:t_bot + 1, heel_idx, 1]
    lead_foot_heel_contact_stability = float(np.std(heel_ys) / (np.mean(np.abs(heel_ys)) + 1e-4))

    # -------------------------------------------------------------------------
    # Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
    # -------------------------------------------------------------------------
    v_foot_stand = kps_stand[toe_idx] - kps_stand[heel_idx]
    v_foot_stand_pacs = R_pacs_stand @ v_foot_stand
    yaw_stand = float(np.degrees(np.arctan2(abs(v_foot_stand_pacs[0]), max(abs(v_foot_stand_pacs[2]), 1e-4))))
    yaw_bot = pacs_lead_foot_yaw_angle
    delta_foot_yaw_standing_to_bottom = float(yaw_bot - yaw_stand)

    # 18. 하강기 무릎 내측 내전 최대 속도
    diff_pacs_x = np.diff(pacs_xs)
    eccentric_knee_inward_velocity_peak = float(np.max(np.abs(diff_pacs_x))) if len(diff_pacs_x) > 0 else 0.0

    # 19. Q-angle 변화량
    ang_q_stand = vector_angle_3d(kps_stand[hip_idx] - kps_stand[knee_idx], kps_stand[ank_idx] - kps_stand[knee_idx])
    ang_q_bot = vector_angle_3d(kps_bot[hip_idx] - kps_bot[knee_idx], kps_bot[ank_idx] - kps_bot[knee_idx])
    delta_q_angle_standing_to_bottom = float(ang_q_bot - ang_q_stand)

    # 20. 최하사점 무릎-발 사잇각 시간 분산
    bot_w_start = max(0, t_bot - 2)
    bot_w_end = min(T, t_bot + 3)
    w_angles = []
    for t_idx in range(bot_w_start, bot_w_end):
        vk = kps_clean[t_idx, knee_idx] - kps_clean[t_idx, hip_idx]
        vf = kps_clean[t_idx, toe_idx] - kps_clean[t_idx, heel_idx]
        w_angles.append(vector_angle_3d(vk, vf))
    bottom_phase_alignment_variance = float(np.var(w_angles))

    # 21. 상승기 저크 평활도
    if t_bot < T - 4:
        concentric_k = kps_clean[t_bot:, knee_idx, 1]
        jerk = np.diff(concentric_k, n=3)
        concentric_knee_rebound_jerk = float(np.mean(jerk ** 2)) if len(jerk) > 0 else 0.0
    else:
        concentric_knee_rebound_jerk = 0.0

    # 22. 하강 구간 상체-골반 비틀림 각도 평균
    yaw_torsions = []
    for t_idx in range(t_bot + 1):
        v_sh = kps_clean[t_idx, 11] - kps_clean[t_idx, 12]
        v_hp = kps_clean[t_idx, 23] - kps_clean[t_idx, 24]
        yaw_torsions.append(vector_angle_3d(v_sh, v_hp))
    trunk_pelvis_yaw_torsion_mean = float(np.mean(yaw_torsions))

    # 23. 골반 Yaw 동적 범위
    pelvic_rotation_dynamic_range = float(np.ptp(yaw_torsions))

    # 24. FPPA 직전 기울기
    pre_bot_start = max(0, t_bot - 5)
    fppa_series = []
    for t_idx in range(pre_bot_start, t_bot + 1):
        vt = kps_clean[t_idx, knee_idx] - kps_clean[t_idx, hip_idx]
        vs = kps_clean[t_idx, ank_idx] - kps_clean[t_idx, knee_idx]
        fppa_series.append(180.0 - vector_angle_3d(vt, vs))
    fppa_pre_bottom_slope = float((fppa_series[-1] - fppa_series[0]) / max(1, len(fppa_series) - 1))

    # 25. 전 구간 발끝 방위각 시간적 분산
    all_foot_yaws = []
    for t_idx in range(T):
        vf = kps_clean[t_idx, toe_idx] - kps_clean[t_idx, heel_idx]
        all_foot_yaws.append(float(np.degrees(np.arctan2(abs(vf[0]), max(abs(vf[2]), 1e-4)))))
    foot_axis_temporal_stability = float(np.var(all_foot_yaws))

    # -------------------------------------------------------------------------
    # Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
    # -------------------------------------------------------------------------
    view_is_view1 = 1.0 if view_code == 'view1' else 0.0 # 시점 E (좌측 후방 45°)
    view_is_view2 = 1.0 if view_code == 'view2' else 0.0 # 시점 B (우측 전방 45°)
    view_is_view3 = 1.0 if view_code == 'view3' else 0.0 # 시점 C (정면 0°)
    view_is_view4 = 1.0 if view_code == 'view4' else 0.0 # 시점 D (좌측 전방 45°)
    view_is_view5 = 1.0 if view_code == 'view5' else 0.0 # 시점 A (우측 후방 45°)

    # -------------------------------------------------------------------------
    # Group E: 크로스 런지 삼방향 일치 기하 지표 (7D)
    # -------------------------------------------------------------------------
    res_geom_bot = compute_cross_lunge_geometric_alignment(kps_bot, hip_idx, knee_idx, heel_idx, toe_idx)
    res_geom_stand = compute_cross_lunge_geometric_alignment(kps_stand, hip_idx, knee_idx, heel_idx, toe_idx)

    user_geom_torso_knee_parallel_dot = res_geom_bot[0]
    user_geom_knee_foot_parallel_dot = res_geom_bot[1]
    user_geom_torso_foot_parallel_dot = res_geom_bot[2]
    user_geom_torso_foot_angle_deg = res_geom_bot[3]
    user_geom_torso_knee_angle_deg = res_geom_bot[4]
    user_geom_knee_foot_angle_deg = res_geom_bot[5]
    user_geom_delta_torso_foot_alignment = float(res_geom_bot[2] - res_geom_stand[2])

    # -------------------------------------------------------------------------
    # Group F: 카메라 광축 폐색 및 단축 피처 (2D)
    # -------------------------------------------------------------------------
    optical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norm_thigh_3d = np.linalg.norm(v_thigh_bot) + 1e-6
    optical_axis_foreshortening = float(abs(v_thigh_bot[2]) / norm_thigh_3d)
    shank_optical_collinearity = vector_angle_3d(v_shank_bot, optical_axis)

    # -------------------------------------------------------------------------
    # Group G: sl2_v7 혁신 피처 (12D)
    # -------------------------------------------------------------------------
    t_30 = max(0, int(0.30 * t_bot))
    t_60 = max(0, int(0.60 * t_bot))

    res_30 = compute_cross_lunge_geometric_alignment(kps_clean[t_30], hip_idx, knee_idx, heel_idx, toe_idx)
    temporal_geom_t30_torso_foot_parallel = res_30[2]
    temporal_geom_t30_torso_knee_parallel = res_30[0]

    res_60 = compute_cross_lunge_geometric_alignment(kps_clean[t_60], hip_idx, knee_idx, heel_idx, toe_idx)
    temporal_geom_t60_torso_foot_parallel = res_60[2]

    descent_dots = []
    for t_step in range(t_bot + 1):
        g = compute_cross_lunge_geometric_alignment(kps_clean[t_step], hip_idx, knee_idx, heel_idx, toe_idx)
        descent_dots.append(g[2])
    temporal_geom_descent_median_torso_foot = float(np.median(descent_dots))
    temporal_geom_descent_stability_std = float(np.std(descent_dots))

    # 광축 폐색 역투영 (3D)
    ray_proj_heel_toe_collinearity = vector_angle_3d(v_foot_bot, optical_axis)
    ray_proj_transverse_deviation = float(abs(v_foot_bot[0]))
    is_rear_view = 1.0 if view_code in ['view1', 'view5'] else 0.0
    ray_proj_occlusion_likelihood_score = float(is_rear_view * (1.0 - abs(v_foot_bot[2]) / (np.linalg.norm(v_foot_bot) + 1e-6)))

    # 정강이 비틀림 기반 역학적 복원 (4D)
    kinematic_shank_torsion_angle = float(pacs_knee_ankle_coronal_tilt)
    kinematic_reconstructed_foot_yaw = float(pacs_lead_foot_yaw_angle * 0.7 + pacs_knee_ankle_coronal_tilt * 0.3)
    kinematic_torso_shank_alignment = float(np.dot(u_fwd_pacs, v_shank_pacs / (np.linalg.norm(v_shank_pacs) + 1e-6)))
    kinematic_foot_vs_observed_deviation = float(abs(kinematic_reconstructed_foot_yaw - pacs_lead_foot_yaw_angle))

    feats = [
        # Group A (8D)
        pacs_trunk_knee_azimuth_error,
        pacs_thigh_heading_to_trunk_angle,
        pacs_knee_ankle_coronal_tilt,
        pacs_lead_hip_rotation_proxy,
        pacs_dkcd_medial_deviation,
        pacs_fppa_3d_collapse,
        pacs_lead_foot_yaw_angle,
        pacs_heel_toe_skew_ratio,

        # Group B (8D)
        shank_torsion_plane_angle,
        knee_trajectory_to_ankle_vector_angle,
        knee_trajectory_linearity_rmse,
        rear_heel_ankle_knee_alignment,
        lead_knee_mediolateral_sway_variance,
        inter_knee_to_ankle_span_ratio,
        lead_shank_sagittal_progression_ratio,
        lead_foot_heel_contact_stability,

        # Group C (9D)
        delta_foot_yaw_standing_to_bottom,
        eccentric_knee_inward_velocity_peak,
        delta_q_angle_standing_to_bottom,
        bottom_phase_alignment_variance,
        concentric_knee_rebound_jerk,
        trunk_pelvis_yaw_torsion_mean,
        pelvic_rotation_dynamic_range,
        fppa_pre_bottom_slope,
        foot_axis_temporal_stability,

        # Group D (5D)
        view_is_view1,
        view_is_view2,
        view_is_view3,
        view_is_view4,
        view_is_view5,

        # Group E (7D)
        user_geom_torso_knee_parallel_dot,
        user_geom_knee_foot_parallel_dot,
        user_geom_torso_foot_parallel_dot,
        user_geom_torso_foot_angle_deg,
        user_geom_torso_knee_angle_deg,
        user_geom_knee_foot_angle_deg,
        user_geom_delta_torso_foot_alignment,

        # Group F (2D)
        optical_axis_foreshortening,
        shank_optical_collinearity,

        # Group G (12D)
        temporal_geom_t30_torso_foot_parallel,
        temporal_geom_t30_torso_knee_parallel,
        temporal_geom_t60_torso_foot_parallel,
        temporal_geom_descent_median_torso_foot,
        temporal_geom_descent_stability_std,
        ray_proj_heel_toe_collinearity,
        ray_proj_transverse_deviation,
        ray_proj_occlusion_likelihood_score,
        kinematic_shank_torsion_angle,
        kinematic_reconstructed_foot_yaw,
        kinematic_torso_shank_alignment,
        kinematic_foot_vs_observed_deviation,
    ]

    assert len(feats) == 51, f"피처 차원이 51이 아닙니다: {len(feats)}"
    return np.array(feats, dtype=np.float32)

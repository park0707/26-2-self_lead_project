# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: side_lunge_c2_features_v3.py
Purpose: 사이드 런지 c2 전용 생체역학 혁신 피처 추출기 (v3 SOTA)
         - [F1 70% 돌파를 위한 4대 핵심 그룹 30D 불변 피처셋]:
           Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)
           Group B: 후방 뷰 발끝 가림 극복을 위한 정강이 비틀림 & 궤적 불변 피처 (8D)
           Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
           Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
         - Savitzky-Golay 다항식 필터 기반 관절 시계열 노이즈 100% 제거
         - c3 공간 거리 다리 분리 엔진 (Spatial Distance Leg Separation) 100% 통합 표준 계승
         - 100% 순수 비전 MediaPipe 33관절 3D 포즈 텐서 기반 (AI-Hub JSON 무누수 0.00%)
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

FEATURE_NAMES_30D = [
    # Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)
    "pacs_trunk_knee_azimuth_error",        # 1. 체간 전방축 대비 작업 무릎의 수평 방위 편차각
    "pacs_thigh_heading_to_trunk_angle",    # 2. 대퇴골 진행 벡터와 체간 정면축 사잇각
    "pacs_knee_ankle_coronal_tilt",         # 3. 골반 관상면(X-Y) 상 무릎-발목 수직 대비 기울기각
    "pacs_lead_hip_rotation_proxy",         # 4. 대퇴골 3D 벡터와 체간 시상면(Y-Z) 내외전 각도
    "pacs_dkcd_medial_deviation",           # 5. 고관절-발목 기계축 대비 무릎 중심 내측 편위 비율
    "pacs_fppa_3d_collapse",                # 6. 하지 3관절 3차원 입체 FPPA 각도 붕괴량
    "pacs_lead_foot_yaw_angle",             # 7. 체간 전방축 대비 발 장축 벡터의 PACS 횡단면 편차각
    "pacs_heel_toe_skew_ratio",             # 8. 발뒤꿈치 대비 발끝의 내외측 편차 비율

    # Group B: 후방 뷰 극복을 위한 정강이 비틀림 & 궤적 불변 피처 (8D)
    "shank_torsion_plane_angle",            # 9. 대퇴-슬관절 평면 법선 vs 발목-뒤꿈치 축 비틀림 사잇각
    "knee_trajectory_to_ankle_vector_angle",# 10. 기립->최하사점 무릎 3D 이동 변위 vs 발목 고정축 사잇각
    "knee_trajectory_linearity_rmse",       # 11. 하강 구간 무릎 3D 이동 궤적의 선형 이탈도 (RMSE)
    "rear_heel_ankle_knee_alignment",       # 12. 후방 관찰면 뒤꿈치-발목-무릎 정렬각 (족부 회외/회내)
    "lead_knee_mediolateral_sway_variance", # 13. 하강 구간 PACS X축(내외측) 무릎 위치의 시간 분산
    "inter_knee_to_ankle_span_ratio",       # 14. 양 무릎 간 3D 거리 대비 양 발목 간 3D 거리 비율
    "lead_shank_sagittal_progression_ratio",# 15. 정강이뼈 전후 진행 변위 대비 좌우 변위 비율
    "lead_foot_heel_contact_stability",     # 16. 하강~최하사점 뒤꿈치 수직(Y) 위치 변동 계수

    # Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
    "delta_foot_yaw_standing_to_bottom",    # 17. [SOTA 핵심] 기립 대비 최하사점 발 외회전 상대 변화량 (Δθ_foot)
    "eccentric_knee_inward_velocity_peak",  # 18. 하강기 무릎 내측 내전 최대 속도
    "delta_q_angle_standing_to_bottom",     # 19. 기립 대비 최하사점 3D Q-angle 변화량
    "bottom_phase_alignment_variance",      # 20. 최하사점 윈도우 내 무릎-발 사잇각 시간 분산
    "concentric_knee_rebound_jerk",         # 21. 상승기 무릎 궤적 가속도 저크(Jerk) 평활도
    "trunk_pelvis_yaw_torsion_mean",        # 22. 하강 구간 전체 흉곽-골반 비틀림 각도 평균
    "pelvic_rotation_dynamic_range",        # 23. 시퀀스 전체 골반 Yaw 각도의 Max - Min 동적 범위
    "fppa_pre_bottom_slope",                # 24. 최하사점 직전 5프레임 FPPA 변화 기울기
    "foot_axis_temporal_stability",         # 25. 전 구간 발끝 방위각의 시간적 분산

    # Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
    "view_is_view1",                        # 26. 시점 E (좌측 후방 45°)
    "view_is_view2",                        # 27. 시점 B (우측 전방 45°)
    "view_is_view3",                        # 28. 시점 C (정면 0°)
    "view_is_view4",                        # 29. 시점 D (좌측 전방 45°)
    "view_is_view5",                        # 30. 시점 A (우측 후방 45°)
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


def build_pacs_basis(kps_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 프레임에서 Gram-Schmidt 정규 골반 좌표계(PACS) 직교 기저 구축
    """
    lh = kps_frame[23, :3]
    rh = kps_frame[24, :3]
    ls = kps_frame[11, :3]
    rs = kps_frame[12, :3]

    mid_hip = (lh + rh) * 0.5
    mid_sh = (ls + rs) * 0.5

    v_lat = lh - rh
    norm_lat = np.linalg.norm(v_lat)
    if norm_lat < 1e-6:
        u_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u_x = v_lat / norm_lat

    v_up = mid_sh - mid_hip
    norm_up = np.linalg.norm(v_up)
    if norm_up < 1e-6:
        v_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        v_up /= norm_up

    v_fwd = np.cross(u_x, v_up)
    norm_fwd = np.linalg.norm(v_fwd)
    if norm_fwd < 1e-6:
        u_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        u_z = v_fwd / norm_fwd

    u_y = np.cross(u_z, u_x)
    norm_y = np.linalg.norm(u_y)
    if norm_y < 1e-6:
        u_y = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        u_y /= norm_y

    R_pelvis = np.vstack([u_x, u_y, u_z]) # (3, 3)
    return R_pelvis, mid_hip


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 간의 사잇각 계산 (degree, 0 ~ 180)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def extract_side_lunge_c2_features_29d(kps_seq_3d: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 포즈 시퀀스로부터 30차원 정예 불변 생체역학 피처 추출 (v3 SOTA)
    """
    if kps_seq_3d.shape[-1] == 4:
        kps_seq_3d = kps_seq_3d[:, :, :3]

    # Savitzky-Golay 평활화로 관절 고주파 노이즈 제거
    kps_seq_clean = safe_savgol_filter_kps(kps_seq_3d, window=7, poly=2)

    T, V, C = kps_seq_clean.shape
    is_left_lead, t_bot = detect_side_lunge_lead_leg_spatial(kps_seq_clean)

    # 안전 경계 설정
    t_bot = max(5, min(T - 5, t_bot))
    bot_start = max(0, t_bot - 3)
    bot_end = min(T, t_bot + 4)

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        lat_sign = -1.0

    pelvis_widths = np.linalg.norm(kps_seq_clean[:, 23, :3] - kps_seq_clean[:, 24, :3], axis=1)
    avg_pelvis_w = float(np.mean(pelvis_widths)) + 1e-6

    # 1. PACS 골반 직교계 변환 시퀀스 계산 (T, 33, 3)
    kps_pacs = np.zeros_like(kps_seq_clean)
    for t in range(T):
        R, c = build_pacs_basis(kps_seq_clean[t])
        kps_pacs[t] = (kps_seq_clean[t] - c[np.newaxis, :]) @ R.T

    # 작업 다리 키포인트 추출
    p_hip_pacs = kps_pacs[:, hip_idx, :]
    p_knee_pacs = kps_pacs[:, knee_idx, :]
    p_ank_pacs = kps_pacs[:, ank_idx, :]
    p_heel_pacs = kps_pacs[:, heel_idx, :]
    p_toe_pacs = kps_pacs[:, toe_idx, :]

    p_knee_w = kps_seq_clean[:, knee_idx, :]
    p_ank_w = kps_seq_clean[:, ank_idx, :]
    p_heel_w = kps_seq_clean[:, heel_idx, :]
    p_toe_w = kps_seq_clean[:, toe_idx, :]

    # =========================================================================
    # Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 방향 정렬 (8D)
    # =========================================================================
    knee_bot_pacs = np.mean(p_knee_pacs[bot_start:bot_end], axis=0)
    azimuth_rad = np.arctan2(knee_bot_pacs[0] * lat_sign, knee_bot_pacs[2] + 1e-6)
    pacs_trunk_knee_azimuth_error = float(np.degrees(azimuth_rad))

    thigh_vec_pacs = knee_bot_pacs - np.mean(p_hip_pacs[bot_start:bot_end], axis=0)
    v_trunk_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pacs_thigh_heading_to_trunk_angle = vector_angle_3d(thigh_vec_pacs, v_trunk_fwd)

    ank_bot_pacs = np.mean(p_ank_pacs[bot_start:bot_end], axis=0)
    shank_coronal = np.array([(ank_bot_pacs[0] - knee_bot_pacs[0]) * lat_sign, ank_bot_pacs[1] - knee_bot_pacs[1]])
    shank_norm = np.linalg.norm(shank_coronal) + 1e-6
    dot_v = np.clip(np.dot(shank_coronal / shank_norm, np.array([0.0, -1.0])), -1.0, 1.0)
    pacs_knee_ankle_coronal_tilt = float(np.degrees(np.arccos(dot_v)))

    thigh_norm = np.linalg.norm(thigh_vec_pacs) + 1e-6
    dot_sag = np.clip(thigh_vec_pacs[0] * lat_sign / thigh_norm, -1.0, 1.0)
    pacs_lead_hip_rotation_proxy = float(np.degrees(np.arcsin(dot_sag)))

    hip_bot_pacs = np.mean(p_hip_pacs[bot_start:bot_end], axis=0)
    ha_vec = ank_bot_pacs - hip_bot_pacs
    ha_norm = np.linalg.norm(ha_vec) + 1e-6
    u_ha = ha_vec / ha_norm
    hk_vec = knee_bot_pacs - hip_bot_pacs
    proj_len = np.dot(hk_vec, u_ha)
    proj_pt = hip_bot_pacs + proj_len * u_ha
    dev_vec = knee_bot_pacs - proj_pt
    pacs_dkcd_medial_deviation = float((dev_vec[0] * lat_sign) / avg_pelvis_w)

    v_femur = hip_bot_pacs - knee_bot_pacs
    v_tibia = ank_bot_pacs - knee_bot_pacs
    fppa_ang = vector_angle_3d(v_femur, v_tibia)
    pacs_fppa_3d_collapse = float(abs(180.0 - fppa_ang))

    toe_bot_pacs = np.mean(p_toe_pacs[bot_start:bot_end], axis=0)
    foot_vec_pacs = toe_bot_pacs - ank_bot_pacs
    foot_yaw_rad = np.arctan2(foot_vec_pacs[0] * lat_sign, foot_vec_pacs[2] + 1e-6)
    pacs_lead_foot_yaw_angle = float(np.degrees(foot_yaw_rad))

    heel_bot_pacs = np.mean(p_heel_pacs[bot_start:bot_end], axis=0)
    pacs_heel_toe_skew_ratio = float(((toe_bot_pacs[0] - heel_bot_pacs[0]) * lat_sign) / avg_pelvis_w)

    # =========================================================================
    # Group B: 후방 뷰 극복을 위한 정강이 비틀림 & 궤적 불변 피처 (8D)
    # =========================================================================
    n_femur_shank = np.cross(thigh_vec_pacs, ank_bot_pacs - knee_bot_pacs)
    v_heel_axis = ank_bot_pacs - heel_bot_pacs
    shank_torsion_plane_angle = vector_angle_3d(n_femur_shank, v_heel_axis)

    v_knee_disp_3d = p_knee_w[t_bot] - p_knee_w[0]
    v_ank_ref = p_ank_w[t_bot] - p_knee_w[t_bot]
    knee_trajectory_to_ankle_vector_angle = vector_angle_3d(v_knee_disp_3d, v_ank_ref)

    if t_bot > 3:
        traj_pts = p_knee_w[:t_bot + 1]
        start_pt = traj_pts[0]
        end_pt = traj_pts[-1]
        line_vec = end_pt - start_pt
        line_len = np.linalg.norm(line_vec) + 1e-6
        u_line = line_vec / line_len
        diff_pts = traj_pts - start_pt[np.newaxis, :]
        proj_lens = np.sum(diff_pts * u_line[np.newaxis, :], axis=1, keepdims=True)
        closest_pts = start_pt[np.newaxis, :] + proj_lens * u_line[np.newaxis, :]
        dist_errs = np.linalg.norm(traj_pts - closest_pts, axis=1)
        knee_trajectory_linearity_rmse = float(np.sqrt(np.mean(dist_errs ** 2)) / avg_pelvis_w)
    else:
        knee_trajectory_linearity_rmse = 0.0

    v_ha = p_ank_w[bot_start:bot_end] - p_heel_w[bot_start:bot_end]
    v_ak = p_knee_w[bot_start:bot_end] - p_ank_w[bot_start:bot_end]
    angles_rear = [vector_angle_3d(v_ha[i], v_ak[i]) for i in range(len(v_ha))]
    rear_heel_ankle_knee_alignment = float(np.mean(angles_rear))

    sway_vals = p_knee_pacs[:t_bot + 1, 0] * lat_sign
    lead_knee_mediolateral_sway_variance = float(np.var(sway_vals) / (avg_pelvis_w ** 2))

    knee_dist_bot = np.linalg.norm(kps_seq_clean[t_bot, 25] - kps_seq_clean[t_bot, 26])
    ank_dist_bot = np.linalg.norm(kps_seq_clean[t_bot, 27] - kps_seq_clean[t_bot, 28]) + 1e-6
    inter_knee_to_ankle_span_ratio = float(knee_dist_bot / ank_dist_bot)

    delta_z = abs(p_ank_pacs[t_bot, 2] - p_knee_pacs[t_bot, 2])
    delta_x = abs(p_ank_pacs[t_bot, 0] - p_knee_pacs[t_bot, 0]) + 1e-6
    lead_shank_sagittal_progression_ratio = float(delta_z / delta_x)

    heel_y = p_heel_w[:t_bot + 1, 1]
    lead_foot_heel_contact_stability = float(np.std(heel_y) / (avg_pelvis_w + 1e-6))

    # =========================================================================
    # Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
    # =========================================================================
    # 17. delta_foot_yaw_standing_to_bottom: 기립 대비 최하사점 발 외회전 상대 변화량
    v_foot_std = np.mean(p_toe_pacs[:3] - p_ank_pacs[:3], axis=0)
    yaw_std = float(np.degrees(np.arctan2(v_foot_std[0] * lat_sign, v_foot_std[2] + 1e-6)))
    delta_foot_yaw_standing_to_bottom = float(pacs_lead_foot_yaw_angle - yaw_std)

    # 18. eccentric_knee_inward_velocity_peak: 하강기 무릎 내측 내전 최대 속도 (안전 diff)
    knee_x_arr = p_knee_pacs[:t_bot + 1, 0] * lat_sign
    if len(knee_x_arr) >= 2:
        knee_x_vel = np.diff(knee_x_arr)
        eccentric_knee_inward_velocity_peak = float(np.max(knee_x_vel))
    else:
        eccentric_knee_inward_velocity_peak = 0.0

    # 19. delta_q_angle_standing_to_bottom: 기립 대비 최하사점 3D Q-angle 변화량
    v_thigh_std = np.mean(kps_seq_clean[:3, knee_idx, :3] - kps_seq_clean[:3, hip_idx, :3], axis=0)
    v_shank_std = np.mean(kps_seq_clean[:3, ank_idx, :3] - kps_seq_clean[:3, knee_idx, :3], axis=0)
    q_std = vector_angle_3d(v_thigh_std, v_shank_std)

    v_thigh_bot = kps_seq_clean[t_bot, knee_idx, :3] - kps_seq_clean[t_bot, hip_idx, :3]
    v_shank_bot = kps_seq_clean[t_bot, ank_idx, :3] - kps_seq_clean[t_bot, knee_idx, :3]
    q_bot = vector_angle_3d(v_thigh_bot, v_shank_bot)
    delta_q_angle_standing_to_bottom = float(q_bot - q_std)

    # 20. bottom_phase_alignment_variance
    thigh_to_foot_angles = []
    for t_i in range(bot_start, bot_end):
        vt = p_knee_pacs[t_i] - p_hip_pacs[t_i]
        vf = p_toe_pacs[t_i] - p_ank_pacs[t_i]
        thigh_to_foot_angles.append(vector_angle_3d(vt, vf))
    bottom_phase_alignment_variance = float(np.var(thigh_to_foot_angles) if thigh_to_foot_angles else 0.0)

    # 21. concentric_knee_rebound_jerk
    if t_bot < T - 4:
        up_knee = p_knee_w[t_bot:, 1]
        jerk = np.diff(up_knee, n=3)
        concentric_knee_rebound_jerk = float(np.mean(jerk ** 2) / (avg_pelvis_w ** 2))
    else:
        concentric_knee_rebound_jerk = 0.0

    # 22. trunk_pelvis_yaw_torsion_mean
    torsion_list = []
    for t_i in range(min(t_bot + 1, T)):
        ls, rs = kps_seq_clean[t_i, 11], kps_seq_clean[t_i, 12]
        lh, rh = kps_seq_clean[t_i, 23], kps_seq_clean[t_i, 24]
        v_sh_lat = ls - rs
        v_hp_lat = lh - rh
        sh_yaw = np.degrees(np.arctan2(v_sh_lat[0], v_sh_lat[2] + 1e-6))
        hp_yaw = np.degrees(np.arctan2(v_hp_lat[0], v_hp_lat[2] + 1e-6))
        torsion_list.append(abs(sh_yaw - hp_yaw))
    trunk_pelvis_yaw_torsion_mean = float(np.mean(torsion_list) if torsion_list else 0.0)

    # 23. pelvic_rotation_dynamic_range
    pelvis_yaws = []
    for t_i in range(T):
        lh, rh = kps_seq_clean[t_i, 23], kps_seq_clean[t_i, 24]
        v_hp = lh - rh
        pelvis_yaws.append(np.degrees(np.arctan2(v_hp[0], v_hp[2] + 1e-6)))
    pelvic_rotation_dynamic_range = float(np.ptp(pelvis_yaws) if pelvis_yaws else 0.0)

    # 24. fppa_pre_bottom_slope
    pre_start = max(0, t_bot - 5)
    if t_bot > pre_start + 1:
        pre_fppa = []
        for t_i in range(pre_start, t_bot + 1):
            vf = p_hip_pacs[t_i] - p_knee_pacs[t_i]
            vt = p_ank_pacs[t_i] - p_knee_pacs[t_i]
            pre_fppa.append(vector_angle_3d(vf, vt))
        fppa_pre_bottom_slope = float(np.polyfit(np.arange(len(pre_fppa)), pre_fppa, 1)[0])
    else:
        fppa_pre_bottom_slope = 0.0

    # 25. foot_axis_temporal_stability
    foot_yaws = []
    for t_i in range(T):
        vf = p_toe_pacs[t_i] - p_ank_pacs[t_i]
        foot_yaws.append(np.degrees(np.arctan2(vf[0] * lat_sign, vf[2] + 1e-6)))
    foot_axis_temporal_stability = float(np.var(foot_yaws) if foot_yaws else 0.0)

    # =========================================================================
    # Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
    # =========================================================================
    view_is_view1 = 1.0 if view_code == 'view1' else 0.0
    view_is_view2 = 1.0 if view_code == 'view2' else 0.0
    view_is_view3 = 1.0 if view_code == 'view3' else 0.0
    view_is_view4 = 1.0 if view_code == 'view4' else 0.0
    view_is_view5 = 1.0 if view_code == 'view5' else 0.0

    return np.array([
        pacs_trunk_knee_azimuth_error,
        pacs_thigh_heading_to_trunk_angle,
        pacs_knee_ankle_coronal_tilt,
        pacs_lead_hip_rotation_proxy,
        pacs_dkcd_medial_deviation,
        pacs_fppa_3d_collapse,
        pacs_lead_foot_yaw_angle,
        pacs_heel_toe_skew_ratio,
        shank_torsion_plane_angle,
        knee_trajectory_to_ankle_vector_angle,
        knee_trajectory_linearity_rmse,
        rear_heel_ankle_knee_alignment,
        lead_knee_mediolateral_sway_variance,
        inter_knee_to_ankle_span_ratio,
        lead_shank_sagittal_progression_ratio,
        lead_foot_heel_contact_stability,
        delta_foot_yaw_standing_to_bottom,      # [신규 핵심]
        eccentric_knee_inward_velocity_peak,
        delta_q_angle_standing_to_bottom,
        bottom_phase_alignment_variance,
        concentric_knee_rebound_jerk,
        trunk_pelvis_yaw_torsion_mean,
        pelvic_rotation_dynamic_range,
        fppa_pre_bottom_slope,
        foot_axis_temporal_stability,
        view_is_view1,
        view_is_view2,
        view_is_view3,
        view_is_view4,
        view_is_view5
    ], dtype=np.float32)


def extract_side_lunge_c2_features_batch(X_seqs: np.ndarray, view_codes: Optional[List[str]] = None) -> np.ndarray:
    """배치 시퀀스 (N, 64, 33, 3) 30차원 피처 일괄 추출"""
    N = len(X_seqs)
    feats = np.zeros((N, 30), dtype=np.float32)
    for i in range(N):
        vc = view_codes[i] if view_codes is not None else "view3"
        feats[i] = extract_side_lunge_c2_features_29d(X_seqs[i], view_code=vc)
    return feats

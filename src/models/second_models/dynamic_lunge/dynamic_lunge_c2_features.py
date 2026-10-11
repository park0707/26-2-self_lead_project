# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드 / 06_스텝_백워드)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: dynamic_lunge_c2_features.py
Purpose: 바벨 런지 bl2_v6 SOTA 피처 엔지니어링 100% 계승
         - 64프레임 순수 MediaPipe 33관절 3D 텐서 기반 18차원 정예 생체역학 피처 추출
         1. [선행 발 지향 및 족부 안정성 4D]:
            - lead_foot_inward_angle_bottom: 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각
            - lead_foot_axis_stability: 하강/상승 구간 발끝 방향 벡터의 요동 분산
            - lead_foot_inward_ratio: 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
            - lead_heel_toe_skew: 발뒤꿈치 대비 발끝의 내측 편차 거리비
         2. [슬관절 동적 정렬 및 외반 변위 4D]:
            - lead_knee_valgus_peak: 최하사점 부근 무릎 외반 최대 피크 각도
            - lead_knee_lateral_ratio: 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
            - lead_knee_velocity_ml: 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
            - lead_knee_ankle_alignment: 무릎 중심과 발목 중심의 수직 정렬 오차
         3. [관상면 동적 변화량 및 하지 연쇄 2D]:
            - fppa_eccentric_to_bottom_delta: 신전 상태 대비 최하사점에서의 FPPA 각도 붕괴량
            - directional_misalignment_thigh_foot: 신체 직교계 상 대퇴 진행 벡터 vs 발 장축 사잇각
         4. [체간 진행 방향 및 골반 회전 2D]:
            - trunk_pelvis_yaw_torsion: 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림 각도
            - pelvic_rotation_bottom: 최하사점에서 골반 축의 수평 회전 편차각
         5. [상하체 결합 및 진행 사잇각 3D]:
            - lead_hip_adduction_bottom: 선행 대퇴골의 관상면 내전 각도
            - trunk_foot_angle_deviation: 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
            - inter_knee_dist_ratio: 양 무릎 간 3D 이격거리 대비 골반 너비 비율
         6. [하지 기계축 편위 프록시 1D]:
            - dkcd_medial_deviation_bottom: 고관절-발목 기계적 축 대비 무릎 중심 내측 변위량 (DKCD)
         7. [생체역학 기립 베이스라인 정규화 & 수평 정사영 2D]:
            - knee_displacement_foot_heading_angle: 기립->최하사점 무릎 수평(XZ) 변위축 vs 발 장축 사잇각
            - delta_q_angle_standing_to_bottom: 기립 시 대비 최하사점 Q-angle의 동적 변화량 (선천적 O/X다리 노이즈 차단)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

FEATURE_NAMES_18D = [
    # Group A: 선행 발 지향 및 족부 안정성 (4D)
    "lead_foot_inward_angle_bottom",        # 1. 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각
    "lead_foot_axis_stability",             # 2. 하강/상승 구간 발끝 방향 벡터의 요동 분산
    "lead_foot_inward_ratio",               # 3. 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
    "lead_heel_toe_skew",                   # 4. 발뒤꿈치 대비 발끝의 내측 편차 거리비

    # Group B: 슬관절 동적 정렬 및 외반 변위 (4D)
    "lead_knee_valgus_peak",                # 5. 최하사점 부근 무릎 외반 최대 피크 각도
    "lead_knee_lateral_ratio",              # 6. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
    "lead_knee_velocity_ml",                # 7. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    "lead_knee_ankle_alignment",            # 8. 무릎 중심과 발목 중심의 수직 정렬 오차

    # Group C: 관상면 동적 변화량 및 하지 연쇄 (2D)
    "fppa_eccentric_to_bottom_delta",       # 9. 신전 상태 대비 최하사점에서의 FPPA 각도 붕괴량
    "directional_misalignment_thigh_foot",  # 10. 신체 직교계 상 대퇴 진행 벡터 vs 발 장축 사잇각

    # Group D: 체간 진행 방향 및 골반 회전 (2D)
    "trunk_pelvis_yaw_torsion",             # 11. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림 각도
    "pelvic_rotation_bottom",               # 12. 최하사점에서 골반 축의 수평 회전 편차각

    # Group E: 상하체 결합 및 진행 사잇각 (3D)
    "lead_hip_adduction_bottom",            # 13. 선행 대퇴골의 관상면 내전 각도
    "trunk_foot_angle_deviation",           # 14. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    "inter_knee_dist_ratio",                # 15. 양 무릎 간 3D 이격거리 대비 골반 너비 비율

    # Group F: 하지 기계축 편위 프록시 (1D)
    "dkcd_medial_deviation_bottom",         # 16. 고관절-발목 기계적 축 대비 무릎 중심 내측 변위량 (DKCD)

    # Group G: 기립 베이스라인 정규화 및 수평 정사영 (2D)
    "knee_displacement_foot_heading_angle", # 17. 기립->최하사점 무릎 수평 변위축 vs 발끝 장축 사잇각
    "delta_q_angle_standing_to_bottom"      # 18. 기립 대비 최하사점 Q-angle 동적 변화량
]


def build_pacs_basis(kps_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 프레임에서 Gram-Schmidt 정규 골반 좌표계(PACS) 직교 기저 구축
    - u_x: 골반 횡단축 (v_lat)
    - u_z: 골반 전방축 (v_lat x v_up)
    - u_y: 골반 상향축 (u_z x u_x)
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

    R_pelvis = np.vstack([u_x, u_y, u_z])
    return R_pelvis, mid_hip


def detect_lead_leg_and_bottom(kps_seq: np.ndarray) -> Tuple[bool, int]:
    """
    (64, 33, 3) 시퀀스에서 선행 다리(Lead Leg) 및 최하사점(Bottom Phase) 탐지
    Returns:
        is_left_lead (bool): 왼발 선행 여부
        bottom_idx (int): 최하사점 프레임 인덱스
    """
    l_knee_y = kps_seq[:, 25, 1]
    r_knee_y = kps_seq[:, 26, 1]
    l_max_y_idx = int(np.argmax(l_knee_y))
    r_max_y_idx = int(np.argmax(r_knee_y))

    if l_knee_y[l_max_y_idx] >= r_knee_y[r_max_y_idx]:
        is_left_lead = True
        bottom_idx = l_max_y_idx
    else:
        is_left_lead = False
        bottom_idx = r_max_y_idx

    l_ankle_z = kps_seq[:, 27, 2]
    r_ankle_z = kps_seq[:, 28, 2]
    l_step_fwd = np.min(l_ankle_z) - np.mean(l_ankle_z[:5])
    r_step_fwd = np.min(r_ankle_z) - np.mean(r_ankle_z[:5])
    if abs(l_knee_y[l_max_y_idx] - r_knee_y[r_max_y_idx]) < 0.05:
        is_left_lead = bool(l_step_fwd < r_step_fwd)

    return is_left_lead, bottom_idx


def extract_delta_features_2d(kps_33_3d: np.ndarray, is_left_lead: bool, t_bot: int) -> Tuple[float, float]:
    """
    17. knee_displacement_foot_heading_angle
    18. delta_q_angle_standing_to_bottom
    """
    hip_idx = 23 if is_left_lead else 24
    knee_idx = 25 if is_left_lead else 26
    ank_idx = 27 if is_left_lead else 28
    heel_idx = 29 if is_left_lead else 30
    toe_idx = 31 if is_left_lead else 32

    # 17. knee_displacement_foot_heading_angle (수평 XZ 평면 사잇각)
    v_foot_xz = np.array([kps_33_3d[t_bot, toe_idx, 0] - kps_33_3d[t_bot, heel_idx, 0],
                          kps_33_3d[t_bot, toe_idx, 2] - kps_33_3d[t_bot, heel_idx, 2]], dtype=np.float32)
    norm_foot = np.linalg.norm(v_foot_xz) + 1e-6
    u_foot = v_foot_xz / norm_foot

    v_disp_xz = np.array([kps_33_3d[t_bot, knee_idx, 0] - kps_33_3d[0, knee_idx, 0],
                          kps_33_3d[t_bot, knee_idx, 2] - kps_33_3d[0, knee_idx, 2]], dtype=np.float32)
    norm_disp = np.linalg.norm(v_disp_xz)

    if norm_disp > 0.02:
        u_disp = v_disp_xz / norm_disp
        dot = np.clip(np.dot(u_foot, u_disp), -1.0, 1.0)
        ang_kf = float(np.arccos(dot) * 180.0 / np.pi)
    else:
        ang_kf = 0.0

    # 18. delta_q_angle_standing_to_bottom (기립 대비 최하사점 Q-angle 변화량)
    v_thigh_std = np.mean(kps_33_3d[:3, knee_idx, :3] - kps_33_3d[:3, hip_idx, :3], axis=0)
    v_shank_std = np.mean(kps_33_3d[:3, ank_idx, :3] - kps_33_3d[:3, knee_idx, :3], axis=0)
    norm_ts = np.linalg.norm(v_thigh_std) * np.linalg.norm(v_shank_std) + 1e-6
    q_std = np.degrees(np.arccos(np.clip(np.dot(v_thigh_std, v_shank_std) / norm_ts, -1.0, 1.0)))

    v_thigh_bot = kps_33_3d[t_bot, knee_idx, :3] - kps_33_3d[t_bot, hip_idx, :3]
    v_shank_bot = kps_33_3d[t_bot, ank_idx, :3] - kps_33_3d[t_bot, knee_idx, :3]
    norm_tb = np.linalg.norm(v_thigh_bot) * np.linalg.norm(v_shank_bot) + 1e-6
    q_bot = np.degrees(np.arccos(np.clip(np.dot(v_thigh_bot, v_shank_bot) / norm_tb, -1.0, 1.0)))

    delta_q = float(q_bot - q_std)
    return ang_kf, delta_q


def extract_dynamic_lunge_c2_features_18d(kps_seq_3d: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 포즈 시퀀스로부터 18차원 정예 피처 추출 (bl2_v6 SOTA 계승)
    """
    if kps_seq_3d.shape[-1] == 4:
        visibilities = kps_seq_3d[:, :, 3]
        kps_seq_3d = kps_seq_3d[:, :, :3]
    else:
        visibilities = np.ones((kps_seq_3d.shape[0], kps_seq_3d.shape[1]), dtype=np.float32)

    T, V, C = kps_seq_3d.shape
    is_left_lead, t_bot = detect_lead_leg_and_bottom(kps_seq_3d)

    bot_start = max(0, t_bot - 4)
    bot_end = min(T, t_bot + 5)

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        lat_sign = -1.0

    pelvis_widths = np.linalg.norm(kps_seq_3d[:, 23, :3] - kps_seq_3d[:, 24, :3], axis=1)
    avg_pelvis_w = float(np.mean(pelvis_widths)) + 1e-6

    # PACS 변환 시퀀스 계산
    kps_pacs = np.zeros_like(kps_seq_3d)
    for t in range(T):
        R, c = build_pacs_basis(kps_seq_3d[t])
        kps_pacs[t] = (kps_seq_3d[t] - c[np.newaxis, :]) @ R.T

    # 1. lead_foot_inward_angle_bottom
    foot_vec_bot = np.mean(kps_seq_3d[bot_start:bot_end, toe_idx, :3] - kps_seq_3d[bot_start:bot_end, ank_idx, :3], axis=0)
    foot_horiz = np.array([foot_vec_bot[0], foot_vec_bot[2]], dtype=np.float32)
    foot_horiz /= (np.linalg.norm(foot_horiz) + 1e-6)
    pelvis_bot = np.mean(kps_seq_3d[bot_start:bot_end, 23, :3] - kps_seq_3d[bot_start:bot_end, 24, :3], axis=0)
    body_fwd = np.array([-pelvis_bot[2], pelvis_bot[0]], dtype=np.float32)
    body_fwd /= (np.linalg.norm(body_fwd) + 1e-6)
    dot_fpa = np.clip(np.dot(body_fwd, foot_horiz), -1.0, 1.0)
    lead_foot_inward_angle_bottom = float(np.degrees(np.arccos(dot_fpa)))

    # 2. lead_foot_axis_stability
    foot_vecs = kps_seq_3d[:, toe_idx, :3] - kps_seq_3d[:, ank_idx, :3]
    foot_yaws = np.degrees(np.arctan2(foot_vecs[:, 0], foot_vecs[:, 2] + 1e-6))
    lead_foot_axis_stability = float(np.var(foot_yaws))

    # 3. lead_foot_inward_ratio
    inward_frames = np.sum(foot_yaws * lat_sign < -12.0)
    lead_foot_inward_ratio = float(inward_frames / T)

    # 4. lead_heel_toe_skew
    heel_toe_dx = (kps_seq_3d[bot_start:bot_end, toe_idx, 0] - kps_seq_3d[bot_start:bot_end, heel_idx, 0]) * lat_sign
    lead_heel_toe_skew = float(np.mean(heel_toe_dx) / avg_pelvis_w)

    # 5. lead_knee_valgus_peak
    thigh_v = kps_seq_3d[:, hip_idx, :3] - kps_seq_3d[:, knee_idx, :3]
    shank_v = kps_seq_3d[:, ank_idx, :3] - kps_seq_3d[:, knee_idx, :3]
    thigh_u = thigh_v / (np.linalg.norm(thigh_v, axis=1, keepdims=True) + 1e-6)
    shank_u = shank_v / (np.linalg.norm(shank_v, axis=1, keepdims=True) + 1e-6)
    knee_angles = np.degrees(np.arccos(np.clip(np.sum(thigh_u * shank_u, axis=1), -1.0, 1.0)))
    lead_knee_valgus_peak = float(np.max(np.abs(180.0 - knee_angles[bot_start:bot_end])))

    # 6. lead_knee_lateral_ratio
    pelvis_c_x = (kps_seq_3d[:, 23, 0] + kps_seq_3d[:, 24, 0]) * 0.5
    knee_dev_x = (kps_seq_3d[:, knee_idx, 0] - pelvis_c_x) * lat_sign
    lead_knee_lateral_ratio = float(np.mean(knee_dev_x[bot_start:bot_end]) / avg_pelvis_w)

    # 7. lead_knee_velocity_ml
    knee_v_ml = np.gradient(kps_seq_3d[:, knee_idx, 0])
    lead_knee_velocity_ml = float(np.mean(knee_v_ml[bot_start:bot_end]) * lat_sign)

    # 8. lead_knee_ankle_alignment
    knee_ank_dx = np.abs(kps_seq_3d[bot_start:bot_end, knee_idx, 0] - kps_seq_3d[bot_start:bot_end, ank_idx, 0])
    lead_knee_ankle_alignment = float(np.mean(knee_ank_dx) / avg_pelvis_w)

    # 9. fppa_eccentric_to_bottom_delta
    hip_loc = kps_pacs[:, hip_idx, :]
    knee_loc = kps_pacs[:, knee_idx, :]
    ank_loc = kps_pacs[:, ank_idx, :]
    v_femur_cor = hip_loc[:, :2] - knee_loc[:, :2]
    v_tibia_cor = ank_loc[:, :2] - knee_loc[:, :2]
    u_f = v_femur_cor / (np.linalg.norm(v_femur_cor, axis=1, keepdims=True) + 1e-6)
    u_t = v_tibia_cor / (np.linalg.norm(v_tibia_cor, axis=1, keepdims=True) + 1e-6)
    fppa_body_arr = np.degrees(np.arccos(np.clip(np.sum(u_f * u_t, axis=1), -1.0, 1.0)))
    fppa_eccentric = np.mean(fppa_body_arr[:10])
    fppa_bottom_mean = np.mean(fppa_body_arr[bot_start:bot_end])
    fppa_eccentric_to_bottom_delta = float(fppa_eccentric - fppa_bottom_mean)

    # 10. directional_misalignment_thigh_foot
    thigh_vec_body = knee_loc[bot_start:bot_end, :] - hip_loc[bot_start:bot_end, :]
    foot_vec_body = kps_pacs[bot_start:bot_end, toe_idx, :] - kps_pacs[bot_start:bot_end, ank_idx, :]
    thigh_plane = np.mean(thigh_vec_body[:, [0, 2]], axis=0)
    foot_plane = np.mean(foot_vec_body[:, [0, 2]], axis=0)
    thigh_p_u = thigh_plane / (np.linalg.norm(thigh_plane) + 1e-6)
    foot_p_u = foot_plane / (np.linalg.norm(foot_plane) + 1e-6)
    directional_misalignment_thigh_foot = float(np.degrees(np.arccos(np.clip(np.dot(thigh_p_u, foot_p_u), -1.0, 1.0))))

    # 11. trunk_pelvis_yaw_torsion
    sh_vec = kps_seq_3d[:, 11, :3] - kps_seq_3d[:, 12, :3]
    sh_yaw = np.degrees(np.arctan2(sh_vec[:, 2], sh_vec[:, 0] + 1e-6))
    pel_vec = kps_seq_3d[:, 23, :3] - kps_seq_3d[:, 24, :3]
    pel_yaw = np.degrees(np.arctan2(pel_vec[:, 2], pel_vec[:, 0] + 1e-6))
    trunk_pelvis_yaw_torsion = float(np.mean(np.abs(sh_yaw[bot_start:bot_end] - pel_yaw[bot_start:bot_end])))

    # 12. pelvic_rotation_bottom
    pel_rot_bot = np.degrees(np.arctan2(np.mean(pel_vec[bot_start:bot_end, 2]), np.mean(pel_vec[bot_start:bot_end, 0]) + 1e-6))
    pelvic_rotation_bottom = float(abs(pel_rot_bot))

    # 13. lead_hip_adduction_bottom
    hip_knee_v = kps_seq_3d[:, knee_idx, :3] - kps_seq_3d[:, hip_idx, :3]
    hip_add_deg = np.degrees(np.arctan2(hip_knee_v[:, 0] * lat_sign, np.abs(hip_knee_v[:, 1]) + 1e-6))
    lead_hip_adduction_bottom = float(np.mean(hip_add_deg[bot_start:bot_end]))

    # 14. trunk_foot_angle_deviation
    trunk_fwd_3d = np.cross(sh_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    norm_tf = np.linalg.norm(trunk_fwd_3d, axis=1, keepdims=True) + 1e-6
    trunk_fwd_3d /= norm_tf
    norm_fv = np.linalg.norm(foot_vecs, axis=1, keepdims=True) + 1e-6
    foot_u_3d = foot_vecs / norm_fv
    trunk_foot_dot = np.clip(np.sum(trunk_fwd_3d * foot_u_3d, axis=1), -1.0, 1.0)
    trunk_foot_angle_deviation = float(np.mean(np.degrees(np.arccos(trunk_foot_dot))[bot_start:bot_end]))

    # 15. inter_knee_dist_ratio
    inter_knee_d = np.linalg.norm(kps_seq_3d[:, 25, :3] - kps_seq_3d[:, 26, :3], axis=1)
    inter_knee_dist_ratio = float(np.mean(inter_knee_d[bot_start:bot_end]) / avg_pelvis_w)

    # 16. dkcd_medial_deviation_bottom
    H_pacs = kps_pacs[:, hip_idx, :]
    K_pacs = kps_pacs[:, knee_idx, :]
    A_pacs = kps_pacs[:, ank_idx, :]
    mech_axis = A_pacs - H_pacs
    norm_mech = np.linalg.norm(mech_axis, axis=1, keepdims=True) + 1e-6
    u_mech = mech_axis / norm_mech
    dot_hk = np.sum((K_pacs - H_pacs) * u_mech, axis=1, keepdims=True)
    K_proj = H_pacs + dot_hk * u_mech
    d_K = K_pacs - K_proj
    medial_dev_arr = d_K[:, 0] * (-lat_sign)
    dkcd_medial_deviation_bottom = float(np.mean(medial_dev_arr[bot_start:bot_end]) / avg_pelvis_w)

    # 17 & 18. Delta features (수평 사잇각 & Q-angle 변화량)
    f17, f18 = extract_delta_features_2d(kps_seq_3d, is_left_lead, t_bot)

    feats_18 = np.array([
        lead_foot_inward_angle_bottom,
        lead_foot_axis_stability,
        lead_foot_inward_ratio,
        lead_heel_toe_skew,
        lead_knee_valgus_peak,
        lead_knee_lateral_ratio,
        lead_knee_velocity_ml,
        lead_knee_ankle_alignment,
        fppa_eccentric_to_bottom_delta,
        directional_misalignment_thigh_foot,
        trunk_pelvis_yaw_torsion,
        pelvic_rotation_bottom,
        lead_hip_adduction_bottom,
        trunk_foot_angle_deviation,
        inter_knee_dist_ratio,
        dkcd_medial_deviation_bottom,
        f17,
        f18
    ], dtype=np.float32)

    feats_18 = np.nan_to_num(feats_18, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats_18


def extract_dynamic_lunge_c2_features_18d_batch(X_seqs: np.ndarray) -> np.ndarray:
    N = X_seqs.shape[0]
    feats = np.zeros((N, 18), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_dynamic_lunge_c2_features_18d(X_seqs[i])
    return feats


if __name__ == '__main__':
    dummy = np.random.randn(64, 33, 3).astype(np.float32)
    f = extract_dynamic_lunge_c2_features_18d(dummy)
    print(f"[+] Dynamic Lunge C2 18D Features Extracted! Shape: {f.shape}")
    assert f.shape == (18,), f"Expected (18,), got {f.shape}"
    print("[+] All assertions passed for Dynamic Lunge 18D SOTA features!")

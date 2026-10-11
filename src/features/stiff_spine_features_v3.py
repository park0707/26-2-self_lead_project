# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Module: stiff_spine_features_v3.py
Purpose: 스티프 데드리프트 척추 중립(sd2 v3) 전용 고도화 생체역학 피처 추출기
  1. 등 굽힘 (Thoracic & Lumbar Flexion / Round Back) 특화 피처
  2. 허리 과신전 (Lumbar Hyperextension / Lockout Posterior Lean / Pelvic Anterior Sway) 특화 피처
  3. 바벨 및 체간 기울임 (Barbell Tilt Roll, Shoulder/Hip Roll, Lateral Barbell Asymmetry) 특화 피처
  4. CoRe (Contrastive Regression) 골드 정상 템플릿 상대 편차 피처 지원
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

FEATURE_NAMES_V3: List[str] = [
    # [1~6] 시상면 척추 만곡 및 요추 동역학 (등 굽힘 굴곡 감지)
    "stiff_lumbar_poly_a",                # [ 1] 요추 2차 포물선 곡률 계수 평균 (굴곡 정도)
    "stiff_lumbar_stiffness_sd",          # [ 2] 전 구간 요추 각도 변동성 (척추 강성 안정도)
    "stiff_spine_lateral_dev_max",        # [ 3] 척추 중심선 횡방향 이탈 최대치 (좌우 비대칭)
    "stiff_trunk_flex_speed_max",         # [ 4] 상체 굴곡 속도 최대 피크
    "stiff_cervical_min",                 # [ 5] 경추-흉추 최소 각도 (시선 및 경추 정렬)
    "stiff_active_rom",                   # [ 6] 손목 기준 수직 가동범위 (ROM)
    "stiff_cobb_mean",                    # [ 7] 요추 Cobb각 평균
    "stiff_shoulder_hip_twist_max",       # [ 8] 어깨선-골반선 3D 비틀림 각도
    "stiff_shear_stress_proxy",           # [ 9] 상체 모멘트암 기반 요추 전단력 프록시
    "stiff_pitch_rom",                    # [10] 상체 굴곡 각도 진폭 (Max - Min)
    "stiff_view_occlusion",               # [11] 후방 시점 가림 지표 (체간 Y축 Z투영)
    "stiff_view_comp_curv",               # [12] 3D 공간 만곡 보정 곡률 (Chord-Displacement)
    "stiff_ratio_arm_to_torso",           # [13] 팔 길이 대 체간 길이 비
    "stiff_ratio_femur_to_tibia",         # [14] 대퇴 대 하퇴 길이 비
    "stiff_ratio_leg_to_torso",           # [15] 다리 전체 대 체간 길이 비
    "stiff_knee_fixation_sd",             # [16] 무릎 관절 각도 고정 안정도
    "stiff_torso_pelvis_lateral_offset",  # [17] 체간 중심과 골반 중심의 횡방향 상대 편차
    "stiff_sagittal_spine_stiffness",     # [18] 하강-상승 전환 시 척추 형태 보존 강성 지표

    # [19~23] ★ 바벨 및 체간 좌우 기울임 특화 피처 (Key 211, 212, 213, 214 대응)
    "stiff_barbell_roll_max",             # [19] 양 손목(바벨) 수평 기울기 최대 각도 (Roll Max)
    "stiff_barbell_roll_mean",            # [20] 양 손목(바벨) 수평 기울기 평균 각도
    "stiff_barbell_lateral_drift",        # [21] 골반 중심 대비 바벨(양 손목 중점) 횡방향 비대칭 편차
    "stiff_shoulder_roll_max",            # [22] 어깨선 수평 기울기 최대 편차 (Shoulder Roll)
    "stiff_hip_roll_max",                 # [23] 골반선 수평 기울기 최대 편차 (Hip Roll)

    # [24~27] ★ 허리 과신전(Hyperextension / Lockout Lean) 특화 피처 (Key 215 대응)
    "stiff_lockout_posterior_lean",       # [24] 기립 락아웃 시 상체 후방 과신전 경사각 (수직선 후방 젖힘)
    "stiff_lockout_pelvic_sway",          # [25] 기립 락아웃 시 발목 대비 골반 전방 스웨이(전방 밀림) 변위
    "stiff_lumbar_lordosis_peak",         # [26] 요추 전만 과신전 음수 피크 (Lordosis Over-extension)
    "stiff_lockout_reversal_accel",       # [27] 락아웃 정점 도달 시 반대 방향 역감속 충격량

    # [28~30] ★ 등 굽힘(Round Back / Thoracic Kyphosis) 특화 피처 (Key 216 대응)
    "stiff_thoracic_kyphosis_peak",       # [28] 상부 흉추 곡률 및 어깨 전방 말림 피크 (Protraction)
    "stiff_thoracolumbar_c-curve_depth",  # [29] C7-S1 직선 대비 흉요추 만곡 최대 함몰 깊이 (C-Curve Depth)
    "stiff_roundback_inflexion_integral"  # [30] 하강 최저 변곡점 부근의 흉추 굴곡 시간 적분값
]


def extract_stiff_spine_features_v3(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None,
    gold_template_kps: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    단일 64프레임 시퀀스로부터 30D 스티프 데드리프트 척추 정예 v3 피처 추출
    kps_33_64f: (64, 33, 3)
    spine_9_64f: (64, 9, 3)
    """
    T = kps_33_64f.shape[0]  # 64
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, MP['LSH'], :]
    r_sh = kps_33_64f[:, MP['RSH'], :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, MP['LHIP'], :]
    r_hip = kps_33_64f[:, MP['RHIP'], :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, MP['LKNE'], :]
    r_knee = kps_33_64f[:, MP['RKNE'], :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, MP['LANK'], :]
    r_ank = kps_33_64f[:, MP['RANK'], :]
    mid_ank = (l_ank + r_ank) / 2.0

    l_wrist = kps_33_64f[:, MP['LWR'], :]
    r_wrist = kps_33_64f[:, MP['RWR'], :]
    mid_wrist = (l_wrist + r_wrist) / 2.0

    # 1. 체간 로컬 정규 직교 기저계
    v_torso = mid_sh - mid_hip
    torso_len = np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7
    y_hat = v_torso / torso_len

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. SpinePose 9분절 3D 좌표
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    # 3. 요추 2차 다항 곡률 poly_a
    z_s1 = np.sum((s_pts[:, 0] - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((s_pts[:, 1] - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((s_pts[:, 2] - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((s_pts[:, 3] - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((s_pts[:, 4] - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        p = np.polyfit(x_coords, lumbar_curves[t_idx], deg=2)
        poly_a_series[t_idx] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    # 4. 척추 횡방향 이탈
    x_s_disp = np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1)
    f3_lat_dev = float(np.max(np.abs(x_s_disp)))

    # 5. 상체 굴곡 속도
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))

    # 6. 경추-흉추 최소 각도
    v_cerv = s_pts[:, 8] - s_pts[:, 6]
    v_thor = s_pts[:, 5] - s_pts[:, 4]
    dot_ct = np.sum(v_cerv * v_thor, axis=-1)
    norm_ct = (np.linalg.norm(v_cerv, axis=-1) * np.linalg.norm(v_thor, axis=-1)) + 1e-7
    ct_angles = np.degrees(np.arccos(np.clip(dot_ct / norm_ct, -1.0, 1.0)))
    f5_cervical_min = float(np.min(ct_angles)) if spine_9_64f is not None else 150.0

    # 7. 가동범위
    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f6_active_rom = float(np.ptp(wrist_y_s))

    # 8. Cobb 각도
    v_l5_s1 = s_pts[:, 0] - s_pts[:, 1]
    v_l3_l1 = s_pts[:, 3] - s_pts[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    cobb_series = gaussian_filter1d(np.abs(cobb_angles), sigma=1.0)
    f7_cobb_mean = float(np.mean(cobb_series))

    # 9. 어깨-골반 비틀림
    v_hip = l_hip - r_hip
    twist = np.degrees(np.arccos(np.clip(
        np.sum(v_sh * v_hip, axis=-1) / (np.linalg.norm(v_sh, axis=-1) * np.linalg.norm(v_hip, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f8_twist_max = float(np.max(twist))

    # 10. 요추 전단력 프록시
    f9_shear = float(np.mean(np.sin(np.radians(torso_pitch))))

    # 11. 체간 각도 진폭
    f10_pitch_rom = float(np.ptp(torso_pitch))

    # 12. 후방 시점 가림 지표
    f11_view_occlusion = float(np.mean(np.abs(y_hat[:, 2])))

    # 13. 3D 공간 만곡 보정 곡률
    v_c7_s1 = s_pts[:, 6] - s_pts[:, 0]
    chord_len = np.linalg.norm(v_c7_s1, axis=-1) + 1e-6
    l3_disp = np.linalg.norm(np.cross(v_c7_s1, s_pts[:, 2] - s_pts[:, 0]), axis=-1) / chord_len
    f12_view_comp_curv = float(np.mean(l3_disp))

    # 14~16. 신체 비율
    mean_torso = float(np.mean(torso_len))
    arm_len = float(np.mean(np.linalg.norm(mid_sh - mid_wrist, axis=-1)))
    femur_len = float(np.mean(np.linalg.norm(mid_hip - mid_knee, axis=-1)))
    tibia_len = float(np.mean(np.linalg.norm(mid_knee - mid_ank, axis=-1)))
    f13_arm_torso = arm_len / max(1e-4, mean_torso)
    f14_femur_tibia = femur_len / max(1e-4, tibia_len)
    f15_leg_torso = (femur_len + tibia_len) / max(1e-4, mean_torso)

    # 17. 무릎 고정 안정도
    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f16_knee_fixation_sd = float(np.std(knee_angle))

    # 18. 체간-골반 횡방향 상대 오프셋
    torso_hip_dx = np.abs(mid_sh[:, 0] - mid_hip[:, 0])
    f17_torso_pelvis_offset = float(np.mean(torso_hip_dx))

    # 19. 척추 강성 유지 지표
    bottom_idx = int(np.argmax(wrist_y_s))
    curv_at_bottom = poly_a_series[bottom_idx]
    f18_sagittal_stiffness = float(curv_at_bottom - f1_poly_a)

    # -------------------------------------------------------------
    # ★ [20~24] 바벨 및 체간 좌우 기울임 특화 (Key 211, 212, 213, 214 대응)
    # -------------------------------------------------------------
    v_wrist = l_wrist - r_wrist  # 양 손목 = 바벨 수평선
    wrist_horiz_dx = v_wrist[:, 0]
    wrist_horiz_dy = v_wrist[:, 1]
    barbell_roll = np.degrees(np.arctan2(wrist_horiz_dy, np.abs(wrist_horiz_dx) + 1e-7))
    f19_barbell_roll_max = float(np.max(np.abs(barbell_roll)))
    f20_barbell_roll_mean = float(np.mean(np.abs(barbell_roll)))

    barbell_center_x = mid_wrist[:, 0]
    pelvis_center_x = mid_hip[:, 0]
    f21_barbell_lateral_drift = float(np.max(np.abs(barbell_center_x - pelvis_center_x)))

    sh_roll = np.degrees(np.arctan2(v_sh[:, 1], np.abs(v_sh[:, 0]) + 1e-7))
    f22_shoulder_roll_max = float(np.max(np.abs(sh_roll)))

    hip_roll = np.degrees(np.arctan2(v_hip[:, 1], np.abs(v_hip[:, 0]) + 1e-7))
    f23_hip_roll_max = float(np.max(np.abs(hip_roll)))

    # -------------------------------------------------------------
    # ★ [25~28] 허리 과신전(Hyperextension / Lockout Lean) 특화 (Key 215 대응)
    # -------------------------------------------------------------
    # 락아웃(기립 직립) 구간: 상위 10% 최저 가동범위 프레임들
    standing_frames = np.where(wrist_y_s < (np.min(wrist_y_s) + 0.15 * np.ptp(wrist_y_s)))[0]
    if len(standing_frames) == 0:
        standing_frames = np.array([0, T - 1])

    # 락아웃 시 체간이 수직 기준 뒤로 젖혀진 각도 (Posterior Lean)
    vertical_vec = np.array([0.0, -1.0, 0.0])  # 위쪽 방향
    torso_vectors = (mid_sh - mid_hip)[standing_frames]
    torso_z_projections = torso_vectors[:, 2]  # Z축 후방 돌출
    # 수직 대비 후방 기울기
    posterior_lean_angles = np.degrees(np.arctan2(torso_z_projections, -torso_vectors[:, 1] + 1e-7))
    f24_lockout_posterior_lean = float(np.max(np.maximum(0.0, posterior_lean_angles)))

    # 골반이 발목보다 앞으로 스웨이(밀림)된 변위 (Anterior Pelvic Sway)
    pelvic_forward_disp = (mid_hip - mid_ank)[standing_frames, 2]
    f25_lockout_pelvic_sway = float(np.max(pelvic_forward_disp))

    # 요추 전만 과신전 음수 피크 (Lordosis Over-extension)
    f26_lordosis_peak = float(np.min(poly_a_series[standing_frames]))

    # 락아웃 정점 가속도 역전 충격량
    standing_vel = torso_pitch_vel[standing_frames]
    f27_lockout_reversal = float(np.max(standing_vel) if len(standing_vel) > 0 else 0.0)

    # -------------------------------------------------------------
    # ★ [29~31] 등 굽힘(Round Back / Thoracic Kyphosis) 특화 (Key 216 대응)
    # -------------------------------------------------------------
    # 흉추 돌출 및 어깨 전방 말림 (Kyphosis Protraction)
    v_c7_midsh = mid_sh - s_pts[:, 6]
    thoracic_forward_disp = np.sum(v_c7_midsh * z_hat, axis=-1)
    f28_thoracic_kyphosis = float(np.max(thoracic_forward_disp))

    # C7-S1 직선 대비 척추 중간점(T8/L1) 함몰 깊이 (C-Curve Depth)
    v_c7_s1_full = s_pts[:, 6] - s_pts[:, 0]
    line_norm = np.linalg.norm(v_c7_s1_full, axis=-1, keepdims=True) + 1e-7
    mid_spine = s_pts[:, 3]  # L1
    proj = s_pts[:, 0] + np.sum((mid_spine - s_pts[:, 0]) * (v_c7_s1_full / line_norm), axis=-1, keepdims=True) * (v_c7_s1_full / line_norm)
    c_curve_dist = np.linalg.norm(mid_spine - proj, axis=-1)
    f29_c_curve_depth = float(np.max(c_curve_dist))

    # 최저 변곡점(하강 전환기) 흉추 굴곡 시간 적분값
    inflexion_window = max(2, int(T * 0.15))
    start_inf = max(0, bottom_idx - inflexion_window)
    end_inf = min(T, bottom_idx + inflexion_window)
    f30_roundback_inflexion = float(np.mean(thoracic_forward_disp[start_inf:end_inf]))

    feats_30d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min,
        f6_active_rom, f7_cobb_mean, f8_twist_max, f9_shear, f10_pitch_rom,
        f11_view_occlusion, f12_view_comp_curv, f13_arm_torso, f14_femur_tibia, f15_leg_torso,
        f16_knee_fixation_sd, f17_torso_pelvis_offset, f18_sagittal_stiffness,
        f19_barbell_roll_max, f20_barbell_roll_mean, f21_barbell_lateral_drift,
        f22_shoulder_roll_max, f23_hip_roll_max,
        f24_lockout_posterior_lean, f25_lockout_pelvic_sway, f26_lordosis_peak, f27_lockout_reversal,
        f28_thoracic_kyphosis, f29_c_curve_depth, f30_roundback_inflexion
    ], dtype=np.float32)

    return feats_30d

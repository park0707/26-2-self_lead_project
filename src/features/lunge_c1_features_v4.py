# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c1_features_v4.py
Purpose: 바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v4 피처 모듈 (30D)
         - v3.1 28D 정예 피처셋 완벽 계승 (Copy-and-Refine)
         - 선행연구 및 체형/시점 편향 극복 2대 신규 피처 추가 (28D -> 30D):
           1. lead_leg_coplanarity: 하지 3관절 평면 법선과 골반 횡축 정합도 (관상면 뒤틀림 감지)
           2. lead_knee_rom_ratio: 직립 기저 대비 최저점 무릎 상대 굴곡 가동범위 비율 (피험자 체형 캘리브레이션)
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# 30차원 공식 v4 도메인 정예 피처 명칭 목록
LUNGE_C1_FEATURE_NAMES_V4: List[str] = [
    # [그룹 1: 무릎 굴곡 및 시상면 90도 핵심 (6D)]
    "lead_knee_angle_min",            # [ 1] 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
    "lead_knee_angle_mean",           # [ 2] 최저점 윈도우 내 평균 굴곡 각도 (°)
    "lead_knee_90_error_abs",         # [ 3] |최소 각도 - 90°| 절대 오차 (°)
    "lead_knee_under_90_ratio",       # [ 4] 90도 미만 체류 프레임 비율
    "lead_knee_flex_velocity_max",    # [ 5] 하강 구간 최대 무릎 굴곡 각속도 (deg/frame)
    "sagittal_knee_90_error",         # [ 6] |ISB 시상면 무릎 굴곡각 - 90°| 절대 오차 (°)

    # [그룹 2: 하지 기하 및 정강이 정렬 + 사용자 권고안 + v4 신규 피처 (13D)]
    "lead_tibia_vertical_tilt",       # [ 7] 지면 수직선 대비 정강이 사잇각 (정강이 수직 유지도)
    "lead_femur_horizontal_tilt",     # [ 8] 지면 수평선 대비 대퇴골 사잇각 (대퇴 수평도)
    "lead_knee_toe_overhang",         # [ 9] 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (신체 스케일 정규화)
    "lead_shank_ankle_coupling",      # [10] 정강이 분절 각도와 발목 배측굴곡 각도 커플링 비율
    "lead_tibia_sagittal_sway",       # [11] 하강 중 정강이뼈 시상면 흔들림 폭
    "lead_femur_length_ratio",        # [12] 체간 대비 앞다리 대퇴골 정규화 길이
    "lead_tibia_length_ratio",        # [13] 체간 대비 앞다리 정강이 정규화 길이
    "rigid_bone_length_consistency",  # [14] 대퇴골-정강이 강체 길이 보존 일관성 지표 (가림 노이즈 감지)
    "rear_knee_ground_clearance",     # [15] 뒷무릎 지면 간격 프록시
    "lead_femur_vertical_drop_ratio", # [16] [v3.1] 대퇴골 정규화 앞다리 고관절-무릎 수직(Y) 거리 (|Δy| / L_femur)
    "front_view_femur_drop_gated",    # [17] [v3.1] 관상면 정면 지향도(Yaw) 게이팅 수직 낙차 비율
    "lead_leg_coplanarity",           # [18] [v4.0 신규] 하지 3관절면과 골반 관상면 직교 정합도 (뒤틀림 감지)
    "lead_knee_rom_ratio",            # [19] [v4.0 신규] 직립 대비 무릎 상대 굴곡 가동범위 비율 (체형 캘리브레이션)

    # [그룹 3: 양 하지 협응 및 보폭 (5D)]
    "lead_rear_knee_ratio",           # [20] 앞무릎 최소각 대비 뒷무릎 최소각 비율
    "rear_knee_angle_min",            # [21] 뒷다리 무릎 최저 각도 (°)
    "pelvic_drop_angle",              # [22] 관상면 상 골반 좌우 기울기 각도
    "stride_length_normalized",       # [23] 양 발목 사이 전후 보폭 (체간 정규화)
    "step_width_normalized",          # [24] 양 발목 사이 좌우 너비 (체간 정규화)

    # [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)]
    "lead_knee_accel_inflection",     # [25] 최저점 전환 순간의 각가속도 (deg/frame^2)
    "bottom_phase_dwell_ratio",       # [26] 전체 64프레임 중 최저점 부근 체류 프레임 비율
    "lead_knee_dtw_distance",         # [27] 이상적 90도 골드 템플릿 곡선 대비 DTW 유사도 거리
    "lead_knee_visibility_mean",      # [28] 최저점 구간 앞무릎 키포인트 평균 신뢰도
    "kinetic_energy_inflection",      # [29] 최저 전환점 시 하지 관절 운동에너지 극소치 프록시
    "jerk_metric_lead_leg",           # [30] 앞다리 무릎 궤적 저크(Jerk) 평활도 지표
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 간의 사잇각 계산 (degree 단위, 0 ~ 180)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    dot = np.dot(v1, v2)
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def safe_savgol_filter(arr: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay 2차 다항식 필터"""
    if len(arr) < window:
        return arr.copy()
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
    except Exception:
        return arr.copy()


def compute_dtw_distance_1d(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """1차원 시계열 간의 Dynamic Time Warping (DTW) 거리 계산"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return float(dtw_matrix[n, m] / (n + m))


def detect_lead_leg_robust(kps_64f: np.ndarray) -> str:
    """하강/최저점 구간 발목 전후(Z축) 변위 분석을 통한 앞다리 판별"""
    mid_hip_y = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    inflection_idx = int(np.argmax(mid_hip_y))
    eval_f = max(10, min(inflection_idx, 54))

    left_disp_z = kps_64f[eval_f, MP_LEFT_ANKLE, 2] - kps_64f[0, MP_LEFT_ANKLE, 2]
    right_disp_z = kps_64f[eval_f, MP_RIGHT_ANKLE, 2] - kps_64f[0, MP_RIGHT_ANKLE, 2]

    if left_disp_z < right_disp_z:
        return 'left'
    elif right_disp_z < left_disp_z:
        return 'right'
    else:
        left_knee_flex = abs(kps_64f[eval_f, MP_LEFT_KNEE, 1] - kps_64f[eval_f, MP_LEFT_HIP, 1])
        right_knee_flex = abs(kps_64f[eval_f, MP_RIGHT_KNEE, 1] - kps_64f[eval_f, MP_RIGHT_HIP, 1])
        return 'left' if left_knee_flex < right_knee_flex else 'right'


def find_dual_anchor_inflection_point(
    kps_64f: np.ndarray,
    angles_sg: np.ndarray
) -> Tuple[int, int, int]:
    """골반 Y 극대점 + 무릎 각도 극소점 이중 앵커링 변곡점 탐색"""
    mid_hip_y = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    hip_inflection = int(np.argmax(mid_hip_y))

    lead_min_idx = int(np.argmin(angles_sg))

    if abs(hip_inflection - lead_min_idx) <= 10:
        inflection_idx = int(round(0.6 * lead_min_idx + 0.4 * hip_inflection))
    else:
        inflection_idx = lead_min_idx

    inflection_idx = max(8, min(inflection_idx, 55))
    win_start = max(0, inflection_idx - 3)
    win_end = min(64, inflection_idx + 4)

    return inflection_idx, win_start, win_end


def extract_lunge_c1_features_v4(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v4 피처 추출기 (30D)
    """
    lead_side = detect_lead_leg_robust(kps_64f)

    if lead_side == 'left':
        lead_hip_idx = MP_LEFT_HIP
        lead_knee_idx = MP_LEFT_KNEE
        lead_ankle_idx = MP_LEFT_ANKLE
        rear_hip_idx = MP_RIGHT_HIP
        rear_knee_idx = MP_RIGHT_KNEE
        rear_ankle_idx = MP_RIGHT_ANKLE
    else:
        lead_hip_idx = MP_RIGHT_HIP
        lead_knee_idx = MP_RIGHT_KNEE
        lead_ankle_idx = MP_RIGHT_ANKLE
        rear_hip_idx = MP_LEFT_HIP
        rear_knee_idx = MP_LEFT_KNEE
        rear_ankle_idx = MP_LEFT_ANKLE

    lead_angles_raw = np.zeros(64, dtype=np.float32)
    rear_angles_raw = np.zeros(64, dtype=np.float32)
    torso_lengths = []
    femur_lengths = []
    tibia_lengths = []

    for f in range(64):
        v_femur = kps_64f[f, lead_hip_idx] - kps_64f[f, lead_knee_idx]
        v_tibia = kps_64f[f, lead_ankle_idx] - kps_64f[f, lead_knee_idx]
        lead_angles_raw[f] = vector_angle_3d(v_femur, v_tibia)

        v_rear_fem = kps_64f[f, rear_hip_idx] - kps_64f[f, rear_knee_idx]
        v_rear_tib = kps_64f[f, rear_ankle_idx] - kps_64f[f, rear_knee_idx]
        rear_angles_raw[f] = vector_angle_3d(v_rear_fem, v_rear_tib)

        torso_len = np.linalg.norm(kps_64f[f, MP_LEFT_SHOULDER] - kps_64f[f, MP_LEFT_HIP])
        femur_len = np.linalg.norm(v_femur)
        tibia_len = np.linalg.norm(v_tibia)
        if torso_len > 1e-4:
            torso_lengths.append(torso_len)
        if femur_len > 1e-4:
            femur_lengths.append(femur_len)
        if tibia_len > 1e-4:
            tibia_lengths.append(tibia_len)

    torso_scale = float(np.median(torso_lengths)) if torso_lengths else 1.0
    L_femur_median = float(np.median(femur_lengths)) if femur_lengths else 1.0

    lead_angles_sg = safe_savgol_filter(lead_angles_raw, window=7, poly=2)
    rear_angles_sg = safe_savgol_filter(rear_angles_raw, window=7, poly=2)

    inflection_idx, win_start, win_end = find_dual_anchor_inflection_point(kps_64f, lead_angles_sg)

    # --- [그룹 1: 무릎 굴곡 및 시상면 90도 핵심 (6D)] ---
    lead_knee_angle_min = float(np.min(lead_angles_sg[win_start:win_end]))
    lead_knee_angle_mean = float(np.mean(lead_angles_sg[win_start:win_end]))
    lead_knee_90_error_abs = float(abs(lead_knee_angle_min - 90.0))
    lead_knee_under_90_ratio = float(np.mean(lead_angles_sg < 90.0))

    vel_traj = np.gradient(lead_angles_sg)
    descent_vel = vel_traj[:inflection_idx]
    lead_knee_flex_velocity_max = float(abs(np.min(descent_vel))) if len(descent_vel) > 0 else 0.0

    # ISB 시상면 투영
    pelvis_x = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    norm_px = np.linalg.norm(pelvis_x)
    if norm_px > 1e-6:
        pelvis_x /= norm_px
    else:
        pelvis_x = np.array([1.0, 0.0, 0.0])

    torso_up = (
        (kps_64f[inflection_idx, MP_LEFT_SHOULDER] + kps_64f[inflection_idx, MP_RIGHT_SHOULDER]) / 2.0 -
        (kps_64f[inflection_idx, MP_LEFT_HIP] + kps_64f[inflection_idx, MP_RIGHT_HIP]) / 2.0
    )
    torso_up_proj = torso_up - np.dot(torso_up, pelvis_x) * pelvis_x
    norm_tup = np.linalg.norm(torso_up_proj)
    pelvis_y = torso_up_proj / norm_tup if norm_tup > 1e-6 else np.array([0.0, -1.0, 0.0])
    pelvis_z = np.cross(pelvis_x, pelvis_y)
    norm_pz = np.linalg.norm(pelvis_z)
    if norm_pz > 1e-6:
        pelvis_z /= norm_pz

    v_femur_bot = kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_knee_idx]
    v_tibia_bot = kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, lead_knee_idx]
    femur_sag = np.array([np.dot(v_femur_bot, pelvis_y), np.dot(v_femur_bot, pelvis_z)])
    tibia_sag = np.array([np.dot(v_tibia_bot, pelvis_y), np.dot(v_tibia_bot, pelvis_z)])
    cos_sag = np.dot(femur_sag, tibia_sag) / (np.linalg.norm(femur_sag) * np.linalg.norm(tibia_sag) + 1e-6)
    sagittal_knee_angle = float(np.degrees(np.arccos(np.clip(cos_sag, -1.0, 1.0))))
    sagittal_knee_90_error = float(abs(sagittal_knee_angle - 90.0))

    # --- [그룹 2: 하지 기하 및 정강이 정렬 + 사용자 권고안 + v4 피처 (13D)] ---
    knee_bot = kps_64f[inflection_idx, lead_knee_idx]
    ankle_bot = kps_64f[inflection_idx, lead_ankle_idx]
    hip_bot = kps_64f[inflection_idx, lead_hip_idx]

    tibia_vec = knee_bot - ankle_bot
    vertical_axis = np.array([0.0, -1.0, 0.0])
    lead_tibia_vertical_tilt = vector_angle_3d(tibia_vec, vertical_axis)

    femur_vec = knee_bot - hip_bot
    horizontal_axis = np.array([femur_vec[0], 0.0, femur_vec[2]])
    lead_femur_horizontal_tilt = vector_angle_3d(femur_vec, horizontal_axis)

    foot_idx = MP_LEFT_FOOT_INDEX if lead_side == 'left' else MP_RIGHT_FOOT_INDEX
    toe_pt = kps_64f[inflection_idx, foot_idx]
    lead_knee_toe_overhang = float((knee_bot[2] - toe_pt[2]) / torso_scale)

    heel_idx = MP_LEFT_HEEL if lead_side == 'left' else MP_RIGHT_HEEL
    heel_pt = kps_64f[inflection_idx, heel_idx]
    foot_vec = toe_pt - heel_pt
    dorsiflex_angle = vector_angle_3d(tibia_vec, foot_vec)
    lead_shank_ankle_coupling = float(lead_tibia_vertical_tilt / (dorsiflex_angle + 1e-4))

    tibia_angles_descent = []
    for f in range(max(0, inflection_idx - 15), inflection_idx):
        k_pt = kps_64f[f, lead_knee_idx]
        a_pt = kps_64f[f, lead_ankle_idx]
        tibia_angles_descent.append(vector_angle_3d(k_pt - a_pt, vertical_axis))
    lead_tibia_sagittal_sway = float(np.ptp(tibia_angles_descent)) if tibia_angles_descent else 0.0

    lead_femur_length_ratio = float(np.median(femur_lengths) / torso_scale)
    lead_tibia_length_ratio = float(np.median(tibia_lengths) / torso_scale)

    combined_bone = femur_lengths + tibia_lengths
    rigid_bone_length_consistency = float(np.std(combined_bone) / (np.mean(combined_bone) + 1e-6))

    rear_knee_bot = kps_64f[inflection_idx, rear_knee_idx]
    rear_knee_ground_clearance = float((ankle_bot[1] - rear_knee_bot[1]) / torso_scale)

    # v3.1 수직 낙차 비율 및 관상면 정면 게이팅
    delta_y = abs(knee_bot[1] - hip_bot[1])
    lead_femur_vertical_drop_ratio = float(delta_y / L_femur_median)

    pelvis_raw_vec = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    yaw = np.arctan2(abs(pelvis_raw_vec[2]), abs(pelvis_raw_vec[0]) + 1e-6)
    front_confidence = float(np.cos(yaw))
    front_view_femur_drop_gated = float(lead_femur_vertical_drop_ratio * front_confidence)

    # === [v4.0 신규 피처 1] 하지 3관절면과 골반 관상면 직교 정합도 (Coplanarity)
    leg_norm = np.cross(v_femur_bot, v_tibia_bot)
    norm_leg = np.linalg.norm(leg_norm)
    leg_norm = leg_norm / (norm_leg + 1e-6) if norm_leg > 1e-6 else np.array([0.0, 0.0, 1.0])
    u_hip_vec = pelvis_raw_vec / (np.linalg.norm(pelvis_raw_vec) + 1e-6)
    lead_leg_coplanarity = float(abs(np.dot(leg_norm, u_hip_vec)))

    # === [v4.0 신규 피처 2] 직립 기저 대비 무릎 상대 굴곡 가동범위 비율 (ROM Ratio)
    standing_angles = []
    for f in range(min(5, len(kps_64f))):
        vf_st = kps_64f[f, lead_hip_idx] - kps_64f[f, lead_knee_idx]
        vt_st = kps_64f[f, lead_ankle_idx] - kps_64f[f, lead_knee_idx]
        standing_angles.append(vector_angle_3d(vf_st, vt_st))
    standing_knee_angle = float(np.median(standing_angles)) if standing_angles else 175.0
    knee_rom = max(0.0, standing_knee_angle - lead_knee_angle_min)
    lead_knee_rom_ratio = float(knee_rom / (standing_knee_angle + 1e-4))

    # --- [그룹 3: 양 하지 협응 및 보폭 (5D)] ---
    rear_knee_min = float(np.min(rear_angles_sg[win_start:win_end]))
    lead_rear_knee_ratio = float(lead_knee_angle_min / (rear_knee_min + 1e-4))

    pelvis_vec = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    horizontal_ground = np.array([1.0, 0.0, 0.0])
    pelvic_drop_angle = float(vector_angle_3d(pelvis_vec, horizontal_ground))

    l_ank = kps_64f[inflection_idx, MP_LEFT_ANKLE]
    r_ank = kps_64f[inflection_idx, MP_RIGHT_ANKLE]
    stride_length_normalized = float(abs(l_ank[2] - r_ank[2]) / torso_scale)
    step_width_normalized = float(abs(l_ank[0] - r_ank[0]) / torso_scale)

    # --- [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)] ---
    accel_traj = np.gradient(vel_traj)
    lead_knee_accel_inflection = float(abs(accel_traj[inflection_idx]))

    bottom_threshold = lead_knee_angle_min + 5.0
    bottom_phase_dwell_ratio = float(np.mean(lead_angles_sg <= bottom_threshold))

    ideal_90_curve = 170.0 - 80.0 * np.sin(np.linspace(0, np.pi, 64))
    lead_knee_dtw_distance = compute_dtw_distance_1d(lead_angles_sg, ideal_90_curve)

    if vis_64f is not None and len(vis_64f) == 64:
        lead_knee_visibility_mean = float(np.mean(vis_64f[win_start:win_end, lead_knee_idx]))
    else:
        lead_knee_visibility_mean = 1.0

    joint_velocities = np.linalg.norm(np.diff(kps_64f[:, [lead_knee_idx, rear_knee_idx], :], axis=0), axis=-1)
    kinetic_energy_inflection = float(np.sum(joint_velocities[max(0, inflection_idx - 1)] ** 2))

    lead_knee_traj_3d = kps_64f[:, lead_knee_idx, :]
    knee_accel_3d = np.diff(lead_knee_traj_3d, n=2, axis=0)
    knee_jerk_3d = np.diff(knee_accel_3d, n=1, axis=0)
    jerk_metric_lead_leg = float(np.mean(np.linalg.norm(knee_jerk_3d, axis=-1)))

    features = {
        "lead_knee_angle_min": lead_knee_angle_min,
        "lead_knee_angle_mean": lead_knee_angle_mean,
        "lead_knee_90_error_abs": lead_knee_90_error_abs,
        "lead_knee_under_90_ratio": lead_knee_under_90_ratio,
        "lead_knee_flex_velocity_max": lead_knee_flex_velocity_max,
        "sagittal_knee_90_error": sagittal_knee_90_error,

        "lead_tibia_vertical_tilt": lead_tibia_vertical_tilt,
        "lead_femur_horizontal_tilt": lead_femur_horizontal_tilt,
        "lead_knee_toe_overhang": lead_knee_toe_overhang,
        "lead_shank_ankle_coupling": lead_shank_ankle_coupling,
        "lead_tibia_sagittal_sway": lead_tibia_sagittal_sway,
        "lead_femur_length_ratio": lead_femur_length_ratio,
        "lead_tibia_length_ratio": lead_tibia_length_ratio,
        "rigid_bone_length_consistency": rigid_bone_length_consistency,
        "rear_knee_ground_clearance": rear_knee_ground_clearance,
        "lead_femur_vertical_drop_ratio": lead_femur_vertical_drop_ratio,
        "front_view_femur_drop_gated": front_view_femur_drop_gated,
        "lead_leg_coplanarity": lead_leg_coplanarity,
        "lead_knee_rom_ratio": lead_knee_rom_ratio,

        "lead_rear_knee_ratio": lead_rear_knee_ratio,
        "rear_knee_angle_min": rear_knee_min,
        "pelvic_drop_angle": pelvic_drop_angle,
        "stride_length_normalized": stride_length_normalized,
        "step_width_normalized": step_width_normalized,

        "lead_knee_accel_inflection": lead_knee_accel_inflection,
        "bottom_phase_dwell_ratio": bottom_phase_dwell_ratio,
        "lead_knee_dtw_distance": lead_knee_dtw_distance,
        "lead_knee_visibility_mean": lead_knee_visibility_mean,
        "kinetic_energy_inflection": kinetic_energy_inflection,
        "jerk_metric_lead_leg": jerk_metric_lead_leg,
    }

    return features


def extract_lunge_c1_feature_vector_v4(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """순수 30D 1차원 피처 벡터 반환 (LUNGE_C1_FEATURE_NAMES_V4 순서 보장)"""
    feat_dict = extract_lunge_c1_features_v4(kps_64f, vis_64f)
    vec = np.array([feat_dict[name] for name in LUNGE_C1_FEATURE_NAMES_V4], dtype=np.float32)
    return vec

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Module: side_lunge_c1_features_v8.py
Purpose: 사이드 런지 c1(앞다리 무릎 각도 90도) SOTA v8 피처 모듈 (40D)
         - sl1_v4(36D, 사용자 최고 선호 성능 버전) 완전 계승
         - sl3(안 굽히는 다리 폄 F1 71.33% 돌파)의 검증된 4대 혁신 기법 융합 (36D -> 40D):
           1. pelvic_tilt_roll_slope: 최하사점 작동 골반 vs 신전 골반 수직 높이차 / 체간 길이 (sl3 기여도 1위 13.78% 지표)
           2. lead_leg_length_compression_ratio: 기립 대비 최하사점 앞다리 골반-발목 3D 거리 압축 비율
           3. trail_to_lead_knee_rom_ratio: 신전 다리 가동범위 대비 작동 다리 가동범위 협응 비율
           4. lead_knee_ratio_to_stand: 피험자 기립 시작 각도 대비 최하사점 굴곡 비율
         - c3 공간 거리 다리 분리 엔진 (Spatial Distance Leg Separation) 100% 통합
         - 100% 순수 비전 MediaPipe 33관절 3D 포즈 텐서 기반 (AI-Hub JSON 무누수 0.00%)
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

# 40차원 공식 v8 도메인 정예 피처 명칭 목록
SIDE_LUNGE_C1_FEATURE_NAMES_V8: List[str] = [
    # [그룹 1: 무릎 굴곡 및 시상면 90도 핵심 (6D)]
    "lead_knee_angle_min",            # [ 1] 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
    "lead_knee_angle_mean",           # [ 2] 최저점 윈도우 내 평균 굴곡 각도 (°)
    "lead_knee_90_error_abs",         # [ 3] |최소 각도 - 90°| 절대 오차 (°)
    "lead_knee_under_90_ratio",       # [ 4] 90도 미만 체류 프레임 비율
    "lead_knee_flex_velocity_max",    # [ 5] 하강 구간 최대 무릎 굴곡 각속도 (deg/frame)
    "sagittal_knee_90_error",         # [ 6] |ISB 시상면 무릎 굴곡각 - 90°| 절대 오차 (°)

    # [그룹 2: 하지 기하 및 정강이 정렬 + 수직 불변량 (13D)]
    "lead_tibia_vertical_tilt",       # [ 7] 지면 수직선 대비 정강이 사잇각 (정강이 수직 유지도)
    "lead_femur_horizontal_tilt",     # [ 8] 지면 수평선 대비 대퇴골 사잇각 (대퇴 수평도)
    "lead_knee_toe_overhang",         # [ 9] 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (신체 스케일 정규화)
    "lead_shank_ankle_coupling",      # [10] 정강이 분절 각도와 발목 배측굴곡 각도 커플링 비율
    "lead_tibia_sagittal_sway",       # [11] 하강 중 정강이뼈 시상면 흔들림 폭
    "lead_femur_length_ratio",        # [12] 체간 대비 앞다리 대퇴골 정규화 길이
    "lead_tibia_length_ratio",        # [13] 체간 대비 앞다리 정강이 정규화 길이
    "rigid_bone_length_consistency",  # [14] 대퇴골-정강이 강체 길이 보존 일관성 지표 (가림 노이즈 감지)
    "rear_knee_ground_clearance",     # [15] 뒷무릎 지면 간격 프록시
    "lead_femur_vertical_drop_ratio", # [16] 대퇴골 정규화 앞다리 고관절-무릎 수직(Y) 거리 (|Δy| / L_femur)
    "front_view_femur_drop_gated",    # [17] 관상면 정면 지향도(Yaw) 게이팅 수직 낙차 비율
    "lead_leg_coplanarity",           # [18] 하지 3관절면과 골반 관상면 직교 정합도 (뒤틀림 감지)
    "lead_knee_rom_ratio",            # [19] 직립 대비 무릎 상대 굴곡 가동범위 비율

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

    # [그룹 5: sl1_v4 핵심 6대 불변 생체역학 피처 (6D)]
    "lead_knee_delta_from_standing",      # [31] 직립 상태 대비 최저점 무릎 굴곡 각도 낙차량 (Δθ, deg)
    "lead_rear_knee_angle_difference",    # [32] 펴는 다리 각도 - 굽히는 다리 각도 차이 (θ_rear - θ_lead, deg)
    "optical_axis_foreshortening_factor", # [33] 하지 3관절면 법선과 카메라 광축 사잇각 코사인 (투영 단축 팩터)
    "femur_horizontal_level_ratio",       # [34] 대퇴골 지면 수평 레벨링 지표 (|y_hip - y_knee| / L_femur)
    "pelvic_drop_to_stride_ratio",        # [35] 보폭 너비 대비 골반 수직 하강비 (체형 무관 런지 심도)
    "lead_knee_angular_rom_extent",       # [36] 직립 각도 대비 굴곡 가동범위 백분율 (Δθ / θ_standing)

    # [그룹 6: sl3 계승 핵심 4대 동적 정규화 및 골반 롤 융합 피처 (4D, 신규 결합)]
    "pelvic_tilt_roll_slope",             # [37] 최하사점 작동 골반 vs 신전 골반 수직 높이차 / 체간 길이 (sl3 1위 피처)
    "lead_leg_length_compression_ratio",  # [38] 기립 대비 최하사점 앞다리 골반-발목 3D 직선거리 압축비
    "trail_to_lead_knee_rom_ratio",       # [39] 신전 다리 가동범위 대비 작동 다리 가동범위 비율
    "lead_knee_ratio_to_stand",           # [40] 기립 상태 무릎 각도 대비 최하사점 굴곡 비율
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
    """Savitzky-Golay 스무딩 필터 적용"""
    if len(arr) < window:
        return arr.copy()
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
    except Exception:
        return arr.copy()


def compute_dtw_distance_1d(seq: np.ndarray, target_curve: np.ndarray) -> float:
    """1차원 시계열 간의 Fast Dynamic Time Warping (DTW) 거리 계산"""
    n, m = len(seq), len(target_curve)
    cost = np.zeros((n + 1, m + 1), dtype=np.float32)
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        diff = (seq[i - 1] - target_curve) ** 2
        for j in range(1, m + 1):
            cost[i, j] = diff[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(math.sqrt(cost[n, m]) / max(n, m))


def separate_legs_by_spatial_distance(
    kps_64f: np.ndarray,
    inflection_idx: int,
    win_start: int,
    win_end: int
) -> Tuple[str, int, int, int, int, int, int, float, float]:
    """c3 공간 거리 다리 분리 엔진 (Spatial Distance Leg Separation)"""
    mid_hip_win = (kps_64f[win_start:win_end, MP_LEFT_HIP] + kps_64f[win_start:win_end, MP_RIGHT_HIP]) / 2.0
    dx_knee_l = np.mean(np.abs(kps_64f[win_start:win_end, MP_LEFT_KNEE, 0] - mid_hip_win[:, 0]))
    dx_knee_r = np.mean(np.abs(kps_64f[win_start:win_end, MP_RIGHT_KNEE, 0] - mid_hip_win[:, 0]))
    dx_ank_l = np.mean(np.abs(kps_64f[win_start:win_end, MP_LEFT_ANKLE, 0] - mid_hip_win[:, 0]))
    dx_ank_r = np.mean(np.abs(kps_64f[win_start:win_end, MP_RIGHT_ANKLE, 0] - mid_hip_win[:, 0]))

    dist_l_body = float(dx_knee_l * 0.6 + dx_ank_l * 0.4)
    dist_r_body = float(dx_knee_r * 0.6 + dx_ank_r * 0.4)

    if dist_l_body < dist_r_body:
        lead_side = 'left'
        lead_hip_idx, lead_knee_idx, lead_ankle_idx = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE
        rear_hip_idx, rear_knee_idx, rear_ankle_idx = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE
    else:
        lead_side = 'right'
        lead_hip_idx, lead_knee_idx, lead_ankle_idx = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE
        rear_hip_idx, rear_knee_idx, rear_ankle_idx = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE

    return (
        lead_side,
        lead_hip_idx, lead_knee_idx, lead_ankle_idx,
        rear_hip_idx, rear_knee_idx, rear_ankle_idx,
        dist_l_body, dist_r_body
    )


def extract_side_lunge_c1_features_v8(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    사이드 런지 64프레임 3차원 포즈 시퀀스로부터 sl1_v4(36D) + sl3 융합(4D) = 총 40D 피처 추출
    """
    assert kps_64f.shape == (64, 33, 3), f"포즈 텐서 규격 오류: {kps_64f.shape}"

    # 1. 체간 스케일
    shoulder_mid = (kps_64f[:, MP_LEFT_SHOULDER] + kps_64f[:, MP_RIGHT_SHOULDER]) / 2.0
    hip_mid = (kps_64f[:, MP_LEFT_HIP] + kps_64f[:, MP_RIGHT_HIP]) / 2.0
    torso_lengths = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    torso_scale = float(np.median(torso_lengths)) if len(torso_lengths) > 0 else 1.0
    if torso_scale < 1e-4:
        torso_scale = 1.0

    # 2. 좌우 무릎 각도 시계열
    v_femur_l = kps_64f[:, MP_LEFT_KNEE] - kps_64f[:, MP_LEFT_HIP]
    v_tibia_l = kps_64f[:, MP_LEFT_ANKLE] - kps_64f[:, MP_LEFT_KNEE]
    v_femur_r = kps_64f[:, MP_RIGHT_KNEE] - kps_64f[:, MP_RIGHT_HIP]
    v_tibia_r = kps_64f[:, MP_RIGHT_ANKLE] - kps_64f[:, MP_RIGHT_KNEE]

    left_angles = np.array([180.0 - vector_angle_3d(v_femur_l[f], v_tibia_l[f]) for f in range(64)])
    right_angles = np.array([180.0 - vector_angle_3d(v_femur_r[f], v_tibia_r[f]) for f in range(64)])

    left_angles_sg = safe_savgol_filter(left_angles)
    right_angles_sg = safe_savgol_filter(right_angles)

    min_combined = np.minimum(left_angles_sg, right_angles_sg)
    inflection_idx = int(np.argmin(min_combined))
    win_start = max(0, inflection_idx - 3)
    win_end = min(64, inflection_idx + 4)

    # 3. 공간 거리 다리 분리
    (
        lead_side,
        lead_hip_idx, lead_knee_idx, lead_ankle_idx,
        rear_hip_idx, rear_knee_idx, rear_ankle_idx,
        dist_l, dist_r
    ) = separate_legs_by_spatial_distance(kps_64f, inflection_idx, win_start, win_end)

    if lead_side == 'left':
        lead_angles_sg = left_angles_sg
        rear_angles_sg = right_angles_sg
    else:
        lead_angles_sg = right_angles_sg
        rear_angles_sg = left_angles_sg

    femur_lengths = np.linalg.norm(kps_64f[:, lead_knee_idx] - kps_64f[:, lead_hip_idx], axis=1)
    tibia_lengths = np.linalg.norm(kps_64f[:, lead_ankle_idx] - kps_64f[:, lead_knee_idx], axis=1)
    L_femur_median = float(np.median(femur_lengths)) if len(femur_lengths) > 0 else 1.0

    # 기립 상태(0~5프레임) 앵커링
    stand_knee_angle = float(np.mean(lead_angles_sg[:6])) if len(lead_angles_sg) >= 6 else 170.0
    stand_lead_hip_ankle_dist = float(np.median(np.linalg.norm(kps_64f[:6, lead_hip_idx] - kps_64f[:6, lead_ankle_idx], axis=1))) + 1e-6

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
    norm_fs = np.linalg.norm(femur_sag)
    norm_ts = np.linalg.norm(tibia_sag)
    if norm_fs > 1e-6 and norm_ts > 1e-6:
        sag_ang = np.degrees(np.arccos(np.clip(np.dot(femur_sag, tibia_sag) / (norm_fs * norm_ts), -1.0, 1.0)))
    else:
        sag_ang = lead_knee_angle_min
    sagittal_knee_90_error = float(abs(sag_ang - 90.0))

    # --- [그룹 2: 하지 기하 및 정강이 정렬 + 수직 불변량 (13D)] ---
    vert_axis = np.array([0.0, -1.0, 0.0])
    horiz_axis = np.array([1.0, 0.0, 0.0])
    lead_tibia_vertical_tilt = float(vector_angle_3d(v_tibia_bot, vert_axis))
    lead_femur_horizontal_tilt = float(vector_angle_3d(v_femur_bot, horiz_axis))

    lead_foot_idx = MP_LEFT_FOOT_INDEX if lead_side == 'left' else MP_RIGHT_FOOT_INDEX
    toe_k = kps_64f[inflection_idx, lead_foot_idx]
    knee_k = kps_64f[inflection_idx, lead_knee_idx]
    lead_knee_toe_overhang = float((knee_k[2] - toe_k[2]) / torso_scale)

    v_foot = kps_64f[inflection_idx, lead_foot_idx] - kps_64f[inflection_idx, lead_ankle_idx]
    ankle_angle = vector_angle_3d(v_tibia_bot, v_foot)
    lead_shank_ankle_coupling = float(lead_tibia_vertical_tilt / (ankle_angle + 1e-5))

    tibia_tilts = [vector_angle_3d(kps_64f[f, lead_ankle_idx] - kps_64f[f, lead_knee_idx], vert_axis) for f in range(inflection_idx + 1)]
    lead_tibia_sagittal_sway = float(np.std(tibia_tilts)) if tibia_tilts else 0.0

    lead_femur_length_ratio = float(L_femur_median / torso_scale)
    L_tibia_median = float(np.median(tibia_lengths)) if len(tibia_lengths) > 0 else 1.0
    lead_tibia_length_ratio = float(L_tibia_median / torso_scale)

    femur_len_std = float(np.std(femur_lengths)) if len(femur_lengths) > 0 else 0.0
    tibia_len_std = float(np.std(tibia_lengths)) if len(tibia_lengths) > 0 else 0.0
    rigid_bone_length_consistency = float(1.0 / (1.0 + (femur_len_std + tibia_len_std) / torso_scale))

    rear_knee_y = kps_64f[inflection_idx, rear_knee_idx, 1]
    ankle_y_mean = (kps_64f[inflection_idx, MP_LEFT_ANKLE, 1] + kps_64f[inflection_idx, MP_RIGHT_ANKLE, 1]) / 2.0
    rear_knee_ground_clearance = float(abs(rear_knee_y - ankle_y_mean) / torso_scale)

    lead_hip_y = kps_64f[inflection_idx, lead_hip_idx, 1]
    lead_knee_y = kps_64f[inflection_idx, lead_knee_idx, 1]
    delta_y_femur = abs(lead_hip_y - lead_knee_y)
    lead_femur_vertical_drop_ratio = float(delta_y_femur / max(1e-4, L_femur_median))

    yaw_indicator = abs(kps_64f[inflection_idx, MP_LEFT_HIP, 2] - kps_64f[inflection_idx, MP_RIGHT_HIP, 2]) / torso_scale
    front_view_weight = math.exp(-2.0 * (yaw_indicator ** 2))
    front_view_femur_drop_gated = float(lead_femur_vertical_drop_ratio * front_view_weight)

    limb_normal = np.cross(v_femur_bot, v_tibia_bot)
    norm_ln = np.linalg.norm(limb_normal)
    if norm_ln > 1e-6:
        limb_normal /= norm_ln
        lead_leg_coplanarity = float(abs(np.dot(limb_normal, pelvis_x)))
    else:
        lead_leg_coplanarity = 0.0

    lead_knee_rom_ratio = float((stand_knee_angle - lead_knee_angle_min) / 90.0)

    # --- [그룹 3: 양 하지 협응 및 보폭 (5D)] ---
    rear_knee_angle_min = float(np.min(rear_angles_sg[win_start:win_end]))
    lead_rear_knee_ratio = float(lead_knee_angle_min / max(1.0, rear_knee_angle_min))

    pelvis_vec = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    pelvis_horiz = np.array([pelvis_vec[0], 0.0, pelvis_vec[2]])
    pelvic_drop_angle = float(vector_angle_3d(pelvis_vec, pelvis_horiz))

    stride_z = abs(kps_64f[inflection_idx, MP_LEFT_ANKLE, 2] - kps_64f[inflection_idx, MP_RIGHT_ANKLE, 2])
    stride_length_normalized = float(stride_z / torso_scale)
    step_x = abs(kps_64f[inflection_idx, MP_LEFT_ANKLE, 0] - kps_64f[inflection_idx, MP_RIGHT_ANKLE, 0])
    step_width_normalized = float(step_x / torso_scale)

    # --- [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)] ---
    accel_traj = np.gradient(vel_traj)
    lead_knee_accel_inflection = float(accel_traj[inflection_idx])

    near_bottom_threshold = lead_knee_angle_min + 5.0
    bottom_phase_dwell_ratio = float(np.mean(lead_angles_sg <= near_bottom_threshold))

    t_eval = np.linspace(0, 1, 64)
    gold_curve = 175.0 - 85.0 * np.exp(-((t_eval - 0.5) ** 2) / 0.03)
    lead_knee_dtw_distance = compute_dtw_distance_1d(lead_angles_sg, gold_curve)

    if vis_64f is not None:
        lead_knee_vis = vis_64f[win_start:win_end, lead_knee_idx]
        lead_knee_visibility_mean = float(np.mean(lead_knee_vis))
    else:
        lead_knee_visibility_mean = 1.0

    hip_vel = np.linalg.norm(np.gradient(kps_64f[:, lead_hip_idx], axis=0), axis=1)
    knee_vel = np.linalg.norm(np.gradient(kps_64f[:, lead_knee_idx], axis=0), axis=1)
    ke_proxy = 0.5 * (hip_vel ** 2 + knee_vel ** 2)
    kinetic_energy_inflection = float(ke_proxy[inflection_idx])

    knee_pos_norm = kps_64f[:, lead_knee_idx] / torso_scale
    vel_3d = np.gradient(knee_pos_norm, axis=0)
    acc_3d = np.gradient(vel_3d, axis=0)
    jerk_3d = np.gradient(acc_3d, axis=0)
    jerk_metric_lead_leg = float(np.mean(np.linalg.norm(jerk_3d, axis=1)))

    # --- [그룹 5: sl1_v4 핵심 6대 불변 생체역학 피처 (6D)] ---
    # 1. 직립 대비 실제 각도 낙차량 (Δθ, deg)
    lead_knee_delta_from_standing = float(stand_knee_angle - lead_knee_angle_min)

    # 2. 펴는다리 - 굽히는다리 각도 차이 (θ_rear - θ_lead, deg)
    lead_rear_knee_angle_difference = float(rear_knee_angle_min - lead_knee_angle_min)

    # 3. 관절면 법선과 카메라 광축(Z축) 사잇각 코사인 (투영 단축 팩터)
    optical_axis_foreshortening_factor = float(abs(limb_normal[2])) if norm_ln > 1e-6 else 0.5

    # 4. 대퇴골 지면 수평 레벨링 지표 (|y_hip - y_knee| / L_femur)
    femur_horizontal_level_ratio = float(delta_y_femur / max(1e-4, L_femur_median))

    # 5. 보폭 너비 대비 골반 수직 하강비
    mid_hip_y_traj = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    pelvis_drop_amount = float(abs(np.max(mid_hip_y_traj) - mid_hip_y_traj[0]))
    actual_stride_width = float(np.linalg.norm(kps_64f[inflection_idx, MP_LEFT_ANKLE] - kps_64f[inflection_idx, MP_RIGHT_ANKLE]))
    pelvic_drop_to_stride_ratio = float(pelvis_drop_amount / max(1e-4, actual_stride_width))

    # 6. 직립 각도 대비 굴곡 가동범위 백분율 (Δθ / θ_standing)
    lead_knee_angular_rom_extent = float(lead_knee_delta_from_standing / max(1.0, stand_knee_angle))

    # --- [그룹 6: sl3 계승 핵심 4대 동적 정규화 및 골반 롤 융합 피처 (4D, 신규 결합)] ---
    # [37] pelvic_tilt_roll_slope: 최하사점 작동 골반 vs 신전 골반 수직 높이차 / 체간길이 (sl3 1위 피처)
    rear_hip_y = kps_64f[inflection_idx, rear_hip_idx, 1]
    pelvic_tilt_roll_slope = float((lead_hip_y - rear_hip_y) / torso_scale)

    # [38] lead_leg_length_compression_ratio: 기립 대비 최하사점 앞다리 골반-발목 3D 직선거리 압축비
    bot_lead_hip_ankle_dist = float(np.linalg.norm(kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_ankle_idx]))
    lead_leg_length_compression_ratio = float(bot_lead_hip_ankle_dist / stand_lead_hip_ankle_dist)

    # [39] trail_to_lead_knee_rom_ratio: 신전 다리 가동범위 대비 작동 다리 가동범위 비율
    rear_knee_rom = float(np.ptp(rear_angles_sg)) + 1e-6
    lead_knee_rom = float(np.ptp(lead_angles_sg)) + 1e-6
    trail_to_lead_knee_rom_ratio = float(rear_knee_rom / lead_knee_rom)

    # [40] lead_knee_ratio_to_stand: 기립 상태 무릎 각도 대비 최하사점 굴곡 비율
    lead_knee_ratio_to_stand = float(lead_knee_angle_min / max(1.0, stand_knee_angle))

    return {
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
        "rear_knee_angle_min": rear_knee_angle_min,
        "pelvic_drop_angle": pelvic_drop_angle,
        "stride_length_normalized": stride_length_normalized,
        "step_width_normalized": step_width_normalized,

        "lead_knee_accel_inflection": lead_knee_accel_inflection,
        "bottom_phase_dwell_ratio": bottom_phase_dwell_ratio,
        "lead_knee_dtw_distance": lead_knee_dtw_distance,
        "lead_knee_visibility_mean": lead_knee_visibility_mean,
        "kinetic_energy_inflection": kinetic_energy_inflection,
        "jerk_metric_lead_leg": jerk_metric_lead_leg,

        "lead_knee_delta_from_standing": lead_knee_delta_from_standing,
        "lead_rear_knee_angle_difference": lead_rear_knee_angle_difference,
        "optical_axis_foreshortening_factor": optical_axis_foreshortening_factor,
        "femur_horizontal_level_ratio": femur_horizontal_level_ratio,
        "pelvic_drop_to_stride_ratio": pelvic_drop_to_stride_ratio,
        "lead_knee_angular_rom_extent": lead_knee_angular_rom_extent,

        "pelvic_tilt_roll_slope": pelvic_tilt_roll_slope,
        "lead_leg_length_compression_ratio": lead_leg_length_compression_ratio,
        "trail_to_lead_knee_rom_ratio": trail_to_lead_knee_rom_ratio,
        "lead_knee_ratio_to_stand": lead_knee_ratio_to_stand,
    }


def extract_side_lunge_c1_feature_vector_v8(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """40D 피처 딕셔너리를 표준 정렬 넘파이 벡터(Shape: (40,))로 변환"""
    feat_dict = extract_side_lunge_c1_features_v8(kps_64f, vis_64f)
    return np.array([feat_dict[k] for k in SIDE_LUNGE_C1_FEATURE_NAMES_V8], dtype=np.float32)

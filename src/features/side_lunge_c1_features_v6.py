# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Module: side_lunge_c1_features_v6.py
Purpose: 사이드 런지 c1(앞다리 무릎 각도 90도) SOTA v6 피처 모듈 (36D)
         - F1 70% 돌파를 위한 6대 신규 불변 생체역학 피처 추가 (30D -> 36D):
           1. lead_knee_delta_from_standing: 직립 베이스라인 대비 상대 굴곡 낙차량 (피험자 유연성 편향 극복)
           2. lead_rear_knee_angle_difference: 펴는 뒷다리와 굽히는 앞다리 사잇각 차이 (편측성 비대칭성 지표)
           3. optical_axis_foreshortening_factor: 관절 굴곡면 법선과 카메라 시선(Z축) 사잇각 정합도 (투영 단축 보정)
           4. femur_horizontal_level_ratio: 최저점 대퇴골 지면 수평성 비율 (|Δy_hip_knee| / L_femur)
           5. pelvic_drop_to_stride_ratio: 보폭 대비 골반 수직 하강 비율 (체형 무관 런지 깊이)
           6. lead_knee_angular_rom_extent: 직립 각도 대비 굴곡 가동범위 백분율 (ROM Ratio)
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

# 36차원 공식 v6 도메인 정예 피처 명칭 목록
SIDE_LUNGE_C1_FEATURE_NAMES_V6: List[str] = [
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

    # [그룹 5: F1 70% 돌파를 위한 6대 신규 불변 생체역학 피처 (6D, 신규 추가)]
    "lead_knee_delta_from_standing",      # [31] 직립 상태 대비 최저점 무릎 굴곡 각도 낙차량 (Δθ, deg)
    "lead_rear_knee_angle_difference",    # [32] 펴는 다리 각도 - 굽히는 다리 각도 차이 (θ_rear - θ_lead, deg)
    "optical_axis_foreshortening_factor", # [33] 하지 3관절면 법선과 카메라 광축 사잇각 코사인 (투영 단축 팩터)
    "femur_horizontal_level_ratio",       # [34] 대퇴골 지면 수평 레벨링 지표 (|y_hip - y_knee| / L_femur)
    "pelvic_drop_to_stride_ratio",        # [35] 보폭 너비 대비 골반 수직 하강비 (체형 무관 런지 심도)
    "lead_knee_angular_rom_extent",       # [36] 직립 각도 대비 굴곡 가동범위 백분율 (Δθ / θ_standing)
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


def detect_side_lunge_working_leg(kps_64f: np.ndarray) -> str:
    """
    [c3 공간 거리 다리 분리 엔진 (Spatial Distance Leg Separation)]
    사이드 런지 체중 이동 시 골반 중심 대비 X축 수평 공간 거리 지표(무릎 0.6 + 발목 0.4) 기반
    작동/작업 다리(Lead Leg) 정밀 판정
    """
    mid_hip_y = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    inflection_idx = int(np.argmax(mid_hip_y))
    eval_f = max(10, min(inflection_idx, 54))

    pelvis_center_x = (kps_64f[eval_f, MP_LEFT_HIP, 0] + kps_64f[eval_f, MP_RIGHT_HIP, 0]) / 2.0

    left_knee_dx = abs(kps_64f[eval_f, MP_LEFT_KNEE, 0] - pelvis_center_x)
    left_ankle_dx = abs(kps_64f[eval_f, MP_LEFT_ANKLE, 0] - pelvis_center_x)
    left_dist_score = 0.6 * left_knee_dx + 0.4 * left_ankle_dx

    right_knee_dx = abs(kps_64f[eval_f, MP_RIGHT_KNEE, 0] - pelvis_center_x)
    right_ankle_dx = abs(kps_64f[eval_f, MP_RIGHT_ANKLE, 0] - pelvis_center_x)
    right_dist_score = 0.6 * right_knee_dx + 0.4 * right_ankle_dx

    if abs(left_dist_score - right_dist_score) > 0.02:
        return 'left' if left_dist_score > right_dist_score else 'right'
    else:
        # 동률 시 무릎 굴곡 각도 비교 (더 많이 굽혀진 쪽이 작업 다리)
        v_l_fem = kps_64f[eval_f, MP_LEFT_HIP] - kps_64f[eval_f, MP_LEFT_KNEE]
        v_l_tib = kps_64f[eval_f, MP_LEFT_ANKLE] - kps_64f[eval_f, MP_LEFT_KNEE]
        l_ang = vector_angle_3d(v_l_fem, v_l_tib)

        v_r_fem = kps_64f[eval_f, MP_RIGHT_HIP] - kps_64f[eval_f, MP_RIGHT_KNEE]
        v_r_tib = kps_64f[eval_f, MP_RIGHT_ANKLE] - kps_64f[eval_f, MP_RIGHT_KNEE]
        r_ang = vector_angle_3d(v_r_fem, v_r_tib)
        return 'left' if l_ang < r_ang else 'right'


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


def extract_side_lunge_c1_features_v6(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    사이드 런지 c1(앞다리 무릎 각도 90도) SOTA v6 피처 추출기 (36D)
    """
    lead_side = detect_side_lunge_working_leg(kps_64f)

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
    L_tibia_median = float(np.median(tibia_lengths)) if tibia_lengths else 1.0
    lead_tibia_length_ratio = float(L_tibia_median / torso_scale)

    femur_len_std = float(np.std(femur_lengths)) if femur_lengths else 0.0
    tibia_len_std = float(np.std(tibia_lengths)) if tibia_lengths else 0.0
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

    stand_knee_angle = float(np.mean(lead_angles_sg[:5])) if len(lead_angles_sg) >= 5 else 170.0
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

    # --- [그룹 5: F1 70% 돌파를 위한 6대 신규 불변 생체역학 피처 (6D)] ---
    # 1. 직립 대비 실제 각도 낙차량 (Δθ, deg) - 피험자 유연성/기저각도 편향 제거
    lead_knee_delta_from_standing = float(stand_knee_angle - lead_knee_angle_min)

    # 2. 펴는다리 - 굽히는다리 각도 차이 (θ_rear - θ_lead, deg) - 편측성 비대칭 굴곡 지표
    lead_rear_knee_angle_difference = float(rear_knee_angle_min - lead_knee_angle_min)

    # 3. 관절면 법선과 카메라 광축(Z축) 사잇각 정합도 (투영 단축 팩터)
    # limb_normal과 Z축([0, 0, 1])의 내적 코사인값
    optical_axis_foreshortening_factor = float(abs(limb_normal[2])) if norm_ln > 1e-6 else 0.5

    # 4. 대퇴골 지면 수평 레벨링 지표 (|y_hip - y_knee| / L_femur)
    # 90도 안착 시 0에 수렴, 덜 앉을수록 큰 값
    femur_horizontal_level_ratio = float(delta_y_femur / max(1e-4, L_femur_median))

    # 5. 보폭 너비 대비 골반 수직 하강비 (체형 무관 런지 심도)
    mid_hip_y_traj = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    pelvis_drop_amount = float(abs(np.max(mid_hip_y_traj) - mid_hip_y_traj[0]))
    actual_stride_width = float(np.linalg.norm(kps_64f[inflection_idx, MP_LEFT_ANKLE] - kps_64f[inflection_idx, MP_RIGHT_ANKLE]))
    pelvic_drop_to_stride_ratio = float(pelvis_drop_amount / max(1e-4, actual_stride_width))

    # 6. 직립 각도 대비 굴곡 가동범위 백분율 (Δθ / θ_standing)
    lead_knee_angular_rom_extent = float(lead_knee_delta_from_standing / max(1.0, stand_knee_angle))

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
    }


def extract_side_lunge_c1_feature_vector_v6(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """36D 피처 딕셔너리를 표준 정렬 넘파이 벡터(Shape: (36,))로 변환"""
    feat_dict = extract_side_lunge_c1_features_v6(kps_64f, vis_64f)
    return np.array([feat_dict[k] for k in SIDE_LUNGE_C1_FEATURE_NAMES_V6], dtype=np.float32)

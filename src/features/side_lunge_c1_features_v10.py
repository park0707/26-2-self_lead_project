# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Module: side_lunge_c1_features_v10.py
Purpose: 사이드 런지 c1(앞다리 무릎 각도 90도) SOTA v10 피처 모듈 (49D)
         - sl1_v8(45D) 완전 계승 (sl1_v4 베이스라인 + sl3 4대 기법 + v8 5대 생체역학)
         - 후방/측방 시점(view5/view1) 관절 가림 및 깊이 왜곡 극복을 위한 4대 후방 불변량 추가 (45D -> 49D):
           46. posterior_pelvis_drop_ratio: 최하사점 골반 중심(Sacrum 프록시) 기립 대비 수직 낙차 / 체간 길이
           47. lead_heel_ground_elevation: 최하사점 앞발 뒤꿈치 지면 이격도 / 발 길이 (보상성 발꿈치 들림 포착)
           48. coronal_trunk_lateral_lean: 최하사점 관상면 상체 측방 기울기 각도 (체간 보상 굴곡)
           49. spine_pelvis_skew_angle: 최하사점 척추-골반 비틀림 사잇각 (체간-골반 해부학적 해리도)
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

# 49차원 공식 v10 도메인 정예 피처 명칭 목록
SIDE_LUNGE_C1_FEATURE_NAMES_V10: List[str] = [
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

    # [그룹 6: sl3 계승 핵심 4대 동적 정규화 및 골반 롤 융합 피처 (4D)]
    "pelvic_tilt_roll_slope",             # [37] 최하사점 작동 골반 vs 신전 골반 수직 높이차 / 체간 길이 (sl3 1위 피처)
    "lead_leg_length_compression_ratio",  # [38] 기립 대비 최하사점 앞다리 골반-발목 3D 직선거리 압축비
    "trail_to_lead_knee_rom_ratio",       # [39] 신전 다리 가동범위 대비 작동 다리 가동범위 비율
    "lead_knee_ratio_to_stand",           # [40] 기립 상태 무릎 각도 대비 최하사점 굴곡 비율

    # [그룹 7: v8 5대 생체역학 고도화 피처 (5D)]
    "femur_slope_at_peak",                # [41] 최하사점 대퇴골 지면 수평 경사각 (도 단위, 90도 심도 물리각)
    "femur_slope_stability",              # [42] 최하사점 전후 5프레임 대퇴골 경사각 안정도 분산
    "lateral_com_shift_ratio",            # [43] 양 발목 너비 대비 신체 질량중심(CoM) 측방 이동 비율 (앞다리 부하율)
    "frontal_q_angle_proxy",              # [44] 관상면 골반-무릎-발목 동적 정렬 사잇각 (외반/내반 보상 검출)
    "pelvic_sagittal_thigh_angle",        # [45] 골반 3D 해부학적 좌표계 투영 대퇴골 고유 시상면 사잇각 (시점 무관)

    # [그룹 8: v9 신규 후방 뷰 가림 극복 및 체간-골반 보상 4대 불변량 피처 (4D, 신규 확장)]
    "posterior_pelvis_drop_ratio",        # [46] 최하사점 골반 중심(Sacrum 프록시) 기립 대비 수직 낙차 / 체간 길이
    "lead_heel_ground_elevation",         # [47] 최하사점 앞발 뒤꿈치 지면 이격도 / 발 길이 (보상성 발꿈치 들림 포착)
    "coronal_trunk_lateral_lean",         # [48] 최하사점 관상면 상체 측방 기울기 각도 (체간 보상 굴곡)
    "spine_pelvis_skew_angle",            # [49] 최하사점 척추-골반 비틀림 사잇각 (체간-골반 해부학적 해리도)
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
    n = len(arr)
    if n < window:
        window = n if n % 2 != 0 else n - 1
    if window < 3:
        return arr.copy()
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly)
    except Exception:
        return arr.copy()


def compute_dtw_distance_1d(seq: np.ndarray, template: np.ndarray) -> float:
    """간이 DTW (Dynamic Time Warping) 거리 계산"""
    n, m = len(seq), len(template)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq[i - 1] - template[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    return float(dtw_matrix[n, m] / (n + m))


def separate_lead_and_trail_legs_c3_engine(
    kps_64f: np.ndarray
) -> Tuple[str, int, int, int, int, int, int]:
    """[c3 공간 거리 다리 분리 엔진 100% 통합]"""
    left_hip_traj = kps_64f[:, MP_LEFT_HIP]
    left_ankle_traj = kps_64f[:, MP_LEFT_ANKLE]
    left_leg_lengths = np.linalg.norm(left_hip_traj - left_ankle_traj, axis=1)
    left_rom = float(np.ptp(left_leg_lengths))

    right_hip_traj = kps_64f[:, MP_RIGHT_HIP]
    right_ankle_traj = kps_64f[:, MP_RIGHT_ANKLE]
    right_leg_lengths = np.linalg.norm(right_hip_traj - right_ankle_traj, axis=1)
    right_rom = float(np.ptp(right_leg_lengths))

    if left_rom >= right_rom:
        lead_side = 'left'
        lead_hip_idx = MP_LEFT_HIP
        lead_knee_idx = MP_LEFT_KNEE
        lead_ankle_idx = MP_LEFT_ANKLE
        rear_hip_idx = MP_RIGHT_HIP
        rear_knee_idx = MP_RIGHT_KNEE
        rear_ankle_idx = MP_RIGHT_ANKLE
    else:
        lead_side = 'right'
        lead_hip_idx = MP_RIGHT_HIP
        lead_knee_idx = MP_RIGHT_KNEE
        lead_ankle_idx = MP_RIGHT_ANKLE
        rear_hip_idx = MP_LEFT_HIP
        rear_knee_idx = MP_LEFT_KNEE
        rear_ankle_idx = MP_LEFT_ANKLE

    return lead_side, lead_hip_idx, lead_knee_idx, lead_ankle_idx, rear_hip_idx, rear_knee_idx, rear_ankle_idx


def extract_side_lunge_c1_features_v10(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """MediaPipe 33관절 3D 64프레임 데이터로부터 49D 공식 v10 피처 딕셔너리 추출"""
    assert kps_64f.shape == (64, 33, 3), f"kps_64f shape must be (64, 33, 3), got {kps_64f.shape}"

    # 1. 작동 다리(lead) / 신전 다리(rear) 공간 분리
    (lead_side, lead_hip_idx, lead_knee_idx, lead_ankle_idx,
     rear_hip_idx, rear_knee_idx, rear_ankle_idx) = separate_lead_and_trail_legs_c3_engine(kps_64f)

    # 2. 신체 고유 스케일 (체간 길이: 양 어깨 중심 - 양 골반 중심)
    mid_shoulder = 0.5 * (kps_64f[:, MP_LEFT_SHOULDER] + kps_64f[:, MP_RIGHT_SHOULDER])
    mid_hip = 0.5 * (kps_64f[:, MP_LEFT_HIP] + kps_64f[:, MP_RIGHT_HIP])
    torso_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    torso_scale = float(np.median(torso_lengths))
    if torso_scale < 1e-4:
        torso_scale = 1.0

    # 3. 양 하지 관절 각도 시계열 계산 (64프레임)
    lead_angles = np.zeros(64, dtype=np.float32)
    rear_angles = np.zeros(64, dtype=np.float32)

    for f in range(64):
        v_thigh = kps_64f[f, lead_hip_idx] - kps_64f[f, lead_knee_idx]
        v_shank = kps_64f[f, lead_ankle_idx] - kps_64f[f, lead_knee_idx]
        lead_angles[f] = vector_angle_3d(v_thigh, v_shank)

        v_rear_thigh = kps_64f[f, rear_hip_idx] - kps_64f[f, rear_knee_idx]
        v_rear_shank = kps_64f[f, rear_ankle_idx] - kps_64f[f, rear_knee_idx]
        rear_angles[f] = vector_angle_3d(v_rear_thigh, v_rear_shank)

    # 스무딩
    lead_angles_sg = safe_savgol_filter(lead_angles, window=7, poly=2)
    rear_angles_sg = safe_savgol_filter(rear_angles, window=7, poly=2)

    # 4. 변곡점 (Inflection point: 앞무릎 최소 굴곡 각도 지점)
    inflection_idx = int(np.argmin(lead_angles_sg))
    inflection_idx = max(5, min(58, inflection_idx))

    # 최저점 윈도우 (+/- 3프레임)
    w_start = max(0, inflection_idx - 3)
    w_end = min(64, inflection_idx + 4)
    lead_window = lead_angles_sg[w_start:w_end]

    # --- [그룹 1: 무릎 굴곡 및 시상면 90도 핵심 (6D)] ---
    lead_knee_angle_min = float(np.min(lead_window))
    lead_knee_angle_mean = float(np.mean(lead_window))
    lead_knee_90_error_abs = float(abs(lead_knee_angle_min - 90.0))
    lead_knee_under_90_ratio = float(np.mean(lead_angles_sg < 90.0))

    lead_vel = np.gradient(lead_angles_sg)
    lead_knee_flex_velocity_max = float(np.max(np.abs(lead_vel[:inflection_idx + 1]))) if inflection_idx > 0 else 0.0

    # ISB 시상면 투영 각도
    lead_hip_f = kps_64f[inflection_idx, lead_hip_idx]
    lead_knee_f = kps_64f[inflection_idx, lead_knee_idx]
    lead_ankle_f = kps_64f[inflection_idx, lead_ankle_idx]

    pelvis_vec = kps_64f[inflection_idx, MP_LEFT_HIP] - kps_64f[inflection_idx, MP_RIGHT_HIP]
    pelvis_norm = np.linalg.norm(pelvis_vec)
    if pelvis_norm > 1e-6:
        n_coronal = pelvis_vec / pelvis_norm
    else:
        n_coronal = np.array([1.0, 0.0, 0.0])

    thigh_3d = lead_knee_f - lead_hip_f
    shank_3d = lead_ankle_f - lead_knee_f
    thigh_sag = thigh_3d - np.dot(thigh_3d, n_coronal) * n_coronal
    shank_sag = shank_3d - np.dot(shank_3d, n_coronal) * n_coronal
    sagittal_knee_angle = vector_angle_3d(-thigh_sag, shank_sag)
    sagittal_knee_90_error = float(abs(sagittal_knee_angle - 90.0))

    # --- [그룹 2: 하지 기하 및 정강이 정렬 + 수직 불변량 (13D)] ---
    tibia_vec = lead_ankle_f - lead_knee_f
    femur_vec = lead_knee_f - lead_hip_f
    vertical_axis = np.array([0.0, 1.0, 0.0])
    horizontal_axis = np.array([1.0, 0.0, 0.0])

    lead_tibia_vertical_tilt = vector_angle_3d(tibia_vec, vertical_axis)
    lead_femur_horizontal_tilt = vector_angle_3d(femur_vec, horizontal_axis)

    foot_idx = MP_LEFT_FOOT_INDEX if lead_side == 'left' else MP_RIGHT_FOOT_INDEX
    toe_pos = kps_64f[inflection_idx, foot_idx]
    lead_knee_toe_overhang = float((lead_knee_f[0] - toe_pos[0]) / torso_scale)

    tibia_angle_from_vert = abs(lead_tibia_vertical_tilt - 180.0)
    foot_vec = toe_pos - lead_ankle_f
    dorsiflex_angle = vector_angle_3d(tibia_vec, foot_vec)
    lead_shank_ankle_coupling = float(dorsiflex_angle / max(1.0, tibia_angle_from_vert))

    descent_tibia_tilts = [
        vector_angle_3d(kps_64f[f, lead_ankle_idx] - kps_64f[f, lead_knee_idx], vertical_axis)
        for f in range(0, inflection_idx + 1)
    ]
    lead_tibia_sagittal_sway = float(np.ptp(descent_tibia_tilts))

    lead_femur_len = float(np.linalg.norm(femur_vec))
    lead_tibia_len = float(np.linalg.norm(tibia_vec))
    lead_femur_length_ratio = float(lead_femur_len / torso_scale)
    lead_tibia_length_ratio = float(lead_tibia_len / torso_scale)

    all_femur_lens = np.linalg.norm(kps_64f[:, lead_knee_idx] - kps_64f[:, lead_hip_idx], axis=1)
    all_tibia_lens = np.linalg.norm(kps_64f[:, lead_ankle_idx] - kps_64f[:, lead_knee_idx], axis=1)
    rigid_bone_length_consistency = float((np.std(all_femur_lens) + np.std(all_tibia_lens)) / (2.0 * torso_scale))

    rear_knee_f = kps_64f[inflection_idx, rear_knee_idx]
    rear_knee_ground_clearance = float((lead_ankle_f[1] - rear_knee_f[1]) / torso_scale)

    # v4 핵심 수직 낙차비
    lead_hip_y = lead_hip_f[1]
    lead_knee_y = lead_knee_f[1]
    lead_femur_vertical_drop_ratio = float(abs(lead_hip_y - lead_knee_y) / max(1e-4, lead_femur_len))

    pelvis_yaw_cos = abs(np.dot(n_coronal, np.array([0.0, 0.0, 1.0])))
    front_view_femur_drop_gated = float(lead_femur_vertical_drop_ratio * (1.0 - pelvis_yaw_cos))

    n_leg_plane = np.cross(femur_vec, tibia_vec)
    norm_leg = np.linalg.norm(n_leg_plane)
    if norm_leg > 1e-6:
        n_leg_plane = n_leg_plane / norm_leg
        lead_leg_coplanarity = float(abs(np.dot(n_leg_plane, n_coronal)))
    else:
        lead_leg_coplanarity = 0.0

    stand_lead_angle = float(lead_angles_sg[0])
    lead_knee_rom_ratio = float((stand_lead_angle - lead_knee_angle_min) / max(1.0, stand_lead_angle))

    # --- [그룹 3: 양 하지 협응 및 보폭 (5D)] ---
    rear_knee_angle_min = float(np.min(rear_angles_sg))
    lead_rear_knee_ratio = float(lead_knee_angle_min / max(1.0, rear_knee_angle_min))

    pelvis_line_y = kps_64f[inflection_idx, MP_LEFT_HIP, 1] - kps_64f[inflection_idx, MP_RIGHT_HIP, 1]
    pelvis_line_x = kps_64f[inflection_idx, MP_LEFT_HIP, 0] - kps_64f[inflection_idx, MP_RIGHT_HIP, 0]
    pelvic_drop_angle = float(np.degrees(np.arctan2(abs(pelvis_line_y), abs(pelvis_line_x) + 1e-6)))

    ankle_dist_3d = kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, rear_ankle_idx]
    stride_length_normalized = float(abs(ankle_dist_3d[2]) / torso_scale)
    step_width_normalized = float(abs(ankle_dist_3d[0]) / torso_scale)

    # --- [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)] ---
    lead_acc = np.gradient(lead_vel)
    lead_knee_accel_inflection = float(lead_acc[inflection_idx])

    dwell_mask = np.abs(lead_angles_sg - lead_knee_angle_min) <= 5.0
    bottom_phase_dwell_ratio = float(np.mean(dwell_mask))

    t_norm = np.linspace(0, np.pi, 64)
    gold_template = 170.0 - 80.0 * np.sin(t_norm)
    lead_knee_dtw_distance = compute_dtw_distance_1d(lead_angles_sg, gold_template)

    if vis_64f is not None:
        lead_knee_visibility_mean = float(np.mean(vis_64f[w_start:w_end, lead_knee_idx]))
    else:
        lead_knee_visibility_mean = 1.0

    knee_v_sq = lead_vel[inflection_idx] ** 2
    hip_v_sq = (np.gradient(kps_64f[:, lead_hip_idx, 1])[inflection_idx]) ** 2
    kinetic_energy_inflection = float(0.5 * (knee_v_sq + hip_v_sq))

    lead_jerk = np.gradient(lead_acc)
    jerk_metric_lead_leg = float(np.sqrt(np.mean(lead_jerk ** 2)))

    # --- [그룹 5: sl1_v4 핵심 6대 불변 생체역학 피처 (6D)] ---
    stand_knee_angle = float(np.mean(lead_angles_sg[:5]))
    lead_knee_delta_from_standing = float(stand_knee_angle - lead_knee_angle_min)

    rear_knee_at_inflection = float(rear_angles_sg[inflection_idx])
    lead_rear_knee_angle_difference = float(rear_knee_at_inflection - lead_knee_angle_min)

    optical_axis = np.array([0.0, 0.0, 1.0])
    optical_axis_foreshortening_factor = float(abs(np.dot(n_leg_plane, optical_axis)))

    femur_horizontal_level_ratio = float(abs(lead_hip_y - lead_knee_y) / max(1e-4, lead_femur_len))

    bot_pelvis_y = min(lead_hip_y, kps_64f[inflection_idx, rear_hip_idx, 1])
    stand_pelvis_y = float(np.mean(0.5 * (kps_64f[:5, lead_hip_idx, 1] + kps_64f[:5, rear_hip_idx, 1])))
    pelvic_vertical_drop = abs(bot_pelvis_y - stand_pelvis_y)
    actual_stride_width = np.linalg.norm(kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, rear_ankle_idx])
    pelvic_drop_to_stride_ratio = float(pelvic_vertical_drop / max(1e-4, actual_stride_width))

    lead_knee_angular_rom_extent = float(lead_knee_delta_from_standing / max(1.0, stand_knee_angle))

    # --- [그룹 6: sl3 계승 핵심 4대 동적 정규화 및 골반 롤 융합 피처 (4D)] ---
    rear_hip_y = kps_64f[inflection_idx, rear_hip_idx, 1]
    pelvic_tilt_roll_slope = float((lead_hip_y - rear_hip_y) / torso_scale)

    stand_lead_hip_ankle_dist = float(np.mean([
        np.linalg.norm(kps_64f[i, lead_hip_idx] - kps_64f[i, lead_ankle_idx])
        for i in range(5)
    ])) + 1e-6
    bot_lead_hip_ankle_dist = float(np.linalg.norm(kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_ankle_idx]))
    lead_leg_length_compression_ratio = float(bot_lead_hip_ankle_dist / stand_lead_hip_ankle_dist)

    rear_knee_rom = float(np.ptp(rear_angles_sg)) + 1e-6
    lead_knee_rom = float(np.ptp(lead_angles_sg)) + 1e-6
    trail_to_lead_knee_rom_ratio = float(rear_knee_rom / lead_knee_rom)

    lead_knee_ratio_to_stand = float(lead_knee_angle_min / max(1.0, stand_knee_angle))

    # --- [그룹 7: v8 5대 생체역학 고도화 피처 (5D)] ---
    vertical_femur_drop = abs(lead_hip_y - lead_knee_y)
    femur_slope_at_peak = float(np.degrees(np.arcsin(np.clip(vertical_femur_drop / max(1e-4, lead_femur_len), 0.0, 1.0))))

    start_w5 = max(0, inflection_idx - 4)
    end_w5 = min(64, inflection_idx + 5)
    w_femur_vecs = kps_64f[start_w5:end_w5, lead_knee_idx] - kps_64f[start_w5:end_w5, lead_hip_idx]
    w_femur_lens = np.linalg.norm(w_femur_vecs, axis=-1) + 1e-6
    w_vert = np.abs(w_femur_vecs[:, 1])
    w_slopes = np.degrees(np.arcsin(np.clip(w_vert / w_femur_lens, 0.0, 1.0)))
    femur_slope_stability = float(np.var(w_slopes))

    mid_hip_inflect = 0.5 * (kps_64f[inflection_idx, lead_hip_idx] + kps_64f[inflection_idx, rear_hip_idx])
    com_x_approx = 0.6 * mid_hip_inflect[0] + 0.2 * lead_knee_f[0] + 0.2 * kps_64f[inflection_idx, rear_ankle_idx, 0]
    lead_ankle_x = lead_ankle_f[0]
    rear_ankle_x = kps_64f[inflection_idx, rear_ankle_idx, 0]
    bos_width = abs(lead_ankle_x - rear_ankle_x) + 1e-6
    lateral_com_shift_ratio = float(abs(com_x_approx - rear_ankle_x) / bos_width)

    v_thigh_frontal = (lead_knee_f - lead_hip_f)[:2]
    v_shank_frontal = (lead_ankle_f - lead_knee_f)[:2]
    n_thigh = np.linalg.norm(v_thigh_frontal) + 1e-6
    n_shank = np.linalg.norm(v_shank_frontal) + 1e-6
    cos_align = np.clip(np.dot(v_thigh_frontal / n_thigh, -v_shank_frontal / n_shank), -1.0, 1.0)
    frontal_q_angle_proxy = float(np.degrees(np.arccos(cos_align)))

    pelvis_x = kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, rear_hip_idx]
    pelvis_x = pelvis_x / (np.linalg.norm(pelvis_x) + 1e-6)
    mid_shoulder_inflect = 0.5 * (kps_64f[inflection_idx, MP_LEFT_SHOULDER] + kps_64f[inflection_idx, MP_RIGHT_SHOULDER])
    spine_vec = mid_shoulder_inflect - mid_hip_inflect
    spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)
    pelvis_z = np.cross(pelvis_x, spine_vec)
    pelvis_z = pelvis_z / (np.linalg.norm(pelvis_z) + 1e-6)
    pelvis_y = np.cross(pelvis_z, pelvis_x)
    thigh_sagittal_y = np.dot(femur_vec, pelvis_y)
    thigh_sagittal_z = np.dot(femur_vec, pelvis_z)
    pelvic_sagittal_thigh_angle = float(np.degrees(np.arctan2(thigh_sagittal_z, thigh_sagittal_y)))

    # --- [그룹 8: v9 신규 후방 뷰 가림 극복 및 체간-골반 보상 4대 불변량 피처 (4D, 신규 확장)] ---
    # [46] posterior_pelvis_drop_ratio: 최하사점 골반 중심(Sacrum 프록시) 기립 대비 수직 낙차 / 체간 길이
    sacrum_traj = 0.5 * (kps_64f[:, MP_LEFT_HIP] + kps_64f[:, MP_RIGHT_HIP])
    stand_sacrum_y = float(np.mean(sacrum_traj[:5, 1]))
    inflect_sacrum_y = float(sacrum_traj[inflection_idx, 1])
    posterior_pelvis_drop_ratio = float(abs(inflect_sacrum_y - stand_sacrum_y) / torso_scale)

    # [47] lead_heel_ground_elevation: 최하사점 앞발 뒤꿈치 지면 이격도 / 발 길이 (보상성 발꿈치 들림 포착)
    lead_heel_idx = MP_LEFT_HEEL if lead_side == 'left' else MP_RIGHT_HEEL
    rear_heel_idx = MP_RIGHT_HEEL if lead_side == 'left' else MP_LEFT_HEEL
    lead_foot_idx = MP_LEFT_FOOT_INDEX if lead_side == 'left' else MP_RIGHT_FOOT_INDEX
    foot_len = float(np.linalg.norm(kps_64f[inflection_idx, lead_foot_idx] - kps_64f[inflection_idx, lead_heel_idx])) + 1e-6
    ground_y = max(float(kps_64f[inflection_idx, rear_heel_idx, 1]), float(kps_64f[inflection_idx, lead_foot_idx, 1]))
    lead_heel_elev = max(0.0, ground_y - float(kps_64f[inflection_idx, lead_heel_idx, 1]))
    lead_heel_ground_elevation = float(lead_heel_elev / foot_len)

    # [48] coronal_trunk_lateral_lean: 최하사점 관상면 상체 측방 기울기 각도 (체간 보상 굴곡)
    trunk_dx = mid_shoulder_inflect[0] - sacrum_traj[inflection_idx, 0]
    trunk_dy = -(mid_shoulder_inflect[1] - sacrum_traj[inflection_idx, 1])
    coronal_trunk_lateral_lean = float(np.degrees(np.arctan2(abs(trunk_dx), abs(trunk_dy) + 1e-6)))

    # [49] spine_pelvis_skew_angle: 최하사점 척추-골반 비틀림 사잇각 (체간-골반 해부학적 해리도)
    pelvis_line = kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, rear_hip_idx]
    trunk_vec_3d = mid_shoulder_inflect - sacrum_traj[inflection_idx]
    n_pel = np.linalg.norm(pelvis_line) + 1e-6
    n_trk = np.linalg.norm(trunk_vec_3d) + 1e-6
    dot_pt = np.clip(np.dot(pelvis_line / n_pel, trunk_vec_3d / n_trk), -1.0, 1.0)
    spine_pelvis_skew_angle = float(abs(90.0 - np.degrees(np.arccos(dot_pt))))

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

        "femur_slope_at_peak": femur_slope_at_peak,
        "femur_slope_stability": femur_slope_stability,
        "lateral_com_shift_ratio": lateral_com_shift_ratio,
        "frontal_q_angle_proxy": frontal_q_angle_proxy,
        "pelvic_sagittal_thigh_angle": pelvic_sagittal_thigh_angle,

        "posterior_pelvis_drop_ratio": posterior_pelvis_drop_ratio,
        "lead_heel_ground_elevation": lead_heel_ground_elevation,
        "coronal_trunk_lateral_lean": coronal_trunk_lateral_lean,
        "spine_pelvis_skew_angle": spine_pelvis_skew_angle,
    }


def extract_side_lunge_c1_feature_vector_v10(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """49D 피처 딕셔너리를 표준 정렬 넘파이 벡터(Shape: (49,))로 변환"""
    feat_dict = extract_side_lunge_c1_features_v10(kps_64f, vis_64f)
    return np.array([feat_dict[k] for k in SIDE_LUNGE_C1_FEATURE_NAMES_V10], dtype=np.float32)

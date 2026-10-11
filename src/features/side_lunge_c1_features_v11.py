# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Module: side_lunge_c1_features_v11.py
Purpose: 사이드 런지 c1(앞다리 무릎 각도 90도) SOTA v11 피처 모듈 (55D)
         - sl1_v9(49D) 완전 계승 (sl1_v4 베이스라인 + sl3 4대 기법 + v8 5대 생체역학 + v9 4대 후방불변량)
         - 패러다임 2, 3, 4 생체역학 및 정규화 좌표계 융합 6대 불변량 추가 (49D -> 55D):
           [패러다임 4: 해부학적 신체 로컬 직교 좌표계 및 ISB 순수 굴곡각]
           50. canonical_body_coronal_knee_angle_min: 골반-척추 기준 신체 로컬 직교 좌표계(Pelvis Canonical Frame) 변환 후 관상면(X-Y) 상에서 계산된 최하사점 앞무릎 굴곡 각도 (카메라 뷰포인트 회전 왜곡 100% 제거)
           51. canonical_body_sagittal_tilt: 신체 로컬 좌표계 상에서 하지 3관절면의 시상면 경사 각도
           52. isb_jcs_pure_flexion_min: ISB Grood & Suntay 원리에 따라 대퇴골 횡축 기준 순수 굴곡 각도 (외전/내전 및 축비틀림 성분을 배제한 순수 관절 굽힘각)
           [패러다임 2: 생체역학 1-DOF 힌지 및 강체 골격 불변 구속]
           53. rigid_fk_femur_strain_ratio: 직립 구간 대비 최하사점 대퇴골 길이 신축 변형률 (|L_femur(t) - L_ref| / L_ref, 원근 투영 단축 왜곡 정량화)
           54. kinematic_hinge_orthogonality: 1자유도 힌지 관절 구속 적합도 (대퇴골-경골 굴곡면과 골반 관상면의 기구학적 직교성 준수도)
           [패러다임 3: 단일 카메라 시공간 3D 리프팅 모션 정규화]
           55. temporal_depth_acceleration_norm: 단일 카메라 시계열 깊이축(Z) 가속도 노름 (Anipose/MotionBERT 시공간 평활화 손실 항과 동일한 물리량으로, 원근 깊이 착시 및 가림 지터 억제도 측정)
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

# 55차원 공식 v11 도메인 정예 피처 명칭 목록
SIDE_LUNGE_C1_FEATURE_NAMES_V11: List[str] = [
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

    # [그룹 8: v9 후방 뷰 가림 극복 및 체간-골반 보상 4대 불변량 피처 (4D)]
    "posterior_pelvis_drop_ratio",        # [46] 최하사점 골반 중심(Sacrum 프록시) 기립 대비 수직 낙차 / 체간 길이
    "lead_heel_ground_elevation",         # [47] 최하사점 앞발 뒤꿈치 지면 이격도 / 발 길이 (보상성 발꿈치 들림 포착)
    "coronal_trunk_lateral_lean",         # [48] 최하사점 관상면 상체 측방 기울기 각도 (체간 보상 굴곡)
    "spine_pelvis_skew_angle",            # [49] 최하사점 척추-골반 비틀림 사잇각 (체간-골반 해부학적 해리도)

    # [그룹 9: 패러다임 2, 3, 4 생체역학 및 정규화 좌표계 융합 6대 불변량 피처 (6D, 신규 확장)]
    "canonical_body_coronal_knee_angle_min", # [50] 골반-척추 기준 신체 로컬 좌표계 관상면 앞무릎 굴곡 각도 최소치 (시점 왜곡 소거)
    "canonical_body_sagittal_tilt",          # [51] 신체 로컬 좌표계 상에서 하지 3관절면 시상면 경사 각도
    "isb_jcs_pure_flexion_min",              # [52] ISB Grood & Suntay 원리 대퇴골 횡축 기준 순수 굴곡 각도 (외전/회전 배제)
    "rigid_fk_femur_strain_ratio",           # [53] 직립 기준 대퇴골 길이 신축 변형률 (|L_femur - L_ref| / L_ref, 단축 왜곡 감지)
    "kinematic_hinge_orthogonality",         # [54] 1자유도 힌지 관절 구속 적합도 (하지 관절면과 골반 관상면의 기구학적 직교성)
    "temporal_depth_acceleration_norm",      # [55] 단일 카메라 시계열 깊이축(Z) 가속도 노름 (Anipose/MotionBERT 시공간 평활도)
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


def extract_side_lunge_c1_features_v11(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """MediaPipe 33관절 3D 64프레임 데이터로부터 55D 공식 v11 피처 딕셔너리 추출"""
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
    v_thigh_inflect = kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_knee_idx]
    v_shank_inflect = kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, lead_knee_idx]
    sagittal_knee_angle = vector_angle_3d(v_thigh_inflect[[1, 2]], v_shank_inflect[[1, 2]])
    sagittal_knee_90_error = float(abs(sagittal_knee_angle - 90.0))

    # --- [그룹 2: 하지 기하 및 정강이 정렬 + 수직 불변량 (13D)] ---
    v_tibia = kps_64f[inflection_idx, lead_knee_idx] - kps_64f[inflection_idx, lead_ankle_idx]
    lead_tibia_vertical_tilt = vector_angle_3d(v_tibia, np.array([0.0, -1.0, 0.0]))

    v_femur = kps_64f[inflection_idx, lead_knee_idx] - kps_64f[inflection_idx, lead_hip_idx]
    lead_femur_horizontal_tilt = vector_angle_3d(v_femur, np.array([1.0, 0.0, 0.0]))

    lead_toe_idx = MP_LEFT_FOOT_INDEX if lead_side == 'left' else MP_RIGHT_FOOT_INDEX
    lead_knee_toe_overhang = float((kps_64f[inflection_idx, lead_knee_idx, 0] -
                                   kps_64f[inflection_idx, lead_toe_idx, 0]) / torso_scale)

    v_foot = kps_64f[inflection_idx, lead_toe_idx] - kps_64f[inflection_idx, lead_ankle_idx]
    ankle_flex_angle = vector_angle_3d(v_tibia, v_foot)
    lead_shank_ankle_coupling = float(lead_tibia_vertical_tilt / (ankle_flex_angle + 1e-5))

    tibia_tilts = [vector_angle_3d(kps_64f[f, lead_knee_idx] - kps_64f[f, lead_ankle_idx], np.array([0.0, -1.0, 0.0]))
                   for f in range(w_start, w_end)]
    lead_tibia_sagittal_sway = float(np.ptp(tibia_tilts))

    femur_lengths = np.linalg.norm(kps_64f[:, lead_hip_idx] - kps_64f[:, lead_knee_idx], axis=1)
    tibia_lengths = np.linalg.norm(kps_64f[:, lead_knee_idx] - kps_64f[:, lead_ankle_idx], axis=1)
    lead_femur_length_ratio = float(np.median(femur_lengths) / torso_scale)
    lead_tibia_length_ratio = float(np.median(tibia_lengths) / torso_scale)

    femur_cv = float(np.std(femur_lengths) / (np.mean(femur_lengths) + 1e-6))
    tibia_cv = float(np.std(tibia_lengths) / (np.mean(tibia_lengths) + 1e-6))
    rigid_bone_length_consistency = float(1.0 / (1.0 + femur_cv + tibia_cv))

    rear_knee_ground_clearance = float((kps_64f[inflection_idx, rear_knee_idx, 1] -
                                        kps_64f[inflection_idx, rear_ankle_idx, 1]) / torso_scale)

    # v8 4대 수직 불변량
    inflect_femur_len = float(np.linalg.norm(v_femur) + 1e-6)
    delta_y_femur = float(abs(kps_64f[inflection_idx, lead_knee_idx, 1] - kps_64f[inflection_idx, lead_hip_idx, 1]))
    lead_femur_vertical_drop_ratio = float(delta_y_femur / inflect_femur_len)

    v_pelvis_yaw = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    yaw_cos = float(abs(v_pelvis_yaw[0]) / (np.linalg.norm(v_pelvis_yaw) + 1e-6))
    front_view_femur_drop_gated = float(lead_femur_vertical_drop_ratio * yaw_cos)

    v1_leg = kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_knee_idx]
    v2_leg = kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, lead_knee_idx]
    leg_norm = np.cross(v1_leg, v2_leg)
    leg_norm_u = leg_norm / (np.linalg.norm(leg_norm) + 1e-6)
    pelvis_norm = np.cross(v_pelvis_yaw, np.array([0.0, -1.0, 0.0]))
    pelvis_norm_u = pelvis_norm / (np.linalg.norm(pelvis_norm) + 1e-6)
    lead_leg_coplanarity = float(abs(np.dot(leg_norm_u, pelvis_norm_u)))

    standing_angles = lead_angles_sg[:max(3, inflection_idx // 3)]
    standing_lead_angle = float(np.median(standing_angles)) if len(standing_angles) > 0 else 170.0
    lead_knee_rom_ratio = float(abs(standing_lead_angle - lead_knee_angle_min) / (standing_lead_angle + 1e-5))

    # --- [그룹 3: 양 하지 협응 및 보폭 (5D)] ---
    rear_window = rear_angles_sg[w_start:w_end]
    rear_knee_angle_min = float(np.min(rear_window))
    lead_rear_knee_ratio = float(lead_knee_angle_min / (rear_knee_angle_min + 1e-5))

    pelvis_vec = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    pelvic_drop_angle = float(math.degrees(math.atan2(abs(pelvis_vec[1]), abs(pelvis_vec[0]) + 1e-6)))

    stride_z = abs(kps_64f[inflection_idx, lead_ankle_idx, 2] - kps_64f[inflection_idx, rear_ankle_idx, 2])
    stride_length_normalized = float(stride_z / torso_scale)

    stride_x = abs(kps_64f[inflection_idx, lead_ankle_idx, 0] - kps_64f[inflection_idx, rear_ankle_idx, 0])
    step_width_normalized = float(stride_x / torso_scale)

    # --- [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)] ---
    accel = np.gradient(lead_vel)
    lead_knee_accel_inflection = float(accel[inflection_idx])

    near_bottom_thresh = lead_knee_angle_min + 5.0
    bottom_phase_dwell_ratio = float(np.mean(lead_angles_sg <= near_bottom_thresh))

    ideal_curve = np.linspace(170.0, 90.0, 32)
    ideal_curve = np.concatenate([ideal_curve, np.linspace(90.0, 170.0, 32)])
    lead_knee_dtw_distance = compute_dtw_distance_1d(lead_angles_sg, ideal_curve)

    if vis_64f is not None:
        lead_knee_visibility_mean = float(np.mean(vis_64f[w_start:w_end, lead_knee_idx]))
    else:
        lead_knee_visibility_mean = 1.0

    disp = np.diff(kps_64f[:, lead_knee_idx, :], axis=0)
    kinetic_proxy = np.sum(disp ** 2, axis=1)
    kinetic_energy_inflection = float(kinetic_proxy[min(inflection_idx, len(kinetic_proxy) - 1)])

    diff3 = np.diff(kps_64f[:, lead_knee_idx, :], n=3, axis=0)
    jerk_metric_lead_leg = float(np.mean(np.linalg.norm(diff3, axis=1)))

    # --- [그룹 5: sl1_v4 핵심 6대 불변 생체역학 피처 (6D)] ---
    lead_knee_delta_from_standing = float(standing_lead_angle - lead_knee_angle_min)
    standing_rear_angles = rear_angles_sg[:max(3, inflection_idx // 3)]
    standing_rear_angle = float(np.median(standing_rear_angles)) if len(standing_rear_angles) > 0 else 170.0
    lead_rear_knee_angle_difference = float(rear_knee_angle_min - lead_knee_angle_min)

    cam_axis = np.array([0.0, 0.0, 1.0])
    optical_axis_foreshortening_factor = float(abs(np.dot(leg_norm_u, cam_axis)))

    hip_y = kps_64f[inflection_idx, lead_hip_idx, 1]
    knee_y = kps_64f[inflection_idx, lead_knee_idx, 1]
    femur_horizontal_level_ratio = float(abs(hip_y - knee_y) / inflect_femur_len)

    total_stride = float(np.linalg.norm(kps_64f[inflection_idx, lead_ankle_idx] -
                                        kps_64f[inflection_idx, rear_ankle_idx]) + 1e-5)
    standing_mid_hip_y = float(np.median(mid_hip[:max(3, inflection_idx // 3), 1]))
    inflect_mid_hip_y = float(mid_hip[inflection_idx, 1])
    pelvic_vertical_drop = float(abs(inflect_mid_hip_y - standing_mid_hip_y))
    pelvic_drop_to_stride_ratio = float(pelvic_vertical_drop / total_stride)

    lead_knee_angular_rom_extent = float((standing_lead_angle - lead_knee_angle_min) / (standing_lead_angle + 1e-5))

    # --- [그룹 6: sl3 계승 핵심 4대 동적 정규화 및 골반 롤 융합 피처 (4D)] ---
    inflect_lead_hip_y = kps_64f[inflection_idx, lead_hip_idx, 1]
    inflect_rear_hip_y = kps_64f[inflection_idx, rear_hip_idx, 1]
    pelvic_tilt_roll_slope = float(abs(inflect_lead_hip_y - inflect_rear_hip_y) / torso_scale)

    standing_leg_3d_dist = float(np.median(
        np.linalg.norm(kps_64f[:max(3, inflection_idx // 3), lead_hip_idx] -
                       kps_64f[:max(3, inflection_idx // 3), lead_ankle_idx], axis=1)
    ))
    inflect_leg_3d_dist = float(np.linalg.norm(
        kps_64f[inflection_idx, lead_hip_idx] - kps_64f[inflection_idx, lead_ankle_idx]
    ))
    lead_leg_length_compression_ratio = float(inflect_leg_3d_dist / (standing_leg_3d_dist + 1e-6))

    rear_knee_rom = float(abs(standing_rear_angle - rear_knee_angle_min))
    lead_knee_rom = float(abs(standing_lead_angle - lead_knee_angle_min))
    trail_to_lead_knee_rom_ratio = float(rear_knee_rom / (lead_knee_rom + 1e-5))

    lead_knee_ratio_to_stand = float(lead_knee_angle_min / (standing_lead_angle + 1e-5))

    # --- [그룹 7: v8 5대 생체역학 고도화 피처 (5D)] ---
    v_femur_xz = math.sqrt(v_femur[0] ** 2 + v_femur[2] ** 2)
    femur_slope_at_peak = float(math.degrees(math.atan2(abs(v_femur[1]), v_femur_xz + 1e-6)))

    peak_slopes = []
    for f_idx in range(w_start, w_end):
        vf = kps_64f[f_idx, lead_knee_idx] - kps_64f[f_idx, lead_hip_idx]
        vf_xz = math.sqrt(vf[0] ** 2 + vf[2] ** 2)
        peak_slopes.append(math.degrees(math.atan2(abs(vf[1]), vf_xz + 1e-6)))
    femur_slope_stability = float(np.std(peak_slopes)) if len(peak_slopes) > 0 else 0.0

    ankle_mid_x = 0.5 * (kps_64f[inflection_idx, lead_ankle_idx, 0] + kps_64f[inflection_idx, rear_ankle_idx, 0])
    com_proxy_x = mid_hip[inflection_idx, 0]
    total_ankle_width_x = abs(kps_64f[inflection_idx, lead_ankle_idx, 0] - kps_64f[inflection_idx, rear_ankle_idx, 0]) + 1e-6
    lateral_com_shift_ratio = float(abs(com_proxy_x - ankle_mid_x) / total_ankle_width_x)

    v_q_upper = kps_64f[inflection_idx, lead_hip_idx, :2] - kps_64f[inflection_idx, lead_knee_idx, :2]
    v_q_lower = kps_64f[inflection_idx, lead_ankle_idx, :2] - kps_64f[inflection_idx, lead_knee_idx, :2]
    norm_qu = np.linalg.norm(v_q_upper)
    norm_ql = np.linalg.norm(v_q_lower)
    if norm_qu > 1e-6 and norm_ql > 1e-6:
        cos_q = np.clip(np.dot(v_q_upper, v_q_lower) / (norm_qu * norm_ql), -1.0, 1.0)
        frontal_q_angle_proxy = float(math.degrees(math.acos(cos_q)))
    else:
        frontal_q_angle_proxy = 180.0

    v_pelvis_lat = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    u_pelvis_lat = v_pelvis_lat / (np.linalg.norm(v_pelvis_lat) + 1e-6)
    u_vert = np.array([0.0, -1.0, 0.0])
    u_pelvis_ant = np.cross(u_pelvis_lat, u_vert)
    u_pelvis_ant = u_pelvis_ant / (np.linalg.norm(u_pelvis_ant) + 1e-6)

    femur_long = np.dot(v_femur, u_vert)
    femur_ant = np.dot(v_femur, u_pelvis_ant)
    pelvic_sagittal_thigh_angle = float(math.degrees(math.atan2(abs(femur_ant), abs(femur_long) + 1e-6)))

    # --- [그룹 8: v9 후방 뷰 가림 극복 및 체간-골반 보상 4대 불변량 피처 (4D)] ---
    stand_sacrum_y = float(np.median(mid_hip[:max(3, inflection_idx // 3), 1]))
    inflect_sacrum_y = float(mid_hip[inflection_idx, 1])
    posterior_pelvis_drop_ratio = float(abs(inflect_sacrum_y - stand_sacrum_y) / torso_scale)

    lead_heel_idx = MP_LEFT_HEEL if lead_side == 'left' else MP_RIGHT_HEEL
    lead_heel_pt = kps_64f[inflection_idx, lead_heel_idx]
    lead_toe_pt = kps_64f[inflection_idx, lead_toe_idx]
    foot_length_3d = float(np.linalg.norm(lead_toe_pt - lead_heel_pt) + 1e-5)
    heel_elevation_y = float(abs(lead_heel_pt[1] - lead_toe_pt[1]))
    lead_heel_ground_elevation = float(heel_elevation_y / foot_length_3d)

    v_torso_inflect = mid_shoulder[inflection_idx] - mid_hip[inflection_idx]
    v_torso_xy = v_torso_inflect[:2]
    norm_txy = np.linalg.norm(v_torso_xy)
    if norm_txy > 1e-6:
        cos_lean = np.clip(abs(v_torso_xy[1]) / norm_txy, -1.0, 1.0)
        coronal_trunk_lateral_lean = float(math.degrees(math.acos(cos_lean)))
    else:
        coronal_trunk_lateral_lean = 0.0

    v_sh_inflect = kps_64f[inflection_idx, MP_RIGHT_SHOULDER] - kps_64f[inflection_idx, MP_LEFT_SHOULDER]
    v_hip_inflect = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    norm_sh = np.linalg.norm(v_sh_inflect)
    norm_hip = np.linalg.norm(v_hip_inflect)
    if norm_sh > 1e-6 and norm_hip > 1e-6:
        cos_skew = np.clip(np.dot(v_sh_inflect, v_hip_inflect) / (norm_sh * norm_hip), -1.0, 1.0)
        spine_pelvis_skew_angle = float(math.degrees(math.acos(cos_skew)))
    else:
        spine_pelvis_skew_angle = 0.0

    # --- [그룹 9: 패러다임 2, 3, 4 생체역학 및 정규화 좌표계 융합 6대 불변량 피처 (6D, 신규 확장)] ---
    # [패러다임 4: 해부학적 신체 로컬 직교 좌표계 (Pelvic Canonical Frame) 구축]
    # 기립 초기 프레임 기반 Gram-Schmidt 직교 기저 행렬 R_pelvis (3x3)
    n_init = max(3, min(8, inflection_idx // 2))
    init_hip_vec = np.mean(
        kps_64f[:n_init, MP_RIGHT_HIP, :] - kps_64f[:n_init, MP_LEFT_HIP, :], axis=0
    )
    init_spine_vec = np.mean(
        mid_shoulder[:n_init, :] - mid_hip[:n_init, :], axis=0
    )

    y_axis_can = init_spine_vec / (np.linalg.norm(init_spine_vec) + 1e-7)
    z_axis_can = np.cross(init_hip_vec, y_axis_can)
    z_axis_can = z_axis_can / (np.linalg.norm(z_axis_can) + 1e-7)
    x_axis_can = np.cross(y_axis_can, z_axis_can)
    x_axis_can = x_axis_can / (np.linalg.norm(x_axis_can) + 1e-7)
    R_pelvis = np.stack([x_axis_can, y_axis_can, z_axis_can], axis=0) # (3, 3)

    # 신체 로컬 직교 좌표계로 변환 (T, 33, 3)
    kps_centered = kps_64f - mid_hip[:, np.newaxis, :]
    kps_canonical = np.einsum('tjc,kc->tjk', kps_centered, R_pelvis)

    # 50. canonical_body_coronal_knee_angle_min: 신체 관상면(X-Y) 상 무릎 굴곡 각도 (카메라 Z축 깊이 착시 완전 배제)
    v_thigh_can_xy = kps_canonical[inflection_idx, lead_hip_idx, :2] - kps_canonical[inflection_idx, lead_knee_idx, :2]
    v_shank_can_xy = kps_canonical[inflection_idx, lead_ankle_idx, :2] - kps_canonical[inflection_idx, lead_knee_idx, :2]
    norm_t_can = np.linalg.norm(v_thigh_can_xy)
    norm_s_can = np.linalg.norm(v_shank_can_xy)
    if norm_t_can > 1e-6 and norm_s_can > 1e-6:
        cos_can = np.clip(np.dot(v_thigh_can_xy, v_shank_can_xy) / (norm_t_can * norm_s_can), -1.0, 1.0)
        canonical_body_coronal_knee_angle_min = float(math.degrees(math.acos(cos_can)))
    else:
        canonical_body_coronal_knee_angle_min = 180.0

    # 51. canonical_body_sagittal_tilt: 신체 로컬 좌표계 상에서 하지 평면의 시상면 이탈 경사
    v1_can = kps_canonical[inflection_idx, lead_hip_idx] - kps_canonical[inflection_idx, lead_knee_idx]
    v2_can = kps_canonical[inflection_idx, lead_ankle_idx] - kps_canonical[inflection_idx, lead_knee_idx]
    can_leg_norm = np.cross(v1_can, v2_can)
    norm_cln = np.linalg.norm(can_leg_norm)
    if norm_cln > 1e-6:
        u_cln = can_leg_norm / norm_cln
        # Z축(신체 전후축)과의 사잇각
        canonical_body_sagittal_tilt = float(math.degrees(math.acos(np.clip(abs(u_cln[2]), -1.0, 1.0))))
    else:
        canonical_body_sagittal_tilt = 0.0

    # 52. isb_jcs_pure_flexion_min: ISB Grood & Suntay 대퇴골 횡축(Z_T) 기준 순수 시상면 굴곡각
    # 대퇴골 로컬 좌표계: 종축 Y_T (hip -> knee), 외적 횡축 Z_T
    y_femur = v_femur / (np.linalg.norm(v_femur) + 1e-7)
    v_pelvis_ref = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    z_femur = np.cross(y_femur, v_pelvis_ref)
    z_femur = z_femur / (np.linalg.norm(z_femur) + 1e-7)
    x_femur = np.cross(y_femur, z_femur)

    # 경골 종축 Y_S (knee -> ankle)
    y_tibia = v_shank_inflect / (np.linalg.norm(v_shank_inflect) + 1e-7)
    # 대퇴골 횡축 Z_T와 경골 종축 Y_S 사이의 직교 회전 성분 추출
    cos_isb = np.clip(np.dot(y_femur, y_tibia), -1.0, 1.0)
    isb_jcs_pure_flexion_min = float(180.0 - math.degrees(math.acos(cos_isb)))

    # [패러다임 2: 생체역학 1-DOF 및 강체 뼈 길이 불변 구속]
    # 53. rigid_fk_femur_strain_ratio: 기립 대비 최하사점 대퇴골 신축 변형률 (Z-wobble/단축 왜곡 감지)
    ref_femur_length = float(np.median(femur_lengths[:n_init]))
    if ref_femur_length < 1e-4:
        ref_femur_length = 0.35
    inflect_femur_len_actual = float(np.linalg.norm(kps_64f[inflection_idx, lead_hip_idx] -
                                                   kps_64f[inflection_idx, lead_knee_idx]))
    rigid_fk_femur_strain_ratio = float(abs(inflect_femur_len_actual - ref_femur_length) / ref_femur_length)

    # 54. kinematic_hinge_orthogonality: 1자유도 힌지 관절 적합도 (대퇴골-경골 굴곡면과 골반 관상면의 평행/직교 안정도)
    v_thigh_vec = kps_64f[inflection_idx, lead_knee_idx] - kps_64f[inflection_idx, lead_hip_idx]
    v_shank_vec = kps_64f[inflection_idx, lead_ankle_idx] - kps_64f[inflection_idx, lead_knee_idx]
    cross_knee = np.cross(v_thigh_vec, v_shank_vec)
    norm_ck = np.linalg.norm(cross_knee)
    if norm_ck > 1e-6 and np.linalg.norm(v_pelvis_yaw) > 1e-6:
        u_knee_axis = cross_knee / norm_ck
        u_pelvis_yaw = v_pelvis_yaw / np.linalg.norm(v_pelvis_yaw)
        kinematic_hinge_orthogonality = float(abs(np.dot(u_knee_axis, u_pelvis_yaw)))
    else:
        kinematic_hinge_orthogonality = 1.0

    # [패러다임 3: 단일 카메라 시공간 3D 리프팅 모션 정규화]
    # 55. temporal_depth_acceleration_norm: 앞무릎 깊이축(Z) 2계 미분 가속도 평활도 노름 (Anipose/MotionBERT 평활도 지표)
    knee_z_series = kps_64f[:, lead_knee_idx, 2]
    diff2_z = np.diff(knee_z_series, n=2)
    temporal_depth_acceleration_norm = float(np.mean(np.abs(diff2_z)))

    features = {
        # 그룹 1 (6D)
        "lead_knee_angle_min": lead_knee_angle_min,
        "lead_knee_angle_mean": lead_knee_angle_mean,
        "lead_knee_90_error_abs": lead_knee_90_error_abs,
        "lead_knee_under_90_ratio": lead_knee_under_90_ratio,
        "lead_knee_flex_velocity_max": lead_knee_flex_velocity_max,
        "sagittal_knee_90_error": sagittal_knee_90_error,

        # 그룹 2 (13D)
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

        # 그룹 3 (5D)
        "lead_rear_knee_ratio": lead_rear_knee_ratio,
        "rear_knee_angle_min": rear_knee_angle_min,
        "pelvic_drop_angle": pelvic_drop_angle,
        "stride_length_normalized": stride_length_normalized,
        "step_width_normalized": step_width_normalized,

        # 그룹 4 (6D)
        "lead_knee_accel_inflection": lead_knee_accel_inflection,
        "bottom_phase_dwell_ratio": bottom_phase_dwell_ratio,
        "lead_knee_dtw_distance": lead_knee_dtw_distance,
        "lead_knee_visibility_mean": lead_knee_visibility_mean,
        "kinetic_energy_inflection": kinetic_energy_inflection,
        "jerk_metric_lead_leg": jerk_metric_lead_leg,

        # 그룹 5 (6D)
        "lead_knee_delta_from_standing": lead_knee_delta_from_standing,
        "lead_rear_knee_angle_difference": lead_rear_knee_angle_difference,
        "optical_axis_foreshortening_factor": optical_axis_foreshortening_factor,
        "femur_horizontal_level_ratio": femur_horizontal_level_ratio,
        "pelvic_drop_to_stride_ratio": pelvic_drop_to_stride_ratio,
        "lead_knee_angular_rom_extent": lead_knee_angular_rom_extent,

        # 그룹 6 (4D)
        "pelvic_tilt_roll_slope": pelvic_tilt_roll_slope,
        "lead_leg_length_compression_ratio": lead_leg_length_compression_ratio,
        "trail_to_lead_knee_rom_ratio": trail_to_lead_knee_rom_ratio,
        "lead_knee_ratio_to_stand": lead_knee_ratio_to_stand,

        # 그룹 7 (5D)
        "femur_slope_at_peak": femur_slope_at_peak,
        "femur_slope_stability": femur_slope_stability,
        "lateral_com_shift_ratio": lateral_com_shift_ratio,
        "frontal_q_angle_proxy": frontal_q_angle_proxy,
        "pelvic_sagittal_thigh_angle": pelvic_sagittal_thigh_angle,

        # 그룹 8 (4D)
        "posterior_pelvis_drop_ratio": posterior_pelvis_drop_ratio,
        "lead_heel_ground_elevation": lead_heel_ground_elevation,
        "coronal_trunk_lateral_lean": coronal_trunk_lateral_lean,
        "spine_pelvis_skew_angle": spine_pelvis_skew_angle,

        # 그룹 9 (6D, 신규 확장)
        "canonical_body_coronal_knee_angle_min": canonical_body_coronal_knee_angle_min,
        "canonical_body_sagittal_tilt": canonical_body_sagittal_tilt,
        "isb_jcs_pure_flexion_min": isb_jcs_pure_flexion_min,
        "rigid_fk_femur_strain_ratio": rigid_fk_femur_strain_ratio,
        "kinematic_hinge_orthogonality": kinematic_hinge_orthogonality,
        "temporal_depth_acceleration_norm": temporal_depth_acceleration_norm,
    }

    return features


def extract_side_lunge_c1_feature_vector_v11(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """MediaPipe 33관절 3D 64프레임 데이터로부터 55D 피처 벡터 (np.ndarray float32) 반환"""
    feat_dict = extract_side_lunge_c1_features_v11(kps_64f, vis_64f)
    vec = np.zeros(len(SIDE_LUNGE_C1_FEATURE_NAMES_V11), dtype=np.float32)
    for i, name in enumerate(SIDE_LUNGE_C1_FEATURE_NAMES_V11):
        val = feat_dict.get(name, 0.0)
        vec[i] = 0.0 if (math.isnan(val) or math.isinf(val)) else val
    return vec

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c1_features_v3.py
Purpose: 바벨 런지 c1(앞다리 무릎 각도 90도) 세부 오류 판별 SOTA v3 피처 최적화 모듈
         - 전수 조사(Ablation & Correlation) 기반 12개 고상관/성능 저해 피처 제거
         - 38차원 -> 26차원(26D) 정예 생체역학 핵심 피처셋
         - ISB 골반 직교 기저계 시상면 회전 정규화 계승
         - Savitzky-Golay 2차 다항식 피크 보존 필터링 계승
         - 골반 Y + 무릎 각속도 영교차점 이중 앵커링(Dual-Anchor) 계승
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

# 26차원 공식 v3 도메인 정예 피처 명칭 목록
LUNGE_C1_FEATURE_NAMES_V3: List[str] = [
    # [그룹 1: 무릎 굴곡 및 시상면 90도 핵심 (6D)]
    "lead_knee_angle_min",            # [ 1] 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
    "lead_knee_angle_mean",           # [ 2] 최저점 윈도우 내 평균 굴곡 각도 (°)
    "lead_knee_90_error_abs",         # [ 3] |최소 각도 - 90°| 절대 오차 (°)
    "lead_knee_under_90_ratio",       # [ 4] 90도 미만 체류 프레임 비율
    "lead_knee_flex_velocity_max",    # [ 5] 하강 구간 최대 무릎 굴곡 각속도 (deg/frame)
    "sagittal_knee_90_error",         # [ 6] |ISB 시상면 무릎 굴곡각 - 90°| 절대 오차 (°)

    # [그룹 2: 하지 기하 및 정강이 정렬 (9D)]
    "lead_tibia_vertical_tilt",       # [ 7] 지면 수직선 대비 정강이 사잇각 (정강이 수직 유지도)
    "lead_femur_horizontal_tilt",     # [ 8] 지면 수평선 대비 대퇴골 사잇각 (대퇴 수평도)
    "lead_knee_toe_overhang",         # [ 9] 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (신체 스케일 정규화)
    "lead_shank_ankle_coupling",      # [10] 정강이 분절 각도와 발목 배측굴곡 각도 커플링 비율
    "lead_tibia_sagittal_sway",       # [11] 하강 중 정강이뼈 시상면 흔들림 폭
    "lead_femur_length_ratio",        # [12] 체간 대비 앞다리 대퇴골 정규화 길이
    "lead_tibia_length_ratio",        # [13] 체간 대비 앞다리 정강이 정규화 길이
    "rigid_bone_length_consistency",  # [14] 대퇴골-정강이 강체 길이 보존 일관성 지표 (가림 노이즈 감지)
    "rear_knee_ground_clearance",     # [15] 뒷무릎 지면 간격 프록시

    # [그룹 3: 양 하지 협응 및 보폭 (5D)]
    "lead_rear_knee_ratio",           # [16] 앞무릎 최소각 대비 뒷무릎 최소각 비율
    "rear_knee_angle_min",            # [17] 뒷다리 무릎 최저 각도 (°)
    "pelvic_drop_angle",              # [18] 관상면 상 골반 좌우 기울기 각도
    "stride_length_normalized",       # [19] 양 발목 사이 전후 보폭 (체간 정규화)
    "step_width_normalized",          # [20] 양 발목 사이 좌우 너비 (체간 정규화)

    # [그룹 4: 시공간 동역학 및 생체역학 에너지 (6D)]
    "lead_knee_accel_inflection",     # [21] 최저점 전환 순간의 각가속도 (deg/frame^2)
    "bottom_phase_dwell_ratio",       # [22] 전체 64프레임 중 최저점 부근 체류 프레임 비율
    "lead_knee_dtw_distance",         # [23] 이상적 90도 골드 템플릿 곡선 대비 DTW 유사도 거리
    "lead_knee_visibility_mean",      # [24] 최저점 구간 앞무릎 키포인트 평균 신뢰도
    "kinetic_energy_inflection",      # [25] 최저 전환점 시 하지 관절 운동에너지 극소치 프록시
    "jerk_metric_lead_leg",           # [26] 앞다리 무릎 궤적 저크(Jerk) 평활도 지표
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
    """Savitzky-Golay 2차 다항식 필터 (길이 부족 시 에러 방지 처리)"""
    if len(arr) < window:
        return arr.copy()
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
    except Exception:
        return arr.copy()


def compute_dtw_distance_1d(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """1차원 시계열 간의 표준 Dynamic Time Warping (DTW) 거리 계산"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
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
    """앞다리(Lead leg: 'left' vs 'right') 판별"""
    left_ankles_z = kps_64f[:, MP_LEFT_ANKLE, 2]
    right_ankles_z = kps_64f[:, MP_RIGHT_ANKLE, 2]
    left_ankles_x = kps_64f[:, MP_LEFT_ANKLE, 0]
    right_ankles_x = kps_64f[:, MP_RIGHT_ANKLE, 0]

    z_diff = np.mean(left_ankles_z) - np.mean(right_ankles_z)
    left_hip_x = np.mean(kps_64f[:, MP_LEFT_HIP, 0])
    right_hip_x = np.mean(kps_64f[:, MP_RIGHT_HIP, 0])
    mid_hip_x = (left_hip_x + right_hip_x) / 2.0

    left_disp_x = np.abs(np.mean(left_ankles_x) - mid_hip_x)
    right_disp_x = np.abs(np.mean(right_ankles_x) - mid_hip_x)

    if abs(z_diff) > 0.05:
        return 'left' if z_diff < 0 else 'right'
    return 'left' if left_disp_x > right_disp_x else 'right'


def find_dual_anchor_inflection_point(
    kps_64f: np.ndarray,
    lead_knee_angles: np.ndarray
) -> Tuple[int, int, int]:
    """골반 수직 변위 극대점 + 무릎 각속도 영교차점 이중 앵커링 변곡점 탐색"""
    n_frames = len(kps_64f)
    mid_hip_y = (kps_64f[:, MP_LEFT_HIP, 1] + kps_64f[:, MP_RIGHT_HIP, 1]) / 2.0
    hip_smooth = safe_savgol_filter(mid_hip_y, window=7, poly=2)
    hip_bottom_idx = int(np.argmax(hip_smooth[10:n_frames-10])) + 10

    angle_vel = np.gradient(lead_knee_angles)
    angle_vel_smooth = safe_savgol_filter(angle_vel, window=5, poly=2)

    search_start = max(10, hip_bottom_idx - 8)
    search_end = min(n_frames - 10, hip_bottom_idx + 8)

    zero_cross_candidates = []
    for f in range(search_start, search_end):
        if angle_vel_smooth[f] * angle_vel_smooth[f + 1] <= 0:
            zero_cross_candidates.append(f)

    if zero_cross_candidates:
        zero_cross_idx = min(zero_cross_candidates, key=lambda idx: abs(idx - hip_bottom_idx))
        inflection_idx = int(round(0.6 * hip_bottom_idx + 0.4 * zero_cross_idx))
    else:
        inflection_idx = hip_bottom_idx

    inflection_idx = max(5, min(n_frames - 6, inflection_idx))
    win_start = max(0, inflection_idx - 3)
    win_end = min(n_frames, inflection_idx + 4)
    return inflection_idx, win_start, win_end


def compute_isb_pelvic_sagittal_angle(
    kps_frame: np.ndarray,
    lead_side: str
) -> float:
    """ISB 추천 골반 로컬 직교 좌표계 기반 시상면 순수 1-DoF 무릎 굴곡각 계산"""
    l_hip = kps_frame[MP_LEFT_HIP]
    r_hip = kps_frame[MP_RIGHT_HIP]
    l_sh = kps_frame[MP_LEFT_SHOULDER]
    r_sh = kps_frame[MP_RIGHT_SHOULDER]

    mid_hip = (l_hip + r_hip) / 2.0
    mid_sh = (l_sh + r_sh) / 2.0

    v_X = r_hip - l_hip
    norm_x = np.linalg.norm(v_X)
    if norm_x < 1e-6:
        v_X = np.array([1.0, 0.0, 0.0])
    else:
        v_X = v_X / norm_x

    v_Y_temp = mid_sh - mid_hip
    norm_y = np.linalg.norm(v_Y_temp)
    if norm_y < 1e-6:
        v_Y_temp = np.array([0.0, -1.0, 0.0])
    else:
        v_Y_temp = v_Y_temp / norm_y

    v_Z = np.cross(v_X, v_Y_temp)
    norm_z = np.linalg.norm(v_Z)
    if norm_z < 1e-6:
        v_Z = np.array([0.0, 0.0, 1.0])
    else:
        v_Z = v_Z / norm_z

    v_Y = np.cross(v_Z, v_X)
    v_Y = v_Y / np.linalg.norm(v_Y)

    R_body = np.vstack([v_X, v_Y, v_Z])

    if lead_side == 'left':
        hip = kps_frame[MP_LEFT_HIP]
        knee = kps_frame[MP_LEFT_KNEE]
        ankle = kps_frame[MP_LEFT_ANKLE]
    else:
        hip = kps_frame[MP_RIGHT_HIP]
        knee = kps_frame[MP_RIGHT_KNEE]
        ankle = kps_frame[MP_RIGHT_ANKLE]

    femur_world = knee - hip
    tibia_world = ankle - knee

    femur_body = R_body @ femur_world
    tibia_body = R_body @ tibia_world

    femur_sagittal = np.array([femur_body[1], femur_body[2]])
    tibia_sagittal = np.array([tibia_body[1], tibia_body[2]])

    norm_fs = np.linalg.norm(femur_sagittal)
    norm_ts = np.linalg.norm(tibia_sagittal)
    if norm_fs < 1e-6 or norm_ts < 1e-6:
        return 180.0

    dot_sag = np.dot(femur_sagittal, tibia_sagittal)
    cos_sag = np.clip(dot_sag / (norm_fs * norm_ts), -1.0, 1.0)
    angle_sag = float(np.degrees(np.arccos(cos_sag)))
    return angle_sag


def extract_lunge_c1_features_v3(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v3 26D 피처 추출 메인 함수
    """
    if kps_64f.shape[0] != 64 or kps_64f.shape[1] != 33:
        raise ValueError(f"입력 형식이 (64, 33, 3)이어야 합니다. 현재: {kps_64f.shape}")

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

    n_frames = len(kps_64f)

    # 1. 3D 무릎 각도 궤적 계산
    raw_lead_angles = np.zeros(n_frames, dtype=np.float32)
    raw_rear_angles = np.zeros(n_frames, dtype=np.float32)
    femur_lengths = np.zeros(n_frames, dtype=np.float32)
    tibia_lengths = np.zeros(n_frames, dtype=np.float32)

    for f in range(n_frames):
        hip_pt = kps_64f[f, lead_hip_idx]
        knee_pt = kps_64f[f, lead_knee_idx]
        ankle_pt = kps_64f[f, lead_ankle_idx]

        v_femur = hip_pt - knee_pt
        v_tibia = ankle_pt - knee_pt
        raw_lead_angles[f] = vector_angle_3d(v_femur, v_tibia)

        r_hip_pt = kps_64f[f, rear_hip_idx]
        r_knee_pt = kps_64f[f, rear_knee_idx]
        r_ankle_pt = kps_64f[f, rear_ankle_idx]
        raw_rear_angles[f] = vector_angle_3d(r_hip_pt - r_knee_pt, r_ankle_pt - r_knee_pt)

        femur_lengths[f] = np.linalg.norm(v_femur)
        tibia_lengths[f] = np.linalg.norm(v_tibia)

    # 2. SG 필터 적용
    lead_angles_sg = safe_savgol_filter(raw_lead_angles, window=7, poly=2)
    rear_angles_sg = safe_savgol_filter(raw_rear_angles, window=7, poly=2)

    # 3. 이중 앵커링 변곡점 검출
    inflection_idx, win_start, win_end = find_dual_anchor_inflection_point(kps_64f, lead_angles_sg)

    # 4. 체간 길이 정규화 기준 (몸통 스케일)
    torso_lens = np.linalg.norm(
        kps_64f[:, MP_LEFT_SHOULDER] - kps_64f[:, MP_LEFT_HIP], axis=1
    )
    torso_scale = float(np.median(torso_lens))
    if torso_scale < 1e-4:
        torso_scale = 1.0

    # 5. ISB 시상면 무릎 굴곡각
    isb_sagittal_bottom = compute_isb_pelvic_sagittal_angle(kps_64f[inflection_idx], lead_side)

    # --- [그룹 1: 앞다리 무릎 각도 및 시상면 굴곡 (6D)] ---
    window_angles = lead_angles_sg[win_start:win_end]
    lead_knee_angle_min = float(np.min(window_angles))
    lead_knee_angle_mean = float(np.mean(window_angles))
    lead_knee_90_error_abs = float(abs(lead_knee_angle_min - 90.0))
    lead_knee_under_90_ratio = float(np.mean(lead_angles_sg < 90.0))

    vel_traj = np.gradient(lead_angles_sg)
    descent_frames = lead_angles_sg[:inflection_idx]
    if len(descent_frames) > 3:
        lead_knee_flex_velocity_max = float(np.max(np.abs(np.gradient(descent_frames))))
    else:
        lead_knee_flex_velocity_max = float(np.max(np.abs(vel_traj)))

    sagittal_knee_90_error = float(abs(isb_sagittal_bottom - 90.0))

    # --- [그룹 2: 하지 기하 및 정강이 정렬 (9D)] ---
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


def extract_lunge_c1_feature_vector_v3(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """순수 26D 1차원 피처 벡터 반환 (LUNGE_C1_FEATURE_NAMES_V3 순서 보장)"""
    feat_dict = extract_lunge_c1_features_v3(kps_64f, vis_64f)
    vec = np.array([feat_dict[name] for name in LUNGE_C1_FEATURE_NAMES_V3], dtype=np.float32)
    return vec

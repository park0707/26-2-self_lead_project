# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Module: lunge_c4_features_v2.py
Purpose: 선행 연구 및 AQA 최신 기법 기반 36차원 바벨 런지 척추 중립 특화 피처 추출기 (v2)
         - [기존 v1 28D 계승]: 경추 3축(8D) + 체간 측굴(8D) + 운동사슬(6D) + 최하사점 위상(6D)
         - [신규 혁신 1: 골드 템플릿 DTW 잔차 4D]: 정상 기준 궤적 대비 비선형 위상 편차 및 L2 적분치
         - [신규 혁신 2: 체간 로컬 좌표계 시점 불변 투영 4D]: 신체 중심 정규화 좌표계 기반 시점 독립적 척추 롤/비틀림 각도
Rules: 100% 순수 비전 MediaPipe 33관절 3D 입력 기반
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe 33 관절 인덱스
MP_NOSE = 0
MP_LEFT_EYE = 2
MP_RIGHT_EYE = 5
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

# 36차원 공식 도메인 피처 명칭 (v2)
LUNGE_C4_FEATURE_NAMES_V2: List[str] = [
    # [그룹 1] 경추 3축 정렬 피처 (8D - v1 계승)
    "cervical_yaw_max",             # [ 1] 두부-시선 횡단면 회전각(Yaw) 최대 절대치 (°)
    "cervical_yaw_mean",            # [ 2] 시퀀스 전체 평균 두부 Yaw 편차각 (°)
    "cervical_pitch_max",           # [ 3] 두부-시선 수평선 대비 상방 젖힘(Pitch Up) 최대각 (°)
    "cervical_pitch_min",           # [ 4] 두부-시선 수평선 대비 하방 숙임(Pitch Down) 최소각 (°)
    "cervical_roll_max",            # [ 5] 양 눈-양 귀 선의 전두면 롤 각도 최대치 (°)
    "ear_shoulder_asymmetry",       # [ 6] 좌/우 귀-어깨 간 수직 거리 비대칭도
    "head_trunk_yaw_diff",          # [ 7] 흉곽 정면 벡터 대비 두부 시선 벡터 상대 편차각 (°)
    "cervical_bottom_pitch",        # [ 8] 최하사점(t_bottom) 순간의 경추 피치 각도 (°)

    # [그룹 2] 전두면 체간 측굴 피처 (8D - v1 계승)
    "trunk_roll_max",               # [ 9] 척추 중심선 전두면 롤 각도 최대치 (°) - 몸통 갸우뚱
    "trunk_roll_mean",              # [10] 시퀀스 전체 평균 체간 롤 각도 (°)
    "shoulder_roll_max",            # [11] 양 어깨선 수평선 대비 롤 각도 최대치 (°)
    "pelvic_obliquity_roll",        # [12] 양 골반선 수평선 대비 롤 각도 최대치 (°)
    "spine_lateral_inflexion",      # [13] 어깨-골반 직선 대비 흉추 횡방향 이탈 깊이
    "torso_bottom_roll",            # [14] 최하사점 순간의 체간 롤 각도 (°)
    "shoulder_pelvis_coupling",     # [15] 어깨 롤과 골반 롤의 비대칭 커플링 지수
    "trunk_roll_integral",          # [16] 임계치(5°) 초과 체간 측굴 누적 적분치

    # [그룹 3] 하지 비대칭 사슬 피처 (6D - v1 계승)
    "split_base_stability_ratio",   # [17] 앞발-뒷발 전후 간격 대비 좌우 기저면 너비비
    "pelvis_lateral_sway",          # [18] 기저면 중심 대비 골반 중심의 좌우 요동폭
    "spine_knee_alignment",         # [19] 선행 무릎 수직선과 척추 축의 관상면 일치도
    "asym_hip_drop_bottom",         # [20] 최하사점에서의 비대칭 골반 하강률
    "kinetic_lateral_drift",        # [21] 발목-골반-어깨-두부로 이어지는 횡방향 중심선 누적 편차
    "counter_tilt_compensation",    # [22] 골반 기울임에 대한 상체 반대 보상 꺾임 계측

    # [그룹 4] 최하사점 위상 및 동역학 피처 (6D - v1 계승)
    "bottom_spine_stability_std",   # [23] 최하사점 윈도우 내 척추 롤 각도 표준편차 (°)
    "bottom_head_jitter_std",       # [24] 최하사점 윈도우 내 두부 시선 각도 표준편차 (°)
    "descent_roll_vel_peak",        # [25] 하강 구간 체간 측굴 각속도 피크 (deg/frame)
    "ascent_recovery_sway",         # [26] 상승 복귀 구간 체간 측굴 요동 누적치
    "bottom_active_roll_mean",      # [27] Active Window 내 평균 체간 롤 각도 (°)
    "bottom_active_yaw_mean",       # [28] Active Window 내 평균 두부 요 각도 (°)

    # [그룹 5 ★ 신규] 골드 템플릿 DTW 및 위상 잔차 편차 피처 (4D)
    "golden_spine_dtw_dist",        # [29] 정상 골드 척추 롤 템플릿 대비 DTW 비선형 정렬 거리
    "golden_cervical_dtw_dist",     # [30] 정상 골드 두부 요 템플릿 대비 DTW 비선형 정렬 거리
    "golden_spine_l2_residual",     # [31] 정상 골드 척추 롤 템플릿 대비 프레임별 L2 잔차 적분치
    "golden_bottom_phase_error",    # [32] 정상 골드 템플릿 대비 최하사점(t_bottom) 순간의 복합 형상 잔차

    # [그룹 6 ★ 신규] 인체 로컬 좌표계 기반 시점 불변 피처 (4D)
    "local_trunk_roll_inv",         # [33] 골반-어깨 로컬 직교 좌표계 상의 시점 불변 체간 롤 각도
    "local_cervical_yaw_inv",       # [34] 체간 정면 법선 로컬 축 상의 시점 불변 두부 회전각
    "local_torso_torsion_max",      # [35] 골반 국소 축 대비 어깨선 3D 비틀림(Torsion) 최대치
    "local_spine_lateral_c_curve"   # [36] 로컬 관상면 상 척추 곡률 C-Curve 적분치
]


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _dtw_distance_1d(s1: np.ndarray, s2: np.ndarray) -> float:
    """1차원 시계열 간의 고속 Dynamic Time Warping (DTW) 거리 계산"""
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


# 정상 골드 템플릿 (바벨 런지 정상 시퀀스들의 평균 기준 곡선 - 64F)
# 척추 롤(정상: 0~2도 미만 유지), 두부 요(정상: 0~3도 미만 정면 유지)
GOLDEN_TRUNK_ROLL_TEMPLATE = np.full(64, 1.2, dtype=np.float32)
GOLDEN_CERVICAL_YAW_TEMPLATE = np.full(64, 2.0, dtype=np.float32)


def extract_lunge_c4_features_v2(seq_33: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 바벨 런지 시퀀스로부터 선행 연구 기반 36차원 척추 중립 특화 피처 추출 (v2)
    """
    assert len(seq_33.shape) == 3, f"Expected 3D tensor (T, 33, C), got {seq_33.shape}"
    T = seq_33.shape[0]
    coords = seq_33[:, :, :3].astype(np.float32)

    # 스무딩
    smoothed = np.zeros_like(coords)
    for j in range(coords.shape[1]):
        for c in range(3):
            smoothed[:, j, c] = gaussian_filter1d(coords[:, j, c], sigma=1.0)

    # 1. 체간 및 두부 벡터
    shoulder_mid = (smoothed[:, MP_LEFT_SHOULDER] + smoothed[:, MP_RIGHT_SHOULDER]) * 0.5
    hip_mid = (smoothed[:, MP_LEFT_HIP] + smoothed[:, MP_RIGHT_HIP]) * 0.5
    torso_vec = shoulder_mid - hip_mid
    torso_len = np.mean(np.linalg.norm(torso_vec, axis=-1)) + 1e-6

    shoulder_vec = smoothed[:, MP_RIGHT_SHOULDER] - smoothed[:, MP_LEFT_SHOULDER]
    chest_normal = np.cross(torso_vec, shoulder_vec)
    chest_fwd = chest_normal / (np.linalg.norm(chest_normal, axis=-1, keepdims=True) + 1e-6)

    nose = smoothed[:, MP_NOSE]
    left_ear = smoothed[:, MP_LEFT_EAR]
    right_ear = smoothed[:, MP_RIGHT_EAR]
    head_fwd = nose - (left_ear + right_ear) * 0.5
    head_fwd_dir = head_fwd / (np.linalg.norm(head_fwd, axis=-1, keepdims=True) + 1e-6)

    # 최하사점 감지
    left_femur = smoothed[:, MP_LEFT_HIP] - smoothed[:, MP_LEFT_KNEE]
    left_tibia = smoothed[:, MP_LEFT_ANKLE] - smoothed[:, MP_LEFT_KNEE]
    right_femur = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_RIGHT_KNEE]
    right_tibia = smoothed[:, MP_RIGHT_ANKLE] - smoothed[:, MP_RIGHT_KNEE]
    left_knee_angles = np.array([_angle_between_vectors(left_femur[t], left_tibia[t]) for t in range(T)])
    right_knee_angles = np.array([_angle_between_vectors(right_femur[t], right_tibia[t]) for t in range(T)])
    min_knee_curve = np.minimum(left_knee_angles, right_knee_angles)
    t_bottom = int(np.argmin(min_knee_curve))
    w_start = max(0, t_bottom - 6)
    w_end = min(T, t_bottom + 7)

    # -------------------------------------------------------------
    # 그룹 1: 경추 3축 정렬 피처 (8D)
    # -------------------------------------------------------------
    cervical_yaw_curve = np.zeros(T, dtype=np.float32)
    for t in range(T):
        hf_xz = np.array([head_fwd_dir[t, 0], 0.0, head_fwd_dir[t, 2]])
        cf_xz = np.array([chest_fwd[t, 0], 0.0, chest_fwd[t, 2]])
        cervical_yaw_curve[t] = _angle_between_vectors(hf_xz, cf_xz)
    cervical_yaw_max = float(np.max(cervical_yaw_curve))
    cervical_yaw_mean = float(np.mean(cervical_yaw_curve))

    cervical_pitch_curve = np.degrees(np.arcsin(np.clip(-head_fwd_dir[:, 1], -1.0, 1.0)))
    cervical_pitch_max = float(np.max(cervical_pitch_curve))
    cervical_pitch_min = float(np.min(cervical_pitch_curve))

    ear_vec = right_ear - left_ear
    cervical_roll_curve = np.abs(np.degrees(np.arctan2(ear_vec[:, 1], ear_vec[:, 0] + 1e-6)))
    cervical_roll_curve = np.minimum(cervical_roll_curve, np.abs(180.0 - cervical_roll_curve))
    cervical_roll_max = float(np.max(cervical_roll_curve))

    left_ear_shoulder_dist = np.abs(smoothed[:, MP_LEFT_EAR, 1] - smoothed[:, MP_LEFT_SHOULDER, 1])
    right_ear_shoulder_dist = np.abs(smoothed[:, MP_RIGHT_EAR, 1] - smoothed[:, MP_RIGHT_SHOULDER, 1])
    ear_shoulder_asymmetry = float(np.max(np.abs(left_ear_shoulder_dist - right_ear_shoulder_dist) / torso_len))

    head_trunk_yaw_diff = float(np.mean([_angle_between_vectors(head_fwd_dir[t], chest_fwd[t]) for t in range(T)]))
    cervical_bottom_pitch = float(cervical_pitch_curve[t_bottom])

    # -------------------------------------------------------------
    # 그룹 2: 전두면 체간 측굴 피처 (8D)
    # -------------------------------------------------------------
    spine_xy = torso_vec[:, :2]
    trunk_roll_curve = np.abs(np.degrees(np.arctan2(spine_xy[:, 0], -spine_xy[:, 1] + 1e-6)))
    trunk_roll_max = float(np.max(trunk_roll_curve))
    trunk_roll_mean = float(np.mean(trunk_roll_curve))

    shoulder_xy = shoulder_vec[:, :2]
    sh_roll_curve = np.abs(np.degrees(np.arctan2(shoulder_xy[:, 1], shoulder_xy[:, 0] + 1e-6)))
    sh_roll_curve = np.minimum(sh_roll_curve, np.abs(180.0 - sh_roll_curve))
    shoulder_roll_max = float(np.max(sh_roll_curve))

    hip_vec = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_LEFT_HIP]
    hip_xy = hip_vec[:, :2]
    hip_roll_curve = np.abs(np.degrees(np.arctan2(hip_xy[:, 1], hip_xy[:, 0] + 1e-6)))
    hip_roll_curve = np.minimum(hip_roll_curve, np.abs(180.0 - hip_roll_curve))
    pelvic_obliquity_roll = float(np.max(hip_roll_curve))

    mid_spine = (shoulder_mid + hip_mid) * 0.5
    inflexion_x = np.abs(mid_spine[:, 0] - (shoulder_mid[:, 0] + hip_mid[:, 0]) * 0.5)
    spine_lateral_inflexion = float(np.max(inflexion_x) / torso_len)

    torso_bottom_roll = float(trunk_roll_curve[t_bottom])
    shoulder_pelvis_coupling = float(np.std(sh_roll_curve - hip_roll_curve))
    excess_roll = np.maximum(0.0, trunk_roll_curve - 5.0)
    trunk_roll_integral = float(np.sum(excess_roll))

    # -------------------------------------------------------------
    # 그룹 3: 하지 비대칭 사슬 피처 (6D)
    # -------------------------------------------------------------
    l_ankle = smoothed[:, MP_LEFT_ANKLE]
    r_ankle = smoothed[:, MP_RIGHT_ANKLE]
    step_ap = np.abs(l_ankle[:, 2] - r_ankle[:, 2])
    step_ml = np.abs(l_ankle[:, 0] - r_ankle[:, 0])
    split_base_stability_ratio = float(np.mean(step_ml / (step_ap + 1e-6)))

    base_mid_x = (l_ankle[:, 0] + r_ankle[:, 0]) * 0.5
    pelvis_lateral_sway = float(np.std(hip_mid[:, 0] - base_mid_x) / torso_len)

    is_left_lead = (left_knee_angles[t_bottom] < right_knee_angles[t_bottom])
    lead_knee_x = smoothed[:, MP_LEFT_KNEE if is_left_lead else MP_RIGHT_KNEE, 0]
    spine_knee_alignment = float(np.mean(np.abs(hip_mid[:, 0] - lead_knee_x) / torso_len))

    asym_hip_drop_bottom = float(np.abs(smoothed[t_bottom, MP_LEFT_HIP, 1] - smoothed[t_bottom, MP_RIGHT_HIP, 1]) / torso_len)
    chain_drift = np.abs(smoothed[:, MP_NOSE, 0] - hip_mid[:, 0]) + np.abs(shoulder_mid[:, 0] - hip_mid[:, 0])
    kinetic_lateral_drift = float(np.mean(chain_drift / torso_len))
    counter_tilt_compensation = float(np.mean(np.abs(sh_roll_curve + hip_roll_curve) / (trunk_roll_curve + 1e-6)))

    # -------------------------------------------------------------
    # 그룹 4: 최하사점 위상 및 동역학 피처 (6D)
    # -------------------------------------------------------------
    bottom_spine_stability_std = float(np.std(trunk_roll_curve[w_start:w_end]))
    bottom_head_jitter_std = float(np.std(cervical_yaw_curve[w_start:w_end]))

    roll_velocity = np.abs(np.diff(trunk_roll_curve))
    descent_roll_vel_peak = float(np.max(roll_velocity[:max(1, t_bottom)]))
    ascent_recovery_sway = float(np.sum(roll_velocity[t_bottom:]))

    bottom_active_roll_mean = float(np.mean(trunk_roll_curve[w_start:w_end]))
    bottom_active_yaw_mean = float(np.mean(cervical_yaw_curve[w_start:w_end]))

    # -------------------------------------------------------------
    # 그룹 5 ★ [선행 연구 신규] 골드 템플릿 DTW 및 위상 잔차 (4D)
    # -------------------------------------------------------------
    golden_spine_dtw_dist = _dtw_distance_1d(trunk_roll_curve, GOLDEN_TRUNK_ROLL_TEMPLATE)
    golden_cervical_dtw_dist = _dtw_distance_1d(cervical_yaw_curve, GOLDEN_CERVICAL_YAW_TEMPLATE)
    golden_spine_l2_residual = float(np.sqrt(np.mean((trunk_roll_curve - GOLDEN_TRUNK_ROLL_TEMPLATE) ** 2)))
    golden_bottom_phase_error = float(abs(torso_bottom_roll - 1.2) + abs(cervical_yaw_curve[t_bottom] - 2.0))

    # -------------------------------------------------------------
    # 그룹 6 ★ [선행 연구 신규] 인체 로컬 좌표계 시점 불변 투영 (4D)
    # -------------------------------------------------------------
    # 체간 축(Y_loc) = 골반중심 -> 어깨중심
    # 좌우 축(X_loc) = 양 골반 벡터를 Y_loc에 수직하게 Gram-Schmidt 정사영
    # 정면 축(Z_loc) = X_loc x Y_loc
    local_rolls = []
    local_yaws = []
    torsions = []
    for t in range(T):
        y_loc = torso_vec[t] / (np.linalg.norm(torso_vec[t]) + 1e-6)
        h_vec = hip_vec[t]
        x_loc = h_vec - np.dot(h_vec, y_loc) * y_loc
        x_loc = x_loc / (np.linalg.norm(x_loc) + 1e-6)
        z_loc = np.cross(x_loc, y_loc)

        # 로컬 관상면(XY) 상에서 척추와 어깨의 기울기
        sh_local_x = np.dot(shoulder_vec[t], x_loc)
        sh_local_y = np.dot(shoulder_vec[t], y_loc)
        sh_local_z = np.dot(shoulder_vec[t], z_loc)

        loc_roll = abs(math.degrees(math.atan2(sh_local_y, abs(sh_local_x) + 1e-6)))
        local_rolls.append(loc_roll)

        # 흉곽-골반 비틀림 (Torsion): 어깨 벡터의 Z_loc 투영각
        torsion = abs(math.degrees(math.atan2(sh_local_z, abs(sh_local_x) + 1e-6)))
        torsions.append(torsion)

        # 로컬 횡단면(XZ) 상에서 두부 시선 벡터 투영 각도
        hd_x = np.dot(head_fwd_dir[t], x_loc)
        hd_z = np.dot(head_fwd_dir[t], z_loc)
        loc_yaw = abs(math.degrees(math.atan2(hd_x, abs(hd_z) + 1e-6)))
        local_yaws.append(loc_yaw)

    local_trunk_roll_inv = float(np.max(local_rolls))
    local_cervical_yaw_inv = float(np.max(local_yaws))
    local_torso_torsion_max = float(np.max(torsions))
    local_spine_lateral_c_curve = float(np.sum(local_rolls) / T)

    feats_36d = np.array([
        # 그룹 1 (8D)
        cervical_yaw_max,
        cervical_yaw_mean,
        cervical_pitch_max,
        cervical_pitch_min,
        cervical_roll_max,
        ear_shoulder_asymmetry,
        head_trunk_yaw_diff,
        cervical_bottom_pitch,

        # 그룹 2 (8D)
        trunk_roll_max,
        trunk_roll_mean,
        shoulder_roll_max,
        pelvic_obliquity_roll,
        spine_lateral_inflexion,
        torso_bottom_roll,
        shoulder_pelvis_coupling,
        trunk_roll_integral,

        # 그룹 3 (6D)
        split_base_stability_ratio,
        pelvis_lateral_sway,
        spine_knee_alignment,
        asym_hip_drop_bottom,
        kinetic_lateral_drift,
        counter_tilt_compensation,

        # 그룹 4 (6D)
        bottom_spine_stability_std,
        bottom_head_jitter_std,
        descent_roll_vel_peak,
        ascent_recovery_sway,
        bottom_active_roll_mean,
        bottom_active_yaw_mean,

        # 그룹 5 (4D 신규)
        golden_spine_dtw_dist,
        golden_cervical_dtw_dist,
        golden_spine_l2_residual,
        golden_bottom_phase_error,

        # 그룹 6 (4D 신규)
        local_trunk_roll_inv,
        local_cervical_yaw_inv,
        local_torso_torsion_max,
        local_spine_lateral_c_curve
    ], dtype=np.float32)

    return np.nan_to_num(feats_36d, nan=0.0, posinf=100.0, neginf=-100.0)

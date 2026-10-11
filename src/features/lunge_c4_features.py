# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c4_features.py
Purpose: 바벨 런지 c4(척추의 중립 위반 / Spine Neutrality Defect) 전용 28차원 생체역학 도메인 피처 추출 모듈
         - 경추 3축(Yaw, Pitch, Roll) 정렬 및 시선 4방향(좌/우/상/하) 이탈 포착 (8D)
         - 하지 분할(Split-Stance)에 따른 전두면 체간 측굴(몸통 갸우뚱, Spine Roll) 포착 (8D)
         - 하지-골반-척추 운동사슬 비대칭 보상 붕괴 계측 (6D)
         - 최하사점(Active Bottom Phase) 위상 안정성 및 동역학 포착 (6D)
Rules: 100% 순수 비전 MediaPipe 33관절 3D 입력 (AI-Hub JSON 엄격 배제, 무결성 보장)
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe 33 관절 인덱스
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
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

# 28차원 공식 도메인 피처 명칭
LUNGE_C4_FEATURE_NAMES: List[str] = [
    # [그룹 1] 경추 3축 정렬 피처 (8D)
    "cervical_yaw_max",             # [ 1] 두부-시선 횡단면 회전각(Yaw) 최대 절대치 (°) - 좌/우 보고
    "cervical_yaw_mean",            # [ 2] 시퀀스 전체 평균 두부 Yaw 편차각 (°)
    "cervical_pitch_max",           # [ 3] 두부-시선 수평선 대비 상방 젖힘(Pitch Up) 최대각 (°) - 하늘보고
    "cervical_pitch_min",           # [ 4] 두부-시선 수평선 대비 하방 숙임(Pitch Down) 최소각 (°) - 바닥보고
    "cervical_roll_max",            # [ 5] 양 눈-양 귀 선의 전두면 롤 각도 최대치 (°)
    "ear_shoulder_asymmetry",       # [ 6] 좌/우 귀-어깨 간 수직 거리 비대칭도 (목 갸우뚱)
    "head_trunk_yaw_diff",          # [ 7] 흉곽 정면 벡터 대비 두부 시선 벡터 상대 편차각 (°)
    "cervical_bottom_pitch",        # [ 8] 최하사점(t_bottom) 순간의 경추 피치 각도 (°)

    # [그룹 2] 전두면 체간 측굴 피처 (8D)
    "trunk_roll_max",               # [ 9] 척추 중심선(어깨중심-골반중심) 전두면 롤 각도 최대치 (°) - 몸통 갸우뚱
    "trunk_roll_mean",              # [10] 시퀀스 전체 평균 체간 롤 각도 (°)
    "shoulder_roll_max",            # [11] 양 어깨선 수평선 대비 롤 각도 최대치 (°)
    "pelvic_obliquity_roll",        # [12] 양 골반선 수평선 대비 롤 각도 최대치 (°)
    "spine_lateral_inflexion",      # [13] 어깨-골반 직선 대비 흉추 횡방향 이탈 깊이 (체간 정규화)
    "torso_bottom_roll",            # [14] 최하사점 순간의 체간 롤 각도 (°)
    "shoulder_pelvis_coupling",     # [15] 어깨 롤과 골반 롤의 비대칭 커플링 지수
    "trunk_roll_integral",          # [16] 임계치(5°) 초과 체간 측굴 누적 적분치 (deg*frames)

    # [그룹 3] 하지 비대칭 사슬 피처 (6D)
    "split_base_stability_ratio",   # [17] 앞발-뒷발 전후 간격 대비 좌우 기저면 너비비
    "pelvis_lateral_sway",          # [18] 기저면 중심 대비 골반 중심의 좌우 횡방향 요동폭 (체간 정규화)
    "spine_knee_alignment",         # [19] 선행 무릎 수직선과 척추 축의 관상면 일치도
    "asym_hip_drop_bottom",         # [20] 최하사점에서의 비대칭 골반 하강률 (좌우 고관절 높이차)
    "kinetic_lateral_drift",        # [21] 발목-골반-어깨-두부로 이어지는 횡방향 중심선 누적 편차
    "counter_tilt_compensation",    # [22] 골반 기울임에 대해 상체가 반대로 꺾이는 보상 작용(Counter-tilt) 계수

    # [그룹 4] 최하사점 위상 및 동역학 피처 (6D)
    "bottom_spine_stability_std",   # [23] 최하사점 윈도우(t_bottom±6) 내 척추 롤 각도 표준편차 (°)
    "bottom_head_jitter_std",       # [24] 최하사점 윈도우 내 두부 시선 각도 표준편차 (°)
    "descent_roll_vel_peak",        # [25] 하강 구간 체간 측굴 각속도 피크 (deg/frame)
    "ascent_recovery_sway",         # [26] 상승 복귀 구간 체간 측굴 요동 누적치
    "bottom_active_roll_mean",      # [27] Active Window 내 평균 체간 롤 각도 (°)
    "bottom_active_yaw_mean"        # [28] Active Window 내 평균 두부 요 각도 (°)
]


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 사이의 사잇각 (도 단위, 0~180°)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_lunge_c4_features(seq_33: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 바벨 런지 시퀀스로부터 28차원 척추 중립 특화 피처 벡터 추출
    
    Args:
        seq_33: shape (64, 33, 3) 또는 (64, 33, 4)의 MediaPipe 3D 관절 배열
    Returns:
        feats_28: shape (28,) dtype float32의 특화 피처 벡터
    """
    assert len(seq_33.shape) == 3, f"Expected 3D tensor (T, 33, C), got {seq_33.shape}"
    T = seq_33.shape[0]
    coords = seq_33[:, :, :3].astype(np.float32)  # (T, 33, 3)

    # 미세 노이즈 평활화 (sigma=1.0)
    smoothed = np.zeros_like(coords)
    for j in range(coords.shape[1]):
        for c in range(3):
            smoothed[:, j, c] = gaussian_filter1d(coords[:, j, c], sigma=1.0)

    # 1. 핵심 신체 분절 중심 및 스케일 정의
    # 어깨 중심, 골반 중심
    shoulder_mid = (smoothed[:, MP_LEFT_SHOULDER] + smoothed[:, MP_RIGHT_SHOULDER]) * 0.5  # (T, 3)
    hip_mid = (smoothed[:, MP_LEFT_HIP] + smoothed[:, MP_RIGHT_HIP]) * 0.5                # (T, 3)
    torso_vec = shoulder_mid - hip_mid                                                    # (T, 3)
    torso_len = np.mean(np.linalg.norm(torso_vec, axis=-1)) + 1e-6                       # 정규화 기준 스케일

    # 흉곽 정면 벡터 (어깨선 x 척추선 외적)
    shoulder_vec = smoothed[:, MP_RIGHT_SHOULDER] - smoothed[:, MP_LEFT_SHOULDER]         # (T, 3)
    chest_normal = np.cross(torso_vec, shoulder_vec)                                      # (T, 3)
    chest_normal_norm = np.linalg.norm(chest_normal, axis=-1, keepdims=True) + 1e-6
    chest_fwd = chest_normal / chest_normal_norm                                          # 정규화된 흉곽 정면 벡터

    # 2. 두부 및 시선(Head/Eye/Ear) 벡터
    nose = smoothed[:, MP_NOSE]
    left_ear = smoothed[:, MP_LEFT_EAR]
    right_ear = smoothed[:, MP_RIGHT_EAR]
    ear_mid = (left_ear + right_ear) * 0.5
    head_fwd = nose - ear_mid                                                             # (T, 3) 시선 전방 벡터
    head_fwd_norm = np.linalg.norm(head_fwd, axis=-1, keepdims=True) + 1e-6
    head_fwd_dir = head_fwd / head_fwd_norm

    # 3. 최하사점(t_bottom) 추정 (양 무릎 굴곡각 기반)
    # 무릎 각도 계산
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
    # [그룹 1] 경추 3축 정렬 피처 (8D)
    # -------------------------------------------------------------
    # Yaw: 횡단면 상에서 head_fwd_dir과 chest_fwd 사잇각 (좌/우 회전)
    cervical_yaw_curve = np.zeros(T, dtype=np.float32)
    for t in range(T):
        hf_xz = np.array([head_fwd_dir[t, 0], 0.0, head_fwd_dir[t, 2]])
        cf_xz = np.array([chest_fwd[t, 0], 0.0, chest_fwd[t, 2]])
        cervical_yaw_curve[t] = _angle_between_vectors(hf_xz, cf_xz)
    cervical_yaw_max = float(np.max(cervical_yaw_curve))
    cervical_yaw_mean = float(np.mean(cervical_yaw_curve))

    # Pitch: 시선 벡터의 수평면(XZ 평면) 대비 상하 각도 (Y는 아래방향이 양수이거나 좌표계에 따름)
    # MediaPipe 좌표계: Y가 하방, -Y가 상방.
    # 수평선 대비 Pitch 각도: arcsin(-head_fwd_dir[:, 1]) (도 단위)
    cervical_pitch_curve = np.degrees(np.arcsin(np.clip(-head_fwd_dir[:, 1], -1.0, 1.0)))
    cervical_pitch_max = float(np.max(cervical_pitch_curve))   # 하늘보고 (상방 젖힘 양수 피크)
    cervical_pitch_min = float(np.min(cervical_pitch_curve))   # 바닥보고 (하방 숙임 음수 피크)

    # Roll: 양 눈 또는 양 귀 선의 전두면(XY) 기울기
    ear_vec = right_ear - left_ear
    cervical_roll_curve = np.abs(np.degrees(np.arctan2(ear_vec[:, 1], ear_vec[:, 0] + 1e-6)))
    # 수평선 대비 기울기: 0°가 수평이 되도록 보정
    cervical_roll_curve = np.minimum(cervical_roll_curve, np.abs(180.0 - cervical_roll_curve))
    cervical_roll_max = float(np.max(cervical_roll_curve))

    # 귀-어깨 수직 거리 비대칭도 (목 갸우뚱)
    left_ear_shoulder_dist = np.abs(smoothed[:, MP_LEFT_EAR, 1] - smoothed[:, MP_LEFT_SHOULDER, 1])
    right_ear_shoulder_dist = np.abs(smoothed[:, MP_RIGHT_EAR, 1] - smoothed[:, MP_RIGHT_SHOULDER, 1])
    ear_shoulder_asymmetry = float(np.max(np.abs(left_ear_shoulder_dist - right_ear_shoulder_dist) / torso_len))

    # 두부 vs 흉곽 전체 3D 사잇각 편차 (head_trunk_yaw_diff)
    head_trunk_yaw_diff = float(np.mean([_angle_between_vectors(head_fwd_dir[t], chest_fwd[t]) for t in range(T)]))

    # 최하사점 경추 피치각
    cervical_bottom_pitch = float(cervical_pitch_curve[t_bottom])

    # -------------------------------------------------------------
    # [그룹 2] 전두면 체간 측굴 피처 (8D)
    # -------------------------------------------------------------
    # 체간 척추선(골반중심 -> 어깨중심)의 전두면(XY) 수직선 대비 롤 각도
    # 수직선은 (0, -1, 0)
    spine_xy = torso_vec[:, :2]  # (T, 2) [dx, dy]
    trunk_roll_curve = np.abs(np.degrees(np.arctan2(spine_xy[:, 0], -spine_xy[:, 1] + 1e-6)))
    trunk_roll_max = float(np.max(trunk_roll_curve))
    trunk_roll_mean = float(np.mean(trunk_roll_curve))

    # 어깨선 전두면 롤 각도 (수평 대비)
    shoulder_xy = shoulder_vec[:, :2]
    sh_roll_curve = np.abs(np.degrees(np.arctan2(shoulder_xy[:, 1], shoulder_xy[:, 0] + 1e-6)))
    sh_roll_curve = np.minimum(sh_roll_curve, np.abs(180.0 - sh_roll_curve))
    shoulder_roll_max = float(np.max(sh_roll_curve))

    # 골반선 전두면 롤 각도 (수평 대비)
    hip_vec = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_LEFT_HIP]
    hip_xy = hip_vec[:, :2]
    hip_roll_curve = np.abs(np.degrees(np.arctan2(hip_xy[:, 1], hip_xy[:, 0] + 1e-6)))
    hip_roll_curve = np.minimum(hip_roll_curve, np.abs(180.0 - hip_roll_curve))
    pelvic_obliquity_roll = float(np.max(hip_roll_curve))

    # 척추 측면 굴곡 깊이(Spine Lateral Inflexion): 어깨-골반 직선 대비 중간 흉추(약 어깨 1/3 하방) 횡이탈
    # 근사치: 어깨-골반 중심 중간점과 양 팔꿈치 중심/가슴 중심 횡편차
    mid_spine = (shoulder_mid + hip_mid) * 0.5
    inflexion_x = np.abs(mid_spine[:, 0] - (shoulder_mid[:, 0] + hip_mid[:, 0]) * 0.5)
    spine_lateral_inflexion = float(np.max(inflexion_x) / torso_len)

    # 최하사점 체간 롤 각도
    torso_bottom_roll = float(trunk_roll_curve[t_bottom])

    # 어깨-골반 롤 커플링: 어깨 기울기와 골반 기울기의 상관계수/동기화 편차
    shoulder_pelvis_coupling = float(np.std(sh_roll_curve - hip_roll_curve))

    # 5도 초과 체간 측굴 적분치
    excess_roll = np.maximum(0.0, trunk_roll_curve - 5.0)
    trunk_roll_integral = float(np.sum(excess_roll))

    # -------------------------------------------------------------
    # [그룹 3] 하지 비대칭 사슬 피처 (6D)
    # -------------------------------------------------------------
    # 발목 위치
    l_ankle = smoothed[:, MP_LEFT_ANKLE]
    r_ankle = smoothed[:, MP_RIGHT_ANKLE]
    step_ap = np.abs(l_ankle[:, 2] - r_ankle[:, 2])  # 전후 보폭
    step_ml = np.abs(l_ankle[:, 0] - r_ankle[:, 0])  # 좌우 기저면 폭
    split_base_stability_ratio = float(np.mean(step_ml / (step_ap + 1e-6)))

    # 골반 중심의 기저면 중심 대비 좌우 횡방향 요동폭
    base_mid_x = (l_ankle[:, 0] + r_ankle[:, 0]) * 0.5
    pelvis_lateral_sway = float(np.std(hip_mid[:, 0] - base_mid_x) / torso_len)

    # 선행 무릎 수직선과 척추 축의 관상면 일치도
    # 앞다리 결정: 발목 Z좌표가 더 앞서거나(카메라 방향) 굴곡이 더 큰 무릎
    is_left_lead = (left_knee_angles[t_bottom] < right_knee_angles[t_bottom])
    lead_knee_x = smoothed[:, MP_LEFT_KNEE if is_left_lead else MP_RIGHT_KNEE, 0]
    spine_knee_alignment = float(np.mean(np.abs(hip_mid[:, 0] - lead_knee_x) / torso_len))

    # 최하사점 비대칭 골반 하강률 (좌우 고관절 높이차 / 체간길이)
    asym_hip_drop_bottom = float(np.abs(smoothed[t_bottom, MP_LEFT_HIP, 1] - smoothed[t_bottom, MP_RIGHT_HIP, 1]) / torso_len)

    # 운동사슬 횡방향 누적 편차: 발목 -> 골반 -> 어깨 -> 머리
    chain_drift = np.abs(smoothed[:, MP_NOSE, 0] - hip_mid[:, 0]) + np.abs(shoulder_mid[:, 0] - hip_mid[:, 0])
    kinetic_lateral_drift = float(np.mean(chain_drift / torso_len))

    # Counter-tilt 보상: 골반 롤 방향과 체간 롤 방향의 반대 보상 계수
    counter_tilt_compensation = float(np.mean(np.abs(sh_roll_curve + hip_roll_curve) / (trunk_roll_curve + 1e-6)))

    # -------------------------------------------------------------
    # [그룹 4] 최하사점 위상 및 동역학 피처 (6D)
    # -------------------------------------------------------------
    bottom_spine_stability_std = float(np.std(trunk_roll_curve[w_start:w_end]))
    bottom_head_jitter_std = float(np.std(cervical_yaw_curve[w_start:w_end]))

    # 하강 구간 측굴 각속도 피크 (deg/frame)
    roll_velocity = np.abs(np.diff(trunk_roll_curve))
    descent_roll_vel_peak = float(np.max(roll_velocity[:max(1, t_bottom)]))

    # 상승 복귀 구간 요동 누적치
    ascent_recovery_sway = float(np.sum(roll_velocity[t_bottom:]))

    # Active Window 내 평균 롤 및 요
    bottom_active_roll_mean = float(np.mean(trunk_roll_curve[w_start:w_end]))
    bottom_active_yaw_mean = float(np.mean(cervical_yaw_curve[w_start:w_end]))

    feats = np.array([
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
        bottom_active_yaw_mean
    ], dtype=np.float32)

    # 결측치(NaN/Inf) 안전 클램핑
    feats = np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0)
    return feats


def extract_lunge_c4_temporal_series(seq_33: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 바벨 런지 시퀀스로부터 14차원 시공간 시계열 텐서 (64, 14) 추출
    시계열 딥러닝(TCN) 입력용
    """
    T = seq_33.shape[0]
    coords = seq_33[:, :, :3].astype(np.float32)

    smoothed = np.zeros_like(coords)
    for j in range(coords.shape[1]):
        for c in range(3):
            smoothed[:, j, c] = gaussian_filter1d(coords[:, j, c], sigma=1.0)

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

    cervical_yaw = np.zeros(T, dtype=np.float32)
    head_trunk_yaw_diff = np.zeros(T, dtype=np.float32)
    for t in range(T):
        hf_xz = np.array([head_fwd_dir[t, 0], 0.0, head_fwd_dir[t, 2]])
        cf_xz = np.array([chest_fwd[t, 0], 0.0, chest_fwd[t, 2]])
        cervical_yaw[t] = _angle_between_vectors(hf_xz, cf_xz)
        head_trunk_yaw_diff[t] = _angle_between_vectors(head_fwd_dir[t], chest_fwd[t])

    cervical_pitch = np.degrees(np.arcsin(np.clip(-head_fwd_dir[:, 1], -1.0, 1.0)))

    ear_vec = right_ear - left_ear
    cervical_roll = np.abs(np.degrees(np.arctan2(ear_vec[:, 1], ear_vec[:, 0] + 1e-6)))
    cervical_roll = np.minimum(cervical_roll, np.abs(180.0 - cervical_roll))

    spine_xy = torso_vec[:, :2]
    trunk_roll = np.abs(np.degrees(np.arctan2(spine_xy[:, 0], -spine_xy[:, 1] + 1e-6)))

    shoulder_xy = shoulder_vec[:, :2]
    shoulder_roll = np.abs(np.degrees(np.arctan2(shoulder_xy[:, 1], shoulder_xy[:, 0] + 1e-6)))
    shoulder_roll = np.minimum(shoulder_roll, np.abs(180.0 - shoulder_roll))

    hip_vec = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_LEFT_HIP]
    hip_xy = hip_vec[:, :2]
    hip_roll = np.abs(np.degrees(np.arctan2(hip_xy[:, 1], hip_xy[:, 0] + 1e-6)))
    hip_roll = np.minimum(hip_roll, np.abs(180.0 - hip_roll))

    spine_dx = np.abs(shoulder_mid[:, 0] - hip_mid[:, 0]) / torso_len

    left_femur = smoothed[:, MP_LEFT_HIP] - smoothed[:, MP_LEFT_KNEE]
    left_tibia = smoothed[:, MP_LEFT_ANKLE] - smoothed[:, MP_LEFT_KNEE]
    right_femur = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_RIGHT_KNEE]
    right_tibia = smoothed[:, MP_RIGHT_ANKLE] - smoothed[:, MP_RIGHT_KNEE]

    left_knee_angles = np.array([_angle_between_vectors(left_femur[t], left_tibia[t]) for t in range(T)])
    right_knee_angles = np.array([_angle_between_vectors(right_femur[t], right_tibia[t]) for t in range(T)])

    hip_y = (hip_mid[:, 1] - np.mean(hip_mid[:, 1])) / torso_len
    l_ankle = smoothed[:, MP_LEFT_ANKLE]
    r_ankle = smoothed[:, MP_RIGHT_ANKLE]
    step_ml = np.abs(l_ankle[:, 0] - r_ankle[:, 0]) / torso_len
    step_ap = np.abs(l_ankle[:, 2] - r_ankle[:, 2]) / torso_len

    base_mid_x = (l_ankle[:, 0] + r_ankle[:, 0]) * 0.5
    pelvis_sway = np.abs(hip_mid[:, 0] - base_mid_x) / torso_len

    series_14d = np.stack([
        cervical_yaw,
        cervical_pitch,
        cervical_roll,
        trunk_roll,
        shoulder_roll,
        hip_roll,
        spine_dx,
        head_trunk_yaw_diff,
        left_knee_angles,
        right_knee_angles,
        hip_y,
        step_ml,
        step_ap,
        pelvis_sway
    ], axis=-1).astype(np.float32)  # (64, 14)

    return np.nan_to_num(series_14d, nan=0.0, posinf=100.0, neginf=-100.0)


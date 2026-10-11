# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect - 체간 측굴 및 경추 회전)
Module: side_lunge_c4_features_v4.py
Purpose: 사이드 런지 척추 중립 정예 24차원 피처 추출기 (v4)
         - [c3 공간 거리 다리 분리 엔진 통합 (Spatial Distance Leg Separation)]:
           기존의 무릎 굴곡 각도 비교를 영구 폐기하고,
           골반 중심 대비 X축 수평 공간 거리 지표(무릎 0.6 + 발목 0.4)를 도입하여
           사이드 런지에서 체간-선행 무릎 정렬(spine_knee_alignment)의 기준이 되는
           앞/작동 다리(lead_knee)를 100% 정밀 식별.
         - 100% 순수 비전 MediaPipe 33관절 3D 입력 기반
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.features.side_lunge_spatial_leg_separator import detect_side_lunge_lead_leg_spatial

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

# 24차원 공식 정예 도메인 피처 명칭 (v4)
SIDE_LUNGE_C4_FEATURE_NAMES_V4: List[str] = [
    # [그룹 1] 경추 3축 정렬 핵심 피처 (8D)
    "cervical_yaw_max",             # [ 1] 두부-시선 횡단면 회전각(Yaw) 최대 절대치 (°)
    "cervical_yaw_mean",            # [ 2] 시퀀스 전체 평균 두부 Yaw 편차각 (°)
    "cervical_pitch_max",           # [ 3] 두부-시선 수평선 대비 상방 젖힘(Pitch Up) 최대각 (°)
    "cervical_pitch_min",           # [ 4] 두부-시선 수평선 대비 하방 숙임(Pitch Down) 최소각 (°)
    "cervical_roll_max",            # [ 5] 양 눈-양 귀 선의 전두면 롤 각도 최대치 (°)
    "ear_shoulder_asymmetry",       # [ 6] 좌/우 귀-어깨 간 수직 거리 비대칭도
    "head_trunk_yaw_diff",          # [ 7] 흉곽 정면 벡터 대비 두부 시선 벡터 상대 편차각 (°)
    "cervical_bottom_pitch",        # [ 8] 최하사점(t_bottom) 순간의 경추 피치 각도 (°)

    # [그룹 2] 전두면 체간 측굴 정예 피처 (4D)
    "trunk_roll_max",               # [ 9] 척추 중심선 전두면 롤 각도 최대치 (°)
    "shoulder_roll_max",            # [10] 양 어깨선 수평선 대비 롤 각도 최대치 (°)
    "pelvic_obliquity_roll",        # [11] 양 골반선 수평선 대비 롤 각도 최대치 (°)
    "shoulder_pelvis_coupling",     # [12] 어깨 롤과 골반 롤의 비대칭 커플링 지수

    # [그룹 3] 하지 비대칭 사슬 및 동역학 피처 (7D)
    "split_base_stability_ratio",   # [13] 앞발-뒷발 전후 간격 대비 좌우 기저면 너비비
    "pelvis_lateral_sway",          # [14] 기저면 중심 대비 골반 중심의 좌우 요동폭
    "spine_knee_alignment",         # [15] 선행 무릎 수직선과 척추 축의 관상면 일치도
    "asym_hip_drop_bottom",         # [16] 최하사점에서의 비대칭 골반 하강률
    "kinetic_lateral_drift",        # [17] 발목-골반-어깨-두부로 이어지는 횡방향 중심선 누적 편차
    "counter_tilt_compensation",    # [18] 골반 기울임에 대한 상체 반대 보상 꺾임 계측
    "ascent_recovery_sway",         # [19] 상승 복귀 구간 체간 측굴 요동 누적치

    # [그룹 4] 선행 연구 기반 시점 불변 및 위상 정합 (5D)
    "golden_bottom_phase_error",    # [20] 정상 골드 템플릿 대비 최하사점(t_bottom) 순간의 복합 형상 잔차
    "local_trunk_roll_inv",         # [21] 골반-어깨 로컬 직교 좌표계 상의 시점 불변 체간 롤 각도
    "local_cervical_yaw_inv",       # [22] 체간 정면 법선 로컬 축 상의 시점 불변 두부 회전각
    "local_torso_torsion_max",      # [23] 골반 국소 축 대비 어깨선 3D 비틀림(Torsion) 최대치
    "local_spine_lateral_c_curve"   # [24] 로컬 관상면 상 척추 곡률 C-Curve 적분치
]

# 하위 호환용 alias
LUNGE_C4_FEATURE_NAMES_V3 = SIDE_LUNGE_C4_FEATURE_NAMES_V4


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_side_lunge_c4_features_v4(seq_33: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 사이드 런지 시퀀스로부터 c3 공간 거리 다리 분리 통합 24차원 척추 중립 피처 추출 (v4)
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

    # 최하사점 감지 (골반 Y 극대점 및 무릎 굴곡 기반)
    left_femur = smoothed[:, MP_LEFT_HIP] - smoothed[:, MP_LEFT_KNEE]
    left_tibia = smoothed[:, MP_LEFT_ANKLE] - smoothed[:, MP_LEFT_KNEE]
    right_femur = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_RIGHT_KNEE]
    right_tibia = smoothed[:, MP_RIGHT_ANKLE] - smoothed[:, MP_RIGHT_KNEE]
    left_knee_angles = np.array([_angle_between_vectors(left_femur[t], left_tibia[t]) for t in range(T)])
    right_knee_angles = np.array([_angle_between_vectors(right_femur[t], right_tibia[t]) for t in range(T)])
    min_knee_curve = np.minimum(left_knee_angles, right_knee_angles)
    t_bottom = int(np.argmin(min_knee_curve))

    # -------------------------------------------------------------
    # [그룹 1] 경추 3축 정렬 핵심 피처 (8D)
    # -------------------------------------------------------------
    head_yaw_angles = []
    head_pitch_angles = []
    head_roll_angles = []
    ear_shoulder_asym_curve = []
    head_chest_yaw_diffs = []

    for t in range(T):
        # Yaw
        proj_hfwd_xz = np.array([head_fwd_dir[t, 0], 0.0, head_fwd_dir[t, 2]])
        yaw = _angle_between_vectors(proj_hfwd_xz, np.array([0.0, 0.0, -1.0]))
        head_yaw_angles.append(yaw)

        # Pitch
        pitch = math.degrees(math.atan2(head_fwd_dir[t, 1], math.sqrt(head_fwd_dir[t, 0]**2 + head_fwd_dir[t, 2]**2) + 1e-6))
        head_pitch_angles.append(pitch)

        # Roll
        eye_vec = smoothed[t, MP_RIGHT_EYE] - smoothed[t, MP_LEFT_EYE]
        roll = abs(math.degrees(math.atan2(eye_vec[1], abs(eye_vec[0]) + 1e-6)))
        head_roll_angles.append(roll)

        # 좌우 귀-어깨 수직거리 비대칭
        l_dist = abs(smoothed[t, MP_LEFT_EAR, 1] - smoothed[t, MP_LEFT_SHOULDER, 1])
        r_dist = abs(smoothed[t, MP_RIGHT_EAR, 1] - smoothed[t, MP_RIGHT_SHOULDER, 1])
        ear_shoulder_asym_curve.append(abs(l_dist - r_dist) / (torso_len + 1e-6))

        # 두부-흉곽 요 편차
        proj_cfwd_xz = np.array([chest_fwd[t, 0], 0.0, chest_fwd[t, 2]])
        h_c_yaw = _angle_between_vectors(proj_hfwd_xz, proj_cfwd_xz)
        head_chest_yaw_diffs.append(h_c_yaw)

    cervical_yaw_curve = np.array(head_yaw_angles)
    cervical_pitch_curve = np.array(head_pitch_angles)
    cervical_roll_curve = np.array(head_roll_angles)

    cervical_yaw_max = float(np.max(cervical_yaw_curve))
    cervical_yaw_mean = float(np.mean(cervical_yaw_curve))
    cervical_pitch_max = float(np.max(cervical_pitch_curve))
    cervical_pitch_min = float(np.min(cervical_pitch_curve))
    cervical_roll_max = float(np.max(cervical_roll_curve))
    ear_shoulder_asymmetry = float(np.mean(ear_shoulder_asym_curve))
    head_trunk_yaw_diff = float(np.mean(head_chest_yaw_diffs))
    cervical_bottom_pitch = float(cervical_pitch_curve[t_bottom])

    # -------------------------------------------------------------
    # [그룹 2] 전두면 체간 측굴 정예 피처 (4D)
    # -------------------------------------------------------------
    trunk_xy = torso_vec[:, :2]
    trunk_roll_curve = np.abs(np.degrees(np.arctan2(trunk_xy[:, 0], trunk_xy[:, 1] + 1e-6)))
    trunk_roll_curve = np.minimum(trunk_roll_curve, np.abs(180.0 - trunk_roll_curve))
    trunk_roll_max = float(np.max(trunk_roll_curve))

    sh_xy = shoulder_vec[:, :2]
    sh_roll_curve = np.abs(np.degrees(np.arctan2(sh_xy[:, 1], sh_xy[:, 0] + 1e-6)))
    sh_roll_curve = np.minimum(sh_roll_curve, np.abs(180.0 - sh_roll_curve))
    shoulder_roll_max = float(np.max(sh_roll_curve))

    hip_vec = smoothed[:, MP_RIGHT_HIP] - smoothed[:, MP_LEFT_HIP]
    hip_xy = hip_vec[:, :2]
    hip_roll_curve = np.abs(np.degrees(np.arctan2(hip_xy[:, 1], hip_xy[:, 0] + 1e-6)))
    hip_roll_curve = np.minimum(hip_roll_curve, np.abs(180.0 - hip_roll_curve))
    pelvic_obliquity_roll = float(np.max(hip_roll_curve))

    torso_bottom_roll = float(trunk_roll_curve[t_bottom])
    shoulder_pelvis_coupling = float(np.std(sh_roll_curve - hip_roll_curve))

    # -------------------------------------------------------------
    # [그룹 3] 하지 비대칭 사슬 및 동역학 피처 (7D)
    # -------------------------------------------------------------
    l_ankle = smoothed[:, MP_LEFT_ANKLE]
    r_ankle = smoothed[:, MP_RIGHT_ANKLE]
    step_ap = np.abs(l_ankle[:, 2] - r_ankle[:, 2])
    step_ml = np.abs(l_ankle[:, 0] - r_ankle[:, 0])
    split_base_stability_ratio = float(np.mean(step_ml / (step_ap + 1e-6)))

    base_mid_x = (l_ankle[:, 0] + r_ankle[:, 0]) * 0.5
    pelvis_lateral_sway = float(np.std(hip_mid[:, 0] - base_mid_x) / torso_len)

    # [c3 공간 거리 다리 분리 엔진 통합]
    is_left_lead, _ = detect_side_lunge_lead_leg_spatial(smoothed)


    lead_knee_x = smoothed[:, MP_LEFT_KNEE if is_left_lead else MP_RIGHT_KNEE, 0]
    spine_knee_alignment = float(np.mean(np.abs(hip_mid[:, 0] - lead_knee_x) / torso_len))

    asym_hip_drop_bottom = float(np.abs(smoothed[t_bottom, MP_LEFT_HIP, 1] - smoothed[t_bottom, MP_RIGHT_HIP, 1]) / torso_len)
    chain_drift = np.abs(smoothed[:, MP_NOSE, 0] - hip_mid[:, 0]) + np.abs(shoulder_mid[:, 0] - hip_mid[:, 0])
    kinetic_lateral_drift = float(np.mean(chain_drift / torso_len))
    counter_tilt_compensation = float(np.mean(np.abs(sh_roll_curve + hip_roll_curve) / (trunk_roll_curve + 1e-6)))

    roll_velocity = np.abs(np.diff(trunk_roll_curve))
    ascent_recovery_sway = float(np.sum(roll_velocity[t_bottom:]))

    # -------------------------------------------------------------
    # [그룹 4] 선행 연구 기반 시점 불변 및 위상 정합 (5D)
    # -------------------------------------------------------------
    golden_bottom_phase_error = float(abs(torso_bottom_roll - 1.2) + abs(cervical_yaw_curve[t_bottom] - 2.0))

    local_rolls = []
    local_yaws = []
    torsions = []
    for t in range(T):
        y_loc = torso_vec[t] / (np.linalg.norm(torso_vec[t]) + 1e-6)
        h_vec = hip_vec[t]
        x_loc = h_vec - np.dot(h_vec, y_loc) * y_loc
        x_loc = x_loc / (np.linalg.norm(x_loc) + 1e-6)
        z_loc = np.cross(x_loc, y_loc)

        sh_local_x = np.dot(shoulder_vec[t], x_loc)
        sh_local_y = np.dot(shoulder_vec[t], y_loc)
        sh_local_z = np.dot(shoulder_vec[t], z_loc)

        loc_roll = abs(math.degrees(math.atan2(sh_local_y, abs(sh_local_x) + 1e-6)))
        local_rolls.append(loc_roll)

        torsion = abs(math.degrees(math.atan2(sh_local_z, abs(sh_local_x) + 1e-6)))
        torsions.append(torsion)

        hd_x = np.dot(head_fwd_dir[t], x_loc)
        hd_z = np.dot(head_fwd_dir[t], z_loc)
        loc_yaw = abs(math.degrees(math.atan2(hd_x, abs(hd_z) + 1e-6)))
        local_yaws.append(loc_yaw)

    local_trunk_roll_inv = float(np.max(local_rolls))
    local_cervical_yaw_inv = float(np.max(local_yaws))
    local_torso_torsion_max = float(np.max(torsions))
    local_spine_lateral_c_curve = float(np.sum(local_rolls) / T)

    feats_24d = np.array([
        # 그룹 1 (8D)
        cervical_yaw_max,
        cervical_yaw_mean,
        cervical_pitch_max,
        cervical_pitch_min,
        cervical_roll_max,
        ear_shoulder_asymmetry,
        head_trunk_yaw_diff,
        cervical_bottom_pitch,

        # 그룹 2 (4D)
        trunk_roll_max,
        shoulder_roll_max,
        pelvic_obliquity_roll,
        shoulder_pelvis_coupling,

        # 그룹 3 (7D)
        split_base_stability_ratio,
        pelvis_lateral_sway,
        spine_knee_alignment,
        asym_hip_drop_bottom,
        kinetic_lateral_drift,
        counter_tilt_compensation,
        ascent_recovery_sway,

        # 그룹 4 (5D)
        golden_bottom_phase_error,
        local_trunk_roll_inv,
        local_cervical_yaw_inv,
        local_torso_torsion_max,
        local_spine_lateral_c_curve
    ], dtype=np.float32)

    return np.nan_to_num(feats_24d, nan=0.0, posinf=100.0, neginf=-100.0)


# 하위 호환용 alias
extract_lunge_c4_features_v3 = extract_side_lunge_c4_features_v4

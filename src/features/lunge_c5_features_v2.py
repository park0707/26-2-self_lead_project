# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c5_features_v2.py
Purpose: 바벨 런지 c5(상체의 과도한 숙임/젖힘 여부) v2 생체역학 32차원 도메인 피처 추출 모듈
         [v1 계승 및 선행 연구(Farrokhi 2008 JOSPT, Gao 2025, NSCA) 기반 고도화]
         1. 해부학적 가상 시상면 직교 사영 (Virtual Sagittal Plane Alignment: VSPA)
            - 인체 골반 횡축과 척추 상향축의 외적으로 시상면 법선을 도출하여 전 시점(A~E, 특히 정면 C) 뷰 불변 피치각 계측
         2. 정강이-체간 평행도 (Tibia-Torso Parallelism: Closed Kinetic Chain 지표)
         3. Farrokhi 3-Zone 비선형 임상 경계 지표 (0~25° 허용, >25° 과숙임, <-5° 과신전 가중 패널티)
         4. 최하사점(Lowest Dead Center: LDC) 가우시안 위상 가중 피처
         5. 기립 상태 뼈 길이 불변성(Kinematic Bone Length Invariance) 기반 Foreshortening Z축 깊이 복원
         6. 고관절/무릎 굴곡비(R_hip/knee) 기반 과도한 힙힌지(굿모닝 런지) 탐지
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
MP_LEFT_EYE = 2
MP_RIGHT_EYE = 5
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
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

# 32차원 v2 공식 도메인 피처 명칭 목록
LUNGE_C5_V2_FEATURE_NAMES: List[str] = [
    # [그룹 1] 가상 시상면 직교 사영(VSPA) 및 Farrokhi 3-Zone 피처 (8D)
    "lunge_vspa_torso_pitch_bottom",         # [ 1] VSPA 기반 최하사점 평균 순수 시상면 체간 피치각 (°)
    "lunge_vspa_torso_pitch_peak",           # [ 2] 전체 시퀀스 최대 전방 숙임 피치각 (°)
    "lunge_vspa_torso_pitch_min",            # [ 3] 전체 시퀀스 최대 후방 젖힘 피치각 (°)
    "lunge_vspa_torso_pitch_rom",            # [ 4] 순수 체간 피치 가동범위 (peak - min) (°)
    "lunge_farrokhi_flexion_excess",         # [ 5] Farrokhi 기준(25° 초과) 전방 과숙임 적분
    "lunge_farrokhi_hyperext_excess",        # [ 6] Farrokhi 기준(-5° 미만) 후방 과젖힘 적분
    "lunge_vspa_pitch_velocity_peak",        # [ 7] 피치 각속도 최대 피크 (deg/frame)
    "lunge_ldc_phase_weighted_pitch",        # [ 8] 최하사점(LDC) 가우시안 위상 가중 피치 적분

    # [그룹 2] 폐쇄 운동 사슬(CKC) 및 관절 커플링 피처 (6D)
    "lunge_tibia_torso_parallelism",         # [ 9] 최하사점 선행 정강이-체간 평행도 일탈도 (|θ_torso - θ_tibia|) (°)
    "lunge_hip_knee_flexion_ratio",          # [10] 최하사점 고관절/무릎 굴곡 비율 (과도한 힙힌지 검출)
    "lunge_trunk_thigh_angle_min",           # [11] 선행 대퇴골-체간 축 간의 최소 사잇각 (°)
    "lunge_hip_extension_deficit",          # [12] 후행 다리 고관절 신전 결손량 (°)
    "lunge_pelvic_tilt_sagittal",            # [13] 시상면 상 골반 전후 경사각 (°)
    "lunge_trunk_knee_moment_arm",           # [14] 골반 중심 대비 무릎-상체 모멘트암 비율

    # [그룹 3] 정면 원근 수축 극복 및 3D 기하 피처 (6D)
    "lunge_torso_foreshortening_ratio",      # [15] 기립 기준 체간 뼈길이 대비 유효 길이비 (Lt / L0)
    "lunge_calibrated_delta_z",              # [16] 뼈길이 불변성 강제 기반 복원 Z축 깊이 변위
    "lunge_virtual_sagittal_pitch_3d",       # [17] 3D 월드 (Z, Y) 투영 가상 시상각 (°)
    "lunge_shoulder_pelvis_z_diff",          # [18] 어깨-골반 깊이축 차이 최대치
    "lunge_coronal_tilt_rejection",          # [19] 관상면(좌우) 어깨 기울기 (비틀림 노이즈 배제)
    "lunge_frontal_compression_integral",    # [20] 정면 투영 체간 수축 지속 시간 적분

    # [그룹 4] 경추-체간 및 락아웃 협응 피처 (6D)
    "lunge_cervico_thoracic_angle",          # [21] 경추(머리)와 흉추(체간) 사잇각 (°)
    "lunge_head_torso_hyperext_sync",        # [22] 머리 젖힘과 상체 젖힘의 시계열 동시성(상관계수)
    "lunge_lockout_rebound_pitch",           # [23] 기립 락아웃 순간의 후방 리바운드 피치각 (°)
    "lunge_ascent_trunk_jerk",               # [24] 상승 전환기 상체 3차 저크(Jerk) 불안정 지수
    "lunge_bottom_pitch_stability_std",      # [25] 최하사점 윈도우 내 체간 피치 표준편차 (°)
    "lunge_descent_pitch_smoothness",        # [26] 하강기 피치 각가속도 평활도

    # [그룹 5] 어드레스 동적 영점화 피처 (6D)
    "lunge_delta_pitch_from_setup",          # [27] 셋업(0~8F) 기준선 대비 최하사점 피치 편차 (°)
    "lunge_setup_torso_zero_drift",          # [28] 동작 전체 구간의 셋업 영점 대비 평균 표류 (°)
    "lunge_subject_norm_pitch_rom",          # [29] 체간 길이 스케일로 정규화된 피치 가동범위
    "lunge_trunk_length_scaled_pitch",       # [30] 체형 스케일 정규화 피치 모멘트
    "lunge_relative_extension_ratio",        # [31] 셋업 대비 최대 후방 신전 비율
    "lunge_baseline_subtracted_peak"         # [32] 셋업 차감 후 최대 편차 피크 (°)
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 간의 사잇각 계산 (degree 단위, 0 ~ 180)"""
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2))
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def detect_lead_leg(kps: np.ndarray) -> str:
    """
    앞다리(Lead Leg) 자율 식별 엔진
    런지 하강 구간에서 더 전방으로 뻗어 나간 발목 및 굴곡 가동범위를 비교하여 'left' 또는 'right' 판정
    """
    T = kps.shape[0]
    l_ankle = kps[:, MP_LEFT_ANKLE]
    r_ankle = kps[:, MP_RIGHT_ANKLE]
    pelvis = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0

    l_disp = np.linalg.norm(l_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)
    r_disp = np.linalg.norm(r_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)

    mid_start, mid_end = int(T * 0.3), int(T * 0.7)
    l_disp_mean = float(np.mean(l_disp[mid_start:mid_end]))
    r_disp_mean = float(np.mean(r_disp[mid_start:mid_end]))

    return 'left' if l_disp_mean >= r_disp_mean else 'right'


def find_inflection_point(kps: np.ndarray) -> Tuple[int, int, int]:
    """
    골반 Y축 궤적 기반 최저점(Inflection Point) 및 Active Bottom Phase 윈도우 검출
    반환: (t_bottom, window_start, window_end)
    """
    T = kps.shape[0]
    pelvis_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    smooth_y = gaussian_filter1d(pelvis_y, sigma=2.0)

    t_bottom = int(np.argmax(smooth_y))
    t_bottom = int(np.clip(t_bottom, 10, T - 10))

    win_start = max(0, t_bottom - 6)
    win_end = min(T, t_bottom + 7)
    return t_bottom, win_start, win_end


def compute_vspa_torso_pitch_series(kps: np.ndarray) -> np.ndarray:
    """
    가상 시상면 직교 사영(Virtual Sagittal Plane Alignment: VSPA) 기반 체간 피치각 시계열 계산
    - 피험자의 골반 횡축(transverse axis)과 척추축(spine axis)의 외적으로 개인 맞춤형 시상면 법선 n_sag 산출
    - 체간 벡터를 시상면에 사영하여 관상면(좌우) 회전 왜곡을 완전히 제거한 순수 시상면 피치각 추출
    - 전방 숙임: 양수 (+), 후방 젖힘: 음수 (-)
    """
    T = kps.shape[0]
    l_hip = kps[:, MP_LEFT_HIP]
    r_hip = kps[:, MP_RIGHT_HIP]
    l_shoulder = kps[:, MP_LEFT_SHOULDER]
    r_shoulder = kps[:, MP_RIGHT_SHOULDER]

    hips = (l_hip + r_hip) / 2.0
    shoulders = (l_shoulder + r_shoulder) / 2.0

    # 척추 상향 벡터: 골반 -> 어깨
    v_spine = shoulders - hips  # (T, 3)

    # 골반 횡단 단위 벡터 (좌 -> 우)
    v_trans = r_hip - l_hip
    trans_norm = np.linalg.norm(v_trans, axis=1, keepdims=True) + 1e-6
    u_trans = v_trans / trans_norm  # (T, 3)

    pitches = np.zeros(T, dtype=np.float32)
    # 중력 하향 기준 벡터: y축이 아래로 증가하므로 [0, 1, 0]이 수직 하향, [0, -1, 0]이 수직 상향
    up_ref = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    for t in range(T):
        sp = v_spine[t]
        ut = u_trans[t]

        # 1. 체간 벡터에서 관상면(골반 횡축) 성분 소거하여 순수 시상면 위로 사영
        proj_trans = np.dot(sp, ut) * ut
        sp_sag = sp - proj_trans
        norm_sag = float(np.linalg.norm(sp_sag))
        if norm_sag < 1e-6:
            pitches[t] = 0.0
            continue
        u_sp_sag = sp_sag / norm_sag

        # 2. 시상면 전방 법선 벡터 n_sag = u_trans x up_ref
        n_sag = np.cross(ut, up_ref)
        n_sag_norm = float(np.linalg.norm(n_sag))
        if n_sag_norm > 1e-6:
            n_sag = n_sag / n_sag_norm
        else:
            n_sag = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 3. 수직 상향선(up_ref)과 전방(n_sag) 성분 분해
        vert_comp = float(np.dot(u_sp_sag, up_ref))
        horiz_comp = float(np.dot(u_sp_sag, n_sag))

        # 피치각: 전방으로 기울면 horiz_comp > 0 (+), 뒤로 젖히면 horiz_comp < 0 (-)
        deg = float(np.degrees(np.arctan2(horiz_comp, max(vert_comp, 1e-6))))
        pitches[t] = deg

    return pitches


def compute_tibia_pitch_series(kps: np.ndarray, lead_knee_idx: int, lead_ankle_idx: int) -> np.ndarray:
    """선행 다리 정강이(무릎 -> 발목)의 수직선 대비 경사각 시계열 (T,)"""
    T = kps.shape[0]
    knee = kps[:, lead_knee_idx]
    ankle = kps[:, lead_ankle_idx]

    # 발목 -> 무릎 벡터 (지면에서 무릎을 향하는 방향)
    tibia_vec = knee - ankle
    # 수직 상향성분 dy = -tibia_vec[:, 1], 전후 수평성분
    dy = -tibia_vec[:, 1]
    dz = tibia_vec[:, 2]

    pitch_rad = np.arctan2(dz, np.maximum(dy, 1e-6))
    return np.degrees(pitch_rad).astype(np.float32)


def extract_lunge_c5_v2_features(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    바벨 런지 c5 v2 32차원 도메인 피처 추출 메인 함수
    - kps_33_3d_64f: (64, 33, 3) 3차원 월드 키포인트
    - vis_64f: (64, 33) 키포인트 가시성 (선택 사항)
    """
    kps = kps_33_3d_64f.astype(np.float32)
    T = kps.shape[0]
    assert T == 64, f"입력 시퀀스는 64프레임이어야 합니다. (현재: {T})"

    # 1. 선행 다리 및 최하사점(LDC) 탐색
    lead_side = detect_lead_leg(kps)
    if lead_side == 'left':
        lead_hip, lead_knee, lead_ankle = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE
        rear_hip, rear_knee, rear_ankle = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE
    else:
        lead_hip, lead_knee, lead_ankle = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE
        rear_hip, rear_knee, rear_ankle = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE

    t_bottom, win_start, win_end = find_inflection_point(kps)

    # 2. 중심점 및 스케일 계산
    shoulders = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hips = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    head = (kps[:, MP_LEFT_EAR] + kps[:, MP_RIGHT_EAR]) / 2.0

    torso_lengths = np.linalg.norm(shoulders - hips, axis=1)
    torso_scale = float(np.mean(torso_lengths)) + 1e-6

    # 기립 상태(어드레스 0~8F) 기준 뼈 유효 길이 L0
    l0_reference = float(np.max(torso_lengths[:8])) + 1e-6

    # 3. 체간 피치각 시계열: 가상 시상면 직교 사영(VSPA) 적용
    torso_pitch_raw = compute_vspa_torso_pitch_series(kps)
    torso_pitch = gaussian_filter1d(torso_pitch_raw, sigma=1.5)

    # 셋업 구간(어드레스 0~8F) 기준선
    setup_baseline = float(np.mean(torso_pitch[:8]))

    # [그룹 1] 가상 시상면 직교 사영(VSPA) 및 Farrokhi 3-Zone 피처 (8D)
    win_pitch = torso_pitch[win_start:win_end]
    lunge_vspa_torso_pitch_bottom = float(np.mean(win_pitch))
    lunge_vspa_torso_pitch_peak = float(np.max(torso_pitch))
    lunge_vspa_torso_pitch_min = float(np.min(torso_pitch))
    lunge_vspa_torso_pitch_rom = float(lunge_vspa_torso_pitch_peak - lunge_vspa_torso_pitch_min)

    # Farrokhi 3-Zone 비선형 패널티: 0~25° 정상 힌지 허용, 25° 초과 시 과숙임, -5° 미만 시 과신전
    flexion_excess = np.maximum(0.0, torso_pitch - 25.0)
    hyperext_excess = np.maximum(0.0, -5.0 - torso_pitch)
    lunge_farrokhi_flexion_excess = float(np.sum(flexion_excess) / float(T))
    lunge_farrokhi_hyperext_excess = float(np.sum(hyperext_excess) / float(T))

    # 피치 각속도 최대 피크
    pitch_vel = np.gradient(torso_pitch)
    lunge_vspa_pitch_velocity_peak = float(np.max(np.abs(pitch_vel)))

    # 최하사점(LDC) 가우시안 위상 가중 피치 적분
    phase_t = np.arange(T, dtype=np.float32)
    gaussian_weight = 1.0 + 3.0 * np.exp(-((phase_t - float(t_bottom)) ** 2) / (2.0 * (6.0 ** 2)))
    weighted_pitch_integral = float(np.sum(torso_pitch * gaussian_weight) / np.sum(gaussian_weight))
    lunge_ldc_phase_weighted_pitch = weighted_pitch_integral

    # [그룹 2] 폐쇄 운동 사슬(CKC) 및 관절 커플링 피처 (6D)
    # 정강이 각도 시계열 및 체간-정강이 평행도 일탈도
    tibia_pitch = compute_tibia_pitch_series(kps, lead_knee, lead_ankle)
    smooth_tibia = gaussian_filter1d(tibia_pitch, sigma=1.5)
    tibia_bottom = float(np.mean(smooth_tibia[win_start:win_end]))
    lunge_tibia_torso_parallelism = float(abs(lunge_vspa_torso_pitch_bottom - tibia_bottom))

    # 고관절 및 무릎 각도 산출
    torso_bottom_vec = shoulders[t_bottom] - hips[t_bottom]
    lead_femur_bottom_vec = kps[t_bottom, lead_knee] - kps[t_bottom, lead_hip]
    lead_tibia_bottom_vec = kps[t_bottom, lead_ankle] - kps[t_bottom, lead_knee]
    rear_femur_bottom_vec = kps[t_bottom, rear_knee] - kps[t_bottom, rear_hip]

    hip_flex_deg = 180.0 - vector_angle_3d(torso_bottom_vec, lead_femur_bottom_vec)
    knee_flex_deg = 180.0 - vector_angle_3d(lead_femur_bottom_vec, lead_tibia_bottom_vec)
    lunge_hip_knee_flexion_ratio = float(hip_flex_deg / max(knee_flex_deg, 10.0))

    # 전 구간 대퇴골-체간 각도 중 최소각
    trunk_thigh_angles = np.zeros(T, dtype=np.float32)
    for t in range(T):
        t_vec = shoulders[t] - hips[t]
        f_vec = kps[t, lead_knee] - kps[t, lead_hip]
        trunk_thigh_angles[t] = vector_angle_3d(t_vec, f_vec)
    lunge_trunk_thigh_angle_min = float(np.min(trunk_thigh_angles))

    # 후행 다리 고관절 신전 결손량
    rear_hip_ext_angle = vector_angle_3d(torso_bottom_vec, rear_femur_bottom_vec)
    lunge_hip_extension_deficit = float(abs(180.0 - rear_hip_ext_angle))

    # 시상면 골반 경사각
    pelvis_sagittal_tilt = float(np.degrees(np.arctan2(hips[t_bottom, 2] - shoulders[t_bottom, 2], torso_scale)))
    lunge_pelvic_tilt_sagittal = pelvis_sagittal_tilt

    # 무릎-상체 모멘트암 비율
    lead_knee_moment = abs(kps[t_bottom, lead_knee, 2] - hips[t_bottom, 2])
    trunk_moment = abs(shoulders[t_bottom, 2] - hips[t_bottom, 2])
    lunge_trunk_knee_moment_arm = float(trunk_moment / (lead_knee_moment + 1e-4))

    # [그룹 3] 정면 원근 수축 극복 및 3D 기하 피처 (6D)
    # 기립 기준 길이 대비 유효 길이비
    lunge_torso_foreshortening_ratio = float(np.min(torso_lengths) / l0_reference)

    # 뼈길이 불변성 강제 기반 복원 Z축 깊이 변위
    delta_xy = np.linalg.norm(shoulders[:, :2] - hips[:, :2], axis=1)
    calibrated_dz = np.sqrt(np.maximum(0.0, l0_reference**2 - delta_xy**2))
    lunge_calibrated_delta_z = float(np.max(calibrated_dz) / torso_scale)

    # 3D 월드 (Z, Y) 가상 시상각
    dy_3d = -(shoulders[:, 1] - hips[:, 1])
    dz_3d = shoulders[:, 2] - hips[:, 2]
    pitch_3d = np.degrees(np.arctan2(dz_3d, np.maximum(dy_3d, 1e-6)))
    lunge_virtual_sagittal_pitch_3d = float(np.mean(pitch_3d[win_start:win_end]))

    # 어깨-골반 깊이 차이 최대치
    lunge_shoulder_pelvis_z_diff = float(np.max(np.abs(dz_3d)) / torso_scale)

    # 관상면(좌우) 어깨 기울기 (비틀림 노이즈 배제 지표)
    shoulder_roll_dy = np.abs(kps[:, MP_LEFT_SHOULDER, 1] - kps[:, MP_RIGHT_SHOULDER, 1])
    shoulder_roll_dx = np.abs(kps[:, MP_LEFT_SHOULDER, 0] - kps[:, MP_RIGHT_SHOULDER, 0]) + 1e-6
    shoulder_roll_deg = np.degrees(np.arctan2(shoulder_roll_dy, shoulder_roll_dx))
    lunge_coronal_tilt_rejection = float(np.mean(shoulder_roll_deg))

    # 정면 투영 체간 수축 지속 적분
    compression_mask = (torso_lengths / l0_reference) < 0.85
    lunge_frontal_compression_integral = float(np.sum(compression_mask) / float(T))

    # [그룹 4] 경추-체간 및 락아웃 협응 피처 (6D)
    head_vec = head[t_bottom] - shoulders[t_bottom]
    spine_vec = shoulders[t_bottom] - hips[t_bottom]
    lunge_cervico_thoracic_angle = vector_angle_3d(head_vec, spine_vec)

    # 머리 젖힘과 상체 젖힘의 시계열 상관계수
    head_dz = head[:, 2] - shoulders[:, 2]
    torso_dz = shoulders[:, 2] - hips[:, 2]
    if np.std(head_dz) > 1e-4 and np.std(torso_dz) > 1e-4:
        r_matrix = np.corrcoef(head_dz, torso_dz)
        lunge_head_torso_hyperext_sync = float(r_matrix[0, 1])
    else:
        lunge_head_torso_hyperext_sync = 0.0

    # 기립 락아웃(종료 8F) 순간의 리바운드 피치각
    lunge_lockout_rebound_pitch = float(np.mean(torso_pitch[-8:]))

    # 상승 전환기 상체 저크(Jerk) 불안정 지수
    pitch_acc = np.gradient(pitch_vel)
    pitch_jerk = np.gradient(pitch_acc)
    ascent_start = t_bottom
    ascent_end = min(T, t_bottom + 16)
    lunge_ascent_trunk_jerk = float(np.max(np.abs(pitch_jerk[ascent_start:ascent_end])))

    # 최하사점 윈도우 내 체간 피치 표준편차
    lunge_bottom_pitch_stability_std = float(np.std(win_pitch))

    # 하강기 피치 각가속도 평활도
    descent_end = t_bottom
    lunge_descent_pitch_smoothness = float(np.mean(np.abs(pitch_acc[:descent_end])))

    # [그룹 5] 어드레스 동적 영점화 피처 (6D)
    lunge_delta_pitch_from_setup = float(lunge_vspa_torso_pitch_bottom - setup_baseline)
    lunge_setup_torso_zero_drift = float(np.mean(np.abs(torso_pitch - setup_baseline)))
    lunge_subject_norm_pitch_rom = float(lunge_vspa_torso_pitch_rom / (torso_scale * 10.0 + 1.0))
    lunge_trunk_length_scaled_pitch = float(lunge_vspa_torso_pitch_bottom * torso_scale)

    max_backward_dev = float(np.max(np.maximum(0.0, setup_baseline - torso_pitch)))
    lunge_relative_extension_ratio = float(max_backward_dev / (lunge_vspa_torso_pitch_rom + 1e-4))
    lunge_baseline_subtracted_peak = float(np.max(np.abs(torso_pitch - setup_baseline)))

    features = {
        "lunge_vspa_torso_pitch_bottom": lunge_vspa_torso_pitch_bottom,
        "lunge_vspa_torso_pitch_peak": lunge_vspa_torso_pitch_peak,
        "lunge_vspa_torso_pitch_min": lunge_vspa_torso_pitch_min,
        "lunge_vspa_torso_pitch_rom": lunge_vspa_torso_pitch_rom,
        "lunge_farrokhi_flexion_excess": lunge_farrokhi_flexion_excess,
        "lunge_farrokhi_hyperext_excess": lunge_farrokhi_hyperext_excess,
        "lunge_vspa_pitch_velocity_peak": lunge_vspa_pitch_velocity_peak,
        "lunge_ldc_phase_weighted_pitch": lunge_ldc_phase_weighted_pitch,
        "lunge_tibia_torso_parallelism": lunge_tibia_torso_parallelism,
        "lunge_hip_knee_flexion_ratio": lunge_hip_knee_flexion_ratio,
        "lunge_trunk_thigh_angle_min": lunge_trunk_thigh_angle_min,
        "lunge_hip_extension_deficit": lunge_hip_extension_deficit,
        "lunge_pelvic_tilt_sagittal": lunge_pelvic_tilt_sagittal,
        "lunge_trunk_knee_moment_arm": lunge_trunk_knee_moment_arm,
        "lunge_torso_foreshortening_ratio": lunge_torso_foreshortening_ratio,
        "lunge_calibrated_delta_z": lunge_calibrated_delta_z,
        "lunge_virtual_sagittal_pitch_3d": lunge_virtual_sagittal_pitch_3d,
        "lunge_shoulder_pelvis_z_diff": lunge_shoulder_pelvis_z_diff,
        "lunge_coronal_tilt_rejection": lunge_coronal_tilt_rejection,
        "lunge_frontal_compression_integral": lunge_frontal_compression_integral,
        "lunge_cervico_thoracic_angle": lunge_cervico_thoracic_angle,
        "lunge_head_torso_hyperext_sync": lunge_head_torso_hyperext_sync,
        "lunge_lockout_rebound_pitch": lunge_lockout_rebound_pitch,
        "lunge_ascent_trunk_jerk": lunge_ascent_trunk_jerk,
        "lunge_bottom_pitch_stability_std": lunge_bottom_pitch_stability_std,
        "lunge_descent_pitch_smoothness": lunge_descent_pitch_smoothness,
        "lunge_delta_pitch_from_setup": lunge_delta_pitch_from_setup,
        "lunge_setup_torso_zero_drift": lunge_setup_torso_zero_drift,
        "lunge_subject_norm_pitch_rom": lunge_subject_norm_pitch_rom,
        "lunge_trunk_length_scaled_pitch": lunge_trunk_length_scaled_pitch,
        "lunge_relative_extension_ratio": lunge_relative_extension_ratio,
        "lunge_baseline_subtracted_peak": lunge_baseline_subtracted_peak,
    }

    return features


def extract_lunge_c5_v2_feature_vector(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> np.ndarray:
    """32차원 float32 numpy 1D 배열로 피처 벡터 반환"""
    f_dict = extract_lunge_c5_v2_features(kps_33_3d_64f, vis_64f)
    vec = np.array([f_dict[name] for name in LUNGE_C5_V2_FEATURE_NAMES], dtype=np.float32)
    return vec

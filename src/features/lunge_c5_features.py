# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c5_features.py
Purpose: 바벨 런지 c5(상체의 과도한 숙임/젖힘 여부) 세부 오류 판별 전용 생체역학 32차원 도메인 피처 추출 모듈
         - 시상면(Sagittal Plane) 순수 체간 피치각 (Torso Pitch) 정밀 계측
         - 체간-대퇴골-골반 기구학적 커플링 수치화
         - 정면 0°(시점 C) 원근 수축(Foreshortening) 극복 3D 가상 시상각 및 깊이축 지표
         - 경추-체간 협응 및 상승 전환기 배치기/저크 감지
         - 동작 전 정지 상태(어드레스) 기반 동적 영점화(Zero-Calibration)
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

# 32차원 공식 도메인 피처 명칭 목록
LUNGE_C5_FEATURE_NAMES: List[str] = [
    # [그룹 1] 시상면 체간 피치각 피처 (8D)
    "lunge_torso_pitch_bottom_win",          # [ 1] 최하사점 윈도우 내 평균 체간 피치각 (°)
    "lunge_torso_pitch_peak",                # [ 2] 전체 시퀀스 최대 전방 숙임 피치각 (°)
    "lunge_torso_pitch_min",                 # [ 3] 전체 시퀀스 최대 후방 젖힘 피치각 (°)
    "lunge_torso_pitch_rom",                 # [ 4] 체간 피치각 가동범위 (peak - min) (°)
    "lunge_torso_backward_integral",         # [ 5] 후방 과신전(-각도) 시간 누적 적분
    "lunge_torso_forward_integral",          # [ 6] 전방 과숙임(+각도) 시간 누적 적분
    "lunge_torso_pitch_velocity_peak",       # [ 7] 피치 각속도 최대 피크 (deg/frame)
    "lunge_torso_bottom_slope",              # [ 8] 최하사점 전후 체간 피치 1차 추세 기울기

    # [그룹 2] 체간-대퇴골 골반 기구학 피처 (6D)
    "lunge_trunk_thigh_angle_min",           # [ 9] 선행 대퇴골과 체간 축 간의 최소 사잇각 (°)
    "lunge_hip_extension_deficit",           # [10] 후행 다리 고관절 신전 결손량 (°)
    "lunge_pelvic_tilt_sagittal",            # [11] 시상면 상 골반 전후 경사각 (°)
    "lunge_trunk_knee_moment_arm",           # [12] 골반 중심 대비 무릎-상체 모멘트암 비율
    "lunge_trunk_pelvis_sagittal_ratio",     # [13] 체간 각도 대비 골반 경사각 상대 비율
    "lunge_femur_torso_bottom_angle",        # [14] 최하사점 순간 대퇴골-체간 사잇각 (°)

    # [그룹 3] 정면 원근 수축 극복 3D 피처 (6D)
    "lunge_torso_foreshortening_ratio",      # [15] 정면(시점 C) 투영 체간 유효 길이비 (Lt / L0)
    "lunge_virtual_sagittal_pitch_3d",       # [16] 3D 월드 (Z, Y) 기반 가상 시상면 투영 피치각 (°)
    "lunge_rib_flare_z_disp",                # [17] 흉골/어깨 z축 전방 돌출 변위 (스케일 정규화)
    "lunge_shoulder_pelvis_z_diff",          # [18] 어깨-골반 깊이축(z축) 차이 최대치
    "lunge_sagittal_depth_span",             # [19] 체간 깊이축 이동 변위 스팬
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
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = np.dot(v1, v2)
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
    l_disp_mean = np.mean(l_disp[mid_start:mid_end])
    r_disp_mean = np.mean(r_disp[mid_start:mid_end])

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
    t_bottom = np.clip(t_bottom, 10, T - 10)

    win_start = max(0, t_bottom - 6)
    win_end = min(T, t_bottom + 7)
    return t_bottom, win_start, win_end


def compute_torso_pitch_series(kps: np.ndarray) -> np.ndarray:
    """
    체간(어깨 중점 -> 골반 중점)의 수직선 대비 시상면 피치각 시계열 (T,) 계산
    - 전방 숙임(Flexion): 양수 (+)
    - 후방 젖힘(Hyperextension): 음수 (-)
    """
    T = kps.shape[0]
    shoulders = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hips = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0

    # 골반 -> 어깨 벡터
    torso_vec = shoulders - hips  # (T, 3)

    # 지면 수직선은 -Y (머리 방향)
    # 시상면 피치각: dy와 전후(dz) 또는 수평(dx/dz) 투영각
    # y축이 아래로 증가하는 이미지 좌표계 기준: 수직 위쪽은 [0, -1, 0]
    # dz > 0 이면 어깨가 뒤로 간 것 (후방 젖힘), dz < 0 이면 어깨가 앞으로 간 것 (전방 숙임)
    # 부호 규약: 전방 숙임 = +, 후방 젖힘 = -
    dy = -torso_vec[:, 1]  # 위쪽 성분 (보통 양수)
    dz = torso_vec[:, 2]   # 전후 깊이 성분

    # arctan2(dz, dy)를 도 단위로 변환
    pitch_rad = np.arctan2(dz, np.maximum(dy, 1e-6))
    pitch_deg = np.degrees(pitch_rad)
    return pitch_deg.astype(np.float32)


def extract_lunge_c5_features(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    바벨 런지 c5(상체의 과도한 숙임/젖힘 여부) 32차원 도메인 피처 추출 메인 함수
    - kps_33_3d_64f: (64, 33, 3) 3차원 월드 키포인트
    - vis_64f: (64, 33) 키포인트 가시성 (선택 사항)
    """
    kps = kps_33_3d_64f.astype(np.float32)
    T = kps.shape[0]
    assert T == 64, f"입력 시퀀스는 64프레임이어야 합니다. (현재: {T})"

    # 1. 선행 다리 및 최저점 탐색
    lead_side = detect_lead_leg(kps)
    if lead_side == 'left':
        lead_hip, lead_knee = MP_LEFT_HIP, MP_LEFT_KNEE
        rear_hip, rear_knee = MP_RIGHT_HIP, MP_RIGHT_KNEE
    else:
        lead_hip, lead_knee = MP_RIGHT_HIP, MP_RIGHT_KNEE
        rear_hip, rear_knee = MP_LEFT_HIP, MP_LEFT_KNEE

    t_bottom, win_start, win_end = find_inflection_point(kps)

    # 2. 중심점 및 스케일 계산
    shoulders = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hips = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    head = (kps[:, MP_LEFT_EAR] + kps[:, MP_RIGHT_EAR]) / 2.0

    torso_lengths = np.linalg.norm(shoulders - hips, axis=1)
    torso_scale = float(np.mean(torso_lengths)) + 1e-6

    # 3. 체간 피치각 시계열 산출 및 스무딩
    torso_pitch_raw = compute_torso_pitch_series(kps)
    torso_pitch = gaussian_filter1d(torso_pitch_raw, sigma=1.5)

    # 셋업 구간(어드레스 0~8F) 기준선
    setup_baseline = float(np.mean(torso_pitch[:8]))

    # [그룹 1] 시상면 체간 피치각 피처 (8D)
    win_pitch = torso_pitch[win_start:win_end]
    lunge_torso_pitch_bottom_win = float(np.mean(win_pitch))
    lunge_torso_pitch_peak = float(np.max(torso_pitch))
    lunge_torso_pitch_min = float(np.min(torso_pitch))
    lunge_torso_pitch_rom = float(lunge_torso_pitch_peak - lunge_torso_pitch_min)

    # 후방 과신전 적분 (-5도 이하) 및 전방 과숙임 적분 (+20도 이상)
    backward_excess = np.maximum(0.0, -5.0 - torso_pitch)
    forward_excess = np.maximum(0.0, torso_pitch - 20.0)
    lunge_torso_backward_integral = float(np.sum(backward_excess) / float(T))
    lunge_torso_forward_integral = float(np.sum(forward_excess) / float(T))

    # 각속도 피크
    pitch_vel = np.gradient(torso_pitch)
    lunge_torso_pitch_velocity_peak = float(np.max(np.abs(pitch_vel)))

    # 최하사점 1차 기울기 (추세)
    if win_end > win_start + 2:
        slope = float(np.polyfit(np.arange(len(win_pitch)), win_pitch, 1)[0])
    else:
        slope = 0.0
    lunge_torso_bottom_slope = slope

    # [그룹 2] 체간-대퇴골 골반 기구학 피처 (6D)
    torso_bottom_vec = shoulders[t_bottom] - hips[t_bottom]
    lead_femur_bottom_vec = kps[t_bottom, lead_knee] - kps[t_bottom, lead_hip]
    rear_femur_bottom_vec = kps[t_bottom, rear_knee] - kps[t_bottom, rear_hip]

    # 선행 대퇴골 - 체간 사잇각
    lunge_femur_torso_bottom_angle = vector_angle_3d(torso_bottom_vec, lead_femur_bottom_vec)

    # 전 구간 대퇴골-체간 각도 중 최소각
    trunk_thigh_angles = np.zeros(T, dtype=np.float32)
    for t in range(T):
        t_vec = shoulders[t] - hips[t]
        f_vec = kps[t, lead_knee] - kps[t, lead_hip]
        trunk_thigh_angles[t] = vector_angle_3d(t_vec, f_vec)
    lunge_trunk_thigh_angle_min = float(np.min(trunk_thigh_angles))

    # 후행 다리 고관절 신전 결손량 (후행 대퇴골과 체간이 이루는 각도 편차)
    rear_hip_ext_angle = vector_angle_3d(torso_bottom_vec, rear_femur_bottom_vec)
    lunge_hip_extension_deficit = float(abs(180.0 - rear_hip_ext_angle))

    # 시상면 골반 경사각
    pelvis_sagittal_tilt = float(np.degrees(np.arctan2(hips[t_bottom, 2] - shoulders[t_bottom, 2], torso_scale)))
    lunge_pelvic_tilt_sagittal = pelvis_sagittal_tilt

    # 무릎-상체 모멘트암 비율
    lead_knee_moment = abs(kps[t_bottom, lead_knee, 2] - hips[t_bottom, 2])
    trunk_moment = abs(shoulders[t_bottom, 2] - hips[t_bottom, 2])
    lunge_trunk_knee_moment_arm = float((trunk_moment + 1e-4) / (lead_knee_moment + 1e-4))
    lunge_trunk_pelvis_sagittal_ratio = float((abs(lunge_torso_pitch_bottom_win) + 1e-4) / (abs(pelvis_sagittal_tilt) + 1e-4))

    # [그룹 3] 정면 원근 수축 극복 3D 피처 (6D)
    # 정면 2D 투영(X, Y) 체간 길이 시계열
    frontal_torso_len = np.linalg.norm(shoulders[:, :2] - hips[:, :2], axis=1)
    setup_frontal_len = float(np.mean(frontal_torso_len[:8])) + 1e-6
    foreshortening = frontal_torso_len / setup_frontal_len
    lunge_torso_foreshortening_ratio = float(np.min(foreshortening))

    # 3D 가상 시상각 (Z, Y 평면에서의 어깨-골반 기울기)
    dz_series = shoulders[:, 2] - hips[:, 2]
    dy_series = -(shoulders[:, 1] - hips[:, 1])
    virt_sagittal_pitch = np.degrees(np.arctan2(dz_series, np.maximum(dy_series, 1e-6)))
    lunge_virtual_sagittal_pitch_3d = float(np.mean(virt_sagittal_pitch[win_start:win_end]))

    # 흉골 돌출 변위 (후방 젖힘 시 어깨 z축 돌출)
    lunge_rib_flare_z_disp = float(np.max(dz_series) / torso_scale)
    lunge_shoulder_pelvis_z_diff = float(np.ptp(dz_series))
    lunge_sagittal_depth_span = float(np.max(np.abs(dz_series)) / torso_scale)
    lunge_frontal_compression_integral = float(np.sum(np.maximum(0.0, 0.9 - foreshortening)) / float(T))

    # [그룹 4] 경추-체간 및 락아웃 협응 피처 (6D)
    head_vec = head[t_bottom] - shoulders[t_bottom]
    lunge_cervico_thoracic_angle = vector_angle_3d(head_vec, torso_bottom_vec)

    # 머리 젖힘과 상체 젖힘 동시성
    head_pitch = gaussian_filter1d(np.degrees(np.arctan2(head[:, 2] - shoulders[:, 2], -(head[:, 1] - shoulders[:, 1]) + 1e-6)), sigma=1.5)
    corr = np.corrcoef(head_pitch, torso_pitch)[0, 1]
    lunge_head_torso_hyperext_sync = float(corr) if not np.isnan(corr) else 0.0

    # 기립 락아웃(동작 후반 50~64F) 후방 리바운드 피치
    lunge_lockout_rebound_pitch = float(np.min(torso_pitch[int(T * 0.75):]))

    # 상승 전환기 상체 저크 (Jerk: 3차 미분)
    jerk_pitch = np.gradient(np.gradient(pitch_vel))
    lunge_ascent_trunk_jerk = float(np.max(np.abs(jerk_pitch[t_bottom:]))) if t_bottom < T - 1 else 0.0
    lunge_bottom_pitch_stability_std = float(np.std(win_pitch))
    descent_accel = np.gradient(pitch_vel[:t_bottom]) if t_bottom > 2 else np.array([0.0])
    lunge_descent_pitch_smoothness = float(np.mean(np.abs(descent_accel)))

    # [그룹 5] 어드레스 동적 영점화 피처 (6D)
    lunge_delta_pitch_from_setup = float(lunge_torso_pitch_bottom_win - setup_baseline)
    lunge_setup_torso_zero_drift = float(np.mean(torso_pitch - setup_baseline))
    lunge_subject_norm_pitch_rom = float(lunge_torso_pitch_rom / torso_scale)
    lunge_trunk_length_scaled_pitch = float(lunge_torso_pitch_bottom_win * torso_scale)
    lunge_relative_extension_ratio = float((setup_baseline - lunge_torso_pitch_min) / (abs(setup_baseline) + 10.0))
    lunge_baseline_subtracted_peak = float(np.max(np.abs(torso_pitch - setup_baseline)))

    # 전체 딕셔너리 조립
    feats: Dict[str, float] = {
        "lunge_torso_pitch_bottom_win": lunge_torso_pitch_bottom_win,
        "lunge_torso_pitch_peak": lunge_torso_pitch_peak,
        "lunge_torso_pitch_min": lunge_torso_pitch_min,
        "lunge_torso_pitch_rom": lunge_torso_pitch_rom,
        "lunge_torso_backward_integral": lunge_torso_backward_integral,
        "lunge_torso_forward_integral": lunge_torso_forward_integral,
        "lunge_torso_pitch_velocity_peak": lunge_torso_pitch_velocity_peak,
        "lunge_torso_bottom_slope": lunge_torso_bottom_slope,

        "lunge_trunk_thigh_angle_min": lunge_trunk_thigh_angle_min,
        "lunge_hip_extension_deficit": lunge_hip_extension_deficit,
        "lunge_pelvic_tilt_sagittal": lunge_pelvic_tilt_sagittal,
        "lunge_trunk_knee_moment_arm": lunge_trunk_knee_moment_arm,
        "lunge_trunk_pelvis_sagittal_ratio": lunge_trunk_pelvis_sagittal_ratio,
        "lunge_femur_torso_bottom_angle": lunge_femur_torso_bottom_angle,

        "lunge_torso_foreshortening_ratio": lunge_torso_foreshortening_ratio,
        "lunge_virtual_sagittal_pitch_3d": lunge_virtual_sagittal_pitch_3d,
        "lunge_rib_flare_z_disp": lunge_rib_flare_z_disp,
        "lunge_shoulder_pelvis_z_diff": lunge_shoulder_pelvis_z_diff,
        "lunge_sagittal_depth_span": lunge_sagittal_depth_span,
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
        "lunge_baseline_subtracted_peak": lunge_baseline_subtracted_peak
    }

    # 안전성 클램핑 (NaN/Inf 원천 차단)
    for k in feats:
        val = feats[k]
        if np.isnan(val) or np.isinf(val):
            feats[k] = 0.0

    return feats


def extract_lunge_c5_feature_vector(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> np.ndarray:
    """32차원 피처 딕셔너리를 고정 순서의 numpy 1D 배열(32,)로 변환"""
    feats = extract_lunge_c5_features(kps_33_3d_64f, vis_64f)
    vec = np.array([feats[name] for name in LUNGE_C5_FEATURE_NAMES], dtype=np.float32)
    return vec

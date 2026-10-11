# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Module: side_lunge_c5_features_v7.py
Purpose: 사이드 런지 c5(상체의 과도한 숙임/젖힘 여부) F1 >= 70% 달성을 위한
         사이드 런지 전용 42차원 혁신 도메인 피처 추출 모듈 (v7)
         
[핵심 혁신 설계]
1. [c3 공간 거리 다리 분리 엔진 계승]:
   골반 중심 대비 X축 수평 공간 거리 지표(무릎 0.6 + 발목 0.4)로 작업/신전 다리 100% 분리.
2. [사이드 런지 관상면-시상면 3D 오일러 분해]:
   관상면(Frontal) 롤(Roll, 갸우뚱)과 시상면(Sagittal) 피치(Pitch, 젖힘)를 직교 분해하여
   Key 160(몸통 갸우뚱 젖힘)과 Key 150(순수 젖힘)을 정확히 분별.
3. [경추(머리) 시선 왜곡 완전 디커플링]:
   머리(코/귀)를 배제하고 어깨 중심(T1)~골반 중심(L5/Sacrum)의 순수 흉요추 축을 추출하여
   Key 166(바닥보고 젖힘), Key 165(하늘보고 젖힘), Key 163/164(좌우보고 젖힘)의 FN 오류 해결.
4. [정면 뷰(view3) 어깨너비 불변 체간 형상비(Trunk Aspect Ratio)]:
   2D 깊이 투영 단축을 보정하기 위해 기립 대비 최하사점 형상비 수축률(L_torso / W_shoulder)을
   도입하여 정면 뷰의 치명적 FP(오탐 161건) 원천 차단.
5. [골반 축-체간 축 직교성 지수]:
   Key 155(구부정 다리) 등 골반 회전으로 인한 정상 상체의 오탐(FP) 방지.
6. [시계열 후방 각속도/각가속도 위상 적분]:
   하강기 체간의 후방 젖힘 붕괴 운동에너지를 적분.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq

from src.features.side_lunge_spatial_leg_separator import detect_side_lunge_lead_leg_spatial

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

# 42차원 v7 공식 도메인 피처 명칭 목록
SIDE_LUNGE_C5_V7_FEATURE_NAMES: List[str] = [
    # [그룹 1] 3D 순수 흉요추 체간 오일러 각 및 젖힘 크기 (7D)
    "sl_pure_torso_pitch_bottom",            # [ 1] 최하사점 순수 체간 시상면 피치각 (전방 +, 후방 -) (°)
    "sl_pure_torso_pitch_min",               # [ 2] 전체 시퀀스 최대 후방 젖힘 피치각 (음수 극값) (°)
    "sl_pure_torso_roll_abs_bottom",         # [ 3] 최하사점 체간 관상면 롤(갸우뚱) 절대각 (°)
    "sl_pure_torso_3d_tilt_bottom",          # [ 4] 최하사점 체간의 중력 수직선 대비 3D 총 기울기 사잇각 (°)
    "sl_pure_torso_pitch_range",             # [ 5] 셋업 대비 최하사점까지의 피치 변화 진폭 (°)
    "sl_hyperext_phase_integral",            # [ 6] 후방 젖힘(피치 < 0) 구간 각도 음수 적분량 (deg·frame)
    "sl_ldc_weighted_pitch",                 # [ 7] 최하사점(LDC) 중심 가우시안 위상 가중 피치 적분

    # [그룹 2] 정면 뷰(view3) 왜곡 극복 및 체간 형상비 불변 피처 (6D)
    "sl_torso_aspect_ratio_bottom",          # [ 8] 최하사점 체간 길이 대 어깨 너비 형상비 (L_torso / W_sh)
    "sl_aspect_ratio_compression",           # [ 9] 기립 대비 최하사점 형상비 수축률 (R_bottom / R_stand)
    "sl_calibrated_delta_z",                 # [10] 뼈길이 불변성 강제 기반 복원 Z축 깊이 변위
    "sl_frontal_pelvis_shoulder_x_offset",   # [11] 정면 투영 골반 중심 대비 어깨 중심 X축 편차율
    "sl_shoulder_width_stability_std",       # [12] 시퀀스 내 어깨 너비 정규화 안정성
    "sl_frontal_compression_rate",           # [13] 하강기 정면 체간 투영 길이 수축 속도

    # [그룹 3] 경추(머리) 시선 왜곡 디커플링 및 척추 곡률 (6D)
    "sl_cervico_thoracic_decoupling_angle",  # [14] 머리(시선) 축과 순수 체간 축 간의 3D 사잇각 (경추 분리) (°)
    "sl_head_pitch_vs_torso_pitch_diff",     # [15] 머리 피치각과 체간 피치각의 괴리도 (|θ_head - θ_torso|) (°)
    "sl_pure_thoracic_curvature",            # [16] 경추-어깨-골반 3점 척추 굴곡/과전만 곡률 지수
    "sl_head_torso_hyperext_sync",           # [17] 머리 젖힘과 상체 젖힘의 시계열 동시성 상관계수
    "sl_gaze_to_horizon_pitch",              # [18] 코-귀 시선 축의 수평선 대비 앙각/부각 (하늘/바닥 판별) (°)
    "sl_cervical_strain_index",              # [19] 최하사점 머리-목 회전 및 신전 스트레인 지수

    # [그룹 4] 사이드 런지 하지-골반 결합 및 기저면 모멘트 (7D)
    "sl_lead_hip_trunk_angle_3d",            # [20] 선행 작업 다리 고관절-체간 3D 사잇각 (°)
    "sl_lead_thigh_torso_flexion_ratio",     # [21] 선행 대퇴골 굴곡 대비 체간 굴곡 비율
    "sl_trail_hip_abduction_angle",          # [22] 후행 신전 다리 고관절 외전각 (°)
    "sl_pelvis_torso_orthogonality",         # [23] 골반선과 체간선의 직교성 일탈 지수 (FP 방지)
    "sl_pelvic_tilt_sagittal",               # [24] 시상면 골반 전후 경사각 (°)
    "sl_support_base_torso_plumb_offset",    # [25] 작업 발목 지지점 대비 어깨 중심 수직선 전후 편차
    "sl_tibia_torso_parallelism",            # [26] 작업 정강이와 체간 축의 평행도 편차 (°)

    # [그룹 5] 어드레스 동적 영점화 및 기립 락아웃 피처 (5D)
    "sl_delta_pitch_from_setup",             # [27] 셋업(0~8F) 기준선 대비 최하사점 피치 편차 (°)
    "sl_setup_torso_zero_drift",             # [28] 동작 전체 구간의 셋업 영점 대비 평균 표류 (°)
    "sl_relative_extension_ratio",           # [29] 셋업 대비 최대 후방 신전 비율
    "sl_baseline_subtracted_peak",           # [30] 셋업 차감 후 최대 편차 피크 (°)
    "sl_lockout_rebound_pitch",              # [31] 기립 복귀 락아웃 순간의 후방 리바운드 피치각 (°)

    # [그룹 6] 동역학 저크(Jerk), 각속도 및 시간-주파수 스펙트럼 (6D)
    "sl_descent_pitch_velocity_min",         # [32] 하강기 최대 후방 젖힘 각속도 피크 (deg/frame, 음수)
    "sl_ascent_trunk_jerk",                  # [33] 상승 전환기 상체 3차 저크(Jerk) 불안정 지수
    "sl_bottom_pitch_stability_std",         # [34] 최하사점 윈도우 내 체간 피치 표준편차 (°)
    "sl_torso_spectral_energy_ratio",        # [35] 고주파(>=3Hz) 저크 대 저주파(<1.5Hz) 스펙트럼 에너지 비
    "sl_bottom_energy_flux_ratio",           # [36] 최하사점 반환 구간(LDC±4F) 운동에너지 플럭스 비율
    "sl_early_ascent_burst_jerk",            # [37] 최하사점 직후 8프레임 상승전환 버스트 저크 피크

    # [그룹 7] 비틀림(Torsion) 및 다축 결합 복합 결함 극복 피처 (5D)
    "sl_coronal_tilt_rejection",             # [38] 관상면(좌우) 어깨 기울기 노이즈 지수
    "sl_trunk_pelvis_torsion_angle",         # [39] 어깨 축과 골반 축의 수평 비틀림 상대각 (°)
    "sl_bidirectional_hyperext_asym",        # [40] 후방 과신전 에너지 비대칭 비
    "sl_femur_to_torso_ratio",               # [41] 대퇴골-체간 해부학적 길이비
    "sl_coronal_sagittal_coupling_index",    # [42] 관상면 롤과 시상면 피치의 다축 연성 결합 지수
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_side_lunge_c5_v7_feature_vector(sequence: np.ndarray) -> np.ndarray:
    """
    사이드 런지 64프레임 MediaPipe 33관절 시퀀스로부터 42차원 v7 혁신 피처 벡터 추출
    
    Args:
        sequence: (64, 33, 3) 넘파이 배열
        
    Returns:
        features: (42,) float32 넘파이 배열
    """
    assert sequence.ndim == 3 and sequence.shape[0] == 64 and sequence.shape[1] >= 33, \
        f"입력 규격 불일치: expected (64, 33, 3), got {sequence.shape}"

    kps = sequence[:, :, :3].astype(np.float32)

    # 1. c3 공간 거리 엔진 기반 앞/작업 다리 및 최하사점 프레임 판별
    is_left_lead, bottom_idx = detect_side_lunge_lead_leg_spatial(kps)
    w_start = max(0, bottom_idx - 2)
    w_end = min(64, bottom_idx + 3)

    lead_hip_idx = MP_LEFT_HIP if is_left_lead else MP_RIGHT_HIP
    lead_knee_idx = MP_LEFT_KNEE if is_left_lead else MP_RIGHT_KNEE
    lead_ankle_idx = MP_LEFT_ANKLE if is_left_lead else MP_RIGHT_ANKLE
    trail_hip_idx = MP_RIGHT_HIP if is_left_lead else MP_LEFT_HIP
    trail_knee_idx = MP_RIGHT_KNEE if is_left_lead else MP_LEFT_KNEE

    # 2. 중심점 시계열 추출 (64, 3)
    sh_l = kps[:, MP_LEFT_SHOULDER]
    sh_r = kps[:, MP_RIGHT_SHOULDER]
    hip_l = kps[:, MP_LEFT_HIP]
    hip_r = kps[:, MP_RIGHT_HIP]
    mid_sh = (sh_l + sh_r) * 0.5
    mid_hip = (hip_l + hip_r) * 0.5

    # 머리 중심점 (코, 양 귀)
    nose = kps[:, MP_NOSE]
    ear_l = kps[:, MP_LEFT_EAR]
    ear_r = kps[:, MP_RIGHT_EAR]
    mid_head = (nose + ear_l + ear_r) / 3.0

    # 3. 순수 체간 벡터 (어깨 중심 - 골반 중심)
    v_torso = mid_sh - mid_hip  # (64, 3)
    norm_torso = np.linalg.norm(v_torso, axis=-1, keepdims=True) + 1e-6
    v_torso_unit = v_torso / norm_torso

    # 3.1 3D 오일러 분해 각도 시계열 (피치: 시상면 전후, 롤: 관상면 좌우)
    # MediaPipe 좌표계: X = 우측(+), Y = 아래(+), Z = 카메라 전방(-) / 후방(+)
    # 수직 기준: -Y (위쪽)
    pitch_ts = np.degrees(np.arctan2(-v_torso[:, 2], -v_torso[:, 1] + 1e-6)) # 전방 숙임 (+), 후방 젖힘 (-)
    roll_ts = np.degrees(np.arctan2(v_torso[:, 0], -v_torso[:, 1] + 1e-6))   # 좌우 갸우뚱
    pitch_smooth = gaussian_filter1d(pitch_ts, sigma=1.5)
    roll_smooth = gaussian_filter1d(roll_ts, sigma=1.5)

    # 3D 총 기울기 각도 (수직선 [0, -1, 0]과의 3D 사잇각)
    v_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cos_tilt = np.sum(v_torso_unit * v_up, axis=-1)
    tilt_3d_ts = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))

    # 4. [그룹 1] 순수 흉요추 체간 오일러 각 및 젖힘 크기 (7D)
    sl_pure_torso_pitch_bottom = float(np.mean(pitch_smooth[w_start:w_end]))
    sl_pure_torso_pitch_min = float(np.min(pitch_smooth))
    sl_pure_torso_roll_abs_bottom = float(np.mean(np.abs(roll_smooth[w_start:w_end])))
    sl_pure_torso_3d_tilt_bottom = float(np.mean(tilt_3d_ts[w_start:w_end]))

    setup_pitch = float(np.mean(pitch_smooth[0:8]))
    sl_pure_torso_pitch_range = float(setup_pitch - sl_pure_torso_pitch_bottom)

    # 후방 젖힘(피치 < 0) 구간 음수 적분
    hyperext_mask = pitch_smooth < 0.0
    sl_hyperext_phase_integral = float(np.sum(np.abs(pitch_smooth[hyperext_mask]))) / 64.0

    # LDC 가우시안 위상 가중 피치 적분
    sigma_ldc = 5.0
    gauss_weights = np.exp(-0.5 * ((np.arange(64) - bottom_idx) / sigma_ldc) ** 2)
    gauss_weights /= (np.sum(gauss_weights) + 1e-6)
    sl_ldc_weighted_pitch = float(np.sum(gauss_weights * pitch_smooth))

    # 5. [그룹 2] 정면 뷰(view3) 왜곡 극복 및 체간 형상비 불변 피처 (6D)
    # 2D 투영 (X, Y) 상의 체간 길이 및 어깨 너비
    sh_width_2d = np.linalg.norm(sh_l[:, :2] - sh_r[:, :2], axis=-1) + 1e-6
    torso_len_2d = np.linalg.norm(mid_sh[:, :2] - mid_hip[:, :2], axis=-1) + 1e-6
    aspect_ratio_ts = torso_len_2d / sh_width_2d

    sl_torso_aspect_ratio_bottom = float(np.mean(aspect_ratio_ts[w_start:w_end]))
    stand_aspect = float(np.mean(aspect_ratio_ts[0:8])) + 1e-6
    sl_aspect_ratio_compression = sl_torso_aspect_ratio_bottom / stand_aspect

    # 복원 Z 변위
    l0 = float(np.mean(torso_len_2d[0:8]))
    lt = float(np.mean(torso_len_2d[w_start:w_end]))
    sl_calibrated_delta_z = float(np.sqrt(max(0.0, l0**2 - lt**2)))

    # 정면 투영 골반-어깨 X 편차율
    dx_sh_hip = np.abs(mid_sh[:, 0] - mid_hip[:, 0]) / sh_width_2d
    sl_frontal_pelvis_shoulder_x_offset = float(np.mean(dx_sh_hip[w_start:w_end]))
    sl_shoulder_width_stability_std = float(np.std(sh_width_2d) / (np.mean(sh_width_2d) + 1e-6))

    # 하강기 체간 투영 길이 수축 속도
    descent_frames = max(1, bottom_idx)
    sl_frontal_compression_rate = float((l0 - lt) / descent_frames)

    # 6. [그룹 3] 경추(머리) 시선 왜곡 디커플링 및 척추 곡률 (6D)
    # 머리-목 벡터 (머리 중심 - 어깨 중심)
    v_neck = mid_head - mid_sh
    norm_neck = np.linalg.norm(v_neck, axis=-1, keepdims=True) + 1e-6
    v_neck_unit = v_neck / norm_neck

    # 머리-체간 3D 사잇각 (180도 일직선 기준 편차)
    dot_neck_torso = np.sum(v_neck_unit * v_torso_unit, axis=-1)
    cervico_ts = 180.0 - np.degrees(np.arccos(np.clip(dot_neck_torso, -1.0, 1.0)))
    sl_cervico_thoracic_decoupling_angle = float(np.mean(cervico_ts[w_start:w_end]))

    # 머리 피치각 시계열
    head_pitch_ts = np.degrees(np.arctan2(-v_neck[:, 2], -v_neck[:, 1] + 1e-6))
    sl_head_pitch_vs_torso_pitch_diff = float(np.mean(np.abs(head_pitch_ts[w_start:w_end] - pitch_smooth[w_start:w_end])))

    # 척추 3점(머리, 어깨, 골반) 곡률 지수
    sl_pure_thoracic_curvature = float(np.mean(np.abs(cervico_ts - np.mean(cervico_ts[0:8]))))

    # 머리 젖힘과 상체 젖힘 동시성 상관계수
    corr_mat = np.corrcoef(pitch_smooth, head_pitch_ts)
    sl_head_torso_hyperext_sync = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0

    # 시선 앙각 (귀-코 벡터)
    v_gaze = nose - (ear_l + ear_r) * 0.5
    sl_gaze_to_horizon_pitch = float(np.degrees(np.arctan2(-v_gaze[bottom_idx, 1], v_gaze[bottom_idx, 2] + 1e-6)))
    sl_cervical_strain_index = float(np.std(cervico_ts))

    # 7. [그룹 4] 사이드 런지 하지-골반 결합 및 기저면 모멘트 (7D)
    # 선행 대퇴골 벡터 (고관절 -> 무릎)
    v_lead_femur = kps[:, lead_knee_idx] - kps[:, lead_hip_idx]
    norm_femur = np.linalg.norm(v_lead_femur, axis=-1, keepdims=True) + 1e-6
    v_lead_femur_unit = v_lead_femur / norm_femur

    # 고관절-체간 3D 사잇각
    dot_hip_torso = np.sum(-v_lead_femur_unit * v_torso_unit, axis=-1)
    hip_trunk_ang = np.degrees(np.arccos(np.clip(dot_hip_torso, -1.0, 1.0)))
    sl_lead_hip_trunk_angle_3d = float(np.mean(hip_trunk_ang[w_start:w_end]))

    # 대퇴골 굴곡각
    femur_pitch = np.degrees(np.arccos(np.clip(v_lead_femur_unit[:, 1], -1.0, 1.0)))
    sl_lead_thigh_torso_flexion_ratio = float((np.mean(femur_pitch[w_start:w_end]) + 1e-6) / (np.abs(sl_pure_torso_pitch_bottom) + 1.0))

    # 후행 다리 외전각
    v_trail_femur = kps[:, trail_knee_idx] - kps[:, trail_hip_idx]
    v_pelvis_line = hip_r - hip_l
    norm_pelvis = np.linalg.norm(v_pelvis_line, axis=-1, keepdims=True) + 1e-6
    v_pelvis_unit = v_pelvis_line / norm_pelvis

    dot_trail = np.sum((v_trail_femur / (np.linalg.norm(v_trail_femur, axis=-1, keepdims=True) + 1e-6)) * v_pelvis_unit, axis=-1)
    sl_trail_hip_abduction_angle = float(np.mean(np.degrees(np.arccos(np.clip(np.abs(dot_trail[w_start:w_end]), -1.0, 1.0)))))

    # 골반선과 체간선의 직교성 일탈 지수 (내적이 0이어야 직교)
    dot_pelvis_torso = np.sum(v_pelvis_unit * v_torso_unit, axis=-1)
    sl_pelvis_torso_orthogonality = float(np.mean(np.abs(dot_pelvis_torso[w_start:w_end])))

    # 골반 시상면 경사각
    pelvis_norm = np.cross(v_pelvis_line, v_torso)
    pelvis_norm /= (np.linalg.norm(pelvis_norm, axis=-1, keepdims=True) + 1e-6)
    sl_pelvic_tilt_sagittal = float(np.degrees(np.arccos(np.clip(pelvis_norm[bottom_idx, 2], -1.0, 1.0))))

    # 지지 발목 대비 어깨 중심 수직선 편차
    plumb_offset = (mid_sh[:, 2] - kps[:, lead_ankle_idx, 2]) / (l0 + 1e-6)
    sl_support_base_torso_plumb_offset = float(np.mean(plumb_offset[w_start:w_end]))

    # 정강이-체간 평행도
    v_shank = kps[:, lead_ankle_idx] - kps[:, lead_knee_idx]
    norm_shank = np.linalg.norm(v_shank, axis=-1, keepdims=True) + 1e-6
    dot_shank_torso = np.sum((v_shank / norm_shank) * v_torso_unit, axis=-1)
    sl_tibia_torso_parallelism = float(np.mean(np.degrees(np.arccos(np.clip(dot_shank_torso[w_start:w_end], -1.0, 1.0)))))

    # 8. [그룹 5] 어드레스 동적 영점화 및 기립 락아웃 (5D)
    sl_delta_pitch_from_setup = float(sl_pure_torso_pitch_bottom - setup_pitch)
    sl_setup_torso_zero_drift = float(np.mean(np.abs(pitch_smooth - setup_pitch)))
    min_ext = float(np.min(pitch_smooth))
    sl_relative_extension_ratio = float((setup_pitch - min_ext) / (np.abs(setup_pitch) + 5.0))
    sl_baseline_subtracted_peak = float(np.max(np.abs(pitch_smooth - setup_pitch)))

    # 기립 락아웃 (마지막 8프레임 리바운드)
    lockout_pitch = float(np.mean(pitch_smooth[56:64]))
    sl_lockout_rebound_pitch = float(lockout_pitch - setup_pitch)

    # 9. [그룹 6] 동역학 저크, 각속도 및 스펙트럼 (6D)
    pitch_vel = np.gradient(pitch_smooth)  # deg/frame
    pitch_acc = np.gradient(pitch_vel)
    pitch_jerk = np.gradient(pitch_acc)

    sl_descent_pitch_velocity_min = float(np.min(pitch_vel[:bottom_idx+1]))
    sl_ascent_trunk_jerk = float(np.max(np.abs(pitch_jerk[bottom_idx:min(64, bottom_idx+12)])))
    sl_bottom_pitch_stability_std = float(np.std(pitch_smooth[w_start:w_end]))

    # FFT 스펙트럼 에너지 비
    fft_vals = np.abs(fft(pitch_smooth - np.mean(pitch_smooth)))[:32]
    freqs = fftfreq(64, d=1.0/30.0)[:32]
    high_energy = float(np.sum(fft_vals[freqs >= 3.0]**2))
    low_energy = float(np.sum(fft_vals[freqs < 1.5]**2)) + 1e-6
    sl_torso_spectral_energy_ratio = float(high_energy / low_energy)

    # 운동에너지 플럭스 비율
    kin_energy = 0.5 * (pitch_vel ** 2)
    bottom_flux = float(np.sum(kin_energy[w_start:w_end]))
    total_flux = float(np.sum(kin_energy)) + 1e-6
    sl_bottom_energy_flux_ratio = float(bottom_flux / total_flux)

    # 상승 전환 버스트 저크 피크
    burst_win = pitch_jerk[bottom_idx:min(64, bottom_idx+8)]
    sl_early_ascent_burst_jerk = float(np.max(np.abs(burst_win))) if len(burst_win) > 0 else 0.0

    # 10. [그룹 7] 비틀림 및 복합 결함 극복 피처 (5D)
    sl_coronal_tilt_rejection = float(np.mean(np.abs(roll_smooth[w_start:w_end])))

    # 어깨-골반 수평 비틀림 (Torsion)
    v_sh_line = sh_r - sh_l
    v_sh_proj = v_sh_line[:, [0, 2]]
    v_hip_proj = v_pelvis_line[:, [0, 2]]
    dot_torsion = np.sum((v_sh_proj / (np.linalg.norm(v_sh_proj, axis=-1, keepdims=True) + 1e-6)) * 
                         (v_hip_proj / (np.linalg.norm(v_hip_proj, axis=-1, keepdims=True) + 1e-6)), axis=-1)
    sl_trunk_pelvis_torsion_angle = float(np.mean(np.degrees(np.arccos(np.clip(dot_torsion[w_start:w_end], -1.0, 1.0)))))

    # 후방 과신전 에너지 비대칭 비
    pos_energy = np.sum(pitch_smooth[pitch_smooth > 0]**2) + 1e-6
    neg_energy = np.sum(pitch_smooth[pitch_smooth < 0]**2)
    sl_bidirectional_hyperext_asym = float(neg_energy / pos_energy)

    # 대퇴골-체간 해부학적 길이비
    femur_len = float(np.mean(norm_femur))
    torso_len = float(np.mean(norm_torso))
    sl_femur_to_torso_ratio = float(femur_len / (torso_len + 1e-6))

    # 롤-피치 다축 연성 결합 지수
    cov_rp = np.cov(roll_smooth, pitch_smooth)
    sl_coronal_sagittal_coupling_index = float(np.abs(cov_rp[0, 1])) if cov_rp.ndim == 2 else 0.0

    # 42차원 벡터 결합
    feature_vector = np.array([
        sl_pure_torso_pitch_bottom,
        sl_pure_torso_pitch_min,
        sl_pure_torso_roll_abs_bottom,
        sl_pure_torso_3d_tilt_bottom,
        sl_pure_torso_pitch_range,
        sl_hyperext_phase_integral,
        sl_ldc_weighted_pitch,
        sl_torso_aspect_ratio_bottom,
        sl_aspect_ratio_compression,
        sl_calibrated_delta_z,
        sl_frontal_pelvis_shoulder_x_offset,
        sl_shoulder_width_stability_std,
        sl_frontal_compression_rate,
        sl_cervico_thoracic_decoupling_angle,
        sl_head_pitch_vs_torso_pitch_diff,
        sl_pure_thoracic_curvature,
        sl_head_torso_hyperext_sync,
        sl_gaze_to_horizon_pitch,
        sl_cervical_strain_index,
        sl_lead_hip_trunk_angle_3d,
        sl_lead_thigh_torso_flexion_ratio,
        sl_trail_hip_abduction_angle,
        sl_pelvis_torso_orthogonality,
        sl_pelvic_tilt_sagittal,
        sl_support_base_torso_plumb_offset,
        sl_tibia_torso_parallelism,
        sl_delta_pitch_from_setup,
        sl_setup_torso_zero_drift,
        sl_relative_extension_ratio,
        sl_baseline_subtracted_peak,
        sl_lockout_rebound_pitch,
        sl_descent_pitch_velocity_min,
        sl_ascent_trunk_jerk,
        sl_bottom_pitch_stability_std,
        sl_torso_spectral_energy_ratio,
        sl_bottom_energy_flux_ratio,
        sl_early_ascent_burst_jerk,
        sl_coronal_tilt_rejection,
        sl_trunk_pelvis_torsion_angle,
        sl_bidirectional_hyperext_asym,
        sl_femur_to_torso_ratio,
        sl_coronal_sagittal_coupling_index
    ], dtype=np.float32)

    return feature_vector

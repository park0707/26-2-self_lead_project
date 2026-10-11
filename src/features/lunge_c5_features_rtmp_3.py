# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c5_features_rtmp_3.py
Purpose: RTMP 2D Pose 기반 c5 (상체 과도한 숙임/젖힘) 초최적화 피처 추출 모듈 (bl5_rtmp_3)
         - 3D Z축 소실 및 다중공선성(Multicollinearity) 저해 피처 전수 제거 (6개 제거)
         - 73차원 정밀 피처셋: 2D 기하 앵커, 원근 수축비, 어깨-골반 속도비, 상호작용 피처
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
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
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

LUNGE_C5_RTMP_3_FEATURE_NAMES: List[str] = [
    # 1. Base 비전 피처 (Z축 노이즈/중복 제거 31D)
    'lunge_vspa_torso_pitch_bottom', 'lunge_vspa_torso_pitch_peak', 'lunge_vspa_torso_pitch_min',
    'lunge_farrokhi_flexion_excess', 'lunge_farrokhi_hyperext_excess', 'lunge_vspa_pitch_velocity_peak',
    'lunge_ldc_phase_weighted_pitch', 'lunge_tibia_torso_parallelism', 'lunge_hip_knee_flexion_ratio',
    'lunge_trunk_thigh_angle_min', 'lunge_hip_extension_deficit', 'lunge_torso_foreshortening_ratio',
    'lunge_calibrated_delta_z', 'lunge_coronal_tilt_rejection', 'lunge_frontal_compression_integral',
    'lunge_cervico_thoracic_angle', 'lunge_lockout_rebound_pitch', 'lunge_ascent_trunk_jerk',
    'lunge_bottom_pitch_stability_std', 'lunge_descent_pitch_smoothness', 'lunge_delta_pitch_from_setup',
    'lunge_setup_torso_zero_drift', 'lunge_relative_extension_ratio', 'lunge_baseline_subtracted_peak',
    'lunge_femur_to_torso_ratio', 'lunge_bidirectional_hyperext_asym', 'lunge_early_ascent_burst_jerk',
    'lunge_trunk_thigh_crp_variability', 'lunge_torso_spectral_energy_ratio', 'lunge_thoracic_curvature_deviation',
    'lunge_bottom_energy_flux_ratio',
    
    # 2. Pure 2D 기하 앵커 및 동역학 피처 (36D)
    'v_trunk_x', 'v_trunk_y', 'v_thigh_lead_x', 'v_thigh_lead_y',
    'cp_trunk_thigh', 'dp_trunk_thigh', 'v_disp_shoulder_y', 'v_disp_hip_y',
    'trunk_compression', 'thigh_compression', 'shoulder_y_jerk', 'crp_2d_variability',
    'knee_toe_x_dist_ratio', 'knee_hip_x_dist_ratio', 'torso_angle_2d_bottom', 'torso_angle_2d_change',
    'spectral_energy_ratio', 'bottom_energy_flux_ratio_y', 'cervico_thoracic_2d_angle', 'hip_extension_deficit_2d',
    'trunk_knee_moment_arm_2d', 'lockout_rebound_y', 'setup_torso_zero_drift', 'baseline_subtracted_peak_y',
    'femur_to_torso_ratio', 'bidirectional_hyperext_asym_y', 'tibia_torso_parallelism_2d', 'v_neck_pelvis_x',
    'v_thigh_rear_x', 'v_thigh_rear_y', 'cp_trunk_rear', 'dp_trunk_rear',
    'v_disp_shoulder_x', 'v_disp_hip_x', 'shoulder_velocity_y_max', 'hip_velocity_y_max',
    
    # 3. 6D 상호작용 피처 (Interaction Features)
    'trunk_to_thigh_compression_ratio', 'shoulder_to_hip_velocity_ratio', 'cervico_angle_x_trunk_comp',
    'torso_angle_minus_tibia_parallel', 'shoulder_hip_x_disp_diff', 'vspa_peak_x_trunk_comp'
]

def map_halpe26_to_mp33(kps26: np.ndarray) -> np.ndarray:
    T = kps26.shape[0]
    kps33 = np.zeros((T, 33, 3), dtype=np.float32)
    kps33[:, 0] = kps26[:, 0]
    kps33[:, 2] = kps26[:, 1]
    kps33[:, 5] = kps26[:, 2]
    kps33[:, 7] = kps26[:, 3]
    kps33[:, 8] = kps26[:, 4]
    kps33[:, 11] = kps26[:, 5]
    kps33[:, 12] = kps26[:, 6]
    kps33[:, 13] = kps26[:, 7]
    kps33[:, 14] = kps26[:, 8]
    kps33[:, 15] = kps26[:, 9]
    kps33[:, 16] = kps26[:, 10]
    kps33[:, 23] = kps26[:, 11]
    kps33[:, 24] = kps26[:, 12]
    kps33[:, 25] = kps26[:, 13]
    kps33[:, 26] = kps26[:, 14]
    kps33[:, 27] = kps26[:, 15]
    kps33[:, 28] = kps26[:, 16]
    kps33[:, 29] = kps26[:, 24] # LEFT_HEEL
    kps33[:, 30] = kps26[:, 25] # RIGHT_HEEL
    kps33[:, 31] = kps26[:, 20] # LEFT_FOOT_INDEX
    kps33[:, 32] = kps26[:, 21] # RIGHT_FOOT_INDEX
    
    left_mp_indices = [1,2,3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4,5,6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33

def extract_lunge_c5_rtmp_3_features(kps_seq: np.ndarray) -> np.ndarray:
    """
    RTMP 2D 키포인트로부터 73차원 bl5_rtmp_3 초최적화 피처 벡터 추출
    kps_seq: (64, 26, 3) 또는 (64, 33, 3)
    """
    if kps_seq.shape[1] == 26:
        kps33 = map_halpe26_to_mp33(kps_seq)
    else:
        kps33 = kps_seq.copy()
        if np.all(kps33[:, :, 2] == 0):
            left_mp_indices = [1,2,3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
            right_mp_indices = [4,5,6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            kps33[:, left_mp_indices, 2] = -0.1
            kps33[:, right_mp_indices, 2] = 0.1

    from src.features.lunge_c5_features_v5 import extract_lunge_c5_v5_feature_vector, LUNGE_C5_V5_FEATURE_NAMES
    base_36d = extract_lunge_c5_v5_feature_vector(kps33)

    from src.features.lunge_c5_features_rtmp_2 import detect_lead_leg_2d, normalize_2d, vector_angle_2d
    
    kps = kps33[:, :, :2].astype(np.float32)
    T = kps.shape[0]

    lead_side = detect_lead_leg_2d(kps33)
    if lead_side == 'left':
        lead_hip, lead_knee, lead_ankle, lead_toe = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_FOOT_INDEX
        rear_hip, rear_knee = MP_RIGHT_HIP, MP_RIGHT_KNEE
    else:
        lead_hip, lead_knee, lead_ankle, lead_toe = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, MP_RIGHT_FOOT_INDEX
        rear_hip, rear_knee = MP_LEFT_HIP, MP_LEFT_KNEE

    pelvis_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    smooth_y = gaussian_filter1d(pelvis_y, sigma=2.0)
    t_bot = int(np.clip(int(np.argmax(smooth_y)), 10, T - 10))
    win_start, win_end = max(0, t_bot - 4), min(T, t_bot + 5)

    shoulders = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hips = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    head = (kps[:, MP_LEFT_EAR] + kps[:, MP_RIGHT_EAR]) / 2.0

    torso_lengths = np.linalg.norm(shoulders - hips, axis=1)
    l0_reference = float(np.mean(torso_lengths[:8])) + 1e-6
    lead_thigh_lengths = np.linalg.norm(kps[:, lead_knee] - kps[:, lead_hip], axis=1)
    thigh0_reference = float(np.mean(lead_thigh_lengths[:8])) + 1e-6

    kps_bot = np.mean(kps[win_start:win_end], axis=0)
    kps_std = np.mean(kps[:8], axis=0)

    sh_bot = (kps_bot[MP_LEFT_SHOULDER] + kps_bot[MP_RIGHT_SHOULDER]) / 2.0
    hip_bot = (kps_bot[MP_LEFT_HIP] + kps_bot[MP_RIGHT_HIP]) / 2.0
    head_bot = (kps_bot[MP_LEFT_EAR] + kps_bot[MP_RIGHT_EAR]) / 2.0

    sh_std = (kps_std[MP_LEFT_SHOULDER] + kps_std[MP_RIGHT_SHOULDER]) / 2.0
    hip_std = (kps_std[MP_LEFT_HIP] + kps_std[MP_RIGHT_HIP]) / 2.0

    v_trunk = normalize_2d(sh_bot - hip_bot)
    v_thigh_lead = normalize_2d(kps_bot[lead_knee] - kps_bot[lead_hip])
    v_thigh_rear = normalize_2d(kps_bot[rear_knee] - kps_bot[rear_hip])
    v_neck_pelvis = normalize_2d(head_bot - hip_bot)

    cp_trunk_thigh = float(v_trunk[0]*v_thigh_lead[1] - v_trunk[1]*v_thigh_lead[0])
    dp_trunk_thigh = float(v_trunk[0]*v_thigh_lead[0] + v_trunk[1]*v_thigh_lead[1])
    cp_trunk_rear = float(v_trunk[0]*v_thigh_rear[1] - v_trunk[1]*v_thigh_rear[0])
    dp_trunk_rear = float(v_trunk[0]*v_thigh_rear[0] + v_trunk[1]*v_thigh_rear[1])

    v_disp_sh = sh_bot - sh_std
    v_disp_hip = hip_bot - hip_std

    trunk_comp = float(np.linalg.norm(sh_bot - hip_bot) / l0_reference)
    thigh_comp = float(np.linalg.norm(kps_bot[lead_knee] - kps_bot[lead_hip]) / thigh0_reference)

    shoulder_y = gaussian_filter1d(shoulders[:, 1], sigma=1.5)
    hip_y = gaussian_filter1d(hips[:, 1], sigma=1.5)
    vel_y = np.gradient(shoulder_y)
    vel_hip_y = np.gradient(hip_y)
    jerk_y = np.gradient(np.gradient(vel_y))

    sh_jerk = float(np.max(np.abs(jerk_y[max(0, t_bot-5):min(T, t_bot+10)])))
    sh_vel_max = float(np.max(np.abs(vel_y)))
    hip_vel_max = float(np.max(np.abs(vel_hip_y)))

    # 2D CRP
    t_norm = (2.0 * (torso_lengths - np.min(torso_lengths)) / (np.max(torso_lengths) - np.min(torso_lengths) + 1e-6)) - 1.0
    th_norm = (2.0 * (lead_thigh_lengths - np.min(lead_thigh_lengths)) / (np.max(lead_thigh_lengths) - np.min(lead_thigh_lengths) + 1e-6)) - 1.0
    vt_norm = np.gradient(torso_lengths) / (np.max(np.abs(np.gradient(torso_lengths))) + 1e-6)
    vth_norm = np.gradient(lead_thigh_lengths) / (np.max(np.abs(np.gradient(lead_thigh_lengths))) + 1e-6)
    crp = np.abs(np.arctan2(vt_norm, t_norm) - np.arctan2(vth_norm, th_norm))
    crp = np.where(crp > np.pi, 2 * np.pi - crp, crp)
    crp_var = float(np.std(np.degrees(crp)[max(0, t_bot-10):min(T, t_bot+10)]))

    knee_toe_x_dist = float(np.abs(kps_bot[lead_knee, 0] - kps_bot[lead_toe, 0]) / l0_reference)
    knee_hip_x_dist = float(np.abs(kps_bot[lead_knee, 0] - hip_bot[0]) / l0_reference)

    def get_ang(p1, p2):
        vec = p1 - p2
        return np.degrees(np.arctan2(vec[:, 0], vec[:, 1]))

    torso_angs = get_ang(hips, shoulders)
    torso_ang_bot = float(np.mean(torso_angs[win_start:win_end]))
    torso_ang_chg = abs(torso_ang_bot - float(np.mean(torso_angs[:8])))

    # Spectral
    zero_sy = shoulder_y - np.mean(shoulder_y)
    fft_v = fft(zero_sy)
    freqs = fftfreq(T, d=1.0/30.0)[:T//2]
    pwr = np.abs(fft_v[:T//2])**2
    p_low = float(np.sum(pwr[(freqs < 1.5) & (freqs > 0.1)])) + 1e-5
    p_high = float(np.sum(pwr[(freqs >= 3.0) & (freqs <= 10.0)]))
    spec_ratio = p_high / p_low

    kin_e = vel_y**2
    bot_e = float(np.sum(kin_e[max(0, t_bot-4):min(T, t_bot+5)]))
    flux_ratio = bot_e / (float(np.sum(kin_e)) + 1e-5)

    cervico_angle = vector_angle_2d(head_bot - sh_bot, sh_bot - hip_bot)
    hip_ext_deficit = vector_angle_2d(sh_bot - hip_bot, kps_bot[rear_knee] - kps_bot[rear_hip])

    dx_trunk = abs(sh_bot[0] - hip_bot[0])
    dx_knee = abs(kps_bot[lead_knee, 0] - hip_bot[0])
    moment_arm_2d = float(dx_trunk / (dx_knee + 1e-4))

    lockout_reb = float(np.mean(shoulders[-8:, 1] - shoulders[:8, 1]))
    zero_drift = float(np.mean(np.abs(torso_lengths - l0_reference)) / l0_reference)
    base_sub_peak = float(np.max(np.abs(shoulder_y - float(np.mean(shoulders[:8, 1])))))
    femur_torso_r = float(thigh0_reference / l0_reference)

    dy = shoulder_y - float(np.mean(shoulders[:8, 1]))
    hyperext_asym = float(np.sum(np.maximum(0, -dy))) / (float(np.sum(np.abs(dy))) + 1e-4)

    tibia_angs = get_ang(kps[:, lead_knee], kps[:, lead_ankle])
    tibia_parallel = abs(torso_ang_bot - float(np.mean(tibia_angs[win_start:win_end])))

    pure_2d_36d = [
        float(v_trunk[0]), float(v_trunk[1]), float(v_thigh_lead[0]), float(v_thigh_lead[1]),
        cp_trunk_thigh, dp_trunk_thigh, float(v_disp_sh[1]), float(v_disp_hip[1]),
        trunk_comp, thigh_comp, sh_jerk, crp_var,
        knee_toe_x_dist, knee_hip_x_dist, torso_ang_bot, torso_ang_chg,
        spec_ratio, flux_ratio, cervico_angle, hip_ext_deficit,
        moment_arm_2d, lockout_reb, zero_drift, base_sub_peak,
        femur_torso_r, hyperext_asym, tibia_parallel, float(v_neck_pelvis[0]),
        float(v_thigh_rear[0]), float(v_thigh_rear[1]), cp_trunk_rear, dp_trunk_rear,
        float(v_disp_sh[0]), float(v_disp_hip[0]), sh_vel_max, hip_vel_max
    ]

    # Interaction 6D
    inter_6d = [
        trunk_comp / (thigh_comp + 1e-4),
        sh_vel_max / (hip_vel_max + 1e-4),
        cervico_angle * trunk_comp,
        torso_ang_bot - tibia_parallel,
        abs(float(v_disp_sh[0]) - float(v_disp_hip[0])),
        base_36d[1] * trunk_comp
    ]

    # Combine Base 31D (excluding 5 dead indices: 9, 10, 14, 25, 26) + 36D + 6D = 73D
    # Dead indices in Base 36D:
    # 9: lunge_pelvic_tilt_sagittal, 10: lunge_trunk_knee_moment_arm, 14: lunge_virtual_sagittal_pitch_3d,
    # 25: lunge_head_torso_hyperext_sync, 26: lunge_pelvis_spine_decoupling_angle
    dead_base_idx = {9, 10, 14, 25, 26}
    base_31d = [base_36d[i] for i in range(len(base_36d)) if i not in dead_base_idx]

    feat_73d = np.array(base_31d + pure_2d_36d + inter_6d, dtype=np.float32)
    return np.nan_to_num(feat_73d, nan=0.0, posinf=0.0, neginf=0.0)


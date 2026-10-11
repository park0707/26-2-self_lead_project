# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c5_features_hybrid.py
Purpose: MediaPipe 3D + RTMPose 2D 하이브리드 피처 추출 모듈 (bl5_hybrid_1)
         - MediaPipe 36D (글로벌 3D 배향, VSPA 시상면 피치, 3D 모멘트암, 위상 적분)
         - RTMPose 37D (초정밀 2D 키포인트 궤적, 수직 저크, 동적 압축비, 스펙트럼 에너지 비)
         - 6D 크로스 모달 상호작용 피처 (Cross-Modal Interactions)
         - 총 79차원 하이브리드 결합 피처셋
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.features.lunge_c5_features_v5 import (
    extract_lunge_c5_v5_feature_vector,
    LUNGE_C5_V5_FEATURE_NAMES
)
from src.features.lunge_c5_features_rtmp_3 import (
    extract_lunge_c5_rtmp_3_features,
    map_halpe26_to_mp33
)

# RTMP 37D feature names
RTMP_2D_37D_NAMES = [
    'v_trunk_x', 'v_trunk_y', 'v_thigh_lead_x', 'v_thigh_lead_y',
    'cp_trunk_thigh', 'dp_trunk_thigh', 'v_disp_shoulder_y', 'v_disp_hip_y',
    'trunk_compression', 'thigh_compression', 'shoulder_y_jerk', 'crp_2d_variability',
    'knee_toe_x_dist_ratio', 'knee_hip_x_dist_ratio', 'torso_angle_2d_bottom', 'torso_angle_2d_change',
    'spectral_energy_ratio', 'bottom_energy_flux_ratio_y', 'cervico_thoracic_2d_angle', 'hip_extension_deficit_2d',
    'trunk_knee_moment_arm_2d', 'lockout_rebound_y', 'setup_torso_zero_drift', 'baseline_subtracted_peak_y',
    'femur_to_torso_ratio', 'bidirectional_hyperext_asym_y', 'tibia_torso_parallelism_2d', 'v_neck_pelvis_x',
    'v_thigh_rear_x', 'v_thigh_rear_y', 'cp_trunk_rear', 'dp_trunk_rear',
    'v_disp_shoulder_x', 'v_disp_hip_x', 'shoulder_velocity_y_max', 'hip_velocity_y_max',
    'rtmp_trunk_extra_ratio'
]

CROSS_MODAL_6D_NAMES = [
    'cross_mp_pitch_rtmp_angle_diff',           # MP 3D 피치각 - RTMP 2D 겉보기 각도 차이 (3D 면외 회전 포착)
    'cross_mp_foreshortening_rtmp_comp_ratio',  # MP 투영 단축비 / RTMP 체간 압축비 (원근 왜곡 보정)
    'cross_mp_peak_pitch_rtmp_comp_mult',       # MP 피크 피치각 * RTMP 체간 압축비 (결함 모멘텀 상승 결합)
    'cross_mp_pitch_vel_rtmp_sh_vel_diff',      # |MP 각속도 - RTMP 어깨 수직 속도| (모달 간 동적 불일치도)
    'cross_rtmp_trunk_thigh_comp_ratio',        # RTMP 체간/대퇴 압축비
    'cross_rtmp_sh_hip_vel_ratio'               # RTMP 어깨/골반 수직 속도비
]

LUNGE_C5_HYBRID_79D_NAMES: List[str] = (
    [f"MP_{f}" for f in LUNGE_C5_V5_FEATURE_NAMES] +
    [f"RTMP_{f}" for f in RTMP_2D_37D_NAMES] +
    CROSS_MODAL_6D_NAMES
)

def compute_cross_interactions(feat_mp_36d: np.ndarray, feat_rtmp_37d: np.ndarray) -> np.ndarray:
    """
    Computes 6D cross-modal interaction features.
    feat_mp_36d: (..., 36)
    feat_rtmp_37d: (..., 37)
    """
    is_1d = (feat_mp_36d.ndim == 1)
    if is_1d:
        feat_mp_36d = feat_mp_36d.reshape(1, -1)
        feat_rtmp_37d = feat_rtmp_37d.reshape(1, -1)

    # Indices in MP 36D:
    # 0: lunge_vspa_torso_pitch_bottom
    # 1: lunge_vspa_torso_pitch_peak
    # 5: lunge_vspa_pitch_velocity_peak
    # 11: lunge_torso_foreshortening_ratio

    # Indices in RTMP 37D:
    # 8: trunk_compression
    # 9: thigh_compression
    # 14: torso_angle_2d_bottom
    # 34: shoulder_velocity_y_max
    # 35: hip_velocity_y_max

    c1 = feat_mp_36d[:, 0] - feat_rtmp_37d[:, 14]
    c2 = feat_mp_36d[:, 11] / (feat_rtmp_37d[:, 8] + 1e-4)
    c3 = feat_mp_36d[:, 1] * feat_rtmp_37d[:, 8]
    c4 = np.abs(feat_mp_36d[:, 5] - feat_rtmp_37d[:, 34])
    c5 = feat_rtmp_37d[:, 8] / (feat_rtmp_37d[:, 9] + 1e-4)
    c6 = feat_rtmp_37d[:, 34] / (feat_rtmp_37d[:, 35] + 1e-4)

    cross = np.column_stack([c1, c2, c3, c4, c5, c6])
    if is_1d:
        return cross[0]
    return cross

def extract_lunge_c5_hybrid_features(
    mp_kps_33_3d: np.ndarray,
    rtmp_kps_26_2d: np.ndarray
) -> np.ndarray:
    """
    Extracts 79D Hybrid Features from MediaPipe 3D and RTMPose 2D sequences.
    mp_kps_33_3d: (64, 33, 3) from MediaPipe
    rtmp_kps_26_2d: (64, 26, 2) or (64, 26, 3) from RTMPose
    """
    # 1. MediaPipe 36D
    mp_feats = extract_lunge_c5_v5_feature_vector(mp_kps_33_3d)

    # 2. RTMP 37D
    # Map to MP33 format and call RTMP feature extraction
    if rtmp_kps_26_2d.shape[1] == 26:
        rtmp_mp = map_halpe26_to_mp33(rtmp_kps_26_2d)
    else:
        rtmp_mp = rtmp_kps_26_2d
    rtmp_feats_all = extract_lunge_c5_rtmp_3_features(rtmp_mp)
    # The 37D pure 2D features are located at indices [31:68] of the 73D RTMP array
    # If using pure 2D extractor or matching the cached 37D features
    rtmp_37d = rtmp_feats_all[31:68]

    # 3. Cross-Modal 6D
    cross_6d = compute_cross_interactions(mp_feats, rtmp_37d)

    # 4. Concatenate (79D)
    hybrid_79d = np.concatenate([mp_feats, rtmp_37d, cross_6d]).astype(np.float32)
    return np.nan_to_num(hybrid_79d, nan=0.0, posinf=0.0, neginf=0.0)

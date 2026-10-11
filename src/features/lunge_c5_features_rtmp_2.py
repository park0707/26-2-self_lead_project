# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c5_features_rtmp_2.py
Purpose: RTMP 2D Pose 기반 c5 (상체 과도한 숙임/젖힘) 고도화 피처 추출 모듈 (bl5_rtmp_2)
         - 3D Z축 소실 극복을 위한 2D 기하 앵커 벡터 및 키네마틱 역학 융합 (79D)
         - 체간-대퇴 2D 압축비, 어깨-골반 Y축 동역학, 대퇴-경골 상호작용 피처 결합
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple

# MediaPipe 33 Landmark Indices (RTMP mapped)
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

def vector_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2))
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def normalize_2d(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return (v / norm).astype(np.float32)

def detect_lead_leg_2d(kps: np.ndarray) -> str:
    T = kps.shape[0]
    l_ankle = kps[:, MP_LEFT_ANKLE, :2]
    r_ankle = kps[:, MP_RIGHT_ANKLE, :2]
    pelvis = (kps[:, MP_LEFT_HIP, :2] + kps[:, MP_RIGHT_HIP, :2]) / 2.0
    l_disp = np.linalg.norm(l_ankle - pelvis, axis=1)
    r_disp = np.linalg.norm(r_ankle - pelvis, axis=1)
    mid_start, mid_end = int(T * 0.3), int(T * 0.7)
    l_disp_mean = float(np.mean(l_disp[mid_start:mid_end]))
    r_disp_mean = float(np.mean(r_disp[mid_start:mid_end]))
    return 'left' if l_disp_mean >= r_disp_mean else 'right'

def map_halpe26_to_mp33(kps26: np.ndarray) -> np.ndarray:
    """Halpe 26 키포인트를 MediaPipe 33 포맷으로 변환"""
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
    
    # 3D Cross-Product 안전 주입
    left_mp_indices = [1,2,3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4,5,6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33

def extract_lunge_c5_rtmp_2_features(kps_seq: np.ndarray) -> np.ndarray:
    """
    RTMP 2D 키포인트로부터 79차원 bl5_rtmp_2 고도화 피처 벡터 추출
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

    # 1. Base 36D (lunge_c5_features_v5)
    from src.features.lunge_c5_features_v5 import extract_lunge_c5_v5_feature_vector
    base_36d = extract_lunge_c5_v5_feature_vector(kps33)

    # 2. Pure 2D 37D (lunge_c5_features_rtmp_3)
    from src.features.lunge_c5_features_rtmp_3 import extract_lunge_c5_features_rtmp_3_vector
    pure_2d_37d = extract_lunge_c5_features_rtmp_3_vector(kps33)

    # 3. 6D Interaction Features
    # pure_2d: 8: trunk_compression, 9: thigh_compression, 14: torso_angle_2d_bottom, 18: cervico_angle, 
    #          26: tibia_torso_parallelism, 33: v_disp_shoulder_x, 34: v_disp_hip_x, 35: shoulder_vel_y, 36: hip_vel_y
    extra_6d = np.array([
        pure_2d_37d[8] / (pure_2d_37d[9] + 1e-4),
        pure_2d_37d[35] / (pure_2d_37d[36] + 1e-4),
        pure_2d_37d[18] * pure_2d_37d[8],
        pure_2d_37d[14] - pure_2d_37d[26],
        abs(pure_2d_37d[33] - pure_2d_37d[34]),
        base_36d[1] * pure_2d_37d[8]
    ], dtype=np.float32)

    feat_79d = np.concatenate([base_36d, pure_2d_37d, extra_6d])
    return np.nan_to_num(feat_79d, nan=0.0, posinf=0.0, neginf=0.0)


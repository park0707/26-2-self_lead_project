# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_bl3_features_v5.py
Purpose: v4 14D 피처에 DTW(Dynamic Time Warping) 기반의 정석 궤적 템플릿 매칭 피처를 추가한 
         v5 차세대 고급 시계열 피처 추출기.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import List

# 기존 v4 피처 계산 로직 재사용 (중복 코드 최소화)
import sys
sys.path.insert(0, 'd:/')
from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_18d_features_v4, calculate_angle_2d

FEATURE_NAMES_19D: List[str] = [
    # v3 직교 정예 11D
    'rear_knee_stability_std',
    'lead_rear_knee_ratio',
    'rear_femur_vertical_pitch',
    'rear_shank_horizontal_pitch',
    'inter_ankle_stride_bottom',
    'knee_velocity_decel_nadir',
    'rear_knee_target_dev_bottom',
    'inter_knee_vertical_disp',
    'torso_to_rear_femur_angle',
    'lead_knee_angle_bottom',
    'rear_knee_angle_bottom',
    # v4 고급 생체역학 7D
    'knee_hysteresis_area',
    'cam_aspect_ratio',
    'femur_tibia_proj_ratio',
    'pseudo_3d_knee_angle',
    'crp_hip_knee_mean',
    'xcorr_hip_knee_lag',
    'phase_time_ratio',
    # v5 신규 피처
    'dtw_gold_standard_dist'         # 19. 정석 무릎 궤적 템플릿과의 DTW 거리
]

# 15D V5 정예 피처셋 (v4 14D + DTW)
FEATURE_INDICES_15D: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18]
FEATURE_NAMES_15D: List[str] = [FEATURE_NAMES_19D[i] for i in FEATURE_INDICES_15D]

def calc_dtw_distance_1d(s1: np.ndarray, s2: np.ndarray) -> float:
    """간단한 1D DTW (Dynamic Time Warping) 거리 계산기"""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Calculate distances
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            
    return dtw_matrix[n, m]

def get_gold_standard_knee_trajectory(T=64):
    """정석적인 런지 뒤다리 무릎 각도 궤적 (170도 -> 90도 -> 170도)"""
    t = np.linspace(0, np.pi * 2, T)
    # Cosine 곡선: 130 + 40*cos(t) -> 시작 170, 중간 90, 끝 170
    gold_traj = 130.0 + 40.0 * np.cos(t)
    return gold_traj

def extract_lunge_bl3_19d_features_v5(kps: np.ndarray) -> np.ndarray:
    """
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (19,) 또는 (N, 19)
    """
    is_batch = (kps.ndim == 4)
    if not is_batch:
        kps = kps[None, ...] # (1, 64, 15, 2)
        
    N, T, J, C = kps.shape
    
    # 1. v4 18D 피처 추출
    feats_18d = extract_lunge_bl3_18d_features_v4(kps) # (N, 18)
    
    # 2. 뒤다리 무릎 각도 시계열 재계산 (DTW용)
    hip_l = kps[:, :, 8, :]
    hip_r = kps[:, :, 9, :]
    knee_l = kps[:, :, 10, :]
    knee_r = kps[:, :, 11, :]
    ank_l = kps[:, :, 12, :]
    ank_r = kps[:, :, 13, :]
    
    vec_femur_l = knee_l - hip_l
    vec_shank_l = ank_l - knee_l
    ang_l_t = 180.0 - calculate_angle_2d(vec_femur_l, vec_shank_l)
    
    vec_femur_r = knee_r - hip_r
    vec_shank_r = ank_r - knee_r
    ang_r_t = 180.0 - calculate_angle_2d(vec_femur_r, vec_shank_r)
    
    if T >= 7:
        ang_l_t = savgol_filter(ang_l_t, window_length=7, polyorder=2, axis=1)
        ang_r_t = savgol_filter(ang_r_t, window_length=7, polyorder=2, axis=1)
        
    sum_knee_ang = ang_l_t + ang_r_t
    bottom_idx = np.argmin(sum_knee_ang, axis=1)
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1]
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1]
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None]
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t) # (N, T)
    
    # 3. Gold Standard DTW 거리 계산
    gold_traj = get_gold_standard_knee_trajectory(T)
    dtw_dists = []
    for i in range(N):
        dist = calc_dtw_distance_1d(rear_knee_ang_t[i], gold_traj)
        # 길이에 대한 정규화
        dtw_dists.append(dist / T)
        
    f19_dtw = np.array(dtw_dists, dtype=np.float32).reshape(N, 1)
    
    # 4. 결합
    feats_19d = np.concatenate([feats_18d, f19_dtw], axis=1)
    
    if not is_batch:
        return feats_19d[0]
    return feats_19d

def extract_lunge_bl3_15d_features_v5(kps_seq: np.ndarray) -> np.ndarray:
    """
    15D 정예 고급 기구학 피처 추출 (v5 챔피언 후보)
    - v4 14D + DTW Gold Standard Distance
    """
    feats_19d = extract_lunge_bl3_19d_features_v5(kps_seq)
    if feats_19d.ndim == 1:
        return feats_19d[FEATURE_INDICES_15D]
    return feats_19d[:, FEATURE_INDICES_15D]

if __name__ == "__main__":
    dummy = np.zeros((64, 15, 2), dtype=np.float32)
    out19 = extract_lunge_bl3_19d_features_v5(dummy)
    out15 = extract_lunge_bl3_15d_features_v5(dummy)
    print("V5 15D Feature names:", FEATURE_NAMES_15D)
    print("Shape:", out15.shape)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c3_rear_knee_features_v2.py
Purpose: 선행 연구(Farrokhi et al. JOSPT, NSCA 런지 생체역학, OpenCap 단안 포즈 추정)를 반영한
         바벨 런지 c3(뒤다리 무릎 각도 90도) 18D 고도화 생체역학 피처 추출기 (v2)
         - 1) 사비츠키-골레이(Savitzky-Golay) 필터링 기반 시계열 지터 제거 및 위상 정밀 고정
         - 2) v1 14D 피처 100% 계승 + 4대 도메인 특화 피처 추가 (총 18D):
              f15: rear_hip_extension_angle (뒤다리 고관절 신전각)
              f16: double_90_bilateral_residual (앞뒤 더블 90도 동시 적중 조화 잔차)
              f17: knee_velocity_decel_nadir (최하사점 전환기 감속도 프로파일)
              f18: stride_to_pelvis_width_ratio (골반 너비 대비 보폭 정규화비)
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any

FEATURE_NAMES_18D: List[str] = [
    # --- v1 계승 14D 피처 ---
    'rear_knee_angle_bottom',        # 1. [최하사점 윈도우] 뒤다리 무릎 2D/3D 사이각 (°)
    'rear_knee_angle_mean',          # 2. 전체 시퀀스 뒤다리 무릎 각도 평균 (°)
    'rear_knee_angle_rom',           # 3. 뒤다리 무릎 관절 가동범위 (max - min) (°)
    'rear_knee_target_dev_bottom',   # 4. [최하사점] 90도 기준 뒤다리 무릎 각도 절대 오차 (|theta - 90|) (°)
    'rear_knee_target_dev_mean',     # 5. 전체 시퀀스 90도 기준 뒤다리 무릎 각도 평균 절대 오차 (°)
    'rear_femur_vertical_pitch',     # 6. [최하사점] 뒤다리 대퇴골의 지면 수직선 대비 피치 각도 (°)
    'rear_shank_horizontal_pitch',   # 7. [최하사점] 뒤다리 정강이의 지면 수평선 대비 각도 (°)
    'lead_rear_knee_ratio',          # 8. [최하사점] 앞다리 무릎 각도 대비 뒤다리 무릎 각도 비율 (theta_lead / theta_rear)
    'lead_knee_angle_bottom',        # 9. [최하사점 윈도우] 앞다리 무릎 각도 (°)
    'inter_knee_vertical_disp',      # 10. [최하사점] 앞뒤 무릎 간 수직 높이차 (정규화 단위)
    'inter_ankle_stride_bottom',     # 11. [최하사점] 양 발목 간 2D 유클리드 거리 (보폭, 정규화 단위)
    'pelvis_drop_to_stride_ratio',   # 12. 최하사점 골반 하강 깊이 대비 보폭 비율
    'rear_knee_stability_std',       # 13. [최하사점 윈도우] 뒤다리 무릎 각도 표준편차 (관절 안정성/떨림)
    'torso_to_rear_femur_angle',     # 14. [최하사점] 체간 축과 뒤다리 대퇴골 간 2D 사잇각 (고관절 신전도) (°)
    # --- v2 신규 4D 선행 연구 기반 특화 피처 ---
    'rear_hip_extension_angle',      # 15. [Farrokhi et al.] 뒤다리 고관절 신전 각도 (Torso-Pelvis vs Pelvis-Knee_rear) (°)
    'double_90_bilateral_residual',  # 16. [NSCA] 앞뒤 무릎 더블 90도 동시 적중 조화 잔차
    'knee_velocity_decel_nadir',     # 17. [OpenCap] 최하사점 직전 뒤다리 무릎 각속도 감속도 (Jerk / 충격 완화)
    'stride_to_pelvis_width_ratio'   # 18. 신체 비례(골반 너비) 정규화 보폭비
]

def calculate_angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 2D 벡터 간의 사이각 (0° ~ 180°) 계산"""
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_lunge_c3_18d_features_v2(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 정규화 관절 배열로부터 c3 v2 18D 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (18,) 또는 (N, 18)
    """
    is_batch = (kps.ndim == 4)
    if not is_batch:
        kps = kps[None, ...] # (1, 64, 15, 2)
        
    N, T, J, C = kps.shape
    
    # 0:PELVIS, 1:TORSO, 8:LEFT_HIP, 9:RIGHT_HIP, 10:LEFT_KNEE, 11:RIGHT_KNEE, 12:LEFT_ANKLE, 13:RIGHT_ANKLE, 14:NECK
    pelvis = kps[:, :, 0, :]   # (N, T, 2)
    torso = kps[:, :, 1, :]    # (N, T, 2)
    hip_l = kps[:, :, 8, :]    # (N, T, 2)
    hip_r = kps[:, :, 9, :]    # (N, T, 2)
    knee_l = kps[:, :, 10, :]  # (N, T, 2)
    knee_r = kps[:, :, 11, :]  # (N, T, 2)
    ank_l = kps[:, :, 12, :]   # (N, T, 2)
    ank_r = kps[:, :, 13, :]   # (N, T, 2)
    
    # 1. 좌우 무릎 각도 시계열 계산 (0° ~ 180°)
    vec_femur_l = knee_l - hip_l # (N, T, 2)
    vec_shank_l = ank_l - knee_l # (N, T, 2)
    ang_l_t = 180.0 - calculate_angle_2d(vec_femur_l, vec_shank_l) # (N, T)
    
    vec_femur_r = knee_r - hip_r # (N, T, 2)
    vec_shank_r = ank_r - knee_r # (N, T, 2)
    ang_r_t = 180.0 - calculate_angle_2d(vec_femur_r, vec_shank_r) # (N, T)
    
    # [선행연구 기법 1: Savitzky-Golay 평활화] 미세 지터 제거 및 위상 잠금
    if T >= 7:
        ang_l_t_smooth = savgol_filter(ang_l_t, window_length=7, polyorder=2, axis=1)
        ang_r_t_smooth = savgol_filter(ang_r_t, window_length=7, polyorder=2, axis=1)
    else:
        ang_l_t_smooth = ang_l_t
        ang_r_t_smooth = ang_r_t
        
    # 2. 최하사점(Bottom Position) 정밀 탐색
    sum_knee_ang = ang_l_t_smooth + ang_r_t_smooth # (N, T)
    bottom_idx = np.argmin(sum_knee_ang, axis=1) # (N,)
    
    # 최하사점 윈도우 마스크 (t_bottom ± 6 프레임)
    t_grid = np.arange(T)[None, :] # (1, T)
    bottom_idx_2d = bottom_idx[:, None] # (N, 1)
    bottom_mask = (t_grid >= np.maximum(0, bottom_idx_2d - 6)) & (t_grid <= np.minimum(T - 1, bottom_idx_2d + 6)) # (N, T)
    win_count = np.sum(bottom_mask, axis=1, keepdims=True) + 1e-7 # (N, 1)
    
    # 3. 선행 다리 vs 후행/뒤다리 동적 자동 분리
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1] # (N,)
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1] # (N,)
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None] # (N, 1)
    
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t) # (N, T)
    lead_knee_ang_t = np.where(is_left_rear, ang_r_t, ang_l_t) # (N, T)
    
    rear_femur_vec = np.where(is_left_rear[:, :, None], vec_femur_l, vec_femur_r) # (N, T, 2)
    rear_shank_vec = np.where(is_left_rear[:, :, None], vec_shank_l, vec_shank_r) # (N, T, 2)
    rear_knee_pts = np.where(is_left_rear[:, :, None], knee_l, knee_r) # (N, T, 2)
    lead_knee_pts = np.where(is_left_rear[:, :, None], knee_r, knee_l) # (N, T, 2)
    rear_hip_pts = np.where(is_left_rear[:, :, None], hip_l, hip_r) # (N, T, 2)
    
    # 4. v1 14대 특화 생체역학 피처 계산
    f1_rear_knee_bot = np.sum(rear_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    f2_rear_knee_mean = np.mean(rear_knee_ang_t, axis=1, keepdims=True)
    f3_rear_knee_rom = np.ptp(rear_knee_ang_t, axis=1, keepdims=True)
    f4_dev_bot = np.abs(f1_rear_knee_bot - 90.0)
    f5_dev_mean = np.mean(np.abs(rear_knee_ang_t - 90.0), axis=1, keepdims=True)
    
    vert_unit = np.array([0.0, 1.0], dtype=np.float32)[None, None, :]
    femur_pitch_t = calculate_angle_2d(rear_femur_vec, vert_unit)
    f6_femur_pitch = np.sum(femur_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    horiz_unit = np.array([1.0, 0.0], dtype=np.float32)[None, None, :]
    shank_pitch_t = calculate_angle_2d(rear_shank_vec, horiz_unit)
    f7_shank_pitch = np.sum(shank_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    lead_knee_bot = np.sum(lead_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    f8_ratio = lead_knee_bot / (f1_rear_knee_bot + 1e-4)
    f9_lead_knee_bot = lead_knee_bot
    
    knee_v_disp_t = np.abs(lead_knee_pts[:, :, 1] - rear_knee_pts[:, :, 1])
    f10_inter_knee_v = np.sum(knee_v_disp_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    ankle_stride_t = np.linalg.norm(ank_l - ank_r, axis=-1)
    f11_stride = np.sum(ankle_stride_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    pelvis_y = pelvis[:, :, 1]
    init_pelvis_y = np.mean(pelvis_y[:, :5], axis=1, keepdims=True)
    bot_pelvis_y = np.sum(pelvis_y * bottom_mask, axis=1, keepdims=True) / win_count
    pelvis_drop = np.maximum(0.0, bot_pelvis_y - init_pelvis_y)
    f12_drop_stride_ratio = pelvis_drop / (f11_stride + 1e-4)
    
    mean_w = f1_rear_knee_bot
    sq_diff = (rear_knee_ang_t - mean_w)**2 * bottom_mask
    f13_knee_std = np.sqrt(np.sum(sq_diff, axis=1, keepdims=True) / win_count)
    
    torso_axis = torso - pelvis
    torso_femur_ang_t = calculate_angle_2d(torso_axis, rear_femur_vec)
    f14_torso_femur = np.sum(torso_femur_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # 5. v2 신규 4D 선행 연구 기반 특화 피처 계산
    # f15: 뒤다리 고관절 신전각 (Farrokhi et al. JOSPT 핵심 지표)
    # 체간 축(Torso-Pelvis)과 뒤다리 대퇴골(Pelvis-Knee_rear) 사잇각
    vec_torso_down = pelvis - torso # (N, T, 2)
    hip_ext_t = calculate_angle_2d(vec_torso_down, rear_femur_vec) # (N, T)
    f15_hip_ext = np.sum(hip_ext_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f16: 앞뒤 무릎 더블 90도 동시 적중 조화 잔차 (NSCA 생체역학)
    # sqrt((rear - 90)^2 + 0.5 * (lead - 90)^2)
    f16_double_90_res = np.sqrt((f1_rear_knee_bot - 90.0)**2 + 0.5 * (lead_knee_bot - 90.0)**2)
    
    # f17: 최하사점 전환기 감속도 프로파일 (OpenCap Jerk/Decel)
    # 최하사점 직전 6프레임 동안의 각속도 변화량
    ang_vel = np.diff(rear_knee_ang_t, axis=1) # (N, T-1)
    bot_idx_clamped = np.clip(bottom_idx - 1, 0, T - 2)
    decel_list = []
    for i in range(N):
        b_idx = bot_idx_clamped[i]
        start_idx = max(0, b_idx - 5)
        vel_win = ang_vel[i, start_idx:b_idx+1]
        decel = float(np.mean(np.abs(vel_win))) if len(vel_win) > 0 else 0.0
        decel_list.append(decel)
    f17_knee_decel = np.array(decel_list, dtype=np.float32).reshape(N, 1)
    
    # f18: 신체 비례(골반 너비) 정규화 보폭비
    pelvis_width_t = np.linalg.norm(hip_l - hip_r, axis=-1) # (N, T)
    mean_pelvis_w = np.mean(pelvis_width_t, axis=1, keepdims=True) + 1e-4
    f18_stride_pelvis_ratio = f11_stride / mean_pelvis_w
    
    # 18D 피처 결합
    feats_18d = np.concatenate([
        f1_rear_knee_bot,
        f2_rear_knee_mean,
        f3_rear_knee_rom,
        f4_dev_bot,
        f5_dev_mean,
        f6_femur_pitch,
        f7_shank_pitch,
        f8_ratio,
        f9_lead_knee_bot,
        f10_inter_knee_v,
        f11_stride,
        f12_drop_stride_ratio,
        f13_knee_std,
        f14_torso_femur,
        f15_hip_ext,
        f16_double_90_res,
        f17_knee_decel,
        f18_stride_pelvis_ratio
    ], axis=1).astype(np.float32)
    
    feats_18d = np.nan_to_num(feats_18d, nan=0.0, posinf=180.0, neginf=-180.0)
    
    if not is_batch:
        return feats_18d[0]
    return feats_18d

if __name__ == "__main__":
    dummy = np.zeros((64, 15, 2), dtype=np.float32)
    out = extract_lunge_c3_18d_features_v2(dummy)
    print("V2 Feature extraction shape:", out.shape)
    assert out.shape == (18,), f"Shape mismatch: {out.shape}"
    print("V2 Feature names count:", len(FEATURE_NAMES_18D))
    print("[+] Lunge C3 V2 18D Feature Extractor initialized successfully!")

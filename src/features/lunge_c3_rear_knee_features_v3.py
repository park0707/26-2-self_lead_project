# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c3_rear_knee_features_v3.py
Purpose: 피처 전수 조사 및 다중공선성(|r| > 0.85)·성능 저해 피처를 전수 제거한
         바벨 런지 c3(뒤다리 무릎 각도 90도) 11D 직교 정예 생체역학 피처 추출기 (v3)
         - 제거된 7개 피처:
           1) pelvis_drop_to_stride_ratio (기여도 0.0의 Dead 피처)
           2) rear_hip_extension_angle (torso_to_rear_femur_angle와 r=1.0000 완전 중복)
           3) double_90_bilateral_residual (rear_knee_target_dev_bottom과 r=0.9458 다중공선성 및 음수 기여)
           4) stride_to_pelvis_width_ratio (inter_ankle_stride_bottom과 r=0.9444 다중공선성 및 음수 기여)
           5) rear_knee_target_dev_mean (rear_knee_angle_mean과 r=0.9703 다중공선성)
           6) rear_knee_angle_mean (순열 기여도 음수)
           7) rear_knee_angle_rom (순열 기여도 음수)
         - 선발된 11대 직교 정예 피처:
           f1: rear_knee_stability_std (최하사점 뒤다리 무릎 각도 표준편차 - 관절 떨림/안정성)
           f2: lead_rear_knee_ratio (앞/뒤 무릎 각도 비율)
           f3: rear_femur_vertical_pitch (뒤다리 대퇴골의 지면 수직선 피치 각도)
           f4: rear_shank_horizontal_pitch (뒤다리 정강이의 지면 수평선 피치 각도)
           f5: inter_ankle_stride_bottom (최하사점 양 발목 간 유클리드 보폭)
           f6: knee_velocity_decel_nadir (최하사점 전환기 뒤다리 무릎 각속도 감속도)
           f7: rear_knee_target_dev_bottom (최하사점 90도 기준 뒤다리 무릎 각도 절대 오차)
           f8: inter_knee_vertical_disp (최하사점 양 무릎 간 수직 높이차)
           f9: torso_to_rear_femur_angle (체간 축과 뒤다리 대퇴골 간 사잇각 - 고관절 신전도)
           f10: lead_knee_angle_bottom (최하사점 앞다리 무릎 각도)
           f11: rear_knee_angle_bottom (최하사점 뒤다리 무릎 2D 절대 각도)
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any

FEATURE_NAMES_11D: List[str] = [
    'rear_knee_stability_std',       # 1. [기여도 1위] 뒤다리 무릎 각도 표준편차 (안정성/떨림)
    'lead_rear_knee_ratio',          # 2. [기여도 2위] 앞다리 대비 뒤다리 무릎 각도 비율
    'rear_femur_vertical_pitch',     # 3. [기여도 3위] 뒤다리 대퇴골의 지면 수직선 대비 피치각 (°)
    'rear_shank_horizontal_pitch',   # 4. [기여도 4위] 뒤다리 정강이의 지면 수평선 대비 피치각 (°)
    'inter_ankle_stride_bottom',     # 5. [기여도 5위] 양 발목 간 2D 유클리드 거리 (보폭)
    'knee_velocity_decel_nadir',     # 6. [기여도 6위] 최하사점 직전 뒤다리 무릎 각속도 감속도
    'rear_knee_target_dev_bottom',   # 7. [기여도 7위] 최하사점 90도 기준 뒤다리 무릎 절대 편차 (°)
    'inter_knee_vertical_disp',      # 8. [기여도 8위] 최하사점 앞뒤 무릎 간 수직 높이차
    'torso_to_rear_femur_angle',     # 9. [기여도 9위] 체간 축과 뒤다리 대퇴골 간 2D 사잇각 (고관절 신전도) (°)
    'lead_knee_angle_bottom',        # 10. 최하사점 앞다리 무릎 각도 (°)
    'rear_knee_angle_bottom'         # 11. 최하사점 뒤다리 무릎 2D/3D 절대각 (°)
]

def calculate_angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 2D 벡터 간의 사이각 (0° ~ 180°) 계산"""
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_lunge_c3_11d_features_v3(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 정규화 관절 배열로부터 c3 v3 11D 직교 정예 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (11,) 또는 (N, 11)
    """
    is_batch = (kps.ndim == 4)
    if not is_batch:
        kps = kps[None, ...] # (1, 64, 15, 2)
        
    N, T, J, C = kps.shape
    
    pelvis = kps[:, :, 0, :]   # (N, T, 2)
    torso = kps[:, :, 1, :]    # (N, T, 2)
    hip_l = kps[:, :, 8, :]    # (N, T, 2)
    hip_r = kps[:, :, 9, :]    # (N, T, 2)
    knee_l = kps[:, :, 10, :]  # (N, T, 2)
    knee_r = kps[:, :, 11, :]  # (N, T, 2)
    ank_l = kps[:, :, 12, :]   # (N, T, 2)
    ank_r = kps[:, :, 13, :]   # (N, T, 2)
    
    # 1. 좌우 무릎 각도 시계열 계산 (0° ~ 180°)
    vec_femur_l = knee_l - hip_l
    vec_shank_l = ank_l - knee_l
    ang_l_t = 180.0 - calculate_angle_2d(vec_femur_l, vec_shank_l)
    
    vec_femur_r = knee_r - hip_r
    vec_shank_r = ank_r - knee_r
    ang_r_t = 180.0 - calculate_angle_2d(vec_femur_r, vec_shank_r)
    
    # Savitzky-Golay 필터링 (Window=7, Poly=2)
    if T >= 7:
        ang_l_t_smooth = savgol_filter(ang_l_t, window_length=7, polyorder=2, axis=1)
        ang_r_t_smooth = savgol_filter(ang_r_t, window_length=7, polyorder=2, axis=1)
    else:
        ang_l_t_smooth = ang_l_t
        ang_r_t_smooth = ang_r_t
        
    # 2. 최하사점(Bottom Position) 정밀 탐색
    sum_knee_ang = ang_l_t_smooth + ang_r_t_smooth
    bottom_idx = np.argmin(sum_knee_ang, axis=1)
    
    # 최하사점 윈도우 마스크 (t_bottom ± 6 프레임)
    t_grid = np.arange(T)[None, :]
    bottom_idx_2d = bottom_idx[:, None]
    bottom_mask = (t_grid >= np.maximum(0, bottom_idx_2d - 6)) & (t_grid <= np.minimum(T - 1, bottom_idx_2d + 6))
    win_count = np.sum(bottom_mask, axis=1, keepdims=True) + 1e-7
    
    # 3. 선행 다리 vs 후행/뒤다리 동적 자동 분리
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1]
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1]
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None]
    
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t)
    lead_knee_ang_t = np.where(is_left_rear, ang_r_t, ang_l_t)
    
    rear_femur_vec = np.where(is_left_rear[:, :, None], vec_femur_l, vec_femur_r)
    rear_shank_vec = np.where(is_left_rear[:, :, None], vec_shank_l, vec_shank_r)
    rear_knee_pts = np.where(is_left_rear[:, :, None], knee_l, knee_r)
    lead_knee_pts = np.where(is_left_rear[:, :, None], knee_r, knee_l)
    
    # 4. 11대 직교 정예 생체역학 피처 산출
    # f11. 최하사점 뒤다리 무릎 각도
    f11_rear_knee_bot = np.sum(rear_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f1. 뒤다리 무릎 각도 표준편차 (안정성)
    sq_diff = (rear_knee_ang_t - f11_rear_knee_bot)**2 * bottom_mask
    f1_knee_std = np.sqrt(np.sum(sq_diff, axis=1, keepdims=True) / win_count)
    
    # f10. 최하사점 앞다리 무릎 각도
    f10_lead_knee_bot = np.sum(lead_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f2. 앞다리 대비 뒤다리 무릎 각도 비율
    f2_ratio = f10_lead_knee_bot / (f11_rear_knee_bot + 1e-4)
    
    # f3. 대퇴골 지면 수직선 피치각
    vert_unit = np.array([0.0, 1.0], dtype=np.float32)[None, None, :]
    femur_pitch_t = calculate_angle_2d(rear_femur_vec, vert_unit)
    f3_femur_pitch = np.sum(femur_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f4. 정강이 지면 수평선 피치각
    horiz_unit = np.array([1.0, 0.0], dtype=np.float32)[None, None, :]
    shank_pitch_t = calculate_angle_2d(rear_shank_vec, horiz_unit)
    f4_shank_pitch = np.sum(shank_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f5. 양 발목 간 유클리드 보폭
    ankle_stride_t = np.linalg.norm(ank_l - ank_r, axis=-1)
    f5_stride = np.sum(ankle_stride_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f6. 최하사점 직전 무릎 각속도 감속도
    ang_vel = np.diff(rear_knee_ang_t, axis=1)
    bot_idx_clamped = np.clip(bottom_idx - 1, 0, T - 2)
    decel_list = []
    for i in range(N):
        b_idx = bot_idx_clamped[i]
        start_idx = max(0, b_idx - 5)
        vel_win = ang_vel[i, start_idx:b_idx+1]
        decel = float(np.mean(np.abs(vel_win))) if len(vel_win) > 0 else 0.0
        decel_list.append(decel)
    f6_knee_decel = np.array(decel_list, dtype=np.float32).reshape(N, 1)
    
    # f7. 최하사점 90도 기준 뒤다리 무릎 편차
    f7_dev_bot = np.abs(f11_rear_knee_bot - 90.0)
    
    # f8. 양 무릎 간 수직 높이차
    knee_v_disp_t = np.abs(lead_knee_pts[:, :, 1] - rear_knee_pts[:, :, 1])
    f8_inter_knee_v = np.sum(knee_v_disp_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f9. 체간 축과 뒤다리 대퇴골 간 사잇각 (고관절 신전도)
    torso_axis = torso - pelvis
    torso_femur_ang_t = calculate_angle_2d(torso_axis, rear_femur_vec)
    f9_torso_femur = np.sum(torso_femur_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # 11D 결합 (선정 순서대로)
    feats_11d = np.concatenate([
        f1_knee_std,         # 1. rear_knee_stability_std
        f2_ratio,            # 2. lead_rear_knee_ratio
        f3_femur_pitch,      # 3. rear_femur_vertical_pitch
        f4_shank_pitch,      # 4. rear_shank_horizontal_pitch
        f5_stride,           # 5. inter_ankle_stride_bottom
        f6_knee_decel,       # 6. knee_velocity_decel_nadir
        f7_dev_bot,          # 7. rear_knee_target_dev_bottom
        f8_inter_knee_v,     # 8. inter_knee_vertical_disp
        f9_torso_femur,      # 9. torso_to_rear_femur_angle
        f10_lead_knee_bot,   # 10. lead_knee_angle_bottom
        f11_rear_knee_bot    # 11. rear_knee_angle_bottom
    ], axis=1).astype(np.float32)
    
    feats_11d = np.nan_to_num(feats_11d, nan=0.0, posinf=180.0, neginf=-180.0)
    
    if not is_batch:
        return feats_11d[0]
    return feats_11d

if __name__ == "__main__":
    dummy = np.zeros((64, 15, 2), dtype=np.float32)
    out = extract_lunge_c3_11d_features_v3(dummy)
    print("V3 Feature extraction shape:", out.shape)
    assert out.shape == (11,), f"Shape mismatch: {out.shape}"
    print("V3 Feature names count:", len(FEATURE_NAMES_11D))
    print("[+] Lunge C3 V3 11D Feature Extractor initialized successfully!")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_bl3_features_v4.py
Purpose: 스포츠 생체역학(IEEE TBME, Journal of Biomechanics) 선행 연구 기반
         원천 라벨을 100% 보존하면서 모델의 순수 비전 분류력을 극대화하는
         바벨 런지 세 번째 결함 '뒤다리 무릎 각도 90도 (bl3)' 18D 고급 시계열 기구학 피처 추출기 (v4)
         
         피처 구성:
         - [v3 계승 11D 직교 정예 피처]
           f1: rear_knee_stability_std (최하사점 뒤다리 무릎 각도 표준편차 - 떨림/안정성)
           f2: lead_rear_knee_ratio (앞다리 대비 뒤다리 무릎 각도 비율)
           f3: rear_femur_vertical_pitch (뒤다리 대퇴골의 지면 수직선 피치각)
           f4: rear_shank_horizontal_pitch (뒤다리 정강이의 지면 수평선 피치각)
           f5: inter_ankle_stride_bottom (최하사점 양 발목 간 2D 유클리드 보폭)
           f6: knee_velocity_decel_nadir (최하사점 전환기 뒤다리 무릎 각속도 감속도)
           f7: rear_knee_target_dev_bottom (최하사점 90도 기준 뒤다리 무릎 절대 편차)
           f8: inter_knee_vertical_disp (최하사점 양 무릎 간 수직 높이차)
           f9: torso_to_rear_femur_angle (체간 축과 뒤다리 대퇴골 간 사잇각 - 고관절 신전도)
           f10: lead_knee_angle_bottom (최하사점 앞다리 무릎 각도)
           f11: rear_knee_angle_bottom (최하사점 뒤다리 무릎 2D 원천 각도)
         - [신규 v4 고급 시계열 기구학 & 시점 자가 보정 7D 피처]
           f12: knee_hysteresis_area (무릎 각도-각속도 위상 루프 적분 면적)
           f13: cam_aspect_ratio (골반폭/체간길이 비: 카메라 방위각 추정)
           f14: femur_tibia_proj_ratio (대퇴/정강이 2D 투영 길이비)
           f15: pseudo_3d_knee_angle (원근 단축 역투영 보정 무릎 각도)
           f16: crp_hip_knee_mean (고관절-무릎 연속 상대 위상 CRP)
           f17: xcorr_hip_knee_lag (엉덩이 하강-무릎 굴곡 상호상관 시간차 지연)
           f18: phase_time_ratio (하강 시간 대비 상승 시간 비율)
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any

FEATURE_NAMES_18D: List[str] = [
    # v3 직교 정예 11D
    'rear_knee_stability_std',       # 1. 뒤다리 무릎 각도 표준편차 (떨림/안정성)
    'lead_rear_knee_ratio',          # 2. 앞다리 대비 뒤다리 무릎 각도 비율
    'rear_femur_vertical_pitch',     # 3. 뒤다리 대퇴골의 지면 수직선 피치각 (°)
    'rear_shank_horizontal_pitch',   # 4. 뒤다리 정강이의 지면 수평선 피치각 (°)
    'inter_ankle_stride_bottom',     # 5. 최하사점 양 발목 간 2D 보폭
    'knee_velocity_decel_nadir',     # 6. 최하사점 전환기 무릎 각속도 감속도
    'rear_knee_target_dev_bottom',   # 7. 최하사점 90도 기준 뒤다리 무릎 편차 (°)
    'inter_knee_vertical_disp',      # 8. 최하사점 양 무릎 간 수직 높이차
    'torso_to_rear_femur_angle',     # 9. 체간 축과 뒤다리 대퇴골 간 사잇각 (°)
    'lead_knee_angle_bottom',        # 10. 최하사점 앞다리 무릎 각도 (°)
    'rear_knee_angle_bottom',        # 11. 최하사점 뒤다리 무릎 원천 각도 (°)
    # v4 고급 생체역학 7D
    'knee_hysteresis_area',          # 12. [동역학] 각도-각속도 위상 평면 히스테리시스 루프 면적
    'cam_aspect_ratio',              # 13. [시점보정] 골반폭 대비 체간길이비 (카메라 방위각 추정)
    'femur_tibia_proj_ratio',        # 14. [시점보정] 최하사점 대퇴/정강이 2D 투영 길이비
    'pseudo_3d_knee_angle',          # 15. [시점보정] 원근 단축 역투영 보정 무릎 각도 (°)
    'crp_hip_knee_mean',             # 16. [협응] 고관절-슬관절 연속 상대 위상 (CRP) 평균 (°)
    'xcorr_hip_knee_lag',            # 17. [동역학] 엉덩이 하강-무릎 굴곡 상호상관 시간차 지연 (프레임)
    'phase_time_ratio'               # 18. [동역학] 하강 시간 대비 상승 시간 비율
]

def calculate_angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 2D 벡터 간의 사이각 (0° ~ 180°) 계산"""
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_lunge_bl3_18d_features_v4(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 관절 배열로부터 bl3 v4 18D 고급 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (18,) 또는 (N, 18)
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
    bottom_idx = np.argmin(sum_knee_ang, axis=1) # (N,)
    
    # 최하사점 윈도우 마스크 (t_bottom ± 6 프레임)
    t_grid = np.arange(T)[None, :]
    bottom_idx_2d = bottom_idx[:, None]
    bottom_mask = (t_grid >= np.maximum(0, bottom_idx_2d - 6)) & (t_grid <= np.minimum(T - 1, bottom_idx_2d + 6))
    win_count = np.sum(bottom_mask, axis=1, keepdims=True) + 1e-7
    
    # 3. 선행 다리 vs 후행/뒤다리 동적 자동 분리
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1]
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1]
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None]
    
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t) # (N, T)
    lead_knee_ang_t = np.where(is_left_rear, ang_r_t, ang_l_t) # (N, T)
    
    rear_femur_vec = np.where(is_left_rear[:, :, None], vec_femur_l, vec_femur_r)
    rear_shank_vec = np.where(is_left_rear[:, :, None], vec_shank_l, vec_shank_r)
    rear_knee_pts = np.where(is_left_rear[:, :, None], knee_l, knee_r)
    lead_knee_pts = np.where(is_left_rear[:, :, None], knee_r, knee_l)
    rear_hip_pts = np.where(is_left_rear[:, :, None], hip_l, hip_r)
    
    # 4. v3의 11대 직교 정예 피처 계산
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
    
    # f5. 최하사점 양 발목 간 유클리드 보폭
    ankle_stride_t = np.linalg.norm(ank_l - ank_r, axis=-1)
    f5_stride = np.sum(ankle_stride_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f6. 최하사점 직전 무릎 각속도 감속도
    ang_vel = np.diff(rear_knee_ang_t, axis=1) # (N, T-1)
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
    
    # 5. [신규 v4 고급 생체역학 7대 피처]
    # f12. 무릎 각도-각속도 위상 루프 적분 면적 (Hysteresis Area)
    # Loop Area = sum(|omega_t * (theta_t - theta_mean)|)
    theta_mean = np.mean(rear_knee_ang_t, axis=1, keepdims=True)
    hyst_area = np.sum(np.abs(ang_vel * (rear_knee_ang_t[:, :-1] - theta_mean)), axis=1, keepdims=True) / (T - 1.0)
    f12_hyst = hyst_area
    
    # f13. 골반폭 대비 체간길이비 (카메라 방위각 추정)
    pelvis_w = np.linalg.norm(hip_l - hip_r, axis=-1) # (N, T)
    torso_l = np.linalg.norm(torso - pelvis, axis=-1) + 1e-4 # (N, T)
    aspect_t = pelvis_w / torso_l
    f13_cam_aspect = np.mean(aspect_t, axis=1, keepdims=True)
    
    # f14. 최하사점 2D 대퇴/정강이 투영 길이비
    len_femur_t = np.linalg.norm(rear_femur_vec, axis=-1) # (N, T)
    len_shank_t = np.linalg.norm(rear_shank_vec, axis=-1) + 1e-4 # (N, T)
    ratio_proj_t = len_femur_t / len_shank_t
    f14_proj_ratio = np.sum(ratio_proj_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f15. 원근 단축 역투영 보정 무릎 각도 (Pseudo-3D Angle)
    # Winter's Biomechanical Ref Ratio = 1.12
    k_proj = np.clip(1.12 / (f14_proj_ratio + 1e-4), 0.5, 2.0)
    cos_2d = np.cos(np.radians(f11_rear_knee_bot))
    # 단축 보정 cos 값 계산
    cos_pseudo = np.clip(cos_2d * k_proj, -1.0, 1.0)
    f15_pseudo_3d = np.degrees(np.arccos(cos_pseudo))
    
    # f16. 고관절-슬관절 연속 상대 위상 (CRP) 평균
    # 고관절 각도 시계열: 체간 축과 대퇴골 사잇각
    hip_ang_t = calculate_angle_2d(torso - pelvis, rear_femur_vec) # (N, T)
    hip_vel = np.diff(hip_ang_t, axis=1) # (N, T-1)
    
    # 위상각 산출 phi = arctan2(vel_norm, pos_norm)
    phi_knee = np.arctan2(ang_vel, rear_knee_ang_t[:, :-1] - np.mean(rear_knee_ang_t[:, :-1], axis=1, keepdims=True) + 1e-4)
    phi_hip = np.arctan2(hip_vel, hip_ang_t[:, :-1] - np.mean(hip_ang_t[:, :-1], axis=1, keepdims=True) + 1e-4)
    crp_t = np.degrees(np.abs(phi_hip - phi_knee))
    f16_crp_mean = np.mean(crp_t, axis=1, keepdims=True)
    
    # f17. 엉덩이 하강-무릎 굴곡 상호상관 시간차 지연 (Cross-Correlation Lag)
    pelvis_y = pelvis[:, :, 1] # (N, T)
    xcorr_lags = []
    for i in range(N):
        py = pelvis_y[i] - np.mean(pelvis_y[i])
        ky = rear_knee_ang_t[i] - np.mean(rear_knee_ang_t[i])
        corr = np.correlate(py, ky, mode='full')
        max_idx = int(np.argmax(corr))
        lag = max_idx - (T - 1)
        xcorr_lags.append(float(lag))
    f17_xcorr_lag = np.array(xcorr_lags, dtype=np.float32).reshape(N, 1)
    
    # f18. 하강 시간 대비 상승 시간 비율 (Eccentric to Concentric Ratio)
    # bot_idx가 몇 번째 프레임인가 (0-indexed)
    ecc_time = np.maximum(1.0, bottom_idx[:, None].astype(np.float32))
    con_time = np.maximum(1.0, (T - 1 - bottom_idx)[:, None].astype(np.float32))
    f18_phase_ratio = ecc_time / con_time
    
    # 18D 피처 결합
    feats_18d = np.concatenate([
        f1_knee_std,         # 1
        f2_ratio,            # 2
        f3_femur_pitch,      # 3
        f4_shank_pitch,      # 4
        f5_stride,           # 5
        f6_knee_decel,       # 6
        f7_dev_bot,          # 7
        f8_inter_knee_v,     # 8
        f9_torso_femur,      # 9
        f10_lead_knee_bot,   # 10
        f11_rear_knee_bot,   # 11
        f12_hyst,            # 12
        f13_cam_aspect,      # 13
        f14_proj_ratio,      # 14
        f15_pseudo_3d,       # 15
        f16_crp_mean,        # 16
        f17_xcorr_lag,       # 17
        f18_phase_ratio      # 18
    ], axis=1).astype(np.float32)
    
    feats_18d = np.nan_to_num(feats_18d, nan=0.0, posinf=180.0, neginf=-180.0)
    
    if not is_batch:
        return feats_18d[0]
    return feats_18d

# 14D 정예 기구학 피처셋 (v3 11D + 히스테리시스 면적 + 대퇴/정강이 투영비 + 고관절-슬관절 CRP)
FEATURE_INDICES_14D: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]
FEATURE_NAMES_14D: List[str] = [FEATURE_NAMES_18D[i] for i in FEATURE_INDICES_14D]

def extract_lunge_bl3_14d_features_v4(kps_seq: np.ndarray) -> np.ndarray:
    """
    14D 정예 고급 기구학 피처 추출 (v4 Champion)
    - v3의 11D 직교 정예 기구학 피처 (f1 ~ f11)
    - + knee_hysteresis_area (무릎 각도-각속도 위상 평면 히스테리시스 루프 적분 면적)
    - + femur_tibia_proj_ratio (최하사점 대퇴/정강이 2D 투영 길이비 - 원근 단축 보정)
    - + crp_hip_knee_mean (고관절-무릎 연속 상대 위상 CRP: 하체 협응 대칭성)
    """
    feats_18d = extract_lunge_bl3_18d_features_v4(kps_seq)
    if feats_18d.ndim == 1:
        return feats_18d[FEATURE_INDICES_14D]
    return feats_18d[:, FEATURE_INDICES_14D]

# 13D 정예 기구학 피처셋 (v3 11D + 히스테리시스 루프 면적 + 고관절-슬관절 CRP)
FEATURE_INDICES_13D: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
FEATURE_NAMES_13D: List[str] = [FEATURE_NAMES_18D[i] for i in FEATURE_INDICES_13D]

def extract_lunge_bl3_13d_features_v4(kps_seq: np.ndarray) -> np.ndarray:
    """
    13D 정예 고급 기구학 피처 추출
    """
    feats_18d = extract_lunge_bl3_18d_features_v4(kps_seq)
    if feats_18d.ndim == 1:
        return feats_18d[FEATURE_INDICES_13D]
    return feats_18d[:, FEATURE_INDICES_13D]

if __name__ == "__main__":
    dummy = np.zeros((64, 15, 2), dtype=np.float32)
    out18 = extract_lunge_bl3_18d_features_v4(dummy)
    out14 = extract_lunge_bl3_14d_features_v4(dummy)
    out13 = extract_lunge_bl3_13d_features_v4(dummy)
    print("V4 18D Feature extraction shape:", out18.shape)
    print("V4 14D Feature extraction shape:", out14.shape)
    print("V4 13D Feature extraction shape:", out13.shape)
    assert out18.shape == (18,), f"Shape mismatch: {out18.shape}"
    assert out14.shape == (14,), f"Shape mismatch: {out14.shape}"
    assert out13.shape == (13,), f"Shape mismatch: {out13.shape}"
    print("V4 14D Feature names:", FEATURE_NAMES_14D)
    print("[+] Lunge BL3 V4 14D Elite Kinematics Feature Extractor initialized successfully!")

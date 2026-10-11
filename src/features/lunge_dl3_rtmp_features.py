# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드 & 06_스텝_백워드)
Defect: Condition 3 - 뒤다리 무릎 각도 90도 (dl3)
Module: lunge_dl3_rtmp_features.py
Purpose: 스포츠 생체역학 선행 연구(IEEE TBME, Journal of Biomechanics) 및
         바벨 런지 bl3_v4 / bl3_rtmp_1 정예 기구학 엔진을 100% 계승하여
         다이나믹 런지 RTMPose-m Halpe26(64, 26, 3) 2D 키포인트에 최적화된
         dl3_rtmp_1 전용 고급 생체역학 피처 추출기 (Copy-and-Refine 계승)

피처 체계:
  [Part 1: bl3 베이스라인 14D 정예 기구학 피처 계승 (Indices 0~13)]
    f01: rear_knee_stability_std (최하사점 뒤다리 무릎 각도 표준편차 - 떨림/안정성)
    f02: lead_rear_knee_ratio (앞다리 대비 뒤다리 무릎 각도 비율)
    f03: rear_femur_vertical_pitch (뒤다리 대퇴골의 지면 수직선 피치각)
    f04: rear_shank_horizontal_pitch (뒤다리 정강이의 지면 수평선 피치각)
    f05: inter_ankle_stride_bottom (최하사점 양 발목 간 2D 보폭)
    f06: knee_velocity_decel_nadir (최하사점 전환기 뒤다리 무릎 각속도 감속도)
    f07: rear_knee_target_dev_bottom (최하사점 90도 기준 뒤다리 무릎 편차)
    f08: inter_knee_vertical_disp (최하사점 양 무릎 간 수직 높이차)
    f09: torso_to_rear_femur_angle (체간 축과 뒤다리 대퇴골 간 사잇각)
    f10: lead_knee_angle_bottom (최하사점 앞다리 무릎 각도)
    f11: rear_knee_angle_bottom (최하사점 뒤다리 무릎 원천 각도)
    f12: knee_hysteresis_area (무릎 각도-각속도 위상 루프 적분 면적)
    f13: femur_tibia_proj_ratio (최하사점 대퇴/정강이 2D 투영 길이비)
    f14: crp_hip_knee_mean (고관절-무릎 연속 상대 위상 CRP 평균)

  [Part 2: 다이나믹 런지 및 RTMPose 고유 동역학 피처 (Indices 14~21)]
    f15: rear_foot_heel_lift (최하사점 뒤꿈치 수직 들림 높이: Heel vs Toe Y차이)
    f16: rear_foot_pitch_bottom (최하사점 뒤발 경사각: 발바닥 접촉/배굴 각도)
    f17: lead_foot_flatness (최하사점 앞발 지면 안착 평평도)
    f18: rear_knee_min_angle (전 시퀀스 뒤다리 무릎 최소 굴곡 각도)
    f19: rear_knee_rom (뒤다리 무릎 관절 가동범위: 기립 - 최소각)
    f20: step_stride_expansion_ratio (기립 보폭 대비 최하사점 스텝 확장비)
    f21: torso_vertical_tilt_bottom (최하사점 체간 수직선 기울기 각도)
    f22: cam_aspect_ratio (골반폭 대비 체간길이비: 카메라 방위각 추정)

  [Part 3: AGENTS.md 공식 5대 카메라 시점 One-Hot 인코딩 (Indices 22~26)]
    f23: view_is_view5 (시점 A: 우측 후방 45°)
    f24: view_is_view2 (시점 B: 우측 전방 45°)
    f25: view_is_view3 (시점 C: 정면 0°)
    f26: view_is_view4 (시점 D: 좌측 전방 45°)
    f27: view_is_view1 (시점 E: 좌측 후방 45°)
"""

import sys
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any, Union

# 14D 베이스라인 피처명 (bl3 계승)
FEATURE_NAMES_14D: List[str] = [
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
    'knee_hysteresis_area',
    'femur_tibia_proj_ratio',
    'crp_hip_knee_mean'
]

# 22D 기구학 피처명 (14D + 8D RTMPose/다이나믹 피처)
FEATURE_NAMES_22D: List[str] = FEATURE_NAMES_14D + [
    'rear_foot_heel_lift',
    'rear_foot_pitch_bottom',
    'lead_foot_flatness',
    'rear_knee_min_angle',
    'rear_knee_rom',
    'step_stride_expansion_ratio',
    'torso_vertical_tilt_bottom',
    'cam_aspect_ratio'
]

# 27D 전체 피처명 (22D + 5D View One-Hot)
FEATURE_NAMES_27D: List[str] = FEATURE_NAMES_22D + [
    'view_is_view5',
    'view_is_view2',
    'view_is_view3',
    'view_is_view4',
    'view_is_view1'
]

VIEW_CODE_LIST = ['view5', 'view2', 'view3', 'view4', 'view1']

def calculate_angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 2D 벡터 간의 사이각 (0° ~ 180°) 계산"""
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_lunge_dl3_rtmp_features_all(
    kps_26: np.ndarray,
    view_code: Optional[str] = None
) -> np.ndarray:
    """
    단일 또는 배치 RTMPose Halpe26 키포인트 (64, 26, 3) 또는 (64, 26, 2)로부터
    dl3_rtmp_1 전용 27D 피처 벡터 추출.
    
    Parameters:
      kps_26: (64, 26, 3) 또는 (64, 26, 2)
      view_code: 'view1' ~ 'view5' (선택적)
      
    Returns:
      feats_27d: (27,) float32 배열
    """
    is_batch = (kps_26.ndim == 4)
    if not is_batch:
        kps_26 = kps_26[None, ...] # (1, 64, 26, C)
        
    N, T, J, C = kps_26.shape
    kps_xy = kps_26[:, :, :, :2] # (N, T, 26, 2)
    
    # Halpe26 관절 인덱스 매핑
    sh_l = kps_xy[:, :, 5, :]     # Left Shoulder
    sh_r = kps_xy[:, :, 6, :]     # Right Shoulder
    hip_l = kps_xy[:, :, 11, :]   # Left Hip
    hip_r = kps_xy[:, :, 12, :]   # Right Hip
    knee_l = kps_xy[:, :, 13, :]  # Left Knee
    knee_r = kps_xy[:, :, 14, :]  # Right Knee
    ank_l = kps_xy[:, :, 15, :]   # Left Ankle
    ank_r = kps_xy[:, :, 16, :]   # Right Ankle
    toe_l = kps_xy[:, :, 20, :]   # Left Big Toe
    toe_r = kps_xy[:, :, 21, :]   # Right Big Toe
    heel_l = kps_xy[:, :, 24, :]  # Left Heel
    heel_r = kps_xy[:, :, 25, :]  # Right Heel
    
    pelvis = (hip_l + hip_r) / 2.0
    torso = (sh_l + sh_r) / 2.0
    
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
    
    # 3. 선행 다리 vs 후행/뒤다리 동적 자동 분리 (최하사점 무릎 Y좌표 기준: 영상 좌표계에서 Y가 클수록 아래)
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1]
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1]
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None]
    
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t) # (N, T)
    lead_knee_ang_t = np.where(is_left_rear, ang_r_t, ang_l_t) # (N, T)
    rear_knee_ang_t_smooth = np.where(is_left_rear, ang_l_t_smooth, ang_r_t_smooth) # (N, T)
    
    rear_femur_vec = np.where(is_left_rear[:, :, None], vec_femur_l, vec_femur_r)
    rear_shank_vec = np.where(is_left_rear[:, :, None], vec_shank_l, vec_shank_r)
    rear_knee_pts = np.where(is_left_rear[:, :, None], knee_l, knee_r)
    lead_knee_pts = np.where(is_left_rear[:, :, None], knee_r, knee_l)
    rear_hip_pts = np.where(is_left_rear[:, :, None], hip_l, hip_r)
    
    rear_toe_pts = np.where(is_left_rear[:, :, None], toe_l, toe_r)
    rear_heel_pts = np.where(is_left_rear[:, :, None], heel_l, heel_r)
    lead_toe_pts = np.where(is_left_rear[:, :, None], toe_r, toe_l)
    lead_heel_pts = np.where(is_left_rear[:, :, None], heel_r, heel_l)
    
    # =========================================================================
    # Part 1: bl3 베이스라인 14D 피처 (Copy-and-Refine 100% 완전 계승)
    # =========================================================================
    
    # f11. 최하사점 뒤다리 무릎 각도
    f11_rear_knee_bot = np.sum(rear_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f01. 뒤다리 무릎 각도 표준편차 (안정성)
    sq_diff = (rear_knee_ang_t - f11_rear_knee_bot)**2 * bottom_mask
    f01_knee_std = np.sqrt(np.sum(sq_diff, axis=1, keepdims=True) / win_count)
    
    # f10. 최하사점 앞다리 무릎 각도
    f10_lead_knee_bot = np.sum(lead_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f02. 앞다리 대비 뒤다리 무릎 각도 비율
    f02_ratio = f10_lead_knee_bot / (f11_rear_knee_bot + 1e-4)
    
    # f03. 대퇴골 지면 수직선 피치각
    vert_unit = np.array([0.0, 1.0], dtype=np.float32)[None, None, :]
    femur_pitch_t = calculate_angle_2d(rear_femur_vec, vert_unit)
    f03_femur_pitch = np.sum(femur_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f04. 정강이 지면 수평선 피치각
    horiz_unit = np.array([1.0, 0.0], dtype=np.float32)[None, None, :]
    shank_pitch_t = calculate_angle_2d(rear_shank_vec, horiz_unit)
    f04_shank_pitch = np.sum(shank_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f05. 최하사점 양 발목 간 2D 보폭
    ankle_stride_t = np.linalg.norm(ank_l - ank_r, axis=-1)
    f05_stride = np.sum(ankle_stride_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f06. 최하사점 직전 무릎 각속도 감속도
    ang_vel = np.diff(rear_knee_ang_t, axis=1) # (N, T-1)
    bot_idx_clamped = np.clip(bottom_idx - 1, 0, T - 2)
    decel_list = []
    for i in range(N):
        b_idx = bot_idx_clamped[i]
        start_idx = max(0, b_idx - 5)
        vel_win = ang_vel[i, start_idx:b_idx+1]
        decel = float(np.mean(np.abs(vel_win))) if len(vel_win) > 0 else 0.0
        decel_list.append(decel)
    f06_knee_decel = np.array(decel_list, dtype=np.float32).reshape(N, 1)
    
    # f07. 최하사점 90도 기준 뒤다리 무릎 편차
    f07_dev_bot = np.abs(f11_rear_knee_bot - 90.0)
    
    # f08. 양 무릎 간 수직 높이차
    knee_v_disp_t = np.abs(lead_knee_pts[:, :, 1] - rear_knee_pts[:, :, 1])
    f08_inter_knee_v = np.sum(knee_v_disp_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f09. 체간 축과 뒤다리 대퇴골 간 사잇각 (고관절 신전도)
    torso_axis = torso - pelvis
    torso_femur_ang_t = calculate_angle_2d(torso_axis, rear_femur_vec)
    f09_torso_femur = np.sum(torso_femur_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f12. 무릎 각도-각속도 위상 루프 적분 면적 (Hysteresis Area)
    theta_mean = np.mean(rear_knee_ang_t, axis=1, keepdims=True)
    hyst_area = np.sum(np.abs(ang_vel * (rear_knee_ang_t[:, :-1] - theta_mean)), axis=1, keepdims=True) / (T - 1.0)
    f12_hyst = hyst_area
    
    # f13. 최하사점 2D 대퇴/정강이 투영 길이비
    len_femur_t = np.linalg.norm(rear_femur_vec, axis=-1)
    len_shank_t = np.linalg.norm(rear_shank_vec, axis=-1) + 1e-4
    ratio_proj_t = len_femur_t / len_shank_t
    f13_proj_ratio = np.sum(ratio_proj_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f14. 고관절-슬관절 연속 상대 위상 (CRP) 평균
    hip_ang_vel = np.diff(torso_femur_ang_t, axis=1) # (N, T-1)
    knee_ang_vel = ang_vel                            # (N, T-1)
    
    hip_pos = torso_femur_ang_t[:, :-1] - np.mean(torso_femur_ang_t, axis=1, keepdims=True)
    knee_pos = rear_knee_ang_t[:, :-1] - theta_mean
    
    phi_hip = np.arctan2(hip_ang_vel, hip_pos + 1e-5)
    phi_knee = np.arctan2(knee_ang_vel, knee_pos + 1e-5)
    
    crp_t = np.degrees(np.abs(phi_hip - phi_knee))
    crp_t = np.where(crp_t > 180.0, 360.0 - crp_t, crp_t)
    f14_crp_mean = np.mean(crp_t, axis=1, keepdims=True)
    
    # =========================================================================
    # Part 2: 다이나믹 런지 및 RTMPose 26Kps 고유 동역학 피처 (8D)
    # =========================================================================
    
    # f15. 뒤다리 뒤꿈치 수직 들림 높이 (Toe vs Heel Y좌표 차이: 영상 좌표계에서 toe.y - heel.y > 0 이면 들림)
    rear_heel_lift_t = rear_toe_pts[:, :, 1] - rear_heel_pts[:, :, 1]
    f15_rear_heel_lift = np.sum(rear_heel_lift_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f16. 최하사점 뒤발 경사각 (Heel-to-Toe 벡터의 지면 수평선 사잇각)
    vec_rear_foot = rear_toe_pts - rear_heel_pts
    rear_foot_pitch_t = calculate_angle_2d(vec_rear_foot, horiz_unit)
    f16_rear_foot_pitch = np.sum(rear_foot_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f17. 최하사점 앞발 지면 안착 평평도 (|lead_toe.y - lead_heel.y|)
    lead_foot_flat_t = np.abs(lead_toe_pts[:, :, 1] - lead_heel_pts[:, :, 1])
    f17_lead_foot_flat = np.sum(lead_foot_flat_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f18. 전 시퀀스 뒤다리 무릎 최소 각도 (최대 굴곡 깊이)
    f18_rear_knee_min = np.min(rear_knee_ang_t_smooth, axis=1, keepdims=True)
    
    # f19. 뒤다리 무릎 관절 가동범위 (초기 기립 5프레임 평균 - 최소 각도)
    init_knee_ang = np.mean(rear_knee_ang_t[:, :5], axis=1, keepdims=True)
    f19_rear_knee_rom = init_knee_ang - f18_rear_knee_min
    
    # f20. 기립 보폭 대비 최하사점 스텝 확장비
    init_stride = np.mean(ankle_stride_t[:, :5], axis=1, keepdims=True) + 1e-4
    f20_stride_ratio = f05_stride / init_stride
    
    # f21. 최하사점 체간 수직선 기울기 각도 (상체 수직도)
    torso_tilt_t = calculate_angle_2d(torso_axis, vert_unit)
    f21_torso_tilt = np.sum(torso_tilt_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f22. 골반폭 대비 체간길이비 (카메라 방위각/시점 추정비)
    pelvis_w = np.linalg.norm(hip_l - hip_r, axis=-1)
    torso_l = np.linalg.norm(torso - pelvis, axis=-1) + 1e-4
    f22_cam_aspect = np.mean(pelvis_w / torso_l, axis=1, keepdims=True)
    
    # =========================================================================
    # Part 3: AGENTS.md 공식 5대 카메라 시점 One-Hot 인코딩 (5D)
    # =========================================================================
    view_one_hot = np.zeros((N, 5), dtype=np.float32)
    if view_code is not None:
        vc = view_code.lower()
        for idx, code in enumerate(VIEW_CODE_LIST):
            if code in vc:
                view_one_hot[:, idx] = 1.0
                break
                
    # 결합: (N, 27)
    feats = np.concatenate([
        # Part 1: bl3 베이스라인 14D
        f01_knee_std,        # 0
        f02_ratio,           # 1
        f03_femur_pitch,     # 2
        f04_shank_pitch,     # 3
        f05_stride,          # 4
        f06_knee_decel,      # 5
        f07_dev_bot,         # 6
        f08_inter_knee_v,    # 7
        f09_torso_femur,     # 8
        f10_lead_knee_bot,   # 9
        f11_rear_knee_bot,   # 10
        f12_hyst,            # 11
        f13_proj_ratio,      # 12
        f14_crp_mean,        # 13
        # Part 2: RTMPose & Dynamic Lunge 8D
        f15_rear_heel_lift,  # 14
        f16_rear_foot_pitch, # 15
        f17_lead_foot_flat,  # 16
        f18_rear_knee_min,   # 17
        f19_rear_knee_rom,   # 18
        f20_stride_ratio,    # 19
        f21_torso_tilt,      # 20
        f22_cam_aspect,      # 21
        # Part 3: View 5D
        view_one_hot         # 22~26
    ], axis=1).astype(np.float32)
    
    feats = np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)
    
    if not is_batch:
        return feats[0]
    return feats

def extract_lunge_dl3_14d_baseline(kps_26: np.ndarray) -> np.ndarray:
    """bl3 베이스라인 14D 피처만 추출"""
    feats_all = extract_lunge_dl3_rtmp_features_all(kps_26)
    if feats_all.ndim == 1:
        return feats_all[:14]
    return feats_all[:, :14]

def extract_lunge_dl3_22d_kinematics(kps_26: np.ndarray) -> np.ndarray:
    """bl3 14D + 8D RTMPose/다이나믹 기구학 피처 (22D) 추출"""
    feats_all = extract_lunge_dl3_rtmp_features_all(kps_26)
    if feats_all.ndim == 1:
        return feats_all[:22]
    return feats_all[:, :22]

if __name__ == '__main__':
    dummy_kps = np.zeros((64, 26, 3), dtype=np.float32)
    f27 = extract_lunge_dl3_rtmp_features_all(dummy_kps, view_code='view3')
    f14 = extract_lunge_dl3_14d_baseline(dummy_kps)
    f22 = extract_lunge_dl3_22d_kinematics(dummy_kps)
    print(f"[+] 27D Feature extraction test: shape={f27.shape}")
    print(f"[+] 14D Feature extraction test: shape={f14.shape}")
    print(f"[+] 22D Feature extraction test: shape={f22.shape}")
    assert f27.shape == (27,), f"Expected (27,), got {f27.shape}"
    assert f14.shape == (14,), f"Expected (14,), got {f14.shape}"
    assert f22.shape == (22,), f"Expected (22,), got {f22.shape}"
    print("[+] All shapes and assertions passed successfully!")

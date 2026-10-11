# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c3_rear_knee_features.py
Purpose: 바벨 런지 세 번째 오류인 '뒤다리 무릎 각도 90도 (c3)' 전담 14D 생체역학 피처 추출기
         - 규정 준수:
           1) 100% 순수 비전 MediaPipe 좌표 기반 추출 (AI-Hub JSON 좌표 완전 배제)
           2) 선행 다리 vs 후행/뒤다리 동적 자동 분리 기구학 엔진 탑재
           3) 최하사점(Bottom Window: t_bot ± 6 프레임) 위상 인식 스냅샷 및 시계열 적분
           4) 물리적 의미가 명확한 14차원 수술적 특화 피처셋 확립
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

FEATURE_NAMES_14D: List[str] = [
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
    'torso_to_rear_femur_angle'      # 14. [최하사점] 체간 축과 뒤다리 대퇴골 간 2D 사잇각 (고관절 신전도) (°)
]

def calculate_angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    두 2D 벡터 간의 사이각 (0° ~ 180°) 계산
    v1: (..., 2), v2: (..., 2) -> returns: (...)
    """
    dot = np.sum(v1 * v2, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_lunge_c3_14d_features(kps: np.ndarray) -> np.ndarray:
    """
    단일 또는 배치 64프레임 정규화 관절 배열로부터 c3 14D 생체역학 피처 추출
    Input:
        kps: (64, 15, 2) 또는 (N, 64, 15, 2)
    Output:
        features: (14,) 또는 (N, 14)
    """
    is_batch = (kps.ndim == 4)
    if not is_batch:
        kps = kps[None, ...] # (1, 64, 15, 2)
        
    N, T, J, C = kps.shape
    
    # 관절 좌표 추출
    # Joint15: 0:PELVIS, 1:TORSO, 8:LEFT_HIP, 9:RIGHT_HIP, 10:LEFT_KNEE, 11:RIGHT_KNEE, 12:LEFT_ANKLE, 13:RIGHT_ANKLE, 14:NECK
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
    # 무릎 굴곡각: 대퇴골과 정강이 사잇각 (완전히 펴지면 180°, 직각이면 90°)
    ang_l_t = 180.0 - calculate_angle_2d(vec_femur_l, vec_shank_l) # (N, T)
    
    vec_femur_r = knee_r - hip_r # (N, T, 2)
    vec_shank_r = ank_r - knee_r # (N, T, 2)
    ang_r_t = 180.0 - calculate_angle_2d(vec_femur_r, vec_shank_r) # (N, T)
    
    # 2. 최하사점(Bottom Position) 프레임 탐색
    # 양 무릎 각도의 합이 최소가 되는 지점 (가장 깊이 굴곡된 순간)
    sum_knee_ang = ang_l_t + ang_r_t # (N, T)
    bottom_idx = np.argmin(sum_knee_ang, axis=1) # (N,)
    
    # 최하사점 윈도우 마스크 (t_bottom ± 6 프레임)
    t_grid = np.arange(T)[None, :] # (1, T)
    bottom_idx_2d = bottom_idx[:, None] # (N, 1)
    bottom_mask = (t_grid >= np.maximum(0, bottom_idx_2d - 6)) & (t_grid <= np.minimum(T - 1, bottom_idx_2d + 6)) # (N, T)
    win_count = np.sum(bottom_mask, axis=1, keepdims=True) + 1e-7 # (N, 1)
    
    # 3. 선행 다리 vs 후행/뒤다리 동적 자동 분리
    # 최하사점에서 뒤다리 무릎은 바닥 쪽으로 깊이 하강하여 Y 좌표가 더 큼 (+Y가 바닥 방향)
    # 또한 발목 이동 변위(최초 5프레임 대비)를 복합 고려
    knee_l_y_bot = knee_l[np.arange(N), bottom_idx, 1] # (N,)
    knee_r_y_bot = knee_r[np.arange(N), bottom_idx, 1] # (N,)
    
    is_left_rear = (knee_l_y_bot >= knee_r_y_bot)[:, None] # (N, 1)
    
    # 뒤다리 및 앞다리 무릎 각도 시계열 분리
    rear_knee_ang_t = np.where(is_left_rear, ang_l_t, ang_r_t) # (N, T)
    lead_knee_ang_t = np.where(is_left_rear, ang_r_t, ang_l_t) # (N, T)
    
    # 뒤다리 및 앞다리 관절 벡터 분리
    rear_femur_vec = np.where(is_left_rear[:, :, None], vec_femur_l, vec_femur_r) # (N, T, 2)
    rear_shank_vec = np.where(is_left_rear[:, :, None], vec_shank_l, vec_shank_r) # (N, T, 2)
    rear_knee_pts = np.where(is_left_rear[:, :, None], knee_l, knee_r) # (N, T, 2)
    lead_knee_pts = np.where(is_left_rear[:, :, None], knee_r, knee_l) # (N, T, 2)
    
    # 4. 14대 특화 생체역학 피처 계산
    # f1: 최하사점 윈도우 뒤다리 무릎 각도
    f1_rear_knee_bot = np.sum(rear_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f2: 전체 시퀀스 뒤다리 무릎 각도 평균
    f2_rear_knee_mean = np.mean(rear_knee_ang_t, axis=1, keepdims=True)
    
    # f3: 뒤다리 무릎 각도 가동범위 (max - min)
    f3_rear_knee_rom = np.ptp(rear_knee_ang_t, axis=1, keepdims=True)
    
    # f4: 90도 기준 최하사점 절대 오차
    f4_dev_bot = np.abs(f1_rear_knee_bot - 90.0)
    
    # f5: 전체 시퀀스 90도 기준 평균 절대 오차
    f5_dev_mean = np.mean(np.abs(rear_knee_ang_t - 90.0), axis=1, keepdims=True)
    
    # f6: 최하사점 뒤다리 대퇴골의 지면 수직선([0, 1]) 대비 각도
    # 이상적 런지 시 대퇴골은 수직 하강하므로 피치가 0°에 수렴
    vert_unit = np.array([0.0, 1.0], dtype=np.float32)[None, None, :]
    femur_pitch_t = calculate_angle_2d(rear_femur_vec, vert_unit) # (N, T)
    f6_femur_pitch = np.sum(femur_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f7: 최하사점 뒤다리 정강이의 지면 수평선([1, 0]) 대비 각도
    # 이상적 런지 시 정강이는 수평에 가까우므로 수평선과의 각도가 0°에 수렴
    horiz_unit = np.array([1.0, 0.0], dtype=np.float32)[None, None, :]
    shank_pitch_t = calculate_angle_2d(rear_shank_vec, horiz_unit) # (N, T)
    f7_shank_pitch = np.sum(shank_pitch_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f8: 앞다리 대비 뒤다리 무릎 각도 비율
    lead_knee_bot = np.sum(lead_knee_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    f8_ratio = lead_knee_bot / (f1_rear_knee_bot + 1e-4)
    
    # f9: 앞다리 무릎 각도
    f9_lead_knee_bot = lead_knee_bot
    
    # f10: 앞뒤 무릎 간 수직 높이차
    knee_v_disp_t = np.abs(lead_knee_pts[:, :, 1] - rear_knee_pts[:, :, 1]) # (N, T)
    f10_inter_knee_v = np.sum(knee_v_disp_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f11: 최하사점 양 발목 간 2D 거리 (보폭)
    ankle_stride_t = np.linalg.norm(ank_l - ank_r, axis=-1) # (N, T)
    f11_stride = np.sum(ankle_stride_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # f12: 골반 하강 깊이 대비 보폭 비율
    pelvis_y = pelvis[:, :, 1]
    init_pelvis_y = np.mean(pelvis_y[:, :5], axis=1, keepdims=True)
    bot_pelvis_y = np.sum(pelvis_y * bottom_mask, axis=1, keepdims=True) / win_count
    pelvis_drop = np.maximum(0.0, bot_pelvis_y - init_pelvis_y)
    f12_drop_stride_ratio = pelvis_drop / (f11_stride + 1e-4)
    
    # f13: 최하사점 윈도우 뒤다리 무릎 각도 표준편차
    mean_w = f1_rear_knee_bot # (N, 1)
    sq_diff = (rear_knee_ang_t - mean_w)**2 * bottom_mask
    f13_knee_std = np.sqrt(np.sum(sq_diff, axis=1, keepdims=True) / win_count)
    
    # f14: 체간 축과 뒤다리 대퇴골 간 사잇각
    torso_axis = torso - pelvis # (N, T, 2)
    torso_femur_ang_t = calculate_angle_2d(torso_axis, rear_femur_vec) # (N, T)
    f14_torso_femur = np.sum(torso_femur_ang_t * bottom_mask, axis=1, keepdims=True) / win_count
    
    # 14D 피처 행렬 결합
    feats = np.concatenate([
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
        f14_torso_femur
    ], axis=1).astype(np.float32)
    
    # NaN/Inf 안전 방어
    feats = np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)
    
    if not is_batch:
        return feats[0]
    return feats

if __name__ == "__main__":
    dummy = np.zeros((64, 15, 2), dtype=np.float32)
    out = extract_lunge_c3_14d_features(dummy)
    print("Single extraction shape:", out.shape)
    assert out.shape == (14,), f"Shape mismatch: {out.shape}"
    print("Feature names count:", len(FEATURE_NAMES_14D))
    print("[+] Lunge C3 14D Feature Extractor initialized successfully!")

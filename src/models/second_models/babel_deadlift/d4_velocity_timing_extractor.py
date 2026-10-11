# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (굿모닝 / 스트리퍼 데드리프트)
Module: d4_velocity_timing_extractor.py
Description:
  생체역학적 각속도 타이밍(Angular Velocity Timing Dynamics) 6대 코어 피처 추출 모듈
  - 1. d4_vel_peak_time_lag: 피크 각속도 발생 시점차 (Time-to-Peak Velocity Lag)
  - 2. d4_vel_xcorr_lag_norm: 무릎-골반 상호상관 최적 시간 지연 (Cross-Correlation Optimal Lag)
  - 3. d4_first_pull_inst_vr_max: First Pull 순간 무릎/골반 각속도비 피크치 (Instantaneous Peak Velocity Ratio)
  - 4. d4_phase_plane_mismatch: 위상평면(Phase-Plane) 동시신전 대각선 궤적 편차 (Phase Mismatch)
  - 5. d4_first_pull_impulse_ratio: First Pull 신전 충격량 분율 (Velocity Impulse Ratio)
  - 6. d4_knee_plateau_decouple_ratio: 무릎 신전 조기 정체-골반 분리 지속 시간비 (Plateau Decoupling Ratio)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple

# MediaPipe 33 랜드마크 인덱스
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

FEATURE_NAMES_TIMING_6D = [
    'd4_vel_peak_time_lag',             # [0] 피크 각속도 발생 시점차
    'd4_vel_xcorr_lag_norm',            # [1] 상호상관 최적 시간 지연
    'd4_first_pull_inst_vr_max',        # [2] First Pull 순간 각속도비 피크치
    'd4_phase_plane_mismatch',          # [3] 위상평면 동시신전 궤적 불일치도
    'd4_first_pull_impulse_ratio',      # [4] First Pull 신전 충격량 분율
    'd4_knee_plateau_decouple_ratio'    # [5] 무릎 신전 조기 정체 분리 시간비
]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """3점 좌표로 3D 관절 각도 계산 (도 단위)"""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    denom = norm1 * norm2 + 1e-7
    cos_ang = np.sum(v1 * v2, axis=-1) / denom
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def extract_d4_velocity_timing_features_from_seq(kps: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 키포인트 시퀀스 (64, 33, 3)로부터 6D 각속도 타이밍 피처 추출
    
    Returns:
        np.ndarray shape (6,), dtype float32
    """
    T = kps.shape[0]  # 64
    
    # 1. 좌우 관절 각도 산출 및 양측 평균
    knee_ang_l = compute_angle_3d(kps[:, MP_LEFT_HIP], kps[:, MP_LEFT_KNEE], kps[:, MP_LEFT_ANKLE])
    knee_ang_r = compute_angle_3d(kps[:, MP_RIGHT_HIP], kps[:, MP_RIGHT_KNEE], kps[:, MP_RIGHT_ANKLE])
    knee_ang_raw = (knee_ang_l + knee_ang_r) / 2.0

    hip_ang_l = compute_angle_3d(kps[:, MP_LEFT_SHOULDER], kps[:, MP_LEFT_HIP], kps[:, MP_LEFT_KNEE])
    hip_ang_r = compute_angle_3d(kps[:, MP_RIGHT_SHOULDER], kps[:, MP_RIGHT_HIP], kps[:, MP_RIGHT_KNEE])
    hip_ang_raw = (hip_ang_l + hip_ang_r) / 2.0

    # 2. 최하사점(nadir) 및 락아웃(lockout) 탐색 (가우시안 보조)
    knee_ref_smooth = gaussian_filter1d(knee_ang_raw, sigma=1.5)
    hip_ref_smooth = gaussian_filter1d(hip_ang_raw, sigma=1.5)

    nadir_idx = int(np.argmin(knee_ref_smooth))
    lock_knee_idx = int(np.argmax(knee_ref_smooth[nadir_idx:])) + nadir_idx
    lock_hip_idx = int(np.argmax(hip_ref_smooth[nadir_idx:])) + nadir_idx
    end_ascent_idx = max(lock_knee_idx, lock_hip_idx)

    if end_ascent_idx <= nadir_idx + 4:
        end_ascent_idx = min(nadir_idx + 25, T - 1)
        if end_ascent_idx <= nadir_idx + 2:
            end_ascent_idx = T - 1
            nadir_idx = max(0, end_ascent_idx - 20)

    ascent_len = max(end_ascent_idx - nadir_idx, 3)
    mid_ascent_idx = nadir_idx + ascent_len // 2

    # 3. Savitzky-Golay 고차 미분 보존 스무딩 (Window=7, Poly=2)
    win = 7 if T >= 9 else 5
    knee_sg = savgol_filter(knee_ang_raw, window_length=win, polyorder=2)
    hip_sg = savgol_filter(hip_ang_raw, window_length=win, polyorder=2)

    # 4. 각속도 산출 (deg / frame)
    knee_vel = np.gradient(knee_sg)
    hip_vel = np.gradient(hip_sg)

    ascent_slice = slice(nadir_idx, end_ascent_idx)
    fp_slice = slice(nadir_idx, mid_ascent_idx)

    k_vel_asc = knee_vel[ascent_slice]
    h_vel_asc = hip_vel[ascent_slice]
    k_vel_fp = knee_vel[fp_slice]
    h_vel_fp = hip_vel[fp_slice]

    eps = 1e-5

    # -------------------------------------------------------------
    # [Feature 1] 피크 각속도 발생 시점차 (Time-to-Peak Velocity Lag)
    # -------------------------------------------------------------
    k_peak_t = np.argmax(k_vel_asc) / float(ascent_len)
    h_peak_t = np.argmax(h_vel_asc) / float(ascent_len)
    d4_vel_peak_time_lag = float(k_peak_t - h_peak_t)

    # -------------------------------------------------------------
    # [Feature 2] 상호상관 최적 시간 지연 (Cross-Correlation Optimal Lag)
    # -------------------------------------------------------------
    k_std = np.std(k_vel_asc) + eps
    h_std = np.std(h_vel_asc) + eps
    k_norm = (k_vel_asc - np.mean(k_vel_asc)) / k_std
    h_norm = (h_vel_asc - np.mean(h_vel_asc)) / h_std
    xcorr = np.correlate(k_norm, h_norm, mode='full')
    lags = np.arange(-len(k_norm) + 1, len(k_norm))
    best_lag = lags[np.argmax(xcorr)]
    d4_vel_xcorr_lag_norm = float(best_lag / float(ascent_len))

    # -------------------------------------------------------------
    # [Feature 3] First Pull 순간 각속도비 피크치 (Instantaneous Peak VR)
    # -------------------------------------------------------------
    # 골반 각속도가 양수인 구간 중심, 극단치 clipping
    h_vel_fp_pos = np.maximum(h_vel_fp, 0.1)
    k_vel_fp_pos = np.maximum(k_vel_fp, 0.0)
    inst_vr = k_vel_fp_pos / h_vel_fp_pos
    d4_first_pull_inst_vr_max = float(np.percentile(inst_vr, 90))
    d4_first_pull_inst_vr_max = float(np.clip(d4_first_pull_inst_vr_max, 0.0, 10.0))

    # -------------------------------------------------------------
    # [Feature 4] 위상평면 동시신전 대각선 궤적 불일치도 (Phase Mismatch)
    # -------------------------------------------------------------
    traj_angles = np.arctan2(k_vel_fp_pos, h_vel_fp_pos)
    d4_phase_plane_mismatch = float(np.mean(np.abs(traj_angles - (np.pi / 4.0))))

    # -------------------------------------------------------------
    # [Feature 5] First Pull 신전 충격량 분율 (Velocity Impulse Ratio)
    # -------------------------------------------------------------
    k_impulse = float(np.sum(k_vel_fp_pos))
    h_impulse = float(np.sum(h_vel_fp_pos))
    d4_first_pull_impulse_ratio = float(k_impulse / (h_impulse + eps))
    d4_first_pull_impulse_ratio = float(np.clip(d4_first_pull_impulse_ratio, 0.0, 10.0))

    # -------------------------------------------------------------
    # [Feature 6] 무릎 신전 조기 정체 분리 시간비 (Plateau Decoupling Ratio)
    # -------------------------------------------------------------
    k_peak_val = max(float(np.max(k_vel_asc)), 0.5)
    h_peak_val = max(float(np.max(h_vel_asc)), 0.5)
    k_idle = k_vel_asc < (0.20 * k_peak_val)
    h_active = h_vel_asc > (0.35 * h_peak_val)
    decoupled_cnt = int(np.sum(k_idle & h_active))
    d4_knee_plateau_decouple_ratio = float(decoupled_cnt / float(ascent_len))

    feats = np.array([
        d4_vel_peak_time_lag,
        d4_vel_xcorr_lag_norm,
        d4_first_pull_inst_vr_max,
        d4_phase_plane_mismatch,
        d4_first_pull_impulse_ratio,
        d4_knee_plateau_decouple_ratio
    ], dtype=np.float32)

    # 결측치 방어
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats

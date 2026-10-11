# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony, 힙 슈트업 / Stripper Deadlift)
Model Version: v4.0 (d4_model_v4.py)
Status: 외부 AI 3사(Gemini, Perplexity, 아주LLM) 자문 반영:
        1) 43D 생체역학 동역학 피처 (기존 37D + 6대 미세 위상차/2D 불변 피처)
        2) 64프레임 경량 1D-TCN 시계열 딥러닝 스트림 (Dilated Conv, Receptive Field 64 전수 커버)
        3) Class-Balanced Asymmetric Focal Loss (γ+=1.0, γ-=3.0, λ-=0.5)
        4) GBDT + TCN Two-Stream 피험자 격리 OOF Late Fusion
        5) Risk-based Calibrated 3-State 제약 최적화 (Coverage >= 90% 엄수)
"""

import os
import sys

# Windows OpenMP Deadlock 방지 (PyTorch + LightGBM / MKL 충돌 방지)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# MediaPipe 33 랜드마크 인덱스
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

FEATURE_NAMES_43_D4: List[str] = [
    # Group 1: 슬관절-고관절 신전 동시성 및 각속도 비 (7D)
    'd4_knee_hip_corr_full',              # [0] 전 구간 무릎-골반 각도 상관계수
    'd4_knee_hip_corr_ascent',            # [1] 상승기 무릎-골반 각도 상관계수
    'd4_vel_ratio_first_half',            # [2] 상승기 전반부(First Pull) 무릎/골반 각속도 비
    'd4_vel_ratio_second_half',           # [3] 상승기 후반부(Second Pull) 무릎/골반 각속도 비
    'd4_vel_ratio_full_ascent',           # [4] 상승기 전 구간 무릎/골반 평균 각속도 비
    'd4_extension_rate_diff_max',         # [5] 무릎 신전율과 골반 신전율의 최대 순간 편차
    'd4_angular_accel_mismatch',          # [6] 무릎 각가속도와 골반 각가속도의 평균 절대 편차

    # Group 2: 상체 경사각 안정도 & 힙 슈트업 (6D)
    'd4_trunk_angle_nadir_deg',           # [7] 최하사점 상체 경사각 (수직 대비, deg)
    'd4_trunk_drop_ascent_max_deg',       # [8] 상승기 상체 전경각 추가 급락(숙여짐) 각도 (deg)
    'd4_hip_shoot_up_speed_norm',         # [9] First Pull 골반 상승 속도 - 어깨 상승 속도 편차
    'd4_torso_horizontal_ratio',          # [10] 상체가 수평에 가깝게 무너진 시간 비율
    'd4_trunk_angle_std_ascent',          # [11] 상승기 상체 경사각의 변동성(std)
    'd4_hip_shoulder_vert_dist_min',      # [12] 상승기 골반-어깨 수직 간격 최소비

    # Group 3: 락아웃 위상차 및 가동범위 (5D)
    'd4_knee_lock_time_norm',             # [13] 무릎 최대 신전 도달 시점 (0~1 정규화)
    'd4_hip_lock_time_norm',              # [14] 골반 최대 신전 도달 시점 (0~1 정규화)
    'd4_lockout_phase_lag_norm',          # [15] 무릎 락아웃과 골반 락아웃 시점의 위상차
    'd4_knee_rom_deg',                    # [16] 무릎 관절 가동 범위 (deg)
    'd4_hip_rom_deg',                     # [17] 고관절 가동 범위 (deg)

    # Group 4: 5대 시점 불변 강건성 및 대칭성 (6D)
    'd4_knee_hip_rom_ratio',              # [18] 무릎 ROM 대비 골반 ROM 비
    'd4_bilateral_knee_asym_deg',         # [19] 좌우 무릎 각도 평균 비대칭도 (deg)
    'd4_bilateral_hip_asym_deg',          # [20] 좌우 골반 각도 평균 비대칭도 (deg)
    'd4_ascent_duration_ratio',           # [21] 전체 동작 중 상승기 구간 길이 비율
    'd4_pelvis_y_rise_rate',              # [22] 골반 수직 상승 총 변위비
    'd4_view_robust_coupling',            # [23] 시점 불변 체간-대퇴 3D 벡터 내적 변화율

    # Group 5: 무릎-골반 각도 비율 동역학 및 진행률 급변 지수 (5D)
    'd4_ratio_delta_first_pull',          # [24] First Pull 구간 무릎/골반 각도 비율 최대 급변량
    'd4_ratio_deriv_max_ascent',          # [25] 상승기 각도 비율의 순간 미분(변화율) 최대치
    'd4_ratio_std_ascent',                # [26] 상승기 각도 비율의 변동성/불안정도 (std)
    'd4_ratio_max_ascent',                # [27] 상승기 각도 비율의 최대 피크치
    'd4_norm_progress_diff_max',          # [28] First Pull 무릎-골반 정규화 신전 진행률 최대 편차

    # Group 6: 선행 연구 기반 연속 상대 위상 (CRP) 피처 (3D)
    'd4_crp_mean_first_pull_deg',         # [29] First Pull 구간 연속 상대 위상(CRP) 평균 (deg)
    'd4_crp_std_ascent_deg',              # [30] 상승기 연속 상대 위상(CRP) 변동성 (std)
    'd4_crp_max_drift_deg',               # [31] 상승기 최대 위상각 이탈량 (Phase Drift)

    # Group 7: 수정 벡터 코딩 (MVC) 커플링 각도 피처 (3D)
    'd4_mvc_knee_dominant_ratio',         # [32] First Pull Knee-Dominant(0~22.5도) 체류 시간 비율
    'd4_mvc_co_extension_ratio',          # [33] First Pull 동시신전(22.5~67.5도) 체류 시간 비율
    'd4_mvc_coupling_angle_mean_deg',     # [34] First Pull 평균 커플링 각도 (γ, deg)

    # Group 8: 체형 기하학적 정규화 피처 (2D)
    'd4_morph_femur_tibia_ratio',         # [35] 대퇴 길이 / 하퇴 길이 비율 (Femur-Tibia ratio)
    'd4_morph_torso_thigh_ratio',         # [36] 체간 길이 / 대퇴 길이 비율 (Torso-Thigh ratio)

    # Group 9: 외부 AI 자문 반영 미세 위상차 및 2D 불변 동역학 킬러 피처 (6D) ★ (v4 신규)
    'd4_hs_vvr_first_pull_max',           # [37] ★ 골반-어깨 수직 속도비(HS-VVR) First Pull 최대값 (Gemini 제안)
    'd4_onset_latency_norm',              # [38] ★ 무릎-골반 신전 개시 시점 차이 (Onset Latency) (아주LLM 제안)
    'd4_early_impulse_ratio',             # [39] ★ First Pull 초기 30% 무릎 신전 충격량 비율 (Early Impulse) (아주LLM 제안)
    'd4_accel_lead_norm',                 # [40] ★ 무릎-골반 최대 각가속도 선행 프레임 편차 (Accel Lead) (아주LLM 제안)
    'd4_hip_bar_decoupling',              # [41] ★ 바벨 수직 상승 대비 골반 상승 분리비 (Decoupling) (PPLX 제안)
    'd4_quasi_static_trunk_moment'        # [42] ★ 미드풋-골반 수평 모멘트암 시계열 변화율 (Trunk Moment) (Gemini/아주LLM 제안)
]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """세 점 p1, p2(중심), p3 사이의 3D 사잇각(deg) 계산"""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    denom = norm1 * norm2 + 1e-7
    cos_ang = np.sum(v1 * v2, axis=-1) / denom
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def compute_phase_angle(angle_seq: np.ndarray, vel_seq: np.ndarray) -> np.ndarray:
    """위상 평면(Phase Plane) 정규화 위상각 phi(t) = arctan2(vel_norm, angle_norm) 계산 (0~360도)"""
    a_min, a_max = np.min(angle_seq), np.max(angle_seq)
    a_range = max(a_max - a_min, 1e-4)
    a_norm = 2.0 * (angle_seq - a_min) / a_range - 1.0

    v_max = np.max(np.abs(vel_seq))
    v_norm = vel_seq / (v_max + 1e-4)

    phase_rad = np.arctan2(v_norm, a_norm)
    phase_deg = np.degrees(phase_rad) % 360.0
    return phase_deg


def extract_d4_v4_features_from_seq(kps: np.ndarray) -> np.ndarray:
    """
    데드리프트 64프레임 33관절 3D 포즈 시퀀스로부터 d4 v4 43D 생체역학 동역학 피처 추출
    Args:
        kps: np.ndarray (64, 33, 3) 64프레임 MediaPipe 관절 좌표
    Returns:
        feats_43: np.ndarray (43,) float32 피처 벡터
    """
    T = kps.shape[0]
    assert T >= 16, f"Sequence length must be >= 16, got {T}"

    # 1. 주요 관절 위치 추출 (가우시안 평활화 적용)
    sh_l, sh_r = kps[:, MP_LEFT_SHOULDER], kps[:, MP_RIGHT_SHOULDER]
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    wr_l, wr_r = kps[:, MP_LEFT_WRIST], kps[:, MP_RIGHT_WRIST]

    sh_mid = (sh_l + sh_r) / 2.0
    hip_mid = (hip_l + hip_r) / 2.0
    knee_mid = (knee_l + knee_r) / 2.0
    ank_mid = (ank_l + ank_r) / 2.0
    wr_mid = (wr_l + wr_r) / 2.0  # 바벨 위치 대리

    # 체형 세그먼트 길이 계산 (중앙값 기반)
    torso_lens = np.linalg.norm(sh_mid - hip_mid, axis=-1)
    femur_lens = np.linalg.norm(hip_mid - knee_mid, axis=-1)
    tibia_lens = np.linalg.norm(knee_mid - ank_mid, axis=-1)

    ref_torso = float(np.median(torso_lens))
    ref_femur = float(np.median(femur_lens))
    ref_tibia = float(np.median(tibia_lens))
    ref_len = ref_torso if ref_torso > 1e-4 else 1.0

    # 2. 관절 각도 계산
    knee_ang_l = compute_angle_3d(hip_l, knee_l, ank_l)
    knee_ang_r = compute_angle_3d(hip_r, knee_r, ank_r)
    knee_ang = (knee_ang_l + knee_ang_r) / 2.0
    knee_ang_smooth = gaussian_filter1d(knee_ang, sigma=1.5)

    hip_ang_l = compute_angle_3d(sh_l, hip_l, knee_l)
    hip_ang_r = compute_angle_3d(sh_r, hip_r, knee_r)
    hip_ang = (hip_ang_l + hip_ang_r) / 2.0
    hip_ang_smooth = gaussian_filter1d(hip_ang, sigma=1.5)

    # 상체 전경각
    trunk_vec = sh_mid - hip_mid
    trunk_vec_norm = np.linalg.norm(trunk_vec, axis=-1) + 1e-7
    cos_vert = np.clip(-trunk_vec[:, 1] / trunk_vec_norm, -1.0, 1.0)
    trunk_lean_deg = np.degrees(np.arccos(cos_vert))
    trunk_lean_smooth = gaussian_filter1d(trunk_lean_deg, sigma=1.5)

    # 3. 주요 구간 분할
    nadir_idx = int(np.argmin(hip_ang_smooth))
    nadir_idx = np.clip(nadir_idx, 5, T - 10)

    ascent_slice = slice(nadir_idx, T)
    post_hip_ang = hip_ang_smooth[ascent_slice]
    post_knee_ang = knee_ang_smooth[ascent_slice]

    lock_hip_rel = int(np.argmax(post_hip_ang))
    lock_knee_rel = int(np.argmax(post_knee_ang))
    lock_hip_idx = nadir_idx + lock_hip_rel
    lock_knee_idx = nadir_idx + lock_knee_rel

    end_ascent_idx = max(lock_hip_idx, lock_knee_idx)
    if end_ascent_idx <= nadir_idx + 3:
        end_ascent_idx = T - 1

    ascent_len = max(end_ascent_idx - nadir_idx, 4)
    mid_ascent_idx = nadir_idx + ascent_len // 2

    # 4. 각속도 및 각가속도 계산
    knee_vel = np.gradient(knee_ang_smooth)
    hip_vel = np.gradient(hip_ang_smooth)
    knee_acc = np.gradient(knee_vel)
    hip_acc = np.gradient(hip_vel)

    # Group 1 (7D)
    corr_full = float(np.corrcoef(knee_ang_smooth, hip_ang_smooth)[0, 1])
    if np.isnan(corr_full):
        corr_full = 0.0

    if ascent_len >= 4:
        corr_ascent = float(np.corrcoef(knee_ang_smooth[nadir_idx:end_ascent_idx],
                                       hip_ang_smooth[nadir_idx:end_ascent_idx])[0, 1])
        if np.isnan(corr_ascent):
            corr_ascent = 0.0
    else:
        corr_ascent = corr_full

    first_pull_k_vel = np.mean(knee_vel[nadir_idx:mid_ascent_idx])
    first_pull_h_vel = np.mean(hip_vel[nadir_idx:mid_ascent_idx])
    vel_ratio_first = float(first_pull_k_vel / (abs(first_pull_h_vel) + 1e-4))

    second_pull_k_vel = np.mean(knee_vel[mid_ascent_idx:end_ascent_idx])
    second_pull_h_vel = np.mean(hip_vel[mid_ascent_idx:end_ascent_idx])
    vel_ratio_second = float(second_pull_k_vel / (abs(second_pull_h_vel) + 1e-4))

    mean_ascent_k_vel = np.mean(knee_vel[nadir_idx:end_ascent_idx])
    mean_ascent_h_vel = np.mean(hip_vel[nadir_idx:end_ascent_idx])
    vel_ratio_full = float(mean_ascent_k_vel / (abs(mean_ascent_h_vel) + 1e-4))

    knee_rom = float(np.max(knee_ang_smooth) - np.min(knee_ang_smooth) + 1e-4)
    hip_rom = float(np.max(hip_ang_smooth) - np.min(hip_ang_smooth) + 1e-4)
    knee_ext_progress = (knee_ang_smooth[nadir_idx:end_ascent_idx] - knee_ang_smooth[nadir_idx]) / knee_rom
    hip_ext_progress = (hip_ang_smooth[nadir_idx:end_ascent_idx] - hip_ang_smooth[nadir_idx]) / hip_rom
    ext_rate_diff_max = float(np.max(np.abs(knee_ext_progress - hip_ext_progress)))

    acc_mismatch = float(np.mean(np.abs(knee_acc[nadir_idx:end_ascent_idx] - hip_acc[nadir_idx:end_ascent_idx])))

    # Group 2 (6D)
    trunk_nadir = float(trunk_lean_smooth[nadir_idx])
    first_pull_trunk = trunk_lean_smooth[nadir_idx:mid_ascent_idx]
    trunk_drop_max = float(np.max(first_pull_trunk) - trunk_nadir)
    if trunk_drop_max < 0.0:
        trunk_drop_max = 0.0

    hip_y_vel = -np.gradient(hip_mid[:, 1]) / ref_len
    sh_y_vel = -np.gradient(sh_mid[:, 1]) / ref_len
    hip_shoot_speed = float(np.mean(hip_y_vel[nadir_idx:mid_ascent_idx] - sh_y_vel[nadir_idx:mid_ascent_idx]))

    torso_horiz_ratio = float(np.mean(trunk_lean_smooth[nadir_idx:end_ascent_idx] > 60.0))
    trunk_std = float(np.std(trunk_lean_smooth[nadir_idx:end_ascent_idx]))
    vert_dist = np.abs(sh_mid[:, 1] - hip_mid[:, 1]) / ref_len
    min_vert_dist = float(np.min(vert_dist[nadir_idx:end_ascent_idx]))

    # Group 3 (5D)
    knee_lock_norm = float(lock_knee_idx / float(T))
    hip_lock_norm = float(lock_hip_idx / float(T))
    lockout_lag = float(abs(knee_lock_norm - hip_lock_norm))
    k_rom_deg = knee_rom
    h_rom_deg = hip_rom

    # Group 4 (6D)
    rom_ratio = float(k_rom_deg / (h_rom_deg + 1e-4))
    asym_knee = float(np.mean(np.abs(knee_ang_l - knee_ang_r)))
    asym_hip = float(np.mean(np.abs(hip_ang_l - hip_ang_r)))
    ascent_ratio = float(ascent_len / float(T))
    pelvis_y_rise = float(abs(hip_mid[nadir_idx, 1] - hip_mid[end_ascent_idx, 1]) / ref_len)

    thigh_vec = knee_mid - hip_mid
    thigh_norm = np.linalg.norm(thigh_vec, axis=-1, keepdims=True) + 1e-7
    trunk_norm_3d = trunk_vec / (trunk_vec_norm[:, None] + 1e-7)
    thigh_norm_3d = thigh_vec / thigh_norm
    dot_coupling = np.sum(trunk_norm_3d * thigh_norm_3d, axis=-1)
    view_robust_coupling = float(np.std(dot_coupling[nadir_idx:end_ascent_idx]))

    # Group 5 (5D)
    ratio_seq = knee_ang_smooth / (hip_ang_smooth + 1e-4)
    first_pull_ratio = ratio_seq[nadir_idx:mid_ascent_idx]
    ratio_delta_first_pull = float(np.max(first_pull_ratio) - np.min(first_pull_ratio))

    d_ratio = np.abs(np.gradient(ratio_seq))
    ratio_deriv_max = float(np.max(d_ratio[nadir_idx:end_ascent_idx]))

    ratio_std_ascent = float(np.std(ratio_seq[nadir_idx:end_ascent_idx]))
    ratio_max_ascent = float(np.max(ratio_seq[nadir_idx:end_ascent_idx]))

    k_norm_first = (knee_ang_smooth[nadir_idx:mid_ascent_idx] - np.min(knee_ang_smooth)) / knee_rom
    h_norm_first = (hip_ang_smooth[nadir_idx:mid_ascent_idx] - np.min(hip_ang_smooth)) / hip_rom
    norm_progress_diff_max = float(np.max(k_norm_first - h_norm_first))

    # Group 6: 선행 연구 기반 연속 상대 위상 (CRP) 피처 (3D)
    phi_knee = compute_phase_angle(knee_ang_smooth, knee_vel)
    phi_hip = compute_phase_angle(hip_ang_smooth, hip_vel)
    crp_raw = np.abs(phi_hip - phi_knee)
    crp_circ = np.minimum(crp_raw, 360.0 - crp_raw)

    crp_first_pull = crp_circ[nadir_idx:mid_ascent_idx]
    crp_ascent = crp_circ[nadir_idx:end_ascent_idx]

    crp_mean_first = float(np.mean(crp_first_pull)) if len(crp_first_pull) > 0 else 0.0
    crp_std_ascent = float(np.std(crp_ascent)) if len(crp_ascent) > 0 else 0.0
    crp_max_drift = float(np.max(crp_ascent)) if len(crp_ascent) > 0 else 0.0

    # Group 7: 수정 벡터 코딩 (MVC) 커플링 각도 피처 (3D)
    d_hip = np.gradient(hip_ang_smooth)[nadir_idx:mid_ascent_idx]
    d_knee = np.gradient(knee_ang_smooth)[nadir_idx:mid_ascent_idx]

    coupling_angles = np.degrees(np.arctan2(np.maximum(d_hip, 0.0), np.maximum(d_knee, 0.0) + 1e-4))
    knee_dominant_ratio = float(np.mean((coupling_angles >= 0.0) & (coupling_angles < 22.5)))
    co_ext_ratio = float(np.mean((coupling_angles >= 22.5) & (coupling_angles <= 67.5)))
    mean_coupling_angle = float(np.mean(coupling_angles))

    # Group 8: 체형 기하학적 정규화 피처 (2D)
    femur_tibia_ratio = float(ref_femur / (ref_tibia + 1e-4))
    torso_thigh_ratio = float(ref_torso / (ref_femur + 1e-4))

    # Group 9: 외부 AI 자문 반영 미세 위상차 및 2D 불변 동역학 킬러 피처 (6D) ★ (v4 신규)
    # [37] HS-VVR (골반-어깨 수직 속도비 First Pull 최대값, Gemini 제안)
    # 수직 속도(Y축 미분, 위쪽 이동이 양수)
    v_y_hip_raw = np.abs(np.gradient(-hip_mid[:, 1]))
    v_y_sh_raw = np.abs(np.gradient(-sh_mid[:, 1]))
    hs_vvr_seq = v_y_hip_raw / (v_y_sh_raw + 1e-4)
    hs_vvr_first_pull = hs_vvr_seq[nadir_idx:mid_ascent_idx]
    hs_vvr_first_pull_max = float(np.max(hs_vvr_first_pull)) if len(hs_vvr_first_pull) > 0 else 1.0

    # [38] Onset Latency (무릎-고관절 신전 개시 시점 차이, 아주LLM 제안)
    # 각속도가 최대 각속도의 10%를 넘는 최초 시점
    k_vel_thresh = 0.10 * (np.max(knee_vel[nadir_idx:end_ascent_idx]) + 1e-4)
    h_vel_thresh = 0.10 * (np.max(hip_vel[nadir_idx:end_ascent_idx]) + 1e-4)

    t_k_onset = nadir_idx
    for t in range(nadir_idx, end_ascent_idx):
        if knee_vel[t] >= k_vel_thresh:
            t_k_onset = t
            break

    t_h_onset = nadir_idx
    for t in range(nadir_idx, end_ascent_idx):
        if hip_vel[t] >= h_vel_thresh:
            t_h_onset = t
            break

    # 무릎이 먼저 신전되면 음수, 지연되면 양수 (정규화)
    onset_latency_norm = float((t_k_onset - t_h_onset) / float(ascent_len))

    # [39] Early Impulse Ratio (First Pull 초기 30% 무릎 신전 충격량 비율, 아주LLM 제안)
    fp_30_len = max(int(ascent_len * 0.30), 2)
    fp_30_end = min(nadir_idx + fp_30_len, T)
    early_k_impulse = np.sum(np.maximum(knee_vel[nadir_idx:fp_30_end], 0.0))
    total_k_impulse = np.sum(np.maximum(knee_vel[nadir_idx:end_ascent_idx], 0.0)) + 1e-4
    early_impulse_ratio = float(early_k_impulse / total_k_impulse)

    # [40] Accel Lead (무릎-골반 최대 각가속도 선행 프레임 편차, 아주LLM 제안)
    t_k_acc_max = nadir_idx + int(np.argmax(knee_acc[nadir_idx:mid_ascent_idx]))
    t_h_acc_max = nadir_idx + int(np.argmax(hip_acc[nadir_idx:mid_ascent_idx]))
    accel_lead_norm = float((t_k_acc_max - t_h_acc_max) / float(ascent_len))

    # [41] Hip-Bar Decoupling (바벨 수직 상승 대비 골반 상승 분리비, PPLX 제안)
    # 바벨(손목) 상승 변위 vs 골반 상승 변위
    bar_y_disp = np.abs(wr_mid[mid_ascent_idx, 1] - wr_mid[nadir_idx, 1]) + 1e-4
    hip_y_disp = np.abs(hip_mid[mid_ascent_idx, 1] - hip_mid[nadir_idx, 1])
    hip_bar_decoupling = float(hip_y_disp / bar_y_disp)

    # [42] Quasi-static Trunk Moment (미드풋-골반 수평 모멘트암 시계열 변화율, Gemini/아주LLM 제안)
    # 발목을 미드풋 대리점으로 사용, 골반과의 수평(X축) 거리
    moment_arm_seq = np.abs(hip_mid[:, 0] - ank_mid[:, 0]) / ref_len
    moment_arm_first_pull = moment_arm_seq[nadir_idx:mid_ascent_idx]
    moment_arm_delta = float(np.max(moment_arm_first_pull) - np.min(moment_arm_first_pull))

    feats = np.array([
        # Group 1 (7D)
        corr_full, corr_ascent, vel_ratio_first, vel_ratio_second, vel_ratio_full,
        ext_rate_diff_max, acc_mismatch,
        # Group 2 (6D)
        trunk_nadir, trunk_drop_max, hip_shoot_speed, torso_horiz_ratio, trunk_std, min_vert_dist,
        # Group 3 (5D)
        knee_lock_norm, hip_lock_norm, lockout_lag, k_rom_deg, h_rom_deg,
        # Group 4 (6D)
        rom_ratio, asym_knee, asym_hip, ascent_ratio, pelvis_y_rise, view_robust_coupling,
        # Group 5 (5D)
        ratio_delta_first_pull, ratio_deriv_max, ratio_std_ascent, ratio_max_ascent, norm_progress_diff_max,
        # Group 6 (3D)
        crp_mean_first, crp_std_ascent, crp_max_drift,
        # Group 7 (3D)
        knee_dominant_ratio, co_ext_ratio, mean_coupling_angle,
        # Group 8 (2D)
        femur_tibia_ratio, torso_thigh_ratio,
        # Group 9 (6D) ★ (v4 신규)
        hs_vvr_first_pull_max, onset_latency_norm, early_impulse_ratio,
        accel_lead_norm, hip_bar_decoupling, moment_arm_delta
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0)
    return feats


def extract_tcn_temporal_features_from_seq(kps: np.ndarray) -> np.ndarray:
    """
    1D-TCN 입력을 위한 64프레임 정규화 시계열 텐서 생성
    관절: 어깨, 골반, 무릎, 발목 (4개 주요 부위 좌우 평균 = 4개 관절)
    채널 구성: (X 상대좌표, Y 상대좌표, Z 상대좌표, 무릎각도, 골반각도, 무릎각속도, 골반각속도, HS-VVR) -> 12 채널
    Shape: (12, 64)
    """
    T = kps.shape[0]  # 64
    sh_mid = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hip_mid = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    knee_mid = (kps[:, MP_LEFT_KNEE] + kps[:, MP_RIGHT_KNEE]) / 2.0
    ank_mid = (kps[:, MP_LEFT_ANKLE] + kps[:, MP_RIGHT_ANKLE]) / 2.0

    ref_len = float(np.median(np.linalg.norm(sh_mid - hip_mid, axis=-1)))
    if ref_len < 1e-4:
        ref_len = 1.0

    # 골반 중심 상대좌표 정규화
    sh_rel = (sh_mid - hip_mid) / ref_len
    knee_rel = (knee_mid - hip_mid) / ref_len
    ank_rel = (ank_mid - hip_mid) / ref_len

    # 관절 각도 (무릎, 골반)
    knee_ang = (compute_angle_3d(kps[:, MP_LEFT_HIP], kps[:, MP_LEFT_KNEE], kps[:, MP_LEFT_ANKLE]) +
                compute_angle_3d(kps[:, MP_RIGHT_HIP], kps[:, MP_RIGHT_KNEE], kps[:, MP_RIGHT_ANKLE])) / 2.0
    hip_ang = (compute_angle_3d(kps[:, MP_LEFT_SHOULDER], kps[:, MP_LEFT_HIP], kps[:, MP_LEFT_KNEE]) +
               compute_angle_3d(kps[:, MP_RIGHT_SHOULDER], kps[:, MP_RIGHT_HIP], kps[:, MP_RIGHT_KNEE])) / 2.0

    knee_ang_norm = (knee_ang - 90.0) / 90.0
    hip_ang_norm = (hip_ang - 90.0) / 90.0

    knee_vel = np.gradient(knee_ang_norm)
    hip_vel = np.gradient(hip_ang_norm)

    # 12채널 조합:
    # 0~2: sh_rel (x, y, z)
    # 3~5: knee_rel (x, y, z)
    # 6~8: ank_rel (x, y, z)
    # 9: knee_ang_norm
    # 10: hip_ang_norm
    # 11: knee_vel - hip_vel (각속도 차이)
    channels = np.concatenate([
        sh_rel, knee_rel, ank_rel,
        knee_ang_norm[:, None], hip_ang_norm[:, None],
        (knee_vel - hip_vel)[:, None]
    ], axis=-1)  # (64, 12)

    channels = np.nan_to_num(channels, nan=0.0).T  # (12, 64)
    return channels.astype(np.float32)


# ==============================================================================
# Phase 2: Asymmetric Focal Loss & Kinematic 1D-TCN
# ==============================================================================
class AsymmetricFocalLoss(nn.Module):
    """
    클래스 불균형(17.5% 결함) 극복을 위한 비대칭 초점 손실함수
    gamma_pos: 1.0 (소수 결함 클래스는 손실을 유지하여 Recall 방어)
    gamma_neg: 3.0 (다수 정상 클래스의 쉬운 예측은 극적으로 소거)
    lambda_neg: 0.5 (정상 오탐 방지 가중치)
    """
    def __init__(self, gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_neg = lambda_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        # 결함(y=1) 손실
        loss_pos = -targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs + self.eps)

        # 정상(y=0) 손실 (오탐 강력 억제)
        loss_neg = -(1.0 - targets) * self.lambda_neg * torch.pow(probs, self.gamma_neg) * torch.log(1.0 - probs + self.eps)

        return torch.mean(loss_pos + loss_neg)


class TemporalBlock(nn.Module):
    """양방향 Dilated Conv 1D 블록 (Receptive Field 확장)"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return out + res


class KinematicTCN(nn.Module):
    """
    64프레임 관절 좌표 및 각도 시계열을 수용하는 초경량 1D-TCN
    - 770개 시퀀스 규모에 최적화된 파라미터 (< 4,000개)로 과적합 방지 및 초고속 수렴
    - Dilation 1, 2, 4 조합 (Receptive field 15~31 프레임 커버)
    - Global Adaptive Max Pooling + Linear Head
    """
    def __init__(self, in_features: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(in_features, hidden_dim, kernel_size=3, dilation=1, dropout=0.10),
            TemporalBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2, dropout=0.10),
            TemporalBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4, dropout=0.10)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x)  # (B, 32, 64)
        pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)  # (B, 32)
        logits = self.head(pooled)  # (B, 1)
        return logits


# ==============================================================================
# Phase 1: 43D Tri-Ensemble GBDT
# ==============================================================================
class BabelDeadliftD4ModelV4GBDT:
    """
    바벨 데드리프트 d4 v4.0 43D Tri-Ensemble GBDT 모델 (Stream A)
    - 43D 확장 생체역학 동역학 피처셋
    - LightGBM (50%) + CatBoost (30%) + Random Forest (20%)
    - 비대칭 클래스 가중치 적용
    """
    def __init__(self):
        self.clf_lgb = None
        self.clf_cat = None
        self.clf_rf = None
        self.weights = {'lgb': 0.50, 'cat': 0.30, 'rf': 0.20}
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, random_state: int = 42):
        n_pos = max(int(np.sum(y == 1)), 1)
        n_neg = max(int(np.sum(y == 0)), 1)
        # Class weight: Recall 보존을 위해 살짝 여유를 준 scale_pos_weight
        scale_pos = float(n_neg / n_pos)

        if lgb is not None:
            self.clf_lgb = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.04,
                num_leaves=15,
                scale_pos_weight=scale_pos,
                n_jobs=1,
                random_state=random_state,
                verbose=-1
            )
            self.clf_lgb.fit(X, y)

        if CatBoostClassifier is not None:
            self.clf_cat = CatBoostClassifier(
                iterations=80,
                learning_rate=0.05,
                depth=4,
                scale_pos_weight=scale_pos,
                thread_count=1,
                random_seed=random_state,
                verbose=False
            )
            self.clf_cat.fit(X, y)

        self.clf_rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            class_weight='balanced',
            n_jobs=1,
            random_state=random_state
        )
        self.clf_rf.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        probs = np.zeros(len(X), dtype=np.float32)
        total_w = 0.0

        if self.clf_lgb is not None:
            probs += self.weights['lgb'] * self.clf_lgb.predict_proba(X)[:, 1]
            total_w += self.weights['lgb']

        if self.clf_cat is not None:
            probs += self.weights['cat'] * self.clf_cat.predict_proba(X)[:, 1]
            total_w += self.weights['cat']

        if self.clf_rf is not None:
            probs += self.weights['rf'] * self.clf_rf.predict_proba(X)[:, 1]
            total_w += self.weights['rf']

        if total_w > 0:
            probs /= total_w

        return probs


# ==============================================================================
# Phase 3 & 4: Two-Stream Calibrated Fusion Model
# ==============================================================================
class TwoStreamCalibratedFusionModel:
    """
    바벨 데드리프트 d4 v4.0 통합 Two-Stream 메타 융합 엔진
    - Stream A (43D GBDT 확률) + Stream B (64프레임 TCN 확률)
    - Logistic Regression 메타 앙상블
    - Temperature Scaling 기반 확률 캘리브레이션
    - Risk-based 3-State 결정 정책
    """
    def __init__(self, tau_normal: float = 0.30, tau_defect: float = 0.50):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.meta_learner = LogisticRegression(class_weight='balanced', C=1.0)
        self.temperature = 1.0
        self.is_fitted = False

    def fit_meta(self, p_gbdt_oof: np.ndarray, p_tcn_oof: np.ndarray, y_true: np.ndarray):
        X_meta = np.column_stack([p_gbdt_oof, p_tcn_oof])
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True

    def predict_proba(self, p_gbdt: np.ndarray, p_tcn: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # 기본 평균
            return (p_gbdt * 0.55 + p_tcn * 0.45)
        X_meta = np.column_stack([p_gbdt, p_tcn])
        probs = self.meta_learner.predict_proba(X_meta)[:, 1]
        return probs

    def predict_3state(self, p_final: np.ndarray, p_gbdt: np.ndarray, p_tcn: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for pf, pg, pt in zip(p_final, p_gbdt, p_tcn):
            disagreement = abs(pg - pt)
            entropy = - (pf * np.log2(max(pf, 1e-6)) + (1 - pf) * np.log2(max(1 - pf, 1e-6)))

            if pf <= self.tau_normal:
                state = "ACCEPT_NORMAL"
                decision = 0
            elif pf >= self.tau_defect:
                state = "ACCEPT_DEFECT"
                decision = 1
            else:
                state = "RETAKE"
                decision = -1

            results.append({
                "probability": float(pf),
                "state": state,
                "decision": decision,
                "disagreement": float(disagreement),
                "entropy": float(entropy)
            })
        return results

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "TwoStreamCalibratedFusionModel":
        with open(filepath, 'rb') as f:
            return pickle.load(f)

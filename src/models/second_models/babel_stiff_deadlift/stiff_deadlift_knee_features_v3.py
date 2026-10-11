# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v3.0 (Subject-Adaptive Normalized 10D & Baseline Succession)
Module: stiff_deadlift_knee_features_v3.py
Purpose: v2의 6D 정예 피처를 계승하고, 솔루션 2(어드레스 기준 상대 굴곡율 무차원화 정규화,
         대퇴골-정강이 골격비 등)를 반영한 10D 적응형 생체역학 불변 지표 및 8ch 시계열 추출
"""

import numpy as np

# 10대 정예 적응형 생체역학 피처 정의
FEATURE_NAMES_10D = [
    "bottom_knee_angle",       # 1. 최저점 무릎 절대 각도 (v2 계승)
    "bottom_knee_delta_mean",  # 2. 어드레스 대비 최저점 무릎 각도 변화량 (v2 계승)
    "femur_rom_mean",          # 3. 대퇴골(허벅지) 지면 경사각 가동범위 (v2 계승)
    "knee_disp_y_rom",         # 4. 무릎 수직(Y축) 이동 가동범위 (v2 계승)
    "shank_svii_mean",         # 5. 정강이 수직 불변성 지수 (v2 계승)
    "hip_hinge_isolation",     # 6. 고관절 힌지 격리 지수 (HHIR, v2 계승)
    "rel_bottom_knee_delta",   # 7. [솔루션 2] 어드레스 각도 대비 상대 굴곡 변화율 (|θ_bot - θ_0| / θ_0)
    "norm_knee_disp_y",        # 8. [솔루션 2] 대퇴골 길이 대비 무릎 수직 이동 무차원비 (Δy / L_femur)
    "address_knee_angle",      # 9. [솔루션 2] 어드레스 초기(0~5F) 평균 무릎 각도 (개인별 유연성 기준선)
    "femur_shank_ratio"        # 10.[솔루션 2] 대퇴골-정강이 분절 길이비 (신체 비율 체형 불변식)
]

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """관절 b를 중심으로 벡터 ba와 bc 사이의 3D 공간 각도(degree) 산출"""
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_knee_invariance_features_10d(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 솔루션 2(어드레스 상대 정규화)를 반영한 10D 적응형 생체역학 피처 추출
    """
    # 1. 관절 궤적 복원
    v_pelvis_torso = x_seq[:, 0:3]
    p_hip_l = x_seq[:, 32:35]
    p_hip_r = x_seq[:, 36:39]
    p_knee_l = p_hip_l + x_seq[:, 40:43]
    p_knee_r = p_hip_r + x_seq[:, 48:51]
    p_ank_l = p_knee_l + x_seq[:, 44:47]
    p_ank_r = p_knee_r + x_seq[:, 52:55]

    # 2. 각도 시계열
    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)
    ang_hip_l = calculate_angle_3d(v_pelvis_torso, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(v_pelvis_torso, p_hip_r, p_knee_r)

    # 3. 최저점(변곡점) 인덱스
    idx_bot = int(np.argmin(0.5 * (ang_hip_l + ang_hip_r)))

    # F1. 최저점 무릎 절대 각도 (v2)
    f1_bot_angle = float(0.5 * (ang_knee_l[idx_bot] + ang_knee_r[idx_bot]))

    # F2. 어드레스 대비 최저점 무릎 각도 변화량 (v2)
    bot_dev_l = abs(ang_knee_l[idx_bot] - ang_knee_l[0])
    bot_dev_r = abs(ang_knee_r[idx_bot] - ang_knee_r[0])
    f2_bot_delta = float(0.5 * (bot_dev_l + bot_dev_r))

    # F3. 대퇴골 지면 경사각 가동범위 (v2)
    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))
    f3_femur_rom = float(0.5 * ((np.max(pitch_femur_l) - np.min(pitch_femur_l)) + (np.max(pitch_femur_r) - np.min(pitch_femur_r))))

    # F4. 무릎 수직(Y축) 이동 가동범위 (v2)
    disp_y_l = p_knee_l[:, 1]
    disp_y_r = p_knee_r[:, 1]
    f4_disp_y_rom = float(0.5 * ((np.max(disp_y_l) - np.min(disp_y_l)) + (np.max(disp_y_r) - np.min(disp_y_r))))

    # F5. 정강이 수직 불변성 지수 (v2)
    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))
    f5_shank_svii = float(0.5 * ((np.max(pitch_shank_l) - np.min(pitch_shank_l)) + (np.max(pitch_shank_r) - np.min(pitch_shank_r))))

    # F6. 고관절 힌지 격리 지수 (HHIR, v2)
    hip_rom_l = np.max(ang_hip_l) - np.min(ang_hip_l)
    hip_rom_r = np.max(ang_hip_r) - np.min(ang_hip_r)
    knee_rom_l = np.max(ang_knee_l) - np.min(ang_knee_l)
    knee_rom_r = np.max(ang_knee_r) - np.min(ang_knee_r)
    hhir_l = hip_rom_l / (knee_rom_l + 1e-5)
    hhir_r = hip_rom_r / (knee_rom_r + 1e-5)
    f6_hhir = float(np.clip(0.5 * (hhir_l + hhir_r), 0.0, 20.0))

    # --------------------------------------------------------------------------
    # 솔루션 2 신규 적응형 피처 (F7 ~ F10)
    # --------------------------------------------------------------------------
    # F9. 어드레스 초기(0~5F) 평균 무릎 각도 (개인별 유연성 기준선)
    init_knee_l = np.mean(ang_knee_l[:6])
    init_knee_r = np.mean(ang_knee_r[:6])
    f9_address_angle = float(0.5 * (init_knee_l + init_knee_r))

    # F7. 어드레스 각도 대비 상대 굴곡 변화율 (|θ_bot - θ_0| / θ_0)
    rel_dev_l = abs(ang_knee_l[idx_bot] - init_knee_l) / (init_knee_l + 1e-5)
    rel_dev_r = abs(ang_knee_r[idx_bot] - init_knee_r) / (init_knee_r + 1e-5)
    f7_rel_delta = float(0.5 * (rel_dev_l + rel_dev_r))

    # 분절 길이 계산 (대퇴골 및 정강이)
    len_femur_l = np.mean(np.linalg.norm(v_femur_l, axis=-1)) + 1e-5
    len_femur_r = np.mean(np.linalg.norm(v_femur_r, axis=-1)) + 1e-5
    mean_len_femur = 0.5 * (len_femur_l + len_femur_r)

    len_shank_l = np.mean(np.linalg.norm(v_shank_l, axis=-1)) + 1e-5
    len_shank_r = np.mean(np.linalg.norm(v_shank_r, axis=-1)) + 1e-5
    mean_len_shank = 0.5 * (len_shank_l + len_shank_r)

    # F8. 대퇴골 길이 대비 무릎 수직 이동 무차원비 (Δy / L_femur)
    f8_norm_disp_y = float(f4_disp_y_rom / mean_len_femur)

    # F10. 대퇴골-정강이 분절 길이비 (체형 무차원 불변식)
    f10_femur_shank_ratio = float(mean_len_femur / mean_len_shank)

    feats_10d = np.array([
        f1_bot_angle, f2_bot_delta, f3_femur_rom, f4_disp_y_rom, f5_shank_svii,
        f6_hhir, f7_rel_delta, f8_norm_disp_y, f9_address_angle, f10_femur_shank_ratio
    ], dtype=np.float32)

    return feats_10d

def extract_knee_temporal_8ch_v3(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 v2를 계승한 8ch 시계열 텐서 추출 (64, 8)
    """
    p_hip_l = x_seq[:, 32:35]
    p_hip_r = x_seq[:, 36:39]
    p_knee_l = p_hip_l + x_seq[:, 40:43]
    p_knee_r = p_hip_r + x_seq[:, 48:51]
    p_ank_l = p_knee_l + x_seq[:, 44:47]
    p_ank_r = p_knee_r + x_seq[:, 52:55]

    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)

    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))

    disp_y_l = p_knee_l[:, 1]
    disp_y_r = p_knee_r[:, 1]

    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))

    series = np.stack([
        ang_knee_l, ang_knee_r,
        pitch_femur_l, pitch_femur_r,
        disp_y_l, disp_y_r,
        pitch_shank_l, pitch_shank_r
    ], axis=-1).astype(np.float32)

    return series

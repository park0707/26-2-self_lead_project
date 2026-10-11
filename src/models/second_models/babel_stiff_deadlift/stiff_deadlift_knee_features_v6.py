# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v6.0 Feature Extraction Module
Module: stiff_deadlift_knee_features_v6.py
Purpose:
  1. Pelvis-Gravity Canonical Transformation (신체 중심 정규 3D 좌표 변환)
  2. KADR (Knee-Ankle Displacement Ratio), PLI (Phase Lag Index), Excess Flexion Integral
  3. v3 10D 적응형 피처 전수 계승 + 컨텍스트 피처 확장 (총 19D)
  4. 8채널 정규 3D 동역학 시계열 텐서 추출
"""

import numpy as np

FEATURE_NAMES_19D = [
    # --- v3 계승 핵심 생체역학 피처 (10D) ---
    "bottom_knee_angle",          # 1. 최저점 무릎 절대 각도
    "bottom_knee_delta_mean",     # 2. 어드레스 대비 최저점 무릎 각도 변화량
    "femur_rom_mean",             # 3. 대퇴골 지면 경사각 가동범위
    "knee_disp_y_rom",            # 4. 무릎 수직(Y축) 이동 가동범위
    "shank_svii_mean",            # 5. 정강이 수직 불변성 지수
    "hip_hinge_isolation",        # 6. 고관절 힌지 격리 지수 (HHIR)
    "rel_bottom_knee_delta",      # 7. 어드레스 각도 대비 상대 굴곡 변화율
    "norm_knee_disp_y",           # 8. 대퇴골 길이 대비 무릎 수직 이동 무차원비
    "address_knee_angle",         # 9. 어드레스 초기(0~5F) 평균 무릎 각도
    "femur_shank_ratio",          # 10.대퇴골-정강이 분절 길이비

    # --- v6 신규 정규 3D 불변 생체역학 피처 (6D) ---
    "kadr_max",                   # 11.발목 기준 무릎 전방 초과 돌출비 최대값 (KT-DC)
    "kadr_mean",                  # 12.발목 기준 무릎 전방 돌출비 평균값
    "pli",                        # 13.고관절-무릎 각속도 위상 분리 지수 (HK-VPL)
    "excess_flex_integral",       # 14.어드레스 대비 단방향 추가 굴곡 적분량
    "excess_flex_max",            # 15.어드레스 대비 단방향 추가 굴곡 최대값
    "knee_disp_z_canonical_rom",  # 16.신체 정규 시상축(Z) 무릎 변위 가동범위

    # --- 컨텍스트 피처 (3D) ---
    "context_sd3_prob",           # 17.바벨 밀착 결함 사전 확률
    "context_torso_coronal_rom",  # 18.골반-체간 관상면(Coronal) 횡경사 가동범위
    "context_torso_pitch_rom"     # 19.체간 시상면 전방 경사각 가동범위
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

def compute_canonical_transform(p_hip_l, p_hip_r, p_ank_l, p_ank_r, p_knee_l, p_knee_r):
    """
    Pelvis-Gravity Canonical Transformation:
    신체 중심 및 중력 기반 3D 정규 좌표계 변환
    - e_x: 골반 좌우 관상축 (오른쪽 -> 왼쪽)
    - e_y: 중력 반대 수직축 (발목 -> 골반 상향)
    - e_z: 시상 전후축 (e_x x e_y, 정면 전방 방향)
    """
    T = p_hip_l.shape[0]
    p_root = 0.5 * (p_hip_l + p_hip_r) # (T, 3)

    # 1. Lateral 축 (e_x)
    v_lateral = p_hip_l - p_hip_r
    e_x = v_lateral / (np.linalg.norm(v_lateral, axis=-1, keepdims=True) + 1e-7)

    # 2. Vertical 축 (e_y): 발목 중심에서 골반으로 향하는 수직 상향 벡터
    p_mid_ank = 0.5 * (p_ank_l + p_ank_r)
    v_up = p_root - p_mid_ank
    dot_xy = np.sum(v_up * e_x, axis=-1, keepdims=True)
    v_y_ortho = v_up - dot_xy * e_x
    e_y = v_y_ortho / (np.linalg.norm(v_y_ortho, axis=-1, keepdims=True) + 1e-7)

    # 3. Sagittal 전방 축 (e_z): e_x x e_y (신체 정면 방향)
    e_z = np.cross(e_x, e_y) # (T, 3)

    # 4. 관절 위치 정규화 (발목 원점 기준)
    knee_mid = 0.5 * (p_knee_l + p_knee_r)
    rel_knee = knee_mid - p_mid_ank

    z_knee_rel = np.sum(rel_knee * e_z, axis=-1) # (T,)
    y_knee_rel = np.sum(rel_knee * e_y, axis=-1) # (T,)

    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    len_shank = 0.5 * (np.linalg.norm(v_shank_l, axis=-1).mean() + np.linalg.norm(v_shank_r, axis=-1).mean()) + 1e-5

    kadr = z_knee_rel / len_shank # (T,)

    return kadr, z_knee_rel, len_shank

def extract_knee_invariance_features_v6(x_seq: np.ndarray, p_sd3: float = 0.0) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 v6.0 19D 복합 생체역학 피처 추출
    """
    # 1. 3D 관절 궤적 복원
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
    ang_knee_mean = 0.5 * (ang_knee_l + ang_knee_r)
    ang_hip_l = calculate_angle_3d(v_pelvis_torso, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(v_pelvis_torso, p_hip_r, p_knee_r)
    ang_hip_mean = 0.5 * (ang_hip_l + ang_hip_r)

    # 3. 최저점(변곡점) 인덱스
    idx_bot = int(np.argmin(ang_hip_mean))

    # F1. 최저점 무릎 절대 각도
    f1_bot_angle = float(ang_knee_mean[idx_bot])

    # F2. 어드레스 대비 최저점 무릎 각도 변화량
    init_knee_l = float(np.mean(ang_knee_l[:6]))
    init_knee_r = float(np.mean(ang_knee_r[:6]))
    init_knee_mean = 0.5 * (init_knee_l + init_knee_r)
    f2_bot_delta = float(abs(ang_knee_mean[idx_bot] - init_knee_mean))

    # F3. 대퇴골 지면 경사각 가동범위
    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))
    f3_femur_rom = float(0.5 * ((np.max(pitch_femur_l) - np.min(pitch_femur_l)) + (np.max(pitch_femur_r) - np.min(pitch_femur_r))))

    # F4. 무릎 수직(Y축) 이동 가동범위
    disp_y_l = p_knee_l[:, 1]
    disp_y_r = p_knee_r[:, 1]
    f4_disp_y_rom = float(0.5 * ((np.max(disp_y_l) - np.min(disp_y_l)) + (np.max(disp_y_r) - np.min(disp_y_r))))

    # F5. 정강이 수직 불변성 지수
    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))
    f5_shank_svii = float(0.5 * ((np.max(pitch_shank_l) - np.min(pitch_shank_l)) + (np.max(pitch_shank_r) - np.min(pitch_shank_r))))

    # F6. 고관절 힌지 격리 지수 (HHIR)
    hip_rom = float(np.max(ang_hip_mean) - np.min(ang_hip_mean))
    knee_rom = float(np.max(ang_knee_mean) - np.min(ang_knee_mean))
    f6_hhir = float(np.clip(hip_rom / (knee_rom + 1e-5), 0.0, 20.0))

    # F7. 어드레스 각도 대비 상대 굴곡 변화율
    f7_rel_delta = float(f2_bot_delta / (init_knee_mean + 1e-5))

    # F8. 대퇴골 길이 대비 무릎 수직 이동 무차원비
    len_femur_l = np.mean(np.linalg.norm(v_femur_l, axis=-1)) + 1e-5
    len_femur_r = np.mean(np.linalg.norm(v_femur_r, axis=-1)) + 1e-5
    mean_len_femur = 0.5 * (len_femur_l + len_femur_r)
    f8_norm_disp_y = float(f4_disp_y_rom / mean_len_femur)

    # F9. 어드레스 초기 평균 무릎 각도
    f9_address_angle = float(init_knee_mean)

    # F10. 대퇴골-정강이 분절 길이비
    len_shank_l = np.mean(np.linalg.norm(v_shank_l, axis=-1)) + 1e-5
    len_shank_r = np.mean(np.linalg.norm(v_shank_r, axis=-1)) + 1e-5
    mean_len_shank = 0.5 * (len_shank_l + len_shank_r)
    f10_femur_shank_ratio = float(mean_len_femur / mean_len_shank)

    # --------------------------------------------------------------------------
    # v6 신규 정규 3D 피처 (F11 ~ F16)
    # --------------------------------------------------------------------------
    kadr_seq, z_knee_seq, _ = compute_canonical_transform(
        p_hip_l, p_hip_r, p_ank_l, p_ank_r, p_knee_l, p_knee_r
    )
    # F11. KADR max
    f11_kadr_max = float(np.max(kadr_seq))

    # F12. KADR mean
    f12_kadr_mean = float(np.mean(kadr_seq))

    # F13. PLI (고관절-무릎 각속도 위상 분리 지수)
    omega_knee = np.sum(np.abs(np.diff(ang_knee_mean)))
    omega_hip = np.sum(np.abs(np.diff(ang_hip_mean)))
    f13_pli = float(np.clip(1.0 - (omega_knee / (omega_hip + 1e-5)), -1.0, 1.0))

    # F14. 단방향 추가 굴곡 적분량 (Excess Flexion Integral)
    excess_flex_seq = np.maximum(0.0, init_knee_mean - ang_knee_mean - 5.0)
    f14_excess_flex_integral = float(np.mean(excess_flex_seq))

    # F15. 단방향 추가 굴곡 최대값 (Excess Flexion Max)
    f15_excess_flex_max = float(np.max(excess_flex_seq))

    # F16. 신체 시상축(Z) 무릎 변위 가동범위
    f16_knee_disp_z_rom = float(np.max(z_knee_seq) - np.min(z_knee_seq))

    # --------------------------------------------------------------------------
    # 컨텍스트 피처 (F17 ~ F19)
    # --------------------------------------------------------------------------
    # F17. 바벨 밀착 결함 확률
    f17_context_sd3 = float(p_sd3)

    # F18. 체간 관상면(Coronal) 횡경사 가동범위
    roll_torso = np.degrees(np.arctan2(v_pelvis_torso[:, 0], np.abs(v_pelvis_torso[:, 1]) + 1e-7))
    f18_torso_coronal_rom = float(np.max(roll_torso) - np.min(roll_torso))

    # F19. 체간 시상면(Sagittal) 전방 경사각 가동범위
    pitch_torso = np.degrees(np.arctan2(v_pelvis_torso[:, 2], np.abs(v_pelvis_torso[:, 1]) + 1e-7))
    f19_torso_pitch_rom = float(np.max(pitch_torso) - np.min(pitch_torso))

    return np.array([
        f1_bot_angle, f2_bot_delta, f3_femur_rom, f4_disp_y_rom, f5_shank_svii,
        f6_hhir, f7_rel_delta, f8_norm_disp_y, f9_address_angle, f10_femur_shank_ratio,
        f11_kadr_max, f12_kadr_mean, f13_pli, f14_excess_flex_integral, f15_excess_flex_max,
        f16_knee_disp_z_rom, f17_context_sd3, f18_torso_coronal_rom, f19_torso_pitch_rom
    ], dtype=np.float32)

def extract_knee_temporal_8ch_v6(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 v6 정규 8ch 시계열 텐서 추출 (64, 8)
    - ch0: 무릎 각도 좌
    - ch1: 무릎 각도 우
    - ch2: 대퇴골 피치 좌
    - ch3: 대퇴골 피치 우
    - ch4: 정규 시상 무릎 변위 (KADR 시계열)
    - ch5: 정강이 피치 좌
    - ch6: 정강이 피치 우
    - ch7: 고관절 각도 평균
    """
    p_hip_l = x_seq[:, 32:35]
    p_hip_r = x_seq[:, 36:39]
    p_knee_l = p_hip_l + x_seq[:, 40:43]
    p_knee_r = p_hip_r + x_seq[:, 48:51]
    p_ank_l = p_knee_l + x_seq[:, 44:47]
    p_ank_r = p_knee_r + x_seq[:, 52:55]
    v_pelvis_torso = x_seq[:, 0:3]

    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)

    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))

    kadr_seq, _, _ = compute_canonical_transform(p_hip_l, p_hip_r, p_ank_l, p_ank_r, p_knee_l, p_knee_r)

    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))

    ang_hip_l = calculate_angle_3d(v_pelvis_torso, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(v_pelvis_torso, p_hip_r, p_knee_r)
    ang_hip_mean = 0.5 * (ang_hip_l + ang_hip_r)

    return np.stack([
        ang_knee_l, ang_knee_r,
        pitch_femur_l, pitch_femur_r,
        kadr_seq,
        pitch_shank_l, pitch_shank_r,
        ang_hip_mean
    ], axis=-1).astype(np.float32)

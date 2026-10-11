# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1)
Version: v2.0 (Curated 6D Feature Diet)
Module: stiff_deadlift_knee_features_v2.py
Purpose: v1의 14D 피처 중 LOFO 전수 감사로 규명된 8대 성능 방해 피처를 전수 제거하고,
         결함 판별에 실질적으로 기여하는 6대 정예 생체역학 불변 지표 및 정예 시계열 텐서 추출
"""

import numpy as np

# 6대 정예 피처 정의 (LOFO 감사에서 검증된 핵심 기여 피처만 선별)
FEATURE_NAMES_6D = [
    "bottom_knee_angle",       # 1. 최저점 무릎 절대 각도 (1위 핵심 지표, LOFO -3.67%p)
    "bottom_knee_delta_mean",  # 2. 어드레스 대비 최저점 무릎 각도 변화량 (변곡점 고정성 지표)
    "femur_rom_mean",          # 3. 대퇴골(허벅지) 지면 경사각 가동범위 (고관절 힌지 수행성)
    "knee_disp_y_rom",         # 4. 무릎 수직(Y축) 이동 가동범위 (스쿼트화 포착 지표)
    "shank_svii_mean",         # 5. 정강이 수직 불변성 지수 (정강이 축 흔들림)
    "hip_hinge_isolation"      # 6. 고관절 힌지 격리 지수 (HHIR = Hip ROM / (Knee ROM + eps))
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

def extract_knee_invariance_features_6d(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 방해 피처 8개를 제거한 6D 정예 생체역학 피처 추출
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

    # F1. 최저점 무릎 절대 각도
    f1_bot_angle = float(0.5 * (ang_knee_l[idx_bot] + ang_knee_r[idx_bot]))

    # F2. 어드레스 대비 최저점 무릎 각도 변화량
    bot_dev_l = abs(ang_knee_l[idx_bot] - ang_knee_l[0])
    bot_dev_r = abs(ang_knee_r[idx_bot] - ang_knee_r[0])
    f2_bot_delta = float(0.5 * (bot_dev_l + bot_dev_r))

    # F3. 대퇴골 가동범위
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    femur_l = p_knee_l - p_hip_l
    femur_r = p_knee_r - p_hip_r
    norm_fl = np.linalg.norm(femur_l, axis=-1) + 1e-7
    norm_fr = np.linalg.norm(femur_r, axis=-1) + 1e-7
    cos_fl = np.clip(np.sum(femur_l * v_vert, axis=-1) / norm_fl, -1.0, 1.0)
    cos_fr = np.clip(np.sum(femur_r * v_vert, axis=-1) / norm_fr, -1.0, 1.0)
    ang_fl = np.degrees(np.arccos(cos_fl))
    ang_fr = np.degrees(np.arccos(cos_fr))
    f3_femur_rom = float(0.5 * ((np.max(ang_fl) - np.min(ang_fl)) + (np.max(ang_fr) - np.min(ang_fr))))

    # F4. 무릎 수직(Y축) 이동 가동범위
    p_knee_mid_y = 0.5 * (p_knee_l[:, 1] + p_knee_r[:, 1])
    f4_disp_y_rom = float(np.max(p_knee_mid_y) - np.min(p_knee_mid_y))

    # F5. 정강이 수직 불변성 지수 평균
    shank_l = p_ank_l - p_knee_l
    shank_r = p_ank_r - p_knee_r
    norm_sl = np.linalg.norm(shank_l, axis=-1) + 1e-7
    norm_sr = np.linalg.norm(shank_r, axis=-1) + 1e-7
    cos_sl = np.clip(np.sum(shank_l * v_vert, axis=-1) / norm_sl, -1.0, 1.0)
    cos_sr = np.clip(np.sum(shank_r * v_vert, axis=-1) / norm_sr, -1.0, 1.0)
    ang_sl = np.degrees(np.arccos(cos_sl))
    ang_sr = np.degrees(np.arccos(cos_sr))
    f5_svii_mean = float(0.5 * (np.std(ang_sl) + np.std(ang_sr)))

    # F6. 고관절 힌지 격리 지수 (HHIR)
    rom_knee_l = float(np.max(ang_knee_l) - np.min(ang_knee_l))
    rom_knee_r = float(np.max(ang_knee_r) - np.min(ang_knee_r))
    f_knee_rom = 0.5 * (rom_knee_l + rom_knee_r)
    rom_hip_l = float(np.max(ang_hip_l) - np.min(ang_hip_l))
    rom_hip_r = float(np.max(ang_hip_r) - np.min(ang_hip_r))
    rom_hip = 0.5 * (rom_hip_l + rom_hip_r)
    f6_hhir = float(rom_hip / (f_knee_rom + 1e-4))

    return np.array([f1_bot_angle, f2_bot_delta, f3_femur_rom, f4_disp_y_rom, f5_svii_mean, f6_hhir], dtype=np.float32)

def extract_knee_temporal_8ch_v2(x_seq: np.ndarray) -> np.ndarray:
    """
    v2 TCN용 8채널 정예 시계열 궤적 추출
    방해 요소였던 Z축 노이즈 채널을 배제하고, 핵심인 각도 및 Y축 수직 변위만 압축
    (64, 8):
      - 0~1: 좌/우 무릎 각도
      - 2~3: 좌/우 고관절 각도
      - 4~5: 좌/우 정강이 각도
      - 6~7: 좌/우 무릎 Y축 수직 변위
    """
    v_pelvis_torso = x_seq[:, 0:3]
    p_hip_l = x_seq[:, 32:35]
    p_hip_r = x_seq[:, 36:39]
    p_knee_l = p_hip_l + x_seq[:, 40:43]
    p_knee_r = p_hip_r + x_seq[:, 48:51]
    p_ank_l = p_knee_l + x_seq[:, 44:47]
    p_ank_r = p_knee_r + x_seq[:, 52:55]

    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l) / 180.0
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r) / 180.0
    ang_hip_l = calculate_angle_3d(v_pelvis_torso, p_hip_l, p_knee_l) / 180.0
    ang_hip_r = calculate_angle_3d(v_pelvis_torso, p_hip_r, p_knee_r) / 180.0

    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    shank_l = p_ank_l - p_knee_l
    shank_r = p_ank_r - p_knee_r
    norm_sl = np.linalg.norm(shank_l, axis=-1) + 1e-7
    norm_sr = np.linalg.norm(shank_r, axis=-1) + 1e-7
    cos_sl = np.clip(np.sum(shank_l * v_vert, axis=-1) / norm_sl, -1.0, 1.0)
    cos_sr = np.clip(np.sum(shank_r * v_vert, axis=-1) / norm_sr, -1.0, 1.0)
    ang_sl = np.arccos(cos_sl) / np.pi
    ang_sr = np.arccos(cos_sr) / np.pi

    disp_ky_l = p_knee_l[:, 1] - p_knee_l[0, 1]
    disp_ky_r = p_knee_r[:, 1] - p_knee_r[0, 1]

    t8 = np.stack([
        ang_knee_l, ang_knee_r,
        ang_hip_l, ang_hip_r,
        ang_sl, ang_sr,
        disp_ky_l, disp_ky_r
    ], axis=-1).astype(np.float32)

    return t8

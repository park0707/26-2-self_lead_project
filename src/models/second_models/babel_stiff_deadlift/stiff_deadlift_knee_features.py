# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (Knee Fixed-Angle Defect)
Module: stiff_deadlift_knee_features.py
Purpose: 순수 비전 64프레임 MediaPipe 3D 관절 좌표로부터 14대 무릎 불변성 생체역학 피처 및 12ch 동역학 텐서 추출
"""

import numpy as np

FEATURE_NAMES_14D = [
    "knee_rom_mean",           # 1. 좌우 평균 무릎 관절 가동범위 (Knee ROM)
    "knee_rom_max",            # 2. 좌우 최대 무릎 관절 가동범위
    "hip_hinge_isolation",     # 3. 고관절 힌지 격리 지수 (HHIR = Hip ROM / (Knee ROM + eps))
    "knee_vel_rms_mean",       # 4. 무릎 각속도 잔차 RMS 에너지 평균
    "knee_vel_rms_max",        # 5. 무릎 각속도 잔차 RMS 에너지 최대
    "shank_svii_mean",         # 6. 정강이 수직 불변성 지수 (SVII = std(theta_shank)) 평균
    "shank_svii_max",          # 7. 정강이 수직 불변성 지수 최대
    "bottom_knee_delta_mean",  # 8. 최저점(변곡점) 무릎 각도 편차 (|theta_bot - theta_start|)
    "bottom_knee_angle",       # 9. 최저점 무릎 절대 각도
    "femur_rom_mean",          # 10. 대퇴골(허벅지) 지면 경사각 가동범위
    "hip_knee_coupling_r",     # 11. 고관절-무릎 각속도 피어슨 상관계수 (동조화 방지 척도)
    "knee_disp_z_rms",         # 12. 무릎 전후(Z축) 공간 변위 RMS (과신전/밀림 지표)
    "knee_disp_y_rom",         # 13. 무릎 수직(Y축) 이동 가동범위 (스쿼트화 지표)
    "pelvis_knee_dist_rom"     # 14. 골반-무릎 상대 거리 변동폭
]

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    관절 b를 중심으로 벡터 ba와 bc 사이의 3D 공간 각도(degree) 산출
    a, b, c: (T, 3) -> return (T,)
    """
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_knee_invariance_features_14d(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 14D 생체역학 불변성 피처 벡터 추출
    x_seq: (64, 98) X_Format_B 텐서
    """
    # 1. 3D 관절 궤적 복원 (골반 원점 기준)
    v_pelvis_torso = x_seq[:, 0:3]
    v_pelvis_hip_l = x_seq[:, 32:35]
    v_pelvis_hip_r = x_seq[:, 36:39]
    v_hip_knee_l = x_seq[:, 40:43]
    v_hip_knee_r = x_seq[:, 48:51]
    v_knee_ank_l = x_seq[:, 44:47]
    v_knee_ank_r = x_seq[:, 52:55]

    # 절대 좌표 계산
    p_hip_l = v_pelvis_hip_l
    p_hip_r = v_pelvis_hip_r
    p_knee_l = p_hip_l + v_hip_knee_l
    p_knee_r = p_hip_r + v_hip_knee_r
    p_ank_l = p_knee_l + v_knee_ank_l
    p_ank_r = p_knee_r + v_knee_ank_r
    p_hip_mid = 0.5 * (p_hip_l + p_hip_r)

    # 2. 무릎 및 고관절 각도 시계열 산출 (64,)
    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)
    ang_hip_l = calculate_angle_3d(v_pelvis_torso, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(v_pelvis_torso, p_hip_r, p_knee_r)

    # 무릎 각도 가동범위
    rom_knee_l = float(np.max(ang_knee_l) - np.min(ang_knee_l))
    rom_knee_r = float(np.max(ang_knee_r) - np.min(ang_knee_r))
    f1_knee_rom_mean = 0.5 * (rom_knee_l + rom_knee_r)
    f2_knee_rom_max = max(rom_knee_l, rom_knee_r)

    # 고관절 가동범위 및 힌지 격리 지수 (HHIR)
    rom_hip_l = float(np.max(ang_hip_l) - np.min(ang_hip_l))
    rom_hip_r = float(np.max(ang_hip_r) - np.min(ang_hip_r))
    rom_hip_mean = 0.5 * (rom_hip_l + rom_hip_r)
    f3_hhir = float(rom_hip_mean / (f1_knee_rom_mean + 1e-4))

    # 무릎 각속도 잔차 RMS 에너지
    vel_knee_l = np.diff(ang_knee_l, prepend=ang_knee_l[0])
    vel_knee_r = np.diff(ang_knee_r, prepend=ang_knee_r[0])
    rms_vel_l = float(np.sqrt(np.mean(vel_knee_l ** 2)))
    rms_vel_r = float(np.sqrt(np.mean(vel_knee_r ** 2)))
    f4_vel_rms_mean = 0.5 * (rms_vel_l + rms_vel_r)
    f5_vel_rms_max = max(rms_vel_l, rms_vel_r)

    # 정강이 수직 불변성 지수 (SVII = std of shank angle to vertical)
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    shank_l = p_ank_l - p_knee_l
    shank_r = p_ank_r - p_knee_r
    norm_sl = np.linalg.norm(shank_l, axis=-1) + 1e-7
    norm_sr = np.linalg.norm(shank_r, axis=-1) + 1e-7
    cos_sl = np.clip(np.sum(shank_l * v_vert, axis=-1) / norm_sl, -1.0, 1.0)
    cos_sr = np.clip(np.sum(shank_r * v_vert, axis=-1) / norm_sr, -1.0, 1.0)
    ang_sl = np.degrees(np.arccos(cos_sl))
    ang_sr = np.degrees(np.arccos(cos_sr))
    svii_l = float(np.std(ang_sl))
    svii_r = float(np.std(ang_sr))
    f6_svii_mean = 0.5 * (svii_l + svii_r)
    f7_svii_max = max(svii_l, svii_r)

    # 최저점(Hip Hinge 최대 지점)에서의 무릎 각도 편차
    idx_bot = int(np.argmin(0.5 * (ang_hip_l + ang_hip_r)))
    bot_dev_l = abs(ang_knee_l[idx_bot] - ang_knee_l[0])
    bot_dev_r = abs(ang_knee_r[idx_bot] - ang_knee_r[0])
    f8_bot_delta_mean = 0.5 * (bot_dev_l + bot_dev_r)
    f9_bot_angle = 0.5 * (ang_knee_l[idx_bot] + ang_knee_r[idx_bot])

    # 대퇴골 경사각 가동범위
    femur_l = p_knee_l - p_hip_l
    femur_r = p_knee_r - p_hip_r
    norm_fl = np.linalg.norm(femur_l, axis=-1) + 1e-7
    norm_fr = np.linalg.norm(femur_r, axis=-1) + 1e-7
    cos_fl = np.clip(np.sum(femur_l * v_vert, axis=-1) / norm_fl, -1.0, 1.0)
    cos_fr = np.clip(np.sum(femur_r * v_vert, axis=-1) / norm_fr, -1.0, 1.0)
    ang_fl = np.degrees(np.arccos(cos_fl))
    ang_fr = np.degrees(np.arccos(cos_fr))
    f10_femur_rom = 0.5 * ((np.max(ang_fl) - np.min(ang_fl)) + (np.max(ang_fr) - np.min(ang_fr)))

    # 고관절-무릎 각속도 피어슨 상관계수
    vel_hip_mean = 0.5 * (np.diff(ang_hip_l, prepend=ang_hip_l[0]) + np.diff(ang_hip_r, prepend=ang_hip_r[0]))
    vel_knee_mean = 0.5 * (vel_knee_l + vel_knee_r)
    if np.std(vel_hip_mean) > 1e-4 and np.std(vel_knee_mean) > 1e-4:
        r_corr = float(np.corrcoef(vel_hip_mean, vel_knee_mean)[0, 1])
    else:
        r_corr = 0.0
    f11_hip_knee_corr = r_corr

    # 무릎 전후(Z축) 공간 변위 RMS (뒤로 밀림/과신전 포착)
    p_knee_mid_z = 0.5 * (p_knee_l[:, 2] + p_knee_r[:, 2])
    f12_knee_disp_z = float(np.std(p_knee_mid_z))

    # 무릎 수직(Y축) 이동 가동범위 (스쿼트화 포착)
    p_knee_mid_y = 0.5 * (p_knee_l[:, 1] + p_knee_r[:, 1])
    f13_knee_disp_y_rom = float(np.max(p_knee_mid_y) - np.min(p_knee_mid_y))

    # 골반-무릎 상대 거리 변동폭
    dist_pelvis_knee = np.linalg.norm(0.5 * (p_knee_l + p_knee_r) - p_hip_mid, axis=-1)
    f14_pelvis_knee_rom = float(np.max(dist_pelvis_knee) - np.min(dist_pelvis_knee))

    feat_14 = np.array([
        f1_knee_rom_mean,
        f2_knee_rom_max,
        f3_hhir,
        f4_vel_rms_mean,
        f5_vel_rms_max,
        f6_svii_mean,
        f7_svii_max,
        f8_bot_delta_mean,
        f9_bot_angle,
        f10_femur_rom,
        f11_hip_knee_corr,
        f12_knee_disp_z,
        f13_knee_disp_y_rom,
        f14_pelvis_knee_rom
    ], dtype=np.float32)

    return feat_14

def extract_knee_temporal_12ch(x_seq: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스(64, 98)에서 TCN/Bi-GRU 모델용 12채널 시계열 궤적 추출
    반환 형태: (64, 12)
    채널 구성:
      - 0~1: 좌/우 무릎 각도
      - 2~3: 좌/우 고관절 각도
      - 4~5: 좌/우 정강이 지면 각도
      - 6~7: 좌/우 대퇴골 지면 각도
      - 8~9: 좌/우 무릎 Z축 변위
      - 10~11: 좌/우 무릎 Y축 변위
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

    femur_l = p_knee_l - p_hip_l
    femur_r = p_knee_r - p_hip_r
    norm_fl = np.linalg.norm(femur_l, axis=-1) + 1e-7
    norm_fr = np.linalg.norm(femur_r, axis=-1) + 1e-7
    cos_fl = np.clip(np.sum(femur_l * v_vert, axis=-1) / norm_fl, -1.0, 1.0)
    cos_fr = np.clip(np.sum(femur_r * v_vert, axis=-1) / norm_fr, -1.0, 1.0)
    ang_fl = np.arccos(cos_fl) / np.pi
    ang_fr = np.arccos(cos_fr) / np.pi

    disp_kz_l = p_knee_l[:, 2] - p_knee_l[0, 2]
    disp_kz_r = p_knee_r[:, 2] - p_knee_r[0, 2]
    disp_ky_l = p_knee_l[:, 1] - p_knee_l[0, 1]
    disp_ky_r = p_knee_r[:, 1] - p_knee_r[0, 1]

    t12 = np.stack([
        ang_knee_l, ang_knee_r,
        ang_hip_l, ang_hip_r,
        ang_sl, ang_sr,
        ang_fl, ang_fr,
        disp_kz_l, disp_kz_r,
        disp_ky_l, disp_ky_r
    ], axis=-1).astype(np.float32)

    return t12

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Version: v2.0-RTM (sd1_rtmp_2)
Module: sd1_rtmp_2_features.py
Purpose: RTMPose-m (Halpe26) 전용 고도화 생체역학 피처 추출기 (v2.0-RTM)
         - 47D 심층 생체역학 피처셋: 근위측(Near-Leg) 추적, 정강이 수직성, 고관절 힌지비율, 발목 배굴각
         - 2D 평면 기하학 무결성 확보 (가짜 Z축 배제, 엄격한 유클리드 투영)
         - Halpe26 및 MediaPipe33 입력 동시 호환
         - 5대 공식 시점(시점 A~E / view1~view5) 직교 투영 원-핫 인코딩
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 공식 5대 시점 매핑 (Official 5-View Taxonomy)
VIEW_INDEX_MAP: Dict[str, int] = {
    'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4,
    '시점 E': 0, '시점 B': 1, '시점 C': 2, '시점 D': 3, '시점 A': 4,
    '시점 E: 좌측 후방 (45도)': 0,
    '시점 B: 우측 전방 (45도)': 1,
    '시점 C: 정면': 2,
    '시점 D: 좌측 전방 (45도)': 3,
    '시점 A: 우측 후방 (45도)': 4
}

FEATURE_NAMES_47D_SD1_RTMP_2: List[str] = [
    'knee_angle_bottom_deg',            # [ 0] 최저점(고관절 최대굴곡) 무릎 2D 내각
    'knee_rom_deg',                     # [ 1] 양 무릎 평균 2D 각도 변화폭 (Max - Min)
    'knee_delta_address_bottom',        # [ 2] 어드레스 대비 최저점 무릎 각도 변화량
    'knee_hip_rom_ratio',               # [ 3] 고관절 대비 무릎 ROM 비율 (ROM_knee / ROM_hip)
    'hip_knee_velocity_corr',           # [ 4] 고관절-무릎 각속도 피어슨 상관계수 (동위상 결합도)
    'mcrp',                             # [ 5] Mean Continuous Relative Phase (평균 연속 상대 위상)
    'vcrp',                             # [ 6] CRP 가변성 (Continuous Relative Phase Variability)
    'knee_angular_accel_max',           # [ 7] 무릎 최대 각가속도 (주저앉는 스파이크)
    'knee_accel_sign_changes',          # [ 8] 무릎 각가속도 부호 반전 횟수 (미세 떨림 지터)
    'excess_flexion_integral',          # [ 9] 어드레스 대비 5도 이상 단방향 추가 굴곡 적분량
    'excess_flexion_max',               # [10] 어드레스 대비 최대 추가 굴곡량
    'kadr_max',                         # [11] 정강이 길이 대비 발목-무릎 수평 변위비 최대값 (KADR Max)
    'kadr_mean',                        # [12] 정강이 길이 대비 발목-무릎 수평 변위비 평균값 (KADR Mean)
    'kadr_rom',                         # [13] 정강이 길이 대비 발목-무릎 수평 변위비 가동범위 (KADR ROM)
    'shank_svii',                       # [14] 정강이 수직 불변성 지수 (지면 수직 대비 정강이 경사각 ROM)
    'femur_rom',                        # [15] 대퇴골 지면 경사각 ROM
    'norm_knee_disp_y',                 # [16] 정강이 길이 대비 무릎 수직(Y축) 하강 변위 ROM (주저앉음)
    'norm_knee_disp_x',                 # [17] 대퇴골 길이 대비 무릎 수평(X축) 변위 ROM
    'pli',                              # [18] 고관절-무릎 각속도 위상 분리 지수 (Phase Lag Index)
    'hyperextension_ratio',             # [19] 무릎 각도가 175도를 초과한 프레임 비율 (과신전)
    'coronal_knee_wobble',              # [20] 관상면(좌우 X축) 무릎 흔들림 정규화 표준편차
    'sagittal_proj_knee_rom',           # [21] 주 관측 시상면 투영 무릎 각도 ROM
    'sagittal_proj_knee_bottom',        # [22] 주 관측 시상면 투영 최저점 무릎 각도
    'bilateral_knee_asym',              # [23] 좌우 무릎 각도 비대칭도 평균
    'near_knee_rom',                    # [24] 카메라 근위측(Near-Leg) 무릎 ROM
    'near_knee_bottom_angle',           # [25] 카메라 근위측 최저점 무릎 각도
    'nadir_knee_drop_window',           # [26] 최저점 전후 윈도우(±4F) 무릎 굴곡 낙폭
    'shank_pitch_nadir',                # [27] 최저점 정강이 전경 경사각
    'shank_pitch_rom',                  # [28] 정강이 전경 경사각 가동범위
    'pelvis_drop_to_shift_ratio',       # [29] 골반 수직 하강폭 대비 전후 후퇴폭 비율
    'knee_past_toe_encroachment_max',   # [30] 발가락 돌출 무릎 전진 지수 최대값
    'near_knee_hip_coupling',           # [31] 근위측 무릎 ROM 대비 고관절 ROM 결합 비율
    'barbell_knee_clearance_nadir',     # [32] 최저점 바벨-무릎 수직 이격거리
    'kawr_nadir',                       # [33] 최저점 관상면 무릎-발목 폭 비율 (KAWR)
    'kawr_rom',                         # [34] 관상면 무릎-발목 폭 비율 가동범위
    'eccentric_knee_vel_peak',          # [35] 하강(Eccentric) 구간 최대 무릎 각속도
    'ankle_dorsiflexion_bottom',        # [36] 최저점 발목 배굴(Dorsiflexion) 각도
    'ankle_dorsiflexion_rom',           # [37] 발목 배굴 각도 가동범위
    'barbell_to_pelvis_drop_ratio',     # [38] 골반 하강폭 대비 바벨 하강폭 비율
    'torso_pitch_bottom',               # [39] 최저점 체간 전경 경사각
    'knee_hip_horizontal_rom',          # [40] 대퇴골 길이 대비 무릎-고관절 수평 변위폭
    'view_code',                        # [41] 시점 코드 (0~4)
    'v_oh_1',                           # [42] 시점 원-핫 1 (view1 / 시점 E: 좌측 후방 45°)
    'v_oh_2',                           # [43] 시점 원-핫 2 (view2 / 시점 B: 우측 전방 45°)
    'v_oh_3',                           # [44] 시점 원-핫 3 (view3 / 시점 C: 정면 0°)
    'v_oh_4',                           # [45] 시점 원-핫 4 (view4 / 시점 D: 좌측 전방 45°)
    'v_oh_5'                            # [46] 시점 원-핫 5 (view5 / 시점 A: 우측 후방 45°)
]


def calc_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """세 점 a, b(중심점), c의 2D 평면 내각(도) 계산 (벡터 클립 및 수치 안정화)"""
    ba = a[:, :2] - b[:, :2]
    bc = c[:, :2] - b[:, :2]
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def convert_mp33_to_halpe26_format(kps_33: np.ndarray) -> np.ndarray:
    """MediaPipe 33 키포인트를 Halpe26 호환 (64, 26, 3) 배열로 변환"""
    T = kps_33.shape[0]
    out = np.zeros((T, 26, 3), dtype=np.float32)
    # Head/Nose
    out[:, 0] = kps_33[:, 0]
    out[:, 17] = kps_33[:, 0]
    # Shoulders
    out[:, 5] = kps_33[:, 11]
    out[:, 6] = kps_33[:, 12]
    # Elbows
    out[:, 7] = kps_33[:, 13]
    out[:, 8] = kps_33[:, 14]
    # Wrists
    out[:, 9] = kps_33[:, 15]
    out[:, 10] = kps_33[:, 16]
    # Hips
    out[:, 11] = kps_33[:, 23]
    out[:, 12] = kps_33[:, 24]
    # Knees
    out[:, 13] = kps_33[:, 25]
    out[:, 14] = kps_33[:, 26]
    # Ankles
    out[:, 15] = kps_33[:, 27]
    out[:, 16] = kps_33[:, 28]
    # Pelvis midpoint
    out[:, 19] = 0.5 * (kps_33[:, 23] + kps_33[:, 24])
    # Toes
    out[:, 20] = kps_33[:, 31]
    out[:, 21] = kps_33[:, 32]
    # Heels
    out[:, 24] = kps_33[:, 29]
    out[:, 25] = kps_33[:, 30]
    return out


def extract_sd1_rtmp_2_features(kps: np.ndarray, view_name: str = 'view3') -> np.ndarray:
    """
    바벨 스티프 데드리프트 sd1 결함 판정용 47D 생체역학 피처 추출
    
    Args:
        kps: (64, 26, 3) Halpe26 또는 (64, 33, 3) MediaPipe33 시퀀스 배열
        view_name: 카메라 시점 명칭 ('view1'~'view5' 또는 '시점 A'~'시점 E')
        
    Returns:
        np.ndarray: shape (47,) float32 고유 피처 벡터
    """
    if kps.shape[1] == 33:
        arr = convert_mp33_to_halpe26_format(kps)
    elif kps.shape[1] >= 26:
        arr = kps[:, :26, :].copy()
    else:
        raise ValueError(f"지원하지 않는 키포인트 형식입니다. Shape: {kps.shape}")

    arr = np.nan_to_num(arr, nan=0.0)

    # 주요 관절점 추출 (X, Y 2D 좌표 사용)
    p_sh_l, p_sh_r = arr[:, 5, :2], arr[:, 6, :2]
    p_wr_l, p_wr_r = arr[:, 9, :2], arr[:, 10, :2]
    p_hip_l, p_hip_r = arr[:, 11, :2], arr[:, 12, :2]
    p_knee_l, p_knee_r = arr[:, 13, :2], arr[:, 14, :2]
    p_ank_l, p_ank_r = arr[:, 15, :2], arr[:, 16, :2]
    p_pelvis = arr[:, 19, :2]
    p_toe_l, p_toe_r = arr[:, 20, :2], arr[:, 21, :2]
    
    p_sh_mid = 0.5 * (p_sh_l + p_sh_r)
    p_knee_mid = 0.5 * (p_knee_l + p_knee_r)
    p_ank_mid = 0.5 * (p_ank_l + p_ank_r)
    p_wr_mid = 0.5 * (p_wr_l + p_wr_r)

    # 1. 2D 관절 각도 계산
    ang_k_l = calc_angle_2d(p_hip_l, p_knee_l, p_ank_l)
    ang_k_r = calc_angle_2d(p_hip_r, p_knee_r, p_ank_r)
    ang_k = 0.5 * (ang_k_l + ang_k_r)

    ang_h_l = calc_angle_2d(p_sh_l, p_hip_l, p_knee_l)
    ang_h_r = calc_angle_2d(p_sh_r, p_hip_r, p_knee_r)
    ang_h = 0.5 * (ang_h_l + ang_h_r)

    # 시점별 근위측(Near-Leg) 및 원위측(Far-Leg) 매핑
    v_clean = str(view_name).strip()
    if v_clean in ['view2', 'view5', '시점 B', '시점 A', '시점 B: 우측 전방 (45도)', '시점 A: 우측 후방 (45도)']:
        near_ang_k = ang_k_r
        near_knee = p_knee_r
        near_ank = p_ank_r
        near_hip = p_hip_r
        near_toe = p_toe_r
    elif v_clean in ['view1', 'view4', '시점 E', '시점 D', '시점 E: 좌측 후방 (45도)', '시점 D: 좌측 전방 (45도)']:
        near_ang_k = ang_k_l
        near_knee = p_knee_l
        near_ank = p_ank_l
        near_hip = p_hip_l
        near_toe = p_toe_l
    else:
        near_ang_k = ang_k
        near_knee = p_knee_mid
        near_ank = p_ank_mid
        near_hip = p_pelvis
        near_toe = 0.5 * (p_toe_l + p_toe_r)

    # 최저점(고관절 최대굴곡 시점) 식별
    idx_bot = int(np.argmin(ang_h))
    w_start = max(0, idx_bot - 4)
    w_end = min(64, idx_bot + 5)

    # 신체 분절 길이 계산 (정규화 스케일러)
    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    len_shank = 0.5 * (np.linalg.norm(v_shank_l, axis=-1).mean() + np.linalg.norm(v_shank_r, axis=-1).mean()) + 1e-5

    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    len_femur = 0.5 * (np.linalg.norm(v_femur_l, axis=-1).mean() + np.linalg.norm(v_femur_r, axis=-1).mean()) + 1e-5

    v_torso = p_sh_mid - p_pelvis
    len_torso = np.linalg.norm(v_torso, axis=-1).mean() + 1e-5

    # 1. 각도 및 가동범위 (ROM)
    f0_bottom_angle = float(ang_k[idx_bot])
    f1_knee_rom = float(np.ptp(ang_k))
    init_knee = float(np.mean(ang_k[:6]))
    f2_delta_address_bot = float(abs(ang_k[idx_bot] - init_knee))
    hip_rom = float(np.ptp(ang_h))
    f3_rom_ratio = float(np.clip(f1_knee_rom / (hip_rom + 1e-5), 0.0, 5.0))

    # 2. 각속도 및 위상 역학
    h_smooth = gaussian_filter1d(ang_h, sigma=1.5)
    k_smooth = gaussian_filter1d(ang_k, sigma=1.5)
    omega_h = np.gradient(h_smooth)
    omega_k = np.gradient(k_smooth)
    vel_corr = float(np.corrcoef(omega_h, omega_k)[0, 1]) if np.std(omega_h) > 1e-6 and np.std(omega_k) > 1e-6 else 0.0
    f4_vel_corr = float(np.nan_to_num(vel_corr))

    theta_h = (h_smooth - np.min(h_smooth)) / (np.ptp(h_smooth) + 1e-7) * 2.0 - 1.0
    theta_k = (k_smooth - np.min(k_smooth)) / (np.ptp(k_smooth) + 1e-7) * 2.0 - 1.0
    phi_h = np.arctan2(omega_h / (np.max(np.abs(omega_h)) + 1e-7), theta_h)
    phi_k = np.arctan2(omega_k / (np.max(np.abs(omega_k)) + 1e-7), theta_k)
    crp_deg = np.degrees(np.abs(phi_h - phi_k))
    crp_deg = np.where(crp_deg > 180.0, 360.0 - crp_deg, crp_deg)
    f5_mcrp = float(np.mean(crp_deg))
    f6_vcrp = float(np.std(crp_deg))

    alpha_k = np.gradient(omega_k)
    f7_accel_max = float(np.max(np.abs(alpha_k)))
    f8_sign_changes = float(np.sum(np.diff(np.sign(alpha_k)) != 0))

    # 3. 초과 굴곡 지표
    excess_flex = np.maximum(0.0, init_knee - ang_k - 5.0)
    f9_excess_integral = float(np.mean(excess_flex))
    f10_excess_max = float(np.max(excess_flex))

    # 4. KADR 및 변위
    rel_knee_x = (p_knee_mid[:, 0] - p_ank_mid[:, 0]) / len_shank
    f11_kadr_max = float(np.max(np.abs(rel_knee_x)))
    f12_kadr_mean = float(np.mean(np.abs(rel_knee_x)))
    f13_kadr_rom = float(np.ptp(rel_knee_x))

    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 0]), np.abs(v_shank_l[:, 1]) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 0]), np.abs(v_shank_r[:, 1]) + 1e-7))
    f14_shank_svii = float(0.5 * (np.ptp(pitch_shank_l) + np.ptp(pitch_shank_r)))

    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 0]), np.abs(v_femur_l[:, 1]) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 0]), np.abs(v_femur_r[:, 1]) + 1e-7))
    f15_femur_rom = float(0.5 * (np.ptp(pitch_femur_l) + np.ptp(pitch_femur_r)))

    f16_norm_disp_y = float(np.ptp(p_knee_mid[:, 1]) / len_shank)
    f17_norm_disp_x = float(np.ptp(p_knee_mid[:, 0]) / len_femur)

    f18_pli = float(np.clip(1.0 - (np.sum(np.abs(np.diff(ang_k))) / (np.sum(np.abs(np.diff(ang_h))) + 1e-5)), -1.0, 1.0))
    f19_hyperext_ratio = float(np.mean(ang_k > 175.0))
    f20_coronal_wobble = float(np.std(p_knee_mid[:, 0]) / len_shank)
    f21_sag_rom = f1_knee_rom
    f22_sag_bottom = f0_bottom_angle
    f23_bilateral_asym = float(np.mean(np.abs(ang_k_l - ang_k_r)))

    # 5. 근위측(Near-Leg) 세부 지표
    f24_near_knee_rom = float(np.ptp(near_ang_k))
    f25_near_knee_bot = float(near_ang_k[idx_bot])
    f26_nadir_knee_drop = float(np.max(near_ang_k[w_start:w_end]) - np.min(near_ang_k[w_start:w_end]))

    near_shank_v = near_ank - near_knee
    near_pitch_shank = np.degrees(np.arctan2(np.abs(near_shank_v[:, 0]), np.abs(near_shank_v[:, 1]) + 1e-7))
    f27_shank_pitch_nadir = float(near_pitch_shank[idx_bot])
    f28_shank_pitch_rom = float(np.ptp(near_pitch_shank))

    pelvis_drop = np.ptp(p_pelvis[:, 1])
    pelvis_shift = np.ptp(p_pelvis[:, 0]) + 1e-5
    f29_pelvis_drop_ratio = float(np.clip(pelvis_drop / pelvis_shift, 0.0, 10.0))

    knee_toe_diff = (near_knee[:, 0] - near_toe[:, 0]) / len_shank
    f30_knee_past_toe_max = float(np.max(np.abs(knee_toe_diff)))
    f31_hinge_coupling = float(f24_near_knee_rom / (hip_rom + 1e-5))

    f32_barbell_knee_dist_nadir = float((p_wr_mid[idx_bot, 1] - p_knee_mid[idx_bot, 1]) / len_shank)

    knee_w = np.abs(p_knee_l[:, 0] - p_knee_r[:, 0])
    ank_w = np.abs(p_ank_l[:, 0] - p_ank_r[:, 0]) + 1e-5
    kawr = knee_w / ank_w
    f33_kawr_nadir = float(kawr[idx_bot])
    f34_kawr_rom = float(np.ptp(kawr))

    ecc_omega = omega_k[:idx_bot] if idx_bot > 5 else omega_k
    f35_ecc_knee_vel_peak = float(np.max(np.abs(ecc_omega))) if len(ecc_omega) > 0 else 0.0

    # 6. 추가 심층 생체역학 지표
    ang_ank_near = calc_angle_2d(near_knee, near_ank, near_toe)
    f36_ankle_dorsi_bot = float(ang_ank_near[idx_bot])
    f37_ankle_dorsi_rom = float(np.ptp(ang_ank_near))

    barbell_drop = np.ptp(p_wr_mid[:, 1])
    f38_bb_pelvis_ratio = float(barbell_drop / (pelvis_drop + 1e-5))

    torso_pitch = np.degrees(np.arctan2(np.abs(v_torso[:, 0]), np.abs(v_torso[:, 1]) + 1e-7))
    f39_torso_pitch_bot = float(torso_pitch[idx_bot])

    knee_hip_horiz = (near_knee[:, 0] - near_hip[:, 0]) / len_femur
    f40_knee_hip_horiz_rom = float(np.ptp(knee_hip_horiz))

    # 7. 시점 원-핫 및 인덱스 인코딩
    v_idx = VIEW_INDEX_MAP.get(v_clean, 2)
    f41_view_code = float(v_idx)
    v_oh = np.zeros(5, dtype=np.float32)
    v_oh[v_idx] = 1.0

    feat = np.array([
        f0_bottom_angle, f1_knee_rom, f2_delta_address_bot, f3_rom_ratio,
        f4_vel_corr, f5_mcrp, f6_vcrp, f7_accel_max, f8_sign_changes,
        f9_excess_integral, f10_excess_max, f11_kadr_max, f12_kadr_mean, f13_kadr_rom,
        f14_shank_svii, f15_femur_rom, f16_norm_disp_y, f17_norm_disp_x,
        f18_pli, f19_hyperext_ratio, f20_coronal_wobble, f21_sag_rom,
        f22_sag_bottom, f23_bilateral_asym,
        f24_near_knee_rom, f25_near_knee_bot, f26_nadir_knee_drop, f27_shank_pitch_nadir,
        f28_shank_pitch_rom, f29_pelvis_drop_ratio, f30_knee_past_toe_max, f31_hinge_coupling,
        f32_barbell_knee_dist_nadir, f33_kawr_nadir, f34_kawr_rom, f35_ecc_knee_vel_peak,
        f36_ankle_dorsi_bot, f37_ankle_dorsi_rom, f38_bb_pelvis_ratio, f39_torso_pitch_bot,
        f40_knee_hip_horiz_rom, f41_view_code,
        v_oh[0], v_oh[1], v_oh[2], v_oh[3], v_oh[4]
    ], dtype=np.float32)

    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == '__main__':
    dummy_halpe26 = np.zeros((64, 26, 3), dtype=np.float32)
    f_h = extract_sd1_rtmp_2_features(dummy_halpe26, 'view3')
    print(f"Halpe26 Feature Extraction Test: shape={f_h.shape}, length={len(FEATURE_NAMES_47D_SD1_RTMP_2)}")
    assert len(f_h) == 47, f"Feature dimension mismatch: expected 47, got {len(f_h)}"

    dummy_mp33 = np.zeros((64, 33, 3), dtype=np.float32)
    f_m = extract_sd1_rtmp_2_features(dummy_mp33, 'view2')
    print(f"MediaPipe33 Feature Extraction Test: shape={f_m.shape}")
    assert len(f_m) == 47, f"Feature dimension mismatch: expected 47, got {len(f_m)}"
    print("[+] All feature tests passed successfully!")

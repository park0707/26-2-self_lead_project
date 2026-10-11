# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Module: sd1_rtmp_features.py
Purpose: RTMPose-m (Halpe26) 전용 고정밀 2D 생체역학 피처 추출기
         - 기존 sd1_v8 피처셋의 생체역학 수학적 정의 100% 계승 (Copy-and-Refine)
         - RTMPose의 2D 고정밀 (X, Y) 좌표계에 맞춘 2D 평면 기하학 무결성 보정
         - 신뢰도(Confidence) 값을 3D Z(Depth)로 오인하는 왜곡 원천 차단
         - Halpe26 (26관절) 및 MediaPipe33 (33관절) 호환 입력 지원
         - 5대 공식 시점(시점 A~E / view1~view5) 정합 30D 마스터 피처셋
"""

import os
import sys
from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 5대 공식 시점 명칭 및 인덱스 매핑 (Official 5-View Taxonomy)
VIEW_INDEX_MAP: Dict[str, int] = {
    'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4,
    '시점 E': 0, '시점 B': 1, '시점 C': 2, '시점 D': 3, '시점 A': 4,
    '시점 E: 좌측 후방 (45도)': 0,
    '시점 B: 우측 전방 (45도)': 1,
    '시점 C: 정면': 2,
    '시점 D: 좌측 전방 (45도)': 3,
    '시점 A: 우측 후방 (45도)': 4
}

# 30D 마스터 피처 명칭 목록
FEATURE_NAMES_30D_SD1_RTMP: List[str] = [
    'knee_angle_bottom_deg',       # [ 0] 최저점(고관절 최대굴곡) 무릎 2D 내각
    'knee_rom_deg',                # [ 1] 무릎 2D 각도 변화폭 (Max - Min)
    'knee_delta_address_bottom',   # [ 2] 어드레스 대비 최저점 무릎 각도 변화량
    'knee_hip_rom_ratio',          # [ 3] 고관절 대비 무릎 ROM 비율 (ROM_knee / ROM_hip)
    'hip_knee_velocity_corr',      # [ 4] 고관절-무릎 각속도 피어슨 상관계수 (동위상 결합도)
    'mcrp',                        # [ 5] Mean Continuous Relative Phase (평균 연속 상대 위상)
    'vcrp',                        # [ 6] CRP 가변성 (Continuous Relative Phase Variability)
    'knee_angular_accel_max',      # [ 7] 무릎 최대 각가속도 (주저앉는 스파이크)
    'knee_accel_sign_changes',     # [ 8] 무릎 각가속도 부호 반전 횟수 (미세 떨림 지터)
    'excess_flexion_integral',     # [ 9] 어드레스 대비 5도 이상 단방향 추가 굴곡 적분량
    'excess_flexion_max',          # [10] 어드레스 대비 최대 추가 굴곡량
    'kadr_max',                    # [11] 정강이 길이 대비 발목-무릎 수평 변위비 최대값 (KADR Max)
    'kadr_mean',                   # [12] 정강이 길이 대비 발목-무릎 수평 변위비 평균값 (KADR Mean)
    'kadr_rom',                    # [13] 정강이 길이 대비 발목-무릎 수평 변위비 가동범위 (KADR ROM)
    'shank_svii',                  # [14] 정강이 수직 불변성 지수 (지면 수직 대비 정강이 경사각 ROM)
    'femur_rom',                   # [15] 대퇴골 지면 경사각 ROM
    'norm_knee_disp_y',            # [16] 정강이 길이 대비 무릎 수직(Y축) 하강 변위 ROM (주저앉음)
    'norm_knee_disp_x',            # [17] 대퇴골 길이 대비 무릎 수평(X축) 변위 ROM
    'pli',                         # [18] 고관절-무릎 각속도 위상 분리 지수 (Phase Lag Index)
    'hyperextension_ratio',        # [19] 무릎 각도가 175도를 초과한 프레임 비율 (과신전)
    'coronal_knee_wobble',         # [20] 관상면(좌우 X축) 무릎 흔들림 표준편차
    'sagittal_proj_knee_rom',      # [21] 주 관측 시상면 투영 무릎 각도 ROM
    'sagittal_proj_knee_bottom',   # [22] 주 관측 시상면 투영 최저점 무릎 각도
    'bilateral_knee_asym',         # [23] 좌우 무릎 각도 비대칭도 평균
    'view_code',                   # [24] 시점 코드 (0~4)
    'v_oh_1',                      # [25] 시점 원-핫 1 (view1 / 시점 E)
    'v_oh_2',                      # [26] 시점 원-핫 2 (view2 / 시점 B)
    'v_oh_3',                      # [27] 시점 원-핫 3 (view3 / 시점 C)
    'v_oh_4',                      # [28] 시점 원-핫 4 (view4 / 시점 D)
    'v_oh_5'                       # [29] 시점 원-핫 5 (view5 / 시점 A)
]


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    2D 평면 상에서 벡터 ba와 bc 사이의 사잇각(0~180도) 산출
    입력 형상: (T, 2)
    """
    ba = a[:, :2] - b[:, :2]
    bc = c[:, :2] - b[:, :2]
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_continuous_relative_phase(ang_hip: np.ndarray, ang_knee: np.ndarray) -> Tuple[float, float, float]:
    """
    고관절-무릎 연속 상대 위상 (CRP) 및 각속도 피어슨 상관계수 계산
    ang_hip, ang_knee: (64,)
    반환: (MCRP, VCRP, vel_corr)
    """
    # 중심차분 각속도 계산 (가우시안 스무딩)
    h_smooth = gaussian_filter1d(ang_hip, sigma=1.5)
    k_smooth = gaussian_filter1d(ang_knee, sigma=1.5)
    
    omega_h = np.gradient(h_smooth)
    omega_k = np.gradient(k_smooth)
    
    # 1. 속도 피어슨 상관계수 (무릎 결함 시 > 0.6 동위상 결합)
    std_h = np.std(omega_h) + 1e-7
    std_k = np.std(omega_k) + 1e-7
    vel_corr = float(np.corrcoef(omega_h, omega_k)[0, 1])
    if np.isnan(vel_corr):
        vel_corr = 0.0

    # 2. 위상각 정규화 [-1, 1]
    rom_h = (np.max(h_smooth) - np.min(h_smooth)) + 1e-7
    rom_k = (np.max(k_smooth) - np.min(k_smooth)) + 1e-7
    
    theta_h = (h_smooth - np.min(h_smooth)) / rom_h * 2.0 - 1.0
    theta_k = (k_smooth - np.min(k_smooth)) / rom_k * 2.0 - 1.0
    
    omega_h_norm = omega_h / (np.max(np.abs(omega_h)) + 1e-7)
    omega_k_norm = omega_k / (np.max(np.abs(omega_k)) + 1e-7)
    
    phi_h = np.arctan2(omega_h_norm, theta_h)
    phi_k = np.arctan2(omega_k_norm, theta_k)
    
    # 3. CRP 시계열 (|phi_h - phi_k|)
    crp_deg = np.degrees(np.abs(phi_h - phi_k))
    crp_deg = np.where(crp_deg > 180.0, 360.0 - crp_deg, crp_deg)
    
    mcrp = float(np.mean(crp_deg))
    vcrp = float(np.std(crp_deg))
    
    return mcrp, vcrp, vel_corr


def extract_sd1_rtmp_features(kps: np.ndarray, view_name: str = "view4") -> np.ndarray:
    """
    (64, 26, 3) Halpe26 또는 (64, 33, 3) MediaPipe 호환 키포인트에서
    RTMPose 최적화 30D 생체역학 피처 추출
    """
    arr = np.array(kps, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 키포인트 인덱스 매핑 (Halpe26 vs MediaPipe33 자동 판별)
    if arr.shape[1] == 26:
        # Halpe 26 Topology
        p_sh_l = arr[:, 5, :2]
        p_sh_r = arr[:, 6, :2]
        p_hip_l = arr[:, 11, :2]
        p_hip_r = arr[:, 12, :2]
        p_knee_l = arr[:, 13, :2]
        p_knee_r = arr[:, 14, :2]
        p_ank_l = arr[:, 15, :2]
        p_ank_r = arr[:, 16, :2]
    else:
        # MediaPipe 33 Topology
        p_sh_l = arr[:, 11, :2]
        p_sh_r = arr[:, 12, :2]
        p_hip_l = arr[:, 23, :2]
        p_hip_r = arr[:, 24, :2]
        p_knee_l = arr[:, 25, :2]
        p_knee_r = arr[:, 26, :2]
        p_ank_l = arr[:, 27, :2]
        p_ank_r = arr[:, 28, :2]

    # 1. 2D 무릎 각도 시계열 (Hip - Knee - Ankle)
    ang_knee_l = calculate_angle_2d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_2d(p_hip_r, p_knee_r, p_ank_r)
    ang_knee = 0.5 * (ang_knee_l + ang_knee_r)  # (64,)

    # 2. 2D 고관절 각도 시계열 (Shoulder - Hip - Knee)
    ang_hip_l = calculate_angle_2d(p_sh_l, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_2d(p_sh_r, p_hip_r, p_knee_r)
    ang_hip = 0.5 * (ang_hip_l + ang_hip_r)  # (64,)

    # 3. 최저점(고관절 최대 굴곡 = 최소 각도 프레임)
    idx_bot = int(np.argmin(ang_hip))

    # [0] 최저점 무릎 2D 내각
    f0_bottom_angle = float(ang_knee[idx_bot])

    # [1] 무릎 2D 각도 변화폭 (ROM)
    f1_knee_rom = float(np.max(ang_knee) - np.min(ang_knee))

    # [2] 어드레스 대비 최저점 무릎 각도 변화량
    init_knee = float(np.mean(ang_knee[:6]))
    f2_delta_address_bot = float(abs(ang_knee[idx_bot] - init_knee))

    # [3] 고관절 대비 무릎 ROM 비율
    hip_rom = float(np.max(ang_hip) - np.min(ang_hip))
    f3_rom_ratio = float(np.clip(f1_knee_rom / (hip_rom + 1e-5), 0.0, 5.0))

    # [4~6] 고관절-무릎 CRP 및 각속도 상관계수
    mcrp, vcrp, vel_corr = compute_continuous_relative_phase(ang_hip, ang_knee)
    f4_vel_corr = float(vel_corr)
    f5_mcrp = float(mcrp)
    f6_vcrp = float(vcrp)

    # [7~8] 무릎 각가속도 및 부호 반전 지터
    k_smooth = gaussian_filter1d(ang_knee, sigma=1.5)
    omega_k = np.gradient(k_smooth)
    alpha_k = np.gradient(omega_k)
    f7_accel_max = float(np.max(np.abs(alpha_k)))
    sign_changes = np.sum(np.diff(np.sign(alpha_k)) != 0)
    f8_sign_changes = float(sign_changes)

    # [9~10] 단방향 초과 굴곡 적분량 & 최대치 (어드레스 대비 5도 이상 굴곡)
    excess_flex = np.maximum(0.0, init_knee - ang_knee - 5.0)
    f9_excess_integral = float(np.mean(excess_flex))
    f10_excess_max = float(np.max(excess_flex))

    # [11~13] 2D KADR (Knee-Ankle Displacement Ratio: 발목 대비 무릎 수평 상대 변위비)
    p_knee_mid = 0.5 * (p_knee_l + p_knee_r)
    p_ank_mid = 0.5 * (p_ank_l + p_ank_r)
    p_hip_mid = 0.5 * (p_hip_l + p_hip_r)

    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    len_shank = 0.5 * (np.linalg.norm(v_shank_l, axis=-1).mean() + np.linalg.norm(v_shank_r, axis=-1).mean()) + 1e-5

    # 무릎의 발목 대비 수평 전후 변위 (정규화)
    rel_knee_x = (p_knee_mid[:, 0] - p_ank_mid[:, 0]) / len_shank
    f11_kadr_max = float(np.max(np.abs(rel_knee_x)))
    f12_kadr_mean = float(np.mean(np.abs(rel_knee_x)))
    f13_kadr_rom = float(np.max(rel_knee_x) - np.min(rel_knee_x))

    # [14] 정강이 수직 불변성 지수 (SVII: 지면 수직 Y축 대비 정강이 경사각 ROM)
    # 수직선 대비 각도: arctan2(|dx|, |dy|)
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 0]), np.abs(v_shank_l[:, 1]) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 0]), np.abs(v_shank_r[:, 1]) + 1e-7))
    f14_shank_svii = float(0.5 * ((np.max(pitch_shank_l) - np.min(pitch_shank_l)) + (np.max(pitch_shank_r) - np.min(pitch_shank_r))))

    # [15] 대퇴골 지면 경사각 ROM
    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    len_femur = 0.5 * (np.mean(np.linalg.norm(v_femur_l, axis=-1)) + np.mean(np.linalg.norm(v_femur_r, axis=-1))) + 1e-5
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 0]), np.abs(v_femur_l[:, 1]) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 0]), np.abs(v_femur_r[:, 1]) + 1e-7))
    f15_femur_rom = float(0.5 * ((np.max(pitch_femur_l) - np.min(pitch_femur_l)) + (np.max(pitch_femur_r) - np.min(pitch_femur_r))))

    # [16] 정강이 길이 대비 무릎 수직(Y축) 변위 ROM (주저앉음 정도)
    disp_y = p_knee_mid[:, 1]
    f16_norm_disp_y = float((np.max(disp_y) - np.min(disp_y)) / (len_shank + 1e-5))

    # [17] 대퇴골 길이 대비 무릎 수평(X축) 변위 ROM
    disp_x = p_knee_mid[:, 0]
    f17_norm_disp_x = float((np.max(disp_x) - np.min(disp_x)) / (len_femur + 1e-5))

    # [18] PLI (Phase Lag Index / 위상 분리 지수)
    sum_omega_k = np.sum(np.abs(np.diff(ang_knee)))
    sum_omega_h = np.sum(np.abs(np.diff(ang_hip)))
    f18_pli = float(np.clip(1.0 - (sum_omega_k / (sum_omega_h + 1e-5)), -1.0, 1.0))

    # [19] 과신전 프레임 비율 (무릎 내각 > 175도)
    f19_hyperext_ratio = float(np.mean(ang_knee > 175.0))

    # [20] 관상면(좌우 X축) 무릎 흔들림 표준편차
    f20_coronal_wobble = float(np.std(p_knee_mid[:, 0]))

    # [21~22] 시상면 투영 무릎 각도 (주요 관측 측면에서의 2D 사잇각)
    f21_sag_rom = f1_knee_rom
    f22_sag_bottom = f0_bottom_angle

    # [23] 좌우 무릎 각도 비대칭도 평균
    f23_bilateral_asym = float(np.mean(np.abs(ang_knee_l - ang_knee_r)))

    # [24~29] 5대 공식 시점 원-핫 인코딩
    v_idx = VIEW_INDEX_MAP.get(view_name, 3)  # default: view4 (시점 D)
    f24_view_code = float(v_idx)
    v_oh = np.zeros(5, dtype=np.float32)
    v_oh[v_idx] = 1.0

    feat_30d = np.array([
        f0_bottom_angle, f1_knee_rom, f2_delta_address_bot, f3_rom_ratio,
        f4_vel_corr, f5_mcrp, f6_vcrp, f7_accel_max, f8_sign_changes,
        f9_excess_integral, f10_excess_max, f11_kadr_max, f12_kadr_mean, f13_kadr_rom,
        f14_shank_svii, f15_femur_rom, f16_norm_disp_y, f17_norm_disp_x,
        f18_pli, f19_hyperext_ratio, f20_coronal_wobble, f21_sag_rom,
        f22_sag_bottom, f23_bilateral_asym, f24_view_code,
        v_oh[0], v_oh[1], v_oh[2], v_oh[3], v_oh[4]
    ], dtype=np.float32)

    return np.nan_to_num(feat_30d, nan=0.0, posinf=0.0, neginf=0.0)


def extract_sd1_temporal_rtmp(kps: np.ndarray) -> np.ndarray:
    """
    Dilated-TCN 시계열 딥러닝 스트림용 12관절 2D 정규화 좌표 텐서 추출 (64, 24)
    골반 중심(Pelvis) 상대화 및 체간 크기 정규화 (Scale Invariant)
    """
    arr = np.array(kps, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape[1] == 26:
        # Halpe 26: 5(LSH), 6(RSH), 11(LHIP), 12(RHIP), 13(LKNE), 14(RKNE), 15(LANK), 16(RANK), 20(LTOE), 21(RTOE), 24(LHEEL), 25(RHEEL)
        joints_12 = [5, 6, 11, 12, 13, 14, 15, 16, 20, 21, 24, 25]
        root = arr[:, 19, :2]  # Hip 중심
    else:
        # MP 33: 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        joints_12 = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        root = 0.5 * (arr[:, 23, :2] + arr[:, 24, :2])

    xy_12 = arr[:, joints_12, :2]  # (64, 12, 2)
    # 골반 중심 상대화
    xy_rel = xy_12 - root[:, None, :]
    # 체간 스케일 정규화
    scale = np.mean(np.linalg.norm(xy_rel, axis=-1)) + 1e-5
    xy_norm = xy_rel / scale  # (64, 12, 2)

    return xy_norm.reshape(64, 24).astype(np.float32)

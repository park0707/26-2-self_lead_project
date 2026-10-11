# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Module: stiff_spine_feature_extractor.py
Purpose: 스티프 데드리프트 척추 중립(Spine Neutrality) 전용 30D 정예 생체역학 피처 추출기
         - 바벨 데드리프트 d1 v4.3 25D의 요추 포물선 곡률 및 시점 보정 피처 100% 계승
         - 스티프 데드리프트 고유 오류(Key 211/212 좌우 기울어짐, Key 216 어깨 말림, Key 215 과신전) 특화 피처 추가
         - AGENTS.md 및 인터페이스 기반 독립 모듈화 규정 준수
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

# 30D 스티프 데드리프트 척추 정예 피처 명칭
FEATURE_NAMES_30_STIFF_SPINE: List[str] = [
    # [1~6] 시상면 척추 만곡 및 요추 동역학
    "stiff_lumbar_poly_a",              # [ 1] 요추 2차 포물선 곡률 계수 평균 (굴곡 vs 과신전)
    "stiff_lumbar_stiffness_sd",        # [ 2] 전 구간 요추 각도 변동성 (척추 강성 안정도)
    "stiff_spine_lateral_dev_max",      # [ 3] 척추 중심선 횡방향 이탈 최대치 (좌우 비대칭)
    "stiff_trunk_flex_speed_max",       # [ 4] 상체 굴곡 속도 최대 피크
    "stiff_cervical_min",               # [ 5] 경추-흉추 최소 각도 (시선 및 경추 정렬)
    "stiff_signed_thoracolumbar_delta", # [ 6] 흉요추 상대 각도 최대 변화폭
    # [7~12] 가동범위 및 불균형 지표
    "stiff_active_rom",                 # [ 7] 손목 기준 수직 가동범위 (ROM)
    "stiff_cobb_mean",                  # [ 8] 요추 Cobb각 평균
    "stiff_shoulder_hip_twist_max",     # [ 9] 어깨선-골반선 3D 비틀림 각도
    "stiff_lateral_asym_integral",      # [10] 척추 좌우 불균형 시계열 누적 적분값
    "stiff_flex_accel_peak",            # [11] 상체 굴곡 각가속도 피크
    "stiff_pitch_standing",             # [12] 기립 어드레스 체간 전경각
    # [13~18] 시점 보정 및 기하학적 정규화
    "stiff_view_normal_z",              # [13] 체간 정면 법선벡터 Z축 투영 (시점 보정 메타)
    "stiff_shear_stress_proxy",         # [14] 상체 모멘트암 기반 요추 전단력 프록시
    "stiff_lockout_hyperextension",     # [15] 기립 락아웃 시 요추 과신전 오프셋 (Key 215)
    "stiff_pitch_rom",                  # [16] 상체 굴곡 각도 진폭 (Max - Min)
    "stiff_view_occlusion",             # [17] 후방 시점 가림 지표 (체간 Y축 Z투영)
    "stiff_view_comp_curv",             # [18] 3D 공간 만곡 보정 곡률 (Chord-Displacement)
    # [19~22] 체형 잔차화 및 신체 비율
    "stiff_res_lumbar_curv",            # [19] 체형 및 회전 디커플링 잔차 척추 곡률
    "stiff_ratio_arm_to_torso",         # [20] 팔 길이 대 체간 길이 비
    "stiff_ratio_femur_to_tibia",       # [21] 대퇴 대 하퇴 길이 비
    "stiff_ratio_leg_to_torso",         # [22] 다리 전체 대 체간 길이 비
    # [23~30] ★ 스티프 데드리프트 고유 결함 특화 피처 (좌우 비대칭, 흉추 말림, 무릎 고정)
    "stiff_shoulder_roll_max",          # [23] 어깨선 수평 기울기 최대 편차 (Key 211, 212)
    "stiff_hip_roll_max",               # [24] 골반선 수평 기울기 최대 편차 (골반 횡경사)
    "stiff_thoracic_kyphosis_peak",     # [25] 상부 흉추 곡률 및 어깨 전방 말림 (Key 216)
    "stiff_lumbopelvic_rhythm_ratio",   # [26] 골반 전경각 속도 대비 요추 굴곡 속도비
    "stiff_knee_fixation_sd",           # [27] 무릎 관절 각도 고정 안정도 (스티프 고유 무릎 제어)
    "stiff_torso_pelvis_lateral_offset",# [28] 체간 중심과 골반 중심의 횡방향 상대 편차
    "stiff_lateral_drift_direction",    # [29] 척추 중심선 좌우 편향 부호/방향성 지표
    "stiff_sagittal_spine_stiffness"    # [30] 하강-상승 전환 시 척추 형태 보존 강성 지표
]


def extract_morphology_stiff(kps_64: np.ndarray) -> np.ndarray:
    """피험자 형태학적 비율 (체형 정규화용 6D)"""
    k_base = np.median(kps_64[:8], axis=0)  # (33, 3)

    sh_mid = 0.5 * (k_base[MP['LSH']] + k_base[MP['RSH']])
    hip_mid = 0.5 * (k_base[MP['LHIP']] + k_base[MP['RHIP']])
    torso_len = max(1e-4, float(np.linalg.norm(sh_mid - hip_mid)))

    l_arm = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['LELB']]) + np.linalg.norm(k_base[MP['LELB']] - k_base[MP['LWR']])
    r_arm = np.linalg.norm(k_base[MP['RSH']] - k_base[MP['RELB']]) + np.linalg.norm(k_base[MP['RELB']] - k_base[MP['RWR']])
    arm_len = 0.5 * (l_arm + r_arm)

    l_femur = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['LKNE']])
    r_femur = np.linalg.norm(k_base[MP['RHIP']] - k_base[MP['RKNE']])
    femur_len = 0.5 * (l_femur + r_femur)

    l_tibia = np.linalg.norm(k_base[MP['LKNE']] - k_base[MP['LANK']])
    r_tibia = np.linalg.norm(k_base[MP['RKNE']] - k_base[MP['RANK']])
    tibia_len = 0.5 * (l_tibia + r_tibia)

    leg_len = femur_len + tibia_len
    sh_w = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['RSH']])
    hip_w = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['RHIP']])

    return np.array([
        arm_len / torso_len,
        femur_len / max(1e-4, tibia_len),
        leg_len / torso_len,
        arm_len / max(1e-4, leg_len),
        sh_w / max(1e-4, hip_w),
        torso_len
    ], dtype=np.float32)


class MorphologyResidualizerStiff:
    """피험자 고유 체형에 의한 척추 곡률 왜곡을 제거하는 잔차화기"""
    def __init__(self, alpha: float = 15.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_curv = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, morph_train: np.ndarray, curv_train: np.ndarray):
        Z = self.scaler.fit_transform(morph_train)
        self.ridge_curv.fit(Z, curv_train)
        self.is_fitted = True
        return self

    def transform(self, morph: np.ndarray, curv_val: float) -> float:
        if not self.is_fitted:
            return 0.0
        Z = self.scaler.transform(morph.reshape(1, -1))
        exp_curv = float(self.ridge_curv.predict(Z)[0])
        return curv_val - exp_curv


def extract_stiff_spine_features_30d(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None,
    residualizer: Optional[MorphologyResidualizerStiff] = None
) -> np.ndarray:
    """
    단일 64프레임 시퀀스로부터 30D 스티프 데드리프트 척추 중립 정예 피처 추출
    """
    T = kps_33_64f.shape[0]  # 64
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, MP['LSH'], :]
    r_sh = kps_33_64f[:, MP['RSH'], :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, MP['LHIP'], :]
    r_hip = kps_33_64f[:, MP['RHIP'], :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, MP['LKNE'], :]
    r_knee = kps_33_64f[:, MP['RKNE'], :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, MP['LANK'], :]
    r_ank = kps_33_64f[:, MP['RANK'], :]
    mid_ank = (l_ank + r_ank) / 2.0

    l_wrist = kps_33_64f[:, MP['LWR'], :]
    r_wrist = kps_33_64f[:, MP['RWR'], :]
    mid_wrist = (l_wrist + r_wrist) / 2.0

    # 1. 체간 로컬 정규 직교 기저계
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. SpinePose 9분절 3D 좌표 (없을 시 선형 보간 대체)
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    # 3. 요추 2차 다항 곡률 poly_a
    z_s1 = np.sum((s_pts[:, 0] - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((s_pts[:, 1] - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((s_pts[:, 2] - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((s_pts[:, 3] - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((s_pts[:, 4] - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        p = np.polyfit(x_coords, lumbar_curves[t_idx], deg=2)
        poly_a_series[t_idx] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    # 4. 척추 횡방향 이탈 (관상면)
    x_s_disp = np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1)
    f3_lat_dev = float(np.max(np.abs(x_s_disp)))

    # 5. 상체 굴곡 각도 및 속도
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))

    # 6. 경추-흉추 각도
    v_cerv = s_pts[:, 8] - s_pts[:, 6]
    v_thor = s_pts[:, 5] - s_pts[:, 4]
    dot_ct = np.sum(v_cerv * v_thor, axis=-1)
    norm_ct = (np.linalg.norm(v_cerv, axis=-1) * np.linalg.norm(v_thor, axis=-1)) + 1e-7
    ct_angles = np.degrees(np.arccos(np.clip(dot_ct / norm_ct, -1.0, 1.0)))
    f5_cervical_min = float(np.min(ct_angles)) if spine_9_64f is not None else 150.0

    # 7. 흉요추 상대 각도 변화폭
    f6_signed_delta = float(np.ptp(poly_a_series))

    # 8. 가동범위
    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f7_active_rom = float(np.ptp(wrist_y_s))

    # 9. 요추 Cobb 각도 평균
    v_l5_s1 = s_pts[:, 0] - s_pts[:, 1]
    v_l3_l1 = s_pts[:, 3] - s_pts[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    cobb_series = gaussian_filter1d(np.abs(cobb_angles), sigma=1.0)
    f8_cobb_mean = float(np.mean(cobb_series))

    # 10. 어깨-골반 비틀림 및 좌우 비대칭 적분
    v_hip = l_hip - r_hip
    twist = np.degrees(np.arccos(np.clip(
        np.sum(v_sh * v_hip, axis=-1) / (np.linalg.norm(v_sh, axis=-1) * np.linalg.norm(v_hip, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f9_twist_max = float(np.max(twist))
    f10_asym_integ = float(np.mean(np.abs(x_s_disp)))

    # 11. 굴곡 가속도
    f11_flex_accel = float(np.max(np.abs(np.gradient(torso_pitch_vel, dt))))

    # 12. 기립 체간 전경각
    f12_pitch_standing = float(np.min(torso_pitch))

    # 13. 시점 법선 Z 투영
    f13_view_z = float(np.mean(z_hat[:, 2]))

    # 14. 요추 전단력 프록시
    f14_shear = float(np.mean(np.sin(np.radians(torso_pitch))))

    # 15. 기립 락아웃 과신전 (Key 215)
    finish_idx = int(np.argmin(wrist_y_s))
    f15_lockout = float(max(0.0, poly_a_series[finish_idx] - 0.05))

    # 16. 체간 굴곡 진폭
    f16_pitch_rom = float(np.ptp(torso_pitch))

    # 17. 후방 시점 가림 지표
    f17_view_occlusion = float(np.mean(np.abs(y_hat[:, 2])))

    # 18. 3D 공간 만곡 보정 곡률
    v_c7_s1 = s_pts[:, 6] - s_pts[:, 0]
    chord_len = np.linalg.norm(v_c7_s1, axis=-1) + 1e-6
    l3_disp = np.linalg.norm(np.cross(v_c7_s1, s_pts[:, 2] - s_pts[:, 0]), axis=-1) / chord_len
    f18_view_comp_curv = float(np.mean(l3_disp))

    # 19. 체형 잔차화 척추 곡률
    morph_6d = extract_morphology_stiff(kps_33_64f)
    if residualizer is not None and residualizer.is_fitted:
        f19_res_curv = float(residualizer.transform(morph_6d, f1_poly_a))
    else:
        f19_res_curv = 0.0

    # 20~22. 신체 비율
    f20_arm_torso = float(morph_6d[0])
    f21_femur_tibia = float(morph_6d[1])
    f22_leg_torso = float(morph_6d[2])

    # -------------------------------------------------------------
    # ★ 23~30: 스티프 데드리프트 고유 결함 특화 피처 (좌우 비대칭, 흉추 말림, 무릎 고정)
    # -------------------------------------------------------------
    # 23. 어깨선 수평 기울기 각도 (Roll) - Key 211(우측), Key 212(좌측)
    sh_horiz_dx = v_sh[:, 0]
    sh_horiz_dy = v_sh[:, 1]
    shoulder_roll = np.degrees(np.arctan2(sh_horiz_dy, np.abs(sh_horiz_dx) + 1e-7))
    f23_shoulder_roll_max = float(np.max(np.abs(shoulder_roll)))

    # 24. 골반선 수평 기울기 각도 (Hip Roll)
    hip_horiz_dx = v_hip[:, 0]
    hip_horiz_dy = v_hip[:, 1]
    hip_roll = np.degrees(np.arctan2(hip_horiz_dy, np.abs(hip_horiz_dx) + 1e-7))
    f24_hip_roll_max = float(np.max(np.abs(hip_roll)))

    # 25. 상부 흉추 곡률 및 어깨 전방 말림 (Key 216)
    # T3(idx 5)와 C7(idx 6) 및 어깨 전방 돌출 거리
    v_c7_midsh = mid_sh - s_pts[:, 6]
    thoracic_forward_disp = np.sum(v_c7_midsh * z_hat, axis=-1)
    f25_thoracic_kyphosis = float(np.max(thoracic_forward_disp))

    # 26. 골반-요추 리듬비 (Lumbopelvic Rhythm)
    # 골반 경사각(Pelvic tilt: S1-Hip)과 요추 곡률 속도비
    hip_tilt = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * np.array([0, 1, 0]), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    pelvic_vel = np.abs(np.gradient(hip_tilt, dt))
    lumbar_curv_vel = np.abs(np.gradient(poly_a_series, dt))
    f26_lumbopelvic_ratio = float(np.mean(lumbar_curv_vel) / (np.mean(pelvic_vel) + 1e-5))

    # 27. 무릎 고정 안정도 (스티프 데드 전제 조건)
    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f27_knee_fixation_sd = float(np.std(knee_angle))

    # 28. 체간-골반 횡방향 상대 오프셋 (정면 시점 C에서 극명)
    torso_hip_dx = np.abs(mid_sh[:, 0] - mid_hip[:, 0])
    f28_torso_pelvis_offset = float(np.mean(torso_hip_dx))

    # 29. 좌우 편향 방향성 지표 (Signed Direction: 음수=좌, 양수=우)
    f29_lateral_drift_dir = float(np.mean(x_s_disp))

    # 30. 시상면 척추 강성 유지 지표 (최저점 통과 시 형상 붕괴율)
    bottom_idx = int(np.argmax(wrist_y_s))
    curv_at_bottom = poly_a_series[bottom_idx]
    f30_sagittal_stiffness = float(curv_at_bottom - f1_poly_a)

    feats_30d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min, f6_signed_delta,
        f7_active_rom, f8_cobb_mean, f9_twist_max, f10_asym_integ, f11_flex_accel, f12_pitch_standing,
        f13_view_z, f14_shear, f15_lockout, f16_pitch_rom, f17_view_occlusion, f18_view_comp_curv,
        f19_res_curv, f20_arm_torso, f21_femur_tibia, f22_leg_torso,
        f23_shoulder_roll_max, f24_hip_roll_max, f25_thoracic_kyphosis, f26_lumbopelvic_ratio,
        f27_knee_fixation_sd, f28_torso_pelvis_offset, f29_lateral_drift_dir, f30_sagittal_stiffness
    ], dtype=np.float32)

    return np.nan_to_num(feats_30d, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_stiff_temporal_24d(kps_33: np.ndarray, spine_9: Optional[np.ndarray] = None) -> np.ndarray:
    """Dilated TCN 입력을 위한 (64, 24) 시계열 텐서"""
    T = 64
    mid_hip = (kps_33[:, MP['LHIP']] + kps_33[:, MP['RHIP']]) / 2.0
    mid_sh = (kps_33[:, MP['LSH']] + kps_33[:, MP['RSH']]) / 2.0
    mid_knee = (kps_33[:, MP['LKNE']] + kps_33[:, MP['RKNE']]) / 2.0
    mid_ank = (kps_33[:, MP['LANK']] + kps_33[:, MP['RANK']]) / 2.0

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = kps_33[:, MP['LSH']] - kps_33[:, MP['RSH']]
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    if spine_9 is not None:
        s_pts = spine_9
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))

    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))

    # SpinePose 5개 대표 마디의 로컬 Z 변위 (굴곡) 및 X 변위 (좌우 비대칭)
    z_disp = np.sum((s_pts[:, [0, 2, 4, 6, 8]] - mid_hip[:, None, :]) * z_hat[:, None, :], axis=-1)  # (64, 5)
    x_disp = np.sum((s_pts[:, [0, 2, 4, 6, 8]] - mid_hip[:, None, :]) * x_hat[:, None, :], axis=-1)  # (64, 5)

    wrist_y = 0.5 * (kps_33[:, MP['LWR'], 1] + kps_33[:, MP['RWR'], 1])
    wrist_y_norm = (wrist_y - np.min(wrist_y)) / (np.ptp(wrist_y) + 1e-5)

    sh_dx = v_sh[:, 0]
    sh_dy = v_sh[:, 1]
    sh_roll = np.degrees(np.arctan2(sh_dy, np.abs(sh_dx) + 1e-7))

    hip_dx = (kps_33[:, MP['LHIP'], 0] - kps_33[:, MP['RHIP'], 0])
    hip_dy = (kps_33[:, MP['LHIP'], 1] - kps_33[:, MP['RHIP'], 1])
    hip_roll = np.degrees(np.arctan2(hip_dy, np.abs(hip_dx) + 1e-7))

    t_channels = [
        torso_pitch / 90.0,
        knee_angle / 180.0,
        wrist_y_norm,
        sh_roll / 45.0,
        hip_roll / 45.0,
        y_hat[:, 0], y_hat[:, 1], y_hat[:, 2],
        z_hat[:, 0], z_hat[:, 1], z_hat[:, 2],
        x_hat[:, 0], x_hat[:, 1], x_hat[:, 2],
        z_disp[:, 0], z_disp[:, 1], z_disp[:, 2], z_disp[:, 3], z_disp[:, 4],
        x_disp[:, 0], x_disp[:, 1], x_disp[:, 2], x_disp[:, 3], x_disp[:, 4],
    ]
    tensor_24d = np.column_stack(t_channels).astype(np.float32)  # (64, 24)
    return np.nan_to_num(tensor_24d, nan=0.0, posinf=1.0, neginf=-1.0)

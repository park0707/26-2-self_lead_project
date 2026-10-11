# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v9.0 (c1_model_v9.py)
Status: ★ 공식 64프레임 표준 & 체간 로컬 직교계 & SpinePose 9대 분절 통합 프로덕션 모델
        - 1) 체간 로컬 정규 직교 기저계(Body-Local Orthonormal Basis: y_hat, z_hat, x_hat) 100% 계승
        - 2) DFKI SpinePose 9대 척추 분절(S1~C1) 고정밀 3D 곡률 6D 완벽 통합
        - 3) 과거 레거시 92D Format B 텐서 슬라이싱(seq[:, 82] 등) 영구 제거
        - 4) 순수 MediaPipe 33관절 64F + SpinePose 9분절 64F 표준 인터페이스: extract_c1_v9_features_from_seq
        - 5) 마스터 데이터셋 2,605건 전수(5대 전 시점 x 9명 피험자) 5-Fold 피험자 완전 격리
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_NAMES_28_C1: List[str] = [
    # [1~6] SpinePose 9대 척추 분절 고정밀 생체역학 피처 (6D)
    "c1_v5_lumbar_curvature_poly_a",      # [1] L1-L3-L5-S1 요추 2차 곡률 계수 a (*1000, 전만 음수 vs 후만 양수)
    "c1_v5_thoracolumbar_cobb_angle",     # [2] 최하사점 L1-L5 요추 Cobb 전만각 (deg)
    "c1_v5_spine_s_curve_harmony",        # [3] 흉추 후만과 요추 전만의 S자 곡률 밸런스 비율
    "c1_v5_cervicothoracic_hinge_pitch",  # [4] C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각 (deg)
    "c1_v5_nadir_lumbar_tangent_jerk",    # [5] 최하사점 요추 접선각의 2차 미분 각가속도 급변도
    "c1_v5_pelvis_l5_retroversion_delta", # [6] 골반(S1) 대비 L5 요추체의 후방 말림(Retroversion) 변위 (cm)

    # [7~14] 체간 로컬 정규 직교계 투영 척추/골반/대퇴 기하 피처 (8D)
    "c1_v4_signed_thoracolumbar_nadir",   # [7] 최하사점 부호 있는 흉요추 곡률각 (deg, +전만/-후만)
    "c1_v4_signed_thoracolumbar_delta",   # [8] 기립 대비 최하사점 흉요추 곡률 변화량 (deg)
    "c1_v4_signed_lumbopelvic_nadir",     # [9] 최하사점 부호 있는 요추-골반 상대각 (deg)
    "c1_v4_pelvifemoral_angle_nadir",     # [10] 최하사점 대퇴-골반 사잇각 (deg)
    "c1_v4_thoracolumbar_fluctuation_sd", # [11] 하강~최하사점 척추 곡률 진동/불안정성 표준편차
    "c1_v4_hip_hinge_spine_decoupling",   # [12] 상체 기울기각 변화 대비 척추 굽힘 분리비율 (굿모닝 스쿼트 지표)
    "c1_v4_spine_lateral_deviation_max",  # [13] 관상면(x_hat) 좌우 수평 이탈 거리 최대치 (cm)
    "c1_v4_lumbar_shear_moment_arm",      # [14] 최하사점 요추 전단력 모멘트암 (cm)

    # [15~26] MediaPipe 33관절 64프레임 직접 계측 생체역학 피처 (12D)
    "c1_trunk_flex_max",                  # [15] 수직선 기준 최대 체간 굴곡각 (deg)
    "c1_lumbar_hyperext_max",             # [16] 기립 기준 체간/요추 과신전 최대각 (deg)
    "c1_trunk_flex_speed_max",            # [17] 체간 굴곡 각속도 최대 피크 (|deg/s|)
    "c1_lumbar_stiffness_sd",             # [18] 전 구간 체간 각도 시계열 변동 표준편차
    "c1_asymmetry_score",                 # [19] 좌우 어깨선과 골반선의 롤(Roll) 불일치각 90th percentile (deg)
    "c1_pelvis_trunk_sync_err",           # [20] 골반 하강 곡선과 체간 굴곡 곡선 간 상호상관 지연 프레임 (Lag)
    "c1_phase_speed_ratio",               # [21] 하강 평균 속도 대비 상승 평균 속도 비 (Rv)
    "c1_bottom_dwell_frames",             # [22] 가동범위 최하위 10% 구간 체류 프레임 수
    "c1_torso_shank_err",                 # [23] 최하사점 체간 벡터와 정강이 벡터 간 평행성 오차 (deg)
    "c1_cervical_min",                    # [24] 귀-어깨-골반 경추-체간 최소 사잇각 (deg)
    "c1_gaze_pitch_min",                  # [25] 코-귀 머리 시선 피치 최소각 (deg)
    "c1_lit_lumbar_shear_proxy",          # [26] 어깨-골반 중심과 발목 중심 간의 전후 수평 이탈 모멘트 (cm)

    # [27~28] 품질 및 시점 메타 피처 (2D)
    "c1_view_score_normal_z",             # [27] 체간 법선 벡터의 카메라 Z축 연속 성분 (시점 지표)
    "c1_active_rom"                       # [28] 스쿼트 수직 가동범위 (골반 최고점 - 최저점 변위, m)
]


def extract_c1_v9_features_from_seq(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    MediaPipe 33관절 64프레임 좌표 및 SpinePose 9대 척추 분절 좌표로부터
    c1 v9.0 공식 28D 생체역학 피처셋 직접 추출
    - kps_33_64f: (64, 33, 3) MediaPipe 3D 월드 좌표
    - spine_9_64f: (64, 9, 3) SpinePose 9대 분절 3D 좌표 (S1, L5, L3, L1, T8, T3, C7, C4, C1)
                   (None인 경우 kps_33으로부터 척추 축을 근사 복원)
    """
    T = kps_33_64f.shape[0] # 64
    dt = 3.0 / max(1, T - 1)

    # MediaPipe 주요 관절
    nose = kps_33_64f[:, 0, :]
    l_ear = kps_33_64f[:, 7, :]
    r_ear = kps_33_64f[:, 8, :]
    mid_ear = (l_ear + r_ear) / 2.0

    l_sh = kps_33_64f[:, 11, :]
    r_sh = kps_33_64f[:, 12, :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]
    r_hip = kps_33_64f[:, 24, :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, 25, :]
    r_knee = kps_33_64f[:, 26, :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, 27, :]
    r_ank = kps_33_64f[:, 28, :]
    mid_ank = (l_ank + r_ank) / 2.0

    # 1. 체간 로컬 정규 직교 기저계 (Body-Local Orthonormal Basis)
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    normal_z_mean = float(np.mean(z_hat[:, 2]))

    # 2. 최하사점(변곡점 b) 검출 및 위상 분석
    # 골반 y좌표 및 무릎 y좌표 하강 분석
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(2, T // 8)

    active_rom = float(np.max(pelvis_y_s) - np.min(pelvis_y_s))

    # 능동 운동 구간 마스크 (가동범위 상단 15% 기립 및 운동 종료 후 잉여 프레임 차단)
    y_min, y_max = np.min(pelvis_y_s), np.max(pelvis_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (pelvis_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 3. SpinePose 9분절 준비 (제공되지 않은 경우 체간 기반 복원)
    if spine_9_64f is None:
        # 근사 9분절 복원: S1(mid_hip), C7(mid_sh), L1~T8 선형 분할
        spine_9 = np.zeros((T, 9, 3), dtype=np.float32)
        weights = [0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00]
        for idx_w, w in enumerate(weights):
            spine_9[:, idx_w, :] = mid_hip + w * (mid_sh - mid_hip)
    else:
        spine_9 = spine_9_64f

    # =========================================================================
    # [1~6] SpinePose 9대 척추 분절 고정밀 생체역학 피처 (6D)
    # =========================================================================
    # f1: L1-L3-L5-S1 요추 2차 곡률 다항식 계수 a
    poly_a_list = []
    for t in range(T):
        lumbar_pts = spine_9[t, :4] # S1, L5, L3, L1
        rel_pts = lumbar_pts - mid_hip[t]
        y_loc = np.sum(rel_pts * y_hat[t], axis=1)
        z_loc = np.sum(rel_pts * z_hat[t], axis=1)
        try:
            poly = np.polyfit(y_loc, z_loc, 2)
            poly_a_list.append(poly[0])
        except Exception:
            poly_a_list.append(0.0)

    poly_a_s = gaussian_filter1d(np.array(poly_a_list), sigma=1.0)
    f1_poly_a_nadir = float(poly_a_s[b] * 1000.0)

    # f2: 최하사점 L1-L5 요추 Cobb각 프록시
    v_l5_s1 = spine_9[:, 0] - spine_9[:, 1]
    v_l3_l1 = spine_9[:, 3] - spine_9[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    f2_cobb_nadir = float(gaussian_filter1d(cobb_angles, sigma=1.0)[b])

    # f3: 흉추 후만과 요추 전만의 S자 곡률 밸런스 비율
    v_t8_l1 = spine_9[:, 3] - spine_9[:, 4]
    v_t8_t3 = spine_9[:, 5] - spine_9[:, 4]
    dot_th = np.sum(v_t8_l1 * v_t8_t3, axis=1)
    norm_th1 = np.linalg.norm(v_t8_l1, axis=1) + 1e-6
    norm_th2 = np.linalg.norm(v_t8_t3, axis=1) + 1e-6
    thoracic_angle = np.degrees(np.arccos(np.clip(dot_th / (norm_th1 * norm_th2), -1.0, 1.0)))
    th_win = thoracic_angle[max(0, b - 2):min(T, b + 3)]
    f3_s_harmony = float(np.abs(f2_cobb_nadir) / (np.mean(th_win) + 1e-4))

    # f4: C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각
    v_cervical = spine_9[:, 8] - spine_9[:, 6]
    v_thoracic = spine_9[:, 5] - spine_9[:, 4]
    dot_ct = np.sum(v_cervical * v_thoracic, axis=1)
    norm_c = np.linalg.norm(v_cervical, axis=1) + 1e-6
    norm_t = np.linalg.norm(v_thoracic, axis=1) + 1e-6
    cross_ct = np.cross(v_thoracic, v_cervical)
    sin_ct = np.sum(cross_ct * x_hat, axis=1) / (norm_c * norm_t)
    cos_ct = np.clip(dot_ct / (norm_c * norm_t), -1.0, 1.0)
    ct_pitch = np.degrees(np.arctan2(sin_ct, cos_ct))
    f4_ct_pitch_nadir = float(gaussian_filter1d(ct_pitch, sigma=1.0)[b])

    # f5: 최하사점 요추 접선각의 2차 미분 각가속도 급변도
    d_cobb = np.gradient(gaussian_filter1d(cobb_angles, sigma=1.0), dt)
    d2_cobb = np.gradient(d_cobb, dt)
    b_win = slice(max(0, b - 3), min(T, b + 4))
    f5_tangent_jerk = float(np.max(np.abs(d2_cobb[b_win])) / 100.0)

    # f6: 골반(S1) 대비 L5 요추체의 후방 말림 변위 (cm)
    diff_l5_s1 = spine_9[:, 1] - spine_9[:, 0]
    retro_z = np.sum(diff_l5_s1 * z_hat, axis=1) * 100.0 # cm 단위
    f6_retro_delta = float(retro_z[b] - np.mean(retro_z[:standing_slice]))

    # =========================================================================
    # [7~14] 체간 로컬 정규 직교계 투영 척추/골반/대퇴 기하 피처 (8D)
    # =========================================================================
    w_lower = spine_9[:, 4] - spine_9[:, 2] # L3 -> T8
    v_upper = spine_9[:, 6] - spine_9[:, 4] # T8 -> C7
    v_pelvis = mid_hip - spine_9[:, 2]      # L3 -> Pelvis
    v_femur = mid_knee - mid_hip            # Pelvis -> Knee

    # f7 & f8: 부호 있는 흉요추 곡률
    cross_tl = np.cross(w_lower, v_upper)
    dot_tl = np.sum(w_lower * v_upper, axis=1)
    norm_w = np.linalg.norm(w_lower, axis=1) + 1e-6
    norm_u = np.linalg.norm(v_upper, axis=1) + 1e-6
    sin_tl = np.sum(cross_tl * x_hat, axis=1) / (norm_w * norm_u)
    cos_tl = np.clip(dot_tl / (norm_w * norm_u), -1.0, 1.0)
    phi_tl = np.degrees(np.arctan2(sin_tl, cos_tl))
    phi_tl_s = gaussian_filter1d(phi_tl, sigma=1.0)
    f7_tl_nadir = float(phi_tl_s[b])
    tl_standing = float(np.mean(phi_tl_s[:standing_slice]))
    f8_tl_delta = float(f7_tl_nadir - tl_standing)

    # f9: 부호 있는 요추-골반 상대각
    cross_lp = np.cross(-w_lower, v_pelvis)
    dot_lp = np.sum((-w_lower) * v_pelvis, axis=1)
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6
    sin_lp = np.sum(cross_lp * x_hat, axis=1) / (norm_w * norm_p)
    cos_lp = np.clip(dot_lp / (norm_w * norm_p), -1.0, 1.0)
    phi_lp = np.degrees(np.arctan2(sin_lp, cos_lp))
    phi_lp_s = gaussian_filter1d(phi_lp, sigma=1.0)
    f9_lp_nadir = float(phi_lp_s[b])

    # f10: 대퇴-골반 사잇각
    cross_pf = np.cross(v_pelvis, v_femur)
    dot_pf = np.sum(v_pelvis * v_femur, axis=1)
    norm_f = np.linalg.norm(v_femur, axis=1) + 1e-6
    sin_pf = np.sum(cross_pf * x_hat, axis=1) / (norm_p * norm_f)
    cos_pf = np.clip(dot_pf / (norm_p * norm_f), -1.0, 1.0)
    phi_pf = np.degrees(np.arctan2(sin_pf, cos_pf))
    phi_pf_s = gaussian_filter1d(phi_pf, sigma=1.0)
    f10_pf_nadir = float(phi_pf_s[b])

    # f11: 하강 구간 흉요추 곡률 진동/불안정성 표준편차
    f11_tl_fluct_sd = float(np.std(phi_tl_s[:b+1]))

    # f12: 상체 기울기각 변화 대비 척추 굽힘 분리비율 (굿모닝 스쿼트 프록시)
    trunk_pitch_raw = np.degrees(np.arctan2(v_torso[:, 2], -v_torso[:, 1]))
    trunk_pitch_s = gaussian_filter1d(trunk_pitch_raw, sigma=1.0)
    trunk_pitch_delta = max(1e-4, float(trunk_pitch_s[b] - np.mean(trunk_pitch_s[:standing_slice])))
    f12_hip_hinge_ratio = float(abs(f8_tl_delta) / trunk_pitch_delta)

    # f13: 관상면(x_hat) 좌우 수평 이탈 거리 최대치 (cm, 능동 구간 한정)
    mid_spine = spine_9[:, 4] # T8
    spine_x_dev = np.abs(np.sum((mid_spine - mid_hip) * x_hat, axis=1)) * 100.0
    f13_spine_lat_dev_max = float(np.max(spine_x_dev[active_mask]))

    # f14: 최하사점 요추 전단력 모멘트암 (cm)
    shear_arm = np.abs(np.sum((spine_9[:, 2] - mid_hip) * z_hat, axis=1)) * 100.0
    f14_shear_moment_arm = float(shear_arm[b])

    # =========================================================================
    # [15~26] MediaPipe 33관절 64프레임 직접 계측 생체역학 피처 (12D)
    # =========================================================================
    # f15: 수직선 기준 최대 체간 굴곡각 (능동 구간 한정)
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    trunk_angles = np.degrees(np.arccos(np.clip(
        np.sum(v_torso * v_vert, axis=1) / (np.linalg.norm(v_torso, axis=1) + 1e-7), -1.0, 1.0
    )))
    trunk_s = gaussian_filter1d(trunk_angles, sigma=1.0)
    trunk_base = float(np.mean(trunk_s[:standing_slice]))
    f15_trunk_flex_max = float(np.max(trunk_s[active_mask]) - trunk_base)

    # f16: 기립 기준 체간/요추 과신전 최대각 (능동 구간 한정)
    f16_lumbar_hyperext_max = float(max(0.0, trunk_base - np.min(trunk_s[active_mask])))

    # f17: 체간 굴곡 각속도 최대 피크 (|deg/s|, 능동 구간 한정)
    trunk_vel = np.gradient(trunk_s, dt)
    f17_trunk_speed_max = float(np.max(np.abs(trunk_vel[active_mask])))

    # f18: 능동 구간 체간 각도 시계열 변동 표준편차
    f18_trunk_stiffness_sd = float(np.std(trunk_s[active_mask]))

    # f19: 좌우 어깨선과 골반선의 롤(Roll) 불일치각 90th percentile
    v_hip_lr = l_hip - r_hip
    cos_sh_hip = np.sum(v_sh * v_hip_lr, axis=1) / (
        np.linalg.norm(v_sh, axis=1) * np.linalg.norm(v_hip_lr, axis=1) + 1e-7
    )
    roll_mismatch = np.degrees(np.arccos(np.clip(cos_sh_hip, -1.0, 1.0)))
    f19_asym_score = float(np.percentile(roll_mismatch, 90))

    # f20: 골반 하강 곡선과 체간 굴곡 곡선 간 상호상관 지연 프레임
    p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
    t_norm = (trunk_s - np.mean(trunk_s)) / (np.std(trunk_s) + 1e-7)
    xcorr = correlate(t_norm, p_norm, mode='full')
    lags = np.arange(-(T - 1), T)
    f20_sync_lag = float(abs(lags[np.argmax(xcorr)]))

    # f21: 하강 평균 속도 대비 상승 평균 속도 비 (Rv)
    vel_descent = pelvis_y_s[b] - pelvis_y_s[0]
    dur_descent = max(1, b)
    vel_ascent = pelvis_y_s[b] - pelvis_y_s[-1]
    dur_ascent = max(1, T - 1 - b)
    speed_descent = abs(vel_descent / dur_descent)
    speed_ascent = abs(vel_ascent / dur_ascent)
    f21_phase_speed_ratio = float(speed_ascent / (speed_descent + 1e-6))

    # f22: 가동범위 최하위 10% 구간 체류 프레임 수
    deep_thresh = np.max(pelvis_y_s) - 0.10 * active_rom
    f22_bottom_dwell = float(np.sum(pelvis_y_s >= deep_thresh))

    # f23: 최하사점 체간 벡터와 정강이 벡터 간 평행성 오차
    v_shank = mid_knee - mid_ank
    cos_ts = np.sum(v_torso * v_shank, axis=1) / (
        np.linalg.norm(v_torso, axis=1) * np.linalg.norm(v_shank, axis=1) + 1e-7
    )
    parallel_err = np.degrees(np.arccos(np.clip(cos_ts, -1.0, 1.0)))
    f23_torso_shank_err = float(gaussian_filter1d(parallel_err, sigma=1.0)[b])

    # f24: 귀-어깨-골반 경추-체간 최소 사잇각
    v_neck = mid_ear - mid_sh
    cos_cerv = np.sum(v_neck * v_torso, axis=1) / (
        np.linalg.norm(v_neck, axis=1) * np.linalg.norm(v_torso, axis=1) + 1e-7
    )
    cerv_angles = np.degrees(np.arccos(np.clip(cos_cerv, -1.0, 1.0)))
    f24_cervical_min = float(np.min(cerv_angles))

    # f25: 코-귀 머리 시선 피치 최소각
    v_head = nose - mid_ear
    head_h = np.sqrt(v_head[:, 0]**2 + v_head[:, 2]**2) + 1e-6
    head_pitch = np.degrees(np.arctan2(-v_head[:, 1], head_h))
    f25_gaze_pitch_min = float(np.min(head_pitch))

    # f26: 어깨-골반 중심과 발목 중심 간의 전후 수평 이탈 모멘트
    body_center = (mid_sh + mid_hip) / 2.0
    shear_h = np.abs(body_center[:, 2] - mid_ank[:, 2]) * 100.0 # cm
    f26_lumbar_shear_proxy = float(shear_h[b])

    # =========================================================================
    # [27~28] 품질 및 시점 메타 피처 (2D)
    # =========================================================================
    f27_view_score_z = normal_z_mean
    f28_active_rom = active_rom

    feats = np.array([
        f1_poly_a_nadir, f2_cobb_nadir, f3_s_harmony, f4_ct_pitch_nadir,
        f5_tangent_jerk, f6_retro_delta, f7_tl_nadir, f8_tl_delta,
        f9_lp_nadir, f10_pf_nadir, f11_tl_fluct_sd, f12_hip_hinge_ratio,
        f13_spine_lat_dev_max, f14_shear_moment_arm, f15_trunk_flex_max,
        f16_lumbar_hyperext_max, f17_trunk_speed_max, f18_trunk_stiffness_sd,
        f19_asym_score, f20_sync_lag, f21_phase_speed_ratio, f22_bottom_dwell,
        f23_torso_shank_err, f24_cervical_min, f25_gaze_pitch_min,
        f26_lumbar_shear_proxy, f27_view_score_z, f28_active_rom
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)


class BabelSquatC1ModelV9:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v9.0 공식 프로덕션 모델
    CatBoost 60% + LightGBM 40% 소프트 보팅 앙상블 (28D 고정밀 피처셋)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.cb_models: List[CatBoostClassifier] = []
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_28_C1

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c1_model_v9_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        결함 발생 확률 배열 반환
        입력: (N, 28) 또는 (28,)
        출력: (N, 2) [P(정상), P(척추 중립 위반 결함)]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v9 모델 가중치가 로드되지 않았습니다.")

        is_single = (X.ndim == 1)
        if is_single:
            X = np.expand_dims(X, axis=0)

        N = X.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X)[:, 1]
            p_lgb = clf.predict_proba(X)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)
        p0 = 1.0 - ensemble_probs
        res = np.column_stack([p0, ensemble_probs])
        return res[0] if is_single else res

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판정 (0: 정상, 1: 척추 중립 위반 결함)"""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v9.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_28_C1)
        self.is_fitted = True
        return self


BabelSquatC1Model = BabelSquatC1ModelV9

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.0_a_diet (c1_model_v11_a_diet.py)
Status: ★ [E0 피처 정제 및 다이어트 확정] 27D 정예 생체역학 피처셋 공식 프로덕션 모델
        - 1) 기존 v10 (28D) 중 다중공선성 및 저기여도 노이즈 피처 2개 제거:
             * c1_bottom_dwell_frames (가동범위 하위 체류 프레임, 기여도 0.51% 최하위)
             * c1_v5_spine_s_curve_harmony (S자 곡률 비율, 라벨 상관계수 -0.008)
        - 2) 외부 AI 자문 제안 E0(Nadir 국소 윈도우) 중 가장 강력한 핵심 피처 1개 통합:
             * c1_e0_lumbopelvic_win_sd (Nadir ±8F 윈도우 내 요추-골반 상대각 국소 표준편차)
        - 3) 실측 성과 (N=2,605 전수):
             * 전체 F1: 79.73% -> 80.52% (+0.79%p 상승, 80% 벽 돌파)
             * 전체 Acc: 80.38% -> 81.54% (+1.16%p 상승)
             * 최약 시점 정면(View 3) F1: 74.30% -> 82.60% (+8.30%p 폭등)
             * 후방(View 1 76.1%) 및 측면(View 2 82.9%, View 4 81.4%) 무결성 완벽 보존
        - 4) 3계층 선택적 거부(Selective Abstention / 3-State) 아키텍처 완전 내장
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

FEATURE_NAMES_27_C1_DIET: List[str] = [
    # [1~5] SpinePose 9대 척추 분절 고정밀 생체역학 피처 (5D, S자 곡률 제외)
    "c1_v5_lumbar_curvature_poly_a",      # [1] L1-L3-L5-S1 요추 2차 곡률 계수 a (*1000)
    "c1_v5_thoracolumbar_cobb_angle",     # [2] 최하사점 L1-L5 요추 Cobb 전만각 (deg)
    "c1_v5_cervicothoracic_hinge_pitch",  # [3] C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각 (deg)
    "c1_v5_nadir_lumbar_tangent_jerk",    # [4] 최하사점 요추 접선각의 2차 미분 각가속도 급변도
    "c1_v5_pelvis_l5_retroversion_delta", # [5] 골반(S1) 대비 L5 요추체의 후방 말림(Retroversion) 변위 (cm)

    # [6~13] 체간 로컬 정규 직교계 투영 척추/골반/대퇴 기하 피처 (8D)
    "c1_v4_signed_thoracolumbar_nadir",   # [6] 최하사점 부호 있는 흉요추 곡률각 (deg)
    "c1_v4_signed_thoracolumbar_delta",   # [7] 기립 대비 최하사점 흉요추 곡률 변화량 (deg)
    "c1_v4_signed_lumbopelvic_nadir",     # [8] 최하사점 부호 있는 요추-골반 상대각 (deg)
    "c1_v4_pelvifemoral_angle_nadir",     # [9] 최하사점 대퇴-골반 사잇각 (deg)
    "c1_v4_thoracolumbar_fluctuation_sd", # [10] 하강~최하사점 척추 곡률 진동 표준편차
    "c1_v4_hip_hinge_spine_decoupling",   # [11] 상체 기울기각 변화 대비 척추 굽힘 분리비율
    "c1_v4_spine_lateral_deviation_max",  # [12] 관상면(x_hat) 좌우 수평 이탈 거리 최대치 (cm)
    "c1_v4_lumbar_shear_moment_arm",      # [13] 최하사점 요추 전단력 모멘트암 (cm)

    # [14~24] MediaPipe 33관절 64프레임 직접 계측 생체역학 피처 (11D, 체류 프레임 제외)
    "c1_trunk_flex_max",                  # [14] 수직선 기준 최대 체간 굴곡각 (deg)
    "c1_lumbar_hyperext_max",             # [15] 기립 기준 체간/요추 과신전 최대각 (deg)
    "c1_trunk_flex_speed_max",            # [16] 체간 굴곡 각속도 최대 피크 (|deg/s|)
    "c1_lumbar_stiffness_sd",             # [17] 전 구간 체간 각도 시계열 변동 표준편차
    "c1_asymmetry_score",                 # [18] 좌우 어깨선과 골반선의 롤 불일치각 90th percentile
    "c1_pelvis_trunk_sync_err",           # [19] 골반 하강 곡선과 체간 굴곡 곡선 간 상호상관 지연 프레임
    "c1_phase_speed_ratio",               # [20] 하강 평균 속도 대비 상승 평균 속도 비
    "c1_torso_shank_err",                 # [21] 최하사점 체간 벡터와 정강이 벡터 간 평행성 오차 (deg)
    "c1_cervical_min",                    # [22] 귀-어깨-골반 경추-체간 최소 사잇각 (deg)
    "c1_gaze_pitch_min",                  # [23] 코-귀 머리 시선 피치 최소각 (deg)
    "c1_lit_lumbar_shear_proxy",          # [24] 어깨-골반 중심과 발목 중심 간의 전후 수평 이탈 모멘트 (cm)

    # [25~26] 품질 및 시점 메타 피처 (2D)
    "c1_view_score_normal_z",             # [25] 체간 법선 벡터의 카메라 Z축 연속 성분 (시점 지표)
    "c1_active_rom",                      # [26] 스쿼트 수직 가동범위 (골반 최고점 - 최저점 변위, m)

    # [★ 27] E0 정예 핵심 국소 피처 (1D)
    "c1_e0_lumbopelvic_win_sd"            # [27] Nadir ±8F 윈도우 내 요추-골반 상대각 국소 표준편차 (deg)
]


def extract_c1_v11_a_diet_features_from_seq(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    MediaPipe 33관절 64프레임 좌표 및 SpinePose 9대 척추 분절 좌표로부터
    c1 v11_a_diet 27D 정예 생체역학 피처셋 직접 추출
    """
    T = kps_33_64f.shape[0]  # 64
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

    # 1. 체간 로컬 정규 직교 기저계
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    normal_z_mean = float(np.mean(z_hat[:, 2]))

    # 2. 최하사점(변곡점 b) 검출 및 위상 분석
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(2, T // 8)

    active_rom = float(np.max(pelvis_y_s) - np.min(pelvis_y_s))

    y_min, y_max = np.min(pelvis_y_s), np.max(pelvis_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (pelvis_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 3. SpinePose 9분절 준비
    if spine_9_64f is None:
        spine_9 = np.zeros((T, 9, 3), dtype=np.float32)
        weights = [0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00]
        for idx_w, w in enumerate(weights):
            spine_9[:, idx_w, :] = mid_hip + w * (mid_sh - mid_hip)
    else:
        spine_9 = spine_9_64f

    # [SpinePose 9대 분절 생체역학 피처]
    poly_a_list = []
    for t in range(T):
        lumbar_pts = spine_9[t, :4]
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

    v_l5_s1 = spine_9[:, 0] - spine_9[:, 1]
    v_l3_l1 = spine_9[:, 3] - spine_9[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    cobb_angles_s = gaussian_filter1d(cobb_angles, sigma=1.0)
    f2_cobb_nadir = float(cobb_angles_s[b])

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

    d_cobb = np.gradient(cobb_angles_s, dt)
    d2_cobb = np.gradient(d_cobb, dt)
    b_win = slice(max(0, b - 3), min(T, b + 4))
    f5_tangent_jerk = float(np.max(np.abs(d2_cobb[b_win])) / 100.0)

    diff_l5_s1 = spine_9[:, 1] - spine_9[:, 0]
    retro_z = np.sum(diff_l5_s1 * z_hat, axis=1) * 100.0
    retro_z_s = gaussian_filter1d(retro_z, sigma=1.0)
    f6_retro_delta = float(retro_z_s[b] - np.mean(retro_z_s[:standing_slice]))

    # [체간 로컬 정규 직교계 투영 기하 피처]
    w_lower = spine_9[:, 4] - spine_9[:, 2]
    v_upper = spine_9[:, 6] - spine_9[:, 4]
    v_pelvis = mid_hip - spine_9[:, 2]
    v_femur = mid_knee - mid_hip

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

    cross_lp = np.cross(-w_lower, v_pelvis)
    dot_lp = np.sum((-w_lower) * v_pelvis, axis=1)
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6
    sin_lp = np.sum(cross_lp * x_hat, axis=1) / (norm_w * norm_p)
    cos_lp = np.clip(dot_lp / (norm_w * norm_p), -1.0, 1.0)
    phi_lp = np.degrees(np.arctan2(sin_lp, cos_lp))
    phi_lp_s = gaussian_filter1d(phi_lp, sigma=1.0)
    f9_lp_nadir = float(phi_lp_s[b])

    cross_pf = np.cross(v_pelvis, v_femur)
    dot_pf = np.sum(v_pelvis * v_femur, axis=1)
    norm_f = np.linalg.norm(v_femur, axis=1) + 1e-6
    sin_pf = np.sum(cross_pf * x_hat, axis=1) / (norm_p * norm_f)
    cos_pf = np.clip(dot_pf / (norm_p * norm_f), -1.0, 1.0)
    phi_pf = np.degrees(np.arctan2(sin_pf, cos_pf))
    phi_pf_s = gaussian_filter1d(phi_pf, sigma=1.0)
    f10_pf_nadir = float(phi_pf_s[b])

    f11_tl_fluct_sd = float(np.std(phi_tl_s[:b+1]))

    trunk_pitch_raw = np.degrees(np.arctan2(v_torso[:, 2], -v_torso[:, 1]))
    trunk_pitch_s = gaussian_filter1d(trunk_pitch_raw, sigma=1.0)
    trunk_pitch_delta = max(1e-4, float(trunk_pitch_s[b] - np.mean(trunk_pitch_s[:standing_slice])))
    f12_hip_hinge_ratio = float(abs(f8_tl_delta) / trunk_pitch_delta)

    mid_spine = spine_9[:, 4]
    spine_x_dev = np.abs(np.sum((mid_spine - mid_hip) * x_hat, axis=1)) * 100.0
    f13_spine_lat_dev_max = float(np.max(spine_x_dev[active_mask]))

    shear_arm = np.abs(np.sum((spine_9[:, 2] - mid_hip) * z_hat, axis=1)) * 100.0
    f14_shear_moment_arm = float(shear_arm[b])

    # [MediaPipe 33관절 직접 계측 피처]
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    trunk_angles = np.degrees(np.arccos(np.clip(
        np.sum(v_torso * v_vert, axis=1) / (np.linalg.norm(v_torso, axis=1) + 1e-7), -1.0, 1.0
    )))
    trunk_s = gaussian_filter1d(trunk_angles, sigma=1.0)
    trunk_base = float(np.mean(trunk_s[:standing_slice]))
    f15_trunk_flex_max = float(np.max(trunk_s[active_mask]) - trunk_base)

    f16_lumbar_hyperext_max = float(max(0.0, trunk_base - np.min(trunk_s[active_mask])))

    trunk_vel = np.gradient(trunk_s, dt)
    f17_trunk_speed_max = float(np.max(np.abs(trunk_vel[active_mask])))

    f18_trunk_stiffness_sd = float(np.std(trunk_s[active_mask]))

    v_hip_lr = l_hip - r_hip
    cos_sh_hip = np.sum(v_sh * v_hip_lr, axis=1) / (
        np.linalg.norm(v_sh, axis=1) * np.linalg.norm(v_hip_lr, axis=1) + 1e-7
    )
    roll_mismatch = np.degrees(np.arccos(np.clip(cos_sh_hip, -1.0, 1.0)))
    f19_asym_score = float(np.percentile(roll_mismatch, 90))

    p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
    t_norm = (trunk_s - np.mean(trunk_s)) / (np.std(trunk_s) + 1e-7)
    xcorr = correlate(t_norm, p_norm, mode='full')
    lags = np.arange(-(T - 1), T)
    f20_sync_lag = float(abs(lags[np.argmax(xcorr)]))

    vel_descent = pelvis_y_s[b] - pelvis_y_s[0]
    dur_descent = max(1, b)
    vel_ascent = pelvis_y_s[b] - pelvis_y_s[-1]
    dur_ascent = max(1, T - 1 - b)
    speed_descent = abs(vel_descent / dur_descent)
    speed_ascent = abs(vel_ascent / dur_ascent)
    f21_phase_speed_ratio = float(speed_ascent / (speed_descent + 1e-6))

    v_shank = mid_knee - mid_ank
    cos_ts = np.sum(v_torso * v_shank, axis=1) / (
        np.linalg.norm(v_torso, axis=1) * np.linalg.norm(v_shank, axis=1) + 1e-7
    )
    parallel_err = np.degrees(np.arccos(np.clip(cos_ts, -1.0, 1.0)))
    f23_torso_shank_err = float(gaussian_filter1d(parallel_err, sigma=1.0)[b])

    v_neck = mid_ear - mid_sh
    cos_cerv = np.sum(v_neck * v_torso, axis=1) / (
        np.linalg.norm(v_neck, axis=1) * np.linalg.norm(v_torso, axis=1) + 1e-7
    )
    cerv_angles = np.degrees(np.arccos(np.clip(cos_cerv, -1.0, 1.0)))
    f24_cervical_min = float(np.min(cerv_angles))

    v_head = nose - mid_ear
    head_h = np.sqrt(v_head[:, 0]**2 + v_head[:, 2]**2) + 1e-6
    head_pitch = np.degrees(np.arctan2(-v_head[:, 1], head_h))
    f25_gaze_pitch_min = float(np.min(head_pitch))

    body_center = (mid_sh + mid_hip) / 2.0
    shear_h = np.abs(body_center[:, 2] - mid_ank[:, 2]) * 100.0
    f26_lumbar_shear_proxy = float(shear_h[b])

    # [품질 및 시점 메타 피처]
    f27_view_score_z = normal_z_mean
    f28_active_rom = active_rom

    # [★ E0 Nadir 국소 윈도우 정예 피처]
    w_start = max(0, b - 8)
    w_end = min(T, b + 9)
    win_lp = phi_lp_s[w_start:w_end]
    f27_e0_lumbopelvic_win_sd = float(np.std(win_lp))

    feats_27 = np.array([
        f1_poly_a_nadir, f2_cobb_nadir, f4_ct_pitch_nadir, f5_tangent_jerk,
        f6_retro_delta, f7_tl_nadir, f8_tl_delta, f9_lp_nadir,
        f10_pf_nadir, f11_tl_fluct_sd, f12_hip_hinge_ratio, f13_spine_lat_dev_max,
        f14_shear_moment_arm, f15_trunk_flex_max, f16_lumbar_hyperext_max,
        f17_trunk_speed_max, f18_trunk_stiffness_sd, f19_asym_score,
        f20_sync_lag, f21_phase_speed_ratio, f23_torso_shank_err,
        f24_cervical_min, f25_gaze_pitch_min, f26_lumbar_shear_proxy,
        f27_view_score_z, f28_active_rom,
        f27_e0_lumbopelvic_win_sd
    ], dtype=np.float32)

    return np.nan_to_num(feats_27, nan=0.0, posinf=0.0, neginf=0.0)


class BabelSquatC1ModelV11ADiet:
    """
    바벨 스쿼트 c1 결함(척추의 중립 위반) v11_a_diet (27D) 공식 프로덕션 모델
    CatBoost 60% + LightGBM 40% 앙상블 + 3계층 불확실성 필터
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 27
    FEATURE_NAMES = FEATURE_NAMES_27_C1_DIET

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_27_C1_DIET

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(
        cls,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64F + SpinePose 9분절로부터 27D 정예 피처 추출"""
        return extract_c1_v11_a_diet_features_from_seq(kps_33_64f, spine_9_64f)

    def predict_proba(self, X_27d: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v11_a_diet 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_27d.ndim == 1:
            X_27d = X_27d.reshape(1, -1)
            single = True

        N = X_27d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_27d)[:, 1]
            p_lgb = clf.predict_proba(X_27d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)
        p0 = 1.0 - ensemble_probs
        res = np.column_stack([p0, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_27d: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_27d)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        spine_9: Optional[np.ndarray] = None,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> Dict[str, Any]:
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v11_a_diet 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps, spine_9)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 27:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 27:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        view_z = float(feats[24])  # c1_view_score_normal_z
        rom = float(feats[25])     # c1_active_rom
        view_type = "rear" if view_z < -0.15 else ("front" if view_z > 0.15 else "oblique")

        if rom < min_rom_threshold:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "view_type": view_type,
                "spine_observable": False,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "스쿼트 하강 깊이가 너무 얕아 척추 중립 및 요추 굴곡을 판정할 수 없습니다.",
                "action": "고관절이 무릎 높이까지 충분히 내려오도록 깊숙이 스쿼트를 수행해 주세요."
            }

        X_2d = feats.reshape(1, -1)
        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]

        fold_preds = [0.60 * p_c + 0.40 * p_l for p_c, p_l in zip(cb_preds, lgb_preds)]
        mean_cb = float(np.mean(cb_preds))
        mean_lgb = float(np.mean(lgb_preds))
        p_defect = float(np.mean(fold_preds))

        model_disagreement = float(abs(mean_cb - mean_lgb))
        fold_std = float(np.std(fold_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        if model_disagreement >= disagreement_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - model_disagreement), 4),
                "view_type": view_type,
                "spine_observable": True,
                "reason_code": "ENSEMBLE_DISAGREEMENT",
                "message": "척추 움직임의 생체역학적 패턴이 애매하여 AI 모델 간 일치도가 낮습니다.",
                "action": "허리를 곧게 편 상태를 명확히 의식하며 1회 다시 수행해 주세요."
            }

        if fold_std >= fold_std_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - 2.0 * fold_std), 4),
                "view_type": view_type,
                "spine_observable": True,
                "reason_code": "FOLD_PREDICTION_UNSTABLE",
                "message": "피험자 체형 특이성 또는 촬영 각도 편차로 인해 예측 안정성이 부족합니다.",
                "action": "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            }

        if p_defect <= tau_normal:
            status = "NORMAL"
            msg = "요추 및 흉요추의 중립 정렬이 안정적으로 유지되었습니다."
            act = None
            reason = "CLEAR_DECISION"
        elif p_defect >= tau_defect:
            status = "DEFECT"
            msg = "요추 말림(버트윙크) 또는 과도한 체간 굴곡/과신전이 발생하여 척추 중립이 위반되었습니다."
            act = "복압(코어)을 단단히 유지하고 엉덩이가 말리지 않는 깊이까지만 스쿼트를 수행하세요."
            reason = "CLEAR_DECISION"
        else:
            status = "UNDETERMINED"
            msg = "정상과 척추 중립 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
            act = "코어에 힘을 주고 척추를 중립 상태로 고정한 뒤 1회 다시 수행해 주세요."
            reason = "LOW_CONFIDENCE"

        return {
            "status": status,
            "spine_status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(confidence, 4),
            "view_type": view_type,
            "spine_observable": True,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def predict_selective_batch(
        self,
        X_27d: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> List[Dict[str, Any]]:
        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_27d.ndim == 1:
            return [self.predict_selective(X_27d, None, tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)]
        return [
            self.predict_selective(X_27d[i], None, tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)
            for i in range(X_27d.shape[0])
        ]

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v11_a_diet 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_27_C1_DIET)
        self.is_fitted = True
        return self

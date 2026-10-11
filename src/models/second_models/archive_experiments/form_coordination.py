# -*- coding: utf-8 -*-
"""
NovelFlow: Intra-Exercise Form Quality & Defect Coordination Feature Extractor
- 동일 운동 내(Intra-Exercise) 정상/오류 및 4대 세부 결함(c1~c4) 분별 특화
- 스쿼트 가동범위(ROM) 및 절대 각도 스냅샷에 불변인 16대 동적 협응 피처
- 학술 근거: Sports Biomechanics (CRP, Vector Coding, Cyclograms, Jerk Metrics)
"""

import numpy as np
from typing import List, Tuple

FEATURE_NAMES_TREND_16: List[str] = [
    # [c1] 척추의 중립 결함 지표
    'c1_cyclogram_hysteresis_norm',    # 무차원 체간-고관절 폐곡선 면적 (Hysteresis)
    'c1_rmse_hysteresis_trunk',        # 동일 무릎각 대비 상승-하강 체간 편차 RMSE
    'c1_marp_trunk_hip',               # 체간-고관절 평균 연속상대위상 (MARP)
    'c1_dsj_torso_pitch',              # 체간 피치 무차원 저크 (Dimensionless Squared Jerk)
    # [c2] 고개 정면 결함 지표
    'c2_neck_drift_auc',               # 목-척추 정렬 누적 드리프트 AUC
    'c2_neck_drift_slope',             # 목-척추 정렬 진행도 회귀 기울기 (beta_1)
    'c2_bottom_cervical_ext_ratio',    # 최하사점 직후 경추 과신전 속도비
    # [c3] 발과 무릎의 방향 일치 결함 지표
    'c3_valgus_collapse_rate_down',    # 하강기 무릎 굴곡 대비 외반 속도비
    'c3_valgus_speed_ratio_total',     # 전체 굴곡 대비 외반 속도비
    'c3_knee_frontal_tortuosity',      # 전두면 무릎 궤적 비틀림 (곡률)
    'c3_dsj_knee_valgus',              # 무릎 외반 무차원 저크 (DSJ)
    # [c4] 발바닥 지면 고정 결함 지표
    'c4_ankle_drift_auc',              # 발목 3D 미세 드리프트 AUC (스탠스 폭 정규화)
    'c4_bottom_vertical_jerk',         # 최하사점 수직 가속도 반환 저크 (바운싱)
    # [공통] 시간 대칭성 및 역동역학 일치도
    'sym_temporal_phase_ratio',        # 하강/상승 시간 대칭비 (T_ecc / T_con)
    'sym_ecc_con_cross_corr',          # 하강-상승 반전 궤적 상호상관 계수
    'rom_hip_knee_ratio_trend'         # 고관절/무릎 상대 가동범위 비
]

def extract_intra_exercise_trend_features(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, T=16, C=92) -> (N, 16) 운동 내 자세 품질 및 결함 협응 경향성 피처 산출
    """
    N, T, C = X_b.shape
    feats = np.zeros((N, 16), dtype=np.float32)
    tau = np.linspace(0.0, 1.0, T)
    dt = 1.0 / max(1, T)

    for i in range(N):
        # 1. 주요 관절각 시계열
        knee_flex = 180.0 - (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0  # 굴곡량 (클수록 깊게 굽힘)
        hip_flex = 180.0 - (X_b[i, :, 58] + X_b[i, :, 59]) / 2.0
        torso_pitch = X_b[i, :, 64]
        valgus = (X_b[i, :, 66] + X_b[i, :, 67]) / 2.0
        neck_align = X_b[i, :, 82]
        stance_width = np.mean(X_b[i, :, 70]) + 1e-5

        # 2. 최하사점(변곡점) 인덱스
        b = int(np.argmax(knee_flex))
        b = int(np.clip(b, 1, T - 2))

        # ----------------------------------------------------
        # [c1] 척추의 중립 결함 지표
        # ----------------------------------------------------
        # A. 정규화 Cyclogram 면적 (Shoelace 무차원 적분)
        k_ptp = np.ptp(knee_flex) + 1e-5
        h_ptp = np.ptp(hip_flex) + 1e-5
        k_norm = (knee_flex - np.min(knee_flex)) / k_ptp
        h_norm = (hip_flex - np.min(hip_flex)) / h_ptp
        shoelace = 0.5 * np.abs(
            np.sum(k_norm[:-1] * h_norm[1:] - k_norm[1:] * h_norm[:-1]) +
            (k_norm[-1] * h_norm[0] - k_norm[0] * h_norm[-1])
        )
        feats[i, 0] = shoelace

        # B. RMSE Hysteresis (하강-상승 동일 무릎각 대비 체간각 오차)
        try:
            m_depths = np.linspace(np.min(knee_flex) + 1.0, np.max(knee_flex) - 1.0, 10)
            trunk_down = np.interp(m_depths, knee_flex[:b+1], torso_pitch[:b+1])
            trunk_up = np.interp(m_depths, knee_flex[b:][::-1], torso_pitch[b:][::-1])
            feats[i, 1] = np.sqrt(np.mean((trunk_up - trunk_down)**2))
        except Exception:
            feats[i, 1] = 0.0

        # C. MARP (체간-고관절 CRP)
        v_torso = np.gradient(torso_pitch, dt)
        v_hip = np.gradient(hip_flex, dt)
        t_ptp = np.ptp(torso_pitch) + 1e-5
        phi_t = np.arctan2(v_torso / (np.max(np.abs(v_torso)) + 1e-5), 2.0 * (torso_pitch - np.min(torso_pitch)) / t_ptp - 1.0)
        phi_h = np.arctan2(v_hip / (np.max(np.abs(v_hip)) + 1e-5), 2.0 * (hip_flex - np.min(hip_flex)) / h_ptp - 1.0)
        crp_th = np.abs(np.arctan2(np.sin(phi_t - phi_h), np.cos(phi_t - phi_h)))
        feats[i, 2] = np.degrees(np.mean(crp_th))

        # D. DSJ Torso Pitch (Dimensionless Squared Jerk)
        jerk_t = np.gradient(np.gradient(v_torso, dt), dt)
        feats[i, 3] = np.log1p((1.0 / (t_ptp**2 + 1e-3)) * np.sum(jerk_t**2) * dt)

        # ----------------------------------------------------
        # [c2] 고개 정면 결함 지표
        # ----------------------------------------------------
        e_neck = np.abs(neck_align - neck_align[0])
        feats[i, 4] = np.mean(e_neck)  # AUC
        try:
            feats[i, 5] = np.polyfit(tau, e_neck, 1)[0]  # Slope (beta_1)
        except Exception:
            feats[i, 5] = 0.0

        # 최하사점 직후 경추 과신전비
        b_end = min(T, b + int(0.20 * T) + 1)
        v_neck = np.gradient(neck_align, dt)
        feats[i, 6] = np.sum(np.maximum(0, v_neck[b:b_end])) / (np.sum(np.abs(v_hip[b:b_end])) + 1e-5)

        # ----------------------------------------------------
        # [c3] 발과 무릎의 방향 일치 결함 지표
        # ----------------------------------------------------
        v_valgus = np.gradient(valgus, dt)
        v_knee = np.gradient(knee_flex, dt)
        feats[i, 7] = np.sum(np.maximum(0, v_valgus[:b])) / (np.sum(np.abs(v_knee[:b])) + 1e-5)
        feats[i, 8] = np.sum(np.abs(v_valgus)) / (np.sum(np.abs(v_knee)) + 1e-5)

        # 무릎 전두면 비틀림 (Tortuosity)
        kd = X_b[i, :, 69]
        kd_curve_len = np.sum(np.sqrt(np.diff(kd)**2 + np.diff(knee_flex)**2))
        kd_chord_len = np.sqrt((kd[b] - kd[0])**2 + (knee_flex[b] - knee_flex[0])**2) + 1e-5
        feats[i, 9] = kd_curve_len / kd_chord_len

        # 무릎 외반 DSJ
        valgus_ptp = np.ptp(valgus) + 1e-5
        jerk_v = np.gradient(np.gradient(v_valgus, dt), dt)
        feats[i, 10] = np.log1p((1.0 / (valgus_ptp**2 + 1e-3)) * np.sum(jerk_v**2) * dt)

        # ----------------------------------------------------
        # [c4] 발바닥 지면 고정 결함 지표
        # ----------------------------------------------------
        dx_ank = np.gradient(X_b[i, :, 87], dt)
        dz_ank = np.gradient(X_b[i, :, 88], dt)
        v_ank = np.sqrt(dx_ank**2 + dz_ank**2) / stance_width
        feats[i, 11] = np.mean(v_ank)

        # 최하사점 수직 가속도 반환 저크 (바운싱)
        acc_knee = np.gradient(v_knee, dt)
        feats[i, 12] = np.abs(acc_knee[min(T-1, b+1)] - acc_knee[max(0, b-1)]) / (2 * dt)

        # ----------------------------------------------------
        # [공통] 시간 대칭성 및 편심-동심 일치도
        # ----------------------------------------------------
        feats[i, 13] = float(b) / max(1.0, float(T - b))  # T_ecc / T_con
        
        # 하강과 상승 궤적 상호상관 계수
        try:
            k_down = np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, b+1), k_norm[:b+1])
            k_up = np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, T-b), k_norm[b:][::-1])
            corr = np.corrcoef(k_down, k_up)[0, 1]
            feats[i, 14] = float(np.nan_to_num(corr, nan=1.0))
        except Exception:
            feats[i, 14] = 1.0

        # 고관절/무릎 ROM 비율
        feats[i, 15] = h_ptp / k_ptp

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats

def build_trend_augmented_features(X_base: np.ndarray, X_b: np.ndarray, base_feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """기존 피처(161D 또는 169D) + 16대 자세 협응 경향성 피처 결합"""
    feats_16 = extract_intra_exercise_trend_features(X_b)
    X_aug = np.concatenate([X_base, feats_16], axis=-1).astype(np.float32)
    if base_feature_names is not None:
        names = list(base_feature_names) + FEATURE_NAMES_TREND_16
    else:
        names = [f"f_{i}" for i in range(X_base.shape[1])] + FEATURE_NAMES_TREND_16
    return X_aug, names

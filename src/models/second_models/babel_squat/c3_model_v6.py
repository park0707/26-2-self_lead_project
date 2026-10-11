# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c3. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Model Version: v6.0 (c3_model_v6.py)
Status: ★ 순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델 v6.0
        - 1) 골반 관상면(Coronal) 로컬 정규 직교 기저계(Body-Local Orthonormal Basis) 투영
        - 2) 관상면 절대 무릎-발목 너비 비율(Absolute Knee-to-Ankle Ratio) 6D 직접 계측
        - 3) 좌우 독립 외반 변위 및 절댓값 합(Bilateral Sum), 동시 붕괴 강도 6D 반영
        - 4) 3D 대퇴-경골 관상면 각도 및 무릎-발끝(Knee-to-Toe) 3차원 정렬 오차 8D 결합
        - 5) 과거 레거시 Format B 텐서 슬라이싱(kd=X_b[:, 69] 등) 100% 영구 제거
        - 6) 3-State 선택적 판정(Balanced tau_normal=0.40, tau_defect=0.60) 탑재
        - 7) 마스터 데이터셋 2,605건 전수(5대 전 시점 x 9명 피험자) 5-Fold 피험자 완전 격리
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

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

FEATURE_NAMES_22_C3: List[str] = [
    # [1~6] 관상면 절대 무릎-발목 너비 비율 (Knee-to-Ankle Ratio 6D)
    "c3_ka_ratio_nadir_abs",              # [1] 최하사점 절대 무릎/발목 너비 비율
    "c3_ka_ratio_min_abs",                # [2] 전 구간 최소 무릎/발목 너비 비율
    "c3_ka_ratio_standing_abs",           # [3] 기립 자세 무릎/발목 너비 비율
    "c3_ka_valgus_duration_ratio",        # [4] 무릎/발목 비율 < 0.85 외반 지속 시간비
    "c3_ka_absolute_deficit_integ",       # [5] 정상기준(0.90) 대비 무릎 폭 부족분 적분값
    "c3_ka_chronic_standing_valgus",      # [6] 기립 시 이미 외반 상태인 만성 플래그 (1.0 / 0.0)

    # [7~12] 골반 관상면(x_hat) 투영 좌우 독립 외반 및 절댓값 합 (6D)
    "c3_valgus_abs_sum_nadir",            # [7] 최하사점 좌우 절대 외반합 (|vl| + |vr|, cm)
    "c3_valgus_abs_sum_max",              # [8] 전 구간 최대 좌우 절대 외반합 (cm)
    "c3_valgus_concurrence_max",          # [9] 양 무릎 동시 내측 붕괴 강도 (cm)
    "c3_valgus_bilateral_min",            # [10] 좌우 중 최소 외반값 (양측 동시성, cm)
    "c3_valgus_asymmetry_p90",            # [11] 좌우 외반 절대 편차 |vl - vr|의 90th percentile (cm)
    "c3_valgus_velocity_peak",            # [12] 무릎 내측 붕괴 속도 피크 (|cm/s|)

    # [13~16] 3D 대퇴골-경골 3차원 관상면 외반각 (4D)
    "c3_3d_femur_tibia_angle_min",        # [13] 양 무릎 중 최소 대퇴-경골 관상각 (deg)
    "c3_3d_femur_tibia_drop_nadir",       # [14] 기립 대비 최하사점 대퇴-경골 관상각 낙폭 (deg)
    "c3_3d_knee_inward_angle_max",        # [15] 안쪽 꺾임 최대 각도 (180 - min_angle, deg)
    "c3_3d_knee_angular_jerk",            # [16] 최하사점 부근 무릎 관상각 2차 미분 각가속도 급변도

    # [17~20] 무릎-발끝(Knee-to-Toe) 3D 정렬 오차 (4D)
    "c3_knee_toe_alignment_err_nadir",    # [17] 최하사점 무릎-발끝 정렬 내측 이탈 거리 (cm)
    "c3_knee_toe_alignment_err_max",      # [18] 전 구간 최대 무릎-발끝 내측 이탈 거리 (cm)
    "c3_foot_progression_mismatch",       # [19] 발끝 진행각과 무릎 진행각 간의 불일치각 (deg)
    "c3_knee_lateral_wobble_std",         # [20] 전 구간 무릎 수평 궤적 떨림/불안정성 표준편차 (cm)

    # [21~22] 품질 및 시점 메타 피처 (2D)
    "c3_view_score_normal_z",             # [21] 골반 법선 벡터의 카메라 Z축 연속 성분 (시점 지표)
    "c3_active_rom"                       # [22] 스쿼트 수직 가동범위 (골반 최고점 - 최저점 변위, m)
]


def extract_c3_v6_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) MediaPipe 3D 월드 좌표 시퀀스로부터 c3 v6.0 22D 생체역학 피처 추출
    """
    T = kps_33_64f.shape[0] # 64
    dt = 3.0 / max(1, T - 1)

    # MediaPipe 주요 관절
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

    l_heel = kps_33_64f[:, 29, :]
    r_heel = kps_33_64f[:, 30, :]

    l_toe = kps_33_64f[:, 31, :]
    r_toe = kps_33_64f[:, 32, :]

    # 1. 골반 및 체간 로컬 정규 직교 기저계 (Orthonormal Basis)
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_hip = l_hip - r_hip
    z_raw = np.cross(y_hat, v_hip)
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

    # 능동 구간 마스크 (가동범위 하위 80% 구간)
    deep_thresh = np.min(pelvis_y_s) + 0.20 * active_rom
    active_mask = (pelvis_y_s >= deep_thresh)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # =========================================================================
    # [1~6] 관상면 절대 무릎-발목 너비 비율 (Knee-to-Ankle Ratio 6D)
    # =========================================================================
    knee_width = np.linalg.norm(l_knee - r_knee, axis=1)
    ankle_width = np.linalg.norm(l_ank - r_ank, axis=1)
    ka_ratio = knee_width / (ankle_width + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)

    f1_ka_nadir = float(ka_ratio_s[b])
    f2_ka_min = float(np.min(ka_ratio_s[active_mask]))
    f3_ka_standing = float(np.mean(ka_ratio_s[:standing_slice]))
    f4_valg_dur_ratio = float(np.mean(ka_ratio_s[active_mask] < 0.85))
    f5_abs_deficit_integ = float(np.sum(np.maximum(0.0, 0.90 - ka_ratio_s[active_mask])))
    f6_chronic_valgus = 1.0 if f3_ka_standing < 0.85 else 0.0

    # =========================================================================
    # [7~12] 골반 관상면(x_hat) 투영 좌우 독립 외반 및 절댓값 합 (6D)
    # =========================================================================
    # 골반 대비 무릎의 상대적 횡방향 변위 (x_hat 기준: 외측 +, 내측 -)
    diff_l = l_knee - l_hip
    diff_r = r_hip - r_knee
    vl = np.sum(diff_l * x_hat, axis=1) * 100.0 # cm 단위
    vr = np.sum(diff_r * x_hat, axis=1) * 100.0 # cm 단위
    vl_s = gaussian_filter1d(vl, sigma=1.0)
    vr_s = gaussian_filter1d(vr, sigma=1.0)

    # 내측 편위(외반) 발생 시 음수 -> -vl, -vr가 클수록 심한 외반
    valg_l = np.maximum(0.0, -vl_s)
    valg_r = np.maximum(0.0, -vr_s)
    valgus_abs_sum = valg_l + valg_r

    f7_valgus_sum_nadir = float(valgus_abs_sum[b])
    f8_valgus_sum_max = float(np.max(valgus_abs_sum[active_mask]))

    # 동시 붕괴 강도 (양 무릎이 둘 다 안으로 모일 때)
    concurrence = np.sqrt(valg_l * valg_r)
    f9_concurrence_max = float(np.max(concurrence[active_mask]))

    # 좌우 최소 외반
    f10_bilateral_min = float(min(valg_l[b], valg_r[b]))

    # 좌우 비대칭도 (편측 붕괴)
    asym_series = np.abs(vl_s - vr_s)
    f11_asym_p90 = float(np.percentile(asym_series, 90))

    # 무릎 내측 붕괴 각속도/선속도 피크
    valg_vel = np.gradient(valgus_abs_sum, dt)
    f12_valg_vel_peak = float(np.max(np.abs(valg_vel[max(0, b - 4):min(T, b + 5)])))

    # =========================================================================
    # [13~16] 3D 대퇴골-경골 3차원 관상면 외반각 (4D)
    # =========================================================================
    # 대퇴 벡터 (hip -> knee) 및 경골 벡터 (knee -> ankle)
    v_femur_l = l_knee - l_hip
    v_tibia_l = l_ank - l_knee
    cos_l = np.sum(v_femur_l * v_tibia_l, axis=1) / (
        np.linalg.norm(v_femur_l, axis=1) * np.linalg.norm(v_tibia_l, axis=1) + 1e-7
    )
    angle_l = np.degrees(np.arccos(np.clip(cos_l, -1.0, 1.0)))

    v_femur_r = r_knee - r_hip
    v_tibia_r = r_ank - r_knee
    cos_r = np.sum(v_femur_r * v_tibia_r, axis=1) / (
        np.linalg.norm(v_femur_r, axis=1) * np.linalg.norm(v_tibia_r, axis=1) + 1e-7
    )
    angle_r = np.degrees(np.arccos(np.clip(cos_r, -1.0, 1.0)))

    knee_angle_min_series = np.minimum(angle_l, angle_r)
    knee_angle_s = gaussian_filter1d(knee_angle_min_series, sigma=1.0)
    angle_standing = float(np.mean(knee_angle_s[:standing_slice]))

    f13_angle_min = float(np.min(knee_angle_s[active_mask]))
    f14_angle_drop = float(max(0.0, angle_standing - knee_angle_s[b]))
    f15_inward_max = float(max(0.0, 180.0 - f13_angle_min))

    d_ang = np.gradient(knee_angle_s, dt)
    d2_ang = np.gradient(d_ang, dt)
    f16_angular_jerk = float(np.max(np.abs(d2_ang[max(0, b - 3):min(T, b + 4)])) / 100.0)

    # =========================================================================
    # [17~20] 무릎-발끝(Knee-to-Toe) 3D 정렬 오차 (4D)
    # =========================================================================
    # 발 진행선: heel -> toe
    v_foot_l = l_toe - l_heel
    v_foot_r = r_toe - r_heel

    # 발 진행선 단위벡터
    norm_foot_l = v_foot_l / (np.linalg.norm(v_foot_l, axis=1, keepdims=True) + 1e-7)
    norm_foot_r = v_foot_r / (np.linalg.norm(v_foot_r, axis=1, keepdims=True) + 1e-7)

    # 발뒤꿈치 기준 무릎의 위치
    rel_k_l = l_knee - l_heel
    rel_k_r = r_knee - r_heel

    # 발 진행선에 직교하는 내측 편차 (cross product z or x_hat 내적)
    cross_foot_l = np.cross(norm_foot_l, rel_k_l)
    cross_foot_r = np.cross(norm_foot_r, rel_k_r)
    err_toe_l = np.sum(cross_foot_l * y_hat, axis=1) * 100.0 # cm
    err_toe_r = np.sum(cross_foot_r * y_hat, axis=1) * 100.0 # cm

    err_toe_sum = np.abs(err_toe_l) + np.abs(err_toe_r)
    err_toe_s = gaussian_filter1d(err_toe_sum, sigma=1.0)

    f17_toe_err_nadir = float(err_toe_s[b])
    f18_toe_err_max = float(np.max(err_toe_s[active_mask]))

    # 발끝 벌림각 vs 무릎 벌림각 불일치
    foot_dist = np.linalg.norm(l_toe - r_toe, axis=1)
    progression_ratio = foot_dist / (knee_width + 1e-6)
    f19_progression_mismatch = float(abs(progression_ratio[b] - 1.0) * 100.0)

    # 무릎 수평 궤적 떨림
    f20_wobble_std = float(np.std(knee_width)) * 100.0

    # =========================================================================
    # [21~22] 품질 및 시점 메타 피처 (2D)
    # =========================================================================
    f21_view_score_z = normal_z_mean
    f22_active_rom = active_rom

    feats = np.array([
        f1_ka_nadir, f2_ka_min, f3_ka_standing, f4_valg_dur_ratio, f5_abs_deficit_integ, f6_chronic_valgus,
        f7_valgus_sum_nadir, f8_valgus_sum_max, f9_concurrence_max, f10_bilateral_min, f11_asym_p90, f12_valg_vel_peak,
        f13_angle_min, f14_angle_drop, f15_inward_max, f16_angular_jerk,
        f17_toe_err_nadir, f18_toe_err_max, f19_progression_mismatch, f20_wobble_std,
        f21_view_score_z, f22_active_rom
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)


class BabelSquatC3ModelV6:
    """
    바벨 스쿼트 c3 결함(무릎-발 방향 일치 / 무릎 외반) 공식 프로덕션 모델 v6.0 (22D)
    3-State 선택적 거부(Balanced: tau_normal 0.40, tau_defect 0.60) 탑재
    """

    DEFECT_CODE = "c3"
    DEFECT_NAME = "발과 무릎의 방향 일치"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 22
    FEATURE_NAMES = FEATURE_NAMES_22_C3

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
        self.feature_names = FEATURE_NAMES_22_C3

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c3_model_v6_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64F 시퀀스로부터 22D 피처 추출"""
        if kps_33_64f.ndim == 3 and kps_33_64f.shape[1] == 33:
            return extract_c3_v6_features_from_seq(kps_33_64f)
        elif kps_33_64f.ndim == 4:
            N = kps_33_64f.shape[0]
            out = np.zeros((N, 22), dtype=np.float32)
            for i in range(N):
                out[i] = extract_c3_v6_features_from_seq(kps_33_64f[i])
            return out
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {kps_33_64f.shape}")

    def predict_proba(self, X_22d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 22) 또는 (22,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C3 v6 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_22d.ndim == 1:
            X_22d = np.expand_dims(X_22d, axis=0)
            single = True

        N = X_22d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_22d)[:, 1]
            p_lgb = clf.predict_proba(X_22d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)
        p0 = 1.0 - ensemble_probs
        res = np.column_stack([p0, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_22d: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_22d)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> Dict[str, Any]:
        """
        c3 3-State 선택적 거부(Selective Abstention / Reject Option) 추론 메서드
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C3 v6 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 22:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 22:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        view_z = float(feats[20]) # c3_view_score_normal_z
        rom = float(feats[21])    # c3_active_rom
        view_type = "rear" if view_z < -0.15 else ("front" if view_z > 0.15 else "oblique")

        # [Tier 1] 가동범위 및 관측 품질 게이트 (Hard Quality Reject)
        if rom < min_rom_threshold:
            return {
                "status": "UNDETERMINED",
                "knee_status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "view_type": view_type,
                "knee_observable": False,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "스쿼트 하강 깊이가 너무 얕아 무릎 굴곡 및 외반 여부를 판정할 수 없습니다.",
                "action": "고관절이 무릎 높이까지 충분히 내려오도록 깊숙이 스쿼트를 수행해 주세요."
            }

        # [Tier 2] 개별 5-Fold 모델 추론 및 불일치/분산 검사
        X_2d = feats.reshape(1, -1)
        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]

        fold_preds = [0.60 * p_c + 0.40 * p_l for p_c, p_l in zip(cb_preds, lgb_preds)]
        mean_cb = float(np.mean(cb_preds))
        mean_lgb = float(np.mean(lgb_preds))
        p_defect = float(np.mean(fold_preds))

        model_disagreement = float(abs(mean_cb - mean_lgb))
        fold_std = float(np.std(fold_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5))) # 0.0 ~ 1.0

        # 2.1 앙상블 모델 간 극심한 불일치 검사
        if model_disagreement >= disagreement_tol:
            return {
                "status": "UNDETERMINED",
                "knee_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - model_disagreement), 4),
                "view_type": view_type,
                "knee_observable": True,
                "reason_code": "ENSEMBLE_DISAGREEMENT",
                "message": "무릎 움직임의 외반 패턴이 애매하여 AI 모델 간 판정 일치도가 낮습니다.",
                "action": "무릎을 발끝 방향으로 확실히 벌리며 1회 다시 수행해 주세요."
            }

        # 2.2 피험자 분할 Fold 모델 간 표준편차 과다 검사
        if fold_std >= fold_std_tol:
            return {
                "status": "UNDETERMINED",
                "knee_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - 2.0 * fold_std), 4),
                "view_type": view_type,
                "knee_observable": True,
                "reason_code": "FOLD_PREDICTION_UNSTABLE",
                "message": "피험자 체형 특이성 또는 촬영 각도로 인해 예측 안정성이 부족합니다.",
                "action": "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            }

        # [Tier 3] 이중 임계치(Double Threshold) 선택적 판정
        if p_defect <= tau_normal:
            status = "NORMAL"
            msg = "무릎과 발의 정렬이 정상적으로 유지되었습니다."
            act = None
            reason = "CLEAR_DECISION"
        elif p_defect >= tau_defect:
            status = "DEFECT"
            msg = "하강 또는 상승 시 무릎이 안쪽으로 모이거나 흔들리는 외반(Valgus)이 감지되었습니다."
            act = "스쿼트 시 무릎을 둘째 발가락 방향으로 바깥쪽으로 벌리며 동작을 수행하세요."
            reason = "CLEAR_DECISION"
        else:
            status = "UNDETERMINED"
            msg = "정상과 무릎 외반 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
            act = "무릎이 안으로 모이지 않도록 발끝 방향으로 명확히 벌리고 1회 다시 수행해 주세요."
            reason = "LOW_CONFIDENCE"

        return {
            "status": status,
            "knee_status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(confidence, 4),
            "view_type": view_type,
            "knee_observable": True,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def predict_selective_batch(
        self,
        X_22d: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> List[Dict[str, Any]]:
        """배치 (N, 22) 선택적 추론"""
        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_22d.ndim == 1:
            return [self.predict_selective(X_22d, tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)]
        return [
            self.predict_selective(X_22d[i], tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)
            for i in range(X_22d.shape[0])
        ]

    def save_weights(self, filepath: str):
        """가중치 저장"""
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
        print(f"[*] c3 v6.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_22_C3)
        self.is_fitted = True
        return self


BabelSquatC3Model = BabelSquatC3ModelV6

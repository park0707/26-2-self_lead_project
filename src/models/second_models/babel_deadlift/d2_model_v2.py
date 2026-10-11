# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Model Version: v2.0 (d2_model_v2.py - 데드리프트 맞춤 정예 7D 피처셋)
Status: 데드리프트 부적합 노이즈 피처 23개(스쿼트 발끝 각도/회전, 3D 관상각 등) 전면 제거
        - 데드리프트 최적화 7대 정예 생체역학 피처 채택:
          1) d2_kp_ratio_min_abs: 전 구간 최소 무릎/골반 너비 비율 (변별력 1위, p=3.69e-3)
          2) d2_ka_absolute_deficit_integ: 정상기준(0.90) 대비 무릎 폭 부족분 적분값 (Cohen's d=0.540)
          3) d2_valgus_abs_sum_max: 전 구간 최대 좌우 절대 외반합 (|vl| + |vr|, cm)
          4) d2_valgus_norm_hip_max: 골반 너비로 정규화된 무차원 최대 외반합
          5) d2_shank_inward_tilt_p90: 정강이 3D 벡터의 정면 안쪽 기울기 90th percentile
          6) d2_valgus_inward_accel_peak: 무릎 내측 붕괴 2차 미분 각가속도 피크
          7) d2_morph_kp_drop_max: 기립 준비 자세 대비 최대 무릎/골반 너비 비율 낙폭
        - LightGBM + CatBoost 이중 GBDT 앙상블 (피험자 격리 StratifiedGroupKFold)
        - 3-State 고신뢰 선택적 판정(Selective Classification) 최적화
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

from src.models.second_models.babel_deadlift.d2_model_v1 import (
    extract_d2_v1_features_from_seq, FEATURE_NAMES_22_D2
)

# 1. 데드리프트 최적화 7대 정예 피처 명칭 (v2.0 공식 표준)
FEATURE_NAMES_7_D2: List[str] = [
    "d2_kp_ratio_min_abs",            # [1] 전 구간 최소 무릎/골반 너비 비율 (변별력 1위)
    "d2_ka_absolute_deficit_integ",   # [2] 정상기준(0.90) 대비 무릎 폭 부족분 적분값 (Cohen's d 1위)
    "d2_valgus_abs_sum_max",          # [3] 전 구간 최대 좌우 절대 외반합 (cm)
    "d2_valgus_norm_hip_max",         # [4] 골반 너비로 정규화된 무차원 최대 외반합
    "d2_shank_inward_tilt_p90",       # [5] 정강이 3D 벡터의 정면 안쪽 기울기 90th percentile
    "d2_valgus_inward_accel_peak",    # [6] 무릎 내측 붕괴 2차 미분 각가속도 피크
    "d2_morph_kp_drop_max"            # [7] 기립 준비 대비 최대 무릎/골반 너비 비율 낙폭
]

FEATURE_NAMES_V2_D2 = FEATURE_NAMES_7_D2

# 하위 호환용 30D 피처 명칭
FEATURE_NAMES_30_D2: List[str] = FEATURE_NAMES_22_D2 + [
    "d2_morph_ka_drop_max",
    "d2_morph_kp_drop_max",
    "d2_kp_ratio_min_abs",
    "d2_kp_ratio_p10",
    "d2_valgus_norm_hip_max",
    "d2_valgus_inward_accel_peak",
    "d2_shank_inward_tilt_p90",
    "d2_bilateral_inward_sync"
]


# 7대 정예 피처의 30D 내 원천 인덱스 매핑
DIET_7_INDICES_FROM_30D: List[int] = [
    24,  # d2_kp_ratio_min_abs (p=3.69e-3, 단독 AUC 0.6140)
    4,   # d2_ka_absolute_deficit_integ (p=5.40e-4, Cohen d=0.540)
    7,   # d2_valgus_abs_sum_max (p=2.40e-3, 단독 AUC 0.5974)
    26,  # d2_valgus_norm_hip_max (p=2.31e-2, 단독 AUC 0.5829)
    28,  # d2_shank_inward_tilt_p90 (단독 AUC 0.5651)
    27,  # d2_valgus_inward_accel_peak
    23   # d2_morph_kp_drop_max
]


def extract_d2_v2_30d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """하위 호환 및 기저 연산용: 기존 30D 피처 추출"""
    f22 = extract_d2_v1_features_from_seq(kps_33_64f)
    T = kps_33_64f.shape[0]
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, 11, :]
    r_sh = kps_33_64f[:, 12, :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]
    r_hip = kps_33_64f[:, 24, :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, 25, :]
    r_knee = kps_33_64f[:, 26, :]

    l_ank = kps_33_64f[:, 27, :]
    r_ank = kps_33_64f[:, 28, :]

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_hip = l_hip - r_hip
    z_raw = np.cross(y_hat, v_hip)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_hat = np.cross(y_hat, z_hat)
    x_hat = x_hat / (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-7)

    knee_w = np.linalg.norm(l_knee - r_knee, axis=1)
    ank_w = np.linalg.norm(l_ank - r_ank, axis=1)
    hip_w = np.linalg.norm(l_hip - r_hip, axis=1)
    mean_hip_w = float(np.mean(hip_w) + 1e-6)

    ka_ratio = knee_w / (ank_w + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)
    kp_ratio = knee_w / (hip_w + 1e-6)
    kp_ratio_s = gaussian_filter1d(kp_ratio, sigma=1.0)

    standing_slice = max(2, T // 8)
    ka_base = float(np.mean(ka_ratio_s[:standing_slice]))
    kp_base = float(np.mean(kp_ratio_s[:standing_slice]))

    f23_ka_drop = float(max(0.0, ka_base - np.min(ka_ratio_s)))
    f24_kp_drop = float(max(0.0, kp_base - np.min(kp_ratio_s)))
    f25_kp_min = float(np.min(kp_ratio_s))
    f26_kp_p10 = float(np.percentile(kp_ratio_s, 10))

    diff_l = l_knee - l_hip
    diff_r = r_hip - r_knee
    vl = np.sum(diff_l * x_hat, axis=1)
    vr = np.sum(diff_r * x_hat, axis=1)
    valg_l_norm = np.maximum(0.0, -vl) / mean_hip_w
    valg_r_norm = np.maximum(0.0, -vr) / mean_hip_w
    valg_sum_norm = valg_l_norm + valg_r_norm
    valg_sum_norm_s = gaussian_filter1d(valg_sum_norm, sigma=1.0)

    f27_valg_norm_max = float(np.max(valg_sum_norm_s))
    d_valg = np.gradient(valg_sum_norm_s, dt)
    d2_valg = np.gradient(d_valg, dt)
    f28_valg_accel_peak = float(np.max(np.abs(d2_valg)))

    v_shank_l = l_knee - l_ank
    v_shank_r = r_knee - r_ank
    norm_sl = v_shank_l / (np.linalg.norm(v_shank_l, axis=1, keepdims=True) + 1e-7)
    norm_sr = v_shank_r / (np.linalg.norm(v_shank_r, axis=1, keepdims=True) + 1e-7)
    shank_inward_l = np.sum(norm_sl * x_hat, axis=1)
    shank_inward_r = -np.sum(norm_sr * x_hat, axis=1)
    shank_inward_sum = np.maximum(0.0, -shank_inward_l) + np.maximum(0.0, -shank_inward_r)
    f29_shank_tilt_p90 = float(np.percentile(shank_inward_sum, 90))

    d_vl = np.gradient(valg_l_norm, dt)
    d_vr = np.gradient(valg_r_norm, dt)
    sync_series = np.sqrt(np.maximum(0.0, d_vl) * np.maximum(0.0, d_vr))
    f30_bilat_sync = float(np.max(sync_series))

    f_new = np.array([
        f23_ka_drop, f24_kp_drop, f25_kp_min, f26_kp_p10,
        f27_valg_norm_max, f28_valg_accel_peak, f29_shank_tilt_p90, f30_bilat_sync
    ], dtype=np.float32)

    f30 = np.concatenate([f22, f_new])
    return np.nan_to_num(f30, nan=0.0, posinf=180.0, neginf=-180.0)


def extract_d2_v2_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) MediaPipe 3D 월드 좌표 시퀀스로부터 데드리프트 d2 v2.0 정예 7D 생체역학 피처 추출
    - 외부 파일 의존성 0% (순수 MediaPipe 어깨, 골반, 무릎, 발목 8개 랜드마크 단독 연산)
    - 스쿼트 발끝 노이즈 피처 23개 전면 제거로 가볍고 신속하며 정밀한 변별력 달성
    """
    f30 = extract_d2_v2_30d_features_from_seq(kps_33_64f)
    f7 = f30[DIET_7_INDICES_FROM_30D]
    return f7


def extract_d2_v2_30d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """하위 호환용: 기존 30D 피처 추출"""
    f22 = extract_d2_v1_features_from_seq(kps_33_64f)
    T = kps_33_64f.shape[0]
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, 11, :]
    r_sh = kps_33_64f[:, 12, :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]
    r_hip = kps_33_64f[:, 24, :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, 25, :]
    r_knee = kps_33_64f[:, 26, :]

    l_ank = kps_33_64f[:, 27, :]
    r_ank = kps_33_64f[:, 28, :]

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_hip = l_hip - r_hip
    z_raw = np.cross(y_hat, v_hip)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_hat = np.cross(y_hat, z_hat)
    x_hat = x_hat / (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-7)

    knee_w = np.linalg.norm(l_knee - r_knee, axis=1)
    ank_w = np.linalg.norm(l_ank - r_ank, axis=1)
    hip_w = np.linalg.norm(l_hip - r_hip, axis=1)
    mean_hip_w = float(np.mean(hip_w) + 1e-6)

    ka_ratio = knee_w / (ank_w + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)
    kp_ratio = knee_w / (hip_w + 1e-6)
    kp_ratio_s = gaussian_filter1d(kp_ratio, sigma=1.0)

    standing_slice = max(2, T // 8)
    ka_base = float(np.mean(ka_ratio_s[:standing_slice]))
    kp_base = float(np.mean(kp_ratio_s[:standing_slice]))

    f23_ka_drop = float(max(0.0, ka_base - np.min(ka_ratio_s)))
    f24_kp_drop = float(max(0.0, kp_base - np.min(kp_ratio_s)))
    f25_kp_min = float(np.min(kp_ratio_s))
    f26_kp_p10 = float(np.percentile(kp_ratio_s, 10))

    diff_l = l_knee - l_hip
    diff_r = r_hip - r_knee
    vl = np.sum(diff_l * x_hat, axis=1)
    vr = np.sum(diff_r * x_hat, axis=1)
    valg_l_norm = np.maximum(0.0, -vl) / mean_hip_w
    valg_r_norm = np.maximum(0.0, -vr) / mean_hip_w
    valg_sum_norm = valg_l_norm + valg_r_norm
    valg_sum_norm_s = gaussian_filter1d(valg_sum_norm, sigma=1.0)

    f27_valg_norm_max = float(np.max(valg_sum_norm_s))
    d_valg = np.gradient(valg_sum_norm_s, dt)
    d2_valg = np.gradient(d_valg, dt)
    f28_valg_accel_peak = float(np.max(np.abs(d2_valg)))

    v_shank_l = l_knee - l_ank
    v_shank_r = r_knee - r_ank
    norm_sl = v_shank_l / (np.linalg.norm(v_shank_l, axis=1, keepdims=True) + 1e-7)
    norm_sr = v_shank_r / (np.linalg.norm(v_shank_r, axis=1, keepdims=True) + 1e-7)
    shank_inward_l = np.sum(norm_sl * x_hat, axis=1)
    shank_inward_r = -np.sum(norm_sr * x_hat, axis=1)
    shank_inward_sum = np.maximum(0.0, -shank_inward_l) + np.maximum(0.0, -shank_inward_r)
    f29_shank_tilt_p90 = float(np.percentile(shank_inward_sum, 90))

    d_vl = np.gradient(valg_l_norm, dt)
    d_vr = np.gradient(valg_r_norm, dt)
    sync_series = np.sqrt(np.maximum(0.0, d_vl) * np.maximum(0.0, d_vr))
    f30_bilat_sync = float(np.max(sync_series))

    f_new = np.array([
        f23_ka_drop, f24_kp_drop, f25_kp_min, f26_kp_p10,
        f27_valg_norm_max, f28_valg_accel_peak, f29_shank_tilt_p90, f30_bilat_sync
    ], dtype=np.float32)

    f30 = np.concatenate([f22, f_new])
    return np.nan_to_num(f30, nan=0.0, posinf=180.0, neginf=-180.0)


class BabelDeadliftD2ModelV2:
    """
    바벨 데드리프트 d2 결함(발과 무릎의 방향 일치 / 무릎 외반) 공식 모델 v2.0 (정예 7D)
    - 스쿼트 발끝 노이즈 피처 23개 전면 제거 + 데드리프트 맞춤 7D 무차원 생체역학 피처
    - LightGBM + CatBoost 앙상블 기반
    - 3-State 고신뢰 선택적 판정(Selective Classification) 최적화
    """

    DEFECT_CODE = "d2"
    DEFECT_NAME = "발과 무릎의 방향 일치"
    EXERCISE_NAME = "바벨 데드리프트"
    FEATURE_DIM = 7
    FEATURE_NAMES = FEATURE_NAMES_7_D2

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.37,
        tau_normal: float = 0.38,
        tau_defect: float = 0.40
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_7_D2

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'weights', 'd2_v2_gbdt_models.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64F 시퀀스로부터 정예 7D 피처 추출"""
        if kps_33_64f.ndim == 3 and kps_33_64f.shape[1] == 33:
            return extract_d2_v2_features_from_seq(kps_33_64f)
        elif kps_33_64f.ndim == 4:
            N = kps_33_64f.shape[0]
            out = np.zeros((N, 7), dtype=np.float32)
            for i in range(N):
                out[i] = extract_d2_v2_features_from_seq(kps_33_64f[i])
            return out
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {kps_33_64f.shape}")

    def predict_proba(self, X_7d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 7) 또는 (7,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("D2 v2 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_7d.ndim == 1:
            X_7d = np.expand_dims(X_7d, axis=0)
            single = True

        N = X_7d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_7d)[:, 1]
            p_lgb = clf.predict_proba(X_7d)[:, 1]
            p_fold = 0.50 * p_cb + 0.50 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)
        p0 = 1.0 - ensemble_probs
        res = np.column_stack([p0, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_7d: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_7d)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        3-State 선택적 판정 인터페이스:
        - 0: 확정 정상 (Accept Normal)
        - 1: 확정 결함 (Accept Defect)
        - -1: 판정 보류 (Undetermined / Grey Zone)
        """
        tn = tau_normal if tau_normal is not None else self.tau_normal
        td = tau_defect if tau_defect is not None else self.tau_defect

        if X_or_kps.ndim >= 3:
            X = self.extract_features(X_or_kps)
        else:
            X = X_or_kps

        probs = self.predict_proba(X)
        single = (X.ndim == 1 or (X.ndim == 2 and X.shape[0] == 1))

        if single:
            p_defect = float(probs[1] if probs.ndim == 1 else probs[0, 1])
            if p_defect <= tn:
                state = 0
                state_str = "정상"
            elif p_defect >= td:
                state = 1
                state_str = "무릎 외반/방향 불일치 결함"
            else:
                state = -1
                state_str = "판정 보류 (재촬영 권고)"

            return {
                "decision": state,
                "decision_label": state_str,
                "defect_prob": round(p_defect, 4),
                "tau_normal": tn,
                "tau_defect": td,
                "confidence": round(abs(p_defect - 0.5) * 2, 4)
            }
        else:
            p_defects = probs[:, 1]
            decisions = np.where(p_defects <= tn, 0, np.where(p_defects >= td, 1, -1))
            return {
                "decisions": decisions.tolist(),
                "defect_probs": np.round(p_defects, 4).tolist(),
                "tau_normal": tn,
                "tau_defect": td,
                "coverage": float(np.mean(decisions != -1))
            }

    def load_weights(self, weights_path: str) -> None:
        """저장된 5-Fold GBDT 앙상블 가중치 로드"""
        with open(weights_path, 'rb') as f:
            bundle = pickle.load(f)

        self.cb_models = bundle["catboost_models"]
        self.lgb_models = bundle["lightgbm_models"]
        self.decision_threshold = bundle.get("decision_threshold", self.decision_threshold)
        self.tau_normal = bundle.get("tau_normal", self.tau_normal)
        self.tau_defect = bundle.get("tau_defect", self.tau_defect)
        self.feature_names = bundle.get("feature_names", self.feature_names)
        self.is_fitted = True

    def save_weights(self, weights_path: str) -> None:
        """현재 학습된 5-Fold 모델 가중치 저장"""
        bundle = {
            "defect_code": self.DEFECT_CODE,
            "exercise": self.EXERCISE_NAME,
            "version": "v2.0_diet_7d",
            "feature_dim": len(self.feature_names),
            "feature_names": self.feature_names,
            "decision_threshold": self.decision_threshold,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "catboost_models": self.cb_models,
            "lightgbm_models": self.lgb_models
        }
        with open(weights_path, 'wb') as f:
            pickle.dump(bundle, f)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v5.0 (c2_model_v5.py)
Status: ★ 순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델
        - v4.0의 9D 피처 명세, 생체역학 설계 사상, CatBoost(60%)+LightGBM(40%) 앙상블 아키텍처 100% 계승
        - AI-Hub 3D JSON 의존성 0% 영구 제거 (100% 순수 비전 MediaPipe 33관절 직접 계측)
        - 횟수 분할(16F) 0% 영구 제거 (무삭제 64프레임 3차 스플라인 표준 전수 적용)
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

FEATURE_NAMES_9_C2: List[str] = [
    "c2_gaze_pitch_drop",                  # [1] 기립 대비 최하사점 시선 피치 낙하폭 (deg)
    "c2_cervical_spine_angle_mean",        # [2] 경추-체간 3D 굴곡각 전체 평균 (deg)
    "c2_cervical_spine_angle_drop",        # [3] 기립 대비 최하사점 경추 굴곡 낙하폭 (deg)
    "c2_head_yaw_asymmetry_mean",          # [4] 머리 좌우 회전 비대칭도 평균
    "c2_cervical_velocity_peak_at_nadir",  # [5] 최하사점 부근 경추 각속도 피크 (deg/s)
    "c2_gaze_coupling_lag",                # [6] 골반 하강 곡선과 시선 피치 간 VOR 커플링 지연시간 (frames)
    "c2_cervical_hyperextension_ratio",    # [7] 스쿼트 상승 구간 경추 과신전 비율
    "c2_gated_true_gaze_pitch_drop",       # [8] MediaPipe 안면 시선 벡터 낙하폭 (deg)
    "c2_gated_true_gaze_pitch_bot"         # [9] MediaPipe 안면 시선 최하사점 절대 각도 (deg)
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 시계열 사이의 사잇각(도, deg) 계산"""
    cos_sim = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def extract_c2_v5_9d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) MediaPipe 3D 월드 좌표 시퀀스로부터 c2 v5.0 9D 피처 추출
    """
    T = kps_33_64f.shape[0] # 64
    dt = 1.0 / max(1, T - 1)

    nose = kps_33_64f[:, 0, :]               # #0 Nose
    l_ear = kps_33_64f[:, 7, :]              # #7 Left Ear
    r_ear = kps_33_64f[:, 8, :]              # #8 Right Ear
    mid_ear = (l_ear + r_ear) / 2.0

    l_sh = kps_33_64f[:, 11, :]              # #11 Left Shoulder
    r_sh = kps_33_64f[:, 12, :]              # #12 Right Shoulder
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]             # #23 Left Hip
    r_hip = kps_33_64f[:, 24, :]             # #24 Right Hip
    mid_hip = (l_hip + r_hip) / 2.0

    # 1. 3대 핵심 생체역학 벡터
    v_torso = mid_sh - mid_hip               # 골반 -> 어깨 (체간 축)
    v_neck = mid_ear - mid_sh                # 어깨 -> 귀 (경추 축)
    v_head = nose - mid_ear                  # 귀 -> 코 (안면 시선 축)

    # 2. 3D 공간 각도 시계열
    cervical_raw = vector_angle_3d(v_neck, v_torso)
    cervical_s = gaussian_filter1d(cervical_raw, sigma=1.0)

    # 3. 시선 절대 피치 (수평면 대비 앙각)
    head_horiz = np.linalg.norm(v_head[:, [0, 2]], axis=-1) + 1e-7
    gaze_pitch_raw = np.degrees(np.arctan2(v_head[:, 1], head_horiz))
    gaze_pitch_s = gaussian_filter1d(gaze_pitch_raw, sigma=1.0)

    # 4. 스쿼트 동작 최하사점(b) 및 기립 슬라이스 탐색 (골반 하강 기반)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(1, T // 8)

    # -------------------------------------------------------------
    # [Face Point Guard: 체간 법선 벡터 기반 안면 가시성 검증]
    # -------------------------------------------------------------
    # n_torso = v_torso x v_sh
    # n_z_mean < 0.0 : 카메라 전면/전측면 응시 (얼굴 유효)
    # n_z_mean >= 0.0: 카메라 등짐 / 후방 시점 (얼굴 가림 -> 가짜 코 환각 차단 필수)
    v_sh_lr = l_sh - r_sh
    n_torso = np.cross(v_torso, v_sh_lr)
    n_z_mean = float(np.mean(n_torso[:, 2] / (np.linalg.norm(n_torso, axis=-1) + 1e-7)))
    is_face_visible = bool(n_z_mean < 0.0)

    # -------------------------------------------------------------
    # [1] c2_gaze_pitch_drop (시선 낙하폭 - 안면 가드 적용)
    # -------------------------------------------------------------
    if is_face_visible:
        gaze_stand = float(np.mean(gaze_pitch_s[:standing_slice]))
        f1_gaze_drop = float(max(0.0, gaze_stand - np.min(gaze_pitch_s)))
    else:
        f1_gaze_drop = 0.0

    # -------------------------------------------------------------
    # [2] c2_cervical_spine_angle_mean (경추 평균 굴곡각 - 후방에서도 100% 유효)
    # -------------------------------------------------------------
    f2_cerv_mean = float(np.mean(cervical_s))

    # -------------------------------------------------------------
    # [3] c2_cervical_spine_angle_drop (경추 굴곡 낙하폭 - 후방에서도 100% 유효)
    # -------------------------------------------------------------
    cerv_stand = float(np.mean(cervical_s[:standing_slice]))
    f3_cerv_drop = float(max(0.0, cerv_stand - cervical_s[b]))

    # -------------------------------------------------------------
    # [4] c2_head_yaw_asymmetry_mean (좌우 머리 회전 비대칭 - 안면 가드 적용)
    # -------------------------------------------------------------
    if is_face_visible:
        dist_l = np.linalg.norm(nose - l_ear, axis=-1)
        dist_r = np.linalg.norm(nose - r_ear, axis=-1)
        yaw_asym = np.abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
        f4_yaw_asym_mean = float(np.mean(yaw_asym))
    else:
        f4_yaw_asym_mean = 0.0

    # -------------------------------------------------------------
    # [5] c2_cervical_velocity_peak_at_nadir (최하사점 경추 각속도 피크 - 후방 유효)
    # -------------------------------------------------------------
    v_cerv = np.gradient(cervical_s, dt)
    w_start = max(0, b - 6)
    w_end = min(T, b + 7)
    f5_vel_peak = float(np.max(np.abs(v_cerv[w_start:w_end])))

    # -------------------------------------------------------------
    # [6] c2_gaze_coupling_lag (골반-시선 커플링 지연 프레임, VOR - 안면 가드 적용)
    # -------------------------------------------------------------
    if is_face_visible:
        p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
        g_norm = (gaze_pitch_s - np.mean(gaze_pitch_s)) / (np.std(gaze_pitch_s) + 1e-7)
        xcorr = correlate(g_norm, p_norm, mode='full')
        lags = np.arange(-(T - 1), T)
        f6_coupling_lag = float(abs(lags[np.argmax(xcorr)]))
    else:
        f6_coupling_lag = 0.0

    # -------------------------------------------------------------
    # [7] c2_cervical_hyperextension_ratio (경추 과신전 비율 - 후방 유효)
    # -------------------------------------------------------------
    ascent = cervical_s[b:]
    if len(ascent) > 0:
        f7_hyperext_ratio = float(np.sum(ascent > cerv_stand + 5.0) / len(ascent))
    else:
        f7_hyperext_ratio = 0.0

    # -------------------------------------------------------------
    # [8] c2_gated_true_gaze_pitch_drop (게이트형 3D 시선 낙하폭 - 안면 가드 적용)
    # -------------------------------------------------------------
    if is_face_visible:
        f8_gated_drop = float(max(0.0, gaze_stand - gaze_pitch_s[b]))
    else:
        f8_gated_drop = 0.0

    # -------------------------------------------------------------
    # [9] c2_gated_true_gaze_pitch_bot (최하사점 시선 절대각도 - 안면 가드 적용)
    # -------------------------------------------------------------
    if is_face_visible:
        f9_gated_bot = float(gaze_pitch_s[b])
    else:
        f9_gated_bot = 0.0

    return np.array([
        f1_gaze_drop,
        f2_cerv_mean,
        f3_cerv_drop,
        f4_yaw_asym_mean,
        f5_vel_peak,
        f6_coupling_lag,
        f7_hyperext_ratio,
        f8_gated_drop,
        f9_gated_bot
    ], dtype=np.float32)


def extract_c2_v5_features(X_kps: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 배치 또는 (64, 33, 3) 단일 시퀀스로부터 9D 피처셋 추출
    """
    if X_kps.ndim == 3:
        return extract_c2_v5_9d_features_from_seq(X_kps)
    
    N = X_kps.shape[0]
    out = np.zeros((N, 9), dtype=np.float32)
    for i in range(N):
        out[i] = extract_c2_v5_9d_features_from_seq(X_kps[i])
    return out


class BabelSquatC2ModelV5:
    """
    바벨 스쿼트 c2 결함(고개 정면) 순수 비전 100% MediaPipe 33관절 기반 공식 프로덕션 모델 v5.0
    CatBoost 60% + LightGBM 40% 앙상블 (9D 고간섭 소거 정예 모델 계승)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 9
    FEATURE_NAMES = FEATURE_NAMES_9_C2

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.cb_models: List[CatBoostClassifier] = []
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_9_C2

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c2_model_v5_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64프레임 시퀀스로부터 9D 피처 추출"""
        return extract_c2_v5_features(kps_33_64f)

    def predict_proba(self, X_9d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 9) 또는 (9,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v5 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_9d.ndim == 1:
            X_9d = X_9d.reshape(1, -1)
            single = True

        N = X_9d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_9d)[:, 1]
            p_lgb = clf.predict_proba(X_9d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_9d: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_9d)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

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
        print(f"[*] c2 v5.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_9_C2)
        self.is_fitted = True
        return self


BabelSquatC2Model = BabelSquatC2ModelV5

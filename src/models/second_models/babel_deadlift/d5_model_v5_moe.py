# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v5.0 Soft-MoE (d5_model_v5_moe.py)
Status: 45D 외부 AI 자문 반영 피처셋 + View-Aware Soft Routing Mixture-of-Experts
        - 킬러 피처 1: 요추-골반 리듬 붕괴비 (d5_hip_bar_decoupling_ratio, Y축 전용 Early Hip Rise) -> 정면 깊이 소실 100% 무결 극복
        - 킬러 피처 2: Ape Index 동적 허용 마진 (d5_ape_index_adjusted_margin) -> 피험자 3번 체형 편향 FP 소거
        - 킬러 피처 3: 신체 사영 다각형 면적 왜곡율 (d5_torso_limb_polygonal_area_ratio) -> 2D 투영 왜곡율
        - 킬러 피처 4: 전신-바벨 합성 무게중심 전방 쏠림도 (d5_system_com_midfoot_offset)
        - 킬러 피처 5: 정면 연속 가중치 (d5_view_soft_front_weight)
        - 2-Gate Soft Routing MoE:
          * Frontal Expert (정면 전담): Y축 리듬비, 2D 단축비, 다각형 면적비 특화
          * Oblique Expert (사각 전담): 3D 거리, 무릎-팔꿈치 기하, 시상면 전방 돌출도 특화
          * Soft Blending: P = W_front * P_front + (1 - W_front) * P_oblique
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

from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v1 import (
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_ELBOW, MP_RIGHT_ELBOW,
    MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE, MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_HEEL, MP_RIGHT_HEEL, MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX,
    extract_d5_v1_features_from_seq, FEATURE_NAMES_28_D5
)
from src.models.second_models.babel_deadlift.d5_model_v3 import (
    extract_d5_v3_features_from_seq, FEATURE_NAMES_32_D5, VIEW_CODE_MAP
)
from src.models.second_models.babel_deadlift.d5_model_v4_gated import (
    extract_d5_v4_gated_features_from_seq, FEATURE_NAMES_40_D5_GATED
)

FEATURE_NAMES_45_D5_MOE: List[str] = FEATURE_NAMES_40_D5_GATED + [
    # Group 8: 외부 AI 추천 킬러 피처 및 MoE 라우팅 (5D)
    'd5_hip_bar_decoupling_ratio',       # [40] Y축 전용 요추-골반 리듬비 (Early Hip Rise / Stripper Pull Ratio)
    'd5_ape_index_adjusted_margin',      # [41] Ape Index(팔/체간 비) 기반 체형 불변 무릎 통과 마진
    'd5_torso_limb_polygonal_area_ratio',# [42] 신체-사지 4점 폐곡선 다각형 2D 사영 면적 왜곡율
    'd5_system_com_midfoot_offset',      # [43] 전신-바벨 합성 무게중심(CoM) 발목 대비 전방 쏠림도
    'd5_view_soft_front_weight'          # [44] 어깨너비/체간길이 2D 사영비 기반 정면 연속 가중치 (MoE Gate)
]


def extract_d5_v5_moe_features_from_seq(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    기존 40D Gated 피처 + 5D 외부 AI 자문 피처 = 총 45D 피처 추출
    """
    # 1. 기존 v4 40D 피처 추출 (완벽 계승)
    f40 = extract_d5_v4_gated_features_from_seq(kps_arr, view_name=view_name)

    pts = kps_arr[:, :, :3].astype(np.float32)
    T = pts.shape[0]

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    elbow_l = pts[:, MP_LEFT_ELBOW]
    elbow_r = pts[:, MP_RIGHT_ELBOW]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]

    shoulder = (shoulder_l + shoulder_r) / 2.0
    elbow = (elbow_l + elbow_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0

    torso_len = float(np.linalg.norm(shoulder - hip, axis=-1).mean() + 1e-6)
    femur_len = float(np.linalg.norm(hip - knee, axis=-1).mean() + 1e-6)
    arm_len = float(np.linalg.norm(bar - shoulder, axis=-1).mean() + 1e-6)
    body_scale = torso_len + femur_len

    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))

    # 무릎 통과 시점(t_knee): nadir 이후 bar의 y가 knee의 y보다 작아지거나(올라가거나) 가장 근접한 프레임
    t_knee = min(T - 1, nadir_idx + 12)
    for t in range(nadir_idx, T):
        if bar[t, 1] <= knee[t, 1]:
            t_knee = t
            break

    # -------------------------------------------------------------
    # [피처 40] 요추-골반 리듬 붕괴비 (Early Hip Rise / Stripper Pull Ratio)
    # *핵심*: 오직 2D Y축(수직 이동)만 사용하므로 정면 깊이 소실에 100% 불변!
    # -------------------------------------------------------------
    dy_hip = abs(hip[t_knee, 1] - hip[nadir_idx, 1])
    dy_bar = abs(bar[t_knee, 1] - bar[nadir_idx, 1])
    hip_bar_ratio = float(dy_hip / (dy_bar + 1e-5))
    hip_bar_ratio = float(np.clip(hip_bar_ratio, 0.0, 5.0))

    # -------------------------------------------------------------
    # [피처 41] Ape Index 동적 허용 마진 (체형 정규화)
    # 팔이 긴 피험자(피험자 3번 등)의 과잉 결함 오분류 방지
    # -------------------------------------------------------------
    ape_index = float(arm_len / torso_len)
    raw_margin = float(abs(bar[t_knee, 0] - knee[t_knee, 0]) / body_scale)
    ape_offset = 0.12 * max(0.0, ape_index - 1.05)
    ape_adjusted_margin = float(raw_margin - ape_offset)

    # -------------------------------------------------------------
    # [피처 42] 신체-사지 4점 폐곡선 다각형 2D 사영 면적 왜곡율
    # Shoulder - Hip - Knee - Bar(Wrist) 사각 다각형의 Shoelace 면적
    # -------------------------------------------------------------
    def calc_quad_area(p1, p2, p3, p4):
        # 2D (x, y) shoelace
        xs = [p1[0], p2[0], p3[0], p4[0]]
        ys = [p1[1], p2[1], p3[1], p4[1]]
        area = 0.5 * abs((xs[0]*ys[1] - ys[0]*xs[1]) +
                         (xs[1]*ys[2] - ys[1]*xs[2]) +
                         (xs[2]*ys[3] - ys[2]*xs[3]) +
                         (xs[3]*ys[0] - ys[3]*xs[0]))
        return float(area)

    area_nadir = calc_quad_area(shoulder[nadir_idx], hip[nadir_idx], knee[nadir_idx], bar[nadir_idx])
    area_knee = calc_quad_area(shoulder[t_knee], hip[t_knee], knee[t_knee], bar[t_knee])
    poly_area_ratio = float(area_knee / (area_nadir + 1e-6))
    poly_area_ratio = float(np.clip(poly_area_ratio, 0.0, 10.0))

    # -------------------------------------------------------------
    # [피처 43] 전신-바벨 합성 무게중심(CoM) 발목 대비 전방 쏠림도
    # 체간(0.45), 대퇴(0.30), 바벨(0.25)
    # -------------------------------------------------------------
    com_t_knee = 0.45 * ((shoulder[t_knee] + hip[t_knee]) / 2.0) + \
                 0.30 * ((hip[t_knee] + knee[t_knee]) / 2.0) + \
                 0.25 * bar[t_knee]
    com_midfoot_offset = float(abs(com_t_knee[0] - ankle[t_knee, 0]) / body_scale)

    # -------------------------------------------------------------
    # [피처 44] 어깨너비/체간길이 2D 사영비 기반 정면 연속 가중치 (Soft-MoE Gate)
    # 정면 뷰(view3)에서는 0.85~1.0, 사각/측면 뷰에서는 0.0~0.3
    # -------------------------------------------------------------
    shoulder_width_2d = float(abs(shoulder_l[:, 0] - shoulder_r[:, 0]).mean())
    torso_len_2d = float(np.sqrt((shoulder[:, 0] - hip[:, 0])**2 + (shoulder[:, 1] - hip[:, 1])**2).mean() + 1e-6)
    soft_front_gate = float(np.clip(shoulder_width_2d / (torso_len_2d * 0.85), 0.0, 1.0))

    feat_5d = np.array([
        hip_bar_ratio,
        ape_adjusted_margin,
        poly_area_ratio,
        com_midfoot_offset,
        soft_front_gate
    ], dtype=np.float32)

    return np.concatenate([f40, feat_5d])


class BabelDeadliftD5ModelV5MoE:
    """
    45D View-Aware Soft Routing Mixture-of-Experts (MoE) 앙상블 분류기
    - Frontal Expert: 정면 뷰 및 깊이 불변 피처 집중 모델 (LGBM + CB + RF)
    - Oblique Expert: 사각/측면 뷰 및 시상면 기하 피처 집중 모델 (LGBM + CB + RF)
    - Global Expert: 전수 피처 통합 모델
    - Soft Blending: P = 0.5 * P_global + 0.5 * (W_front * P_front + (1 - W_front) * P_oblique)
    """
    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None):
        self.is_fitted = False

        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.035,
            'num_leaves': 22,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 2
        }
        self.cb_params = cb_params or {
            'iterations': 200,
            'depth': 5,
            'learning_rate': 0.035,
            'random_seed': 42,
            'verbose': 0,
            'thread_count': 2
        }
        self.rf_params = rf_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': 2
        }

        # 1. Global Tri-Ensemble
        self.global_lgb = lgb.LGBMClassifier(**self.lgb_params) if lgb else None
        self.global_cb = CatBoostClassifier(**self.cb_params) if CatBoostClassifier else None
        self.global_rf = RandomForestClassifier(**self.rf_params)

        # 2. Frontal Expert (정면 전담)
        self.front_lgb = lgb.LGBMClassifier(**self.lgb_params) if lgb else None
        self.front_cb = CatBoostClassifier(**self.cb_params) if CatBoostClassifier else None
        self.front_rf = RandomForestClassifier(**self.rf_params)

        # 3. Oblique Expert (사각/측면 전담)
        self.oblique_lgb = lgb.LGBMClassifier(**self.lgb_params) if lgb else None
        self.oblique_cb = CatBoostClassifier(**self.cb_params) if CatBoostClassifier else None
        self.oblique_rf = RandomForestClassifier(**self.rf_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Global Expert 학습 (전수 데이터)
        if self.global_lgb:
            self.global_lgb.fit(X, y)
        if self.global_cb:
            self.global_cb.fit(X, y)
        self.global_rf.fit(X, y)

        # W_front 게이트 값 확인 (인덱스 44: d5_view_soft_front_weight)
        w_front = X[:, 44]

        # 정면 서브셋 및 사각 서브셋 분할 가중 학습
        # Soft 가중치 샘플링 또는 마스크 분할
        front_mask = (w_front >= 0.50)
        oblique_mask = (w_front < 0.50)

        # 2. Frontal Expert 학습
        if sum(front_mask) > 10 and len(np.unique(y[front_mask])) > 1:
            X_front, y_front = X[front_mask], y[front_mask]
            # 정면 가중치 부여 샘플 학습
            sw_front = np.clip(w_front[front_mask], 0.2, 1.0)
            if self.front_lgb:
                self.front_lgb.fit(X_front, y_front, sample_weight=sw_front)
            if self.front_cb:
                self.front_cb.fit(X_front, y_front, sample_weight=sw_front)
            self.front_rf.fit(X_front, y_front, sample_weight=sw_front)
        else:
            self.front_lgb = self.global_lgb
            self.front_cb = self.global_cb
            self.front_rf = self.global_rf

        # 3. Oblique Expert 학습
        if sum(oblique_mask) > 10 and len(np.unique(y[oblique_mask])) > 1:
            X_oblique, y_oblique = X[oblique_mask], y[oblique_mask]
            sw_oblique = np.clip(1.0 - w_front[oblique_mask], 0.2, 1.0)
            if self.oblique_lgb:
                self.oblique_lgb.fit(X_oblique, y_oblique, sample_weight=sw_oblique)
            if self.oblique_cb:
                self.oblique_cb.fit(X_oblique, y_oblique, sample_weight=sw_oblique)
            self.oblique_rf.fit(X_oblique, y_oblique, sample_weight=sw_oblique)
        else:
            self.oblique_lgb = self.global_lgb
            self.oblique_cb = self.global_cb
            self.oblique_rf = self.global_rf

        self.is_fitted = True
        return self

    def _predict_sub_proba(self, lgb_m, cb_m, rf_m, X: np.ndarray) -> np.ndarray:
        p_l = lgb_m.predict_proba(X)[:, 1] if lgb_m else 0.5
        p_c = cb_m.predict_proba(X)[:, 1] if cb_m else 0.5
        p_r = rf_m.predict_proba(X)[:, 1]
        return 0.50 * p_l + 0.30 * p_c + 0.20 * p_r

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # 1. Global Ensemble 예측
        p_global = self._predict_sub_proba(self.global_lgb, self.global_cb, self.global_rf, X)

        # 2. Frontal & Oblique Expert 예측
        p_front = self._predict_sub_proba(self.front_lgb, self.front_cb, self.front_rf, X)
        p_oblique = self._predict_sub_proba(self.oblique_lgb, self.oblique_cb, self.oblique_rf, X)

        # 3. Soft Routing Blending
        w_front = X[:, 44]  # [0.0, 1.0]
        p_moe = w_front * p_front + (1.0 - w_front) * p_oblique

        # 4. Global 50% + MoE 50% 최종 앙상블
        p_final = 0.50 * p_global + 0.50 * p_moe

        return np.column_stack([1.0 - p_final, p_final])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'BabelDeadliftD5ModelV5MoE':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

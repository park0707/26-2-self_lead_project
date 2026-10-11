# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c3. 발과 무릎의 방향 일치 (Knee-Toe Alignment / Knee Valgus)
Model Version: v5.0 (c3_model_v5.py)
Status: [차세대 모델] 절대 무릎-발목 너비 비율(Absolute Width Ratio) 및 좌우 독립/절댓값 합(Bilateral Sum) 탑재 (53D)

[v4.0 대비 혁신 내용]:
1. [기준 1] 골반 중심 관상면(Coronal / XY) 절대 무릎-발목 너비 비율 (Absolute Knee-to-Ankle Ratio):
   - 기립 대비 변화량(Delta)에만 의존하던 기존의 치명적 맹점 극복
   - 서 있을 때부터 이미 무릎이 모여 있던 만성 외반(와이드 스탠스 무릎 모임)을 완벽 포착
   - 주요 피처: 최하사점 절대비율, 최소비율, 기립 절대비율, 외반(비율<0.85) 지속 시간 점유율, 절대 부족분 적분값
2. [기준 2] 좌우 무릎 외반의 부호 상쇄 제거 및 양측 절댓값 합 (Bilateral Independent & Sum):
   - 기존의 v_l - v_r 감산에 의한 좌우 대칭 외반 상쇄 버그 원천 소거
   - 주요 피처: 최하사점 절대 외반합 (|v_l| + |v_r|), 최대 절대 외반합, 양 무릎 동시 내측 붕괴 강도
3. 모델 앙상블:
   - 20D 기하학 + 6D 좌우독립 + 17D 딥러닝 임베딩 + 10D 절대/좌우외반 피처 = 총 53D 정예 피처셋
   - CatBoost (60%) + LightGBM (40%) 앙상블
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES

V5_FEATURE_NAMES_10: List[str] = [
    'c3_ka_ratio_nadir_abs',         # 최하사점 절대 무릎/발목 너비 비율
    'c3_ka_ratio_min_abs',           # 전 구간 최소 무릎/발목 너비 비율
    'c3_ka_ratio_standing_abs',      # 기립 자세 무릎/발목 너비 비율
    'c3_ka_valgus_duration_ratio',   # 무릎/발목 비율 < 0.85 지속 시간비
    'c3_ka_absolute_deficit_integ',  # 정상기준(0.90) 대비 무릎 폭 부족분 적분값
    'c3_ka_chronic_standing_valgus', # 기립 시 이미 외반 상태인 만성 플래그
    'c3_valgus_abs_sum_nadir',       # 최하사점 좌우 절대 외반합 (|vl| + |vr|)
    'c3_valgus_abs_sum_max',         # 전 구간 최대 좌우 절대 외반합
    'c3_valgus_concurrence_max',     # 양 무릎 동시 내측 붕괴 강도
    'c3_valgus_bilateral_min'        # 좌우 중 최소 외반값 (양측 동시성 검증)
]


class BabelSquatC3ModelV5:
    """
    바벨 스쿼트 c3 결함(무릎 외반/정렬 불일치) 확정 모델 v5.0 (53D)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.cb_models: List[CatBoostClassifier] = []
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.feature_names: List[str] = []
        self.is_fitted = False

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c3_model_v5_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @staticmethod
    def extract_v5_new_features(X_b: np.ndarray) -> np.ndarray:
        """
        Format B 시계열 (N, T, 92) 또는 (T, 92)에서 신규 10D 절대/좌우외반 피처 추출
        """
        single = False
        if X_b.ndim == 2:
            X_b = X_b[np.newaxis, ...]
            single = True

        N, T, C = X_b.shape
        kd = X_b[:, :, 69]
        ad = np.maximum(X_b[:, :, 70], 1e-4)
        ratio = kd / ad
        vl = X_b[:, :, 66]
        vr = X_b[:, :, 67]

        nadir_idx = T // 2
        f_ka_nadir = ratio[:, nadir_idx:nadir_idx+1]
        f_ka_min = np.min(ratio, axis=1, keepdims=True)
        f_ka_standing = np.mean(ratio[:, :max(1, T // 8)], axis=1, keepdims=True)
        f_valg_duration = np.mean(ratio < 0.85, axis=1, keepdims=True)
        f_abs_deficit = np.mean(np.maximum(0.0, 0.90 - ratio), axis=1, keepdims=True)
        f_chronic = (f_ka_standing < 0.85).astype(np.float32)

        f_abs_sum_nadir = (np.abs(vl[:, nadir_idx]) + np.abs(vr[:, nadir_idx]))[:, None]
        f_abs_sum_max = np.max(np.abs(vl) + np.abs(vr), axis=1, keepdims=True)
        f_concurrence = np.max(np.sqrt(np.maximum(0.0, vl) * np.maximum(0.0, vr)), axis=1, keepdims=True)
        f_bilateral_min = np.min(np.maximum(0.0, np.minimum(vl, vr)), axis=1, keepdims=True)

        feats_10d = np.hstack([
            f_ka_nadir, f_ka_min, f_ka_standing, f_valg_duration, f_abs_deficit, f_chronic,
            f_abs_sum_nadir, f_abs_sum_max, f_concurrence, f_bilateral_min
        ]).astype(np.float32)

        return feats_10d[0] if single else feats_10d

    def predict_proba(self, X_53d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 53) 또는 (53,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C3 v5 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_53d.ndim == 1:
            X_53d = X_53d[np.newaxis, ...]
            single = True

        N = X_53d.shape[0]
        n_folds = len(self.cb_models)
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, lgb_m in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_53d)[:, 1]
            p_lgb = lgb_m.predict_proba(X_53d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / n_folds

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_53d: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_53d)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'feature_names': self.feature_names,
            'decision_threshold': self.decision_threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c3 v5.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.feature_names = payload.get('feature_names', [])
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.is_fitted = True
        return self


BabelSquatC3Model = BabelSquatC3ModelV5

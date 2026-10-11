# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v7.0 (c1_model_v7.py)
Status: ★ 최종 확정 프로덕션 모델 (46D Deep-Hybrid: 29D 정예 생체역학 + 17D 딥러닝 시계열 임베딩)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 29D 정예 생체역학 기하학 피처 (SpinePose 3D 척추 곡률, 버트윙크, 굿모닝 스쿼트 분리율 등)
   - 17D 딥러닝 시계열 피처 (64프레임 Conv-BiGRU-Attention 추출 16D 임베딩 + 1D 사후확률)
   - 총 46D 융합 피처셋
2. 실측 성능 (5-Fold StratifiedGroupKFold 완전 격리):
   - Precision: 81.59% (v6.0 대비 +0.16%p)
   - Recall: 76.98% (v6.0 대비 +1.30%p)
   - F1-Score: 79.22% (v6.0 대비 +0.77%p 상승, 역대 최고치 경신!)
   - ROC-AUC: 88.58% (v6.0 대비 +0.15%p)
   - 미인식 오류 12건 즉각 구제 (224건 -> 212건)
   - 가짜 정상(Level 4, 52건) 배제 후 실무 Clean F1: 79.89% (80% 코앞 진입!)
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.c1_model_v6 import C1_V6A_FEATURE_NAMES_29

C1_DEEP_FEATURE_NAMES_17 = [f"c1_deep_temporal_emb_{i}" for i in range(16)] + ["c1_deep_temporal_prob"]
C1_V7_FEATURE_NAMES_45 = [f for f in C1_V6A_FEATURE_NAMES_29 if f != "ear_sh_hip_min"] + C1_DEEP_FEATURE_NAMES_17


class BabelSquatC1ModelV7:
    """
    바벨 스쿼트 c1 결함(척추의 중립 위반) 최종 확정 프로덕션 모델 v7.1
    CatBoost 60% + LightGBM 40% 앙상블 (45D Deep-Hybrid)
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
        self.feature_names = C1_V7_FEATURE_NAMES_45

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c1_model_v7_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(self, X_46d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 46) 또는 (46,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v7 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_46d.ndim == 1:
            X_46d = X_46d.reshape(1, -1)
            single = True

        # 46D 입력 시 간섭 피처 ear_sh_hip_min (index 11) 자동 소거 (45D 변환)
        if X_46d.shape[1] == 46:
            keep_idx = [i for i in range(46) if i != 11]
            X_46d = X_46d[:, keep_idx]

        N = X_46d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_46d)[:, 1]
            p_lgb = clf.predict_proba(X_46d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_46d: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X_46d)
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
        print(f"[*] c1 v7.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.feature_names = payload.get('feature_names', C1_V7_FEATURE_NAMES_45)
        self.is_fitted = True
        return self


BabelSquatC1Model = BabelSquatC1ModelV7

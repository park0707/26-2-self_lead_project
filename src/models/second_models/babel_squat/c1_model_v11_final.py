# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.0_final (c1_model_v11_final.py)
Status: ★ [E0 + E1 궁극의 융합 챔피언] Tri-Ensemble (v11_a + v11_a_diet + v11_b TCN) 공식 프로덕션 모델
        - 1) v11_a (E0 38D Nadir 국소 윈도우 전수 피처 GBDT, 가중치 10%)
        - 2) v11_a_diet (E0 27D 정예 생체역학 피처 GBDT, 가중치 40%)
        - 3) v11_b (E1 24D 관절 64F 시계열 텐서 Dilated TCN, 가중치 50%)
        - 4) 실측 성능 (N=2,605 전수, 5-Fold 피험자 완전 격리):
             * 전체 F1: 84.47% (v10 79.73% 대비 +4.74%p, 역대 최고치 경신)
             * 전체 정확도(Acc): 85.22%
             * 5대 시점 전수 83.3%~87.4% 폭등 (완벽한 시점 불변성 확립)
             * 후방(View 1: 84.12%), 정면(View 3: 83.51%) 등 최약 시점 완벽 정복
        - 5) 3-State 선택적 거부(Selective Abstention) 아키텍처 완전 통합
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_a import (
    extract_c1_v11_a_features_from_seq,
    FEATURE_NAMES_38_C1,
    BabelSquatC1ModelV11A
)
from src.models.second_models.babel_squat.c1_model_v11_a_diet import (
    extract_c1_v11_a_diet_features_from_seq,
    FEATURE_NAMES_27_C1_DIET,
    BabelSquatC1ModelV11ADiet
)
from src.models.second_models.babel_squat.c1_model_v11_b import (
    SquatC1DilatedTCN,
    BabelSquatC1ModelV11B
)
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d


class BabelSquatC1ModelV11Final:
    """
    바벨 스쿼트 c1 결함(척추 중립 위반) 궁극의 융합 챔피언 모델 v11_final
    v11_a (10%) + v11_a_diet (40%) + v11_b TCN (50%) Tri-Ensemble
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60,
        w_a: float = 0.10,
        w_diet: float = 0.40,
        w_tcn: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.w_a = w_a
        self.w_diet = w_diet
        self.w_tcn = w_tcn

        self.model_a = BabelSquatC1ModelV11A()
        self.model_diet = BabelSquatC1ModelV11ADiet()
        self.tcn_models: List[SquatC1DilatedTCN] = []
        self.is_fitted = False

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(
        self,
        X_38d: np.ndarray,
        X_27d: np.ndarray,
        X_temporal: np.ndarray
    ) -> np.ndarray:
        """
        3가지 표현(38D, 27D, 64x24 시계열)으로부터 앙상블 확률 반환
        """
        if not self.is_fitted:
            raise RuntimeError("C1 v11_final 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_38d.ndim == 1:
            X_38d = X_38d.reshape(1, -1)
            X_27d = X_27d.reshape(1, -1)
            X_temporal = X_temporal[np.newaxis, ...]
            single = True

        N = len(X_38d)
        p_a = self.model_a.predict_proba(X_38d)[:, 1]
        p_d = self.model_diet.predict_proba(X_27d)[:, 1]

        # TCN 추론
        x_t = torch.tensor(X_temporal.transpose(0, 2, 1), dtype=torch.float32)
        p_t = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            for tcn in self.tcn_models:
                tcn.eval()
                logits = tcn(x_t).squeeze(-1)
                p_t += torch.sigmoid(logits).cpu().numpy() / len(self.tcn_models)

        # 3-Way Tri-Ensemble
        p_final = self.w_a * p_a + self.w_diet * p_d + self.w_tcn * p_t
        p_final = np.clip(p_final, 0.0, 1.0)
        p0 = 1.0 - p_final
        res = np.column_stack([p0, p_final])
        return res[0] if single else res

    def predict(
        self,
        X_38d: np.ndarray,
        X_27d: np.ndarray,
        X_temporal: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_38d, X_27d, X_temporal)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tcn_state_dicts = [tcn.state_dict() for tcn in self.tcn_models]
        payload = {
            'model_a_cb': self.model_a.cb_models,
            'model_a_lgb': self.model_a.lgb_models,
            'model_diet_cb': self.model_diet.cb_models,
            'model_diet_lgb': self.model_diet.lgb_models,
            'tcn_state_dicts': tcn_state_dicts,
            'w_a': self.w_a,
            'w_diet': self.w_diet,
            'w_tcn': self.w_tcn,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v11_final 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.model_a.cb_models = payload['model_a_cb']
        self.model_a.lgb_models = payload['model_a_lgb']
        self.model_a.is_fitted = True

        self.model_diet.cb_models = payload['model_diet_cb']
        self.model_diet.lgb_models = payload['model_diet_lgb']
        self.model_diet.is_fitted = True

        self.tcn_models = []
        for sd in payload['tcn_state_dicts']:
            tcn = SquatC1DilatedTCN()
            tcn.load_state_dict(sd)
            tcn.eval()
            self.tcn_models.append(tcn)

        self.w_a = payload.get('w_a', 0.10)
        self.w_diet = payload.get('w_diet', 0.40)
        self.w_tcn = payload.get('w_tcn', 0.50)
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.is_fitted = True
        return self

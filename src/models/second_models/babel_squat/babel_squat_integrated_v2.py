# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v2.py
Purpose: 바벨 스쿼트 4대 세부 결함(c1~c4) 통합 추론 및 실서비스 다중 라벨 결함 진단 엔진 v2.0
- 핵심 업그레이드 사항 (v1.0 대비):
  1) c1 모델: 척추 중립 위반 c1 v1.0 (22D, F1 72.73%) -> c1 v6-A (29D, Opt F1 80.04%, 실무군 AUC 92.88%)로 전면 개량
  2) c2 모델: 고개 정면 c2 v3.0 (13D, F1 85.85%)
  3) c3 모델: 발-무릎 일치 c3 v3.0 (20D, F1 90.57%)
  4) c4 모델: 발바닥 지면 고정 c4 v3.0 (9D, F1 89.70%)
- 소프트 위험도 융합 게이트 (Soft Risk-Fusion Gate):
  Defect_Score = 1 - ∏ (1 - clip(w_k * P(c_k), 0, 1))
- 생체역학 우선순위 큐: c1(척추 디스크) > c3(무릎 ACL) > c4(발바닥 지면) > c2(경추)
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.c1_model_v6 import BabelSquatC1ModelV6
from src.models.second_models.babel_squat.c2_model_v3 import BabelSquatC2ModelV3
from src.models.second_models.babel_squat.c3_model_v3 import BabelSquatC3ModelV3
from src.models.second_models.babel_squat.c4_model_v3 import BabelSquatC4ModelV3

# 생체역학 부상 위험도 기본 가중치 (척추 > 무릎 > 발바닥 > 고개)
DEFAULT_RISK_WEIGHTS_V2 = {
    'c1': 1.15, # 척추 중립 위반 (요추 추간판 디스크 최고 위험)
    'c3': 1.10, # 무릎 외반 (전방십자인대 ACL 파열 고위험)
    'c4': 1.00, # 발바닥 지면 고정 위반 (체중 분산 불안정)
    'c2': 0.90  # 고개 정면 위반 (경추 긴장/상대적 저위험)
}

DEFECT_NAMES = {
    'c1': '척추의 중립 위반 (Spine Neutrality)',
    'c2': '고개 정면 미유지 (Head/Gaze Alignment)',
    'c3': '발과 무릎 방향 불일치 (Knee Valgus)',
    'c4': '발바닥 지면 미고정 (Foot Ground Fixation)'
}

PRIORITY_ORDER = ['c1', 'c3', 'c4', 'c2']

class BabelSquatIntegratedEngineV2:
    """
    바벨 스쿼트 4대 세부 결함 결합 실서비스 통합 진단 엔진 v2.0
    - c1 척추 모델 v6-A 29D (Opt F1 80.04% 챔피언) 탑재
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        threshold_clean: float = 0.79,
        c1_variant: str = "v6-a",
        decision_thresholds: Optional[Dict[str, float]] = None,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.c1_variant = c1_variant.lower()

        self.threshold_clean = threshold_clean
        self.decision_thresholds = decision_thresholds or {
            'c1': 0.50, 'c2': 0.50, 'c3': 0.50, 'c4': 0.50
        }
        self.risk_weights = risk_weights or DEFAULT_RISK_WEIGHTS_V2

        # 4대 세부 결함 최신 모델 초기화
        self.model_c1 = BabelSquatC1ModelV6(mode="cascade", variant=self.c1_variant)
        self.model_c2 = BabelSquatC2ModelV3(mode="cascade")
        self.model_c3 = BabelSquatC3ModelV3(mode="cascade")
        self.model_c4 = BabelSquatC4ModelV3(mode="cascade")

        self.is_loaded = False
        self._load_all_weights()

    def _load_all_weights(self):
        """4대 모델의 사전 학습 가중치 로드"""
        w_c1 = os.path.join(self.weights_dir, 'c1_model_v6_weights.pkl')
        w_c2 = os.path.join(self.weights_dir, 'c2_model_v3_weights.pkl')
        w_c3 = os.path.join(self.weights_dir, 'c3_model_v3_weights.pkl')
        w_c4 = os.path.join(self.weights_dir, 'c4_model_v3_weights.pkl')

        if os.path.exists(w_c1):
            self.model_c1.load_weights(w_c1)
        if os.path.exists(w_c2):
            self.model_c2.load_weights(w_c2)
        if os.path.exists(w_c3):
            self.model_c3.load_weights(w_c3)
        if os.path.exists(w_c4):
            self.model_c4.load_weights(w_c4)

        self.is_loaded = True

    def compute_defect_score(self, probs: Dict[str, float]) -> float:
        """
        소프트 위험도 융합 게이트:
        Defect_Score = 1 - ∏ (1 - clip(w_k * P(c_k), 0, 1))
        """
        prod_clean = 1.0
        for k, p in probs.items():
            w = self.risk_weights.get(k, 1.0)
            p_weighted = float(np.clip(w * p, 0.0, 1.0))
            prod_clean *= (1.0 - p_weighted)
        defect_score = 1.0 - prod_clean
        return float(np.clip(defect_score, 0.0, 1.0))

    def evaluate_probabilities(
        self,
        feats_c1: np.ndarray,
        feats_c2: np.ndarray,
        feats_c3: np.ndarray,
        feats_c4: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """각 모델의 결함 확률 배열 반환 (N,)"""
        p_c1 = self.model_c1.predict_proba(feats_c1)[:, 1]
        p_c2 = self.model_c2.predict_proba(feats_c2)[:, 1]
        p_c3 = self.model_c3.predict_proba(feats_c3)[:, 1]
        p_c4 = self.model_c4.predict_proba(feats_c4)[:, 1]

        return {
            'c1': p_c1,
            'c2': p_c2,
            'c3': p_c3,
            'c4': p_c4
        }

    def diagnose_batch(
        self,
        probs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """배치 단위 종합 진단 결과 반환"""
        N = len(probs_dict['c1'])
        defect_scores = np.zeros(N, dtype=np.float32)
        is_clean_arr = np.zeros(N, dtype=bool)

        preds_multi = np.zeros((N, 4), dtype=int)
        keys = ['c1', 'c2', 'c3', 'c4']

        for i in range(N):
            sample_probs = {k: probs_dict[k][i] for k in keys}
            ds = self.compute_defect_score(sample_probs)
            defect_scores[i] = ds
            is_clean_arr[i] = (ds < self.threshold_clean)

            for col_idx, k in enumerate(keys):
                th = self.decision_thresholds.get(k, 0.50)
                preds_multi[i, col_idx] = int(sample_probs[k] >= th)

        y_overall_pred = (~is_clean_arr).astype(int)

        return {
            'defect_scores': defect_scores,
            'is_clean': is_clean_arr,
            'y_overall_pred': y_overall_pred,
            'preds_multi': preds_multi
        }

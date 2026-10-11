# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v1.py
Purpose: 바벨 스쿼트 4대 세부 결함(c1~c4) 결합 소프트 위험도 융합 기반 실서비스 통합 진단 엔진 v1.0
- 구성 모델:
  1) c1 (척추의 중립): BabelSquatC1ModelV1 (22D)
  2) c2 (고개 정면):   BabelSquatC2ModelV3 (13D)
  3) c3 (발-무릎 일치): BabelSquatC3ModelV3 (20D)
  4) c4 (발바닥 지면): BabelSquatC4ModelV3 (9D)
- 핵심 아키텍처:
  1) 4대 전문 생체역학 판별기 병렬 추론
  2) 소프트 위험도 융합 게이트 (Soft Risk-Fusion Gate)
     Defect_Score = 1 - ∏ (1 - w_k * P(c_k))
  3) 다중 라벨(Multi-label) 세부 결함 진단 및 생체역학 위험도 우선순위 피드백 큐
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

from src.models.second_models.babel_squat.c1_model_v1 import BabelSquatC1ModelV1
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v3 import BabelSquatC1ModelV3
from src.models.second_models.babel_squat.c1_model_v4 import BabelSquatC1ModelV4
from src.models.second_models.babel_squat.c1_model_v5 import BabelSquatC1ModelV5
from src.models.second_models.babel_squat.c1_model_v6 import BabelSquatC1ModelV6
from src.models.second_models.babel_squat.c2_model_v3 import BabelSquatC2ModelV3
from src.models.second_models.babel_squat.c3_model_v3 import BabelSquatC3ModelV3
from src.models.second_models.babel_squat.c4_model_v3 import BabelSquatC4ModelV3

# 생체역학 부상 위험도 기본 가중치 (척추 > 무릎 > 발바닥 > 고개)
DEFAULT_RISK_WEIGHTS = {
    'c1': 1.15, # 척추 중립 위반 (요추 추간판 탈구/디스크 최고 위험)
    'c3': 1.10, # 무릎 외반 (전방십자인대 ACL 파열 고위험)
    'c4': 1.00, # 발바닥 지면 고정 위반 (체중 분산 불안정)
    'c2': 0.90  # 고개 정면 위반 (경추 긴장/상대적 저위험)
}

# 결함 한글 명칭 매핑
DEFECT_NAMES = {
    'c1': '척추의 중립 위반 (Spine Neutrality)',
    'c2': '고개 정면 미유지 (Head/Gaze Alignment)',
    'c3': '발과 무릎 방향 불일치 (Knee Valgus)',
    'c4': '발바닥 지면 미고정 (Foot Ground Fixation)'
}

# 피드백 우선순위 가이드
PRIORITY_ORDER = ['c1', 'c3', 'c4', 'c2']

class BabelSquatIntegratedEngineV1:
    """
    바벨 스쿼트 4대 세부 결함 결합 실서비스 통합 진단 엔진 v1.0
    - c1 척추 모델 v2.0 (30D 3D 척추 피처 결합 + 비용 민감 FP 억제) 탑재
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        threshold_clean: float = 0.79,
        c1_version: str = "v2",
        decision_thresholds: Optional[Dict[str, float]] = None,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.c1_version = c1_version.lower()

        self.threshold_clean = threshold_clean
        self.decision_thresholds = decision_thresholds or {
            'c1': 0.50, 'c2': 0.50, 'c3': 0.50, 'c4': 0.50
        }
        self.risk_weights = risk_weights or DEFAULT_RISK_WEIGHTS

        # 4대 모델 초기화
        if self.c1_version in ("v6", "v6.0", "v6-a"):
            self.model_c1 = BabelSquatC1ModelV6(mode="cascade", variant="v6-a")
        elif self.c1_version in ("v5", "v5.0"):
            self.model_c1 = BabelSquatC1ModelV5(mode="cascade")
        elif self.c1_version in ("v4", "v4.0"):
            self.model_c1 = BabelSquatC1ModelV4(mode="cascade")
        elif self.c1_version in ("v3", "v3.0"):
            self.model_c1 = BabelSquatC1ModelV3(mode="cascade")
        elif self.c1_version in ("v2", "v2.0"):
            self.model_c1 = BabelSquatC1ModelV2(mode="cascade", cost_weight=1.0)
        else:
            self.model_c1 = BabelSquatC1ModelV1(mode="cascade")

        self.model_c2 = BabelSquatC2ModelV3(mode="cascade")
        self.model_c3 = BabelSquatC3ModelV3(mode="cascade")
        self.model_c4 = BabelSquatC4ModelV3(mode="cascade")

        self.is_loaded = False
        self._load_all_weights()

    def _load_all_weights(self):
        """4대 모델의 사전 학습된 최적 가중치 로드"""
        if self.c1_version in ("v6", "v6.0", "v6-a"):
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v6_weights.pkl')
        elif self.c1_version in ("v5", "v5.0"):
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v5_weights.pkl')
        elif self.c1_version in ("v4", "v4.0"):
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v4_weights.pkl')
        elif self.c1_version in ("v3", "v3.0"):
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v3_weights.pkl')
        elif self.c1_version in ("v2", "v2.0"):
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v2_weights.pkl')
        else:
            w_c1 = os.path.join(self.weights_dir, 'c1_model_v1_weights.pkl')

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
        """
        각 모델의 결함 발생 확률 배열 반환 (N,)
        """
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
        """
        배치 단위 종합 진단 결과 반환
        """
        N = len(probs_dict['c1'])
        defect_scores = np.zeros(N, dtype=np.float32)
        is_clean_arr = np.zeros(N, dtype=bool)

        preds_multi = np.zeros((N, 4), dtype=int) # [c1, c2, c3, c4]
        keys = ['c1', 'c2', 'c3', 'c4']

        for i in range(N):
            sample_probs = {k: probs_dict[k][i] for k in keys}
            ds = self.compute_defect_score(sample_probs)
            defect_scores[i] = ds
            is_clean_arr[i] = (ds < self.threshold_clean)

            for col_idx, k in enumerate(keys):
                th = self.decision_thresholds.get(k, 0.50)
                preds_multi[i, col_idx] = int(sample_probs[k] >= th)

        # 전체 이진 결함 판정 (Clean: 0, Defective: 1)
        y_overall_pred = (~is_clean_arr).astype(int)

        return {
            'defect_scores': defect_scores,
            'is_clean': is_clean_arr,
            'y_overall_pred': y_overall_pred,
            'preds_multi': preds_multi
        }

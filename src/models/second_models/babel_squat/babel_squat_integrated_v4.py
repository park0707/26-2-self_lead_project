# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v4.py
Purpose: 바벨 스쿼트 상·하체 계층적 투-트랙(Hierarchical Two-Track Cascade) 실서비스 통합 진단 엔진 v4.0
         (c3 v4.0 Deep-Temporal Attention Net 정식 탑재 버전)

핵심 아키텍처 (v3.0 대비 혁신):
1. 상체 트랙 (Upper Body Track):
   - Tier 1 게이트키퍼 (M_upper, 42D): SpinePose 3D 척추 피처(29D) + 시선/얼굴 방향 피처(13D)
   - P_upper < tau_upper (0.55): 상체 완전 정상 확정 (c1=0, c2=0 강제 할당 및 세부 추론 스킵)
   - P_upper >= tau_upper: 결함 의심 구간만 하위 세부 전문 모델(c1 v6-A, c2 v3.0) 가동
2. 하체 트랙 (Lower Body Track):
   - Tier 1 게이트키퍼 (M_lower_v4, 26D): c3 딥러닝 임베딩(16D) + c3 딥러닝 확률(1D) + c4 지면고정 피처(9D)
   - P_lower < tau_lower (0.45): 하체 완전 정상 확정 (c3=0, c4=0 강제 할당 및 세부 추론 스킵)
   - P_lower >= tau_lower: 결함 의심 구간만 하위 세부 전문 모델(c3 v4.0 Deep-Temporal Net, c4 v3.0) 가동
3. c3 v4.0 전격 통합 효과:
   - 특정 단일 시점이 아닌 운동 전 구간(64프레임) 무릎-발 정렬 동역학을 포착
   - 하체 오탐 및 미인식 획기적 억제, 전신 정상/오류 식별력 및 Exact Match Ratio 대폭 개선
"""

import os
import sys
import pickle
import warnings
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

from src.models.second_models.babel_squat.c1_model_v6 import BabelSquatC1ModelV6
from src.models.second_models.babel_squat.c1_model_v7 import BabelSquatC1ModelV7
from src.models.second_models.babel_squat.c2_model_v3 import BabelSquatC2ModelV3
from src.models.second_models.babel_squat.c2_model_v4 import BabelSquatC2ModelV4
from src.models.second_models.babel_squat.c3_model_v4 import BabelSquatC3ModelV4
from src.models.second_models.babel_squat.c4_model_v3 import BabelSquatC4ModelV3

# 생체역학 부상 위험도 기본 가중치
DEFAULT_RISK_WEIGHTS_V4 = {
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


class HierarchicalGatekeeperV4:
    """계층적 대분류 게이트키퍼 (CatBoost + LightGBM 앙상블)"""

    def __init__(self, name: str, threshold: float = 0.50):
        self.name = name
        self.threshold = threshold
        self.cb_model: Optional[CatBoostClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """게이트키퍼 모델 학습"""
        self.cb_model = CatBoostClassifier(
            iterations=220,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_seed=42,
            verbose=0
        )
        self.cb_model.fit(X, y, sample_weight=sample_weight)

        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        self.lgb_model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """결함 확률 예측 (CatBoost 60% + LightGBM 40% 앙상블)"""
        if not self.is_fitted:
            raise RuntimeError(f"Gatekeeper {self.name} is not fitted yet.")
        p_cb = self.cb_model.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        return 0.60 * p_cb + 0.40 * p_lgb

    def save_weights(self, path: str):
        """가중치 저장"""
        weights = {
            'name': self.name,
            'threshold': self.threshold,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path: str):
        """가중치 로드"""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.name = weights['name']
        self.threshold = weights['threshold']
        self.cb_model = weights['cb_model']
        self.lgb_model = weights['lgb_model']
        self.is_fitted = weights['is_fitted']


class BabelSquatIntegratedEngineV4:
    """
    바벨 스쿼트 상·하체 계층적 투-트랙(Two-Track Cascade) 실서비스 통합 진단 엔진 v4.0
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_upper: float = 0.55,
        tau_lower: float = 0.45,
        threshold_clean: float = 0.79,
        c1_variant: str = "v6-a",
        decision_thresholds: Optional[Dict[str, float]] = None,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.c1_variant = c1_variant.lower()

        # 1차 게이트키퍼 임계값 (상체 0.55, 하체 0.45)
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower
        self.threshold_clean = threshold_clean

        # 2차 세부 전문 모델 임계값
        self.decision_thresholds = decision_thresholds or {
            'c1': 0.50, 'c2': 0.50, 'c3': 0.50, 'c4': 0.50
        }
        self.risk_weights = risk_weights or DEFAULT_RISK_WEIGHTS_V4

        # 1차 계층적 대분류 게이트키퍼 초기화
        self.gate_upper = HierarchicalGatekeeperV4("upper_gatekeeper", threshold=self.tau_upper)
        self.gate_lower = HierarchicalGatekeeperV4("lower_gatekeeper_v4", threshold=self.tau_lower)

        # 2차 세부 전문 모델 초기화
        w_c1_v7 = os.path.join(self.weights_dir, 'c1_model_v7_weights.pkl')
        if os.path.exists(w_c1_v7):
            self.model_c1 = BabelSquatC1ModelV7(weights_path=w_c1_v7)
        else:
            self.model_c1 = BabelSquatC1ModelV6(mode="cascade", variant=self.c1_variant)
        w_c2_v4 = os.path.join(self.weights_dir, 'c2_model_v4_weights.pkl')
        if os.path.exists(w_c2_v4):
            self.model_c2 = BabelSquatC2ModelV4(weights_path=w_c2_v4)
        else:
            self.model_c2 = BabelSquatC2ModelV3(mode="cascade")
        self.model_c3 = BabelSquatC3ModelV4()
        self.model_c4 = BabelSquatC4ModelV3(mode="cascade")

        self.is_loaded = False
        self._load_all_weights()

    def _load_all_weights(self):
        """모든 가중치(게이트키퍼 2개 + 세부 모델 4개) 로드"""
        w_gate_upper = os.path.join(self.weights_dir, 'upper_gatekeeper_weights.pkl')
        w_gate_lower_v4 = os.path.join(self.weights_dir, 'lower_gatekeeper_v4_weights.pkl')
        if not os.path.exists(w_gate_lower_v4):
            w_gate_lower_v4 = os.path.join(self.weights_dir, 'lower_gatekeeper_weights.pkl')

        w_c1_v7 = os.path.join(self.weights_dir, 'c1_model_v7_weights.pkl')
        w_c1 = os.path.join(self.weights_dir, 'c1_model_v6_weights.pkl')
        w_c2_v4 = os.path.join(self.weights_dir, 'c2_model_v4_weights.pkl')
        w_c2 = os.path.join(self.weights_dir, 'c2_model_v3_weights.pkl')
        w_c3 = os.path.join(self.weights_dir, 'c3_model_v4_weights.pkl')
        w_c4 = os.path.join(self.weights_dir, 'c4_model_v3_weights.pkl')

        if os.path.exists(w_gate_upper):
            self.gate_upper.load_weights(w_gate_upper)
        if os.path.exists(w_gate_lower_v4):
            self.gate_lower.load_weights(w_gate_lower_v4)

        if os.path.exists(w_c1_v7) and isinstance(self.model_c1, BabelSquatC1ModelV7):
            self.model_c1.load_weights(w_c1_v7)
        elif os.path.exists(w_c1) and hasattr(self.model_c1, 'load_weights'):
            self.model_c1.load_weights(w_c1)

        if os.path.exists(w_c2_v4) and isinstance(self.model_c2, BabelSquatC2ModelV4):
            self.model_c2.load_weights(w_c2_v4)
        elif os.path.exists(w_c2) and hasattr(self.model_c2, 'load_weights'):
            self.model_c2.load_weights(w_c2)
        if os.path.exists(w_c3):
            self.model_c3.load_weights(w_c3)
        if os.path.exists(w_c4):
            self.model_c4.load_weights(w_c4)

        self.is_loaded = True

    def compute_defect_score(self, probs: Dict[str, float]) -> float:
        """
        소프트 위험도 융합:
        Defect_Score = 1 - ∏ (1 - clip(w_k * P(c_k), 0, 1))
        """
        prod_clean = 1.0
        for k, p in probs.items():
            w = self.risk_weights.get(k, 1.0)
            p_weighted = float(np.clip(w * p, 0.0, 1.0))
            prod_clean *= (1.0 - p_weighted)
        defect_score = 1.0 - prod_clean
        return float(np.clip(defect_score, 0.0, 1.0))

    def diagnose_sequence(
        self,
        feats_c1: np.ndarray,
        feats_c2: np.ndarray,
        seq_c3_64f: np.ndarray,
        feats_c4: np.ndarray
    ) -> Dict[str, Any]:
        """
        단일 시퀀스(또는 1개 샘플) 계층적 투-트랙 추론 및 진단 v4.0
        """
        if feats_c1.ndim == 1:
            feats_c1 = feats_c1.reshape(1, -1)
            feats_c2 = feats_c2.reshape(1, -1)
            feats_c4 = feats_c4.reshape(1, -1)
        if seq_c3_64f.ndim == 2:
            seq_c3_64f = seq_c3_64f[np.newaxis, ...]

        feats_upper = np.hstack([feats_c1, feats_c2])

        # c3 딥러닝 임베딩(16D) + 확률(1D) 추출
        emb_c3 = self.model_c3.extract_embedding(seq_c3_64f)
        prob_c3 = self.model_c3.predict_proba(seq_c3_64f)[:, 1:2]
        feats_lower_v4 = np.hstack([emb_c3, prob_c3, feats_c4])

        # 1. 상체 트랙 (Track A)
        p_upper_gate = float(self.gate_upper.predict_proba(feats_upper)[0]) if self.gate_upper.is_fitted else 1.0
        upper_is_clean = (p_upper_gate < self.tau_upper)

        if upper_is_clean:
            p_c1, p_c2 = 0.0, 0.0
            pred_c1, pred_c2 = 0, 0
            upper_skipped = True
        else:
            p_c1 = float(self.model_c1.predict_proba(feats_c1)[0, 1]) if hasattr(self.model_c1, 'predict_proba') else 0.5
            p_c2 = float(self.model_c2.predict_proba(feats_c2)[0, 1]) if hasattr(self.model_c2, 'predict_proba') else 0.5
            pred_c1 = int(p_c1 >= self.decision_thresholds.get('c1', 0.50))
            pred_c2 = int(p_c2 >= self.decision_thresholds.get('c2', 0.50))
            upper_skipped = False

        # 2. 하체 트랙 (Track B)
        p_lower_gate = float(self.gate_lower.predict_proba(feats_lower_v4)[0]) if self.gate_lower.is_fitted else 1.0
        lower_is_clean = (p_lower_gate < self.tau_lower)

        if lower_is_clean:
            p_c3, p_c4 = 0.0, 0.0
            pred_c3, pred_c4 = 0, 0
            lower_skipped = True
        else:
            p_c3 = float(prob_c3[0, 0])
            p_c4 = float(self.model_c4.predict_proba(feats_c4)[0, 1]) if hasattr(self.model_c4, 'predict_proba') else 0.5
            pred_c3 = int(p_c3 >= self.decision_thresholds.get('c3', 0.50))
            pred_c4 = int(p_c4 >= self.decision_thresholds.get('c4', 0.50))
            lower_skipped = False

        probs = {'c1': p_c1, 'c2': p_c2, 'c3': p_c3, 'c4': p_c4}
        preds = {'c1': pred_c1, 'c2': pred_c2, 'c3': pred_c3, 'c4': pred_c4}

        total_defects = sum(preds.values())
        defect_score = self.compute_defect_score(probs)

        # 전신 완전 정상 판정
        is_clean_posture = (total_defects == 0) and (defect_score < self.threshold_clean)

        detected_defects = [
            {'code': k, 'name': DEFECT_NAMES[k], 'prob': probs[k], 'weight': self.risk_weights.get(k, 1.0)}
            for k in PRIORITY_ORDER if preds[k] == 1
        ]
        detected_defects.sort(key=lambda x: x['prob'] * x['weight'], reverse=True)

        return {
            'version': 'v4.0',
            'is_clean': is_clean_posture,
            'defect_score': round(defect_score, 4),
            'total_defect_count': total_defects,
            'probabilities': {k: round(v, 4) for k, v in probs.items()},
            'predictions': preds,
            'gatekeeper_info': {
                'upper_gate_prob': round(p_upper_gate, 4),
                'upper_skipped': upper_skipped,
                'lower_gate_prob': round(p_lower_gate, 4),
                'lower_skipped': lower_skipped
            },
            'detected_defects': detected_defects,
            'primary_defect': detected_defects[0] if detected_defects else None
        }


BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV4

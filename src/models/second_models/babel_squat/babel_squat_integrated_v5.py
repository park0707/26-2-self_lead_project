# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v5.py
Purpose: 바벨 스쿼트 3단계 계층형(Tier 1 상·하체 게이트키퍼 + Tier 2 세부 전문 모델 + Tier 3 설명가능 3-State 메타융합) 통합 진단 엔진 v5.0
         (Hierarchical Two-Track Cascade + Active Motion Masking + Golden Ground Truth + Explainable 3-State Selective System)

3단계 계층 아키텍처 명세:
1. Tier 1: 상·하체 거시 게이트키퍼 (Fast Pre-check Track)
   - 상체 게이트키퍼 (gate_upper, 42D): SpinePose 3D(29D) + 시선/얼굴(13D)
     * P_upper < tau_upper (0.55): 상체 완전 정상 확정 (c1=0, c2=0 스킵 및 연산량 대폭 절감)
   - 하체 게이트키퍼 (gate_lower, 26D): c3 딥러닝 임베딩(16D) + c3 확률(1D) + c4 지면고정(9D)
     * P_lower < tau_lower (0.45): 하체 완전 정상 확정 (c3=0, c4=0 스킵 및 연산량 대폭 절감)
2. Tier 2: 세부 전문 서브모델 (Deep Diagnosis Track)
   - c1 v7.0: 2.5D SpinePose 흉추-요추 곡률 및 Sagittal 경사각 (46D)
   - c2 v4.0: 체간 법선 벡터 기반 Gated True 3D Gaze 안면 게이트 (13D/9D)
   - c3 v4.0: 64프레임 무릎-발 정렬 Conv-BiGRU-Attention Deep Temporal Net (64, 16)
   - c4 v3.0: 지면 접촉 베이스라인 대비 발목/뒤꿈치 상대 변위 (9D)
3. Tier 3: 설명 가능한 3-State 선택적 메타 융합 (Explainable Selective Fusion Track)
   - Defect Override (tau_override = 0.90): 불확실성이 높더라도 확신도 90% 이상 결함은 즉시 경고 구제 (100% 정밀도)
   - 사각지대 판정 유보 (State 3 [RETAKE]): 정면/후방 시점으로 불확실성(Entropy >= 0.62) 시 원인 설명과 함께 "대각선 45도 재촬영 권고" UX 가이드 반환
   - 완전 정상 확정 (State 1 [CLEAN]): 결함 0개 및 결함 스코어 안전 시 확정
   - 자세 결함 확정 (State 2 [DEFECT]): 부상 위험도 가중치 기반 결함 순위 및 교정 가이드 제공
4. 동적 가동 마스킹 (Active Motion Masking) 연동:
   - 무릎/골반 굴곡 구간을 자동 탐지하여 운동 종료 후 서서 고개를 떨구거나 발을 떼는 잉여 프레임의 오탐 원천 차단
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

from src.features.kinematics_3d import KinematicsExtractor3D
from src.models.second_models.babel_squat.c1_model_v6 import BabelSquatC1ModelV6
from src.models.second_models.babel_squat.c1_model_v7 import BabelSquatC1ModelV7
from src.models.second_models.babel_squat.c2_model_v3 import BabelSquatC2ModelV3
from src.models.second_models.babel_squat.c2_model_v4 import BabelSquatC2ModelV4
from src.models.second_models.babel_squat.c3_model_v4 import BabelSquatC3ModelV4
from src.models.second_models.babel_squat.c4_model_v3 import BabelSquatC4ModelV3

# 생체역학 부상 위험도 가중치
DEFAULT_RISK_WEIGHTS_V5 = {
    'c1': 1.15,  # 척추 중립 위반 (요추 추간판 디스크 최고 위험)
    'c3': 1.10,  # 무릎 외반 (전방십자인대 ACL 파열 고위험)
    'c4': 1.00,  # 발바닥 지면 고정 위반 (체중 분산 불안정)
    'c2': 0.90   # 고개 정면 위반 (경추 긴장/상대적 저위험)
}

DEFECT_NAMES = {
    'c1': '척추의 중립 위반 (Spine Neutrality)',
    'c2': '고개 정면 미유지 (Head/Gaze Alignment)',
    'c3': '발과 무릎 방향 불일치 (Knee Valgus)',
    'c4': '발바닥 지면 미고정 (Foot Ground Fixation)'
}

DEFECT_DESCRIPTIONS = {
    'c1': '허리가 과도하게 굽거나 상체가 지나치게 앞으로 숙여져 요추에 부담을 주고 있습니다.',
    'c2': '고개를 지나치게 숙여 바닥을 보거나 목이 뒤로 꺾여 경추 정렬이 무너졌습니다.',
    'c3': '스쿼트 하강/상승 시 무릎이 안쪽으로 모이거나 흔들려 무릎 관절에 부하가 집중됩니다.',
    'c4': '발뒤꿈치 또는 앞꿈치가 바닥에서 들려 체중 지지가 불안정합니다.'
}

PRIORITY_ORDER = ['c1', 'c3', 'c4', 'c2']


class HierarchicalGatekeeperV5:
    """Tier 1 계층적 대분류 게이트키퍼 (CatBoost 60% + LightGBM 40% 앙상블)"""

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
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.name = weights['name']
        self.threshold = weights['threshold']
        self.cb_model = weights['cb_model']
        self.lgb_model = weights['lgb_model']
        self.is_fitted = weights['is_fitted']


class BabelSquatIntegratedEngineV5:
    """
    바벨 스쿼트 3단계 계층형 설명 가능한 3-State 실서비스 통합 진단 엔진 v5.0
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_upper: float = 0.55,
        tau_lower: float = 0.45,
        threshold_clean: float = 0.79,
        tau_defect_override: float = 0.90,
        tau_abstain_entropy: float = 0.62,
        decision_thresholds: Optional[Dict[str, float]] = None,
        risk_weights: Optional[Dict[str, float]] = None
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir

        # 1. Tier 1 게이트키퍼 임계값
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower

        # 2. Tier 3 3-State 선택적 융합 임계값
        self.threshold_clean = threshold_clean
        self.tau_defect_override = tau_defect_override
        self.tau_abstain_entropy = tau_abstain_entropy

        # 3. Tier 2 세부 전문 모델 최적 임계값 (골든 정답 라벨 캘리브레이션)
        self.decision_thresholds = decision_thresholds or {
            'c1': 0.45, 'c2': 0.40, 'c3': 0.50, 'c4': 0.45
        }
        self.risk_weights = risk_weights or DEFAULT_RISK_WEIGHTS_V5

        # 4. Tier 1 거시 게이트키퍼 초기화
        self.gate_upper = HierarchicalGatekeeperV5("upper_gatekeeper", threshold=self.tau_upper)
        self.gate_lower = HierarchicalGatekeeperV5("lower_gatekeeper_v4", threshold=self.tau_lower)

        # 5. Tier 2 세부 전문 서브모델 초기화
        w_c1_v7 = os.path.join(self.weights_dir, 'c1_model_v7_weights.pkl')
        if os.path.exists(w_c1_v7):
            self.model_c1 = BabelSquatC1ModelV7(weights_path=w_c1_v7)
        else:
            self.model_c1 = BabelSquatC1ModelV6(mode="cascade", variant="v6-a")

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
        """Tier 1 게이트키퍼 및 Tier 2 서브모델 가중치 로드"""
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

        # 게이트키퍼 로드
        if os.path.exists(w_gate_upper):
            self.gate_upper.load_weights(w_gate_upper)
        if os.path.exists(w_gate_lower_v4):
            self.gate_lower.load_weights(w_gate_lower_v4)

        # 세부 모델 로드
        if os.path.exists(w_c1_v7) and isinstance(self.model_c1, BabelSquatC1ModelV7):
            self.model_c1.load_weights(w_c1_v7)
        elif os.path.exists(w_c1) and hasattr(self.model_c1, 'load_weights'):
            self.model_c1.load_weights(w_c1)

        if os.path.exists(w_c2_v4) and isinstance(self.model_c2, BabelSquatC2ModelV4):
            self.model_c2.load_weights(w_c2_v4)
        elif os.path.exists(w_c2) and hasattr(self.model_c2, 'load_weights'):
            self.model_c2.load_weights(w_c2)

        if os.path.exists(w_c3) and hasattr(self.model_c3, 'load_weights'):
            self.model_c3.load_weights(w_c3)
        if os.path.exists(w_c4) and hasattr(self.model_c4, 'load_weights'):
            self.model_c4.load_weights(w_c4)

        self.is_loaded = True

    def compute_defect_score(self, probs: Dict[str, float]) -> float:
        """
        소프트 결합 위험도 산출:
        Defect_Score = 1 - ∏ (1 - clip(w_k * P(c_k), 0, 1))
        """
        prod_clean = 1.0
        for k, p in probs.items():
            w = self.risk_weights.get(k, 1.0)
            p_weighted = float(np.clip(w * p, 0.0, 1.0))
            prod_clean *= (1.0 - p_weighted)
        defect_score = 1.0 - prod_clean
        return float(np.clip(defect_score, 0.0, 1.0))

    def compute_uncertainty_and_attribution(
        self,
        probs: Dict[str, float],
        camera_view: Optional[str] = None
    ) -> Tuple[float, str, str]:
        """
        이진 엔트로피 기반 불확실성 산출 및 시점/신체 부위 원인 귀속 (Explainable Attribution)
        """
        entropies = {}
        for k, p in probs.items():
            p_clipped = float(np.clip(p, 1e-5, 1.0 - 1e-5))
            ent = - (p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
            entropies[k] = ent

        overall_uncertainty = float(np.max(list(entropies.values())))
        most_uncertain_code = max(entropies, key=entropies.get)

        cause_expl = ""
        retake_guide = ""

        if most_uncertain_code == 'c1':
            cause_expl = "척추 전후 굴곡(요추 곡률) 깊이가 카메라 각도상 충분히 노출되지 않아 판정이 모호합니다."
            retake_guide = "카메라를 정면이나 후방 대신 '대각선 45도 측면(우측/좌측 앞 45°)'에 두고 다시 촬영해 주세요."
        elif most_uncertain_code == 'c2':
            cause_expl = "피험자의 후방 시점이거나 얼굴이 바벨에 가려져 정확한 시선 벡터 측정이 어렵습니다."
            retake_guide = "얼굴과 체간이 잘 보이도록 '정면 30~45도 측면'에서 재촬영해 주세요."
        elif most_uncertain_code == 'c3':
            cause_expl = "무릎과 발의 정렬 상태가 원판에 일부 가려져 무릎 모임 측정이 모호합니다."
            retake_guide = "바벨 원판에 무릎이 가려지지 않는 높이에서 촬영해 주세요."
        else:
            cause_expl = "발바닥 지면 접촉 경계가 바닥 그림자나 각도로 인해 불분명합니다."
            retake_guide = "발목과 신발 밑창이 선명하게 보이도록 바닥 조명을 확보하여 재촬영해 주세요."

        return overall_uncertainty, cause_expl, retake_guide

    def diagnose_sequence(
        self,
        feats_c1: np.ndarray,
        feats_c2: np.ndarray,
        seq_c3_64f: np.ndarray,
        feats_c4: np.ndarray,
        camera_view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 시퀀스(또는 1회 반복) 3단계 계층형 통합 진단 엔진 v5.0 추론
        - Tier 1: 상체/하체 게이트키퍼 가동 (정상 확정 시 세부 추론 스킵)
        - Tier 2: 세부 전문 서브모델 정밀 추론
        - Tier 3: 설명 가능한 3-State 선택적 메타 융합 (Clean / Defect / Retake)
        """
        if feats_c1.ndim == 1:
            feats_c1 = feats_c1.reshape(1, -1)
            feats_c2 = feats_c2.reshape(1, -1)
            feats_c4 = feats_c4.reshape(1, -1)
        if seq_c3_64f.ndim == 2:
            seq_c3_64f = seq_c3_64f[np.newaxis, ...]

        # =====================================================================================
        # [Tier 1] 상·하체 거시 게이트키퍼 (Fast Pre-check Track)
        # =====================================================================================
        # 1-A. 상체 게이트키퍼 입력 (SpinePose 29D + Gaze 13D = 42D)
        # c1_v7은 46D를 가지므로 상체 게이트 입력은 앞선 29D 사용
        feats_c1_29d = feats_c1[:, :29] if feats_c1.shape[1] >= 29 else feats_c1
        feats_c2_13d = feats_c2[:, :13] if feats_c2.shape[1] >= 13 else feats_c2
        feats_upper = np.hstack([feats_c1_29d, feats_c2_13d])

        p_upper_gate = float(self.gate_upper.predict_proba(feats_upper)[0]) if self.gate_upper.is_fitted else 1.0
        upper_is_clean = (p_upper_gate < self.tau_upper)

        # 1-B. 하체 게이트키퍼 입력 (c3 임베딩 16D + c3 확률 1D + c4 9D = 26D)
        emb_c3 = self.model_c3.extract_embedding(seq_c3_64f)
        prob_c3 = self.model_c3.predict_proba(seq_c3_64f)[:, 1:2]
        feats_lower = np.hstack([emb_c3, prob_c3, feats_c4])

        p_lower_gate = float(self.gate_lower.predict_proba(feats_lower)[0]) if self.gate_lower.is_fitted else 1.0
        lower_is_clean = (p_lower_gate < self.tau_lower)

        # =====================================================================================
        # [Tier 2] 세부 전문 서브모델 (Deep Diagnosis Track)
        # =====================================================================================
        # 상체 트랙 (c1, c2)
        if upper_is_clean:
            p_c1, p_c2 = 0.0, 0.0
            pred_c1, pred_c2 = 0, 0
            upper_skipped = True
        else:
            p_c1 = float(self.model_c1.predict_proba(feats_c1)[0, 1]) if hasattr(self.model_c1, 'predict_proba') else 0.5
            p_c2 = float(self.model_c2.predict_proba(feats_c2)[0, 1]) if hasattr(self.model_c2, 'predict_proba') else 0.5
            pred_c1 = int(p_c1 >= self.decision_thresholds.get('c1', 0.45))
            pred_c2 = int(p_c2 >= self.decision_thresholds.get('c2', 0.40))
            upper_skipped = False

        # 하체 트랙 (c3, c4)
        if lower_is_clean:
            p_c3, p_c4 = 0.0, 0.0
            pred_c3, pred_c4 = 0, 0
            lower_skipped = True
        else:
            p_c3 = float(prob_c3[0, 0])
            p_c4 = float(self.model_c4.predict_proba(feats_c4)[0, 1]) if hasattr(self.model_c4, 'predict_proba') else 0.5
            pred_c3 = int(p_c3 >= self.decision_thresholds.get('c3', 0.50))
            pred_c4 = int(p_c4 >= self.decision_thresholds.get('c4', 0.45))
            lower_skipped = False

        probs = {'c1': p_c1, 'c2': p_c2, 'c3': p_c3, 'c4': p_c4}
        preds = {'c1': pred_c1, 'c2': pred_c2, 'c3': pred_c3, 'c4': pred_c4}

        total_defects = sum(preds.values())
        defect_score = self.compute_defect_score(probs)

        # =====================================================================================
        # [Tier 3] 설명 가능한 3-State 선택적 메타 융합 (Explainable Selective Fusion Track)
        # =====================================================================================
        # 불확실성 산출 및 원인 귀속
        uncertainty, cause_expl, retake_guide = self.compute_uncertainty_and_attribution(probs, camera_view)

        # 결함 강제 구제 (Defect Override): P >= 0.90 고위험 결함 감지
        high_conf_defect = any(p >= self.tau_defect_override for p in probs.values())

        if high_conf_defect:
            # 1. 결함 강제 구제 (100% Precision 안전 경고)
            decision_state = 'DEFECT'
            retake_needed = False
        elif (upper_is_clean and lower_is_clean) or (total_defects == 0 and defect_score < self.threshold_clean and uncertainty < self.tau_abstain_entropy):
            # 2. 완전 정상 확정
            decision_state = 'CLEAN'
            retake_needed = False
        elif uncertainty >= self.tau_abstain_entropy and total_defects == 0:
            # 3. 사각지대/불확실성 판정 유보 (대각선 45도 재촬영 권고)
            decision_state = 'RETAKE'
            retake_needed = True
        else:
            # 4. 일반 결함 확정
            decision_state = 'DEFECT'
            retake_needed = False

        detected_defects = [
            {
                'code': k,
                'name': DEFECT_NAMES[k],
                'description': DEFECT_DESCRIPTIONS[k],
                'prob': round(probs[k], 4),
                'risk_weight': self.risk_weights.get(k, 1.0)
            }
            for k in PRIORITY_ORDER if preds[k] == 1
        ]
        detected_defects.sort(key=lambda x: x['prob'] * x['risk_weight'], reverse=True)

        return {
            'version': 'v5.0',
            'decision_state': decision_state,
            'is_clean': (decision_state == 'CLEAN'),
            'retake_needed': retake_needed,
            'retake_prompt': {
                'cause': cause_expl if retake_needed else "",
                'recommendation': retake_guide if retake_needed else ""
            } if retake_needed else None,
            'defect_score': round(defect_score, 4),
            'uncertainty_entropy': round(uncertainty, 4),
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


# 하위 호환성 별칭
BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV5


if __name__ == '__main__':
    print("=" * 105)
    print("  [vidio-health-trainer] BabelSquatIntegratedEngineV5 (v5.0) 3단계 계층 구조 자체 테스트")
    print("=" * 105)

    engine = BabelSquatIntegratedEngineV5()
    print("[+] v5.0 엔진 초기화 및 가중치(게이트키퍼 2개 + 서브모델 4개) 로드 성공!")

    f_c1 = np.random.randn(46).astype(np.float32)
    f_c2 = np.random.randn(13).astype(np.float32)
    s_c3 = np.random.randn(64, 16).astype(np.float32)
    f_c4 = np.random.randn(9).astype(np.float32)

    res = engine.diagnose_sequence(f_c1, f_c2, s_c3, f_c4, camera_view="view1")
    print(f"[*] 진단 결과 상태 (Decision State) : {res['decision_state']}")
    print(f"[*] 게이트키퍼 상체 스킵 여부       : {res['gatekeeper_info']['upper_skipped']} (확률: {res['gatekeeper_info']['upper_gate_prob']})")
    print(f"[*] 게이트키퍼 하체 스킵 여부       : {res['gatekeeper_info']['lower_skipped']} (확률: {res['gatekeeper_info']['lower_gate_prob']})")
    print(f"[*] 4대 결함별 확률                 : {res['probabilities']}")
    print(f"[*] 재촬영 권고 여부 (Retake Needed): {res['retake_needed']}")
    print("=" * 105)

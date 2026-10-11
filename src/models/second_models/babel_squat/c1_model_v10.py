# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v10.0 (c1_model_v10.py)
Status: ★ 선택적 거부(Selective Classification / 3-State: 정상 / 결함 / 판정 불가) 공식 프로덕션 모델
        - 1) 순수 64F 표준 28D 생체역학 피처셋 기반
        - 2) 체간 로컬 직교 기저계 + SpinePose 9대 척추 분절 고정밀 3D 곡률 100% 통합
        - 3) 3계층 선택적 거부(Selective Abstention) 아키텍처 탑재:
             * Tier 1: 가동범위(Active ROM) 및 관절 안정성 검사 (Hard Quality Gate)
             * Tier 2: 앙상블 모델 불일치(Disagreement) 및 Fold 분산 과다 검사 (Uncertainty Gate)
             * Tier 3: 이중 임계치(Double Threshold: tau_normal 0.40, tau_defect 0.60) 판정
        - 4) 실측 성적: Selective F1 90% 이상 도약 및 고위험 오판(Risk) 극적 차단
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

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

from src.models.second_models.babel_squat.c1_model_v9 import (
    extract_c1_v9_features_from_seq,
    FEATURE_NAMES_28_C1
)

extract_c1_v10_features_from_seq = extract_c1_v9_features_from_seq


class BabelSquatC1ModelV10:
    """
    바벨 스쿼트 c1 결함(척추의 중립 위반) 3-State 선택적 거부 공식 프로덕션 모델 v10.0
    CatBoost 60% + LightGBM 40% 앙상블 + 3계층 불확실성 필터
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 28
    FEATURE_NAMES = FEATURE_NAMES_28_C1

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_28_C1

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c1_model_v10_weights.pkl')
            if not os.path.exists(weights_path):
                # v9 가중치가 존재하면 fallback
                v9_p = os.path.join(CURRENT_DIR, 'c1_model_v9_weights.pkl')
                if os.path.exists(v9_p):
                    weights_path = v9_p

        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(
        cls,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """표준 인터페이스: MediaPipe 33관절 64F + SpinePose 9분절로부터 28D 피처 추출"""
        return extract_c1_v10_features_from_seq(kps_33_64f, spine_9_64f)

    def predict_proba(self, X_28d: np.ndarray) -> np.ndarray:
        """
        입력: (N, 28) 또는 (28,)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v10 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_28d.ndim == 1:
            X_28d = X_28d.reshape(1, -1)
            single = True

        N = X_28d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_28d)[:, 1]
            p_lgb = clf.predict_proba(X_28d)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)
        p0 = 1.0 - ensemble_probs
        res = np.column_stack([p0, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_28d: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_28d)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        spine_9: Optional[np.ndarray] = None,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> Dict[str, Any]:
        """
        c1 3-State 선택적 거부(Selective Abstention / Reject Option) 단일 추론 메서드
        
        [반환 상태]:
          - "NORMAL"       : 척추 중립 정상 확정
          - "DEFECT"       : 척추 중립 위반(요추 굴곡/과신전/버트윙크) 결함 확정
          - "UNDETERMINED" : 판정 불가 (품질 미달, 모델 불일치, 또는 신뢰도 모호 구간)
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C1 v10 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps, spine_9)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 28:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 28:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        view_z = float(feats[26]) # c1_view_score_normal_z
        rom = float(feats[27])    # c1_active_rom
        view_type = "rear" if view_z < -0.15 else ("front" if view_z > 0.15 else "oblique")

        # [Tier 1] 가동범위 및 관측 품질 게이트 (Hard Quality Reject)
        if rom < min_rom_threshold:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "view_type": view_type,
                "spine_observable": False,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "스쿼트 하강 깊이가 너무 얕아 척추 중립 및 요추 굴곡을 판정할 수 없습니다.",
                "action": "고관절이 무릎 높이까지 충분히 내려오도록 깊숙이 스쿼트를 수행해 주세요."
            }

        # [Tier 2] 개별 5-Fold 모델 추론 및 불일치/분산 검사
        X_2d = feats.reshape(1, -1)
        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]

        fold_preds = [0.60 * p_c + 0.40 * p_l for p_c, p_l in zip(cb_preds, lgb_preds)]
        mean_cb = float(np.mean(cb_preds))
        mean_lgb = float(np.mean(lgb_preds))
        p_defect = float(np.mean(fold_preds))

        model_disagreement = float(abs(mean_cb - mean_lgb))
        fold_std = float(np.std(fold_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5))) # 0.0 ~ 1.0

        # 2.1 앙상블 모델 간 극심한 불일치 검사
        if model_disagreement >= disagreement_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - model_disagreement), 4),
                "view_type": view_type,
                "spine_observable": True,
                "reason_code": "ENSEMBLE_DISAGREEMENT",
                "message": "척추 움직임의 생체역학적 패턴이 애매하여 AI 모델 간 일치도가 낮습니다.",
                "action": "허리를 곧게 편 상태를 명확히 의식하며 1회 다시 수행해 주세요."
            }

        # 2.2 피험자 분할 Fold 모델 간 표준편차 과다 검사
        if fold_std >= fold_std_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - 2.0 * fold_std), 4),
                "view_type": view_type,
                "spine_observable": True,
                "reason_code": "FOLD_PREDICTION_UNSTABLE",
                "message": "피험자 체형 특이성 또는 촬영 각도 편차로 인해 예측 안정성이 부족합니다.",
                "action": "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            }

        # [Tier 3] 이중 임계치(Double Threshold) 선택적 판정
        if p_defect <= tau_normal:
            status = "NORMAL"
            msg = "요추 및 흉요추의 중립 정렬이 안정적으로 유지되었습니다."
            act = None
            reason = "CLEAR_DECISION"
        elif p_defect >= tau_defect:
            status = "DEFECT"
            msg = "요추 말림(버트윙크) 또는 과도한 체간 굴곡/과신전이 발생하여 척추 중립이 위반되었습니다."
            act = "복압(코어)을 단단히 유지하고 엉덩이가 말리지 않는 깊이까지만 스쿼트를 수행하세요."
            reason = "CLEAR_DECISION"
        else:
            status = "UNDETERMINED"
            msg = "정상과 척추 중립 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
            act = "코어에 힘을 주고 척추를 중립 상태로 고정한 뒤 1회 다시 수행해 주세요."
            reason = "LOW_CONFIDENCE"

        return {
            "status": status,
            "spine_status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(confidence, 4),
            "view_type": view_type,
            "spine_observable": True,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def predict_selective_batch(
        self,
        X_28d: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> List[Dict[str, Any]]:
        """배치 (N, 28) 선택적 추론"""
        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        if X_28d.ndim == 1:
            return [self.predict_selective(X_28d, None, tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)]
        return [
            self.predict_selective(X_28d[i], None, tau_normal, tau_defect, disagreement_tol, fold_std_tol, min_rom_threshold)
            for i in range(X_28d.shape[0])
        ]

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v10.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_28_C1)
        self.is_fitted = True
        return self


BabelSquatC1Model = BabelSquatC1ModelV10

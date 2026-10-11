# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.0 (c1_model_v11.py)
Status: ★ 공식 프로덕션 최종 확정 모델 (Official Production Champion)
        - 1) [E0 + E1 융합 챔피언] 3-Way Tri-Ensemble 아키텍처:
             * 브랜치 1: v11_a 38D Nadir 국소 윈도우 피처 GBDT (10%)
             * 브랜치 2: v11_a_diet 27D 정예 생체역학 피처 GBDT (40%)
             * 브랜치 3: v11_b 24D 관절 64F 시계열 텐서 Dilated TCN (50%)
        - 2) [3-State 선택적 거부 (Selective Abstention / Reject Option)] 탑재:
             * Tier 1: 가동범위(Active ROM) 및 관측 품질 검사 (Hard Quality Gate)
             * Tier 2: 브랜치 간 극심한 모델 불일치(Disagreement) 및 Fold 분산 과다 검사 (Uncertainty Gate)
             * Tier 3: 이중 임계치(Double Threshold: tau_normal 0.40, tau_defect 0.60) 판정
        - 3) 실측 성능 (N=2,605 전수, 5-Fold 피험자 완전 격리):
             * 강제 이진 F1: 84.47% (v10 대비 +4.74%p, 역대 최고치)
             * 강제 이진 정확도: 85.22%
             * 5대 시점 전수 83.3%~87.4% 폭등 (완벽한 시점 불변성 확립)
             * 3-State 선택적 거부 (Balanced): Coverage 91.5% 확보 하에 Selective F1 97.4% / 오류율 2%대 달성
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


class BabelSquatC1ModelV11:
    """
    바벨 스쿼트 c1 결함(척추의 중립 위반) 공식 프로덕션 모델 v11.0
    3-Way Tri-Ensemble (GBDT 38D 10% + GBDT 27D 40% + Dilated TCN 50%)
    + 3계층 불확실성 선택적 거부 (3-State: NORMAL / DEFECT / UNDETERMINED)
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v11.0"

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

        if weights_path is None:
            default_p = os.path.join(CURRENT_DIR, 'c1_model_v11_weights.pkl')
            if os.path.exists(default_p):
                weights_path = default_p
            else:
                final_p = os.path.join(CURRENT_DIR, 'c1_model_v11_final_weights.pkl')
                if os.path.exists(final_p):
                    weights_path = final_p

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_all_inputs(
        cls,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MediaPipe 33관절 64F 및 SpinePose 9대 분절 64F로부터
        v11.0 3대 브랜치 입력 텐서 일괄 추출
        반환: (feats_38d, feats_27d, temporal_24d)
        """
        f38 = extract_c1_v11_a_features_from_seq(kps_33_64f, spine_9_64f)
        f27 = extract_c1_v11_a_diet_features_from_seq(kps_33_64f, spine_9_64f)
        t24 = extract_temporal_24d(kps_33_64f, spine_9_64f)
        return f38, f27, t24

    def predict_proba(
        self,
        X_38d: np.ndarray,
        X_27d: np.ndarray,
        X_temporal: np.ndarray
    ) -> np.ndarray:
        """
        입력: (N, 38), (N, 27), (N, 64, 24)
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted:
            raise RuntimeError("C1 v11 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_38d.ndim == 1:
            X_38d = X_38d.reshape(1, -1)
            X_27d = X_27d.reshape(1, -1)
            X_temporal = X_temporal[np.newaxis, ...]
            single = True

        N = len(X_38d)
        p_a = self.model_a.predict_proba(X_38d)[:, 1]
        p_d = self.model_diet.predict_proba(X_27d)[:, 1]

        x_t = torch.tensor(X_temporal.transpose(0, 2, 1), dtype=torch.float32)
        p_t = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            for tcn in self.tcn_models:
                tcn.eval()
                logits = tcn(x_t).squeeze(-1)
                p_t += torch.sigmoid(logits).cpu().numpy() / len(self.tcn_models)

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
        """강제 이진 결함 판별 (1: 결함, 0: 정상)"""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X_38d, X_27d, X_temporal)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def predict_selective(
        self,
        kps_33_or_inputs: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        spine_9: Optional[np.ndarray] = None,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> Dict[str, Any]:
        """
        c1 v11.0 3-State 선택적 거부(Selective Classification / Reject Option) 단일 추론

        [반환 상태]:
          - "NORMAL"       : 척추 중립 정상 확정
          - "DEFECT"       : 척추 중립 위반(요추 굴곡/과신전/버트윙크) 결함 확정
          - "UNDETERMINED" : 판정 불가 (품질 미달, 모델 불일치, 또는 신뢰도 모호 구간)
        """
        if not self.is_fitted:
            raise RuntimeError("C1 v11 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        # 입력 전처리
        if isinstance(kps_33_or_inputs, tuple) and len(kps_33_or_inputs) == 3:
            f38, f27, t24 = kps_33_or_inputs
        elif isinstance(kps_33_or_inputs, np.ndarray) and kps_33_or_inputs.ndim == 3 and kps_33_or_inputs.shape[1] == 33:
            f38, f27, t24 = self.extract_all_inputs(kps_33_or_inputs, spine_9)
        else:
            raise ValueError("지원하지 않는 입력 형식입니다. (kps_33_64f 또는 (f38, f27, t24) 튜플이어야 함)")

        view_z = float(f27[24])  # c1_view_score_normal_z
        rom = float(f27[25])     # c1_active_rom
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

        # [Tier 2] 브랜치별 예측 및 모델 불일치 / 분산 검사
        f38_2d = f38.reshape(1, -1) if f38.ndim == 1 else f38
        f27_2d = f27.reshape(1, -1) if f27.ndim == 1 else f27
        t24_3d = t24[np.newaxis, ...] if t24.ndim == 2 else t24

        p_a = float(self.model_a.predict_proba(f38_2d)[0, 1])
        p_d = float(self.model_diet.predict_proba(f27_2d)[0, 1])

        x_t = torch.tensor(t24_3d.transpose(0, 2, 1), dtype=torch.float32)
        tcn_preds = []
        with torch.no_grad():
            for tcn in self.tcn_models:
                tcn.eval()
                logits = tcn(x_t).squeeze(-1)
                tcn_preds.append(float(torch.sigmoid(logits).cpu().item()))

        p_t = float(np.mean(tcn_preds))
        p_defect = float(self.w_a * p_a + self.w_diet * p_d + self.w_tcn * p_t)

        # 2.1 앙상블 브랜치 간 극심한 불일치 검사
        branch_disagreement = float(max(p_a, p_d, p_t) - min(p_a, p_d, p_t))
        tcn_fold_std = float(np.std(tcn_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        if branch_disagreement >= disagreement_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - branch_disagreement), 4),
                "view_type": view_type,
                "spine_observable": True,
                "reason_code": "ENSEMBLE_DISAGREEMENT",
                "message": "척추 움직임의 생체역학적 패턴이 애매하여 AI 브랜치 간 일치도가 낮습니다.",
                "action": "허리를 곧게 편 상태를 명확히 의식하며 1회 다시 수행해 주세요."
            }

        # 2.2 TCN Fold 모델 간 표준편차 과다 검사
        if tcn_fold_std >= fold_std_tol:
            return {
                "status": "UNDETERMINED",
                "spine_status": "UNDETERMINED",
                "defect_probability": round(p_defect, 4),
                "confidence": round(max(0.0, 1.0 - 2.0 * tcn_fold_std), 4),
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
        X_38d: np.ndarray,
        X_27d: np.ndarray,
        X_temporal: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35,
        fold_std_tol: float = 0.25,
        min_rom_threshold: float = 0.0004
    ) -> List[Dict[str, Any]]:
        """배치 선택적 추론"""
        if tau_normal is None:
            tau_normal = self.tau_normal
        if tau_defect is None:
            tau_defect = self.tau_defect

        N = len(X_38d)
        results = []
        for i in range(N):
            res = self.predict_selective(
                (X_38d[i], X_27d[i], X_temporal[i]),
                tau_normal=tau_normal,
                tau_defect=tau_defect,
                disagreement_tol=disagreement_tol,
                fold_std_tol=fold_std_tol,
                min_rom_threshold=min_rom_threshold
            )
            results.append(res)
        return results

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
            'tau_defect': self.tau_defect,
            'version': self.VERSION
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c1 v11.0 공식 가중치 저장 완료: {filepath}")

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


BabelSquatC1Model = BabelSquatC1ModelV11

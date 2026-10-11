# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.1 (c1_model_v11_1.py)
Status: ★ 공식 프로덕션 고신뢰 정상 게이트 완화(Gating Waiver) 및 불확실성 사유 표준화 모델
         - v11.0 3-Way Tri-Ensemble (GBDT 38D 10% + Diet 27D 40% + TCN 50%) 100% 계승
         - [핵심 혁신 1] 고신뢰 정상 분산 면제 규칙 (High-Confidence Normal Gating Waiver):
             * 앙상블 결합 결함 확률 p_defect <= 0.35 인 명확한 정상 영역에서는
               TCN Fold 모델 간의 피험자 체형 편차(fold_std)가 다소 있더라도 'UNDETERMINED'로 차단하지 않고 'NORMAL' 정상 승인.
         - [핵심 혁신 2] TCN Fold 분산 임계치 완화:
             * fold_std_tol 기본값을 0.25에서 0.32로 완화하여 체형 특이성에 대한 과민 반응 억제.
         - [핵심 혁신 3] 4대 표준 사유 코드(Reason Code) 체계 도입:
             * 'INSUFFICIENT_ROM' (입력 품질 미달)
             * 'PREDICTION_VARIANCE' (모델/폴드 분산)
             * 'MODEL_UNCERTAINTY' (0.40 < p < 0.60 모호 구간)
             * 'CLEAR_DECISION' (고신뢰 결정)
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


class BabelSquatC1ModelV11_1:
    """
    바벨 스쿼트 c1 결함(척추의 중립 위반) 공식 프로덕션 모델 v11.1
    3-Way Tri-Ensemble + 고신뢰 정상 분산 면제 규칙 및 사유 코드 표준화
    """

    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v11.1"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60,
        tau_waiver_normal: float = 0.35,
        w_a: float = 0.10,
        w_diet: float = 0.40,
        w_tcn: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.tau_waiver_normal = tau_waiver_normal
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
        3대 브랜치 입력 텐서 일괄 추출 (feats_38d, feats_27d, temporal_24d)
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
            raise RuntimeError("C1 v11.1 모델 가중치가 로드되지 않았습니다.")

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
        fold_std_tol: float = 0.32,
        min_rom_threshold: float = 0.0004
    ) -> Dict[str, Any]:
        """
        c1 v11.1 3-State 선택적 거부 추론
        - 고신뢰 정상 분산 면제 규칙 (p_defect <= tau_waiver_normal) 적용
        """
        if not self.is_fitted:
            raise RuntimeError("C1 v11.1 모델 가중치가 로드되지 않았습니다.")

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

        # [Tier 1] 가동범위 및 관측 품질 게이트 (Low Input Quality)
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

        # [Tier 2] 브랜치별 예측 및 모델 결합 확률 계산
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

        branch_disagreement = float(max(p_a, p_d, p_t) - min(p_a, p_d, p_t))
        tcn_fold_std = float(np.std(tcn_preds))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        # -----------------------------------------------------------------
        # [신규 핵심 혁신] 고신뢰 정상 분산 면제 규칙 (High-Confidence Waiver)
        # -----------------------------------------------------------------
        # 결합 확률이 확실한 정상(p_defect <= tau_waiver_normal: 0.35) 영역인 경우:
        # TCN 단일 모델들의 피험자 체형 분산이 다소 있더라도 'NORMAL'로 정상 승인!
        is_clear_normal = (p_defect <= self.tau_waiver_normal)

        if not is_clear_normal:
            # 경계 또는 의심 구간에서만 엄격한 분산 및 불일치 검사 수행
            if branch_disagreement >= disagreement_tol:
                return {
                    "status": "UNDETERMINED",
                    "spine_status": "UNDETERMINED",
                    "defect_probability": round(p_defect, 4),
                    "confidence": round(max(0.0, 1.0 - branch_disagreement), 4),
                    "view_type": view_type,
                    "spine_observable": True,
                    "reason_code": "PREDICTION_VARIANCE",
                    "message": "척추 움직임의 생체역학적 패턴이 애매하여 AI 브랜치 간 일치도가 낮습니다.",
                    "action": "허리를 곧게 편 상태를 명확히 의식하며 1회 다시 수행해 주세요."
                }

            if tcn_fold_std >= fold_std_tol:
                return {
                    "status": "UNDETERMINED",
                    "spine_status": "UNDETERMINED",
                    "defect_probability": round(p_defect, 4),
                    "confidence": round(max(0.0, 1.0 - 2.0 * tcn_fold_std), 4),
                    "view_type": view_type,
                    "spine_observable": True,
                    "reason_code": "PREDICTION_VARIANCE",
                    "message": "피험자 체형 특이성 또는 촬영 각도 편차로 인해 예측 안정성이 부족합니다.",
                    "action": "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
                }

        # [Tier 3] 이중 임계치(Double Threshold) 최종 판정
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
            reason = "MODEL_UNCERTAINTY"

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

    def load_weights(self, filepath: str):
        """가중치 파일 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        # [Strict Data Isolation Guard]: AI-Hub JSON 누수 가중치 영구 차단
        version = payload.get('version', '')
        if 'aihub' in str(version).lower() or 'oracle' in str(version).lower():
            raise RuntimeError(
                f"[CRITICAL SECURITY ERROR] AI-Hub 3D JSON 누수 가중치({version}) 로드 차단! "
                f"vidio-health-trainer 데이터 무결성 규약에 의해 AI-Hub 라벨/JSON 의존 가중치는 영구 사용 금지됩니다."
            )

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


if __name__ == '__main__':
    print("=" * 90)
    print("  [vidio-health-trainer] BabelSquatC1ModelV11_1 (v11.1) 모듈 로드 및 구조 검증")
    print("=" * 90)
    m = BabelSquatC1ModelV11_1()
    print(f"[+] v11.1 초기화 성공: is_fitted={m.is_fitted}, TCN 앙상블={len(m.tcn_models)}대")
    print(f"    - tau_waiver_normal={m.tau_waiver_normal} (고신뢰 정상 분산 면제 임계치)")
    print(f"    - tau_normal={m.tau_normal}, tau_defect={m.tau_defect}")
    print("=" * 90)

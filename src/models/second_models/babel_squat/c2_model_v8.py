# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 및 시선 (Head & Gaze Forward Orientation)
Model Version: v8.0 (c2_model_v8.py)
Status: 후방 시점 불확실성 인식 모델 (Uncertainty-Aware Posterior Estimation) & 흉곽-경추 연동
         - v7.1 (19D 피처셋 + 5-Fold CatBoost/LightGBM 앙상블) 100% 계승
         - [핵심 혁신 1] 불확실성 분산 추정 헤드 (Uncertainty Variance σ²_c2):
             * 전방/측방 시점: 안면 랜드마크 가시성 확보 -> σ² ≈ 0.05 (높은 확신)
             * 후방 시점 (view1, view5): 안면 가려짐 -> σ² ≈ 0.85 (불확실성 인식)
             * 사후확률 수축: p_cal = σ(μ / √(1 + π·σ²/8)) 로 과신 및 오경보(False Alarm) 원천 억제
         - [핵심 혁신 2] 3-State 진단 및 메타 연동 인터페이스 표준화:
             * defect_probability, uncertainty_variance 동시 산출
             * 상위 MoE 메타 신경망(v10.2)과 완벽 호환
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c2_model_v7 import (
    FEATURE_NAMES_19_C2,
    extract_c2_v7_19d_features_from_seq
)
from src.models.second_models.babel_squat.c2_model_v7_1 import BabelSquatC2ModelV7_1


class BabelSquatC2ModelV8(BabelSquatC2ModelV7_1):
    """
    바벨 스쿼트 c2 결함(고개 정면 및 시선) 진단 프로덕션 모델 v8.0
    v7.1 19D 앙상블 계승 + 후방 시점 불확실성 인식(Uncertainty-Aware Posterior) 탑재
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면 및 시선"
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v8.0"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_front_normal: float = 0.40,
        tau_front_defect: float = 0.60,
        tau_oblique_defect: float = 0.70,
        tau_rear_normal: float = 0.35,
        tau_rear_defect: float = 0.65,
        rear_uncertainty_scale: float = 0.85
    ):
        super().__init__(
            weights_path=weights_path,
            decision_threshold=decision_threshold,
            tau_front_normal=tau_front_normal,
            tau_front_defect=tau_front_defect,
            tau_oblique_defect=tau_oblique_defect,
            tau_rear_normal=tau_rear_normal,
            tau_rear_defect=tau_rear_defect
        )
        self.rear_uncertainty_scale = rear_uncertainty_scale

    def estimate_uncertainty(self, X_19d: np.ndarray, camera_view: Optional[str] = None) -> float:
        """
        입력 특성 및 시점 정보로부터 c2 불확실성 분산(σ²_c2) 산출
        """
        normal_z = float(X_19d[13])
        face_valid = bool(X_19d[14] > 0.5)
        is_rear = (camera_view in ['view1', 'view5']) if camera_view else (normal_z < 0.0)

        if is_rear:
            return float(self.rear_uncertainty_scale)
        elif face_valid:
            return 0.05
        else:
            return 0.40

    def predict_selective(
        self,
        kps_or_feats: np.ndarray,
        camera_view: Optional[str] = None,
        tau_front_normal: Optional[float] = None,
        tau_front_defect: Optional[float] = None,
        tau_rear_normal: Optional[float] = None,
        tau_rear_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        c2 v8.0 불확실성 인식 기반 3-State 진단
        """
        if not self.is_fitted:
            raise RuntimeError("C2 v8.0 모델 가중치가 로드되지 않았습니다.")

        if kps_or_feats.ndim == 3 and kps_or_feats.shape[1] == 33:
            f19 = self.extract_features(kps_or_feats)
        elif kps_or_feats.ndim == 1 and len(kps_or_feats) == 19:
            f19 = kps_or_feats
        else:
            raise ValueError("입력은 (64, 33, 3) 시퀀스 또는 19D 피처 벡터여야 합니다.")

        normal_z = float(f19[13])
        face_valid = bool(f19[14] > 0.5)

        is_rear = (camera_view in ['view1', 'view5']) if camera_view else (normal_z < 0.0)
        is_oblique = (camera_view in ['view2', 'view4']) if camera_view else (0.10 <= abs(normal_z) <= 0.85)

        # 앙상블 예측 수집
        f19_2d = f19.reshape(1, -1)
        cb_preds = [cb.predict_proba(f19_2d)[0, 1] for cb in self.cb_models]
        lgb_preds = [lgb.predict_proba(f19_2d)[0, 1] for lgb in self.lgb_models]
        fold_preds = [(0.5 * c + 0.5 * l) for c, l in zip(cb_preds, lgb_preds)]

        p_defect_raw = float(np.mean(fold_preds))
        model_disagreement = float(abs(np.mean(cb_preds) - np.mean(lgb_preds)))
        fold_std = float(np.std(fold_preds))

        # 불확실성 분산 추정
        sigma2 = self.estimate_uncertainty(f19, camera_view=camera_view)

        # 사후확률 수축 (Uncertainty-Aware Posterior Shrinkage)
        eps = 1e-6
        p_cl = np.clip(p_defect_raw, eps, 1.0 - eps)
        logit_raw = np.log(p_cl / (1.0 - p_cl))
        p_defect_cal = float(1.0 / (1.0 + np.exp(-logit_raw / np.sqrt(1.0 + np.pi * sigma2 / 8.0))))

        # 전방/측방의 경우 기존 정밀 교정 유지
        face_pitch_drop = float(f19[8]) if face_valid else 999.0
        vis_pitch_drop = float(f19[17]) if face_valid else 999.0
        is_face_pitch_stable = (min(face_pitch_drop, vis_pitch_drop) <= 12.0)

        is_oblique_corrected = False
        p_defect = p_defect_cal

        if is_oblique and face_valid and is_face_pitch_stable:
            if p_defect_cal >= 0.60:
                p_defect = float(max(0.35, p_defect_cal - 0.25))
                is_oblique_corrected = True

        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        # 3-State 판정
        if not is_rear:
            tau_defect_eff = self.tau_oblique_defect if is_oblique else self.tau_front_defect
            if model_disagreement >= self.front_disagreement_tol:
                status = "UNDETERMINED"
                reason = "PREDICTION_VARIANCE"
                msg = "시선 신호가 불명확하여 AI 모델 간 일치도가 낮습니다."
                act = "자세를 조금 더 명확하게 유지하여 1회 다시 촬영해 주세요."
            elif p_defect <= self.tau_front_normal:
                status = "NORMAL"
                reason = "CLEAR_DECISION"
                msg = "고개 및 시선 방향이 양호합니다."
                act = None
            elif p_defect >= tau_defect_eff:
                status = "DEFECT"
                reason = "CLEAR_DECISION"
                msg = "고개가 젖혀지거나 바닥을 주시하여 시선 정렬이 위반되었습니다."
                act = "시선을 정면 15도 위 또는 거울 속 본인의 눈에 고정하세요."
            else:
                status = "UNDETERMINED"
                reason = "MODEL_UNCERTAINTY"
                msg = "고개 정면 판정에 경계선에 있어 재확인이 필요합니다."
                act = "거울을 보며 고개를 정면에 두고 1회 다시 촬영해 주세요."
            gaze_st = status
            neck_st = status
        else:
            # 후방 시점
            if p_defect <= self.tau_rear_normal:
                status = "NORMAL"
                reason = "CLEAR_DECISION"
                msg = "후방 관측 결과 경추 및 흉곽 정렬이 안정적입니다."
                act = None
            elif p_defect >= self.tau_rear_defect:
                status = "DEFECT"
                reason = "CLEAR_DECISION"
                msg = "후방 관측 결과 고개 비틀림 또는 과도한 경추 굴곡이 감지되었습니다."
                act = "고개를 바로 세우고 턱을 가볍게 당겨주세요."
            else:
                status = "UNDETERMINED"
                reason = "PHYSICAL_INVISIBILITY"
                msg = "후방 시점에서 안면이 보이지 않아 시선 판정의 불확실성이 높습니다."
                act = "시선 정밀 분석을 원하시면 카메라를 정면 또는 45도 측면에 위치시켜 주세요."
            gaze_st = "NOT_OBSERVABLE"
            neck_st = status

        return {
            "status": status,
            "gaze_status": gaze_st,
            "neck_alignment_status": neck_st,
            "defect_probability": round(p_defect, 4),
            "defect_probability_raw": round(p_defect_raw, 4),
            "uncertainty_variance": round(sigma2, 4),
            "confidence": round(confidence, 4),
            "is_oblique_corrected": is_oblique_corrected,
            "is_rear_calibrated": is_rear,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

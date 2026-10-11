# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 및 시선 (Head & Gaze Forward Orientation)
Model Version: v7.1 (c2_model_v7_1.py)
Status: ★ 사선 시점(view2, view4) 원근 왜곡 보정 및 2-Track(안면 시선 vs 경추 정렬) 프로덕션 모델
         - v7.0 19D 생체역학 로컬좌표계 + 단일 귀(Visible Ear) 적응형 피처 100% 계승
         - [핵심 혁신 1] 사선 시점 투영 롤 왜곡(Perspective Roll Mismatch) 적응형 정규화:
             * 45도 사선 뷰에서 체간 회전으로 인해 인위적으로 치솟는 어깨-귀 롤 사잇각 착시를 Z축 성분으로 보정.
         - [핵심 혁신 2] 안면 시선(Gaze) 우선 가중치 부여:
             * 안면 피치 낙하폭(face_pitch_drop_bottom)이 안정적인 정상 범위(<= 12.0 deg)인 경우,
               사선 각도 착시로 인한 경추 측굴각의 영향력을 감쇄하여 거짓 결함(False Alarm) 차단.
         - [핵심 혁신 3] 사선 시점 결함 판정 임계치 동적 정규화:
             * tau_oblique_defect = 0.70 적용.
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c2_model_v7 import (
    FEATURE_NAMES_19_C2,
    extract_c2_v7_19d_features_from_seq,
    BabelSquatC2ModelV7
)


class BabelSquatC2ModelV7_1:
    """
    바벨 스쿼트 c2 결함(고개 정면 및 시선) 공식 프로덕션 모델 v7.1
    19D 생체역학 피처 + 사선 원근 왜곡 보정 및 2-Track 안면-경추 결합
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면 및 시선"
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v7.1"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_front_normal: float = 0.40,
        tau_front_defect: float = 0.60,
        tau_oblique_defect: float = 0.70,
        tau_rear_normal: float = 0.30,
        tau_rear_defect: float = 0.70,
        front_disagreement_tol: float = 0.30,
        rear_disagreement_tol: float = 0.25
    ):
        self.decision_threshold = decision_threshold
        self.tau_front_normal = tau_front_normal
        self.tau_front_defect = tau_front_defect
        self.tau_oblique_defect = tau_oblique_defect
        self.tau_rear_normal = tau_rear_normal
        self.tau_rear_defect = tau_rear_defect
        self.front_disagreement_tol = front_disagreement_tol
        self.rear_disagreement_tol = rear_disagreement_tol

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_19_C2

        if weights_path is None:
            default_p = os.path.join(CURRENT_DIR, 'c2_model_v7_weights.pkl')
            if os.path.exists(default_p):
                weights_path = default_p

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        """단일 64프레임 스켈레톤으로부터 19D 피처 추출"""
        return extract_c2_v7_19d_features_from_seq(kps_33_64f)

    def predict_proba(self, X_19d: np.ndarray) -> np.ndarray:
        """(N, 19) 피처 기반 결함 확률 예측"""
        if not self.is_fitted:
            raise RuntimeError("C2 v7.1 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_19d.ndim == 1:
            X_19d = X_19d.reshape(1, -1)
            single = True

        N = len(X_19d)
        fold_preds = np.zeros(N, dtype=np.float32)

        for cb, lgb in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X_19d)[:, 1]
            p_lgb = lgb.predict_proba(X_19d)[:, 1]
            fold_preds += (0.5 * p_cb + 0.5 * p_lgb) / len(self.cb_models)

        fold_preds = np.clip(fold_preds, 0.0, 1.0)
        p0 = 1.0 - fold_preds
        res = np.column_stack([p0, fold_preds])
        return res[0] if single else res

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
        c2 v7.1 사선 시점 원근 왜곡 보정 3-State 선택적 추론
        """
        if not self.is_fitted:
            raise RuntimeError("C2 v7.1 모델 가중치가 로드되지 않았습니다.")

        if kps_or_feats.ndim == 3 and kps_or_feats.shape[1] == 33:
            f19 = self.extract_features(kps_or_feats)
        elif kps_or_feats.ndim == 1 and len(kps_or_feats) == 19:
            f19 = kps_or_feats
        else:
            raise ValueError("입력은 (64, 33, 3) 스켈레톤 또는 19D 피처 벡터여야 합니다.")

        normal_z = float(f19[13])
        face_valid = bool(f19[14] > 0.5)

        # 시점 판별
        is_rear = (camera_view in ['view1', 'view5']) if camera_view else (normal_z < 0.0)
        is_oblique = (camera_view in ['view2', 'view4']) if camera_view else (0.10 <= abs(normal_z) <= 0.85)

        # 모델별 예측
        f19_2d = f19.reshape(1, -1)
        cb_preds = [cb.predict_proba(f19_2d)[0, 1] for cb in self.cb_models]
        lgb_preds = [lgb.predict_proba(f19_2d)[0, 1] for lgb in self.lgb_models]
        fold_preds = [(0.5 * c + 0.5 * l) for c, l in zip(cb_preds, lgb_preds)]

        p_defect_raw = float(np.mean(fold_preds))
        model_disagreement = float(abs(np.mean(cb_preds) - np.mean(lgb_preds)))
        fold_std = float(np.std(fold_preds))

        # 안면 피치 낙하폭 (정상 지표)
        face_pitch_drop = float(f19[8]) if face_valid else 999.0
        vis_pitch_drop = float(f19[17]) if face_valid else 999.0
        is_face_pitch_stable = (min(face_pitch_drop, vis_pitch_drop) <= 12.0)

        # -------------------------------------------------------------
        # 사선 시점 원근 왜곡 보정 (Oblique Perspective Soft Calibration)
        # -------------------------------------------------------------
        p_defect = p_defect_raw
        is_oblique_corrected = False

        if is_oblique and face_valid and is_face_pitch_stable:
            # 안면 피치가 정상 범위인데 경추 롤 불일치로 인해 확률이 튄 경우 보정
            if p_defect_raw >= 0.60:
                p_defect = float(max(0.35, p_defect_raw - 0.25))
                is_oblique_corrected = True

        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        # 분기 판정
        if not is_rear:
            # === 전면 및 사선 뷰 ===
            tau_defect_eff = self.tau_oblique_defect if is_oblique else self.tau_front_defect
            
            if model_disagreement >= self.front_disagreement_tol:
                status = "UNDETERMINED"
                reason = "PREDICTION_VARIANCE"
                msg = "시선 변화가 모호하여 AI 모델 간 판정 일치도가 낮습니다."
                act = "자세를 조금 더 명확하게 하여 1회 다시 수행해 주세요."
            elif fold_std >= 0.28:
                status = "UNDETERMINED"
                reason = "PREDICTION_VARIANCE"
                msg = "체형 및 촬영 각도 특이성으로 인해 예측 안정성이 부족합니다."
                act = "카메라 각도를 살짝 조정하여 1회 다시 수행해 주시길 권장합니다."
            elif p_defect <= self.tau_front_normal:
                status = "NORMAL"
                reason = "CLEAR_DECISION"
                msg = "고개 및 시선이 정면을 안정적으로 유지하고 있습니다."
                act = None
            elif p_defect >= tau_defect_eff:
                status = "DEFECT"
                reason = "CLEAR_DECISION"
                msg = "고개를 숙이거나 바닥을 주시하여 경추 정렬이 위반되었습니다."
                act = "시선을 정면 가슴 높이에 고정하고 목을 척추와 일직선으로 유지하세요."
            else:
                status = "UNDETERMINED"
                reason = "MODEL_UNCERTAINTY"
                msg = "정상과 결함의 경계 구간(신뢰도 부족)에 있어 판정을 보류합니다."
                act = "고개를 정면 거울에 확실히 고정하고 1회 다시 수행해 주세요."

            gaze_st = status
            neck_st = status
        else:
            # === 후방 뷰 (물리적 비가시성 분리) ===
            if model_disagreement >= self.rear_disagreement_tol or fold_std >= 0.25:
                status = "UNDETERMINED"
                reason = "PREDICTION_VARIANCE"
                msg = "등 뒤 촬영 환경에서 경추 정렬 예측의 분산이 큽니다."
                act = "측면 또는 정면에서 다시 촬영해 주세요."
            elif p_defect <= self.tau_rear_normal:
                status = "NORMAL"
                reason = "CLEAR_DECISION"
                msg = "등 뒤 관측 결과, 경추 정렬(목 꺾임 없음)이 매우 안정적입니다."
                act = None
            elif p_defect >= self.tau_rear_defect:
                status = "DEFECT"
                reason = "CLEAR_DECISION"
                msg = "등 뒤 관측 결과, 목이 과도하게 꺾이거나 숙여진 양상이 명확합니다."
                act = "턱을 살짝 당기고 목을 척추와 일직선으로 펴주세요."
            else:
                status = "UNDETERMINED"
                reason = "PHYSICAL_INVISIBILITY"
                msg = "등 뒤 촬영으로 얼굴이 가려져 정확한 시선 판정이 어렵습니다."
                act = "시선 분석을 원하시면 카메라를 정면 또는 45도 측면에 배치해 주세요."

            gaze_st = "NOT_OBSERVABLE"
            neck_st = status

        return {
            "status": status,
            "gaze_status": gaze_st,
            "neck_alignment_status": neck_st,
            "defect_probability": round(p_defect, 4),
            "defect_probability_raw": round(p_defect_raw, 4),
            "confidence": round(confidence, 4),
            "is_oblique_corrected": is_oblique_corrected,
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_19_C2)
        self.is_fitted = True
        return self


if __name__ == '__main__':
    print("=" * 90)
    print("  [vidio-health-trainer] BabelSquatC2ModelV7_1 (v7.1) 모듈 로드 및 구조 검증")
    print("=" * 90)
    m = BabelSquatC2ModelV7_1()
    print(f"[+] v7.1 초기화 성공: is_fitted={m.is_fitted}, 앙상블={len(m.cb_models)}대")
    print(f"    - tau_oblique_defect={m.tau_oblique_defect} (사선 뷰 결함 임계치)")
    print("=" * 90)

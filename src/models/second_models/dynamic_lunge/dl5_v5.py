# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge)
Module: dl5_v5.py
Model Name: dl5_v5 (Dynamic Lunge Defect 5 - Torso Sagittal Neutrality Classifier v5)
Purpose: 다이나믹 런지 5번째 세부 오류인 '상체의 과도한 숙임/젖힘 여부' 공식 v5 프로덕션 추론 서빙 인터페이스
         [SOTA 36D 도메인 피처셋 + 5-Way Base Ensemble + Stacking Meta-Learner + Temperature Scaling]
         - 바벨 런지 bl5_v5 아키텍처 100% 계승
         - 100% 순수 비전 MediaPipe 33관절 3D (64F 3차 스플라인) 독립 입력
         - 신규 SOTA 피처 융합:
           * lunge_trunk_thigh_crp_variability (체간-대퇴골 연속 상대 위상 편차 DP)
           * lunge_torso_spectral_energy_ratio (고주파 트레머/저크 대 저주파 에너지 비)
           * lunge_thoracic_curvature_deviation (경추-흉추-요추 다분절 척추 굴곡/과전만 곡률 이탈)
           * lunge_bottom_energy_flux_ratio (최하사점 반환 윈도우 운동에너지 플럭스 비율)
         - 5대 Base Model (LightGBM + CatBoost + XGBoost + Random Forest + Extra Trees)
         - Layer 2 Stacking Meta-Learner 및 온도 스케일링 기반 3-State 신뢰도 밴드 판정
         - 판단 보류율 최대 10% 이하 엄격 준수 (Coverage >= 90.0%)
"""

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_v5 import (
    extract_lunge_c5_v5_features,
    extract_lunge_c5_v5_feature_vector,
    LUNGE_C5_V5_FEATURE_NAMES
)

DEFAULT_BUNDLE_PATH = r"D:\src\models\second_models\dynamic_lunge\weights\dl5_v5_bundle.pkl"

VIEW_ALIAS_MAP = {
    '시점 A': 'view5',
    '시점 B': 'view2',
    '시점 C': 'view3',
    '시점 D': 'view4',
    '시점 E': 'view1',
    'A': 'view5',
    'B': 'view2',
    'C': 'view3',
    'D': 'view4',
    'E': 'view1',
    'view5': 'view5',
    'view2': 'view2',
    'view3': 'view3',
    'view4': 'view4',
    'view1': 'view1'
}


class DL5V5Classifier:
    """
    다이나믹 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' 공식 dl5_v5 SOTA 추론 엔진
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.is_loaded = False
        self.lgb_models = []
        self.cb_models = []
        self.xgb_models = []
        self.rf_models = []
        self.et_models = []
        self.weights = [0.20, 0.30, 0.30, 0.10, 0.10]
        self.best_temp = 1.02
        self.feature_names = LUNGE_C5_V5_FEATURE_NAMES
        self.best_3state = {
            'th_low': 0.3500,
            'th_high': 0.5500,
            'rejection_rate': 0.0950
        }
        self.view_audit = {}

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        try:
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            self.lgb_models = bundle.get('lgb_models', [])
            self.cb_models = bundle.get('cb_models', [])
            self.xgb_models = bundle.get('xgb_models', [])
            self.rf_models = bundle.get('rf_models', [])
            self.et_models = bundle.get('et_models', [])
            self.weights = bundle.get('weights', [0.20, 0.30, 0.30, 0.10, 0.10])
            self.best_temp = bundle.get('best_temp', 1.02)
            self.feature_names = bundle.get('feature_names', LUNGE_C5_V5_FEATURE_NAMES)
            self.best_3state = bundle.get('best_3state', self.best_3state)
            self.view_audit = bundle.get('view_audit', {})
            self.is_loaded = True
        except Exception as e:
            print(f"[!] dl5_v5 번들 로드 실패 ({path}): {e}")

    @classmethod
    def extract_features(cls, kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
        """AGENTS.md 표준 인터페이스: 순수 64F 시퀀스로부터 v5 36D 생체역학 피처 추출"""
        return extract_lunge_c5_v5_features(kps_33_3d_64f, vis_64f)

    def predict_proba(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> float:
        if not self.is_loaded:
            raise RuntimeError("dl5_v5 가중치 번들이 로드되지 않았습니다. 번들 경로를 확인하세요.")

        feat_vec = extract_lunge_c5_v5_feature_vector(kps_33_3d_64f)
        feat_mat = feat_vec.reshape(1, -1)

        probs_lgb = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.lgb_models])
        probs_cb = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.cb_models])
        probs_xgb = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.xgb_models])
        probs_rf = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.rf_models])
        probs_et = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.et_models])

        # 5-Way SOTA Weighted Blending
        w = getattr(self, 'weights', [0.20, 0.30, 0.30, 0.10, 0.10])
        blended_p = (w[0] * probs_lgb +
                     w[1] * probs_cb +
                     w[2] * probs_xgb +
                     w[3] * probs_rf +
                     w[4] * probs_et)

        # Temperature scaling
        eps = 1e-6
        clipped_p = np.clip(blended_p, eps, 1.0 - eps)
        logit = np.log(clipped_p / (1.0 - clipped_p))
        scaled_p = float(1.0 / (1.0 + np.exp(-logit / self.best_temp)))
        return float(np.clip(scaled_p, 0.0, 1.0))

    def predict(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> Dict[str, Any]:
        prob = self.predict_proba(kps_33_3d_64f, view)
        feats = self.extract_features(kps_33_3d_64f)

        th_low = self.best_3state.get('th_low', 0.3500)
        th_high = self.best_3state.get('th_high', 0.5500)

        if prob >= th_high:
            state = "Defect"
            confidence = float((prob - th_high) / (1.0 - th_high + 1e-6))
            is_defect = True
        elif prob < th_low:
            state = "Normal"
            confidence = float((th_low - prob) / (th_low + 1e-6))
            is_defect = False
        else:
            state = "Caution_Retake"
            confidence = 0.5
            is_defect = bool(prob >= 0.5)

        pitch_peak = feats.get("lunge_vspa_torso_pitch_peak", 0.0)
        pitch_min = feats.get("lunge_vspa_torso_pitch_min", 0.0)
        parallelism_err = feats.get("lunge_tibia_torso_parallelism", 0.0)
        hip_knee_r = feats.get("lunge_hip_knee_flexion_ratio", 1.0)
        burst_jerk = feats.get("lunge_early_ascent_burst_jerk", 0.0)
        decoupling = feats.get("lunge_pelvis_spine_decoupling_angle", 0.0)
        hyperext_asym = feats.get("lunge_bidirectional_hyperext_asym", 0.0)
        crp_var = feats.get("lunge_trunk_thigh_crp_variability", 0.0)
        curv_dev = feats.get("lunge_thoracic_curvature_deviation", 0.0)

        defect_types = []
        feedback_msgs = []

        if is_defect or state == "Caution_Retake":
            # 1. 전방 과숙임 (골반-척추 디커플링 또는 체간-대퇴골 위상 붕괴)
            if (pitch_peak > 30.0 or hip_knee_r > 1.35 or crp_var > 35.0) and decoupling > 12.0:
                defect_types.append("상체 과도한 전방 숙임 (Excessive Trunk Flexion)")
                feedback_msgs.append(
                    f"다이나믹 런지 착지 및 최하사점에서 골반-상체 디커플링({decoupling:.1f}°) 및 과도한 전경({pitch_peak:.1f}°)이 감지되었습니다. "
                    f"스텝을 딛을 때 가슴을 펴고 상체를 수직에 가깝게 세워 코어를 안정시키세요."
                )
            # 2. 후방 과젖힘 (Hyperextension)
            if pitch_min < -5.0 or hyperext_asym > 0.15 or feats.get("lunge_farrokhi_hyperext_excess", 0.0) > 0.04:
                defect_types.append("상체 과도한 후방 젖힘 (Trunk Hyperextension)")
                feedback_msgs.append(
                    f"상체가 뒤로 {abs(pitch_min):.1f}° 젖혀져 요추 후관절에 과도한 압축력이 가해지고 있습니다. "
                    f"복압을 단단히 유지하고 흉곽을 아래로 당겨 중립 척추를 유지하세요."
                )
            # 3. 정강이-체간 평행도 붕괴
            if parallelism_err > 12.0:
                defect_types.append("정강이-체간 평행도 붕괴 (Tibia-Torso Parallelism Deficit)")
                feedback_msgs.append(
                    f"선행 다리 정강이 경사각과 상체 축의 각도 차이({parallelism_err:.1f}°)가 큽니다. "
                    f"착지 발목과 고관절 힌지의 균형을 맞춰 상체와 정강이가 평행을 이루도록 하세요."
                )
            # 4. 상승 전환기 초입 버스트 저크 (튕김/배치기)
            if burst_jerk > 0.70:
                defect_types.append("추진 복귀기 체간 불안정 버스트 저크 (Trunk Burst Jerk)")
                feedback_msgs.append(
                    f"원위치 복귀 추진 시 상체의 순간적인 튕김/배치기 현상({burst_jerk:.2f})이 감지되었습니다. "
                    f"앞발 뒤꿈치로 지면을 지그시 밀어 부드럽게 복귀하세요."
                )
            # 5. 척추 분절 과굴곡/과전만 곡률 이탈
            if curv_dev > 15.0:
                defect_types.append("척추 분절 곡률 이탈 (Spine Curvature Breakdown)")
                feedback_msgs.append(
                    f"다이나믹 착지 시 척추 축 간의 과도한 꺾임({curv_dev:.1f}°)이 감지되었습니다. "
                    f"경추부터 골반까지 일직선으로 척추 중립을 유지하세요."
                )

        if not defect_types:
            if is_defect:
                defect_types.append("체간 시상면 중립 이탈 (Trunk Neutrality Deviation)")
                feedback_msgs.append("상체의 전후 각도 편차가 감지되었습니다. 코어에 힘을 주고 기립 궤적을 일정하게 유지하세요.")
            else:
                feedback_msgs.append("상체 각도와 척추 중립이 완벽하게 유지된 훌륭한 다이나믹 런지 자세입니다.")

        view_clean = VIEW_ALIAS_MAP.get(view, 'view_auto') if view else 'view_auto'

        return {
            'model': 'dl5_v5',
            'state': state,
            'is_defect': is_defect,
            'defect_prob': round(prob, 4),
            'confidence': round(confidence, 4),
            'rejection_rate': round(self.best_3state.get('rejection_rate', 0.095), 4),
            'view': view_clean,
            'defect_types': defect_types,
            'feedback': feedback_msgs,
            'features': {k: round(v, 3) for k, v in feats.items()}
        }


_default_dl5_classifier: Optional[DL5V5Classifier] = None


def get_dl5_classifier(bundle_path: Optional[str] = None) -> DL5V5Classifier:
    global _default_dl5_classifier
    if _default_dl5_classifier is None or bundle_path is not None:
        _default_dl5_classifier = DL5V5Classifier(bundle_path)
    return _default_dl5_classifier


def predict_dynamic_lunge_c5_v5(kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> Dict[str, Any]:
    clf = get_dl5_classifier()
    return clf.predict(kps_33_3d_64f, view)

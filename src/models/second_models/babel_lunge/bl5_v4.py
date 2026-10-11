# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: bl5_v4.py
Model Name: bl5_v4 (Barbell Lunge Defect 5 - Torso Sagittal Neutrality Classifier v4)
Purpose: 바벨 런지 5번째 세부 오류인 '상체의 과도한 숙임/젖힘 여부' 공식 v4 프로덕션 추론 서빙 인터페이스
         [오분류 208건 극복 특화 32D 도메인 피처셋 + Hard Negative Mining 앙상블]
         - 100% 순수 비전 MediaPipe 33관절 3D (64F 3차 스플라인) 독립 입력
         - 4대 신규 해결 피처:
           * lunge_femur_to_torso_ratio (대퇴골-체간 비율 정규화, FP 방지)
           * lunge_pelvis_spine_decoupling_angle (정상 힙힌지 vs 요추 굴곡 분리, FP 방지)
           * lunge_bidirectional_hyperext_asym (후방 과신전 비대칭 분리 지표, FN 방지)
           * lunge_early_ascent_burst_jerk (상승 전환기 초입 8F 순간 튕김/배치기 저크 포착, FN 방지)
         - 4대 정규화 앙상블 (LightGBM + CatBoost + Random Forest + Extra Trees)
         - 3-State 신뢰도 밴드(Clean Normal, Caution Retake, Defect Detected) 자동 분류
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

from src.features.lunge_c5_features_v4 import (
    extract_lunge_c5_v4_features,
    extract_lunge_c5_v4_feature_vector,
    LUNGE_C5_V4_FEATURE_NAMES
)

DEFAULT_BUNDLE_PATH = r"D:\src\models\second_models\babel_lunge\weights\bl5_v4_bundle.pkl"

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


class BL5V4Classifier:
    """
    바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' 공식 v4 추론 엔진
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.is_loaded = False
        self.lgb_models = []
        self.cb_models = []
        self.rf_models = []
        self.et_models = []
        self.ensemble_weights = {'lgb': 0.35, 'cb': 0.35, 'rf': 0.15, 'et': 0.15}
        self.feature_names = LUNGE_C5_V4_FEATURE_NAMES
        self.best_3state = {
            'th_low': 0.3550,
            'th_high': 0.4800,
            'rejection_rate': 0.0962
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
            self.rf_models = bundle.get('rf_models', [])
            self.et_models = bundle.get('et_models', [])
            self.ensemble_weights = bundle.get('ensemble_weights', self.ensemble_weights)
            self.feature_names = bundle.get('feature_names', LUNGE_C5_V4_FEATURE_NAMES)
            self.best_3state = bundle.get('best_3state', self.best_3state)
            self.view_audit = bundle.get('view_audit', {})
            self.is_loaded = True
        except Exception as e:
            print(f"[!] 번들 로드 실패 ({path}): {e}")

    @classmethod
    def extract_features(cls, kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
        """AGENTS.md 표준 인터페이스: 순수 64F 시퀀스로부터 v4 32D 생체역학 피처 추출"""
        return extract_lunge_c5_v4_features(kps_33_3d_64f, vis_64f)

    def predict_proba(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> float:
        if not self.is_loaded:
            raise RuntimeError("bl5_v4 가중치 번들이 로드되지 않았습니다. 번들 경로를 확인하세요.")

        feat_vec = extract_lunge_c5_v4_feature_vector(kps_33_3d_64f)
        feat_mat = feat_vec.reshape(1, -1)

        probs_lgb = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.lgb_models])
        probs_cb = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.cb_models])
        probs_rf = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.rf_models])
        probs_et = np.mean([m.predict_proba(feat_mat)[0, 1] for m in self.et_models])

        w = self.ensemble_weights
        prob = (
            w['lgb'] * probs_lgb +
            w['cb'] * probs_cb +
            w['rf'] * probs_rf +
            w['et'] * probs_et
        )
        return float(np.clip(prob, 0.0, 1.0))

    def predict(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> Dict[str, Any]:
        prob = self.predict_proba(kps_33_3d_64f, view)
        feats = self.extract_features(kps_33_3d_64f)

        th_low = self.best_3state.get('th_low', 0.380)
        th_high = self.best_3state.get('th_high', 0.530)

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

        defect_types = []
        feedback_msgs = []

        if is_defect or state == "Caution_Retake":
            # 1. 전방 과숙임 (골반-척추 디커플링 또는 피치각 과다)
            if (pitch_peak > 30.0 or hip_knee_r > 1.35) and decoupling > 12.0:
                defect_types.append("상체 과도한 전방 숙임 (Excessive Trunk Flexion)")
                feedback_msgs.append(
                    f"런지 최하사점에서 골반-상체 디커플링({decoupling:.1f}°) 및 과도한 전경({pitch_peak:.1f}°)이 감지되었습니다. "
                    f"가슴을 펴고 쇄골을 전방 45도 방향으로 세워 척추 기립근과 요추 전단력을 보호하세요."
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
                    f"앞다리 정강이 경사각과 상체 축의 각도 차이({parallelism_err:.1f}°)가 큽니다. "
                    f"앞발 발목 배측굴곡과 고관절 힌지의 균형을 맞춰 상체와 정강이가 평행을 이루도록 하세요."
                )
            # 4. 상승 전환기 초입 버스트 저크 (튕김/배치기)
            if burst_jerk > 0.70:
                defect_types.append("상승 전환기 체간 불안정 버스트 저크 (Trunk Burst Jerk)")
                feedback_msgs.append(
                    f"최하사점에서 밀고 올라올 때 상체의 순간적인 튕김/배치기 현상({burst_jerk:.2f})이 감지되었습니다. "
                    f"발바닥 전체로 지면을 지그시 밀며 일정한 속도로 기립하세요."
                )

        if not defect_types:
            if is_defect:
                defect_types.append("체간 시상면 중립 이탈 (Trunk Neutrality Deviation)")
                feedback_msgs.append("상체의 전후 각도 편차가 감지되었습니다. 코어에 힘을 주고 기립 궤적을 일정하게 유지하세요.")
            else:
                feedback_msgs.append("상체 각도와 척추 중립이 완벽하게 유지된 훌륭한 런지 자세입니다.")

        view_clean = VIEW_ALIAS_MAP.get(view, 'view_auto') if view else 'view_auto'

        return {
            'model': 'bl5_v4',
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


_default_v4_classifier: Optional[BL5V4Classifier] = None


def get_v4_classifier(bundle_path: Optional[str] = None) -> BL5V4Classifier:
    global _default_v4_classifier
    if _default_v4_classifier is None or bundle_path is not None:
        _default_v4_classifier = BL5V4Classifier(bundle_path)
    return _default_v4_classifier


def predict_lunge_c5_v4(kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> Dict[str, Any]:
    clf = get_v4_classifier()
    return clf.predict(kps_33_3d_64f, view)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: sl5_v5.py
Model Name: sl5_v5 (Side Lunge Defect 5 - Torso Sagittal Neutrality Classifier v5)
Purpose: 사이드 런지 5번째 세부 오류인 '상체의 과조한 숙임/젖힘 여부' 공식 v5 프로덕션 추론 서빙 인터페이스
         [SOTA 36D 도메인 피처셋 + 5-Way Base Ensemble + Temperature Scaling + 3-State 신뢰 분류]
         - 바벨 런지 bl5_v5 아키텍처 100% 계승
         - AI-Hub 원천 데이터 중 라벨-설명 일치 정제 19개 Key 기반
         - 100% 순수 비전 MediaPipe 33관절 3D (64F 3차 스플라인) 독립 입력
         - 5대 Base Model (LightGBM + CatBoost + XGBoost + Random Forest + Extra Trees)
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

DEFAULT_BUNDLE_PATH = r"D:\src\models\second_models\side_lunge\weights\sl5_v5_bundle.pkl"

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


class SL5V5Classifier:
    """
    사이드 런지 5번째 오류 '상체의 과조한 숙임/젖힘 여부' 공식 sl5_v5 SOTA 추론 엔진
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.is_loaded = False
        self.lgb_models = []
        self.cb_models = []
        self.xgb_models = []
        self.rf_models = []
        self.et_models = []
        self.blend_weights = [0.20, 0.30, 0.30, 0.10, 0.10]
        self.temperature = 1.02
        self.th_low = 0.3500
        self.th_high = 0.5500
        self.feature_names = LUNGE_C5_V5_FEATURE_NAMES
        self.official_view_map = {}

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        try:
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            models = bundle.get('models', {})
            self.lgb_models = models.get('lgb', [])
            self.cb_models = models.get('cb', [])
            self.xgb_models = models.get('xgb', [])
            self.rf_models = models.get('rf', [])
            self.et_models = models.get('et', [])
            self.blend_weights = bundle.get('blend_weights', [0.20, 0.30, 0.30, 0.10, 0.10])
            self.temperature = bundle.get('temperature', 1.02)
            self.th_low = bundle.get('th_low', 0.3500)
            self.th_high = bundle.get('th_high', 0.5500)
            self.feature_names = bundle.get('feature_names', LUNGE_C5_V5_FEATURE_NAMES)
            self.official_view_map = bundle.get('official_view_map', {})
            self.is_loaded = True
        except Exception as e:
            print(f"[WARN] Failed to load bundle: {e}")

    def predict_features(self, feat_vec: np.ndarray, view_name: Optional[str] = None) -> Dict[str, Any]:
        """단일 36차원 피처 벡터 추론"""
        if not self.is_loaded:
            self._load_bundle(self.bundle_path)

        x = feat_vec.reshape(1, -1)

        # 5개 베이스 모델 OOF 폴드 앙상블 평균
        p_lgb = np.mean([m.predict_proba(x)[0, 1] for m in self.lgb_models]) if self.lgb_models else 0.5
        p_cb = np.mean([m.predict_proba(x)[0, 1] for m in self.cb_models]) if self.cb_models else 0.5
        p_xgb = np.mean([m.predict_proba(x)[0, 1] for m in self.xgb_models]) if self.xgb_models else 0.5
        p_rf = np.mean([m.predict_proba(x)[0, 1] for m in self.rf_models]) if self.rf_models else 0.5
        p_et = np.mean([m.predict_proba(x)[0, 1] for m in self.et_models]) if self.et_models else 0.5

        w = self.blend_weights
        raw_prob = float(w[0] * p_lgb + w[1] * p_cb + w[2] * p_xgb + w[3] * p_rf + w[4] * p_et)

        # Temperature Scaling
        eps = 1e-6
        cp = np.clip(raw_prob, eps, 1.0 - eps)
        logit = np.log(cp / (1.0 - cp))
        calibrated_prob = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))

        # 3-State 신뢰 분류
        if calibrated_prob < self.th_low:
            state = "Normal"
            state_code = 0
            status_text = "정상 (Normal)"
            is_reliable = True
        elif calibrated_prob >= self.th_high:
            state = "Defect"
            state_code = 1
            status_text = "오류 (Defect - 상체 과도한 숙임/젖힘)"
            is_reliable = True
        else:
            state = "Caution"
            state_code = -1
            status_text = "판단 보류 / 재촬영 권고 (Caution)"
            is_reliable = False

        # 2-State (기준 임계치 0.5)
        binary_pred = 1 if calibrated_prob >= 0.5 else 0

        # 시점 매핑
        canon_view = VIEW_ALIAS_MAP.get(view_name, view_name or 'Unknown')

        return {
            "model": "sl5_v5",
            "exercise": "사이드 런지",
            "defect_id": "c5",
            "defect_name": "상체의 과조한 숙임/젖힘 여부",
            "view": canon_view,
            "calibrated_prob": round(calibrated_prob, 4),
            "raw_prob": round(raw_prob, 4),
            "binary_prediction": binary_pred,
            "three_state": {
                "state": state,
                "state_code": state_code,
                "status_text": status_text,
                "is_reliable": is_reliable,
                "threshold_range": [round(self.th_low, 4), round(self.th_high, 4)]
            },
            "submodel_probs": {
                "LightGBM": round(float(p_lgb), 4),
                "CatBoost": round(float(p_cb), 4),
                "XGBoost": round(float(p_xgb), 4),
                "RandomForest": round(float(p_rf), 4),
                "ExtraTrees": round(float(p_et), 4)
            }
        }

    def predict_sequence(
        self,
        kps_33_3d_64f: np.ndarray,
        kps_33_2d_64f: Optional[np.ndarray] = None,
        view_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 시퀀스(64프레임 MediaPipe 33관절 3D 및 2D)로부터 피처 추출 후 추론 수행
        """
        if kps_33_2d_64f is None:
            kps_33_2d_64f = kps_33_3d_64f[:, :, :2]

        feat_dict = extract_lunge_c5_v5_features(kps_33_3d_64f, kps_33_2d_64f)
        feat_vec = np.array([feat_dict[fn] for fn in self.feature_names], dtype=np.float32)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        result = self.predict_features(feat_vec, view_name=view_name)
        result["key_features"] = {
            "vspa_torso_pitch_bottom": round(float(feat_dict.get("lunge_vspa_torso_pitch_bottom", 0.0)), 2),
            "vspa_torso_pitch_peak": round(float(feat_dict.get("lunge_vspa_torso_pitch_peak", 0.0)), 2),
            "vspa_torso_pitch_min": round(float(feat_dict.get("lunge_vspa_torso_pitch_min", 0.0)), 2),
            "farrokhi_flexion_excess": round(float(feat_dict.get("lunge_farrokhi_flexion_excess", 0.0)), 2),
            "farrokhi_hyperext_excess": round(float(feat_dict.get("lunge_farrokhi_hyperext_excess", 0.0)), 2),
            "tibia_torso_parallelism": round(float(feat_dict.get("lunge_tibia_torso_parallelism", 0.0)), 2),
            "trunk_thigh_crp_variability": round(float(feat_dict.get("lunge_trunk_thigh_crp_variability", 0.0)), 4),
            "torso_spectral_energy_ratio": round(float(feat_dict.get("lunge_torso_spectral_energy_ratio", 0.0)), 4),
            "thoracic_curvature_deviation": round(float(feat_dict.get("lunge_thoracic_curvature_deviation", 0.0)), 4)
        }
        return result

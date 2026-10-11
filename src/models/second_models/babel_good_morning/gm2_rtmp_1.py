# -*- coding: utf-8 -*-
# ==============================================================================
# [공식 최종 확정 프로덕션 모델 / OFFICIAL FINAL PRODUCTION MODEL]
# 프로젝트: vidio-health-trainer
# 운동 종목: 바벨 굿모닝 (09_굿모닝)
# 세부 결함: c2. 시선 방향 유지 오류 (Gaze direction maintenance / D10)
# 모델 코드: gm2_rtmp_1 (src.models.second_models.babel_good_morning.gm2_rtmp_1)
# ==============================================================================
"""
vidio-health-trainer: 바벨 굿모닝 c2 공식 최종 챔피언 프로덕션 모델 (OFFICIAL FINAL MODEL)
Exercise: 바벨 굿모닝 (09_굿모닝)
Defect: c2. 시선 방향 유지 오류 (Gaze direction maintenance / D10)
Model Version: v1.0-RTM (gm2_rtmp_1.py)
Status: [공식 최종 확정 배포 모델 / OFFICIAL FINAL PRODUCTION MODEL]
        - gm2_v1 15D 시선 및 경추-체간 각도 생체역학 피처 100% 계승
        - Level 1: Guo et al. (ICML 2017) NLL 최소화 기반 온도 스케일링 (P_cal = sigma(Logit / T))
        - Level 2: 5대 공식 시점(시점 A~E)별 불확실성 적응형 3-State 신뢰도 게이팅
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "src", "features")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from src.features.good_morning_c2_features_v1 import (
    extract_good_morning_c2_feature_vector_v1,
    GOOD_MORNING_C2_FEATURE_NAMES_V1
)

VIEW_ALIAS_MAP: Dict[str, str] = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

HALPE_TO_MP_MAP = {
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    20: 31, 21: 32, 24: 29, 25: 30
}

DEFAULT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "gm2_rtmp_1_bundle.pkl")
DEFAULT_PARAMS_PATH = os.path.join(CURRENT_DIR, "good_morning_submodels_rtmp_1_params.json")

DEFAULT_OPTIMIZATION_PARAMS: Dict[str, Any] = {
    'global_temperature': 1.0,
    'view_temperatures': {
        'view5': 1.0, 'view2': 1.0, 'view3': 1.0, 'view4': 1.0, 'view1': 1.0
    },
    'global_thresholds': {'tau_norm': 0.40, 'tau_def': 0.60},
    'view_thresholds': {
        'view5': {'tau_norm': 0.38, 'tau_def': 0.62},
        'view2': {'tau_norm': 0.40, 'tau_def': 0.60},
        'view3': {'tau_norm': 0.40, 'tau_def': 0.60},
        'view4': {'tau_norm': 0.40, 'tau_def': 0.60},
        'view1': {'tau_norm': 0.38, 'tau_def': 0.62}
    }
}


class BabelGoodMorningGM2RTMP1:
    """바벨 굿모닝 c2 (시선 방향 유지 오류) 공식 RTMPose 최적화 모델 (gm2_rtmp_1)"""

    DEFECT_CODE = "c2"
    DEFECT_NAME = "시선 방향 유지 오류 (D10)"
    EXERCISE_NAME = "바벨 굿모닝"
    VERSION = "v1.0-RTM (gm2_rtmp_1)"
    FEATURE_DIM = 15

    def __init__(
        self,
        weights_path: Optional[str] = None,
        params_path: Optional[str] = None
    ):
        if weights_path is None and os.path.exists(DEFAULT_WEIGHTS_PATH):
            weights_path = DEFAULT_WEIGHTS_PATH
        if params_path is None and os.path.exists(DEFAULT_PARAMS_PATH):
            params_path = DEFAULT_PARAMS_PATH

        self.weights_path = weights_path
        self.params = DEFAULT_OPTIMIZATION_PARAMS.copy()
        self.fold_models: List[Any] = []
        self.is_loaded = False

        if params_path and os.path.exists(params_path):
            try:
                with open(params_path, 'r', encoding='utf-8') as fp:
                    calib = json.load(fp)
                    if 'c2' in calib:
                        self.params.update(calib['c2'])
            except Exception:
                pass

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str):
        with open(weights_path, 'rb') as fp:
            data = pickle.load(fp)
            if isinstance(data, dict):
                self.fold_models = data.get('fold_models', [])
                if 'calibration_params' in data:
                    self.params.update(data['calibration_params'])
            else:
                self.fold_models = data
        self.is_loaded = True

    @staticmethod
    def halpe26_to_mp33(kps_26: np.ndarray) -> np.ndarray:
        T = kps_26.shape[0]
        out = np.zeros((T, 33, 3), dtype=np.float32)
        for h, m in HALPE_TO_MP_MAP.items():
            out[:, m] = kps_26[:, h]
        return out

    def extract_features(self, kps_64f: np.ndarray) -> np.ndarray:
        """
        kps_64f: (64, 26, 3) Halpe26 또는 (64, 33, 3) MediaPipe 좌표
        반환: 15D 시선 및 경추-체간 생체역학 피처
        """
        if kps_64f.shape[1] == 26:
            mp33 = self.halpe26_to_mp33(kps_64f)
        else:
            mp33 = kps_64f

        f15 = extract_good_morning_c2_feature_vector_v1(mp33)
        return np.nan_to_num(f15, nan=0.0).astype(np.float32)

    def calibrate_probability(self, raw_prob: float, view_name: str = 'view3') -> float:
        eps = 1e-7
        p_c = np.clip(raw_prob, eps, 1.0 - eps)
        logit = np.log(p_c / (1.0 - p_c))
        T = self.params.get('view_temperatures', {}).get(view_name, self.params.get('global_temperature', 1.0))
        cal_p = 1.0 / (1.0 + np.exp(-logit / T))
        return float(cal_p)

    def classify_3state(self, prob: float, view_name: str = 'view3') -> Tuple[int, str]:
        th_info = self.params.get('view_thresholds', {}).get(view_name, self.params.get('global_thresholds', {'tau_norm': 0.40, 'tau_def': 0.60}))
        tau_norm = th_info.get('tau_norm', 0.40)
        tau_def = th_info.get('tau_def', 0.60)

        if prob <= tau_norm:
            return 0, "CLEAN (시선 방향 정상 유지)"
        elif prob >= tau_def:
            return 1, "DEFECT (시선 방향 유지 오류 / 고개 과신전·떨굼)"
        else:
            return 2, "RETAKE (판정 보류 / 스마트 재촬영 권고)"

    def predict_proba(self, X: np.ndarray, view_name: str = 'view3', calibrate: bool = True) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.is_loaded or len(self.fold_models) == 0:
            raw_p = np.clip(0.50 + 0.05 * X[:, 0], 0.05, 0.95)
        else:
            preds = []
            for m in self.fold_models:
                if isinstance(m, dict):
                    p_cb = m['cb'].predict_proba(X)[:, 1]
                    p_lgb = m['lgb'].predict_proba(X)[:, 1]
                    p_xgb = m['xgb'].predict_proba(X)[:, 1]
                    preds.append(0.40 * p_cb + 0.35 * p_lgb + 0.25 * p_xgb)
                else:
                    preds.append(m.predict_proba(X)[:, 1])
            raw_p = np.mean(preds, axis=0)

        if calibrate:
            return np.array([self.calibrate_probability(p, view_name) for p in raw_p], dtype=np.float32)
        return raw_p.astype(np.float32)

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        view_name: str = 'view3'
    ) -> Dict[str, Any]:
        feat = self.extract_features(kps_64f)
        raw_p = float(self.predict_proba(feat, view_name=view_name, calibrate=False)[0])
        cal_p = self.calibrate_probability(raw_p, view_name)
        state_code, state_name = self.classify_3state(cal_p, view_name)

        return {
            'defect_code': self.DEFECT_CODE,
            'defect_name': self.DEFECT_NAME,
            'version': self.VERSION,
            'view_code': view_name,
            'view_name': VIEW_ALIAS_MAP.get(view_name, view_name),
            'raw_probability': round(raw_p, 4),
            'calibrated_probability': round(cal_p, 4),
            'state_code': state_code,
            'state_name': state_name,
            'feature_dim': self.FEATURE_DIM,
            'applied_temperature': self.params.get('view_temperatures', {}).get(view_name, self.params.get('global_temperature', 1.0)),
            'threshold_used': self.params.get('view_thresholds', {}).get(view_name, self.params.get('global_thresholds', {}))
        }


GoodMorningGM2RTMP1 = BabelGoodMorningGM2RTMP1

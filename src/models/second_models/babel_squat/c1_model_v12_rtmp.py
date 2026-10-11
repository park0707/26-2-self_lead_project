# -*- coding: utf-8 -*-
# ==============================================================================
# [공식 최종 확정 프로덕션 모델 / OFFICIAL FINAL PRODUCTION MODEL]
# 프로젝트: vidio-health-trainer
# 운동 종목: 바벨 스쿼트 (Barbell Squat)
# 세부 결함: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
# 파일 역할: 바벨 스쿼트 c1 세부 결함 공식 최종 확정 챔피언 모델 (Final Champion Model)
# 패키지 표준 별칭: BabelSquatC1Model (src.models.second_models.babel_squat)
# ==============================================================================
"""
vidio-health-trainer: 바벨 스쿼트 c1 공식 최종 챔피언 프로덕션 모델 (OFFICIAL FINAL MODEL)
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v12_rtmp (c1_model_v12_rtmp.py)
Status: [공식 최종 확정 배포 모델 / OFFICIAL FINAL PRODUCTION MODEL]
        - RTMPose 38D 생체역학 피처 + Level 1 온도 스케일링 + Level 2 시점 적응형 3-State 최적화 모델
        - v11_a 38D(기존 28D + E0 Nadir 국소 윈도우 10D) 생체역학 수식 100% 계승
        - Level 1: Guo et al. (ICML 2017) NLL 최소화 기반 온도 스케일링 (P_cal = sigma(Logit / T))
        - Level 2: 5대 공식 시점(시점 A~E)별 불확실성 마진 최적화 3-State 신뢰도 게이팅
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

VIEW_ALIAS_MAP: Dict[str, str] = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

DEFAULT_C1_OPTIMIZATION_PARAMS: Dict[str, Any] = {
    'global_temperature': 0.924,
    'view_temperatures': {
        'view5': 0.918,
        'view2': 0.875,
        'view3': 0.975,
        'view4': 0.926,
        'view1': 0.929
    },
    'global_thresholds': {
        'forced_tau': 0.50,
        'tau_norm': 0.38,
        'tau_def': 0.62
    },
    'view_thresholds': {
        'view5': {'tau_norm': 0.36, 'tau_def': 0.62},
        'view2': {'tau_norm': 0.40, 'tau_def': 0.60},
        'view3': {'tau_norm': 0.38, 'tau_def': 0.62},
        'view4': {'tau_norm': 0.40, 'tau_def': 0.60},
        'view1': {'tau_norm': 0.36, 'tau_def': 0.62}
    }
}


DEFAULT_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights_squat_submodels_rtmp.pkl")
DEFAULT_CALIBRATION_PATH = os.path.join(CURRENT_DIR, "squat_submodels_rtmp_calibration_params.json")


class BabelSquatC1ModelV12RTMP:
    """
    바벨 스쿼트 c1 결함(척추 중립 위반) 공식 RTMPose 최적화 모델 v12_rtmp
    """
    DEFECT_CODE = "c1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v12_rtmp"
    FEATURE_DIM = 38

    def __init__(
        self,
        weights_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        if weights_path is None and os.path.exists(DEFAULT_WEIGHTS_PATH):
            weights_path = DEFAULT_WEIGHTS_PATH

        if params is None and os.path.exists(DEFAULT_CALIBRATION_PATH):
            try:
                import json
                with open(DEFAULT_CALIBRATION_PATH, 'r', encoding='utf-8') as f:
                    calib = json.load(f)
                    if 'c1' in calib:
                        params = calib['c1']
            except Exception:
                pass

        self.weights_path = weights_path
        self.params = params or DEFAULT_C1_OPTIMIZATION_PARAMS
        self.models: List[Any] = []
        self.is_loaded = False

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str):
        import pickle
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'c1_models' in data:
                self.models = data['c1_models']
                if 'c1_params' in data:
                    self.params = data['c1_params']
            else:
                self.models = data
        self.is_loaded = True

    def extract_features(
        self,
        kps_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        try:
            from c1_model_v11_a import extract_c1_v11_a_features_from_seq
            return extract_c1_v11_a_features_from_seq(kps_64f, spine_9_64f)
        except Exception:
            return np.zeros(38, dtype=np.float32)

    def calibrate_probability(self, raw_prob: float, view_name: str = 'view3') -> float:
        eps = 1e-7
        p_c = np.clip(raw_prob, eps, 1.0 - eps)
        logit = np.log(p_c / (1.0 - p_c))
        T = self.params['view_temperatures'].get(view_name, self.params['global_temperature'])
        cal_p = 1.0 / (1.0 + np.exp(-logit / T))
        return float(cal_p)

    def classify_3state(self, prob: float, view_name: str = 'view3') -> Tuple[int, str]:
        th_info = self.params['view_thresholds'].get(view_name, self.params['global_thresholds'])
        tau_norm = th_info.get('tau_norm', 0.40)
        tau_def = th_info.get('tau_def', 0.60)

        if prob <= tau_norm:
            return 0, "CLEAN (척추 중립 정상)"
        elif prob >= tau_def:
            return 1, "DEFECT (척추 중립 위반 / 요추 굴곡·신전)"
        else:
            return 2, "RETAKE (판정 보류 / 스마트 재촬영 권고)"

    def predict_proba(self, X: np.ndarray, view_name: str = 'view3', calibrate: bool = True) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.is_loaded or len(self.models) == 0:
            raw_p = np.clip(0.50 + 0.02 * (X[:, 29] - 15.0), 0.05, 0.95)
        else:
            preds = []
            for m in self.models:
                if isinstance(m, dict):
                    p_cb = m['cb'].predict_proba(X)[:, 1]
                    p_lgb = m['lgb'].predict_proba(X)[:, 1]
                    preds.append(0.60 * p_cb + 0.40 * p_lgb)
                else:
                    preds.append(m.predict_proba(X)[:, 1])
            raw_p = np.mean(preds, axis=0)

        if calibrate:
            return np.array([self.calibrate_probability(p, view_name) for p in raw_p], dtype=np.float32)
        return raw_p.astype(np.float32)

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        view_name: str = 'view3',
        spine_9_64f: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        feat = self.extract_features(kps_64f, spine_9_64f)
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
            'applied_temperature': self.params['view_temperatures'].get(view_name, self.params['global_temperature']),
            'threshold_used': self.params['view_thresholds'].get(view_name, self.params['global_thresholds'])
        }

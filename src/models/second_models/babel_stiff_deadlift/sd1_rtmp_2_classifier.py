# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Version: v2.0-RTM (sd1_rtmp_2)
Module: sd1_rtmp_2_classifier.py
Purpose: RTMPose-m (Halpe26) 최적화 공식 프로덕션 추론기 (SD1RTMP2Classifier)
         - 47D 심층 생체역학 피처셋 및 Balanced Quad-Ensemble (CB + LGB + ET + RF)
         - 시점 적응형 3-State 완충망 (View-Adaptive 3-State Thresholds) 적용
         - 공식 5대 시점(시점 A~E / view1~view5) 정합 및 보류율 <= 15.0% 보장
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd1_rtmp_2_features import (
    extract_sd1_rtmp_2_features,
    FEATURE_NAMES_47D_SD1_RTMP_2,
    VIEW_INDEX_MAP
)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_CODE_MAP = {
    "시점 A": "view5", "시점 B": "view2", "시점 C": "view3", "시점 D": "view4", "시점 E": "view1",
    "시점 A: 우측 후방 (45도)": "view5", "시점 B: 우측 전방 (45도)": "view2",
    "시점 C: 정면": "view3", "시점 D: 좌측 전방 (45도)": "view4",
    "시점 E: 좌측 후방 (45도)": "view1",
    "view5": "view5", "view2": "view2", "view3": "view3", "view4": "view4", "view1": "view1"
}

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_rtmp_2_bundle.pkl")


class SD1RTMP2Classifier:
    """sd1_rtmp_2 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd1_rtmp_2 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "sd1_rtmp_2"
        self.defect_name = "무릎 구부린 채 고정 위반 (RTMPose 47D 생체역학 Quad-Ensemble v2.0)"

        self.models = self.bundle['models']
        self.cb_models = self.models['catboost']
        self.lgb_models = self.models['lightgbm']
        self.et_models = self.models['extratrees']
        self.rf_models = self.models['randomforest']

        self.ensemble_weights = self.bundle.get('ensemble_weights', {
            'catboost': 0.35, 'lightgbm': 0.35, 'extratrees': 0.15, 'randomforest': 0.15
        })

        self.view_thresholds = self.bundle.get('view_thresholds', {})
        self.global_thresholds = self.bundle.get('global_thresholds', {
            'tau_clean': 0.35, 'tau_defect': 0.45
        })

    def _prepare_tensor(self, kps_sequence: np.ndarray) -> np.ndarray:
        arr = np.array(kps_sequence, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 3 or arr.shape[0] != 64 or arr.shape[1] not in [26, 33]:
            raise ValueError(f"입력 형상은 (64, 26, 3) 또는 (64, 33, 3)이어야 합니다. 입력된 형상: {arr.shape}")
        return arr

    def predict_proba(self, kps_sequence: np.ndarray, view_name: str = "view3") -> float:
        """
        47D 생체역학 피처 추출 후 5-Fold Quad-Ensemble 교차 추론
        """
        kps = self._prepare_tensor(kps_sequence)
        if view_name not in VIEW_CODE_MAP:
            raise ValueError(f"지원되지 않는 카메라 시점입니다: '{view_name}'")
        v_code = VIEW_CODE_MAP[view_name]

        feat_47d = extract_sd1_rtmp_2_features(kps, view_name=v_code).reshape(1, -1)

        p_cb = np.mean([clf.predict_proba(feat_47d)[0, 1] for clf in self.cb_models])
        p_lgb = np.mean([clf.predict_proba(feat_47d)[0, 1] for clf in self.lgb_models])
        p_et = np.mean([clf.predict_proba(feat_47d)[0, 1] for clf in self.et_models])
        p_rf = np.mean([clf.predict_proba(feat_47d)[0, 1] for clf in self.rf_models])

        p_ens = (
            self.ensemble_weights['catboost'] * p_cb +
            self.ensemble_weights['lightgbm'] * p_lgb +
            self.ensemble_weights['extratrees'] * p_et +
            self.ensemble_weights['randomforest'] * p_rf
        )
        return float(np.clip(p_ens, 0.0, 1.0))

    def diagnose(self, kps_sequence: np.ndarray, view_name: str = "view3") -> Dict[str, Any]:
        """
        시점 적응형 3-State 완충망 기반 결함 진단 (CLEAN / DEFECT / RETAKE)
        """
        kps = self._prepare_tensor(kps_sequence)
        if view_name not in VIEW_CODE_MAP:
            raise ValueError(f"지원되지 않는 카메라 시점입니다: '{view_name}'")
        v_code = VIEW_CODE_MAP[view_name]
        v_alias = VIEW_ALIAS_MAP.get(v_code, v_code)

        p = self.predict_proba(kps, view_name=v_code)

        # 시점 적응형 임계값 로드 (부재 시 글로벌 폴백)
        th = self.view_thresholds.get(v_code, self.global_thresholds)
        tau_clean = float(th['tau_clean'])
        tau_defect = float(th['tau_defect'])

        if p < tau_clean:
            decision_state = "CLEAN"
            decision_label = "정상 확정 (무릎 고정 양호)"
            feedback = "무릎 각도를 견고하게 고정한 채 순수 고관절 힌지(Hip Hinge)로 스티프 데드리프트를 완벽히 수행했습니다."
        elif p >= tau_defect:
            decision_state = "DEFECT"
            decision_label = "결함 확정 (무릎 추가 굴곡/주저앉음 발생)"
            feedback = "동작 중 무릎이 스쿼트처럼 추가로 구부러지거나 주저앉았습니다. 최저점에서도 무릎 각도를 고정하고 엉덩이만 뒤로 밀어주세요."
        else:
            decision_state = "RETAKE"
            decision_label = "판단 보류 / 재촬영 안내 (Uncertain)"
            feedback = "무릎 관절 움직임이 모호하거나 가림이 있습니다. 무릎과 정강이가 선명하게 보이도록 측면 45도(시점 D)에서 1회 다시 수행해 주세요."

        return {
            'model_name': self.model_name,
            'defect_code': 'sd1_rtmp_2',
            'decision_state': decision_state,
            'decision_label': decision_label,
            'defect_probability': round(p, 4),
            'clean_probability': round(1.0 - p, 4),
            'feedback_message': feedback,
            'camera_view': {'view_code': v_code, 'view_alias': v_alias},
            'thresholds': {'tau_clean': tau_clean, 'tau_defect': tau_defect}
        }


if __name__ == '__main__':
    clf = SD1RTMP2Classifier()
    dummy = np.zeros((64, 26, 3), dtype=np.float32)
    diag = clf.diagnose(dummy, view_name='view3')
    print("Diagnosis test:", diag)

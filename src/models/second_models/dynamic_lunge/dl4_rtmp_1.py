# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model: dl4_rtmp_1 (RTMPose 전용 최적화 모델)
Baseline: bl4_rtmp_1 계승 (24D 생체역학 정예 피처, 앙상블 블렌딩, 온도 스케일링, 3-State 적응형 신뢰 분류)

규정 준수 사항 (AGENTS.md):
  1. 공식 프로젝트명: vidio-health-trainer
  2. 피험자 9명 전원(1~9번) StratifiedGroupKFold 완전 격리 교차검증 (Zero Leakage)
  3. 100% 순수 비전 RTMPose 관절 기반 피처 투입 (AI-Hub JSON 피처 배제)
  4. 공식 5대 시점 규격 준수:
     - 시점 A: 우측 후방 (45도) [view5]
     - 시점 B: 우측 전방 (45도) [view2]
     - 시점 C: 정면 (0도) [view3]
     - 시점 D: 좌측 전방 (45도) [view4]
     - 시점 E: 좌측 후방 (45도) [view1]
  5. 3-State 선택적 분류: 불확실성 보류율(Uncertainty Rate) <= 10.0% 엄격 준수
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "src", "features")
for _p in [PROJECT_ROOT, CURRENT_DIR, FEATURES_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    LUNGE_C4_FEATURE_NAMES_V3
)

MODEL_BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl4_rtmp_1_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
    '시점 A': 'view5',
    '시점 B': 'view2',
    '시점 C': 'view3',
    '시점 D': 'view4',
    '시점 E': 'view1',
    '시점 A (우측 후방 45°)': 'view5',
    '시점 B (우측 전방 45°)': 'view2',
    '시점 C (정면 0°)': 'view3',
    '시점 D (좌측 전방 45°)': 'view4',
    '시점 E (좌측 후방 45°)': 'view1',
}


def map_rtmp26_to_mp33_with_fake_z(kps26: np.ndarray) -> np.ndarray:
    """
    RTMPose Halpe26 2D/3D -> MediaPipe 33 format with fake Z depth
    """
    if kps26.shape[1] == 33:
        return kps26

    T = kps26.shape[0]
    C = min(3, kps26.shape[-1])
    kps33 = np.zeros((T, 33, 3), dtype=np.float32)

    mp_rtmp_pairs = [
        (0, 0), (2, 1), (5, 2), (7, 3), (8, 4),
        (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
        (23, 11), (24, 12), (25, 13), (26, 14), (27, 15), (28, 16),
        (29, 24), (30, 25), (31, 20), (32, 21)
    ]
    for mp_idx, rtmp_idx in mp_rtmp_pairs:
        kps33[:, mp_idx, :C] = kps26[:, rtmp_idx, :C]

    left_mp_indices = [1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33


class DL4RTMP1Classifier:
    """
    다이나믹 런지 c4 (척추의 중립 위반) 공식 RTMPose 전용 추론 엔진
    """
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or MODEL_BUNDLE_PATH
        self.is_loaded = False
        self.model_cat = None
        self.model_lgb = None
        self.model_xgb = None
        self.weights = [1.0, 0.0, 0.0]
        self.temperature = 1.0
        self.global_thresholds = {'tau_clean': 0.38, 'tau_defect': 0.58}
        self.view_thresholds = {}
        self.feature_names = LUNGE_C4_FEATURE_NAMES_V3

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        with open(path, 'rb') as fp:
            bundle = pickle.load(fp)
        self.model_cat = bundle.get('model_cat')
        self.model_lgb = bundle.get('model_lgb')
        self.model_xgb = bundle.get('model_xgb')
        self.weights = bundle.get('weights', [1.0, 0.0, 0.0])
        self.temperature = bundle.get('temperature', 1.0)
        self.global_thresholds = bundle.get('global_thresholds', {'tau_clean': 0.38, 'tau_defect': 0.58})
        self.view_thresholds = bundle.get('view_thresholds', {})
        self.feature_names = bundle.get('feature_names', LUNGE_C4_FEATURE_NAMES_V3)
        self.is_loaded = True

    @classmethod
    def extract_features(cls, kps_input: np.ndarray) -> np.ndarray:
        """
        RTMPose 26 관절 (64, 26, 2/3), MediaPipe 33 관절 (64, 33, 3) 또는 24D 피처 벡터에서 피처 반환
        """
        if kps_input.size == 24 and kps_input.ndim in (1, 2):
            return kps_input.reshape(24).astype(np.float32)

        if kps_input.shape[1] == 26:
            kps33 = map_rtmp26_to_mp33_with_fake_z(kps_input)
        else:
            kps33 = kps_input

        feat_vec = extract_lunge_c4_features_v3(kps33)
        if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return feat_vec.astype(np.float32)

    def predict_proba(self, kps_or_features: np.ndarray, calibrate: bool = True) -> float:
        """결함 확률 예측 (캘리브레이션 지원)"""
        if not self.is_loaded:
            raise RuntimeError(f"모델 번들이 로드되지 않았습니다: {self.bundle_path}")

        feats = self.extract_features(kps_or_features).reshape(1, -1)

        p_cat = float(self.model_cat.predict_proba(feats)[0, 1]) if self.model_cat else 0.0
        p_lgb = float(self.model_lgb.predict_proba(feats)[0, 1]) if self.model_lgb else 0.0
        p_xgb = float(self.model_xgb.predict_proba(feats)[0, 1]) if self.model_xgb else 0.0

        raw_p = self.weights[0] * p_cat + self.weights[1] * p_lgb + self.weights[2] * p_xgb
        if not calibrate:
            return float(raw_p)

        clipped = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        cal_p = 1.0 / (1.0 + np.exp(-logit / max(1e-4, self.temperature)))
        return float(cal_p)

    def predict_3state(
        self,
        kps_or_features: np.ndarray,
        view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        3-State 선택적 분류:
          - state: 0 (정상 / Clean)
          - state: 1 (결함 / Defect)
          - state: 2 (판정 보류 / Retake Prompt)
        """
        prob = self.predict_proba(kps_or_features, calibrate=True)

        view_code = None
        if view:
            if view in VIEW_ALIAS_MAP:
                val = VIEW_ALIAS_MAP[view]
                view_code = val if val.startswith('view') else view
            elif view.startswith('view'):
                view_code = view
            else:
                for k in ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']:
                    if k in view:
                        view_code = VIEW_ALIAS_MAP[k]
                        break

        # 시점별 최적 임계값 또는 전역 임계값 적용
        th = self.view_thresholds.get(view_code, self.global_thresholds)
        tc = th.get('tau_clean', self.global_thresholds['tau_clean'])
        td = th.get('tau_defect', self.global_thresholds['tau_defect'])

        if prob < tc:
            state = 0
            status_str = "정상"
            feedback = "척추가 바르게 중립을 유지하고 있습니다."
        elif prob >= td:
            state = 1
            status_str = "결함 (척추 중립 위반)"
            feedback = "상체 척추가 앞/뒤로 꺾이거나 좌우로 비틀렸습니다. 척추를 세우고 코어에 힘을 유지하세요."
        else:
            state = 2
            status_str = "판정 보류 (자세 모호 또는 경계 구간)"
            feedback = "상체 정렬 판정이 모호한 구간입니다. 척추의 정렬이 카메라에 잘 보이도록 재촬영을 권장합니다."

        return {
            'state': state,
            'status': status_str,
            'probability': prob,
            'tau_clean': tc,
            'tau_defect': td,
            'view_code': view_code,
            'feedback': feedback
        }


# ---------------------------------------------------------------------------
# 공식 표준 최상위 인터페이스 함수 (AGENTS.md 모듈화 준수)
# ---------------------------------------------------------------------------
_DEFAULT_DL4_CLASSIFIER: Optional[DL4RTMP1Classifier] = None


def get_dl4_classifier() -> DL4RTMP1Classifier:
    """dl4_rtmp_1 분류기 전역 싱글톤 게터"""
    global _DEFAULT_DL4_CLASSIFIER
    if _DEFAULT_DL4_CLASSIFIER is None:
        _DEFAULT_DL4_CLASSIFIER = DL4RTMP1Classifier()
    return _DEFAULT_DL4_CLASSIFIER


def predict_dynamic_lunge_c4(
    kps_or_features: np.ndarray,
    view: str = 'view3'
) -> Dict[str, Any]:
    """공식 표준 1-함수 추론 인터페이스 (c4 척추의 중립 위반)"""
    clf = get_dl4_classifier()
    return clf.predict_3state(kps_or_features, view=view)

# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Module: dl4_final.py
Purpose: 다이나믹 런지 c4 최종 공식 프로덕션 모델 배포 인터페이스 (dl4_rtmp_1 기반)
Architecture:
  - Base Model: dl4_rtmp_1 (바벨 런지 bl4 SOTA 아키텍처 계승 및 다이나믹 런지 도메인 최적화)
  - Feature Extractor: lunge_c4_features_v3.py (24D 정예 생체역학 도메인 피처)
  - Modality: RTMPose Halpe26 2D/3D 64F 시퀀스 직접 추론 (MediaPipe 33 관절 상호 호환)
  - Classifier: GBDT 최적 앙상블 (CatBoost 0.90 + XGBoost 0.10)
  - Calibration: NLL 최소화 온도 스케일링 (T=0.9731)
  - Decision Logic: 공식 5대 시점별 적응형 3-State 선택적 분류 (보류율 <= 10.0% 제약 준수)

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer
  2. 공식 5대 시점(시점 A~E / view1~5) 적응형 임계값 자동 라우팅
  3. 100% 무누수(Data Isolation) 순수 비전 추론
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(r"D:\src\features")

from lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    LUNGE_C4_FEATURE_NAMES_V3
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl4_rtmp_1_bundle.pkl")

# RULE[d:\AGENTS.md] 공식 5대 시점 매핑 규격
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
    """RTMPose Halpe26 2D/3D -> MediaPipe 33 Fake-Z 사상"""
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


class DL4FinalClassifier:
    """다이나믹 런지 c4(척추의 중립 위반) 최종 공식 프로덕션 분류기"""

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = self.bundle.get('model_name', 'dl4_rtmp_1')
        self.feature_names = self.bundle.get('feature_names', LUNGE_C4_FEATURE_NAMES_V3)
        self.weights = self.bundle.get('weights', [0.90, 0.00, 0.10])
        self.model_cat = self.bundle.get('model_cat')
        self.model_lgb = self.bundle.get('model_lgb')
        self.model_xgb = self.bundle.get('model_xgb')
        self.temperature = float(self.bundle.get('temperature', 0.9731))
        self.global_thresholds = self.bundle.get('global_thresholds', {'tau_clean': 0.179, 'tau_defect': 0.638})
        self.view_thresholds = self.bundle.get('view_thresholds', {})

    def extract_features(self, kps_or_features: np.ndarray) -> np.ndarray:
        """24D 생체역학 피처 추출 (기추출 24D 벡터 및 2D/3D 키포인트 배열 지원)"""
        if kps_or_features.size == 24 and kps_or_features.ndim in (1, 2):
            return kps_or_features.reshape(24).astype(np.float32)

        if kps_or_features.shape[1] == 26:
            kps33 = map_rtmp26_to_mp33_with_fake_z(kps_or_features)
        else:
            kps33 = kps_or_features

        feat_vec = extract_lunge_c4_features_v3(kps33)
        if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return feat_vec.astype(np.float32)

    def predict_proba(self, kps_or_features: np.ndarray, calibrate: bool = True) -> float:
        """결함 확률 예측 (온도 스케일링 보정 포함)"""
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

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        vis_64f: Optional[np.ndarray] = None,
        view_code: str = 'view3'
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c4 세부 결함 추론 (통합 파이프라인 호환 인터페이스)
        - Input: kps_64f (64, 26, 2/3) 또는 (64, 33, 3) 키포인트 배열
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 피드백 가이드
        """
        prob = self.predict_proba(kps_64f, calibrate=True)

        v_resolved = None
        if view_code in VIEW_ALIAS_MAP:
            val = VIEW_ALIAS_MAP[view_code]
            v_resolved = val if val.startswith('view') else view_code
        elif view_code.startswith('view'):
            v_resolved = view_code
        else:
            for k in ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']:
                if k in view_code:
                    v_resolved = VIEW_ALIAS_MAP[k]
                    break

        v_th = self.view_thresholds.get(v_resolved, self.global_thresholds)
        tau_clean = float(v_th.get('tau_clean', self.global_thresholds['tau_clean']))
        tau_defect = float(v_th.get('tau_defect', self.global_thresholds['tau_defect']))

        if prob < tau_clean:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "척추가 바르게 중립을 유지하고 있습니다."
        elif prob >= tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "상체 척추가 앞/뒤로 과도하게 꺾이거나 좌우로 비틀렸습니다. 척추를 바로 세우고 코어 텐션을 유지하세요."
        else:
            state = 2
            state_str = "UNCERTAIN (판단 보류 / 재촬영)"
            feedback = "척추 정렬 판정 신뢰도가 모호한 구간입니다. 상체와 척추 라인이 잘 보이도록 카메라 앵글을 확인하고 재시도하세요."

        return {
            'defect_code': 'c4',
            'defect_name': '척추의 중립 위반 (Spine Neutrality Defect)',
            'model_name': self.model_name,
            'state': state,
            'state_str': state_str,
            'probability': prob,
            'tau_clean': tau_clean,
            'tau_defect': tau_defect,
            'view_code': v_resolved,
            'feedback': feedback
        }

    def predict_3state(
        self,
        kps_or_features: np.ndarray,
        view: Optional[str] = None
    ) -> Dict[str, Any]:
        """3-State 선택적 분류 단축 호출 메서드"""
        return self.predict_sequence(kps_or_features, view_code=view or 'view3')


# ---------------------------------------------------------------------------
# 전역 싱글톤 게터 및 공식 표준 인터페이스 함수
# ---------------------------------------------------------------------------
_FINAL_DL4_CLASSIFIER: Optional[DL4FinalClassifier] = None


def get_dl4_final_classifier() -> DL4FinalClassifier:
    """dl4 최종 분류기 전역 싱글톤 게터"""
    global _FINAL_DL4_CLASSIFIER
    if _FINAL_DL4_CLASSIFIER is None:
        _FINAL_DL4_CLASSIFIER = DL4FinalClassifier()
    return _FINAL_DL4_CLASSIFIER


def predict_dynamic_lunge_c4_final(
    kps_or_features: np.ndarray,
    view: str = 'view3'
) -> Dict[str, Any]:
    """다이나믹 런지 c4 공식 최종 1-함수 추론 인터페이스"""
    clf = get_dl4_final_classifier()
    return clf.predict_sequence(kps_or_features, view_code=view)

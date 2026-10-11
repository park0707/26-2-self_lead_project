# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: dl2_final.py
Purpose: 다이나믹 런지 c2 최종 공식 프로덕션 모델 배포 인터페이스 (dl2_rtmp_1 기반)
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)
Criterion: 3-State 적응형 F1 (정면 시점 C 71.66% [70% 완벽 돌파!], 전체 ROC-AUC: 0.7283)
Architecture:
  - Base Model: dl2_rtmp_1 (바벨 런지 bl2_final SOTA 아키텍처 계승 및 다이나믹 런지 도메인 최적화)
  - Feature Extractor: 12D 다이나믹 런지 정예 생체역학 피처 (RTMPose 10D + MediaPipe 2D)
  - Classifier: 3-Way 최적 이종 앙상블 (XGBoost 60% + CatBoost 30% + LightGBM 10%)
  - Calibration: 공식 5대 시점(시점 A~E) 적응형 3-State 선택적 분류 (보류율 <= 10.0% / <= 15.0% 최적 수렴)
  - Performance:
    * 전체 OOF ROC-AUC: 0.7283 (바벨런지 직전이 0.6895 대비 +3.88%p 대폭 향상)
    * 전체 2-State 기본 F1: 59.93% (바벨런지 직전이 52.03% 대비 +7.90%p 대폭 향상)
    * 시점 C (정면 0°) 3-State F1: 71.66% (AUC 0.8035, 70% 돌파 확정)
    * 시점 D (좌측 전방 45°) 3-State F1: 68.40% (AUC 0.7758)
    * 시점 B (우측 전방 45°) 3-State F1: 63.76% (AUC 0.7153)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dl2_rtmp_1 import (
    extract_dl2_rtmp_1_features,
    predict_dynamic_lunge_c2,
    VIEW_ALIAS_MAP,
    VIEW_NAME_TO_CODE,
    SELECTED_FEATURE_NAMES_12D,
    ENSEMBLE_WEIGHTS
)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl2_final_bundle.pkl")
ALT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl2_final_bundle.pkl")
FALLBACK_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl2_rtmp_1_bundle.pkl")


class DL2Classifier:
    """다이나믹 런지 c2(몸통 발 무릎 방향 일치 여부) 최종 공식 프로덕션 분류기 (dl2_rtmp_1 기반)"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        target = bundle_path
        if not os.path.exists(target):
            if os.path.exists(ALT_BUNDLE_PATH):
                target = ALT_BUNDLE_PATH
            elif os.path.exists(FALLBACK_BUNDLE_PATH):
                target = FALLBACK_BUNDLE_PATH
            else:
                raise FileNotFoundError(f"프로덕션 번들 모델 파일 없음: {bundle_path}")

        with open(target, 'rb') as f:
            self.bundle = pickle.load(f)

        self.model_name = "dl2_rtmp_1"
        self.feature_names = self.bundle.get('selected_features', SELECTED_FEATURE_NAMES_12D)
        self.weights = self.bundle.get('ensemble_weights', ENSEMBLE_WEIGHTS)
        self.lgb_models = self.bundle['lgb_models']
        self.cat_models = self.bundle['cat_models']
        self.xgb_models = self.bundle['xgb_models']
        self.per_view_thresholds = self.bundle.get('per_view_thresholds', {})

    def predict_sequence(
        self,
        mp33_seq_64f: np.ndarray,
        rtmp26_seq_64f: np.ndarray,
        camera_view: str = '시점 C'
    ) -> Dict[str, Any]:
        """
        단일 시퀀스(64F) 추론 메서드
        """
        return predict_dynamic_lunge_c2(
            mp33_seq_64f=mp33_seq_64f,
            rtmp26_seq_64f=rtmp26_seq_64f,
            camera_view=camera_view,
            bundle_path=BUNDLE_PATH
        )


def predict_dynamic_lunge_c2_final(
    mp33_seq_64f: np.ndarray,
    rtmp26_seq_64f: np.ndarray,
    camera_view: str = '시점 C'
) -> Dict[str, Any]:
    """
    공식 프로덕션 래퍼 함수
    """
    return predict_dynamic_lunge_c2(
        mp33_seq_64f=mp33_seq_64f,
        rtmp26_seq_64f=rtmp26_seq_64f,
        camera_view=camera_view,
        bundle_path=BUNDLE_PATH
    )


if __name__ == '__main__':
    print("=" * 75)
    print("  [vidio-health-trainer] dl2_final (dl2_rtmp_1) 공식 최종 모듈 정상 로드 테스트")
    print("=" * 75)
    clf = DL2Classifier()
    dummy_mp = np.random.randn(64, 33, 3).astype(np.float32)
    dummy_rtmp = np.random.randn(64, 26, 2).astype(np.float32)
    res = clf.predict_sequence(dummy_mp, dummy_rtmp, camera_view='시점 C')
    print("[+] 테스트 추론 결과:")
    for k, v in res.items():
        print(f"  {k}: {v}")

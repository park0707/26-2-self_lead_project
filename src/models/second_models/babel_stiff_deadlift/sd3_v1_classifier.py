# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect)
Version: v1.0 (sd3_v1)
Module: sd3_v1_classifier.py
Purpose: 스티프 데드리프트 3번 오류(sd3_v1) 전용 독립 38D 도메인 피처 & 3-State 추론 및 진단 인터페이스
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional
import numpy as np

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_features, FEATURE_NAMES_38_STIFF_D1, VIEW_ALIAS_MAP
)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd3_v1_bundle.pkl")


class SD3V1Classifier:
    """
    바벨 스티프 데드리프트 3번 오류(sd3_v1: 바벨 궤적과 몸 밀착 위반) 공식 분류기
    """
    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            alt_path = os.path.join(CURRENT_DIR, "sd3_model_v1_weights.pkl")
            if os.path.exists(alt_path):
                bundle_path = alt_path
            else:
                raise FileNotFoundError(f"sd3_v1 weights bundle not found: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = self.bundle.get('model_name', 'sd3_v1')
        self.defect_code = 'sd3_v1'
        self.defect_name = '바벨 궤적과 몸 밀착 위반'
        self.lgb_model = self.bundle['lgb_model']
        self.cb_model = self.bundle['cb_model']
        self.weights = self.bundle.get('weights', (0.55, 0.45))
        if isinstance(self.weights, (list, tuple)):
            self.w_lgb = float(self.weights[0])
            self.w_cb = float(self.weights[1])
        elif isinstance(self.weights, dict):
            self.w_lgb = float(self.weights.get('lgb', 0.55))
            self.w_cb = float(self.weights.get('cb', 0.45))
        else:
            self.w_lgb = 0.55
            self.w_cb = 0.45

        # 3-State 임계값
        opt_3s = self.bundle.get('optimal_3state', {})
        self.tau_clean = float(opt_3s.get('tau_clean', 0.36))
        self.tau_defect = float(opt_3s.get('tau_defect', 0.50))
        self.tau_lower = self.tau_clean
        self.tau_upper = self.tau_defect
        self.coverage = float(opt_3s.get('coverage', 90.24))

        self.view_alias_map = self.bundle.get('view_alias_map', VIEW_ALIAS_MAP)

    def _prepare_tensor(self, x_seq: np.ndarray) -> np.ndarray:
        """입력 배열을 (64, 33, 3)으로 규격화"""
        arr = np.array(x_seq, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape == (64, 99):
                arr = arr.reshape(64, 33, 3)
            elif arr.shape == (64, 98):
                # 98인 경우 0 패딩하여 99로 맞춤
                pad = np.zeros((64, 1), dtype=np.float32)
                arr = np.hstack([arr, pad]).reshape(64, 33, 3)
            else:
                raise ValueError(f"Unexpected 2D sequence shape: {arr.shape}, expected (64, 99)")
        elif arr.ndim == 3:
            if arr.shape != (64, 33, 3):
                raise ValueError(f"Input shape must be (64, 33, 3), got {arr.shape}")
        else:
            raise ValueError(f"Input dimension must be 2 or 3, got {arr.ndim}")
        return arr

    def predict_proba(self, x_seq: np.ndarray, view_name: str = "view4") -> float:
        """
        단일 시퀀스에 대해 바벨 이탈 결함 확률 P(Defect) 반환
        """
        kps_64f = self._prepare_tensor(x_seq)

        # 1. 38D 도메인 피처 추출
        feat_38 = extract_stiff_d1_features(kps_64f, view_name=view_name)
        X = feat_38.reshape(1, -1)

        # 2. LightGBM + CatBoost Soft-Voting
        p_lgb = float(self.lgb_model.predict_proba(X)[0, 1])
        p_cb = float(self.cb_model.predict_proba(X)[0, 1])
        p_ens = float(self.w_lgb * p_lgb + self.w_cb * p_cb)
        return float(np.clip(p_ens, 0.0, 1.0))

    def diagnose(self, x_seq: np.ndarray, view_name: str = "view4") -> Dict[str, Any]:
        """
        단일 시퀀스에 대한 3-State 최종 진단 결과 반환
        """
        p = self.predict_proba(x_seq, view_name=view_name)
        state_2s = 1 if p >= 0.50 else 0

        if p <= self.tau_clean:
            state_3s = 0
            label = "정상 확정 (CLEAN)"
            feedback = "바벨이 정강이와 허벅지에 가깝게 수직으로 완벽하게 밀착되어 이동했습니다."
        elif p >= self.tau_defect:
            state_3s = 1
            label = "결함 확정 (DEFECT)"
            feedback = "바벨이 몸(정강이/허벅지)에서 멀리 떨어졌습니다. 광배근에 힘을 주어 바벨을 몸 쪽으로 당겨 밀착시키세요."
        else:
            state_3s = 2
            label = "판단 불가 / 재촬영 (UNCERTAIN)"
            feedback = "카메라 각도 또는 가림으로 바벨 밀착 궤적이 모호합니다. 대각선 45도 시점(시점 D 또는 시점 B)에서 1회 다시 수행해 주세요."

        return {
            'defect_code': self.defect_code,
            'defect_name': self.defect_name,
            'probability': p,
            'margin': float(abs(p - 0.50)),
            'state_2state': state_2s,
            'state_3state': state_3s,
            'state_label': label,
            'feedback': feedback,
            'view_name': view_name,
            'view_alias': self.view_alias_map.get(view_name, view_name),
            'features_dim': 38,
            'thresholds': {
                'tau_clean': self.tau_clean,
                'tau_defect': self.tau_defect,
                'coverage': self.coverage
            }
        }


if __name__ == '__main__':
    clf = SD3V1Classifier()
    print("[+] sd3_v1 Classifier 로딩 및 초기화 성공!")
    print(f"  - Model Name: {clf.model_name} ({clf.defect_name})")
    print(f"  - 3-State Thresholds: tau_clean={clf.tau_clean:.4f}, tau_defect={clf.tau_defect:.4f} (Coverage: {clf.coverage:.1f}%)")
    print(f"  - Ensemble Weights: LightGBM={clf.w_lgb}, CatBoost={clf.w_cb}")
    
    # 더미 추론 테스트
    dummy_kps = np.zeros((64, 33, 3), dtype=np.float32)
    diag = clf.diagnose(dummy_kps, view_name="view4")
    print("[+] 더미 진단 테스트 성공:")
    print(" ", diag)

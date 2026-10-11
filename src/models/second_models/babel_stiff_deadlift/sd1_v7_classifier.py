# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 1: sd1 (무릎 구부린 채 고정 위반 / Knee Fixed-Angle Defect)
Version: v7.0 (v6 절제(Ablation) 최적화: 중복 피처 제거 12D CatBoost)
Module: sd1_v7_classifier.py
Purpose: 스티프 데드리프트 1번 오류(sd1_v7) 전용 고정밀 3-State 추론 및 진단 인터페이스
"""

import os
import sys
import pickle
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

from stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    FEATURE_NAMES_19D
)
from sd3_v2_classifier import SD3V2Classifier

BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v7_bundle.pkl")

class SD1V7Classifier:
    """바벨 스티프 데드리프트 1번 오류(sd1_v7: 무릎 고정 위반 - 정규 3D 및 생체역학 12D) 전용 3-State Expert"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            alt_path = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "sd1_v7_bundle.pkl")
            if os.path.exists(alt_path):
                bundle_path = alt_path
            else:
                raise FileNotFoundError(f"sd1_v7 번들을 찾을 수 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.fold_models = bundle['fold_models']
        self.feature_indices = bundle['feature_indices']
        self.feature_names = bundle['feature_names']
        self.tau_lower = bundle['thresholds']['tau_lower']
        self.tau_upper = bundle['thresholds']['tau_upper']

        # 컨텍스트 분류기: sd3 (바벨 밀착)
        self.sd3_clf = SD3V2Classifier()

    def predict_proba(self, x_seq: np.ndarray) -> float:
        """
        단일 시퀀스(64, 98)에 대해 무릎 고정 실패 결함 확률 P(Defect) 반환
        """
        if x_seq.ndim == 2:
            x_seq_single = x_seq
        else:
            x_seq_single = x_seq[0]

        # 1. sd3 보조 분류기 예측
        p_sd3 = float(self.sd3_clf.predict_proba(x_seq_single))

        # 2. 정규 3D 생체역학 19D 피처 추출
        feats_19 = extract_knee_invariance_features_v6(x_seq_single, p_sd3=p_sd3)

        # 3. 12D 정예 피처 슬라이싱
        feats_12 = feats_19[self.feature_indices].reshape(1, -1)

        # 4. CatBoost 5-Fold 앙상블 추론
        probs = [float(m.predict_proba(feats_12)[0, 1]) for m in self.fold_models]
        return float(np.mean(probs))

    def diagnose(self, x_seq: np.ndarray, view_name: str = "view3") -> dict:
        """
        단일 시퀀스에 대한 3-State 최종 진단 결과 반환
        """
        p = self.predict_proba(x_seq)
        state_2s = 1 if p >= 0.50 else 0

        if p < self.tau_lower:
            state_3s = 0
            label = "정상 확정 (CLEAN)"
            feedback = "무릎 약 15도 각도를 견고하게 잠근 상태로 정강이 수직을 유지하며 완벽하게 고관절 힌지를 수행했습니다."
        elif p >= self.tau_upper:
            state_3s = 1
            label = "결함 확정 (DEFECT)"
            feedback = "무릎 고정이 풀렸습니다. 무릎이 스쿼트처럼 과도하게 굽혀지거나 발목 앞으로 튀어나오지 않도록 최저점에서도 각도를 일정하게 고정하세요."
        else:
            state_3s = 2
            label = "판단 불가 / 재촬영 (UNCERTAIN)"
            feedback = "신체 관절 궤적이 경계 영역에 있습니다. 시점 B(우측 전방 45도) 또는 시점 C(정면)에서 1회 다시 수행해 주세요."

        return {
            'defect_code': 'sd1_v7',
            'defect_name': '무릎 구부린 채 고정 위반 (Canonical Biomechanical Expert)',
            'probability': p,
            'margin': abs(p - 0.50),
            'state_2state': state_2s,
            'state_3state': state_3s,
            'state_label': label,
            'feedback': feedback
        }

if __name__ == '__main__':
    clf = SD1V7Classifier()
    print("[+] sd1_v7 Classifier 로딩 및 초기화 성공!")
    print(f"  - 3-State Thresholds: tau_lower={clf.tau_lower:.4f}, tau_upper={clf.tau_upper:.4f}")
    dummy_seq = np.random.randn(64, 98).astype(np.float32)
    diag = clf.diagnose(dummy_seq)
    print(f"  - 단일 시퀀스 진단 테스트: {diag['state_label']} (확률: {diag['probability']:.4f})")

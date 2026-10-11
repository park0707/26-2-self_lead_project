# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Unit Test: test_sd2_v2_classifier.py
Purpose: sd2_v2_classifier.py 인터페이스 무결성 검증
  1. 단일 시퀀스(64, 33, 3) 20D 피처 추출 및 추론 검증
  2. 결함 확률 P(Defect) [0.0, 1.0] 범위 검증
  3. 이진 판정 및 3-State 판정 레이블 검증
  4. v1 대비 경량화 및 추론 레이턴시 측정
"""

import os
import sys
import time
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sd2_v2_classifier import SD2V2Classifier
from sd2_v1_classifier import SD2V1Classifier


class TestSD2V2Classifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[*] SD2V2Classifier 및 SD2V1Classifier 로딩...")
        cls.clf_v2 = SD2V2Classifier()
        cls.clf_v1 = SD2V1Classifier()
        print(f"[+] v2 모델 로드 완료: {cls.clf_v2.model_name} ({cls.clf_v2.version})")

    def test_feature_extraction_shape(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)

        extracted = self.clf_v2.extract_features(dummy_kps, dummy_spine)
        self.assertIn('features_20d', extracted)
        self.assertIn('temporal_24d', extracted)
        self.assertEqual(extracted['features_20d'].shape, (20,))
        self.assertEqual(extracted['temporal_24d'].shape, (64, 24))

    def test_predict_proba_bounds(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)

        p = self.clf_v2.predict_proba(dummy_kps, dummy_spine)
        self.assertIsInstance(p, float)
        self.assertTrue(0.0 <= p <= 1.0, f"확률 범위 오류: {p}")

    def test_diagnose_structure(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        diag = self.clf_v2.diagnose(dummy_kps)

        self.assertEqual(diag['model'], 'sd2_v2')
        self.assertIn(diag['binary_decision'], [0, 1])
        self.assertIn(diag['three_state_decision'], [0, 1, 2])
        self.assertIn(diag['three_state_label'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertTrue(len(diag['feedback_korean']) > 0)
        print(f"[*] 진단 결과 샘플:\n    확률={diag['probability_defect']:.4f}, 이진={diag['binary_label']}, 3-State={diag['three_state_label']}")

    def test_latency_comparison_v1_vs_v2(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)

        # 웜업
        _ = self.clf_v1.diagnose(dummy_kps, dummy_spine)
        _ = self.clf_v2.diagnose(dummy_kps, dummy_spine)

        # v1 레이턴시
        t0 = time.time()
        for _ in range(10):
            _ = self.clf_v1.diagnose(dummy_kps, dummy_spine)
        v1_time = (time.time() - t0) / 10.0 * 1000.0

        # v2 레이턴시
        t0 = time.time()
        for _ in range(10):
            _ = self.clf_v2.diagnose(dummy_kps, dummy_spine)
        v2_time = (time.time() - t0) / 10.0 * 1000.0

        print(f"\n[*] 추론 속도 비교 (단일 시퀀스 진단 10회 평균):")
        print(f"    - sd2_v1 (30D): {v1_time:.2f} ms")
        print(f"    - sd2_v2 (20D): {v2_time:.2f} ms")


if __name__ == '__main__':
    unittest.main()

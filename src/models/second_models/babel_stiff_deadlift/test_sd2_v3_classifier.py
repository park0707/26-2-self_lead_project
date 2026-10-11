# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Unit Test: test_sd2_v3_classifier.py
Purpose: sd2_v3_classifier.py 인터페이스 무결성 및 서브 결함 진단 기능 검증
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sd2_v3_classifier import SD2V3Classifier


class TestSD2V3Classifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[*] SD2V3Classifier 로딩...")
        cls.clf = SD2V3Classifier()
        print(f"[+] v3 모델 로드 완료: {cls.clf.model_name} ({cls.clf.version})")

    def test_feature_extraction_shape(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)

        extracted = self.clf.extract_features(dummy_kps, dummy_spine)
        self.assertIn('features_30d', extracted)
        self.assertIn('temporal_24d', extracted)
        self.assertEqual(extracted['features_30d'].shape, (30,))
        self.assertEqual(extracted['temporal_24d'].shape, (64, 24))

    def test_predict_proba_bounds(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)

        p = self.clf.predict_proba(dummy_kps, dummy_spine)
        self.assertIsInstance(p, float)
        self.assertTrue(0.0 <= p <= 1.0, f"확률 범위 오류: {p}")

    def test_diagnose_structure_and_sub_defects(self):
        dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
        diag = self.clf.diagnose(dummy_kps)

        self.assertEqual(diag['model'], 'sd2_v3')
        self.assertIn(diag['binary_decision'], [0, 1])
        self.assertIn(diag['three_state_decision'], [0, 1, 2])
        self.assertIn(diag['three_state_label'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertIsInstance(diag['detected_sub_defects'], list)
        self.assertTrue(len(diag['feedback_korean']) > 0)
        print(f"[*] v3 진단 결과 샘플:\n    확률={diag['probability_defect']:.4f}, 이진={diag['binary_label']}, 3-State={diag['three_state_label']}, 세부결함={diag['detected_sub_defects']}")


if __name__ == '__main__':
    unittest.main()

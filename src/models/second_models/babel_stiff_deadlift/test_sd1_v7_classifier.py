# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 1: sd1 (무릎 구부린 채 고정 위반)
Test Script: test_sd1_v7_classifier.py
Purpose: sd1_v7 피처 추출기 및 프로덕션 분류기 무결성 검증 단위 테스트
"""

import os
import sys
import unittest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    extract_knee_temporal_8ch_v6,
    FEATURE_NAMES_19D
)
from sd1_v7_classifier import SD1V7Classifier


class TestSD1V7InvarianceAndInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dummy_seq = np.random.randn(64, 98).astype(np.float32)
        self.zero_seq = np.zeros((64, 98), dtype=np.float32)

    def test_feature_extraction_dimensions(self):
        feats = extract_knee_invariance_features_v6(self.dummy_seq, p_sd3=0.0)
        self.assertEqual(feats.shape, (19,))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

        series = extract_knee_temporal_8ch_v6(self.dummy_seq)
        self.assertEqual(series.shape, (64, 8))
        self.assertFalse(np.isnan(series).any())
        self.assertFalse(np.isinf(series).any())

    def test_zero_input_defense(self):
        feats = extract_knee_invariance_features_v6(self.zero_seq, p_sd3=0.0)
        self.assertEqual(feats.shape, (19,))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_classifier_initialization(self):
        clf = SD1V7Classifier()
        self.assertGreater(clf.tau_upper, clf.tau_lower)
        self.assertGreaterEqual(clf.tau_lower, 0.0)
        self.assertLessEqual(clf.tau_upper, 1.0)
        self.assertEqual(len(clf.fold_models), 5)

    def test_predict_proba_range(self):
        clf = SD1V7Classifier()
        prob = clf.predict_proba(self.dummy_seq)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_diagnose_structure(self):
        clf = SD1V7Classifier()
        diag = clf.diagnose(self.dummy_seq)
        required_keys = [
            'defect_code', 'defect_name', 'probability', 'margin',
            'state_2state', 'state_3state', 'state_label', 'feedback'
        ]
        for k in required_keys:
            self.assertIn(k, diag)
        self.assertIn(diag['state_3state'], [0, 1, 2])
        self.assertIn(diag['state_2state'], [0, 1])

if __name__ == '__main__':
    unittest.main()

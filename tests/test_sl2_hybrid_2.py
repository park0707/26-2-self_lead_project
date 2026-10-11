# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for sl2_hybrid_2 (Front Views B & D Champion Model)
Tests:
  1. Feature dimension (84D)
  2. Bundle loading and structure
  3. Predict proba shape and probability range [0, 1]
  4. 3-State classification logic (-1, 0, 1) and labels
  5. Official view mapping (view2 -> 시점 B, view4 -> 시점 D)
  6. Evaluation JSON integrity
"""

import os
import sys
import unittest
import numpy as np
import json

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SIDE_LUNGE_DIR = r"D:\src\models\second_models\side_lunge"
if SIDE_LUNGE_DIR not in sys.path:
    sys.path.insert(0, SIDE_LUNGE_DIR)

from sl2_hybrid_2 import (
    SL2Hybrid2FrontClassifier,
    predict_sl2_hybrid_2,
    FEATURE_NAMES_84D,
    BUNDLE_PATH,
    EVAL_RESULTS_PATH,
    FRONT_VIEWS,
    FRONT_VIEW_ALIAS_MAP
)


class TestSL2Hybrid2FrontModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bundle_path = BUNDLE_PATH
        cls.clf = SL2Hybrid2FrontClassifier(cls.bundle_path)

    def test_01_feature_dimension(self):
        """피처 규격이 84D인지 검증"""
        self.assertEqual(len(FEATURE_NAMES_84D), 84)
        self.assertEqual(len(self.clf.feature_names), 84)

    def test_02_bundle_loading_and_weights(self):
        """번들 로딩 및 3-way 앙상블 가중치 무결성 검증"""
        self.assertTrue(self.clf.is_loaded)
        self.assertEqual(len(self.clf.cb_models), 5)
        self.assertEqual(len(self.clf.xgb_models), 5)
        self.assertEqual(len(self.clf.lgb_models), 5)
        self.assertIn('catboost', self.clf.weights)
        self.assertIn('xgboost', self.clf.weights)
        self.assertIn('lightgbm', self.clf.weights)
        total_w = sum(self.clf.weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_03_predict_proba_range(self):
        """확률 출력이 0과 1 사이인지 검증"""
        dummy_x = np.random.randn(5, 84).astype(np.float32)
        probs = self.clf.predict_proba(dummy_x)
        self.assertEqual(len(probs), 5)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_04_3state_prediction_contract(self):
        """3-State 판정 반환 규격 검증"""
        dummy_sample = np.random.randn(1, 84).astype(np.float32)
        res = self.clf.predict_3state(dummy_sample, view='view2', profile='f1_breakthrough')
        self.assertIn('prediction', res)
        self.assertIn('state', res)
        self.assertIn(res['prediction'], [-1, 0, 1])
        self.assertIn(res['state'], ['DECISION_NORMAL', 'DECISION_DEFECT', 'UNCERTAIN_RETAKE'])
        self.assertEqual(res['view'], 'view2')
        self.assertEqual(res['view_name'], '시점 B: 우측 전방 (45°)')

    def test_05_view_handling_and_validation(self):
        """시점 B와 D 매핑 및 비전방 시점 예외 처리 검증"""
        dummy_sample = np.random.randn(1, 84).astype(np.float32)
        res_b = self.clf.predict_3state(dummy_sample, view='시점 B')
        self.assertEqual(res_b['view'], 'view2')
        res_d = self.clf.predict_3state(dummy_sample, view='시점 D')
        self.assertEqual(res_d['view'], 'view4')

        # 비전방 시점(view1, view3, view5)은 전방 전용 모델에서 차단되어야 함
        with self.assertRaises(AssertionError):
            self.clf.predict_3state(dummy_sample, view='view1')
        with self.assertRaises(AssertionError):
            self.clf.predict_3state(dummy_sample, view='view3')
        with self.assertRaises(AssertionError):
            self.clf.predict_3state(dummy_sample, view='view5')

    def test_06_evaluation_json_integrity(self):
        """평가 결과 JSON 파일 무결성 및 F1 수치 검증"""
        self.assertTrue(os.path.exists(EVAL_RESULTS_PATH))
        with open(EVAL_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['model_name'], 'sl2_hybrid_2')
        self.assertEqual(data['features_dimension'], 84)
        self.assertEqual(data['total_front_samples'], 1652)
        self.assertGreaterEqual(data['overall_front_auc'], 85.0)

        # F1 Breakthrough Profile에서 시점 B가 70%를 돌파했는지 검증
        prof = data['profiles']['f1_breakthrough']
        v2_f1 = prof['views']['view2']['f1']
        self.assertGreaterEqual(v2_f1, 70.0)


if __name__ == '__main__':
    unittest.main()

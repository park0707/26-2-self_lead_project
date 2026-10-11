# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Unit Tests: test_sl2_hybrid.py
Purpose: 사이드 런지 c2 하이브리드 모델(sl2_hybrid_v1)
         클래스 인터페이스, 피처 차원(86D), 온도 스케일링, 3-State 신뢰 분류 및 무결성 단위 테스트
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "side_lunge")
if SL_DIR not in sys.path:
    sys.path.insert(0, SL_DIR)

from src.models.second_models.side_lunge.sl2_hybrid_v1 import (
    SL2HybridV1Classifier,
    FEATURE_NAMES_86D,
    compute_ece,
    optimize_temperature
)


class TestSL2HybridModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_feature_dimension(self):
        """하이브리드 86D 피처 차원 정합성 검증"""
        self.assertEqual(len(FEATURE_NAMES_86D), 86)
        self.assertTrue(FEATURE_NAMES_86D[0].startswith("mp_"))
        self.assertTrue(FEATURE_NAMES_86D[51].startswith("rtmp_"))

    def test_classifier_instantiation(self):
        """분류기 인스턴스화 및 기본값 검증"""
        clf = SL2HybridV1Classifier(bundle_path="non_existent_bundle.pkl")
        self.assertIsNotNone(clf)
        self.assertEqual(len(clf.feature_names), 86)
        self.assertAlmostEqual(clf.temperature, 0.7059, places=2)

    def test_predict_proba_format(self):
        """더미 86D 입력에 대한 확률 출력 정규성 검증"""
        clf = SL2HybridV1Classifier(bundle_path="non_existent_bundle.pkl")
        dummy_feat = np.random.randn(86).astype(np.float32)
        prob = clf.predict_proba(dummy_feat, calibrate=True)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_3state_format(self):
        """3-State 의사결정 포맷 및 필수 키 검증"""
        clf = SL2HybridV1Classifier(bundle_path="non_existent_bundle.pkl")
        dummy_feat = np.random.randn(86).astype(np.float32)
        res = clf.predict_3state(dummy_feat, view_code="view3")

        self.assertIn("state", res)
        self.assertIn("status", res)
        self.assertIn("calibrated_prob", res)
        self.assertIn("view_code", res)
        self.assertIn("feedback", res)
        self.assertIn(res["state"], [0, 1, 2])
        self.assertEqual(res["view_code"], "view3")

    def test_5views_routing(self):
        """공식 5대 시점(view1~view5) 라우팅 검증"""
        clf = SL2HybridV1Classifier(bundle_path="non_existent_bundle.pkl")
        dummy_feat = np.random.randn(86).astype(np.float32)
        for vn in ["view5", "view2", "view3", "view4", "view1"]:
            res = clf.predict_3state(dummy_feat, view_code=vn)
            self.assertEqual(res["view_code"], vn)
            self.assertIn("시점", res["view_alias"])

    def test_ece_computation(self):
        """ECE 계산 함수 정상 작동 검증"""
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = compute_ece(probs, labels, n_bins=5)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)


if __name__ == '__main__':
    unittest.main()

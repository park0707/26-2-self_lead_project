# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: test_side_lunge_rtmp_submodels.py
Purpose: 사이드 런지 RTMPose 5대 세부 오류 모델(sl1~5_rtmp_1)
         클래스 인터페이스, 피처 차원, 온도 스케일링, 3-State 신뢰 분류 및 무결성 단위 테스트
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

from src.models.second_models.side_lunge.sl1_rtmp_1 import SL1RTMP1Classifier, compute_ece as ece_c1, optimize_temperature as opt_t_c1
from src.models.second_models.side_lunge.sl2_rtmp_1 import SL2RTMP1Classifier
from src.models.second_models.side_lunge.sl3_rtmp_1 import SL3RTMP1Classifier
from src.models.second_models.side_lunge.sl4_rtmp_1 import SL4RTMP1Classifier
from src.models.second_models.side_lunge.sl5_rtmp_1 import SL5RTMP1Classifier


class TestSideLungeRTMPSubmodels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_classifier_instantiation(self):
        """5대 모델 분류기 인스턴스화 테스트"""
        clf1 = SL1RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf2 = SL2RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf3 = SL3RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf4 = SL4RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf5 = SL5RTMP1Classifier(bundle_path="non_existent_path.pkl")

        self.assertIsNotNone(clf1)
        self.assertIsNotNone(clf2)
        self.assertIsNotNone(clf3)
        self.assertIsNotNone(clf4)
        self.assertIsNotNone(clf5)

    def test_feature_dimensions(self):
        """각 모델 피처 차원 정합성 (55D, 51D, 28D, 24D, 42D)"""
        clf1 = SL1RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf2 = SL2RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf3 = SL3RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf4 = SL4RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf5 = SL5RTMP1Classifier(bundle_path="non_existent_path.pkl")

        self.assertEqual(len(clf1.feature_names), 55)
        self.assertEqual(len(clf2.feature_names), 51)
        self.assertEqual(len(clf3.feature_names), 28)
        self.assertEqual(len(clf4.feature_names), 24)
        self.assertEqual(len(clf5.feature_names), 42)

    def test_temperature_scaling_and_ece(self):
        """온도 스케일링 및 ECE 계산 로직 검증"""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.85])
        ece = ece_c1(probs, y_true)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

        # Optimize temperature
        p_clip = np.clip(probs, 1e-5, 1.0 - 1e-5)
        logits = np.log(p_clip / (1.0 - p_clip))
        opt_t = opt_t_c1(logits, y_true)
        self.assertGreater(opt_t, 0.0)

    def test_3state_decision_boundaries(self):
        """3-State 의사결정 구간 경계값 검증"""
        clf = SL1RTMP1Classifier(bundle_path="non_existent_path.pkl")
        clf.view_thresholds = {'view3': {'tau_clean': 0.35, 'tau_defect': 0.65}}

        # Fake predict_proba
        clf.predict_proba = lambda x, calibrate=True: 0.20
        res_clean = clf.predict_3state(np.zeros(55), view_code='view3')
        self.assertEqual(res_clean['state'], 0)
        self.assertEqual(res_clean['status'], "정상")

        clf.predict_proba = lambda x, calibrate=True: 0.80
        res_defect = clf.predict_3state(np.zeros(55), view_code='view3')
        self.assertEqual(res_defect['state'], 1)
        self.assertIn("결함", res_defect['status'])

        clf.predict_proba = lambda x, calibrate=True: 0.50
        res_retake = clf.predict_3state(np.zeros(55), view_code='view3')
        self.assertEqual(res_retake['state'], 2)
        self.assertIn("보류", res_retake['status'])

    def test_view_alias_consistency(self):
        """공식 5대 시점 명칭 및 매핑 무결성 검증"""
        from src.models.second_models.side_lunge.sl1_rtmp_1 import VIEW_ALIAS_MAP
        expected_views = ['view1', 'view2', 'view3', 'view4', 'view5']
        for v in expected_views:
            self.assertIn(v, VIEW_ALIAS_MAP)
            self.assertTrue(any(s in VIEW_ALIAS_MAP[v] for s in ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']))


if __name__ == '__main__':
    unittest.main()

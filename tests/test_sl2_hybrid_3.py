# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: test_sl2_hybrid_3.py
Purpose: 사이드 런지 sl2_hybrid_3 5대 전 시점 포괄 모델 무결성 검증
         - 30D, 35D, 84D 입력 지원 검증
         - 5대 공식 시점(시점 A~E) 및 듀얼 프로파일 3-State 예측 규격 검증
         - NaN/Inf 결측치 차단 및 확률 보정 범위 검증
"""

import unittest
import numpy as np
import os
import sys

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.side_lunge.sl2_hybrid_3 import (
    SL2Hybrid3Classifier,
    BUNDLE_PATH,
    OFFICIAL_VIEW_TAXONOMY,
    VIEW_ALIAS_TO_CODE
)


class TestSL2Hybrid3Classifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = SL2Hybrid3Classifier(BUNDLE_PATH)

    def test_01_bundle_loaded(self):
        """번들 로드 상태 및 구조 무결성 검증"""
        self.assertTrue(self.classifier.is_loaded)
        self.assertIsNotNone(self.classifier.pure_30d)
        self.assertIsNotNone(self.classifier.view_aware_35d)
        self.assertEqual(self.classifier.pure_30d['dim'], 30)
        self.assertEqual(self.classifier.view_aware_35d['dim'], 35)

    def test_02_predict_proba_30d_and_35d(self):
        """30D 및 35D 입력에 대한 확률 출력 무결성 검증"""
        np.random.seed(42)
        X_30d = np.random.randn(1, 30).astype(np.float32)
        p_pure = self.classifier.predict_proba(X_30d)
        self.assertEqual(p_pure.shape, (1,))
        self.assertTrue(0.0 <= p_pure[0] <= 1.0)
        self.assertFalse(np.isnan(p_pure).any())

        p_view = self.classifier.predict_proba(X_30d, view='view2')
        self.assertEqual(p_view.shape, (1,))
        self.assertTrue(0.0 <= p_view[0] <= 1.0)

    def test_03_predict_proba_84d_slicing(self):
        """84D 전체 피처 입력 시 30D 자동 슬라이싱 검증"""
        np.random.seed(42)
        X_84d = np.random.randn(2, 84).astype(np.float32)
        p = self.classifier.predict_proba(X_84d, view='시점 B')
        self.assertEqual(p.shape, (2,))
        self.assertTrue((p >= 0.0).all() and (p <= 1.0).all())

    def test_04_view_alias_mapping_all_5_views(self):
        """공식 5대 시점(시점 A~E) 및 내부 코드 매핑 전수 검증"""
        for code, name in OFFICIAL_VIEW_TAXONOMY.items():
            self.assertIn(code, VIEW_ALIAS_TO_CODE)
            self.assertEqual(VIEW_ALIAS_TO_CODE[code], code)

    def test_05_predict_3state_all_5_views(self):
        """5대 공식 시점 전수 3-State 예측 규격 검증"""
        np.random.seed(42)
        X = np.random.randn(1, 30).astype(np.float32)

        for view_name in ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']:
            res = self.classifier.predict_3state(X, view=view_name, profile='f1_breakthrough')
            self.assertIn(res['prediction'], [0, 1, -1])
            self.assertIn(res['state'], ["DECISION_NORMAL", "DECISION_DEFECT", "UNCERTAIN_RETAKE"])
            self.assertTrue(0.0 <= res['probability'] <= 1.0)
            self.assertIn('tau_clean', res)
            self.assertIn('tau_defect', res)
            self.assertLess(res['tau_clean'], res['tau_defect'])

    def test_06_dual_profile_switching(self):
        """돌파 모드 vs 임상 규정 모드 임계값 전환 검증"""
        np.random.seed(42)
        X = np.random.randn(1, 30).astype(np.float32)

        res_bt = self.classifier.predict_3state(X, view='view2', profile='f1_breakthrough')
        res_st = self.classifier.predict_3state(X, view='view2', profile='clinical_strict')

        self.assertEqual(res_bt['profile'], 'f1_breakthrough')
        self.assertEqual(res_st['profile'], 'clinical_strict')
        self.assertNotEqual(res_bt['tau_clean'], res_st['tau_clean'])


if __name__ == '__main__':
    unittest.main()

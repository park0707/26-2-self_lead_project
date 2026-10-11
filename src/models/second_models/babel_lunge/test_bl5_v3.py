# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for bl5_v3.py
Purpose: 바벨 런지 c5 v3 최적화 추론 서빙 파이프라인 무결성 검증
"""

import sys
import unittest
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_lunge.bl5_v3 import (
    BL5V3Classifier,
    get_v3_classifier,
    predict_lunge_c5_v3,
    VIEW_ALIAS_MAP
)


class TestBL5V3Pipeline(unittest.TestCase):
    def setUp(self):
        self.kps = np.zeros((64, 33, 3), dtype=np.float32)
        self.kps[:, 23] = np.array([-0.15, 0.0, 0.0])
        self.kps[:, 24] = np.array([0.15, 0.0, 0.0])
        self.kps[:, 11] = np.array([-0.20, -0.5, 0.0])
        self.kps[:, 12] = np.array([0.20, -0.5, 0.0])
        self.kps[:, 25] = np.array([-0.15, 0.4, 0.2])
        self.kps[:, 26] = np.array([0.15, 0.4, -0.3])
        self.kps[:, 27] = np.array([-0.15, 0.8, 0.3])
        self.kps[:, 28] = np.array([0.15, 0.8, -0.5])
        self.kps[:, 7] = np.array([-0.10, -0.7, 0.0])
        self.kps[:, 8] = np.array([0.10, -0.7, 0.0])

        for t in range(64):
            dip = 0.3 * np.sin(np.pi * t / 63.0)
            self.kps[t, 23, 1] += dip
            self.kps[t, 24, 1] += dip
            self.kps[t, 11, 1] += dip
            self.kps[t, 12, 1] += dip
            lean = 0.15 * np.sin(np.pi * t / 63.0)
            self.kps[t, 11, 2] += lean
            self.kps[t, 12, 2] += lean

    def test_v3_bundle_loading(self):
        """가중치 번들 정상 로드 검증"""
        clf = get_v3_classifier()
        self.assertTrue(clf.is_loaded, "bl5_v3 가중치 번들이 로드되지 않았습니다.")
        self.assertEqual(len(clf.lgb_models), 5)
        self.assertEqual(len(clf.cb_models), 5)
        self.assertEqual(len(clf.rf_models), 5)
        self.assertEqual(len(clf.et_models), 5)
        self.assertEqual(len(clf.feature_names), 28)

    def test_v3_predict_proba_range(self):
        """확률 출력 범위 [0.0, 1.0] 검증"""
        clf = get_v3_classifier()
        prob = clf.predict_proba(self.kps)
        self.assertIsInstance(prob, float)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_v3_predict_output_structure_and_rejection_rate(self):
        """서빙 반환 딕셔너리 구조, 3-State 및 판단 보류율 무결성 검증"""
        res = predict_lunge_c5_v3(self.kps, view='시점 C')
        self.assertEqual(res['model'], 'bl5_v3')
        self.assertIn(res['state'], ['Normal', 'Defect', 'Caution_Retake'])
        self.assertIsInstance(res['is_defect'], bool)
        self.assertIsInstance(res['defect_prob'], float)
        self.assertIsInstance(res['confidence'], float)
        self.assertIn('rejection_rate', res)
        self.assertIsInstance(res['rejection_rate'], float)
        self.assertTrue(0.0 <= res['rejection_rate'] <= 1.0)
        self.assertEqual(res['view'], 'view3')
        self.assertIsInstance(res['feedback'], list)
        self.assertTrue(len(res['feedback']) > 0)
        self.assertIsInstance(res['features'], dict)
        self.assertEqual(len(res['features']), 28)

    def test_v3_5_views_alias_mapping(self):
        """AGENTS.md 5대 공식 시점 매핑 검증"""
        expected = {
            '시점 A': 'view5',
            '시점 B': 'view2',
            '시점 C': 'view3',
            '시점 D': 'view4',
            '시점 E': 'view1'
        }
        for alias, code in expected.items():
            res = predict_lunge_c5_v3(self.kps, view=alias)
            self.assertEqual(res['view'], code)


if __name__ == '__main__':
    unittest.main()

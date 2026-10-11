# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: test_dl1_final.py
Purpose: 다이나믹 런지 c1 공식 최종 확정 모델(dl1_v4 기반 dl1_final.py) 단위 테스트 및 검증
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\models\second_models\dynamic_lunge")

from dl1_final import (
    DL1Classifier,
    get_dl1_classifier,
    predict_dynamic_lunge_c1,
    BUNDLE_PATH,
    VIEW_ALIAS_MAP
)


class TestDL1Final(unittest.TestCase):
    """dl1_final 공식 확정 모델 단위 테스트 스위트"""

    def setUp(self):
        self.clf = get_dl1_classifier()

    def test_01_bundle_file_exists(self):
        """1. 번들 파일 물리적 로드 무결성"""
        self.assertTrue(os.path.exists(BUNDLE_PATH), f"번들 파일 없음: {BUNDLE_PATH}")
        self.assertGreater(os.path.getsize(BUNDLE_PATH), 100000)

    def test_02_model_spec(self):
        """2. 모델 사양 및 피처 차원 검증"""
        self.assertEqual(self.clf.model_name, "dl1_v4")
        self.assertEqual(len(self.clf.feature_names), 30)
        self.assertEqual(len(self.clf.weights), 3)

    def test_03_inference_schema(self):
        """3. 단일 64프레임 시퀀스 추론 및 출력 규격 검증"""
        dummy_kps = np.random.uniform(0.2, 0.8, size=(64, 33, 3)).astype(np.float32)
        res = predict_dynamic_lunge_c1(dummy_kps, view_code='view3')

        required_keys = [
            'model_name', 'exercise', 'defect_code', 'defect_name',
            'view_code', 'view_name', 'state', 'state_str',
            'defect_probability', 'feedback', 'metrics', 'thresholds'
        ]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertIn(res['state'], [0, 1, 2])
        self.assertGreaterEqual(res['defect_probability'], 0.0)
        self.assertLessEqual(res['defect_probability'], 1.0)

    def test_04_all_views_mapping(self):
        """4. 5대 시점 전수 매핑 검증"""
        dummy_kps = np.ones((64, 33, 3), dtype=np.float32) * 0.5
        for v_code, v_name in VIEW_ALIAS_MAP.items():
            res = predict_dynamic_lunge_c1(dummy_kps, view_code=v_code)
            self.assertEqual(res['view_code'], v_code)
            self.assertEqual(res['view_name'], v_name)
            self.assertIn('tau_clean', res['thresholds'])
            self.assertIn('tau_defect', res['thresholds'])
            self.assertLessEqual(res['thresholds']['tau_clean'], res['thresholds']['tau_defect'])


if __name__ == "__main__":
    unittest.main()

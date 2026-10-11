# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_dl4_final.py
Purpose: 다이나믹 런지 c4 최종 공식 배포 모듈(dl4_final.py) 무결성 및 호환성 검증
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.dynamic_lunge.dl4_final import (
    DL4FinalClassifier,
    get_dl4_final_classifier,
    predict_dynamic_lunge_c4_final,
    VIEW_ALIAS_MAP
)

CACHE_PATH = r"D:\Data\피처_데이터\dl4_rtmp_1_dataset_cache.npz"


class TestDL4FinalClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = DL4FinalClassifier()

    def test_01_initialization(self):
        """최종 모델 초기화 및 메타데이터 무결성 검증"""
        self.assertEqual(self.clf.model_name, 'dl4_rtmp_1')
        self.assertEqual(len(self.clf.feature_names), 24)
        self.assertAlmostEqual(self.clf.temperature, 0.9731, places=2)
        self.assertIn('tau_clean', self.clf.global_thresholds)
        self.assertIn('tau_defect', self.clf.global_thresholds)

    def test_02_predict_sequence_2d_and_3d(self):
        """2D(64, 26, 2) 및 3D(64, 26, 3) 시퀀스 추론 인터페이스 검증"""
        # 3D
        dummy_3d = np.random.randn(64, 26, 3).astype(np.float32)
        res_3d = self.clf.predict_sequence(dummy_3d, view_code='시점 B (우측 전방 45°)')
        self.assertEqual(res_3d['defect_code'], 'c4')
        self.assertEqual(res_3d['view_code'], 'view2')
        self.assertIn(res_3d['state'], [0, 1, 2])
        self.assertIsInstance(res_3d['probability'], float)

        # 2D
        dummy_2d = np.random.randn(64, 26, 2).astype(np.float32)
        res_2d = self.clf.predict_sequence(dummy_2d, view_code='view5')
        self.assertEqual(res_2d['view_code'], 'view5')
        self.assertIn(res_2d['state'], [0, 1, 2])

    def test_03_predict_sequence_features_input(self):
        """(1, 24) 또는 (24,) 피처 직접 입력 검증"""
        dummy_feats = np.random.randn(24).astype(np.float32)
        res_1d = self.clf.predict_sequence(dummy_feats, view_code='view3')
        self.assertIn(res_1d['state'], [0, 1, 2])

        dummy_batch = dummy_feats.reshape(1, 24)
        res_2d = self.clf.predict_sequence(dummy_batch, view_code='view1')
        self.assertIn(res_2d['state'], [0, 1, 2])

    def test_04_singleton_and_wrapper(self):
        """싱글톤 게터 및 최종 함수 래퍼 검증"""
        singleton = get_dl4_final_classifier()
        self.assertIs(singleton, get_dl4_final_classifier())

        dummy_kps = np.random.randn(64, 26, 3).astype(np.float32)
        res = predict_dynamic_lunge_c4_final(dummy_kps, view='시점 C (정면 0°)')
        self.assertEqual(res['view_code'], 'view3')
        self.assertIn(res['state'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

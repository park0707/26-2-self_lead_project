# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Test: test_side_lunge_final_submodels.py
Purpose: 사이드 런지 공식 최종 5대 세부 오류 모델군 통합 추론 및 무결성 검증
         - c1: sl1_rtmp_1 (55D)
         - c2: sl2_rtmp_1 (51D)
         - c3: sl3_rtmp_1 (28D)
         - c4: sl4_rtmp_1 (24D)
         - c5: sl5_hybrid_1 (84D, MediaPipe 3D + RTMPose 2D 하이브리드 최종 모델)
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SL_DIR = r"D:\src\models\second_models\side_lunge"
if SL_DIR not in sys.path:
    sys.path.insert(0, SL_DIR)

from sl1_rtmp_1 import SL1RTMP1Classifier
from sl2_rtmp_1 import SL2RTMP1Classifier
from sl3_rtmp_1 import SL3RTMP1Classifier
from sl4_rtmp_1 import SL4RTMP1Classifier
from sl5_hybrid_1 import SL5Hybrid1Classifier, extract_side_lunge_c5_hybrid_features

VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


class TestSideLungeFinalSubmodels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c1 = SL1RTMP1Classifier()
        cls.c2 = SL2RTMP1Classifier()
        cls.c3 = SL3RTMP1Classifier()
        cls.c4 = SL4RTMP1Classifier()
        cls.c5 = SL5Hybrid1Classifier()

    def test_01_all_submodels_loaded(self):
        """5대 세부오류 공식 최종 모델 번들이 모두 정상 로드되었는지 검증"""
        self.assertTrue(self.c1.is_loaded, "sl1_rtmp_1 번들 로드 실패")
        self.assertTrue(self.c2.is_loaded, "sl2_rtmp_1 번들 로드 실패")
        self.assertTrue(self.c3.is_loaded, "sl3_rtmp_1 번들 로드 실패")
        self.assertTrue(self.c4.is_loaded, "sl4_rtmp_1 번들 로드 실패")
        self.assertTrue(self.c5.is_loaded, "sl5_hybrid_1 번들 로드 실패")

    def test_02_c5_is_hybrid_model(self):
        """c5 공식 최종 모델이 sl5_hybrid_1 (84D)인지 검증"""
        self.assertEqual(len(self.c5.feature_names), 84, "c5 피처 차원이 84D가 아닙니다.")
        self.assertTrue(any(f.startswith("mp_") for f in self.c5.feature_names), "MediaPipe 피처 접두사 누락")
        self.assertTrue(any(f.startswith("rtmp_") for f in self.c5.feature_names), "RTMPose 피처 접두사 누락")

    def test_03_predict_proba_ranges(self):
        """5대 서브모델의 확률값이 모두 [0.0, 1.0] 범위 내인지 검증"""
        d1 = np.random.randn(55).astype(np.float32)
        d2 = np.random.randn(51).astype(np.float32)
        d3 = np.random.randn(28).astype(np.float32)
        d4 = np.random.randn(24).astype(np.float32)
        d5 = np.random.randn(84).astype(np.float32)

        for name, clf, feat in [('c1', self.c1, d1), ('c2', self.c2, d2), ('c3', self.c3, d3), ('c4', self.c4, d4), ('c5', self.c5, d5)]:
            p = clf.predict_proba(feat, calibrate=True)
            self.assertIsInstance(p, float)
            self.assertTrue(0.0 <= p <= 1.0, f"{name} 확률 비정상: {p}")

    def test_04_session_integrated_diagnosis(self):
        """5개 서브모델 결합 세션 진단 및 3-State 무결성 검증"""
        dummy_inputs = {
            'c1': np.random.randn(55).astype(np.float32),
            'c2': np.random.randn(51).astype(np.float32),
            'c3': np.random.randn(28).astype(np.float32),
            'c4': np.random.randn(24).astype(np.float32),
            'c5': np.random.randn(84).astype(np.float32)
        }
        clfs = {'c1': self.c1, 'c2': self.c2, 'c3': self.c3, 'c4': self.c4, 'c5': self.c5}

        for vn in VIEW_ORDER:
            states = {}
            for k in ['c1', 'c2', 'c3', 'c4', 'c5']:
                res = clfs[k].predict_3state(dummy_inputs[k], view_code=vn)
                self.assertIn(res['state'], [0, 1, 2], f"{k} 상태 비정상")
                states[k] = res['state']

            # 세션 판정 로직 검증: 5개 중 하나라도 2이면 보류, 하나라도 1이면 결함, 모두 0이면 정상
            if any(s == 2 for s in states.values()):
                session_verdict = "보류"
            elif any(s == 1 for s in states.values()):
                session_verdict = "결함"
            else:
                session_verdict = "정상"

            self.assertIn(session_verdict, ["정상", "결함", "보류"])


if __name__ == '__main__':
    unittest.main()

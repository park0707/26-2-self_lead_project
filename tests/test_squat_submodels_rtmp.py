# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: tests/test_squat_submodels_rtmp.py
Purpose: 바벨 스쿼트 세부 오류 4대 모델 RTMPose 신규 모듈 및 최적화 엔진 단위 테스트 (Harness Engineering)
"""

import os
import sys
import unittest
import numpy as np

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

MODULE_DIR = r"D:\src\models\second_models\babel_squat"
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from c1_model_v12_rtmp import BabelSquatC1ModelV12RTMP, DEFAULT_C1_OPTIMIZATION_PARAMS
from c2_model_v9_rtmp import BabelSquatC2ModelV9RTMP, DEFAULT_C2_OPTIMIZATION_PARAMS
from c3_model_v7_rtmp import BabelSquatC3ModelV7RTMP, DEFAULT_C3_OPTIMIZATION_PARAMS
from c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP, DEFAULT_C4_OPTIMIZATION_PARAMS
from squat_submodels_rtmp import BabelSquatSubmodelsRTMP, VIEW_ALIAS_MAP


class TestSquatSubmodelsRTMP(unittest.TestCase):

    def setUp(self):
        self.c1 = BabelSquatC1ModelV12RTMP()
        self.c2 = BabelSquatC2ModelV9RTMP()
        self.c3 = BabelSquatC3ModelV7RTMP()
        self.c4 = BabelSquatC4ModelV6RTMP()
        self.engine = BabelSquatSubmodelsRTMP()

    def test_dimensions(self):
        """4대 서브모델 피처 차원 정합성 검증"""
        self.assertEqual(self.c1.FEATURE_DIM, 38)
        self.assertEqual(self.c2.FEATURE_DIM, 19)
        self.assertEqual(self.c3.FEATURE_DIM, 22)
        self.assertEqual(self.c4.FEATURE_DIM, 25)

    def test_view_alias_map(self):
        """공식 5대 시점 명칭 및 코드 정합성 검증"""
        expected_views = {'view1', 'view2', 'view3', 'view4', 'view5'}
        self.assertEqual(set(VIEW_ALIAS_MAP.keys()), expected_views)
        self.assertIn("시점 A", VIEW_ALIAS_MAP['view5'])
        self.assertIn("시점 B", VIEW_ALIAS_MAP['view2'])
        self.assertIn("시점 C", VIEW_ALIAS_MAP['view3'])
        self.assertIn("시점 D", VIEW_ALIAS_MAP['view4'])
        self.assertIn("시점 E", VIEW_ALIAS_MAP['view1'])

    def test_temperature_calibration(self):
        """온도 스케일링 단조성 및 유효 범위 [0, 1] 검증"""
        for p_in in [0.05, 0.20, 0.50, 0.80, 0.95]:
            p_out = self.c1.calibrate_probability(p_in, view_name='view3')
            self.assertGreaterEqual(p_out, 0.0)
            self.assertLessEqual(p_out, 1.0)

        # 0.50은 T에 관계없이 0.50 보존
        p_half = self.c1.calibrate_probability(0.50, view_name='view3')
        self.assertAlmostEqual(p_half, 0.50, places=3)

    def test_c2_nan_robustness(self):
        """c2 후방 시점 안면 결측(NaN) 방어 검증"""
        x_with_nans = np.full((1, 19), np.nan, dtype=np.float32)
        x_with_nans[0, 5:] = 1.0  # 비안면 피처 일부 채움
        prob = self.c2.predict_proba(x_with_nans, view_name='view5', calibrate=True)
        self.assertFalse(np.isnan(prob[0]))
        self.assertGreaterEqual(prob[0], 0.0)
        self.assertLessEqual(prob[0], 1.0)

    def test_3state_classification(self):
        """3-State 의사결정 상태 코드 (0: CLEAN, 1: DEFECT, 2: RETAKE) 유효성 검증"""
        # 아주 낮은 확률 -> CLEAN
        s_clean, _ = self.c3.classify_3state(0.10, view_name='view3')
        self.assertEqual(s_clean, 0)

        # 아주 높은 확률 -> DEFECT
        s_defect, _ = self.c3.classify_3state(0.90, view_name='view3')
        self.assertEqual(s_defect, 1)

        # 불확실성 중간값 -> RETAKE
        s_retake, _ = self.c3.classify_3state(0.50, view_name='view3')
        self.assertEqual(s_retake, 2)

    def test_integrated_engine_inference(self):
        """통합 배포 엔진 종합 진단 파이프라인 검증"""
        feats_dict = {
            'c1': np.zeros(38, dtype=np.float32),
            'c2': np.zeros(19, dtype=np.float32),
            'c3': np.zeros(22, dtype=np.float32),
            'c4': np.zeros(25, dtype=np.float32)
        }
        res = self.engine.predict_features_dict(feats_dict, view_name='view3')
        self.assertIn('session_state', res)
        self.assertIn('session_decision', res)
        self.assertIn('p_clean_fusion', res)
        self.assertIn('submodels', res)
        self.assertEqual(len(res['submodels']), 4)

    def test_canonical_package_exports_and_default_loading(self):
        """패키지 진입점(__init__.py) 표준 별칭 및 기본 가중치 자동 로딩 검증"""
        from src.models.second_models.babel_squat import (
            BabelSquatC1Model,
            BabelSquatC2Model,
            BabelSquatC3Model,
            BabelSquatC4Model,
            BabelSquatIntegratedEngine,
            BabelSquatEngine
        )
        c1 = BabelSquatC1Model()
        c2 = BabelSquatC2Model()
        c3 = BabelSquatC3Model()
        c4 = BabelSquatC4Model()
        eng = BabelSquatIntegratedEngine()

        self.assertTrue(c1.is_loaded)
        self.assertTrue(c2.is_loaded)
        self.assertTrue(c3.is_loaded)
        self.assertTrue(c4.is_loaded)
        self.assertTrue(eng.is_loaded)

    def test_predict_sequence_and_flexible_interface(self):
        """64프레임 원시 골격 시퀀스(64, 17, 3) 대상 종단간 통합 추론 검증"""
        dummy_seq = np.random.randn(64, 17, 3).astype(np.float32)
        res = self.engine.predict(dummy_seq, view_name='view3')
        self.assertIn('session_state', res)
        self.assertIn('submodels', res)
        self.assertIn('c1', res['submodels'])
        self.assertIn('c2', res['submodels'])
        self.assertIn('c3', res['submodels'])
        self.assertIn('c4', res['submodels'])



if __name__ == '__main__':
    unittest.main()

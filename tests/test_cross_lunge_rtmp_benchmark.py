# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: tests/test_cross_lunge_rtmp_benchmark.py
Purpose: 크로스 런지 세부 오류 3대 모델 RTMPose 벤치마크 및 피처 파이프라인 무결성 검증 (Harness Engineering)
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

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODULE_DIR = r"D:\src\models\second_models\cross_lunge"
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from run_cross_lunge_mediapipe_vs_rtmpose_benchmark import (
    halpe26_to_mediapipe33,
    evaluate_metrics,
    VIEW_MAP,
    VIEW_ALIAS_MAP,
    RETAINED_C2_INDICES
)
from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)
from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    extract_cross_lunge_c2_v7_features_51d,
    safe_savgol_filter_kps
)
from src.models.second_models.cross_lunge.cross_lunge_c2_v2_features import (
    compute_v2_cross_lunge_specialized_features_8d
)
from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial
from src.features.cross_lunge_c3_features import (
    extract_cross_lunge_c3_features_22d,
    FEATURE_NAMES_22D
)


class TestCrossLungeRTMPBenchmark(unittest.TestCase):

    def test_halpe26_to_mediapipe33_mapping(self):
        """Halpe26 -> MediaPipe33 관절 매핑 무결성 검증"""
        dummy_halpe = np.zeros((64, 26, 3), dtype=np.float32)
        # 11: left_hip, 12: right_hip, 13: left_knee, 14: right_knee
        dummy_halpe[:, 11, :] = [1.0, 2.0, 3.0]
        dummy_halpe[:, 12, :] = [4.0, 5.0, 6.0]
        dummy_halpe[:, 13, :] = [7.0, 8.0, 9.0]
        dummy_halpe[:, 14, :] = [10.0, 11.0, 12.0]

        mp33 = halpe26_to_mediapipe33(dummy_halpe)
        self.assertEqual(mp33.shape, (64, 33, 3))
        # MP33: 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee
        np.testing.assert_array_almost_equal(mp33[0, 23, :], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(mp33[0, 24, :], [4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(mp33[0, 25, :], [7.0, 8.0, 9.0])
        np.testing.assert_array_almost_equal(mp33[0, 26, :], [10.0, 11.0, 12.0])

    def test_submodel_feature_dimensions(self):
        """크로스 런지 3대 서브모델 피처 차원(c1: 55D, c2: 48D, c3: 22D) 정합성 검증"""
        self.assertEqual(len(SIDE_LUNGE_C1_FEATURE_NAMES_V11), 55)
        self.assertEqual(len(RETAINED_C2_INDICES), 48)
        self.assertEqual(len(FEATURE_NAMES_22D), 22)

    def test_view_alias_taxonomy(self):
        """AGENTS.md 공식 5대 시점 Taxonomy 매핑 검증"""
        expected_views = {'view1', 'view2', 'view3', 'view4', 'view5'}
        self.assertEqual(set(VIEW_ALIAS_MAP.keys()), expected_views)
        self.assertIn("시점 A", VIEW_ALIAS_MAP['view5'])
        self.assertIn("시점 B", VIEW_ALIAS_MAP['view2'])
        self.assertIn("시점 C", VIEW_ALIAS_MAP['view3'])
        self.assertIn("시점 D", VIEW_ALIAS_MAP['view4'])
        self.assertIn("시점 E", VIEW_ALIAS_MAP['view1'])

    def test_metrics_evaluation_logic(self):
        """2-State 및 3-State 선택적 판정 지표 계산 검증"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])
        
        # 3-State: [0.35, 0.65] 모호 구간 -> 0.45, 0.55가 보류됨 (2건 / 6건 = 33.33%)
        metrics = evaluate_metrics(y_true, y_prob, threshold=0.50, tau_clean=0.35, tau_defect=0.65)
        
        self.assertEqual(metrics['coverage'], round(4 / 6 * 100.0, 2))
        self.assertEqual(metrics['retake'], round(2 / 6 * 100.0, 2))
        self.assertGreater(metrics['sel_f1'], 0.0)
        self.assertGreater(metrics['forced_f1'], 0.0)


if __name__ == "__main__":
    unittest.main()

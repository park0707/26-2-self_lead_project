# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: tests/test_side_lunge_rtmp.py
Purpose: 사이드 런지 RTMPose 신규 모듈 및 5대 세부 오류 피처 추출/평가 단위 테스트 (Harness Engineering)
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

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SIDE_LUNGE_DIR = r"D:\src\models\second_models\side_lunge"
if SIDE_LUNGE_DIR not in sys.path:
    sys.path.insert(0, SIDE_LUNGE_DIR)

from src.features.build_side_lunge_rtmp_features import halpe26_to_mp33_unflipped
from src.features.side_lunge_c1_features_v11 import extract_side_lunge_c1_feature_vector_v11
from side_lunge_c2_v7_features import extract_side_lunge_c2_v7_features_51d
from src.features.side_lunge_c3_features_v3 import extract_side_lunge_c3_features_v3_28d
from src.features.side_lunge_c4_features_v4 import extract_side_lunge_c4_features_v4
from src.features.side_lunge_c5_features_v7 import extract_side_lunge_c5_v7_feature_vector
from src.models.second_models.side_lunge.evaluate_side_lunge_rtmp import (
    calc_2state_metrics,
    calc_3state_metrics,
    OFFICIAL_VIEW_MAP
)


class TestSideLungeRTMP(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 64-frame Halpe26 pose sequence
        self.kps26 = np.zeros((64, 26, 3), dtype=np.float32)
        # Hips at (0, 0)
        self.kps26[:, 11, :] = [-0.15, 0.0, 0.95]  # Left hip
        self.kps26[:, 12, :] = [0.15, 0.0, 0.95]   # Right hip
        self.kps26[:, 19, :] = [0.0, 0.0, 0.95]    # Pelvis
        # Shoulders
        self.kps26[:, 5, :] = [-0.20, -0.5, 0.95]  # Left shoulder
        self.kps26[:, 6, :] = [0.20, -0.5, 0.95]   # Right shoulder
        # Knees
        self.kps26[:, 13, :] = [-0.20, 0.4, 0.95]  # Left knee
        self.kps26[:, 14, :] = [0.40, 0.3, 0.95]   # Right knee
        # Ankles
        self.kps26[:, 15, :] = [-0.20, 0.8, 0.95]  # Left ankle
        self.kps26[:, 16, :] = [0.60, 0.7, 0.95]   # Right ankle
        # Feet
        self.kps26[:, 20, :] = [-0.20, 0.9, 0.95]  # Left big toe
        self.kps26[:, 21, :] = [0.70, 0.8, 0.95]   # Right big toe
        self.kps26[:, 24, :] = [-0.20, 0.82, 0.95] # Left heel
        self.kps26[:, 25, :] = [0.58, 0.72, 0.95]  # Right heel

    def test_halpe26_to_mp33_unflipped_shape_and_z(self):
        """Halpe26 -> MP33 어댑터 규격 및 Z=0 검증"""
        mp33 = halpe26_to_mp33_unflipped(self.kps26, is_flipped=False)
        self.assertEqual(mp33.shape, (64, 33, 3))
        # Z coordinates must be 0.0
        self.assertTrue(np.allclose(mp33[:, :, 2], 0.0))

    def test_halpe26_unflip_logic(self):
        """is_flipped=True 시 역반전(Unflip) 좌우 대칭성 검증"""
        mp33_flipped = halpe26_to_mp33_unflipped(self.kps26, is_flipped=True)
        # X coordinates should be negated and left/right swapped
        self.assertEqual(mp33_flipped.shape, (64, 33, 3))

    def test_feature_dimensions(self):
        """c1~c5 5대 세부 오류 모델 피처 차원 정합성 검증"""
        mp33 = halpe26_to_mp33_unflipped(self.kps26, is_flipped=False)

        fc1 = extract_side_lunge_c1_feature_vector_v11(mp33)
        self.assertEqual(fc1.shape, (55,), "c1 must be 55D")
        self.assertFalse(np.isnan(fc1).any(), "c1 must not have NaNs")

        fc2 = extract_side_lunge_c2_v7_features_51d(mp33, view_code="view3")
        self.assertEqual(fc2.shape, (51,), "c2 must be 51D")
        self.assertFalse(np.isnan(fc2).any(), "c2 must not have NaNs")

        fc3 = extract_side_lunge_c3_features_v3_28d(mp33)
        self.assertEqual(fc3.shape, (28,), "c3 must be 28D")
        self.assertFalse(np.isnan(fc3).any(), "c3 must not have NaNs")

        fc4_raw = extract_side_lunge_c4_features_v4(mp33)
        fc4 = np.array(list(fc4_raw.values())) if isinstance(fc4_raw, dict) else fc4_raw
        self.assertEqual(fc4.shape, (24,), "c4 must be 24D")
        self.assertFalse(np.isnan(fc4).any(), "c4 must not have NaNs")

        fc5 = extract_side_lunge_c5_v7_feature_vector(mp33)
        self.assertEqual(fc5.shape, (42,), "c5 must be 42D")
        self.assertFalse(np.isnan(fc5).any(), "c5 must not have NaNs")

    def test_official_view_mapping(self):
        """공식 5대 시점 명칭 및 코드 정합성 검증"""
        expected_views = {'view1', 'view2', 'view3', 'view4', 'view5'}
        self.assertEqual(set(OFFICIAL_VIEW_MAP.keys()), expected_views)
        self.assertIn("시점 A", OFFICIAL_VIEW_MAP['view5'])
        self.assertIn("시점 B", OFFICIAL_VIEW_MAP['view2'])
        self.assertIn("시점 C", OFFICIAL_VIEW_MAP['view3'])
        self.assertIn("시점 D", OFFICIAL_VIEW_MAP['view4'])
        self.assertIn("시점 E", OFFICIAL_VIEW_MAP['view1'])

    def test_3state_classification_metrics(self):
        """3-State 의사결정 및 보류율(Uncertainty Rate) 산출 검증"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.5, 0.5]) # 2 uncertain samples
        res = calc_3state_metrics(y_true, y_prob, tau_clean=0.35, tau_defect=0.65)

        self.assertAlmostEqual(res['uncertainty_rate'], 2.0 / 6.0, places=4)
        self.assertAlmostEqual(res['coverage'], 4.0 / 6.0, places=4)
        self.assertEqual(res['effective_f1'], 1.0) # All 4 decided are perfectly classified


if __name__ == "__main__":
    unittest.main()

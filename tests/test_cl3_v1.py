# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for Cross Lunge c3 (cl3_v1) Feature & Model Pipeline
Harness Engineering Verification Suite
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.cross_lunge_c3_features import (
    extract_cross_lunge_c3_features_22d,
    FEATURE_NAMES_22D
)


class TestCrossLungeC3Pipeline(unittest.TestCase):
    def setUp(self):
        # 합성 64프레임 33관절 3D 시퀀스 (64, 33, 3)
        self.kps_seq_33 = np.random.randn(64, 33, 3).astype(np.float32)
        # MediaPipe 일반적 관절 깊이/스케일 모사
        self.kps_seq_33[:, 11, :] = [-0.2, -0.5, 0.0]  # Left shoulder
        self.kps_seq_33[:, 12, :] = [0.2, -0.5, 0.0]   # Right shoulder
        self.kps_seq_33[:, 23, :] = [-0.15, 0.0, 0.0]  # Left hip
        self.kps_seq_33[:, 24, :] = [0.15, 0.0, 0.0]   # Right hip
        self.kps_seq_33[:, 25, :] = [-0.15, 0.4, 0.1]  # Left knee
        self.kps_seq_33[:, 26, :] = [0.15, 0.4, -0.2]  # Right knee
        self.kps_seq_33[:, 27, :] = [-0.15, 0.8, 0.0]  # Left ankle
        self.kps_seq_33[:, 28, :] = [0.15, 0.8, -0.3]  # Right ankle

    def test_feature_dimension_and_names(self):
        """피처 벡터 차원 및 피처 이름 개수 일치성 검증"""
        feat_vec = extract_cross_lunge_c3_features_22d(self.kps_seq_33)
        self.assertEqual(len(feat_vec), 22)
        self.assertEqual(len(FEATURE_NAMES_22D), 22)
        self.assertEqual(feat_vec.dtype, np.float32)

    def test_no_nan_or_inf_in_features(self):
        """피처 벡터에 NaN 또는 Inf가 존재하지 않는지 엄격 검증"""
        feat_vec = extract_cross_lunge_c3_features_22d(self.kps_seq_33)
        self.assertFalse(np.isnan(feat_vec).any(), "피처에 NaN이 검출되었습니다.")
        self.assertFalse(np.isinf(feat_vec).any(), "피처에 Inf가 검출되었습니다.")

    def test_zero_sequence_robustness(self):
        """모든 관절이 0인 극단적 엣지 케이스 안정성 검증"""
        zero_seq = np.zeros((64, 33, 3), dtype=np.float32)
        feat_vec = extract_cross_lunge_c3_features_22d(zero_seq)
        self.assertEqual(len(feat_vec), 22)
        self.assertFalse(np.isnan(feat_vec).any())
        self.assertFalse(np.isinf(feat_vec).any())

    def test_15_joints_compatibility(self):
        """15관절 입력 포맷 하위 호환성 검증"""
        seq_15 = np.random.randn(64, 15, 3).astype(np.float32)
        feat_vec = extract_cross_lunge_c3_features_22d(seq_15)
        self.assertEqual(len(feat_vec), 22)
        self.assertFalse(np.isnan(feat_vec).any())


if __name__ == '__main__':
    unittest.main()

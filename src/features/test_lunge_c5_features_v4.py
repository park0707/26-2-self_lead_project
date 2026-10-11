# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_lunge_c5_features_v4.py
Verification of lunge_c5_features_v4.py 32D domain features
"""

import sys
import unittest
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_v4 import (
    extract_lunge_c5_v4_features,
    extract_lunge_c5_v4_feature_vector,
    LUNGE_C5_V4_FEATURE_NAMES
)


class TestLungeC5FeaturesV4(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic standard 64-frame sequence (64, 33, 3)
        self.kps = np.random.randn(64, 33, 3).astype(np.float32)
        # Ensure realistic y coordinates (hips above knees, knees above ankles)
        self.kps[:, 23:25, 1] = 0.5   # Hips
        self.kps[:, 25:27, 1] = 0.8   # Knees
        self.kps[:, 27:29, 1] = 1.1   # Ankles
        self.kps[:, 11:13, 1] = 0.1   # Shoulders

    def test_feature_count(self):
        self.assertEqual(len(LUNGE_C5_V4_FEATURE_NAMES), 32)

    def test_feature_dict_extraction(self):
        feats = extract_lunge_c5_v4_features(self.kps)
        self.assertEqual(len(feats), 32)
        for name in LUNGE_C5_V4_FEATURE_NAMES:
            self.assertIn(name, feats)
            val = feats[name]
            self.assertFalse(np.isnan(val), f"Feature {name} is NaN")
            self.assertFalse(np.isinf(val), f"Feature {name} is Inf")

    def test_feature_vector_extraction(self):
        vec = extract_lunge_c5_v4_feature_vector(self.kps)
        self.assertEqual(vec.shape, (32,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertFalse(np.any(np.isnan(vec)))
        self.assertFalse(np.any(np.isinf(vec)))

    def test_v4_specialized_features(self):
        feats = extract_lunge_c5_v4_features(self.kps)
        self.assertIn("lunge_femur_to_torso_ratio", feats)
        self.assertIn("lunge_pelvis_spine_decoupling_angle", feats)
        self.assertIn("lunge_bidirectional_hyperext_asym", feats)
        self.assertIn("lunge_early_ascent_burst_jerk", feats)
        self.assertGreater(feats["lunge_femur_to_torso_ratio"], 0.0)
        self.assertGreaterEqual(feats["lunge_bidirectional_hyperext_asym"], 0.0)


if __name__ == '__main__':
    unittest.main()

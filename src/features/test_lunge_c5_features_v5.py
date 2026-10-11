# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_lunge_c5_features_v5.py
Verification of lunge_c5_features_v5.py 36D SOTA domain features
"""

import sys
import unittest
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_v5 import (
    extract_lunge_c5_v5_features,
    extract_lunge_c5_v5_feature_vector,
    LUNGE_C5_V5_FEATURE_NAMES
)


class TestLungeC5FeaturesV5(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic standard 64-frame sequence (64, 33, 3)
        self.kps = np.random.randn(64, 33, 3).astype(np.float32) * 0.05
        # Ensure realistic positions
        self.kps[:, 23:25, 1] = 0.5   # Hips
        self.kps[:, 25:27, 1] = 0.8   # Knees
        self.kps[:, 27:29, 1] = 1.1   # Ankles
        self.kps[:, 11:13, 1] = 0.1   # Shoulders
        self.kps[:, 7:9, 1] = -0.1    # Ears / Head

    def test_feature_count(self):
        self.assertEqual(len(LUNGE_C5_V5_FEATURE_NAMES), 36)

    def test_feature_dict_extraction(self):
        feats = extract_lunge_c5_v5_features(self.kps)
        self.assertEqual(len(feats), 36)
        for name in LUNGE_C5_V5_FEATURE_NAMES:
            self.assertIn(name, feats)
            val = feats[name]
            self.assertFalse(np.isnan(val), f"Feature {name} is NaN")
            self.assertFalse(np.isinf(val), f"Feature {name} is Inf")

    def test_feature_vector_extraction(self):
        vec = extract_lunge_c5_v5_feature_vector(self.kps)
        self.assertEqual(vec.shape, (36,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertFalse(np.any(np.isnan(vec)))
        self.assertFalse(np.any(np.isinf(vec)))

    def test_v5_sota_features(self):
        feats = extract_lunge_c5_v5_features(self.kps)
        self.assertIn("lunge_trunk_thigh_crp_variability", feats)
        self.assertIn("lunge_torso_spectral_energy_ratio", feats)
        self.assertIn("lunge_thoracic_curvature_deviation", feats)
        self.assertIn("lunge_bottom_energy_flux_ratio", feats)
        self.assertGreaterEqual(feats["lunge_trunk_thigh_crp_variability"], 0.0)
        self.assertGreaterEqual(feats["lunge_torso_spectral_energy_ratio"], 0.0)
        self.assertGreaterEqual(feats["lunge_bottom_energy_flux_ratio"], 0.0)


if __name__ == '__main__':
    unittest.main()

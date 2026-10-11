# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Integration Tests for SD3V2Classifier (sd3_v2)
"""

import unittest
import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd3_v2_classifier import SD3V2Classifier
from stiff_d1_features import VIEW_ALIAS_MAP


class TestSD3V2Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = SD3V2Classifier()

    def test_classifier_metadata(self):
        self.assertEqual(self.clf.defect_code, 'sd3_v2')
        self.assertEqual(self.clf.model_name, 'sd3_v2')
        self.assertAlmostEqual(self.clf.tau_clean, 0.36, places=2)
        self.assertAlmostEqual(self.clf.tau_defect, 0.50, places=2)
        self.assertEqual(len(self.clf.feature_names), 32)
        self.assertEqual(len(self.clf.pruned_features), 6)

    def test_predict_proba_shape_3d(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        p = self.clf.predict_proba(x, view_name="view4")
        self.assertIsInstance(p, float)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_predict_proba_shape_2d(self):
        x99 = np.random.randn(64, 99).astype(np.float32)
        p99 = self.clf.predict_proba(x99, view_name="view2")
        self.assertIsInstance(p99, float)
        self.assertTrue(0.0 <= p99 <= 1.0)

        x98 = np.random.randn(64, 98).astype(np.float32)
        p98 = self.clf.predict_proba(x98, view_name="view2")
        self.assertIsInstance(p98, float)
        self.assertTrue(0.0 <= p98 <= 1.0)

    def test_diagnose_schema(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        diag = self.clf.diagnose(x, view_name="view4")
        
        required_keys = [
            'defect_code', 'defect_name', 'probability', 'margin',
            'state_2state', 'state_3state', 'state_label', 'feedback',
            'view_name', 'view_alias', 'features_dim', 'pruned_features_count', 'thresholds'
        ]
        for k in required_keys:
            self.assertIn(k, diag)

        self.assertIn(diag['state_2state'], [0, 1])
        self.assertIn(diag['state_3state'], [0, 1, 2])
        self.assertEqual(diag['features_dim'], 32)
        self.assertEqual(diag['pruned_features_count'], 6)
        self.assertEqual(diag['defect_code'], 'sd3_v2')

    def test_all_5_views(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        for v_code, v_alias in VIEW_ALIAS_MAP.items():
            diag = self.clf.diagnose(x, view_name=v_code)
            self.assertEqual(diag['view_name'], v_code)
            self.assertEqual(diag['view_alias'], v_alias)
            self.assertTrue(0.0 <= diag['probability'] <= 1.0)

    def test_nan_inf_allzero_safety(self):
        # All zeros
        x_zero = np.zeros((64, 33, 3), dtype=np.float32)
        diag_z = self.clf.diagnose(x_zero, view_name="view3")
        self.assertFalse(np.isnan(diag_z['probability']))

        # NaN injection
        x_nan = np.random.randn(64, 33, 3).astype(np.float32)
        x_nan[10:15, :, :] = np.nan
        diag_nan = self.clf.diagnose(x_nan, view_name="view1")
        self.assertFalse(np.isnan(diag_nan['probability']))

        # Inf injection
        x_inf = np.random.randn(64, 33, 3).astype(np.float32)
        x_inf[20:25, :, :] = np.inf
        diag_inf = self.clf.diagnose(x_inf, view_name="view5")
        self.assertFalse(np.isnan(diag_inf['probability']))


if __name__ == '__main__':
    unittest.main()

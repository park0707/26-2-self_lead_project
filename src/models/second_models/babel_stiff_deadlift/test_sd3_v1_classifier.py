# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for SD3V1Classifier (sd3_v1)
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

from sd3_v1_classifier import SD3V1Classifier


class TestSD3V1Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = SD3V1Classifier()

    def test_classifier_metadata(self):
        self.assertEqual(self.clf.defect_code, 'sd3_v1')
        self.assertEqual(self.clf.model_name, 'sd3_v1')
        self.assertAlmostEqual(self.clf.tau_clean, 0.36, places=2)
        self.assertAlmostEqual(self.clf.tau_defect, 0.50, places=2)

    def test_predict_proba_shape_3d(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        p = self.clf.predict_proba(x, view_name="view4")
        self.assertIsInstance(p, float)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_predict_proba_shape_2d(self):
        x = np.random.randn(64, 99).astype(np.float32)
        p = self.clf.predict_proba(x, view_name="view2")
        self.assertIsInstance(p, float)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_diagnose_schema(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        diag = self.clf.diagnose(x, view_name="view4")
        
        required_keys = [
            'defect_code', 'defect_name', 'probability', 'margin',
            'state_2state', 'state_3state', 'state_label', 'feedback',
            'view_name', 'view_alias', 'features_dim', 'thresholds'
        ]
        for k in required_keys:
            self.assertIn(k, diag)

        self.assertIn(diag['state_2state'], [0, 1])
        self.assertIn(diag['state_3state'], [0, 1, 2])
        self.assertEqual(diag['features_dim'], 38)
        self.assertEqual(diag['defect_code'], 'sd3_v1')

    def test_all_five_views(self):
        x = np.random.randn(64, 33, 3).astype(np.float32)
        views = ['view1', 'view2', 'view3', 'view4', 'view5']
        for v in views:
            diag = self.clf.diagnose(x, view_name=v)
            self.assertEqual(diag['view_name'], v)
            self.assertTrue(len(diag['view_alias']) > 0)


if __name__ == '__main__':
    unittest.main()

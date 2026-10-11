# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Test Suite: test_bl3_v3.py
Purpose: 바벨 런지 c3 v3 11D 직교 정예 모델(bl3_v3.py) 하네스 엔지니어링 단위 테스트
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v3 import extract_lunge_c3_11d_features_v3, FEATURE_NAMES_11D
from src.models.sub_models.bl3_v3 import BL3V3Classifier

class TestBL3V3Classifier(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.clf = BL3V3Classifier()
        cls.dummy_single = np.random.randn(64, 15, 2).astype(np.float32)
        cls.dummy_batch = np.random.randn(5, 64, 15, 2).astype(np.float32)
        
    def test_01_feature_dimension_and_names(self):
        """11D 피처 추출 규격 및 결측치 0% 검증"""
        feat = extract_lunge_c3_11d_features_v3(self.dummy_single)
        self.assertEqual(feat.shape, (11,), f"Feature dimension mismatch: {feat.shape} != (11,)")
        self.assertEqual(len(FEATURE_NAMES_11D), 11, f"Feature names count mismatch: {len(FEATURE_NAMES_11D)}")
        self.assertFalse(np.isnan(feat).any(), "NaN detected in 11D features!")
        self.assertFalse(np.isinf(feat).any(), "Inf detected in 11D features!")
        
    def test_02_batch_feature_extraction(self):
        """배치 (N, 64, 15, 2) 피처 추출 검증"""
        batch_feats = extract_lunge_c3_11d_features_v3(self.dummy_batch)
        self.assertEqual(batch_feats.shape, (5, 11), f"Batch shape mismatch: {batch_feats.shape}")
        self.assertFalse(np.isnan(batch_feats).any(), "NaN in batch features!")
        
    def test_03_classifier_init_and_thresholds(self):
        """v3 분류기 초기화 및 최적 3-State 임계값 유효성 검증"""
        self.assertIsNotNone(self.clf.cb_model, "CatBoost model failed to initialize or load.")
        self.assertGreater(self.clf.TAU_HIGH, self.clf.TAU_LOW, "TAU_HIGH must be greater than TAU_LOW.")
        self.assertTrue(0.0 <= self.clf.TAU_LOW <= 1.0, "TAU_LOW out of bounds [0, 1].")
        self.assertTrue(0.0 <= self.clf.TAU_HIGH <= 1.0, "TAU_HIGH out of bounds [0, 1].")
        self.assertTrue(self.clf.TAU_LOW <= self.clf.OPTIMAL_TAU <= self.clf.TAU_HIGH, "OPTIMAL_TAU must lie between TAU_LOW and TAU_HIGH.")
        
    def test_04_predict_single_contract(self):
        """단일 추론 인터페이스 규격 및 키 무결성 검증"""
        res = self.clf.predict_single(self.dummy_single)
        required_keys = [
            'exercise', 'defect_name', 'model_version', 'status', 'decision',
            'defect_probability', 'optimal_threshold', 'certainty_margin',
            'biomechanics', 'coaching_message'
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Missing key in prediction result: {key}")
            
        self.assertIn(res['status'], ["NORMAL", "DEFECT", "UNDETERMINED"])
        self.assertIn(res['decision'], [0, 1, -1])
        self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
        self.assertEqual(res['model_version'], "v3.0")
        
        bio = res['biomechanics']
        for bkey in FEATURE_NAMES_11D:
            self.assertIn(bkey, bio, f"Missing biomechanics metric: {bkey}")
            
    def test_05_inference_latency(self):
        """단일 추론 레이턴시 20ms 이내 초고속성 검증"""
        import time
        t0 = time.time()
        for _ in range(50):
            _ = self.clf.predict_single(self.dummy_single)
        avg_ms = ((time.time() - t0) / 50.0) * 1000.0
        print(f"\n[+] Measured BL3_v3 Single Inference Latency: {avg_ms:.3f} ms")
        self.assertLess(avg_ms, 20.0, f"Latency exceeded 20ms threshold: {avg_ms:.3f} ms")

if __name__ == '__main__':
    unittest.main()

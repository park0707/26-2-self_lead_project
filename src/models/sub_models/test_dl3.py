# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Integration Test Suite: test_dl3.py
Purpose: 다이나믹 런지 세 번째 결함 '뒤다리 무릎 각도 90도 (dl3)' 프로덕션 파이프라인 무결성 검증
"""

import unittest
import numpy as np
import time
import os
import sys

sys.path.insert(0, 'd:/')
from src.features.lunge_dl3_features import (
    extract_lunge_dl3_14d_features,
    FEATURE_NAMES_14D
)
from src.models.sub_models.dl3 import DL3Classifier

class TestDL3Pipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.classifier = DL3Classifier()
        cls.dummy_kps = np.random.uniform(0.2, 0.8, (64, 15, 2)).astype(np.float32)
        
    def test_feature_names_count(self):
        """14D 정예 기구학 피처 명칭 목록이 정확히 14개인지 검증"""
        self.assertEqual(len(FEATURE_NAMES_14D), 14)
        self.assertIn('knee_hysteresis_area', FEATURE_NAMES_14D)
        self.assertIn('femur_tibia_proj_ratio', FEATURE_NAMES_14D)
        self.assertIn('crp_hip_knee_mean', FEATURE_NAMES_14D)
        
    def test_feature_extraction_contract(self):
        """14D 피처 추출 함수의 출력 규격 및 결측치(NaN/Inf) 무결성 검증"""
        feats = extract_lunge_dl3_14d_features(self.dummy_kps)
        self.assertEqual(feats.shape, (14,))
        self.assertFalse(np.isnan(feats).any(), "NaN detected in 14D features!")
        self.assertFalse(np.isinf(feats).any(), "Inf detected in 14D features!")
        
    def test_classifier_initialization(self):
        """DL3Classifier 번들 로드 및 임계값 정합성 검증"""
        self.assertIsNotNone(self.classifier.cb_model)
        self.assertTrue(0.0 < self.classifier.TAU_LOW < self.classifier.OPTIMAL_TAU < self.classifier.TAU_HIGH < 1.0)
        self.assertEqual(self.classifier.MODEL_CODE, "dl3")
        self.assertEqual(self.classifier.VERSION, "v1.0")
        
    def test_predict_single_contract(self):
        """단일 시퀀스 추론 출력 딕셔너리 규격 및 3-State 결정 유효성 검증"""
        res = self.classifier.predict_single(self.dummy_kps)
        required_keys = [
            'exercise', 'defect_code', 'defect_name', 'state',
            'state_label', 'defect_probability', 'confidence',
            'kinematics', 'coaching_feedback'
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing key in prediction: {k}")
            
        self.assertIn(res['state'], [-1, 0, 1])
        self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
        self.assertTrue(0.0 <= res['confidence'] <= 1.0)
        self.assertIn('rear_knee_angle_bottom_deg', res['kinematics'])
        self.assertTrue(len(res['coaching_feedback']) > 0)
        
    def test_inference_latency(self):
        """실시간 서비스 30fps(33ms)를 충족하는 초저지연(10ms 이하) 추론 검증"""
        t0 = time.time()
        n_iters = 100
        for _ in range(n_iters):
            _ = self.classifier.predict_single(self.dummy_kps)
        avg_lat_ms = (time.time() - t0) / n_iters * 1000.0
        print(f"\n[+] Verified Average DL3 Inference Latency: {avg_lat_ms:.3f} ms / sequence (Threshold: < 10.0 ms)")
        self.assertLess(avg_lat_ms, 10.0)

if __name__ == '__main__':
    unittest.main()

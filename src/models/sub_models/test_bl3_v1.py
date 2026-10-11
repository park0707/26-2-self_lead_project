# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_bl3_v1.py
Purpose: 바벨 런지 c3 세부 오류 분류기 (BL3V1Classifier) 하네스 엔지니어링 단위 테스트
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.sub_models.bl3_v1 import BL3V1Classifier

class TestBL3V1Classifier(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.classifier = BL3V1Classifier()
        
    def test_model_initialization(self):
        """1. 모델 로드 및 버전 무결성 검증"""
        self.assertIsNotNone(self.classifier.model, "Classifier model must be loaded.")
        self.assertEqual(self.classifier.VERSION, "v1.0")
        self.assertEqual(self.classifier.DEFECT_NAME, "뒤다리 무릎 각도 90도")
        
    def test_feature_extraction_shape(self):
        """2. 14D 피처 규격 검증"""
        dummy_seq = np.random.randn(64, 15, 2).astype(np.float32)
        feats = self.classifier.extract_features(dummy_seq)
        self.assertEqual(feats.shape, (14,), "Features shape must be (14,).")
        self.assertFalse(np.isnan(feats).any(), "Extracted features must not contain NaNs.")
        
    def test_single_prediction_structure(self):
        """3. 단일 추론 출력 인터페이스 및 키 무결성 검증"""
        dummy_seq = np.random.randn(64, 15, 2).astype(np.float32)
        res = self.classifier.predict_single(dummy_seq)
        
        required_keys = [
            'exercise', 'defect_name', 'model_version', 'status',
            'decision', 'defect_probability', 'optimal_threshold',
            'certainty_margin', 'biomechanics', 'coaching_message'
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Result must contain key '{k}'.")
            
        self.assertIn(res['status'], ['NORMAL', 'DEFECT', 'UNDETERMINED'])
        self.assertIn(res['decision'], [0, 1, -1])
        self.assertGreaterEqual(res['defect_probability'], 0.0)
        self.assertLessEqual(res['defect_probability'], 1.0)
        self.assertGreaterEqual(res['certainty_margin'], 0.0)
        
    def test_zero_tensor_defense(self):
        """4. 제로 텐서 및 극단치 방어 무결성 검증"""
        zero_seq = np.zeros((64, 15, 2), dtype=np.float32)
        res = self.classifier.predict_single(zero_seq)
        self.assertIn(res['status'], ['NORMAL', 'DEFECT', 'UNDETERMINED'])
        self.assertFalse(np.isnan(res['defect_probability']))
        
    def test_biomechanics_keys(self):
        """5. 생체역학 지표 딕셔너리 무결성 검증"""
        dummy_seq = np.random.randn(64, 15, 2).astype(np.float32)
        res = self.classifier.predict_single(dummy_seq)
        bio = res['biomechanics']
        bio_keys = [
            'rear_knee_angle_bottom', 'rear_knee_target_dev_bottom',
            'rear_femur_vertical_pitch', 'rear_shank_horizontal_pitch',
            'lead_rear_knee_ratio', 'inter_knee_vertical_disp',
            'inter_ankle_stride_bottom', 'pelvis_drop_to_stride_ratio'
        ]
        for bk in bio_keys:
            self.assertIn(bk, bio, f"Biomechanics must contain key '{bk}'.")
            self.assertFalse(np.isnan(bio[bk]))

if __name__ == '__main__':
    unittest.main()

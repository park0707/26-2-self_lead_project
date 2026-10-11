# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering Test Suite: test_dl3_rtmp_1_harness.py
Purpose: dl3_rtmp_1 프로덕션 모델 및 피처 추출기 단위 테스트 및 무결성 하네스 검증
"""

import os
import sys
import time
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")
sys.path.append(r"D:\src\models\second_models\dynamic_lunge")

from lunge_dl3_rtmp_features import (
    extract_lunge_dl3_rtmp_features_all,
    extract_lunge_dl3_22d_kinematics,
    extract_lunge_dl3_14d_baseline,
    FEATURE_NAMES_27D,
    FEATURE_NAMES_22D,
    FEATURE_NAMES_14D
)
from dl3_rtmp_1 import DL3RTMP1Classifier

class TestDL3RTMP1Harness(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.classifier = DL3RTMP1Classifier()
        # Mock random but plausible skeleton
        np.random.seed(42)
        cls.dummy_kps = np.random.uniform(-0.5, 0.5, size=(64, 26, 3)).astype(np.float32)
        cls.dummy_kps[:, :, 2] = 0.95 # Confidence
        
    def test_01_feature_extractor_shapes_and_nan(self):
        """피처 추출기의 출력 shape 및 NaN/Inf 무결성 검증"""
        f27 = extract_lunge_dl3_rtmp_features_all(self.dummy_kps, view_code='view3')
        f22 = extract_lunge_dl3_22d_kinematics(self.dummy_kps)
        f14 = extract_lunge_dl3_14d_baseline(self.dummy_kps)
        
        self.assertEqual(f27.shape, (27,))
        self.assertEqual(f22.shape, (22,))
        self.assertEqual(f14.shape, (14,))
        
        self.assertFalse(np.isnan(f27).any(), "NaN found in 27D features")
        self.assertFalse(np.isinf(f27).any(), "Inf found in 27D features")
        self.assertFalse(np.isnan(f22).any(), "NaN found in 22D features")
        self.assertFalse(np.isnan(f14).any(), "NaN found in 14D features")
        
    def test_02_all_views_feature_extraction(self):
        """공식 5대 시점별 원핫 인코딩 정상 작동 검증"""
        views = ['view1', 'view2', 'view3', 'view4', 'view5']
        for idx, v in enumerate(views):
            f27 = extract_lunge_dl3_rtmp_features_all(self.dummy_kps, view_code=v)
            # view encoding starts at index 22
            one_hot = f27[22:27]
            self.assertEqual(np.sum(one_hot), 1.0, f"View {v} one hot sum should be 1.0")
            
    def test_03_classifier_predict_sequence_structure(self):
        """추론기 출력 딕셔너리 구조 및 규격 검증"""
        res = self.classifier.predict_sequence(self.dummy_kps, view_code='view2')
        self.assertIn('state', res)
        self.assertIn(res['state'], [0, 1, 2])
        self.assertIn('decision', res)
        self.assertIn(res['decision'], ['NORMAL', 'DEFECT', 'UNDETERMINED'])
        self.assertIn('defect_probability', res)
        self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
        self.assertIn('coaching_message', res)
        self.assertTrue(len(res['coaching_message']) > 10)
        self.assertIn('biomechanics', res)
        
    def test_04_inference_latency(self):
        """단일 시퀀스 실시간 추론 지연시간 벤치마크 (<= 15ms)"""
        latencies = []
        for _ in range(50):
            t0 = time.time()
            _ = self.classifier.predict_sequence(self.dummy_kps, view_code='view2')
            latencies.append((time.time() - t0) * 1000.0)
            
        avg_latency = np.mean(latencies)
        print(f"\n  [Latency Benchmark] dl3_rtmp_1 Avg Inference Latency: {avg_latency:.2f} ms")
        self.assertLess(avg_latency, 15.0, f"Inference too slow: {avg_latency:.2f} ms")
        
    def test_05_real_sample_inference(self):
        """실제 다이나믹 런지 RTMPose 파일에 대한 엔드투엔드 추론 검증"""
        fpath = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지"
        import glob
        files = glob.glob(os.path.join(fpath, "*.npz"))
        self.assertTrue(len(files) > 0, "Real sample file not found")
        
        sample_data = np.load(files[0])
        kps_real = sample_data['kps_26_64f']
        res = self.classifier.predict_sequence(kps_real, view_code='view2')
        
        self.assertIn(res['state'], [0, 1, 2])
        self.assertFalse(np.isnan(res['defect_probability']))
        print(f"\n  [Real Sample] {os.path.basename(files[0])} -> State: {res['state_label']} (Prob: {res['defect_probability']:.3f})")

if __name__ == '__main__':
    unittest.main()

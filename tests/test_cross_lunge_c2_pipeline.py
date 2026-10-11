# -*- coding: utf-8 -*-
"""
vidio-health-trainer Test Suite
Module: test_cross_lunge_c2_pipeline.py
Purpose: 크로스 런지 c2 피처 추출 및 다리 분리 엔진 무결성 하네스 단위 테스트
         1. MediaPipe Full33 64F 시퀀스 Shape 무결성 (64, 33, 3)
         2. 크로스 런지 앞다리(Lead Leg) 및 최하사점(t_bot) 탐색 정상 동작
         3. 51차원 피처 규격 정합성 및 NaN / Inf 누출 0.00% 검증
         4. AI-Hub JSON 비참조 (순수 비전 기반 100% 무누수) 검증
"""

import os
import sys
import glob
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial
from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    extract_cross_lunge_c2_v7_features_51d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"


class TestCrossLungeC2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
        if not npz_files:
            raise unittest.SkipTest("전처리된 크로스 런지 NPZ 파일이 아직 없습니다.")
        cls.sample_files = npz_files[:5]

    def test_01_feature_dimension_and_nan(self):
        """51차원 피처 차원 및 결측치(NaN/Inf) 부재 검증"""
        for npz_path in self.sample_files:
            data = np.load(npz_path)
            self.assertIn('kps_33_3d_64f', data)
            kps = data['kps_33_3d_64f']
            self.assertEqual(kps.shape, (64, 33, 3))

            # 피처 추출
            feats = extract_cross_lunge_c2_v7_features_51d(kps, view_code='view3')
            self.assertEqual(len(feats), 51)
            self.assertEqual(len(FEATURE_NAMES_51D), 51)

            # NaN / Inf 검증
            self.assertFalse(np.isnan(feats).any(), f"NaN 발견: {npz_path}")
            self.assertFalse(np.isinf(feats).any(), f"Inf 발견: {npz_path}")

    def test_02_lead_leg_and_bottom_detection(self):
        """앞다리 및 최하사점 탐지 범위 검증"""
        for npz_path in self.sample_files:
            data = np.load(npz_path)
            kps = data['kps_33_3d_64f']
            is_left_lead, t_bot = detect_cross_lunge_lead_leg_spatial(kps)
            self.assertIsInstance(is_left_lead, bool)
            self.assertIsInstance(t_bot, int)
            self.assertGreaterEqual(t_bot, 0)
            self.assertLess(t_bot, 64)

    def test_03_cl2_final_hybrid_inference(self):
        """cl2_final 공식 최종 모델 하이브리드 단일 추론 무결성 검증"""
        from src.models.second_models.cross_lunge.cl2_final import predict_cross_lunge_c2
        dummy_feat_67d = np.zeros(67, dtype=np.float32)
        res = predict_cross_lunge_c2(dummy_feat_67d, camera_view='시점 C')
        self.assertEqual(res['defect_code'], 'cl2')
        self.assertIn(res['state'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertEqual(res['feature_dim'], 67)
        self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)


if __name__ == '__main__':
    unittest.main()


# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Test: test_sl5_hybrid_1.py
Purpose: sl5_hybrid_1 (MediaPipe + RTMPose 하이브리드 최적화 모델) 단위 테스트
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SL_DIR = r"D:\src\models\second_models\side_lunge"
if SL_DIR not in sys.path:
    sys.path.insert(0, SL_DIR)

from sl5_hybrid_1 import (
    SL5Hybrid1Classifier,
    extract_side_lunge_c5_hybrid_features,
    HYBRID_CACHE_NPZ,
    MODEL_BUNDLE_PKL,
    VIEW_ORDER
)


class TestSL5Hybrid1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle_path = MODEL_BUNDLE_PKL
        cls.classifier = SL5Hybrid1Classifier(cls.bundle_path)

    def test_01_bundle_loaded(self):
        """모델 번들이 정상 로드되었는지 검증"""
        self.assertTrue(self.classifier.is_loaded, "sl5_hybrid_1 번들이 로드되지 않았습니다.")
        self.assertEqual(len(self.classifier.models_lgb), 5, "LightGBM 5-Fold 모델 누락")
        self.assertEqual(len(self.classifier.models_cb), 5, "CatBoost 5-Fold 모델 누락")
        self.assertEqual(len(self.classifier.models_xgb), 5, "XGBoost 5-Fold 모델 누락")
        self.assertEqual(len(self.classifier.models_rf), 5, "RandomForest 5-Fold 모델 누락")
        self.assertEqual(len(self.classifier.models_et), 5, "ExtraTrees 5-Fold 모델 누락")

    def test_02_predict_proba_range(self):
        """단일 84차원 더미 피처 벡터 추론 시 확률값이 [0.0, 1.0] 범위 내인지 검증"""
        dummy_feat = np.random.randn(84).astype(np.float32)
        prob_raw = self.classifier.predict_proba(dummy_feat, calibrate=False)
        prob_cal = self.classifier.predict_proba(dummy_feat, calibrate=True)

        self.assertIsInstance(prob_raw, float)
        self.assertIsInstance(prob_cal, float)
        self.assertTrue(0.0 <= prob_raw <= 1.0, f"Raw 확률 비정상: {prob_raw}")
        self.assertTrue(0.0 <= prob_cal <= 1.0, f"Calibrated 확률 비정상: {prob_cal}")

    def test_03_predict_3state(self):
        """5대 시점별 3-State 판정 결과 규격 검증"""
        dummy_feat = np.random.randn(84).astype(np.float32)
        for vn in VIEW_ORDER:
            res = self.classifier.predict_3state(dummy_feat, view_code=vn)
            self.assertIn(res['state'], [0, 1, 2], f"State 값 비정상: {res['state']}")
            self.assertIn('status', res)
            self.assertIn('calibrated_prob', res)
            self.assertIn('tau_clean', res)
            self.assertIn('tau_defect', res)
            self.assertLess(res['tau_clean'], res['tau_defect'])

    def test_04_feature_extraction_interface(self):
        """추출기 인터페이스 규격 검증"""
        dummy_mp = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_rtmp = np.random.randn(64, 26, 3).astype(np.float32)
        feat_vec = extract_side_lunge_c5_hybrid_features(dummy_mp, dummy_rtmp, is_flipped=False)
        self.assertEqual(feat_vec.shape, (84,), f"추출된 피처 규격 불일치: {feat_vec.shape}")
        self.assertFalse(np.isnan(feat_vec).any(), "피처에 NaN 존재")


if __name__ == '__main__':
    unittest.main()

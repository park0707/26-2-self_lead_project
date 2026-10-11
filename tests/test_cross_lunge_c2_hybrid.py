# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering Test Suite
Test Module: tests/test_cross_lunge_c2_hybrid.py
Target Model: cl2_hybrid_1 (Cross Lunge Condition 2 Dual-Pose Hybrid Classifier)
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge.cl2_hybrid_1 import (
    extract_cl2_hybrid_1_features,
    CL2_HYBRID_1_FEATURE_NAMES,
    CL2Hybrid1Classifier,
    predict_cross_lunge_c2_hybrid
)

MP_SAMPLE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
RTM_SAMPLE_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\08_크로스_런지"
BUNDLE_PATH = r"D:\src\models\second_models\cross_lunge\cl2_hybrid_1_bundle.pkl"


class TestCrossLungeC2HybridModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import glob
        mp_files = sorted(glob.glob(os.path.join(MP_SAMPLE_DIR, "*.npz")))
        rtm_files = sorted(glob.glob(os.path.join(RTM_SAMPLE_DIR, "*.npz")))
        assert len(mp_files) > 0 and len(rtm_files) > 0, "샘플 데이터 파일 부재"

        cls.sample_mp = np.load(mp_files[0])['kps_33_3d_64f']
        cls.sample_rtm = np.load(rtm_files[0])['kps_26_64f']

        assert cls.sample_mp.shape == (64, 33, 3), f"MP 규격 오류: {cls.sample_mp.shape}"
        assert cls.sample_rtm.shape == (64, 26, 3), f"RTM 규격 오류: {cls.sample_rtm.shape}"

    def test_01_feature_extraction_dimension(self):
        """1. 하이브리드 86차원 피처 추출 차원 정합성 검증"""
        feat = extract_cl2_hybrid_1_features(self.sample_mp, self.sample_rtm, view_code='view3')
        self.assertEqual(feat.shape, (86,))
        self.assertEqual(len(CL2_HYBRID_1_FEATURE_NAMES), 86)
        self.assertFalse(np.isnan(feat).any(), "피처에 NaN이 존재합니다.")
        self.assertFalse(np.isinf(feat).any(), "피처에 Inf가 존재합니다.")

    def test_02_bundle_loading_and_weights(self):
        """2. 번들 로드 및 앙상블 파라미터 무결성 검증"""
        clf = CL2Hybrid1Classifier(bundle_path=BUNDLE_PATH)
        self.assertEqual(len(clf.models_cat), 5)
        self.assertEqual(len(clf.models_lgb), 5)
        self.assertEqual(len(clf.models_xgb), 5)
        self.assertEqual(len(clf.weights), 3)
        self.assertAlmostEqual(sum(clf.weights), 1.0, places=4)
        self.assertGreater(clf.temperature, 0.1)

    def test_03_view_coverage_and_thresholds(self):
        """3. 공식 5대 시점별 임계값 존재 및 무결성 검증"""
        clf = CL2Hybrid1Classifier(bundle_path=BUNDLE_PATH)
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            self.assertIn(v, clf.view_thresholds)
            th = clf.view_thresholds[v]
            self.assertIn('tau_clean', th)
            self.assertIn('tau_defect', th)
            self.assertLessEqual(th['tau_clean'], th['tau_defect'])

    def test_04_single_inference_pipeline(self):
        """4. 단일 시퀀스 3-State 추론 파이프라인 및 응답 규격 검증"""
        views_to_test = ['view1', 'view2', 'view3', 'view4', 'view5', '시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
        for v in views_to_test:
            res = predict_cross_lunge_c2_hybrid(self.sample_mp, self.sample_rtm, view_code=v, bundle_path=BUNDLE_PATH)
            self.assertIn('defect_code', res)
            self.assertEqual(res['defect_code'], 'cl2')
            self.assertIn('state', res)
            self.assertIn(res['state'], ['CLEAN', 'DEFECT', 'RETAKE'])
            self.assertIn('confidence', res)
            self.assertTrue(0.0 <= res['confidence'] <= 1.0)
            self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
            self.assertEqual(res['features_86d'].shape, (86,))


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering Test Suite
Test Module: tests/test_cross_lunge_rtmp_optimized.py
Target Models: cl1_rtmp_1, cl2_rtmp_1, cl3_rtmp_1 (Cross Lunge RTMPose Optimized)
"""

import os
import sys
import unittest
import numpy as np
import pickle

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge.cl1_rtmp_1 import (
    extract_cl1_rtmp_1_features,
    CL1_RTMP_1_FEATURE_NAMES,
    CL1RTMP1Classifier,
    predict_cross_lunge_c1_rtmp_1
)
from src.models.second_models.cross_lunge.cl2_rtmp_1 import (
    extract_cl2_rtmp_1_features,
    CL2_RTMP_1_FEATURE_NAMES,
    CL2RTMP1Classifier,
    predict_cross_lunge_c2_rtmp_1
)
from src.models.second_models.cross_lunge.cl3_rtmp_1 import (
    extract_cl3_rtmp_1_features,
    CL3_RTMP_1_FEATURE_NAMES,
    CL3RTMP1Classifier,
    predict_cross_lunge_c3_rtmp_1
)

BUNDLE_DIR = r"D:\src\models\second_models\cross_lunge"
BUNDLE_CL1 = os.path.join(BUNDLE_DIR, "cl1_rtmp_1_bundle.pkl")
BUNDLE_CL2 = os.path.join(BUNDLE_DIR, "cl2_rtmp_1_bundle.pkl")
BUNDLE_CL3 = os.path.join(BUNDLE_DIR, "cl3_rtmp_1_bundle.pkl")

SAMPLE_NPZ_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\08_크로스_런지"


class TestCrossLungeRTMPOptimizedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import glob
        sample_files = glob.glob(os.path.join(SAMPLE_NPZ_DIR, "*.npz"))
        assert len(sample_files) > 0, "샘플 데이터 파일이 존재하지 않습니다."
        cls.sample_path = sample_files[0]
        data = np.load(cls.sample_path)
        cls.sample_kps_26 = data['kps_26_64f']
        assert cls.sample_kps_26.shape == (64, 26, 3), f"샘플 규격 불일치: {cls.sample_kps_26.shape}"

    def test_01_feature_extraction_dimensions(self):
        """1. 세 모델별 RTMPose 고도화 피처 차원 정합성 검증"""
        f1 = extract_cl1_rtmp_1_features(self.sample_kps_26, view_code='view3')
        f2 = extract_cl2_rtmp_1_features(self.sample_kps_26, view_code='view3')
        f3 = extract_cl3_rtmp_1_features(self.sample_kps_26, view_code='view3')

        self.assertEqual(f1.shape, (74,), f"CL1 피처 차원 오류: {f1.shape} != (74,)")
        self.assertEqual(len(CL1_RTMP_1_FEATURE_NAMES), 74)
        self.assertEqual(f2.shape, (82,), f"CL2 피처 차원 오류: {f2.shape} != (82,)")
        self.assertEqual(len(CL2_RTMP_1_FEATURE_NAMES), 82)
        self.assertEqual(f3.shape, (46,), f"CL3 피처 차원 오류: {f3.shape} != (46,)")
        self.assertEqual(len(CL3_RTMP_1_FEATURE_NAMES), 46)

        # NaN / Inf 결측치 무결성 검증
        self.assertFalse(np.isnan(f1).any(), "CL1 피처에 NaN 존재")
        self.assertFalse(np.isnan(f2).any(), "CL2 피처에 NaN 존재")
        self.assertFalse(np.isnan(f3).any(), "CL3 피처에 NaN 존재")

    def test_02_bundles_exist_and_valid(self):
        """2. 가중치 번들 파일 생성 및 메타데이터 무결성 검증"""
        for b_path, exp_dim, name in [
            (BUNDLE_CL1, 74, "cl1_rtmp_1"),
            (BUNDLE_CL2, 82, "cl2_rtmp_1"),
            (BUNDLE_CL3, 46, "cl3_rtmp_1")
        ]:
            self.assertTrue(os.path.exists(b_path), f"번들 부재: {b_path}")
            with open(b_path, 'rb') as fp:
                b = pickle.load(fp)
            self.assertEqual(b['model_name'], name)
            self.assertEqual(b['feature_dim'], exp_dim)
            self.assertIn('temperature', b)
            self.assertIn('view_thresholds', b)
            self.assertIn('models_cat', b)
            self.assertIn('models_lgb', b)
            self.assertIn('models_xgb', b)

            # 5대 공식 시점 임계치 완비 여부
            for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
                self.assertIn(v, b['view_thresholds'])
                tc = b['view_thresholds'][v]['tau_clean']
                td = b['view_thresholds'][v]['tau_defect']
                self.assertLess(tc, td, f"임계치 역전 오류 ({v}): {tc} >= {td}")

    def test_03_cl1_rtmp_1_inference_all_views(self):
        """3. cl1_rtmp_1 공식 5대 시점 전수 추론 규격 검증"""
        views = ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
        for v in views:
            res = predict_cross_lunge_c1_rtmp_1(self.sample_kps_26, camera_view=v)
            self.assertEqual(res['model_version'], 'cl1_rtmp_1')
            self.assertIn(res['prediction_2state'], ['NORMAL', 'DEFECT'])
            self.assertIn(res['prediction_3state'], ['NORMAL', 'DEFECT', 'UNCERTAIN'])
            self.assertGreaterEqual(res['defect_probability'], 0.0)
            self.assertLessEqual(res['defect_probability'], 1.0)
            self.assertLess(res['tau_clean'], res['tau_defect'])

    def test_04_cl2_rtmp_1_inference_all_views(self):
        """4. cl2_rtmp_1 공식 5대 시점 전수 추론 규격 검증"""
        views = ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
        for v in views:
            res = predict_cross_lunge_c2_rtmp_1(self.sample_kps_26, camera_view=v)
            self.assertEqual(res['model_version'], 'cl2_rtmp_1')
            self.assertIn(res['prediction_2state'], ['NORMAL', 'DEFECT'])
            self.assertIn(res['prediction_3state'], ['NORMAL', 'DEFECT', 'UNCERTAIN'])
            self.assertGreaterEqual(res['defect_probability'], 0.0)
            self.assertLessEqual(res['defect_probability'], 1.0)

    def test_05_cl3_rtmp_1_inference_all_views(self):
        """5. cl3_rtmp_1 공식 5대 시점 전수 추론 규격 검증"""
        views = ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
        for v in views:
            res = predict_cross_lunge_c3_rtmp_1(self.sample_kps_26, camera_view=v)
            self.assertEqual(res['model_version'], 'cl3_rtmp_1')
            self.assertIn(res['prediction_2state'], ['NORMAL', 'DEFECT'])
            self.assertIn(res['prediction_3state'], ['NORMAL', 'DEFECT', 'UNCERTAIN'])
            self.assertGreaterEqual(res['defect_probability'], 0.0)
            self.assertLessEqual(res['defect_probability'], 1.0)

    def test_06_benchmark_results_and_reports_exist(self):
        """6. 공식 벤치마크 JSON 및 Word 보고서 생성 확인"""
        json_path = os.path.join(BUNDLE_DIR, "cross_lunge_rtmp_optimized_benchmark_results.json")
        docx_path = r"D:\자주프 문서\7주차\vidio-health-trainer_크로스런지_세부오류3종_RTMPose_최적화_모델_cl1_cl2_cl3_성능평가보고서.docx"
        self.assertTrue(os.path.exists(json_path), f"JSON 결과 파일 부재: {json_path}")
        self.assertTrue(os.path.exists(docx_path), f"Word 보고서 부재: {docx_path}")
        self.assertGreater(os.path.getsize(docx_path), 5000, "Word 보고서 파일이 비정상적으로 작습니다.")


if __name__ == '__main__':
    unittest.main()

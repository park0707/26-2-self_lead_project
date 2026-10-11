# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering Test Suite
Test Module: tests/test_cross_lunge_hybrid_1.py
Target Model: cl1_hybrid_1 (MediaPipe 3D + RTMPose 2D Hybrid 134D Classifier)
"""

import os
import sys
import unittest
import numpy as np
import pickle

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge.cl1_hybrid_1 import (
    extract_cl1_hybrid_1_features,
    CL1_HYBRID_1_FEATURE_NAMES,
    CL1Hybrid1Classifier,
    predict_cross_lunge_c1_hybrid_1,
    VIEW_NAME_TO_CODE,
    VIEW_CODE_TO_NAME
)

BUNDLE_DIR = r"D:\src\models\second_models\cross_lunge"
BUNDLE_CL1_HYBRID = os.path.join(BUNDLE_DIR, "cl1_hybrid_1_bundle.pkl")

SAMPLE_RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\08_크로스_런지"
SAMPLE_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"


class TestCrossLungeHybrid1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import glob
        rtm_files = sorted(glob.glob(os.path.join(SAMPLE_RTM_DIR, "*.npz")))
        assert len(rtm_files) > 0, "RTMPose 샘플 데이터 부재"
        cls.sample_rtm_path = rtm_files[0]
        fname = os.path.basename(cls.sample_rtm_path)
        cls.sample_mp_path = os.path.join(SAMPLE_MP_DIR, fname)
        assert os.path.exists(cls.sample_mp_path), f"MediaPipe 대응 샘플 부재: {cls.sample_mp_path}"

        d_rtm = np.load(cls.sample_rtm_path)
        cls.sample_kps_26 = d_rtm['kps_26_64f']
        assert cls.sample_kps_26.shape == (64, 26, 3), f"RTMPose 텐서 규격 오류: {cls.sample_kps_26.shape}"

        d_mp = np.load(cls.sample_mp_path)
        cls.sample_kps_33 = d_mp['kps_33_3d_64f']
        assert cls.sample_kps_33.shape == (64, 33, 3), f"MediaPipe 텐서 규격 오류: {cls.sample_kps_33.shape}"
        cls.sample_vis = d_mp['vis_64f'] if 'vis_64f' in d_mp.files else None

    def test_01_feature_extraction_dimensions(self):
        """1. cl1_hybrid_1 134D 피처 추출 차원 및 무결성 검증"""
        feats = extract_cl1_hybrid_1_features(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            view_code='view3',
            vis_64f=self.sample_vis
        )
        self.assertEqual(len(feats), 134, f"피처 차원이 134가 아닙니다: {len(feats)}")
        self.assertEqual(len(CL1_HYBRID_1_FEATURE_NAMES), 134)
        self.assertFalse(np.isnan(feats).any(), "피처에 NaN이 존재합니다.")
        self.assertFalse(np.isinf(feats).any(), "피처에 Inf가 존재합니다.")

    def test_02_bundle_file_integrity(self):
        """2. 가중치 번들 구조 및 메타데이터 무결성 검증"""
        self.assertTrue(os.path.exists(BUNDLE_CL1_HYBRID), f"번들 파일 부재: {BUNDLE_CL1_HYBRID}")
        with open(BUNDLE_CL1_HYBRID, 'rb') as fp:
            bundle = pickle.load(fp)

        self.assertEqual(bundle['model_name'], 'cl1_hybrid_1')
        self.assertEqual(bundle['feature_dim'], 134)
        self.assertEqual(len(bundle['models_cat']), 5)
        self.assertEqual(len(bundle['models_lgb']), 5)
        self.assertEqual(len(bundle['models_xgb']), 5)
        self.assertAlmostEqual(sum(bundle['ensemble_weights']), 1.0, places=4)
        self.assertGreater(bundle['temperature'], 0.1)

        # 5대 시점 임계치 전수 확인
        for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
            self.assertIn(v, bundle['view_thresholds'])
            th = bundle['view_thresholds'][v]
            self.assertIn('tau_clean', th)
            self.assertIn('tau_defect', th)
            self.assertLess(th['tau_clean'], th['tau_defect'])

    def test_03_classifier_feature_inference(self):
        """3. CL1Hybrid1Classifier 134D 피처 벡터 추론 검증"""
        clf = CL1Hybrid1Classifier(bundle_path=BUNDLE_CL1_HYBRID)
        feats = extract_cl1_hybrid_1_features(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            view_code='view3'
        )

        res = clf.predict_features(feats, camera_view='시점 C')
        self.assertEqual(res['model_version'], 'cl1_hybrid_1')
        self.assertEqual(res['camera_view_code'], 'view3')
        self.assertIn(res['prediction_2state'], ['NORMAL', 'DEFECT'])
        self.assertIn(res['prediction_3state'], ['NORMAL', 'DEFECT', 'UNCERTAIN'])
        self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
        self.assertTrue(0.0 <= res['raw_probability'] <= 1.0)

    def test_04_end_to_end_inference(self):
        """4. 양대 관절 텐서 기반 엔드투엔드 추론 검증"""
        clf = CL1Hybrid1Classifier(bundle_path=BUNDLE_CL1_HYBRID)
        res = clf.predict(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            camera_view='시점 B'
        )
        self.assertEqual(res['camera_view_code'], 'view2')
        self.assertIn('tau_clean', res)
        self.assertIn('tau_defect', res)

    def test_05_top_level_functional_api(self):
        """5. predict_cross_lunge_c1_hybrid_1 함수 규격 검증"""
        # 텐서 직접 입력
        res_tensor = predict_cross_lunge_c1_hybrid_1(
            kps_26_or_features=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            camera_view='시점 D'
        )
        self.assertEqual(res_tensor['camera_view_code'], 'view4')

        # 피처 직접 입력
        feats = extract_cl1_hybrid_1_features(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            view_code='view1'
        )
        res_feat = predict_cross_lunge_c1_hybrid_1(
            kps_26_or_features=feats,
            camera_view='시점 E'
        )
        self.assertEqual(res_feat['camera_view_code'], 'view1')


if __name__ == '__main__':
    unittest.main()

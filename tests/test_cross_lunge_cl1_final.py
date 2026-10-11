# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering Test Suite
Test Module: tests/test_cross_lunge_cl1_final.py
Target Model: cl1_final (Cross Lunge C1 Official Final Champion Model)
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.cross_lunge import (
    predict_cross_lunge_c1,
    extract_cl1_final_features,
    CL1FinalClassifier
)

BUNDLE_DIR = r"D:\src\models\second_models\cross_lunge"
BUNDLE_CL1_FINAL = os.path.join(BUNDLE_DIR, "cl1_final_bundle.pkl")

SAMPLE_RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\08_크로스_런지"
SAMPLE_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"


class TestCrossLungeCL1Final(unittest.TestCase):
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

        d_mp = np.load(cls.sample_mp_path)
        cls.sample_kps_33 = d_mp['kps_33_3d_64f']
        cls.sample_vis = d_mp['vis_64f'] if 'vis_64f' in d_mp.files else None

    def test_01_bundle_existence(self):
        """1. cl1_final_bundle.pkl 파일 존재 및 로드 검증"""
        self.assertTrue(os.path.exists(BUNDLE_CL1_FINAL), f"번들 부재: {BUNDLE_CL1_FINAL}")
        clf = CL1FinalClassifier(bundle_path=BUNDLE_CL1_FINAL)
        self.assertEqual(clf.model_name, "cl1_final")
        self.assertEqual(len(clf.feature_names), 134)

    def test_02_feature_extraction(self):
        """2. cl1_final 134D 피처 추출 무결성 검증"""
        feats = extract_cl1_final_features(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            view_code='view3',
            vis_64f=self.sample_vis
        )
        self.assertEqual(feats.shape, (134,))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_03_predict_function_dual_tensor(self):
        """3. predict_cross_lunge_c1 듀얼 텐서 엔드투엔드 추론 검증"""
        views_to_test = ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
        expected_codes = ['view5', 'view2', 'view3', 'view4', 'view1']

        for v_name, v_code in zip(views_to_test, expected_codes):
            res = predict_cross_lunge_c1(
                kps_26_or_features=self.sample_kps_26,
                kps_33_3d_64f=self.sample_kps_33,
                camera_view=v_name
            )
            self.assertEqual(res['model_version'], 'cl1_final')
            self.assertEqual(res['camera_view_code'], v_code)
            self.assertIn(res['prediction_2state'], ['NORMAL', 'DEFECT'])
            self.assertIn(res['prediction_3state'], ['NORMAL', 'DEFECT', 'UNCERTAIN'])
            self.assertTrue(0.0 <= res['defect_probability'] <= 1.0)
            self.assertTrue(0.0 <= res['raw_probability'] <= 1.0)
            self.assertIn('tau_clean', res)
            self.assertIn('tau_defect', res)

    def test_04_predict_function_feature_input(self):
        """4. predict_cross_lunge_c1 134D 피처 벡터 직접 입력 추론 검증"""
        feats = extract_cl1_final_features(
            kps_26_64f=self.sample_kps_26,
            kps_33_3d_64f=self.sample_kps_33,
            view_code='view2'
        )
        res = predict_cross_lunge_c1(
            kps_26_or_features=feats,
            camera_view='시점 B'
        )
        self.assertEqual(res['model_version'], 'cl1_final')
        self.assertEqual(res['camera_view_code'], 'view2')


if __name__ == '__main__':
    unittest.main()

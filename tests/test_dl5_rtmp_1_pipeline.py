# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Pipeline Test: test_dl5_rtmp_1_pipeline.py
Purpose: dl5_rtmp_1 프로덕션 추론 파이프라인 무결성 및 단위 검증
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.dynamic_lunge.dl5_rtmp_1 import (
    DL5RTMP1Classifier,
    get_dl5_rtmp_1_classifier,
    predict_dynamic_lunge_c5_rtmp_1,
    VIEW_THRESHOLDS_10,
    VIEW_THRESHOLDS_15
)

DATASET_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_rtmp_dataset.npz"
MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]

class TestDL5RTMP1Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clf = get_dl5_rtmp_1_classifier()
        cls.dataset = np.load(DATASET_PATH, allow_pickle=True)

    def test_01_bundle_integrity(self):
        """1. 번들 로드 및 메타데이터 무결성 검증"""
        self.assertTrue(self.clf.is_loaded)
        self.assertEqual(self.clf.model_name, "dl5_rtmp_1")
        self.assertEqual(len(self.clf.feature_names), 79)
        self.assertEqual(len(self.clf.weights), 3)
        self.assertAlmostEqual(sum(self.clf.weights), 1.0)
        self.assertIsNotNone(self.clf.model_cat)
        self.assertIsNotNone(self.clf.model_lgb)
        self.assertIsNotNone(self.clf.model_xgb)

    def test_02_feature_extraction(self):
        """2. 79차원 하이브리드 피처 추출 및 결측치(NaN/Inf) 무결성 검증"""
        sample_mp = np.random.randn(64, 33, 3).astype(np.float32)
        sample_rtmp = np.random.randn(64, 26, 3).astype(np.float32)

        feats = self.clf.extract_features(sample_mp, sample_rtmp)
        self.assertEqual(feats.shape, (79,))
        self.assertFalse(np.isnan(feats).any(), "피처에 NaN이 존재합니다.")
        self.assertFalse(np.isinf(feats).any(), "피처에 Inf가 존재합니다.")

    def test_03_real_samples_inference(self):
        """3. 실측 다이나믹 런지 데이터(포워드 및 백워드) 전수 규격 추론 검증"""
        fnames = list(self.dataset['filenames'])[:5]
        for fname in fnames:
            p_mp = os.path.join(MP33_DIR, fname)
            p_rtmp = os.path.join(RTMP_DIRS[0], fname)
            if not os.path.exists(p_rtmp):
                p_rtmp = os.path.join(RTMP_DIRS[1], fname)

            d_mp = np.load(p_mp)
            d_rtmp = np.load(p_rtmp)

            kps_mp = d_mp['kps_33_3d_64f']
            kps_rtmp = d_rtmp['kps_26_64f']

            res = self.clf.predict_sequence(kps_mp, kps_rtmp, view='view3')
            self.assertIn(res['status'], ['Normal', 'Defect', 'Caution_Retake'])
            self.assertIn(res['state_code'], [0, 1, 2])
            self.assertIsInstance(res['is_defect'], bool)
            self.assertGreaterEqual(res['defect_prob'], 0.0)
            self.assertLessEqual(res['defect_prob'], 1.0)
            self.assertGreater(len(res['feedback']), 0)
            self.assertIn('torso_pitch_peak_deg', res['key_metrics'])

    def test_04_all_5_views_adaptation(self):
        """4. 공식 5대 시점(A~E) 및 view1~view5 적응형 임계값 매핑 검증"""
        sample_mp = np.random.randn(64, 33, 3).astype(np.float32)
        sample_rtmp = np.random.randn(64, 26, 3).astype(np.float32)

        view_test_cases = [
            ('view5', '시점 A (우측 후방 45°)'),
            ('시점 A', '시점 A (우측 후방 45°)'),
            ('view2', '시점 B (우측 전방 45°)'),
            ('시점 B', '시점 B (우측 전방 45°)'),
            ('view3', '시점 C (정면 0°)'),
            ('시점 C', '시점 C (정면 0°)'),
            ('view4', '시점 D (좌측 전방 45°)'),
            ('시점 D', '시점 D (좌측 전방 45°)'),
            ('view1', '시점 E (좌측 후방 45°)'),
            ('시점 E', '시점 E (좌측 후방 45°)'),
        ]

        for v_input, v_expected_display in view_test_cases:
            res = self.clf.predict_sequence(sample_mp, sample_rtmp, view=v_input)
            self.assertEqual(res['applied_view'], v_expected_display)

    def test_05_convenience_function(self):
        """5. 모듈 레벨 퀵 추론 함수 predict_dynamic_lunge_c5_rtmp_1 작동 검증"""
        sample_mp = np.zeros((64, 33, 3), dtype=np.float32)
        sample_rtmp = np.zeros((64, 26, 3), dtype=np.float32)
        res = predict_dynamic_lunge_c5_rtmp_1(sample_mp, sample_rtmp, view='view2')
        self.assertEqual(res['model'], 'dl5_rtmp_1')
        self.assertIn('feedback', res)

if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Module: tests/test_side_lunge_c1_hybrid.py
Purpose: MediaPipe + RTMPose 하이브리드 파이프라인 및 sl1_hybrid_1 모델 단위 테스트 스위트
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_NPZ = r"D:\Data\피처_데이터\sl1_hybrid_features.npz"
BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_hybrid_1_bundle.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_hybrid_1_evaluation_results.json"


class TestSideLungeC1Hybrid(unittest.TestCase):

    def test_cache_dimensions_and_no_nan(self):
        """하이브리드 피처 캐시 차원 및 결측치(NaN/Inf) 무결성 검증"""
        self.assertTrue(os.path.exists(CACHE_NPZ), f"캐시 파일 부재: {CACHE_NPZ}")
        data = np.load(CACHE_NPZ)

        X_55d = data['X_hybrid_55d']
        X_110d = data['X_hybrid_110d']
        y = data['y']
        groups = data['groups']
        views = data['views']

        self.assertEqual(X_55d.shape, (4130, 55))
        self.assertEqual(X_110d.shape, (4130, 110))
        self.assertEqual(len(y), 4130)
        self.assertEqual(len(groups), 4130)
        self.assertEqual(len(views), 4130)

        self.assertEqual(np.isnan(X_55d).sum(), 0)
        self.assertEqual(np.isinf(X_55d).sum(), 0)
        self.assertEqual(np.isnan(X_110d).sum(), 0)
        self.assertEqual(np.isinf(X_110d).sum(), 0)

        # 피험자 9명 전수, 5대 시점 전수 확인
        self.assertEqual(len(np.unique(groups)), 9)
        self.assertEqual(len(np.unique(views)), 5)

    def test_5view_mapping_and_order(self):
        """5대 공식 카메라 시점 매핑 및 순서 무결성 검증"""
        expected_views = {'view1', 'view2', 'view3', 'view4', 'view5'}
        data = np.load(CACHE_NPZ)
        actual_views = set(np.unique(data['views']))
        self.assertEqual(expected_views, actual_views)

    def test_inference_bundle_and_3state(self):
        """모델 번들 로드 및 단일 시퀀스 3-State 추론 규격 검증"""
        import pickle
        if not os.path.exists(BUNDLE_PKL):
            self.skipTest(f"번들 파일 아직 미생성: {BUNDLE_PKL}")

        with open(BUNDLE_PKL, 'rb') as f:
            clf = pickle.load(f)

        data = np.load(CACHE_NPZ)
        feat_110d = data['X_hybrid_110d'][0]

        # 단일 추론 확률
        p = clf.predict_proba(feat_110d)
        self.assertEqual(len(p), 1)
        self.assertTrue(0.0 <= p[0] <= 1.0)

        # 3-State 판정
        res = clf.predict_3state(feat_110d, view_code='view3')
        self.assertIn('state', res)
        self.assertIn(res['state'], [0, 1, 2])
        self.assertIn('state_label', res)
        self.assertIn('feedback', res)
        self.assertTrue(0.0 <= res['calibrated_prob'] <= 1.0)


if __name__ == '__main__':
    unittest.main()

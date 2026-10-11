# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Pipeline Test Suite for dl2_rtmp_1
Script: test_dl2_rtmp_1_pipeline.py
Purpose: harness-engineering 규정에 따른 dl2_rtmp_1 단일 추론 파이프라인 무결성 검증
"""

import os
import sys
import glob
import unittest
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dl2_rtmp_1 import (
    predict_dynamic_lunge_c2,
    extract_dl2_rtmp_1_features,
    load_dl2_rtmp_1_bundle,
    BUNDLE_PATH,
    WEIGHTS_BUNDLE_PATH,
    VIEW_ALIAS_MAP,
    VIEW_NAME_TO_CODE
)

class TestDL2RTMP1Pipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("  [Harness-Engineering] dl2_rtmp_1 파이프라인 무결성 테스트 스위트")
        print("=" * 70)
        cls.bundle = load_dl2_rtmp_1_bundle()

    def test_01_bundle_integrity(self):
        """모델 번들 내부 키 및 컴포넌트 무결성 검증"""
        self.assertIsNotNone(self.bundle)
        self.assertIn('lgb_models', self.bundle)
        self.assertIn('cat_models', self.bundle)
        self.assertIn('xgb_models', self.bundle)
        self.assertEqual(len(self.bundle['lgb_models']), 5)
        self.assertEqual(len(self.bundle['cat_models']), 5)
        self.assertEqual(len(self.bundle['xgb_models']), 5)
        self.assertIn('selected_features', self.bundle)
        self.assertEqual(len(self.bundle['selected_features']), 12)
        self.assertIn('per_view_thresholds', self.bundle)
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            self.assertIn(v, self.bundle['per_view_thresholds'])
            self.assertIn('tau_clean', self.bundle['per_view_thresholds'][v])
            self.assertIn('tau_defect', self.bundle['per_view_thresholds'][v])
            self.assertLess(
                self.bundle['per_view_thresholds'][v]['tau_clean'],
                self.bundle['per_view_thresholds'][v]['tau_defect']
            )

    def test_02_feature_extraction_dimension_and_nan(self):
        """피처 추출기 규격 (12D) 및 NaN/Inf 무결성 검증"""
        dummy_mp = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_rtmp = np.random.randn(64, 26, 3).astype(np.float32)
        feat = extract_dl2_rtmp_1_features(dummy_mp, dummy_rtmp)
        self.assertEqual(feat.shape, (12,))
        self.assertFalse(np.isnan(feat).any())
        self.assertFalse(np.isinf(feat).any())

    def test_03_inference_schema_all_views(self):
        """공식 5대 시점 전체에 대한 추론 반환 스키마 및 제약 조건 검증"""
        dummy_mp = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_rtmp = np.random.randn(64, 26, 3).astype(np.float32)

        view_test_cases = [
            '시점 A', '시점 B', '시점 C', '시점 D', '시점 E',
            'view5', 'view2', 'view3', 'view4', 'view1'
        ]

        for v in view_test_cases:
            res = predict_dynamic_lunge_c2(dummy_mp, dummy_rtmp, camera_view=v)
            self.assertEqual(res['defect_code'], 'dl2')
            self.assertIn(res['state'], ['CLEAN', 'DEFECT', 'RETAKE'])
            self.assertIn(res['state_kr'], ['정상', '오류 (몸통-발-무릎 방향 불일치)', '판정 보류 (재촬영 권고)'])
            if res['state'] == 'CLEAN':
                self.assertFalse(res['is_defect'])
            elif res['state'] == 'DEFECT':
                self.assertTrue(res['is_defect'])
            else:
                self.assertIsNone(res['is_defect'])
                
            self.assertGreaterEqual(res['probability'], 0.0)
            self.assertLessEqual(res['probability'], 1.0)
            self.assertGreaterEqual(res['confidence'], 0.0)
            self.assertLessEqual(res['confidence'], 1.0)
            self.assertIn(res['view_code'], ['view1', 'view2', 'view3', 'view4', 'view5'])
            self.assertIn(res['view_name'], VIEW_ALIAS_MAP.values())
            self.assertTrue(len(res['feedback']) > 10)

    def test_04_shape_tolerance(self):
        """(64, 26, 2) vs (64, 26, 3) RTMPose 입력 형상 허용 오차 검증"""
        dummy_mp = np.random.randn(64, 33, 3).astype(np.float32)
        dummy_rtmp_2d = np.random.randn(64, 26, 2).astype(np.float32)
        dummy_rtmp_3d = np.random.randn(64, 26, 3).astype(np.float32)

        res2d = predict_dynamic_lunge_c2(dummy_mp, dummy_rtmp_2d, camera_view='시점 C')
        res3d = predict_dynamic_lunge_c2(dummy_mp, dummy_rtmp_3d, camera_view='시점 C')

        self.assertIn(res2d['state'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertIn(res3d['state'], ['CLEAN', 'DEFECT', 'RETAKE'])

    def test_05_real_samples_verification(self):
        """실제 다이나믹 런지 데이터셋 샘플을 로드하여 추론 동작 검증"""
        mp_files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지\*.npz")[:5]
        rtmp_dirs = [
            r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
            r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
        ]

        tested_count = 0
        for mp_path in mp_files:
            fname = os.path.basename(mp_path)
            rtmp_path = None
            for rd in rtmp_dirs:
                c = os.path.join(rd, fname)
                if os.path.exists(c):
                    rtmp_path = c
                    break
            if not rtmp_path:
                continue

            mp = np.load(mp_path)['kps_33_3d_64f']
            rtmp = np.load(rtmp_path)['kps_26_64f']

            view_code = 'view3'
            for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
                if f"_{v}.npz" in fname:
                    view_code = v
                    break

            res = predict_dynamic_lunge_c2(mp, rtmp, camera_view=view_code)
            self.assertIn(res['state'], ['CLEAN', 'DEFECT', 'RETAKE'])
            self.assertGreaterEqual(res['probability'], 0.0)
            self.assertLessEqual(res['probability'], 1.0)
            tested_count += 1

        self.assertGreater(tested_count, 0, "실제 샘플 검증 최소 1건 이상 실행되어야 함")
        print(f"\n[+] 실제 다이나믹 런지 샘플 {tested_count}건 정밀 추론 검증 성공!")


if __name__ == '__main__':
    unittest.main()

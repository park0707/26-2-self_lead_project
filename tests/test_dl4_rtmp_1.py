# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_dl4_rtmp_1.py
Purpose: 다이나믹 런지 c4 (척추의 중립 위반) dl4_rtmp_1 모델 및 추론 엔진 무결성 검증
         - 1. 번들 로드 및 파라미터 무결성
         - 2. 2D/3D 키포인트 (64, 26, 2) 및 (64, 26, 3) Fake-Z 사상 규격 검증
         - 3. 24D 생체역학 피처 추출 및 (1, 24) 배치 피처 입력 검증
         - 4. 확률 예측 범위 [0, 1] 검증
         - 5. 5대 공식 시점 코드 및 전체 정식 명칭 라우팅 엄격 검증
         - 6. 실제 데이터셋 샘플 무작위 추출 실측 추론 검증
         - 7. 비정상 입력(Zero 텐서, 극단값 노이즈)에 대한 예외 방어 및 안정성 검증
         - 8. 모듈 최상위 싱글톤 및 편의 함수(predict_dynamic_lunge_c4) 검증
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.dynamic_lunge.dl4_rtmp_1 import (
    DL4RTMP1Classifier,
    get_dl4_classifier,
    predict_dynamic_lunge_c4,
    map_rtmp26_to_mp33_with_fake_z,
    VIEW_ALIAS_MAP
)

CACHE_PATH = r"D:\Data\피처_데이터\dl4_rtmp_1_dataset_cache.npz"


class TestDL4RTMP1Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = DL4RTMP1Classifier()

    def test_01_model_loaded(self):
        """모델 번들이 정상 로드되었는지 검증"""
        self.assertTrue(self.classifier.is_loaded, "dl4_rtmp_1_bundle.pkl이 로드되지 않았습니다.")
        self.assertIsNotNone(self.classifier.model_cat, "CatBoost 모델이 누락되었습니다.")
        self.assertEqual(len(self.classifier.feature_names), 24, "피처 차원이 24D가 아닙니다.")
        self.assertGreater(self.classifier.temperature, 0.0, "온도 파라미터가 유효하지 않습니다.")

    def test_02_map_rtmp26_to_mp33_2d_and_3d(self):
        """RTMPose Halpe26 2D(64, 26, 2) 및 3D(64, 26, 3) Fake-Z 사상 규격 검증"""
        # 1) 3D 입력 (T, 26, 3)
        dummy_kps3d = np.random.randn(64, 26, 3).astype(np.float32)
        mapped_3d = map_rtmp26_to_mp33_with_fake_z(dummy_kps3d)
        self.assertEqual(mapped_3d.shape, (64, 33, 3))
        self.assertAlmostEqual(float(mapped_3d[0, 11, 2]), -0.1, places=4)
        self.assertAlmostEqual(float(mapped_3d[0, 12, 2]), 0.1, places=4)

        # 2) 2D 입력 (T, 26, 2) - 결함 1 방어 테스트
        dummy_kps2d = np.random.randn(64, 26, 2).astype(np.float32)
        mapped_2d = map_rtmp26_to_mp33_with_fake_z(dummy_kps2d)
        self.assertEqual(mapped_2d.shape, (64, 33, 3))
        self.assertAlmostEqual(float(mapped_2d[0, 11, 2]), -0.1, places=4)
        self.assertAlmostEqual(float(mapped_2d[0, 12, 2]), 0.1, places=4)

    def test_03_extract_features_shape_and_2d_batch(self):
        """24D 생체역학 피처 추출 및 (1, 24) 배치 입력 지원 검증 (결함 3 방어)"""
        dummy_kps26 = np.random.randn(64, 26, 3).astype(np.float32)
        feats = self.classifier.extract_features(dummy_kps26)
        self.assertEqual(feats.shape, (24,))
        self.assertFalse(np.isnan(feats).any(), "피처에 NaN이 포함되어 있습니다.")
        self.assertFalse(np.isinf(feats).any(), "피처에 Inf가 포함되어 있습니다.")

        # (1, 24) 직접 입력 검증
        batch_feat = feats.reshape(1, 24)
        feats_from_batch = self.classifier.extract_features(batch_feat)
        self.assertEqual(feats_from_batch.shape, (24,))

    def test_04_predict_proba_range(self):
        """예측 확률값 범위 [0.0, 1.0] 검증"""
        dummy_kps26 = np.random.randn(64, 26, 3).astype(np.float32)
        prob = self.classifier.predict_proba(dummy_kps26, calibrate=True)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_05_predict_3state_all_views_and_full_aliases(self):
        """공식 5대 시점 코드, 단축명, 정식 명칭 라우팅 검증 (결함 2 방어)"""
        dummy_kps26 = np.random.randn(64, 26, 3).astype(np.float32)
        test_views = [
            ('view5', 'view5'),
            ('시점 A', 'view5'),
            ('시점 A (우측 후방 45°)', 'view5'),
            ('view2', 'view2'),
            ('시점 B', 'view2'),
            ('시점 B (우측 전방 45°)', 'view2'),
            ('view3', 'view3'),
            ('시점 C', 'view3'),
            ('시점 C (정면 0°)', 'view3'),
            ('view4', 'view4'),
            ('시점 D', 'view4'),
            ('시점 D (좌측 전방 45°)', 'view4'),
            ('view1', 'view1'),
            ('시점 E', 'view1'),
            ('시점 E (좌측 후방 45°)', 'view1'),
        ]
        for v_input, expected_code in test_views:
            res = self.classifier.predict_3state(dummy_kps26, view=v_input)
            self.assertEqual(res['view_code'], expected_code, f"시점 라우팅 실패: {v_input} -> {res['view_code']}")
            self.assertIn(res['state'], [0, 1, 2])
            self.assertGreaterEqual(res['probability'], 0.0)
            self.assertLessEqual(res['probability'], 1.0)
            self.assertLessEqual(res['tau_clean'], res['tau_defect'])

    def test_06_real_sample_inference(self):
        """실제 캐시 데이터셋 샘플 추론 검증"""
        if os.path.exists(CACHE_PATH):
            data = np.load(CACHE_PATH, allow_pickle=True)
            X = data['X']
            y = data['y']
            views = data['views']

            indices = np.random.choice(len(y), size=min(10, len(y)), replace=False)
            for idx in indices:
                feat = X[idx]
                v = views[idx]
                res = self.classifier.predict_3state(feat, view=v)
                self.assertIn(res['state'], [0, 1, 2])
                self.assertIsInstance(res['feedback'], str)
                self.assertGreater(len(res['feedback']), 5)

    def test_07_extreme_inputs_safety(self):
        """비정상 입력(Zero 텐서, 극단값 노이즈)에 대한 예외 방어 및 안정성 검증"""
        zeros_kps = np.zeros((64, 26, 3), dtype=np.float32)
        res_zero = self.classifier.predict_3state(zeros_kps, view='view3')
        self.assertIn(res_zero['state'], [0, 1, 2])
        self.assertFalse(np.isnan(res_zero['probability']))

        noisy_kps = np.full((64, 26, 3), 999.0, dtype=np.float32)
        res_noisy = self.classifier.predict_3state(noisy_kps, view='view2')
        self.assertIn(res_noisy['state'], [0, 1, 2])
        self.assertFalse(np.isnan(res_noisy['probability']))

    def test_08_module_convenience_functions(self):
        """모듈 최상위 싱글톤 및 편의 함수(predict_dynamic_lunge_c4) 검증"""
        clf = get_dl4_classifier()
        self.assertIsInstance(clf, DL4RTMP1Classifier)

        dummy_kps26 = np.random.randn(64, 26, 3).astype(np.float32)
        res = predict_dynamic_lunge_c4(dummy_kps26, view='시점 B (우측 전방 45°)')
        self.assertEqual(res['view_code'], 'view2')
        self.assertIn(res['state'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

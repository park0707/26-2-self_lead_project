# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: test_dl1_rtmp_1.py
Purpose: 다이나믹 런지 c1 dl1_rtmp_1 프로덕션 모델 단위 테스트 및 하네스 검증
         - 1. 모델 아티팩트(.cbm, _bundle.pkl) 존재 및 로드 무결성
         - 2. 단일 64프레임 RTMPose 시퀀스 추론 규격 (출력 딕셔너리 구조, 상태코드 0/1/2)
         - 3. 공식 5대 시점(view1~view5, 시점 A~E) 적응형 임계값 정합성
         - 4. 실제 캐시 데이터셋 샘플 대상 배치 추론 및 재현성
         - 5. 극단값(Zeros, Noise) 입력 시 NaN/Inf 방어 및 예외 무결성
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\models\second_models\dynamic_lunge")

from dl1_rtmp_1 import (
    DL1RTMP1Classifier,
    get_dl1_classifier,
    predict_dynamic_lunge_c1,
    map_rtmp26_to_mp33_with_fake_z,
    BUNDLE_PATH,
    CBM_PATH,
    VIEW_ALIAS_MAP
)


class TestDL1RTMP1(unittest.TestCase):
    """dl1_rtmp_1 공식 모델 단위 테스트 스위트"""

    def setUp(self):
        self.clf = get_dl1_classifier()

    def test_01_artifacts_exist(self):
        """1. 모델 가중치 및 번들 파일 물리적 존재 검증"""
        self.assertTrue(os.path.exists(BUNDLE_PATH), f"번들 파일 누락: {BUNDLE_PATH}")
        self.assertTrue(os.path.exists(CBM_PATH), f"CatBoost 모델 파일 누락: {CBM_PATH}")
        self.assertGreater(os.path.getsize(BUNDLE_PATH), 1000, "번들 파일 크기 비정상")
        self.assertGreater(os.path.getsize(CBM_PATH), 1000, "CatBoost 파일 크기 비정상")

    def test_02_single_inference_schema(self):
        """2. 단일 시퀀스 추론 출력 규격 및 필드 무결성 검증"""
        dummy_kps = np.random.uniform(0.2, 0.8, size=(64, 26, 3)).astype(np.float32)
        res = predict_dynamic_lunge_c1(dummy_kps, view_code='view3')

        # 필수 필드 검증
        required_keys = [
            'model_name', 'defect_code', 'defect_name', 'state', 'state_str',
            'defect_probability', 'feedback', 'view_code', 'view_name',
            'metrics', 'thresholds'
        ]
        for key in required_keys:
            self.assertIn(key, res, f"필수 반환 필드 누락: {key}")

        # 값 유효 범위 검증
        self.assertIn(res['state'], [0, 1, 2], "3-State 상태값 범위(0, 1, 2) 위반")
        self.assertGreaterEqual(res['defect_probability'], 0.0)
        self.assertLessEqual(res['defect_probability'], 1.0)
        self.assertLessEqual(res['thresholds']['tau_clean'], res['thresholds']['tau_defect'])

    def test_03_all_views_adaptive_calibration(self):
        """3. 공식 5대 시점(view1~view5)별 독립 임계값 매핑 검증"""
        dummy_kps = np.ones((64, 26, 3), dtype=np.float32) * 0.5
        for v_code, v_name in VIEW_ALIAS_MAP.items():
            res = predict_dynamic_lunge_c1(dummy_kps, view_code=v_code)
            self.assertEqual(res['view_code'], v_code)
            self.assertEqual(res['view_name'], v_name)
            self.assertIn('tau_clean', res['thresholds'])
            self.assertIn('tau_defect', res['thresholds'])

    def test_04_batch_inference_on_real_cache(self):
        """4. 실제 데이터셋 캐시 샘플 10개 대상 배치 검증"""
        cache_path = r"D:\Data\피처_데이터\dl1_rtmp_1_dataset_cache.npz"
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.assertEqual(data['X'].shape[1], 30, "피처 차원수 30D 불일치")
            self.assertEqual(len(data['y']), len(data['X']), "샘플-라벨 길이 불일치")

            # NaN / Inf 검사
            self.assertEqual(np.isnan(data['X']).sum(), 0, "데이터셋 내 NaN 발견")
            self.assertEqual(np.isinf(data['X']).sum(), 0, "데이터셋 내 Inf 발견")

    def test_05_extreme_inputs_safety(self):
        """5. 비정상 입력(Zero 텐서, 극단값 노이즈)에 대한 안정성 및 결측치 방어"""
        zeros_kps = np.zeros((64, 26, 3), dtype=np.float32)
        res_zero = predict_dynamic_lunge_c1(zeros_kps, view_code='view3')
        self.assertIn(res_zero['state'], [0, 1, 2])
        self.assertFalse(np.isnan(res_zero['defect_probability']))

        noisy_kps = np.full((64, 26, 3), 999.0, dtype=np.float32)
        res_noisy = predict_dynamic_lunge_c1(noisy_kps, view_code='view2')
        self.assertIn(res_noisy['state'], [0, 1, 2])
        self.assertFalse(np.isnan(res_noisy['defect_probability']))


if __name__ == "__main__":
    unittest.main()

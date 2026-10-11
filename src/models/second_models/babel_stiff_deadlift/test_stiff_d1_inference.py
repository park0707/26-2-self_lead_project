# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: d1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Module: test_stiff_d1_inference.py
Purpose: 스티프 데드리프트 d1 피처 추출기(stiff_d1_features.py, 38D) 및
         단일 시퀀스 추론 파이프라인(stiff_d1_inference.py)에 대한
         단위 테스트 및 통합 추론 검증 (입력 규격, 극단값 방어, 5대 시점 전수, 3-State 판정)
Rule: AGENTS.md 규정 (5대 공식 시점 taxonomy, 64-Frame Pipeline, 비전 100% 무결성)
"""

import os
import sys
import glob
import time
import pickle
import warnings
import tempfile
import unittest
from typing import Dict, Any, List

import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_features,
    FEATURE_NAMES_38_STIFF_D1,
    VIEW_ALIAS_MAP,
    VIEW_CODE_MAP
)
from src.models.second_models.babel_stiff_deadlift.stiff_d1_inference import (
    infer_stiff_d1_sequence,
    load_model_bundle,
    WEIGHTS_PATH
)
import src.models.second_models.babel_stiff_deadlift.stiff_d1_inference as d1_inf_module

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"


class TestStiffD1FeatureExtraction(unittest.TestCase):
    """
    stiff_d1_features.py 단위 테스트 슈트
    - 입력 규격 검증 (64, 33, 3) -> 38D
    - 극단값 / 엣지 케이스 안전 방어 (Zeros, NaN, Inf, 극한 스케일, 제로-디비전)
    - 5대 공식 시점(view1~view5) 전수 피처 추출 및 시점 인디케이터 검증
    """

    def setUp(self):
        # 재현성을 위한 시드 고정
        np.random.seed(42)
        # 정상 범위의 인체 관절 더미 텐서 (64 프레임, 33 관절, 3차원 XYZ)
        self.normal_kps = np.random.uniform(low=0.2, high=0.8, size=(64, 33, 3)).astype(np.float32)

    def test_01_normal_tensor_shape_and_dimension(self):
        """[항목 1] (64, 33, 3) 정상 텐서에 대한 38D 피처 추출 및 차원/타입 일치 검증"""
        features = extract_stiff_d1_features(self.normal_kps, view_name="view3")

        # 1. 반환 타입 및 1차원 배열 검증
        self.assertIsInstance(features, np.ndarray, "반환 객체는 np.ndarray여야 합니다.")
        self.assertEqual(features.ndim, 1, "피처 벡터는 1차원이어야 합니다.")

        # 2. 38차원 정확성 검증
        self.assertEqual(features.shape, (38,), f"피처 차원은 정확히 (38,)이어야 하나 {features.shape}가 반환되었습니다.")
        self.assertEqual(len(FEATURE_NAMES_38_STIFF_D1), 38, "피처 명칭 리스트 길이가 38이어야 합니다.")
        self.assertEqual(features.dtype, np.float32, "피처 벡터의 dtype은 np.float32여야 합니다.")

        # 3. 결측치(NaN/Inf) 전무성 검증
        self.assertFalse(np.isnan(features).any(), "정상 텐서 추출 결과에 NaN이 포함되어 있습니다.")
        self.assertFalse(np.isinf(features).any(), "정상 텐서 추출 결과에 Inf가 포함되어 있습니다.")

    def test_02_real_cache_sample_extraction(self):
        """[항목 1 확장] 실제 전처리 캐시 디렉토리 내 npz 데이터셋 단일 샘플 피처 추출 검증"""
        npz_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
        if not npz_files:
            self.skipTest(f"캐시 디렉토리에 테스트용 npz 파일이 존재하지 않습니다: {CACHE_DIR}")

        sample_path = npz_files[0]
        with np.load(sample_path, allow_pickle=True) as data:
            self.assertIn('kps_33_3d_64f', data, "npz 파일 내 'kps_33_3d_64f' 키가 존재해야 합니다.")
            real_kps = data['kps_33_3d_64f']

        self.assertEqual(real_kps.shape, (64, 33, 3), f"실제 관절 데이터 shape가 (64, 33, 3)이어야 합니다: {real_kps.shape}")

        real_features = extract_stiff_d1_features(real_kps, view_name="view3")
        self.assertEqual(real_features.shape, (38,), "실제 데이터 추출 피처 차원이 38D여야 합니다.")
        self.assertFalse(np.isnan(real_features).any(), "실제 데이터 추출 피처에 NaN이 존재하지 않아야 합니다.")
        self.assertFalse(np.isinf(real_features).any(), "실제 데이터 추출 피처에 Inf가 존재하지 않아야 합니다.")

    def test_03_zero_array_safety(self):
        """[항목 2] All-zero 입력 텐서 시 제로 디비전 방어 및 NaN/Inf 미발생 검증"""
        zero_kps = np.zeros((64, 33, 3), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = extract_stiff_d1_features(zero_kps, view_name="view3")

        self.assertEqual(features.shape, (38,), "All-zero 입력 시에도 (38,) 차원이 반환되어야 합니다.")
        self.assertFalse(np.isnan(features).any(), "All-zero 입력 시 NaN이 발생해서는 안 됩니다.")
        self.assertFalse(np.isinf(features).any(), "All-zero 입력 시 Inf가 발생해서는 안 됩니다.")

    def test_04_all_nan_array_safety(self):
        """[항목 2] All-NaN 입력 텐서 시 np.nan_to_num을 통한 안전 방어 및 NaN 제거 검증"""
        nan_kps = np.full((64, 33, 3), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = extract_stiff_d1_features(nan_kps, view_name="view4")

        self.assertEqual(features.shape, (38,), "All-NaN 입력 시에도 (38,) 차원이 반환되어야 합니다.")
        self.assertFalse(np.isnan(features).any(), "All-NaN 입력 후 결과 피처에 NaN이 남아있지 않아야 합니다.")
        self.assertFalse(np.isinf(features).any(), "All-NaN 입력 후 결과 피처에 Inf가 남아있지 않아야 합니다.")

    def test_05_partial_nan_array_safety(self):
        """[항목 2] 특정 관절 및 특정 프레임 결측(Partial NaN) 시의 안전 방어 검증"""
        partial_nan_kps = self.normal_kps.copy()
        # 무릎, 발목 결측 모의 (오클루전 상황)
        partial_nan_kps[:, [25, 26, 27, 28], :] = np.nan
        # 셋업 시작 프레임 일부 결측
        partial_nan_kps[:5, :, :] = np.nan
        # 최하사점 부근 결측
        partial_nan_kps[30:35, :, :] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = extract_stiff_d1_features(partial_nan_kps, view_name="view2")

        self.assertEqual(features.shape, (38,), "Partial NaN 입력 시에도 (38,) 형상이 유지되어야 합니다.")
        self.assertFalse(np.isnan(features).any(), "Partial NaN 입력 후 결과에 NaN이 없어야 합니다.")
        self.assertFalse(np.isinf(features).any(), "Partial NaN 입력 후 결과에 Inf가 없어야 합니다.")

    def test_06_inf_and_negative_inf_safety(self):
        """[항목 2] 양의 무한대(+Inf) 및 음의 무한대(-Inf) 입력 시 오버플로우 방어 검증"""
        inf_kps = np.full((64, 33, 3), np.inf, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features_pos_inf = extract_stiff_d1_features(inf_kps, view_name="view5")

        self.assertEqual(features_pos_inf.shape, (38,))
        self.assertFalse(np.isnan(features_pos_inf).any(), "+Inf 입력 시 NaN 미발생 보장")
        self.assertFalse(np.isinf(features_pos_inf).any(), "+Inf 입력 시 Inf 미발생 보장")

        # 음의 무한대 및 혼합 무한대
        mixed_inf_kps = self.normal_kps.copy()
        mixed_inf_kps[::2, :, :] = np.inf
        mixed_inf_kps[1::2, :, :] = -np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features_mixed_inf = extract_stiff_d1_features(mixed_inf_kps, view_name="view1")

        self.assertEqual(features_mixed_inf.shape, (38,))
        self.assertFalse(np.isnan(features_mixed_inf).any(), "혼합 무한대 시 NaN 미발생 보장")
        self.assertFalse(np.isinf(features_mixed_inf).any(), "혼합 무한대 시 Inf 미발생 보장")

    def test_07_extreme_scale_and_static_sequence(self):
        """[항목 2] 극단적 스케일(1e8, 1e-8) 및 정적(움직임 없음, 수직변위 0) 시퀀스 제로-디비전 방어 검증"""
        # 1. 매우 큰 스케일
        huge_kps = self.normal_kps * 1e8
        features_huge = extract_stiff_d1_features(huge_kps, view_name="view3")
        self.assertEqual(features_huge.shape, (38,))
        self.assertFalse(np.isnan(features_huge).any())
        self.assertFalse(np.isinf(features_huge).any())

        # 2. 매우 미세한 스케일
        tiny_kps = self.normal_kps * 1e-8
        features_tiny = extract_stiff_d1_features(tiny_kps, view_name="view3")
        self.assertEqual(features_tiny.shape, (38,))
        self.assertFalse(np.isnan(features_tiny).any())
        self.assertFalse(np.isinf(features_tiny).any())

        # 3. 64프레임 동안 완전히 정지한 정적 시퀀스 (수직 변위 = 0, nadir_idx = 0)
        static_kps = np.ones((64, 33, 3), dtype=np.float32) * 0.5
        features_static = extract_stiff_d1_features(static_kps, view_name="view3")
        self.assertEqual(features_static.shape, (38,))
        self.assertFalse(np.isnan(features_static).any())
        self.assertFalse(np.isinf(features_static).any())

    def test_08_five_views_full_coverage(self):
        """[항목 3] 공식 5대 시점 전수(view1~view5)별 피처 추출 및 시점 지시자 정합성 검증"""
        expected_view_specs = {
            'view5': {'alias': '시점 A (우측 후방 45°)', 'code': 0.0, 'frontal': 0.0, 'rear': 1.0, 'oblique': 1.0},
            'view2': {'alias': '시점 B (우측 전방 45°)', 'code': 1.0, 'frontal': 0.0, 'rear': 0.0, 'oblique': 1.0},
            'view3': {'alias': '시점 C (정면 0°)',       'code': 2.0, 'frontal': 1.0, 'rear': 0.0, 'oblique': 0.0},
            'view4': {'alias': '시점 D (좌측 전방 45°)', 'code': 3.0, 'frontal': 0.0, 'rear': 0.0, 'oblique': 1.0},
            'view1': {'alias': '시점 E (좌측 후방 45°)', 'code': 4.0, 'frontal': 0.0, 'rear': 1.0, 'oblique': 1.0},
        }

        for view_code, spec in expected_view_specs.items():
            features = extract_stiff_d1_features(self.normal_kps, view_name=view_code)

            # 1. 기본 규격 검증
            self.assertEqual(features.shape, (38,), f"{view_code} 추출 피처 차원은 (38,)이어야 합니다.")
            self.assertFalse(np.isnan(features).any(), f"{view_code} 피처에 NaN이 없어야 합니다.")
            self.assertFalse(np.isinf(features).any(), f"{view_code} 피처에 Inf가 없어야 합니다.")

            # 2. Group 5 & 6 시점 지시자 인덱스 정합성 검증
            # [23]: stiff_d1_view_code
            # [24]: stiff_d1_is_frontal
            # [25]: stiff_d1_is_rear
            # [26]: stiff_d1_gate_is_oblique
            self.assertAlmostEqual(features[23], spec['code'], places=4,
                                   msg=f"{view_code} view_code 불일치 (기대값: {spec['code']}, 실제값: {features[23]})")
            self.assertAlmostEqual(features[24], spec['frontal'], places=4,
                                   msg=f"{view_code} is_frontal 불일치 (기대값: {spec['frontal']}, 실제값: {features[24]})")
            self.assertAlmostEqual(features[25], spec['rear'], places=4,
                                   msg=f"{view_code} is_rear 불일치 (기대값: {spec['rear']}, 실제값: {features[25]})")
            self.assertAlmostEqual(features[26], spec['oblique'], places=4,
                                   msg=f"{view_code} gate_is_oblique 불일치 (기대값: {spec['oblique']}, 실제값: {features[26]})")

            # 3. 정면 뷰(view3) 특화 마스킹 검증: is_oblique == 0일 때 f27(SEK 사잇각), f30(Z-diff)는 0.0으로 마스킹되어야 함
            if view_code == 'view3':
                self.assertAlmostEqual(features[27], 0.0, places=4, msg="정면 뷰에서는 f27 SEK 각도가 0.0으로 마스킹되어야 합니다.")
                self.assertAlmostEqual(features[30], 0.0, places=4, msg="정면 뷰에서는 f30 Z-diff가 0.0으로 마스킹되어야 합니다.")

    def test_09_view_alias_map_integrity(self):
        """[항목 3 확장] AGENTS.md 표준 시점 명칭 및 VIEW_ALIAS_MAP, VIEW_CODE_MAP 일치성 검증"""
        required_views = {'view1', 'view2', 'view3', 'view4', 'view5'}
        self.assertEqual(set(VIEW_ALIAS_MAP.keys()), required_views, "VIEW_ALIAS_MAP은 5대 시점 전수를 포함해야 합니다.")
        self.assertEqual(set(VIEW_CODE_MAP.keys()), required_views, "VIEW_CODE_MAP은 5대 시점 전수를 포함해야 합니다.")

        # 공식 명칭 텍스트 검증
        self.assertIn("시점 A", VIEW_ALIAS_MAP['view5'])
        self.assertIn("시점 B", VIEW_ALIAS_MAP['view2'])
        self.assertIn("시점 C", VIEW_ALIAS_MAP['view3'])
        self.assertIn("시점 D", VIEW_ALIAS_MAP['view4'])
        self.assertIn("시점 E", VIEW_ALIAS_MAP['view1'])


class MockClassifier:
    """통합 추론 단위 테스트를 위한 경량 Mock 분류기"""
    def __init__(self, fixed_prob: float = 0.5):
        self.fixed_prob = fixed_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        prob_1 = np.full((n_samples, 1), self.fixed_prob, dtype=np.float32)
        prob_0 = 1.0 - prob_1
        return np.hstack([prob_0, prob_1])


class TestStiffD1InferencePipeline(unittest.TestCase):
    """
    stiff_d1_inference.py 통합 추론 파이프라인 검증
    - 입력 텐서 shape 유효성 검사 (ValueError)
    - 3-State 판정 로직 무결성 (Clean=0, Defect=1, Abstain=-1)
    - 5대 공식 시점별 단일 시퀀스 추론 응답 인터페이스 검증
    - 가중치 번들 영속성 및 실제 파일 로드 I/O 검증
    """

    def setUp(self):
        np.random.seed(42)
        self.normal_kps = np.random.uniform(low=0.2, high=0.8, size=(64, 33, 3)).astype(np.float32)
        self.orig_cache = d1_inf_module._MODEL_CACHE

    def tearDown(self):
        # 모듈 캐시 원복
        d1_inf_module._MODEL_CACHE = self.orig_cache

    def test_10_input_shape_validation(self):
        """[통합 추론] 잘못된 입력 shape 입력 시 ValueError 발생 검증"""
        # 프레임 수 부족 (32, 33, 3)
        with self.assertRaises(ValueError):
            infer_stiff_d1_sequence(np.zeros((32, 33, 3), dtype=np.float32))

        # 관절 수 불일치 (64, 25, 3)
        with self.assertRaises(ValueError):
            infer_stiff_d1_sequence(np.zeros((64, 25, 3), dtype=np.float32))

        # 2D 좌표만 입력된 경우 (64, 33, 2)
        with self.assertRaises(ValueError):
            infer_stiff_d1_sequence(np.zeros((64, 33, 2), dtype=np.float32))

    def _setup_mock_bundle(self, prob: float, tau_clean: float = 0.40, tau_defect: float = 0.60):
        """모의 모델 번들 등록 헬퍼"""
        mock_bundle = {
            'model_name': 'mock_stiff_d1_bundle',
            'feature_dim': 38,
            'lgb_model': MockClassifier(fixed_prob=prob),
            'cb_model': MockClassifier(fixed_prob=prob),
            'weights': (0.55, 0.45),
            'optimal_3state': {
                'tau_clean': tau_clean,
                'tau_defect': tau_defect,
                'coverage': 92.5
            }
        }
        d1_inf_module._MODEL_CACHE = mock_bundle

    def test_11_three_state_decision_logic(self):
        """[통합 추론] 3-State Selective Gating 판정 로직(정상, 결함, 판정보류) 분기 무결성 검증"""
        # Case 1: prob <= tau_clean (정상 판정: prediction = 0, is_decided = True)
        self._setup_mock_bundle(prob=0.25, tau_clean=0.40, tau_defect=0.60)
        res_clean = infer_stiff_d1_sequence(self.normal_kps, view_name="view4")
        self.assertEqual(res_clean['prediction'], 0, "확률 0.25는 정상(0)으로 판정되어야 합니다.")
        self.assertTrue(res_clean['is_decided'], "정상 판정 시 is_decided는 True여야 합니다.")
        self.assertIn("정상", res_clean['decision_label'])

        # Case 2: prob >= tau_defect (결함 판정: prediction = 1, is_decided = True)
        self._setup_mock_bundle(prob=0.75, tau_clean=0.40, tau_defect=0.60)
        res_defect = infer_stiff_d1_sequence(self.normal_kps, view_name="view4")
        self.assertEqual(res_defect['prediction'], 1, "확률 0.75는 결함(1)으로 판정되어야 합니다.")
        self.assertTrue(res_defect['is_decided'], "결함 판정 시 is_decided는 True여야 합니다.")
        self.assertIn("결함", res_defect['decision_label'])

        # Case 3: tau_clean < prob < tau_defect (판정 보류: prediction = -1, is_decided = False)
        self._setup_mock_bundle(prob=0.50, tau_clean=0.40, tau_defect=0.60)
        res_abstain = infer_stiff_d1_sequence(self.normal_kps, view_name="view4")
        self.assertEqual(res_abstain['prediction'], -1, "경계구간 확률 0.50은 판정보류(-1)여야 합니다.")
        self.assertFalse(res_abstain['is_decided'], "보류 판정 시 is_decided는 False여야 합니다.")
        self.assertIn("보류", res_abstain['decision_label'])

    def test_12_five_views_inference_response_schema(self):
        """[통합 추론] 5대 시점 전수 추론 호출 및 반환 규격 스키마 전수 검증"""
        self._setup_mock_bundle(prob=0.30, tau_clean=0.40, tau_defect=0.60)

        required_keys = {
            'defect_prob', 'prediction', 'is_decided', 'decision_label',
            'view_name', 'view_alias', 'tau_clean', 'tau_defect', 'features_dim'
        }

        for vcode in ['view1', 'view2', 'view3', 'view4', 'view5']:
            res = infer_stiff_d1_sequence(self.normal_kps, view_name=vcode)

            # 1. 필수 키 존재 검증
            self.assertTrue(required_keys.issubset(res.keys()), f"{vcode} 응답에 필수 키 누락")

            # 2. 필드 타입 및 값 범위 검증
            self.assertIsInstance(res['defect_prob'], float)
            self.assertTrue(0.0 <= res['defect_prob'] <= 1.0)
            self.assertIn(res['prediction'], [0, 1, -1])
            self.assertIsInstance(res['is_decided'], bool)
            self.assertIsInstance(res['decision_label'], str)
            self.assertEqual(res['view_name'], vcode)
            self.assertEqual(res['view_alias'], VIEW_ALIAS_MAP[vcode])
            self.assertEqual(res['features_dim'], 38)

    def test_13_weights_bundle_persistence_and_inference(self):
        """[통합 추론] 가중치 번들 I/O 및 앙상블 로드 후 추론 검증 (실 가중치 또는 검증용 번들)"""
        d1_inf_module._MODEL_CACHE = None

        if os.path.exists(WEIGHTS_PATH):
            target_weights_path = WEIGHTS_PATH
            bundle = load_model_bundle(target_weights_path)
        else:
            # 실 가중치 생성 전 단계일 경우, 독립 번들 직렬화/역직렬화 I/O 파이프라인 무결성 검증
            mock_bundle = {
                'model_name': 'verified_stiff_d1_bundle',
                'feature_dim': 38,
                'lgb_model': MockClassifier(fixed_prob=0.35),
                'cb_model': MockClassifier(fixed_prob=0.45),
                'weights': (0.55, 0.45),
                'optimal_3state': {
                    'tau_clean': 0.44,
                    'tau_defect': 0.56,
                    'coverage': 91.8
                }
            }
            temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            target_weights_path = temp_file.name
            temp_file.close()
            with open(target_weights_path, 'wb') as fp:
                pickle.dump(mock_bundle, fp)

            bundle = load_model_bundle(target_weights_path)
            # 임시 파일 정리 등록
            self.addCleanup(lambda: os.path.exists(target_weights_path) and os.remove(target_weights_path))

        self.assertIsNotNone(bundle)
        self.assertIn('lgb_model', bundle)
        self.assertIn('cb_model', bundle)
        self.assertEqual(bundle['feature_dim'], 38)

        # 5대 시점 전수 추론 실행
        for vcode in ['view1', 'view2', 'view3', 'view4', 'view5']:
            res = infer_stiff_d1_sequence(self.normal_kps, view_name=vcode)
            self.assertEqual(res['features_dim'], 38)
            self.assertIn(res['prediction'], [0, 1, -1])
            self.assertIn('view_alias', res)


def run_comprehensive_inspection() -> bool:
    """콘솔 실행용 종합 무결성 감사 리포트 출력 함수"""
    print("\n" + "=" * 88)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 d1 피처 & 추론 QA 검증 스위트")
    print("=" * 88)

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestStiffD1FeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestStiffD1InferencePipeline))

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "-" * 88)
    print("  [QA 검증 최종 요약]")
    print(f"  - 실행 테스트 수   : {result.testsRun}개")
    print(f"  - 성공(Passed)     : {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}개")
    print(f"  - 실패(Failures)   : {len(result.failures)}개")
    print(f"  - 오류(Errors)     : {len(result.errors)}개")
    print(f"  - 스킵(Skipped)    : {len(result.skipped)}개")
    print(f"  - 소요 시간        : {elapsed:.3f}초")
    print(f"  - 최종 판정        : {'★ 100% 무결성 검증 통과 (PASSED)' if result.wasSuccessful() else '❌ 검증 실패 (FAILED)'}")
    print("=" * 88 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_inspection()
    sys.exit(0 if success else 1)

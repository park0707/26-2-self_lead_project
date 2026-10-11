# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for lunge_c5_features_v2.py
Purpose: v2 32차원 도메인 피처 추출기 무결성 검증
"""

import sys
import unittest
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_v2 import (
    extract_lunge_c5_v2_features,
    extract_lunge_c5_v2_feature_vector,
    LUNGE_C5_V2_FEATURE_NAMES,
    compute_vspa_torso_pitch_series,
    detect_lead_leg,
    find_inflection_point
)


class TestLungeC5FeaturesV2(unittest.TestCase):
    def setUp(self):
        # 합성 64프레임 런지 랜드마크 생성 (T=64, 33, 3)
        self.kps = np.zeros((64, 33, 3), dtype=np.float32)
        # 골반 (좌 23, 우 24)
        self.kps[:, 23] = np.array([-0.15, 0.0, 0.0])
        self.kps[:, 24] = np.array([0.15, 0.0, 0.0])
        # 어깨 (좌 11, 우 12)
        self.kps[:, 11] = np.array([-0.20, -0.5, 0.0])
        self.kps[:, 12] = np.array([0.20, -0.5, 0.0])
        # 무릎 (좌 25, 우 26) & 발목 (좌 27, 우 28)
        self.kps[:, 25] = np.array([-0.15, 0.4, 0.2])
        self.kps[:, 26] = np.array([0.15, 0.4, -0.3])
        self.kps[:, 27] = np.array([-0.15, 0.8, 0.3])
        self.kps[:, 28] = np.array([0.15, 0.8, -0.5])
        # 귀 (좌 7, 우 8)
        self.kps[:, 7] = np.array([-0.10, -0.7, 0.0])
        self.kps[:, 8] = np.array([0.10, -0.7, 0.0])

        # 하강 궤적 모사 (중간 32프레임이 최저점)
        for t in range(64):
            dip = 0.3 * np.sin(np.pi * t / 63.0)
            self.kps[t, 23, 1] += dip
            self.kps[t, 24, 1] += dip
            self.kps[t, 11, 1] += dip
            self.kps[t, 12, 1] += dip
            # 상체 약간 전경 (t=32에서 숙임)
            lean = 0.15 * np.sin(np.pi * t / 63.0)
            self.kps[t, 11, 2] += lean
            self.kps[t, 12, 2] += lean

    def test_feature_count_and_names(self):
        """32차원 규격 및 명칭 일치 검증"""
        self.assertEqual(len(LUNGE_C5_V2_FEATURE_NAMES), 32)
        f_dict = extract_lunge_c5_v2_features(self.kps)
        self.assertEqual(len(f_dict), 32)
        for name in LUNGE_C5_V2_FEATURE_NAMES:
            self.assertIn(name, f_dict)

    def test_feature_vector_shape_and_no_nan(self):
        """1D float32 32차원 벡터 반환 및 NaN/Inf 무결성 검증"""
        vec = extract_lunge_c5_v2_feature_vector(self.kps)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (32,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertFalse(np.isnan(vec).any(), "NaN 값이 검출되었습니다.")
        self.assertFalse(np.isinf(vec).any(), "Inf 값이 검출되었습니다.")

    def test_vspa_pitch_series(self):
        """VSPA 기반 시상면 피치각 시계열 계산 검증"""
        pitches = compute_vspa_torso_pitch_series(self.kps)
        self.assertEqual(pitches.shape, (64,))
        self.assertFalse(np.isnan(pitches).any())

    def test_lead_leg_detection(self):
        """앞다리 식별 검증"""
        lead = detect_lead_leg(self.kps)
        self.assertEqual(lead, 'right')

    def test_inflection_point(self):
        """최하사점 탐색 검증"""
        t_bot, w_start, w_end = find_inflection_point(self.kps)
        self.assertTrue(25 <= t_bot <= 38)
        self.assertTrue(0 <= w_start < t_bot < w_end <= 64)


if __name__ == '__main__':
    unittest.main()

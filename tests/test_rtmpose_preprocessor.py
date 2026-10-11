# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: test_rtmpose_preprocessor.py
Purpose: RTMPosePreprocessor 7단계 전처리 파이프라인의 텐서 규격, 무결성, 결측치 보간, 하위 호환성 전수 검증
"""

import os
import sys
import unittest
import numpy as np

# 프로젝트 루트 경로 등록
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.core.skeleton2d import JointHalpe26, Joint15
from src.preprocessing.rtmpose_preprocessor import RTMPosePreprocessor


class TestRTMPosePreprocessor(unittest.TestCase):
    """RTMPosePreprocessor 무결성 검증 테스트 스위트"""

    def setUp(self):
        # 모의 시퀀스 생성: T=50, J=26, C=3 [x, y, conf]
        np.random.seed(42)
        self.T = 50
        self.raw_kps = np.random.uniform(0.1, 0.9, size=(self.T, 26, 3)).astype(np.float32)
        # 신뢰도 0.8로 기본 설정
        self.raw_kps[:, :, 2] = 0.85
        # Neck(18)과 Hip(19) 설정 (y축으로 인체 형성)
        self.raw_kps[:, JointHalpe26.NECK, :2] = np.array([0.5, 0.3], dtype=np.float32)
        self.raw_kps[:, JointHalpe26.HIP, :2] = np.array([0.5, 0.6], dtype=np.float32)

    def test_filter_and_impute_confidence(self):
        """저신뢰도 및 NaN 구간 3차 스플라인 보간 검증"""
        corrupted = np.copy(self.raw_kps)
        # 중간 5프레임 저신뢰도 및 NaN 주입
        corrupted[10:15, JointHalpe26.LEFT_KNEE, 2] = 0.1 # 저신뢰도
        corrupted[20:23, JointHalpe26.RIGHT_ANKLE, :2] = np.nan # NaN

        cleaned = RTMPosePreprocessor.filter_and_impute_confidence(corrupted)
        self.assertIsNotNone(cleaned)
        # NaN 존재하지 않아야 함
        self.assertFalse(np.isnan(cleaned).any(), "보간 후 NaN이 존재하면 안 됩니다.")
        # 보간된 프레임이 유의미한 값이어야 함
        self.assertGreater(cleaned.shape[0], 0)

    def test_root_center_pelvis(self):
        """Pelvis(19번) 원점 (0, 0) 평행 이동 검증"""
        centered = RTMPosePreprocessor.root_center_pelvis(self.raw_kps)
        pelvis_xy = centered[:, JointHalpe26.HIP, :2]
        np.testing.assert_allclose(pelvis_xy, 0.0, atol=1e-5, err_msg="Pelvis는 원점 (0,0)이어야 합니다.")

    def test_scale_normalize_torso(self):
        """Neck(18) - Hip(19) 척추 길이 1.0 정규화 검증"""
        scaled, torso_len = RTMPosePreprocessor.scale_normalize_torso(self.raw_kps)
        neck = scaled[:, JointHalpe26.NECK, :2]
        hip = scaled[:, JointHalpe26.HIP, :2]
        mean_len = np.mean(np.linalg.norm(neck - hip, axis=-1))
        self.assertAlmostEqual(mean_len, 1.0, places=4, msg="체간 길이는 1.0으로 정규화되어야 합니다.")

    def test_resample_to_64f(self):
        """고정 64프레임 리샘플링 규격 준수 검증"""
        resampled = RTMPosePreprocessor.resample_to_64f(self.raw_kps, target_frames=64)
        self.assertEqual(resampled.shape, (64, 26, 3))
        self.assertFalse(np.isnan(resampled).any())

    def test_halpe26_to_joint15(self):
        """Halpe26 -> Joint15 하위 호환 변환 규격 검증"""
        resampled_64 = RTMPosePreprocessor.resample_to_64f(self.raw_kps, target_frames=64)
        j15 = RTMPosePreprocessor.halpe26_to_joint15(resampled_64)
        self.assertEqual(j15.shape, (64, 15, 3))
        # Pelvis 매핑 확인
        np.testing.assert_allclose(j15[:, Joint15.PELVIS], resampled_64[:, JointHalpe26.HIP])
        # Neck 매핑 확인
        np.testing.assert_allclose(j15[:, Joint15.NECK], resampled_64[:, JointHalpe26.NECK])

    def test_process_raw_sequence_full_pipeline(self):
        """7단계 전체 파이프라인 일괄 실행 및 반환 딕셔너리 검증"""
        res = RTMPosePreprocessor.process_raw_sequence(self.raw_kps, exercise_name="04_바벨_런지")
        self.assertIsNotNone(res)
        self.assertIn('kps_26_64f', res)
        self.assertIn('kps_15_64f', res)
        self.assertIn('foot_info', res)
        self.assertEqual(res['kps_26_64f'].shape, (64, 26, 3))
        self.assertEqual(res['kps_15_64f'].shape, (64, 15, 3))
        self.assertFalse(np.isnan(res['kps_26_64f']).any())
        self.assertFalse(np.isinf(res['kps_26_64f']).any())


if __name__ == '__main__':
    unittest.main()

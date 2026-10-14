# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_tier1_active_period_gater.py
Purpose: [Harness-Engineering] Tier 1 ActivePeriodGater 단위 테스트 및 통합 검증
         1. 초단기 시퀀스 (< 2초) 폴백 방어 검증
         2. 결측치 및 비정상 스켈레톤 내결함성(Fault Tolerance) 검증
         3. 기하학적 관절 각도(Knee/Hip Angle) 계산 정확성 검증
         4. 정지 상태 셋업 미세 진동(Noise Wobble) 100% 필터링 검증
         5. 실제 5회 바벨 스쿼트 일반화 영상 5회 완벽 검출 및 무결성 검증
"""

import os
import sys
import unittest
import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.core.skeleton2d import Joint15
from src.web.pipeline.active_period_gater import ActivePeriodGater


class TestActivePeriodGater(unittest.TestCase):
    """Tier 1 ActivePeriodGater 하네스 단위 테스트 스위트"""

    def setUp(self):
        self.fps = 30.0

    def test_short_sequence_fallback(self):
        """2초 미만 초단기 시퀀스 입력 시 안전 폴백 검증"""
        short_kps = np.zeros((20, 15, 2), dtype=np.float32)
        start_f, end_f, reps = ActivePeriodGater.filter_and_segment_repetitions(short_kps, fps=self.fps)
        self.assertEqual(start_f, 0)
        self.assertEqual(end_f, 19)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0], (0, 19, 10))

    def test_joint_angle_computation_geometry(self):
        """기하학적 관절 각도 계산 공식 검증 (직선 180도, 직각 90도)"""
        kps = np.zeros((1, 15, 2), dtype=np.float32)
        # 고관절 (0, 0), 무릎 (0, 100), 발목 (0, 200) -> 일직선 180도
        kps[0, Joint15.LEFT_HIP] = [0.0, 0.0]
        kps[0, Joint15.LEFT_KNEE] = [0.0, 100.0]
        kps[0, Joint15.LEFT_ANKLE] = [0.0, 200.0]
        # 우측 다리: 고관절 (100, 0), 무릎 (100, 100), 발목 (200, 100) -> 90도 직각
        kps[0, Joint15.RIGHT_HIP] = [100.0, 0.0]
        kps[0, Joint15.RIGHT_KNEE] = [100.0, 100.0]
        kps[0, Joint15.RIGHT_ANKLE] = [200.0, 100.0]

        angles = ActivePeriodGater.compute_joint_angles(kps)
        self.assertAlmostEqual(float(angles["left_knee"][0]), 180.0, places=1)
        self.assertAlmostEqual(float(angles["right_knee"][0]), 90.0, places=1)

    def test_stationary_noise_filtering(self):
        """정지 서 있는 자세 중 3~5프레임의 미세 노이즈 버스트 필터링 검증"""
        T = 150  # 5초 분량
        kps = np.zeros((T, 15, 2), dtype=np.float32)
        # 직립 기립 자세 기본 배치
        for t in range(T):
            kps[t, Joint15.PELVIS] = [300.0, 400.0]
            kps[t, Joint15.NECK] = [300.0, 200.0]
            kps[t, Joint15.TORSO] = [300.0, 300.0]
            kps[t, Joint15.LEFT_HIP] = [280.0, 400.0]
            kps[t, Joint15.RIGHT_HIP] = [320.0, 400.0]
            kps[t, Joint15.LEFT_KNEE] = [280.0, 550.0]
            kps[t, Joint15.RIGHT_KNEE] = [320.0, 550.0]
            kps[t, Joint15.LEFT_ANKLE] = [280.0, 700.0]
            kps[t, Joint15.RIGHT_ANKLE] = [320.0, 700.0]

        # 30~33번 프레임에 4프레임짜리 일시적 노이즈 주입 (0.13초)
        for t in range(30, 34):
            kps[t, Joint15.LEFT_KNEE] = [350.0, 550.0]  # 순간적 굴곡

        start_f, end_f, reps = ActivePeriodGater.filter_and_segment_repetitions(kps, fps=self.fps)
        # 4프레임 노이즈는 min_burst_sec(0.35초 = 10프레임) 미만이므로 무시되고 폴백되어야 함
        self.assertEqual(len(reps), 1)
        self.assertEqual(start_f, 0)
        self.assertEqual(end_f, T - 1)

    def test_real_generalization_video_5reps(self):
        """실제 일반화 바벨 스쿼트 5회 영상에 대한 5회 전수 분할 및 언랙 필터링 검증"""
        cache_path = r"D:\data\일반화\test_video_landmarks_cache.npz"
        if not os.path.exists(cache_path):
            self.skipTest(f"캐시 파일이 존재하지 않아 스킵: {cache_path}")

        data = np.load(cache_path)
        kps_15 = data["kps_15_2d"]
        fps = float(data["fps"])

        start_f, end_f, reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps=fps)

        # 1. 셋업/언랙 구간(F0~F150)이 정확히 필터링되었는지 검증
        self.assertGreaterEqual(start_f, 120, "셋업 및 언랙 구간이 시작 지점 이전에 필터링되어야 합니다.")
        self.assertLessEqual(start_f, 165, "첫 번째 스쿼트의 시작 지점이 너무 늦게 잡히지 않아야 합니다.")

        # 2. 정확히 5회 반복 검출 검증
        self.assertEqual(len(reps), 5, f"기대 횟수 5회 != 실제 {len(reps)}회")

        # 3. 각 회차의 유효 시간 (1.5초 ~ 4.5초) 검증
        for idx, (s, e, b) in enumerate(reps, 1):
            dur = (e - s + 1) / fps
            self.assertGreaterEqual(dur, 1.5, f"Rep #{idx} 소요 시간이 너무 짧습니다: {dur:.2f}s")
            self.assertLessEqual(dur, 4.5, f"Rep #{idx} 소요 시간이 너무 깁니다: {dur:.2f}s")
            self.assertGreater(b, s, f"Rep #{idx} 최하사점은 시작점 이후여야 합니다.")
            self.assertLess(b, e, f"Rep #{idx} 최하사점은 종료점 이전이어야 합니다.")


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 멀티비전 키포인트 품질(KQS) 평가 및 재촬영 게이팅 하네스 테스트
(Test Suite: tests/test_kqs_gating_harness.py)
"""

import sys
import unittest
import numpy as np

# 프로젝트 루트 등록
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.core.feedback_engine import FeedbackEngine, RETAKE_PROMPT_CATALOG
from src.web.core.schemas import RepetitionDetail, DefectPrediction
from src.web.pipeline.stage2_engine import Stage2InferenceEngine


class TestKQSGatingHarness(unittest.TestCase):
    def setUp(self):
        self.fb_engine = FeedbackEngine.get_instance()
        self.stage2_engine = Stage2InferenceEngine.get_instance(prewarm=False)

    def test_01_normal_quality_pass(self):
        """정상 가시성(0.9+) 시퀀스: KQS >= 0.85, is_retake=False 통과 검증"""
        T = 64
        mp_vis = np.full((T, 33), 0.95, dtype=np.float32)
        rtm_conf = np.full((T, 26), 0.92, dtype=np.float32)

        res = self.fb_engine.evaluate_keypoint_quality(
            exercise_id=0,  # 바벨 스쿼트
            mp_vis_seq=mp_vis,
            rtm_conf_seq=rtm_conf,
            threshold=0.60
        )

        self.assertFalse(res['is_retake'], "정상 가시성에서는 재촬영이 요구되지 않아야 합니다.")
        self.assertGreaterEqual(res['kqs_score'], 0.85, "정상 가시성의 KQS는 0.85 이상이어야 합니다.")
        self.assertEqual(len(res['occluded_joints']), 0)
        self.assertIsNone(res['retake_code'])

    def test_02_knee_occlusion_gating_and_prompt(self):
        """슬관절(무릎) 가림 시: OCC_KNEE 트리거, 바벨 원판 언급 없는 교정 큐, 바이패스 검증"""
        T = 64
        mp_vis = np.full((T, 33), 0.90, dtype=np.float32)
        rtm_conf = np.full((T, 26), 0.90, dtype=np.float32)

        # 무릎 관절 가림 주입: MediaPipe 25, 26 / RTMPose 13, 14
        mp_vis[:, [25, 26]] = 0.10
        rtm_conf[:, [13, 14]] = 0.10

        res = self.fb_engine.evaluate_keypoint_quality(
            exercise_id=0,  # 바벨 스쿼트
            mp_vis_seq=mp_vis,
            rtm_conf_seq=rtm_conf,
            threshold=0.60
        )

        self.assertTrue(res['is_retake'], "무릎 가림 발생 시 is_retake는 True여야 합니다.")
        self.assertIn("슬관절(무릎)", res['occluded_joints'])
        self.assertEqual(res['retake_code'], "OCC_KNEE")
        self.assertIsNotNone(res['retake_cue'])

        # 사용자 명시 규칙 검증: 바벨 원판 언급이 없어야 함
        self.assertNotIn("바벨 원판", res['retake_cue'], "무릎 재촬영 큐에 '바벨 원판' 단어가 포함되어서는 안 됩니다.")
        self.assertIn("무릎 관절이 화면 또는 물체에 가려져", res['retake_cue'])

        # 우선순위 큐 연동 검증
        dummy_preds = [{'defect_code': 'c1', 'state': 1, 'probability': 0.95}]
        coaching = self.fb_engine.prioritize_rep_defects(
            exercise_id=0,
            defect_predictions=dummy_preds,
            view_code='view3',
            kqs_info=res
        )

        self.assertEqual(coaching['overall_status'], '판정 보류 (재촬영 권고)')
        self.assertFalse(coaching['is_clean'])
        self.assertTrue(coaching['is_retake_required'])
        self.assertIn("[재촬영 안내]", coaching['primary_action_cue'])
        self.assertIn("무릎", coaching['primary_action_cue'])

    def test_03_feet_occlusion_gating(self):
        """하지 족관절/발목 화각 이탈 시: OCC_FEET 트리거 및 발목 가이드 검증"""
        T = 64
        mp_vis = np.full((T, 33), 0.90, dtype=np.float32)
        # 발목/발끝 (27~32) 결손 주입
        mp_vis[:, [27, 28, 29, 30, 31, 32]] = 0.15

        res = self.fb_engine.evaluate_keypoint_quality(
            exercise_id=0,
            mp_vis_seq=mp_vis,
            threshold=0.60
        )

        self.assertTrue(res['is_retake'])
        self.assertIn("하지 족관절(발목/발바닥)", res['occluded_joints'])
        self.assertEqual(res['retake_code'], "OCC_FEET")
        self.assertIn("화면 하단 밖으로 벗어났습니다", res['retake_cue'])

    def test_04_consecutive_streak_penalty(self):
        """시간축 연속 결손(Streak Penalty) 동작 검증"""
        T = 64
        mp_vis = np.full((T, 33), 0.85, dtype=np.float32)
        # 중간 24프레임(약 0.8초) 동안 전 관절 0.1로 드롭
        mp_vis[20:44, :] = 0.10

        res = self.fb_engine.evaluate_keypoint_quality(
            exercise_id=0,
            mp_vis_seq=mp_vis,
            threshold=0.60
        )

        self.assertTrue(res['is_retake'])
        self.assertLess(res['kqs_score'], 0.60, "연속 결손 페널티로 KQS 점수가 대폭 감점되어야 합니다.")

    def test_05_stage2_engine_e2e_integration(self):
        """Stage2InferenceEngine diagnose E2E 바이패스 및 정상 분기 검증"""
        T = 64
        dummy_kps = np.zeros((T, 33, 3), dtype=np.float32)

        # 1. 무릎 가림 조건
        vis_occluded = np.full((T, 33), 0.95, dtype=np.float32)
        vis_occluded[:, [25, 26]] = 0.05

        diag_occluded = self.stage2_engine.diagnose(
            exercise=0,
            keypoints_seq=dummy_kps,
            camera_view="시점 C",
            mp_vis_seq=vis_occluded
        )

        self.assertFalse(diag_occluded['is_clean'])
        self.assertTrue(diag_occluded['coaching']['is_retake_required'])
        self.assertEqual(diag_occluded['coaching']['overall_status'], '판정 보류 (재촬영 권고)')
        # 결함들이 보류 상태(state=2)로 바이패스되었는지 확인
        for d in diag_occluded['defects']:
            self.assertEqual(d['state'], 2, "가림 발생 시 세부 결함은 2(보류) 상태여야 합니다.")

    def test_06_schema_compatibility(self):
        """RepetitionDetail Pydantic 스키마 직렬화 호환성 검증"""
        rep = RepetitionDetail(
            rep_index=1,
            start_frame=0,
            end_frame=64,
            start_sec=0.0,
            end_sec=2.13,
            duration_sec=2.13,
            is_clean=False,
            defects=[],
            primary_feedback="[재촬영 안내] 무릎 관절이 화면 또는 물체에 가려져 정밀 분석이 어렵습니다.",
            kqs_score=0.4521,
            is_retake_required=True,
            occluded_joints=["슬관절(무릎)"],
            retake_cue="무릎 관절이 화면 또는 물체에 가려져 정밀 분석이 어렵습니다."
        )

        d = rep.model_dump()
        self.assertEqual(d['kqs_score'], 0.4521)
        self.assertTrue(d['is_retake_required'])
        self.assertIn("슬관절(무릎)", d['occluded_joints'])
        self.assertIn("물체에 가려져", d['retake_cue'])


if __name__ == '__main__':
    unittest.main()

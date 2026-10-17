# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 보류(Caution/Pending) 케이스 전용 피드백 및 환경/모호성 팁 하네스 테스트
(Test Suite: tests/test_pending_feedback_harness.py)
"""

import sys
import unittest
import numpy as np

# 프로젝트 루트 등록
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.core.feedback_engine import (
    FeedbackEngine, FEEDBACK_CATALOG, EXERCISE_CLARITY_CATALOG
)
from src.web.core.schemas import RepetitionDetail, ExercisePrediction
from src.web.core.config import EXERCISE_CATALOG


class TestPendingFeedbackHarness(unittest.TestCase):
    def setUp(self):
        self.fb_engine = FeedbackEngine.get_instance()

    def test_01_all_33_defects_caution_cue_validity(self):
        """8대 운동 33종 전 결함의 State 2 주의 문구(caution_cue) 완성도 전수 검증"""
        self.assertEqual(len(FEEDBACK_CATALOG), 33)
        for code, item in FEEDBACK_CATALOG.items():
            self.assertTrue(hasattr(item, 'caution_cue'), f"{code}에 caution_cue 필드가 없습니다.")
            self.assertIsNotNone(item.caution_cue, f"{code} caution_cue가 None입니다.")
            self.assertGreater(len(item.caution_cue.strip()), 8, f"{code} caution_cue가 너무 짧습니다: {item.caution_cue}")
            # 주의 큐에 부적절한 단정 표현("틀렸습니다", "부상 위험")이 없는지 검증
            self.assertNotIn("부상 위험이 있습니다", item.caution_cue)

    def test_02_pure_state_2_caution_coaching(self):
        """State 1 결함 없이 State 2 주의 항목만 존재할 때: 소프트 예방 코칭 피드백 검증"""
        # 스쿼트 c1 (척추 중립 State 2)
        dummy_preds = [
            {'defect_code': 'c1', 'state': 2, 'probability': 0.52},
            {'defect_code': 'c2', 'state': 0, 'probability': 0.10},
            {'defect_code': 'c3', 'state': 0, 'probability': 0.15},
            {'defect_code': 'c4', 'state': 0, 'probability': 0.20},
        ]

        res = self.fb_engine.prioritize_rep_defects(
            exercise_id=0,
            defect_predictions=dummy_preds,
            view_code='view3'
        )

        self.assertEqual(res['overall_status'], '주의/보류')
        self.assertFalse(res['is_clean'])
        self.assertIn("[자세 주의]", res['primary_action_cue'])
        self.assertIn("요추 아치가 다소 불안정합니다", res['primary_action_cue'])
        self.assertIsNotNone(res['caution_cue'])
        self.assertIn("요추 아치", res['caution_cue'])

        # 2개 이상의 State 2 주의 항목이 있을 때 2순위 주의 큐 연계 검증
        dummy_preds_multi = [
            {'defect_code': 'c1', 'state': 2, 'probability': 0.55},
            {'defect_code': 'c3', 'state': 2, 'probability': 0.48},
        ]
        res_multi = self.fb_engine.prioritize_rep_defects(
            exercise_id=0,
            defect_predictions=dummy_preds_multi
        )
        self.assertEqual(res_multi['overall_status'], '주의/보류')
        self.assertTrue(len(res_multi['secondary_cues']) > 0)
        self.assertIn("[주의]", res_multi['secondary_cues'][0])
        self.assertIn("슬관절/외측 둔근", res_multi['secondary_cues'][0])

    def test_03_state_1_with_state_2_secondary(self):
        """1순위 State 1 결함 + 2순위 State 2 주의 항목 결합 시 1+1 코칭 검증"""
        dummy_preds = [
            {'defect_code': 'c1', 'state': 1, 'probability': 0.85},  # State 1 (결함)
            {'defect_code': 'c4', 'state': 2, 'probability': 0.45},  # State 2 (주의)
        ]

        res = self.fb_engine.prioritize_rep_defects(
            exercise_id=0,
            defect_predictions=dummy_preds
        )

        self.assertEqual(res['overall_status'], '결함 검출')
        # 최우선 큐는 State 1 결함
        self.assertIn("요추/척추 기립근", res['primary_action_cue'])
        # 2순위 보조 큐에 State 2 주의 항목 연계 확인
        self.assertTrue(len(res['secondary_cues']) > 0)
        self.assertIn("[주의] 족관절/하지 안정성", res['secondary_cues'][0])
        self.assertIn("발바닥 지지가 다소 불안정합니다", res['secondary_cues'][0])
        self.assertIsNotNone(res['caution_cue'])

    def test_04_marginal_kqs_environment_tip(self):
        """KQS 경계선 품질 구간(0.60~0.72) 환경 팁 생성 검증"""
        T = 64
        # 전반적으로 보통 가시성이며, 발목(27~32)이 다소 낮은 조건
        mp_vis = np.full((T, 33), 0.70, dtype=np.float32)
        mp_vis[:, [27, 28, 29, 30, 31, 32]] = 0.48

        kqs_res = self.fb_engine.evaluate_keypoint_quality(
            exercise_id=0,
            mp_vis_seq=mp_vis,
            threshold=0.60
        )

        self.assertFalse(kqs_res['is_retake'], "KQS 0.60 이상이므로 재촬영 대상은 아니어야 합니다.")
        self.assertGreaterEqual(kqs_res['kqs_score'], 0.60)
        self.assertLess(kqs_res['kqs_score'], 0.72)
        self.assertIsNotNone(kqs_res['environment_tip'])
        self.assertIn("[촬영 환경 팁]", kqs_res['environment_tip'])
        self.assertIn("스마트폰을 50cm 뒤로 물려", kqs_res['environment_tip'])

        # prioritize_rep_defects 연동 확인
        dummy_preds = [{'defect_code': 'c1', 'state': 0, 'probability': 0.1}]
        rep_out = self.fb_engine.prioritize_rep_defects(
            exercise_id=0,
            defect_predictions=dummy_preds,
            kqs_info=kqs_res
        )
        self.assertEqual(rep_out['environment_tip'], kqs_res['environment_tip'])

    def test_05_exercise_clarity_tip_pairs(self):
        """1단계 운동 종목 분류 모호성 안내 팁 검증"""
        # 스쿼트(0) vs 데드리프트(1) 확률 근소 차이
        tip_squat_dead = self.fb_engine.get_exercise_clarity_tip(
            top1_exercise_id=0,
            top2_exercise_id=1,
            confidence=0.55,
            prob_gap=0.10
        )
        self.assertIsNotNone(tip_squat_dead)
        self.assertIn("[동작 명확화 팁]", tip_squat_dead)
        self.assertIn("스쿼트와 데드리프트의 중간 형태", tip_squat_dead)

        # 데드리프트(1) vs 스티프데드(2)
        tip_dead_stiff = self.fb_engine.get_exercise_clarity_tip(
            top1_exercise_id=1,
            top2_exercise_id=2,
            confidence=0.58,
            prob_gap=0.12
        )
        self.assertIsNotNone(tip_dead_stiff)
        self.assertIn("일반 데드리프트와 스티프 데드리프트의 경계선", tip_dead_stiff)

        # 확신도가 충분히 높은 경우 (명확화 팁 불필요)
        tip_clear = self.fb_engine.get_exercise_clarity_tip(
            top1_exercise_id=0,
            top2_exercise_id=1,
            confidence=0.95,
            prob_gap=0.85
        )
        self.assertIsNone(tip_clear, "확신도가 높을 때는 명확화 팁을 남발하지 않아야 합니다.")

    def test_06_repetition_detail_schema_serialization(self):
        """RepetitionDetail Pydantic 직렬화 검증 (caution_cue 및 environment_tip)"""
        rep = RepetitionDetail(
            rep_index=2,
            start_frame=65,
            end_frame=130,
            start_sec=2.16,
            end_sec=4.33,
            duration_sec=2.17,
            is_clean=False,
            defects=[],
            primary_feedback="[자세 주의] 요추 아치가 다소 불안정합니다.",
            caution_cue="요추 아치가 다소 불안정합니다. 최하단 깊이를 살짝 줄이고 코어 텐션을 끝까지 유지하세요.",
            environment_tip="[촬영 환경 팁] 화면 하단 여백이 좁아 발목 접지 분석 신뢰도가 다소 낮습니다. 스마트폰을 50cm 뒤로 물려 전신을 확보하세요.",
            kqs_score=0.685,
            is_retake_required=False
        )

        dumped = rep.model_dump()
        self.assertEqual(dumped['rep_index'], 2)
        self.assertIn("요추 아치가 다소 불안정합니다", dumped['caution_cue'])
        self.assertIn("스마트폰을 50cm 뒤로 물려", dumped['environment_tip'])
        self.assertEqual(dumped['kqs_score'], 0.685)
        self.assertFalse(dumped['is_retake_required'])


if __name__ == '__main__':
    unittest.main()

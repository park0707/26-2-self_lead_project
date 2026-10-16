# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 3단계 자세 교정 피드백 엔진 단위 및 커버리지 하네스 테스트
(Test Suite: tests/test_stage3_feedback_engine.py)
"""

import sys
import time
import unittest
import numpy as np

# 프로젝트 루트 등록
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.core.feedback_engine import FeedbackEngine, FEEDBACK_CATALOG, CAUSAL_CHAIN_RULES
from src.web.core.config import EXERCISE_CATALOG


class TestStage3FeedbackEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FeedbackEngine.get_instance()

    def test_01_catalog_coverage_33_defects(self):
        """8대 운동 33종 전 결함 카탈로그 100% 등록 검증"""
        expected_total = sum(ex['defects_count'] for ex in EXERCISE_CATALOG.values())
        self.assertEqual(expected_total, 33, "전체 결함 수는 정확히 33개여야 합니다.")
        self.assertEqual(len(FEEDBACK_CATALOG), 33, "피드백 카탈로그에는 정확히 33개 결함이 등록되어야 합니다.")

        # 각 결함의 필수 필드 유효성 검증
        for code, item in FEEDBACK_CATALOG.items():
            self.assertEqual(code, item.defect_code.lower())
            self.assertTrue(len(item.defect_name) > 0)
            self.assertTrue(len(item.target_joint) > 0)
            self.assertTrue(0.0 <= item.risk_weight <= 1.0)
            self.assertTrue(len(item.action_cue) > 10, f"{code} action_cue가 너무 짧습니다.")
            self.assertTrue(len(item.biomechanical_reason) > 10, f"{code} biomechanical_reason이 너무 짧습니다.")
            self.assertTrue(len(item.caution_cue) > 5, f"{code} caution_cue가 누락되었습니다.")
            self.assertTrue(len(item.normal_cue) > 5, f"{code} normal_cue가 누락되었습니다.")
            self.assertTrue(len(item.optimal_view) > 0, f"{code} optimal_view가 누락되었습니다.")
            self.assertTrue(len(item.recommended_drill) > 0, f"{code} recommended_drill이 누락되었습니다.")

    def test_02_single_feedback_formatting_3states(self):
        """단일 결함 3-State 상태별 포맷팅 검증"""
        # State 1 (결함)
        fb_defect = self.engine.format_single_feedback("c1", state=1, prob=0.85)
        self.assertIn("[교정 조치]", fb_defect)
        self.assertIn("가슴을", fb_defect)

        # State 2 (주의/보류)
        fb_caution = self.engine.format_single_feedback("c1", state=2, prob=0.45)
        self.assertIn("[주의]", fb_caution)
        self.assertIn("요추 아치", fb_caution)

        # State 0 (정상)
        fb_clean = self.engine.format_single_feedback("c1", state=0, prob=0.05)
        self.assertIn("정상 동작입니다", fb_clean)

    def test_03_prioritize_rep_defects_and_priority_queue(self):
        """다중 결함 발생 시 생체역학 위험도 기반 우선순위 큐 검증"""
        # 스쿼트에서 c1 (척추 1.0)과 c2 (고개 0.4)가 동시 발생한 경우
        # (단, c1+c2는 연쇄 규칙이 있으므로 단독 우선순위 확인을 위해 c1과 c4 테스트)
        dummy_preds = [
            {'defect_code': 'c4', 'state': 1, 'probability': 0.70}, # 발바닥 (0.60 * 0.70 = 0.42)
            {'defect_code': 'c1', 'state': 1, 'probability': 0.85}, # 척추 (1.00 * 0.85 = 0.85)
        ]
        result = self.engine.prioritize_rep_defects(exercise_id=0, defect_predictions=dummy_preds, view_code='view3')
        self.assertFalse(result['is_clean'])
        self.assertEqual(result['overall_status'], '결함 검출')
        # 최우선 순위는 위험도가 가장 높은 요추/척추(c1)여야 함
        self.assertIn("요추/척추 기립근", result['primary_action_cue'])
        self.assertTrue(len(result['secondary_cues']) > 0)
        self.assertIn("족관절/하지 안정성", result['secondary_cues'][0])

    def test_04_causal_chaining_detection(self):
        """복합 결함 연쇄 인과관계 검출 및 합성 큐 검증"""
        # 스쿼트: c3(무릎 모임) + c4(발바닥 뜸) 동시 발생
        dummy_preds = [
            {'defect_code': 'c3', 'state': 1, 'probability': 0.90},
            {'defect_code': 'c4', 'state': 1, 'probability': 0.85}
        ]
        res = self.engine.prioritize_rep_defects(exercise_id=0, defect_predictions=dummy_preds)
        self.assertIsNotNone(res['causal_chain_detected'])
        self.assertIn("지면 접지력 붕괴로 인한 무릎 외반 연쇄", res['causal_chain_detected'])
        self.assertIn("[핵심 원인 교정]", res['primary_action_cue'])
        self.assertIn("발바닥 삼각점", res['primary_action_cue'])

    def test_05_multi_rep_set_summary(self):
        """다중 반복 세트 종합 맞춤 처방전 도출 검증"""
        rep_results = [
            {'rep_index': 1, 'is_clean': False, 'defects': [{'defect_code': 'c1', 'state': 1}, {'defect_code': 'c3', 'state': 0}]},
            {'rep_index': 2, 'is_clean': False, 'defects': [{'defect_code': 'c1', 'state': 1}, {'defect_code': 'c3', 'state': 0}]},
            {'rep_index': 3, 'is_clean': False, 'defects': [{'defect_code': 'c1', 'state': 1}, {'defect_code': 'c3', 'state': 0}]},
            {'rep_index': 4, 'is_clean': True, 'defects': [{'defect_code': 'c1', 'state': 0}, {'defect_code': 'c3', 'state': 0}]},
            {'rep_index': 5, 'is_clean': False, 'defects': [{'defect_code': 'c3', 'state': 1}, {'defect_code': 'c1', 'state': 0}]},
        ]
        summary = self.engine.generate_set_summary(exercise_id=0, exercise_name="바벨 스쿼트", rep_results=rep_results)
        self.assertEqual(summary['total_reps'], 5)
        self.assertEqual(summary['clean_reps'], 1)
        self.assertEqual(summary['clean_ratio_percent'], 20.0)
        self.assertEqual(summary['most_frequent_defect']['code'], 'c1')
        self.assertEqual(summary['most_frequent_defect']['count'], 3)
        self.assertIn("요추/척추 기립근", summary['summary_title'])
        self.assertTrue(len(summary['recommended_drills']) > 0)
        self.assertIn("데드버그", summary['recommended_drills'][0])

    def test_06_execution_latency_under_1ms(self):
        """추론 지연시간 벤치마크 (목표: < 1ms)"""
        dummy_preds = [{'defect_code': f'c{i}', 'state': 1, 'probability': 0.8} for i in range(1, 5)]
        t0 = time.perf_counter()
        for _ in range(100):
            _ = self.engine.prioritize_rep_defects(exercise_id=0, defect_predictions=dummy_preds)
        elapsed_per_call_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
        print(f"\n[Latency Benchmark] 1-Rep 피드백 산출 평균 지연시간: {elapsed_per_call_ms:.4f} ms")
        self.assertLess(elapsed_per_call_ms, 1.0, "피드백 엔진 1회 처리 시간은 1ms 미만이어야 합니다.")


if __name__ == '__main__':
    unittest.main()

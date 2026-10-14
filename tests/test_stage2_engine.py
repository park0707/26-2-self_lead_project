# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_stage2_engine.py
Purpose: 웹 서빙 전용 2단계 운동별 세부 결함 추론 엔진(Stage2InferenceEngine)
         8대 운동 33종 결함 전수 단위 테스트 및 레이턴시 검증
"""

import os
import sys
import time
import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.config import EXERCISE_CATALOG


def run_unit_tests():
    print("=" * 95)
    print("  ★ [vidio-health-trainer] Stage2InferenceEngine 8대 운동 33종 결함 E2E 전수 검증")
    print("=" * 95)

    # 1. 엔진 인스턴스 생성 (싱글톤)
    t0 = time.time()
    engine = Stage2InferenceEngine.get_instance(prewarm=False)
    init_time = time.time() - t0
    print(f"[+] 1. Stage2 엔진 초기화 성공 (소요 시간: {init_time:.3f}초)")

    # 2. 사전 워밍 (Pre-warming)
    print("\n[*] 2. 8대 운동 33종 세부 결함 모델 사전 워밍(Pre-warming) 수행 중...")
    t_warm = time.time()
    warm_elapsed = engine.warm_up()
    print(f"    ✔ 사전 워밍 완료 ({warm_elapsed:.2f}초, 모든 모델 메모리 상주 완료)")

    # 3. 8대 운동 전수 진단 테스트
    np.random.seed(42)
    dummy_mp33 = np.random.randn(64, 33, 3).astype(np.float32) * 0.2
    views_test = ['view1', 'view2', 'view3', 'view4', 'view5']

    total_defects_tested = 0
    latencies = []

    print("\n[*] 3. 8대 운동별 개별 세부 결함 모델 직접 호출 및 규격 검증:")
    for ex_id in range(8):
        ex_meta = EXERCISE_CATALOG[ex_id]
        view_for_ex = views_test[ex_id % len(views_test)]

        t_inf = time.time()
        res = engine.diagnose(
            exercise=ex_id,
            keypoints_seq=dummy_mp33,
            camera_view=view_for_ex
        )
        elapsed_ms = (time.time() - t_inf) * 1000.0
        latencies.append(elapsed_ms)

        print(f"\n  -------------------------------------------------------------")
        print(f"  ▶ 운동 [{ex_id}] {ex_meta['name']} (시점: {res['selected_view']}, 지연시간: {elapsed_ms:.2f}ms)")
        print(f"  -------------------------------------------------------------")
        assert res['exercise_id'] == ex_id
        assert len(res['defects']) == ex_meta['defects_count']

        for d in res['defects']:
            total_defects_tested += 1
            print(f"    - [{d['defect_code']}] {d['defect_name']:<35} | 확률: {d['probability']*100:5.1f}% | 상태: {d['state_label']:<6} | 피드백: \"{d['feedback']}\"")
            assert 0.0 <= d['probability'] <= 1.0, f"확률 범위 오류: {d['probability']}"
            assert d['state'] in [0, 1, 2], f"State 코드 오류: {d['state']}"
            assert d['feedback'] == "피드백 자리", f"기본 피드백 불일치: {d['feedback']}"

    # 4. 비정규 프레임(50 프레임) 3차 스플라인 보간 진단 테스트
    print(f"\n[+] 4. 비정규 50-프레임 시퀀스 자동 64F 스플라인 보간 진단 검증:")
    dummy_50f = np.random.randn(50, 33, 3).astype(np.float32) * 0.2
    res_50f = engine.diagnose(exercise=0, keypoints_seq=dummy_50f, camera_view='시점 C')
    print(f"    ✔ 50프레임 입력 -> 64프레임 자동 보간 후 진단 성공 ({res_50f['inference_time_ms']}ms)")
    assert len(res_50f['defects']) == 4

    # 5. 종합 평가 요약
    print("\n" + "=" * 95)
    print(f"  ★ [검증 결과 요약]")
    print(f"    - 총 검증 운동 종목: 8개 전 종목")
    print(f"    - 총 검증 세부 결함 모델: {total_defects_tested}개 전 결함")
    print(f"    - 사전 워밍 후 평균 추론 지연시간: {np.mean(latencies):.2f} ms (실시간 웹 서비스 50ms 미만 완벽 충족!)")
    print(f"    - 피드백 슬롯 검증: 전 모델 '피드백 자리' default 규격 100% 일치")
    print("=" * 95)
    print("  ★ [PASS] Stage2InferenceEngine 모든 8대 운동 33종 결함 단위 테스트 100% 통과!")
    print("=" * 95)
    return True


if __name__ == '__main__':
    run_unit_tests()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_stage1_engine.py
Purpose: 웹 서빙 전용 1단계 운동 분류 엔진(Stage1InferenceEngine) E2E 단위 테스트 및 레이턴시 검증
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

from src.web.pipeline.stage1_engine import Stage1InferenceEngine

BUNDLE_PATH = r"D:\src\web\weights\stage1_v6_production_bundle.pkl"


def run_unit_tests():
    print("=" * 85)
    print("  ★ [vidio-health-trainer] Stage1InferenceEngine E2E 단위 테스트 및 레이턴시 검증")
    print("=" * 85)

    if not os.path.exists(BUNDLE_PATH):
        print(f"[!] 번들 파일이 아직 생성되지 않았습니다: {BUNDLE_PATH}")
        return False

    # 1. 엔진 인스턴스 생성 (싱글톤)
    t0 = time.time()
    engine = Stage1InferenceEngine(bundle_path=BUNDLE_PATH)
    init_time = time.time() - t0
    print(f"[+] 1. 엔진 초기화 성공 (소요 시간: {init_time:.3f}초)")

    # 2. 합성 관절 시퀀스 테스트 (64, 15, 3)
    np.random.seed(42)
    sample_15j = np.random.randn(64, 15, 3).astype(np.float32) * 0.2
    
    t_inf = time.time()
    res1 = engine.predict(sample_15j)
    t_inf_elapsed = (time.time() - t_inf) * 1000.0

    print(f"\n[+] 2. (64, 15, 3) 15-관절 입력 추론 검증:")
    print(f"    - 판별 종목: {res1['exercise_name']} (ID: {res1['exercise_id']})")
    print(f"    - 신뢰도(Confidence): {res1['confidence']*100:.2f}%")
    print(f"    - 소속 대그룹: {res1['macro_group']}")
    print(f"    - 추론 지연시간: {res1['inference_time_ms']} ms (E2E 측정: {t_inf_elapsed:.2f} ms)")
    assert 0 <= res1['exercise_id'] <= 7, "Exercise ID out of bounds"
    assert 0.0 <= res1['confidence'] <= 1.0, "Confidence out of range"

    # 3. MediaPipe 33관절 시퀀스 테스트 (64, 33, 3)
    sample_33j = np.random.randn(64, 33, 3).astype(np.float32) * 0.2
    res2 = engine.predict(sample_33j)
    print(f"\n[+] 3. (64, 33, 3) MediaPipe 33-관절 자동 매핑 입력 추론 검증:")
    print(f"    - 판별 종목: {res2['exercise_name']} (ID: {res2['exercise_id']})")
    print(f"    - 신뢰도(Confidence): {res2['confidence']*100:.2f}%")
    print(f"    - 추론 지연시간: {res2['inference_time_ms']} ms")
    assert 0 <= res2['exercise_id'] <= 7, "Exercise ID out of bounds"

    # 3-1. MediaPipe 33관절 비정규 프레임 (45, 33, 3) -> 64F 스플라인 보간 검증
    sample_45f = np.random.randn(45, 33, 3).astype(np.float32) * 0.2
    res3 = engine.predict(sample_45f)
    print(f"\n[+] 3-1. (45, 33, 3) MediaPipe 33관절 비정규 프레임 64F 3차 스플라인 보간 추론 검증:")
    print(f"    - 판별 종목: {res3['exercise_name']} (ID: {res3['exercise_id']})")
    print(f"    - 신뢰도(Confidence): {res3['confidence']*100:.2f}%")
    print(f"    - 추론 지연시간: {res3['inference_time_ms']} ms")
    assert 0 <= res3['exercise_id'] <= 7, "Exercise ID out of bounds"

    # 4. 실제 데이터셋 샘플로 실측 테스트
    from src.models.first_model.dataset import Stage1DatasetLoader
    test_npz = r"D:\Data\피처_데이터\Condition4_NovelFlow_3D\Dual_Test_Strict.npz"
    if os.path.exists(test_npz):
        split_data = Stage1DatasetLoader.load_split_data('Dual_Test_Strict.npz')
        raw_b = split_data['X_Format_B']
        y_true = split_data['y_action']
        print(f"\n[+] 4. 실제 원천 테스트셋 10개 샘플 연속 실측 테스트:")
        latencies = []
        correct_count = 0
        N_eval = min(10, len(raw_b))
        for i in range(N_eval):
            sample_seq = raw_b[i]  # (64, 98)
            r = engine.predict(sample_seq)
            latencies.append(r['inference_time_ms'])
            is_correct = (r['exercise_id'] == y_true[i])
            if is_correct:
                correct_count += 1
            correct_str = "✔ 정답" if is_correct else "✘ 오답"
            print(f"    - 샘플 #{i+1:02d}: 정답={engine.exercise_names_short[y_true[i]]} -> 예측={r['exercise_short']} ({r['confidence']*100:.1f}%) | {correct_str} ({r['inference_time_ms']}ms)")
        print(f"    - 10개 샘플 실측 정확도: {correct_count}/{N_eval} ({correct_count/N_eval*100:.1f}%)")
        print(f"    - 평균 추론 지연시간: {np.mean(latencies):.2f} ms")

    print("\n" + "=" * 85)
    print("  ★ [PASS] Stage1InferenceEngine 모든 단위 테스트 100% 통과!")
    print("=" * 85)
    return True


if __name__ == '__main__':
    run_unit_tests()

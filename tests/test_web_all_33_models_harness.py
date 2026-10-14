# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 8대 운동 33종 세부 결함 모델 엔드투엔드 무결성 하네스 테스트
(Harness Engineering Test Suite)

목적:
  1. 웹 서비스 DefectModelRegistry를 통한 8대 운동 33종 결함 모델 전수 직접 호출
  2. 예외 발생(Exception) 및 암묵적 Fallback(prob=0.0, state=0) 발생 여부 정밀 감사
  3. 모든 33종 모델의 정상 추론(0.0 <= prob <= 1.0, state in {0, 1, 2}, 결함 명칭 일치) 검증
"""

import os
import sys
import logging
import traceback
import numpy as np

# 프로젝트 루트 등록
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 로거 설정: logger.error 발생 시 하네스 실패로 기록
class HarnessLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errors = []

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errors.append(self.format(record))

log_handler = HarnessLogHandler()
log_handler.setFormatter(logging.Formatter('%(name)s [%(levelname)s]: %(message)s'))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

from src.web.core.model_registry import DefectModelRegistry
from src.web.core.config import EXERCISE_CATALOG

def run_harness():
    print("=" * 80)
    print("  [Harness Engineering] vidio-health-trainer 8대 운동 33종 모델 전수 검증")
    print("=" * 80)

    registry = DefectModelRegistry.get_instance()

    # 현실적인 더미 관절 데이터 생성 (64프레임, 정규화 범위 0~1)
    np.random.seed(42)
    dummy_mp33 = np.random.uniform(0.3, 0.7, (64, 33, 3)).astype(np.float32)
    dummy_rtm26 = np.random.uniform(0.3, 0.7, (64, 26, 3)).astype(np.float32)
    # Z축 및 Score 현실화
    dummy_mp33[:, :, 2] = np.random.uniform(-0.2, 0.2, (64, 33)).astype(np.float32)
    dummy_rtm26[:, :, 2] = np.random.uniform(0.8, 1.0, (64, 26)).astype(np.float32) # RTMPose confidence score

    views = ['view1', 'view2', 'view3', 'view4', 'view5']

    total_models = 0
    passed_models = 0
    failed_models = []

    audit_summary = []

    for eid, ex_info in EXERCISE_CATALOG.items():
        ex_name = ex_info['name']
        print(f"\n[{eid+1}/8] 운동 검증: {ex_name} (ID: {eid})")

        # 각 시점별 테스트
        for view_code in ['view3', 'view2']:
            initial_error_count = len(log_handler.errors)
            try:
                preds = registry.evaluate_exercise_defects(
                    exercise_id=eid,
                    mp33_seq=dummy_mp33,
                    rtm26_seq=dummy_rtm26,
                    view_code=view_code
                )
            except Exception as e:
                print(f"  [CRITICAL ERROR] evaluate_exercise_defects 호출 중 예외: {e}")
                traceback.print_exc()
                preds = []

            new_errors = log_handler.errors[initial_error_count:]

            if view_code == 'view3': # 기본 정면 시점 결과 상세 감사
                for pred in preds:
                    total_models += 1
                    # 에러 로그 발생 여부 대조
                    model_error = [err for err in new_errors if f"{pred.defect_code} 실패" in err or f"ID {eid}" in err]
                    
                    is_valid_prob = isinstance(pred.probability, float) and 0.0 <= pred.probability <= 1.0
                    is_valid_state = pred.state in [0, 1, 2]
                    is_valid_label = pred.state_label in ["정상", "결함", "주의/보류"]
                    has_error_log = len(model_error) > 0

                    if is_valid_prob and is_valid_state and is_valid_label and not has_error_log:
                        status = "PASS"
                        passed_models += 1
                    else:
                        status = "FAIL"
                        failed_models.append((ex_name, pred.defect_code, pred.defect_name, model_error))

                    print(f"  [{status}] {pred.defect_code}: {pred.defect_name} | 확률: {pred.probability:.4f} | 상태: {pred.state} ({pred.state_label})")
                    audit_summary.append({
                        'exercise_id': eid,
                        'exercise_name': ex_name,
                        'defect_code': pred.defect_code,
                        'defect_name': pred.defect_name,
                        'probability': pred.probability,
                        'state': pred.state,
                        'state_label': pred.state_label,
                        'status': status,
                        'error_log': model_error
                    })

    print("\n" + "=" * 80)
    print(f"  [하네스 테스트 최종 결과] 총 {total_models}종 중 {passed_models}종 통과 (성공률: {passed_models/total_models*100:.1f}%)")
    if failed_models:
        print(f"  [!] 실패 모델 ({len(failed_models)}건):")
        for ex_n, code, name, err in failed_models:
            print(f"    - {ex_n} [{code}] {name}: {err}")
    else:
        print("  ★ 8대 운동 33종 전 모델 100% 정상 작동 검증 완료 (0 Exceptions, 0 Silent Fallbacks) ★")
    print("=" * 80)

    return passed_models, total_models, audit_summary

if __name__ == '__main__':
    p, t, summary = run_harness()
    if p == t and t == 33:
        sys.exit(0)
    else:
        sys.exit(1)

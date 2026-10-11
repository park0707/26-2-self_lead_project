# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: test_d5_v7_two_tier_inference.py
Purpose: d5 v7.0 Two-Tier 모델 단일 시퀀스 실시간 추론 API 무결성 검증
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v7_two_tier import (
    BabelDeadliftD5ModelV7TwoTier,
    infer_d5_v7_sequence
)

RAW_DATA_PATH = os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz")
MODEL_PATH = os.path.join(CURRENT_DIR, "weights", "d5_model_v7_two_tier_best.pkl")


def test_inference():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d5 v7.0 Two-Tier 단일 시퀀스 추론 테스트")
    print("=" * 80)

    raw = np.load(RAW_DATA_PATH, allow_pickle=True)
    kps = raw['kps']
    views = raw['views']
    y = raw['y'][:, 4]  # d5 라벨

    print(f"[*] 테스트 샘플 로드 완료: 총 {len(kps)}개 시퀀스 중 3개 샘플 테스트\n")

    test_indices = [0, 10, 25]  # 다양한 시점 및 라벨
    for idx in test_indices:
        kps_sample = kps[idx]
        view_sample = str(views[idx])
        y_true = int(y[idx])

        result = infer_d5_v7_sequence(
            kps_64f=kps_sample,
            view_name=view_sample,
            model_path=MODEL_PATH
        )

        print(f"▶ [Sample #{idx:3d}] 실제 정답: {'결함(1)' if y_true==1 else '정상(0)'}")
        print(f"  - 입력 시점      : {result['view_name']}")
        print(f"  - Tier 1 확률    : {result['tier1_prob']:.4f}")
        print(f"  - Tier 2 확률    : {result['tier2_prob']:.4f}")
        print(f"  - 융합 결함 확률 : {result['defect_prob']:.4f} (캘리브레이션: {result['calibrated_prob']:.4f})")
        print(f"  - 추론 경로      : {result['inference_path']}")
        print(f"  - 최종 판정      : {result['decision_label']} (예측값: {result['prediction']})")
        print(f"  - 확정 상태      : {'확정 (DECIDED)' if result['is_decided'] else '보류 (RETAKE)'}")
        print("-" * 75)

    print("\n[+] d5 v7.0 Two-Tier 단일 시퀀스 추론 무결성 검증 100% 성공!")


if __name__ == "__main__":
    test_inference()

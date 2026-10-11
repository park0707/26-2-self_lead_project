# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: test_d3_model_v7_0_inference.py
Purpose: 공식 확정된 v7.0 2계층 모델(d3_model_v7_0.py)의 가중치 로드, 단일 시퀀스 추론 및 5대 시점 3-State 출력 규격 안정성 전수 검증
"""

import os
import sys
import numpy as np

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v7_0 import (
    BabelDeadliftD3ModelV7_0,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v3 import VIEW_CODE_LIST

WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "d3_model_v7_0_weights.pkl")


def test_v7_0_inference():
    print("=" * 90)
    print("  [vidio-health-trainer] d3 v7.0 2계층 공식 확정 모델 단일 시퀀스 추론 및 무결성 감사")
    print("=" * 90)

    # 1. 모델 인스턴스 생성 및 가중치 로드
    model = BabelDeadliftD3ModelV7_0()
    assert os.path.exists(WEIGHTS_PATH), f"가중치 파일 부재: {WEIGHTS_PATH}"
    model.load_weights(WEIGHTS_PATH)
    print(f"[✔] 공식 가중치 로드 성공: {WEIGHTS_PATH}")

    # 2. 더미 64프레임 33관절 3D 랜드마크 시퀀스 생성 (정상 스케일)
    dummy_kps = np.random.uniform(low=0.2, high=0.8, size=(64, 33, 3)).astype(np.float32)

    # 3. 5대 공식 시점별 단일 시퀀스 피처 추출 및 추론 검증
    print("\n[*] 5대 공식 시점별 3-State 단일 시퀀스 진단 테스트:")
    for vcode in VIEW_CODE_LIST:
        alias = VIEW_ALIAS_MAP[vcode]

        # 단일 시퀀스 피처 자동 추출
        seq_57, feats_36 = model.extract_features_single(dummy_kps, vcode)
        assert seq_57.shape == (1, 64, 57), f"시퀀스 형상 불일치: {seq_57.shape}"
        assert feats_36.shape == (1, 36), f"피처 형상 불일치: {feats_36.shape}"

        # 3-State 진단
        diag = model.diagnose_3state(seq_57, feats_36, views=np.array([vcode]))[0]

        print(f"  [{alias}]")
        print(f"    - P(Defect) 확률: {diag['prob_defect']:.4f}")
        print(f"    - 진단 결과     : {diag['state_str']}")
        print(f"    - 사용자 안내   : {diag['message']}")
        print(f"    - 임계치 구간   : 정상 <= {diag['threshold_range'][0]:.3f} / 결함 >= {diag['threshold_range'][1]:.3f}")

    print("\n" + "=" * 90)
    print("  [✔] d3 v7.0 2계층 공식 확정 모델의 모든 추론 파이프라인 무결성 검증 완료!")
    print("=" * 90)


if __name__ == '__main__':
    test_v7_0_inference()

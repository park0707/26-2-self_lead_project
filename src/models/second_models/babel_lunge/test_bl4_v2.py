# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_bl4_v2.py
Purpose: bl4_v2 추론 모듈 무결성 및 선행 연구 기반 3-State 진단 인터페이스 단위 테스트
"""

import os
import sys
import glob
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_lunge.bl4_v2 import BL4V2Classifier


def test_v2_classifier():
    print("=" * 60)
    print("  [Harness Test] bl4_v2.py 공식 배포 추론 모듈 단위 테스트")
    print("=" * 60)

    clf = BL4V2Classifier()
    print(f"[✓] BL4V2Classifier 초기화 성공: {clf.model_name} ({clf.version})")
    print(f"    - 피처 수: {len(clf.feature_names)}차원")
    print(f"    - 3-State 임계치: Clean <= {clf.thresholds['tau_clean']:.2f}, Defect >= {clf.thresholds['tau_def']:.2f}")

    # 1. 더미 텐서 (64, 33, 4) 추론 테스트
    dummy_seq = np.random.randn(64, 33, 4).astype(np.float32)
    prob = clf.predict_proba(dummy_seq)
    pred_2s = clf.predict(dummy_seq)
    res_3s = clf.predict_3state(dummy_seq)

    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    assert pred_2s in (0, 1), f"2-State prediction invalid: {pred_2s}"
    assert res_3s['state'] in ('Clean', 'Defect', 'Retake'), f"Invalid state: {res_3s['state']}"
    print(f"[✓] 더미 텐서 추론 성공: Prob={prob:.4f}, 2-State={pred_2s}, 3-State={res_3s['state']}")

    # 2. 실제 전처리된 npz 샘플 추론 테스트
    sample_files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지\*.npz")
    if sample_files:
        sample_p = sample_files[0]
        data = np.load(sample_p)
        seq = data['kps_33_3d_64f']
        s_prob = clf.predict_proba(seq)
        s_res = clf.predict_3state(seq)
        print(f"[✓] 실제 샘플({os.path.basename(sample_p)}) 추론 성공:")
        print(f"    - 결함 확률: {s_prob:.4f}")
        print(f"    - 3-State 판정: {s_res['state']} (코드 {s_res['state_code']})")
        print(f"    - 코칭 피드백: {s_res['feedback']}")
        print(f"    - 주요 피처: {s_res['features']}")

    print("\n[★ APPROVED] bl4_v2.py 배포 인터페이스 단위 테스트 100% 통과!\n")


if __name__ == '__main__':
    test_v2_classifier()

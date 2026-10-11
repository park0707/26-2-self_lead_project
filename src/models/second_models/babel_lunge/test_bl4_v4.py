# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_bl4_v4.py
Purpose: bl4_v4.py 배포 추론 모듈 인터페이스 및 무결성 단위 테스트
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, r"D:/")
from src.models.second_models.babel_lunge.bl4_v4 import BL4V4Classifier, DEFAULT_BUNDLE_PATH


def test_bl4_v4_module():
    print("=" * 60)
    print("  [Harness Test] bl4_v4.py 공식 배포 추론 모듈 단위 테스트")
    print("=" * 60)

    if not os.path.exists(DEFAULT_BUNDLE_PATH):
        print(f"[!] 가중치 번들이 아직 생성되지 않았습니다: {DEFAULT_BUNDLE_PATH}")
        return

    clf = BL4V4Classifier(DEFAULT_BUNDLE_PATH)
    assert clf.is_loaded, "가중치 번들 로드 실패!"
    print(f"[PASS] BL4V4Classifier 초기화 성공: {clf.bundle_path}")
    print(f"    - 정밀 피처 수: {len(clf.feature_names)}차원")
    print(f"    - 3-State 임계치: Clean <= {clf.thresholds['tau_clean']:.2f}, Defect >= {clf.thresholds['tau_def']:.2f}")

    # 1. 더미 텐서 추론 테스트
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    p_dummy = clf.predict_proba(dummy_seq)
    pred_2s = clf.predict(dummy_seq)
    res_3s = clf.predict_3state(dummy_seq)

    assert 0.0 <= p_dummy <= 1.0, f"확률 범위 오류: {p_dummy}"
    assert pred_2s in (0, 1), f"2-State 예측값 오류: {pred_2s}"
    assert res_3s['state'] in ("Clean", "Defect", "Retake"), f"3-State 상태 오류: {res_3s['state']}"
    print(f"[PASS] 더미 텐서 추론 성공: Prob={p_dummy:.4f}, 2-State={pred_2s}, 3-State={res_3s['state']}")

    # 2. 실제 시퀀스 파일 추론 테스트
    sample_files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지\*.npz")
    if sample_files:
        sample_p = sample_files[0]
        d = np.load(sample_p)
        seq = d['kps_33_3d_64f']
        res = clf.predict_3state(seq)
        print(f"[PASS] 실제 샘플({os.path.basename(sample_p)}) 추론 성공:")
        print(f"    - 결함 확률: {res['defect_prob']:.4f}")
        print(f"    - 3-State 판정: {res['state']} (코드 {res['state_code']})")
        print(f"    - 코칭 피드백: {res['feedback']}")
        print(f"    - 주요 피처: {res['features']}")

    print("\n[★ APPROVED] bl4_v4.py 배포 인터페이스 단위 테스트 100% 통과!")


if __name__ == "__main__":
    test_bl4_v4_module()

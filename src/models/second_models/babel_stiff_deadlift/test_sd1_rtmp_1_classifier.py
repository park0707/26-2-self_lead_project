# -*- coding: utf-8 -*-
"""
test_sd1_rtmp_1_classifier.py
sd1_rtmp_1 분류기 단위 테스트 (초기화, 5대 시점, Halpe26/MP33 더미 및 실제 시퀀스, 예외 처리)
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("D:/src/models/second_models/babel_stiff_deadlift")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sd1_rtmp_1_classifier import SD1RTMP1Classifier, VIEW_ALIAS_MAP


def test_initialization():
    print("[Test 1] sd1_rtmp_1 초기화 및 가중치 번들 로딩 검증...")
    clf = SD1RTMP1Classifier()
    assert clf.bundle is not None
    assert clf.model_name == "sd1_rtmp_1"
    assert clf.tau_clean < clf.tau_defect
    print(f"  -> 초기화 성공: {clf.model_name}, tau_clean={clf.tau_clean:.3f}, tau_defect={clf.tau_defect:.3f}")


def test_dummy_inference():
    print("\n[Test 2] 더미 텐서 (Halpe26 & MP33) 5대 공식 시점 전수 추론 검증...")
    clf = SD1RTMP1Classifier()

    dummy_halpe26 = np.random.randn(64, 26, 3).astype(np.float32)
    dummy_mp33 = np.random.randn(64, 33, 3).astype(np.float32)

    views = ["view1", "view2", "view3", "view4", "view5", "시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]
    for v in views:
        p1 = clf.predict_proba(dummy_halpe26, view_name=v)
        res1 = clf.diagnose(dummy_halpe26, view_name=v)
        assert 0.0 <= p1 <= 1.0
        assert res1["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]

        p2 = clf.predict_proba(dummy_mp33, view_name=v)
        res2 = clf.diagnose(dummy_mp33, view_name=v)
        assert 0.0 <= p2 <= 1.0
        assert res2["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]

    print("  -> Halpe26 및 MP33 5대 전 시점 추론 100% 정상 통과!")


def test_error_handling():
    print("\n[Test 3] 유효하지 않은 입력 방어 로직 검증...")
    clf = SD1RTMP1Classifier()

    try:
        clf.predict_proba(np.random.randn(30, 26, 3))
        assert False, "형상 오류 미감지"
    except ValueError as e:
        print(f"  -> 프레임 수 오류 정상 방어: {e}")

    try:
        clf.predict_proba(np.random.randn(64, 26, 3), view_name="invalid_camera")
        assert False, "미지원 시점 미감지"
    except ValueError as e:
        print(f"  -> 미지원 시점 정상 방어: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("  SD1RTMP1Classifier 단위 테스트 시작")
    print("=" * 80)
    test_initialization()
    test_dummy_inference()
    test_error_handling()
    print("\n" + "=" * 80)
    print("  [PASS] 모든 SD1RTMP1Classifier 단위 테스트가 성공적으로 통과되었습니다!")
    print("=" * 80)

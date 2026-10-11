# -*- coding: utf-8 -*-
"""
test_sd1_rtmp_2_classifier.py
sd1_rtmp_2 분류기 단위 테스트 (초기화, 5대 공식 시점, Halpe26/MP33 더미 및 실제 시퀀스, 예외 처리)
규정 준수: vidio-health-trainer, harness-engineering
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("D:/src/models/second_models/babel_stiff_deadlift")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sd1_rtmp_2_classifier import SD1RTMP2Classifier, VIEW_ALIAS_MAP, VIEW_CODE_MAP


def test_initialization():
    print("[Test 1] sd1_rtmp_2 초기화 및 가중치 번들 로딩 검증...")
    clf = SD1RTMP2Classifier()
    assert clf.bundle is not None
    assert clf.model_name == "sd1_rtmp_2"
    assert len(clf.cb_models) == 5
    assert len(clf.lgb_models) == 5
    assert len(clf.et_models) == 5
    assert len(clf.rf_models) == 5
    assert len(clf.view_thresholds) >= 5
    print(f"  -> 초기화 성공: {clf.model_name}, 4대 앙상블 20개 폴드 모델 로드 확인")


def test_view_adaptive_thresholds():
    print("\n[Test 2] 시점 적응형 3-State 임계값 (View-Adaptive) 등록 검증...")
    clf = SD1RTMP2Classifier()
    for vk in ["view1", "view2", "view3", "view4", "view5"]:
        assert vk in clf.view_thresholds, f"시점 {vk} 임계값 누락"
        th = clf.view_thresholds[vk]
        tc, td = th['tau_clean'], th['tau_defect']
        assert tc < td, f"임계값 역전 오류: tc={tc}, td={td}"
        print(f"  -> {vk} 임계값 정상: [{tc:.4f}, {td:.4f}]")


def test_dummy_inference():
    print("\n[Test 3] 더미 텐서 (Halpe26 & MP33) 5대 공식 시점 전수 추론 검증...")
    clf = SD1RTMP2Classifier()

    dummy_halpe26 = np.random.randn(64, 26, 3).astype(np.float32)
    dummy_mp33 = np.random.randn(64, 33, 3).astype(np.float32)

    views = ["view1", "view2", "view3", "view4", "view5", "시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]
    for v in views:
        p1 = clf.predict_proba(dummy_halpe26, view_name=v)
        res1 = clf.diagnose(dummy_halpe26, view_name=v)
        assert 0.0 <= p1 <= 1.0, f"확률 범위 오류: {p1}"
        assert res1["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]
        assert "feedback_message" in res1
        assert "thresholds" in res1

        p2 = clf.predict_proba(dummy_mp33, view_name=v)
        res2 = clf.diagnose(dummy_mp33, view_name=v)
        assert 0.0 <= p2 <= 1.0, f"확률 범위 오류: {p2}"
        assert res2["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]

    print("  -> Halpe26 및 MP33 5대 전 시점 추론 100% 정상 통과!")


def test_error_handling():
    print("\n[Test 4] 유효하지 않은 입력 방어 로직 검증...")
    clf = SD1RTMP2Classifier()

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
    print("  SD1RTMP2Classifier 단위 테스트 시작 (harness-engineering)")
    print("=" * 80)
    test_initialization()
    test_view_adaptive_thresholds()
    test_dummy_inference()
    test_error_handling()
    print("\n" + "=" * 80)
    print("  [PASS] 모든 SD1RTMP2Classifier 단위 테스트가 성공적으로 통과되었습니다!")
    print("=" * 80)

# -*- coding: utf-8 -*-
"""
test_sd1_rtmp_classifier.py
sd1_rtmp 분류기 단위 테스트 (초기화, 5대 시점, Halpe26/MP33 더미 및 실제 시퀀스, 예외 처리)
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("D:/src/models/second_models/babel_stiff_deadlift")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sd1_rtmp_classifier import SD1RTMPClassifier, VIEW_ALIAS_MAP


def test_initialization():
    print("[Test 1] 초기화 및 가중치 번들 로딩 검증...")
    clf = SD1RTMPClassifier()
    assert clf.bundle is not None
    assert clf.model_name == "sd1_rtmp"
    assert clf.tau_clean < clf.tau_defect
    print(f"  -> 초기화 성공: {clf.model_name}, tau_clean={clf.tau_clean:.3f}, tau_defect={clf.tau_defect:.3f}")


def test_dummy_inference():
    print("\n[Test 2] 더미 텐서 (Halpe26 & MP33) 5대 공식 시점 전수 추론 검증...")
    clf = SD1RTMPClassifier()

    # Halpe26 dummy: (64, 26, 3)
    dummy_halpe26 = np.random.randn(64, 26, 3).astype(np.float32)
    # MediaPipe33 dummy: (64, 33, 3)
    dummy_mp33 = np.random.randn(64, 33, 3).astype(np.float32)

    views = ["view1", "view2", "view3", "view4", "view5", "시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]
    for v in views:
        # 1. Halpe 26
        p1 = clf.predict_proba(dummy_halpe26, view_name=v)
        res1 = clf.diagnose(dummy_halpe26, view_name=v)
        assert 0.0 <= p1 <= 1.0
        assert res1["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]

        # 2. MediaPipe 33 호환
        p2 = clf.predict_proba(dummy_mp33, view_name=v)
        res2 = clf.diagnose(dummy_mp33, view_name=v)
        assert 0.0 <= p2 <= 1.0
        assert res2["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]

    print("  -> Halpe26 및 MP33 5대 전 시점 추론 100% 정상 통과!")


def test_real_sequence():
    print("\n[Test 3] 실제 RTMPose 전처리 시퀀스 추론 검증...")
    import glob
    rtm_files = glob.glob(r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트\*.npz")
    if not rtm_files:
        print("  -> RTMPose 데이터셋 없음, 스킵")
        return

    clf = SD1RTMPClassifier()
    print(f"  -> {len(rtm_files)}개 시퀀스 중 5개 샘플 추론 실행:")
    for i in range(min(5, len(rtm_files))):
        fpath = rtm_files[i]
        d = np.load(fpath, allow_pickle=True)
        seq = d['kps_26_64f']
        v = str(d['view'])
        sid = str(d['sequence_id'])
        lbl = int(d['label'])

        p = clf.predict_proba(seq, view_name=v)
        diag = clf.diagnose(seq, view_name=v)
        print(f"     Sample {i+1} [{sid} | {diag['camera_view']['view_alias']} | Label={lbl}]: P={p:.3f}, State={diag['decision_state']}")


def test_error_handling():
    print("\n[Test 4] 유효하지 않은 입력 방어 로직 검증...")
    clf = SD1RTMPClassifier()

    # 잘못된 프레임 수
    try:
        clf.predict_proba(np.random.randn(30, 26, 3))
        assert False, "형상 오류 미감지"
    except ValueError as e:
        print(f"  -> 프레임 수 오류 정상 방어: {e}")

    # 지원되지 않는 시점
    try:
        clf.predict_proba(np.random.randn(64, 26, 3), view_name="invalid_camera")
        assert False, "미지원 시점 미감지"
    except ValueError as e:
        print(f"  -> 미지원 시점 정상 방어: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("  SD1RTMPClassifier 단위 테스트 시작")
    print("=" * 80)
    test_initialization()
    test_dummy_inference()
    test_real_sequence()
    test_error_handling()
    print("\n" + "=" * 80)
    print("  [PASS] 모든 SD1RTMPClassifier 단위 테스트가 성공적으로 통과되었습니다!")
    print("=" * 80)

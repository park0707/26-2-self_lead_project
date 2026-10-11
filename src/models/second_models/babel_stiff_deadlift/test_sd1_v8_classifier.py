"""
test_sd1_v8_classifier.py
sd1_v8 분류기 단위 테스트 (초기화, 5대 시점, 실제 3D 시퀀스, 예외 처리)
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("D:/src/models/second_models/babel_stiff_deadlift")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sd1_v8_classifier import SD1V8Classifier, VIEW_ALIAS_MAP

def test_initialization():
    print("[Test 1] 초기화 및 번들 로딩 테스트...")
    clf = SD1V8Classifier()
    assert clf.bundle is not None
    assert clf.model_name == "sd1_v8"
    assert clf.tau_clean < clf.tau_defect
    print(f"  -> 초기화 성공: {clf.model_name}, tau_clean={clf.tau_clean}, tau_defect={clf.tau_defect}")

def test_dummy_inference():
    print("\n[Test 2] 더미 텐서 (64, 33, 3) 5대 시점 전수 추론 테스트...")
    clf = SD1V8Classifier()
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)

    views = ["view1", "view2", "view3", "view4", "view5", "시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]
    for v in views:
        p = clf.predict_proba(dummy_seq, view_name=v)
        res = clf.diagnose(dummy_seq, view_name=v)
        assert 0.0 <= p <= 1.0
        assert res["decision_state"] in ["CLEAN", "DEFECT", "RETAKE"]
        print(f"  -> View [{v}] 추론 통과: P={p:.4f}, State={res['decision_state']}, View={res['camera_view']['view_alias']}")

def test_real_sequence():
    print("\n[Test 3] 실제 AI-Hub 3D 시퀀스 추론 테스트...")
    import pickle
    cache_path = Path("D:/src/models/second_models/babel_stiff_deadlift/stiff_d1_v3_dataset_cache.pkl")
    if not cache_path.exists():
        print("  -> 캐시 부재로 스킵")
        return

    with open(cache_path, "rb") as fp:
        c = pickle.load(fp)
    df = c["df"]
    
    clf = SD1V8Classifier()
    print(f"  -> {len(df)}개 시퀀스 중 5개 샘플 추론 실행:")
    for i in range(5):
        row = df.iloc[i]
        fpath = row["path"]
        d = np.load(fpath, allow_pickle=True)
        seq = d["kps_33_3d_64f"]
        v = row["view"]
        k = row["key"]
        p = clf.predict_proba(seq, view_name=v)
        diag = clf.diagnose(seq, view_name=v)
        print(f"     Sample {i} (Key {k}, {diag['camera_view']['view_alias']}): P={p:.3f}, State={diag['decision_state']}")

def test_error_handling():
    print("\n[Test 4] 유효하지 않은 입력 방어 로직 검증...")
    clf = SD1V8Classifier()
    
    # 잘못된 형상
    try:
        clf.predict_proba(np.random.randn(30, 33, 3))
        assert False, "형상 오류 미감지"
    except ValueError as e:
        print(f"  -> 형상 오류 정상 감지: {e}")

    # 지원되지 않는 시점
    try:
        clf.predict_proba(np.random.randn(64, 33, 3), view_name="invalid_view")
        assert False, "미지원 시점 미감지"
    except ValueError as e:
        print(f"  -> 미지원 시점 정상 감지: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("  SD1V8Classifier 단위 테스트 시작")
    print("=" * 80)
    test_initialization()
    test_dummy_inference()
    test_real_sequence()
    test_error_handling()
    print("\n" + "=" * 80)
    print("  [PASS] 모든 SD1V8Classifier 단위 테스트가 성공적으로 통과되었습니다!")
    print("=" * 80)

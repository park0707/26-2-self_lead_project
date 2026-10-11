"""
test_stiff_integrated_predictor.py
바벨 스티프 데드리프트 통합 추론기 (StiffIntegratedPredictor v4.1) 단위 테스트
- Test 1: v4.1 번들 로딩, 정직한 성능 메타데이터, 하위 버전(v4.0) 번들 호환 로딩
- Test 2: 5대 시점 전수(공식 코드 및 시점 A~E 별칭) 더미 추론, 출력 규격/확률 합 검증
- Test 3: 5-class 디코딩 일관성 (결함 목록 <-> 유효 조합, sd1 결함 시 sd2 동반)
- Test 4: 피처 모듈 규격(45D, NaN 방어) 및 잘못된 입력 방어
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("D:/src/models/second_models/babel_stiff_deadlift/integrated")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stiff_integrated_predictor import StiffIntegratedPredictor
from stiff_meta_v41 import build_meta_features_v41, FEATURE_NAMES, COMBOS

VIEWS = ["view1", "view2", "view3", "view4", "view5", "시점 A", "시점 B", "시점 C", "시점 D", "시점 E"]


def test_initialization():
    print("[Test 1] v4.1 초기화/메타데이터/하위 호환 로딩...")
    p = StiffIntegratedPredictor()
    info = p.get_model_info()
    assert info["bundle_loaded"] and info["version"] == "v4.1"
    assert info["bundle_name"] == "stiff_integrated_v4_1_bundle.pkl"
    hm = info["honest_metrics"]
    assert "중첩 CV" in hm["evaluation"]
    assert info["selective_coverage"] >= 0.90, info["selective_coverage"]
    print(f"  -> 정직한 성능: 3-State EMR={hm['emr_3state']:.2f}%, Clean F1={hm['clean_f1_3state']:.2f}%, 커버리지={hm['coverage']:.1f}%")
    old = StiffIntegratedPredictor(str(ROOT / "stiff_integrated_v4_bundle.pkl"))
    assert old.get_model_info()["version"] == "v4.0"
    print("  -> v4.0 번들 하위 호환 로딩 OK")


def test_dummy_inference():
    print("\n[Test 2] 5대 시점 전수 더미 추론...")
    p = StiffIntegratedPredictor()
    rng = np.random.RandomState(0)
    seq = rng.randn(64, 33, 3).astype(np.float32)
    for v in VIEWS:
        r = p.predict(seq, view=v)
        assert r["final_status"] in ("CLEAN", "DEFECT", "RETAKE")
        for k in ("sd1", "sd2", "sd3"):
            assert k in r["submodel_predictions"]
        assert 0.0 <= r["probabilities"]["clean"] <= 1.0
        assert abs(r["probabilities"]["clean"] + r["probabilities"]["defect"] - 1.0) < 1e-3
        assert len(r["feedback"]) >= 1
        print(f"  -> {v:7s} ({r['meta_info']['view_official']}): {r['final_status']}")


def test_decoding_consistency():
    print("\n[Test 3] 5-class 디코딩 일관성 (랜덤 20개 시퀀스 x 5시점)...")
    p = StiffIntegratedPredictor()
    rng = np.random.RandomState(1)
    valid = {tuple(k for k, f in zip(("sd1", "sd2", "sd3"), c) if f) for c in COMBOS}
    seen = set()
    for i in range(20):
        seq = (rng.randn(64, 33, 3) * rng.uniform(0.3, 2.0)).astype(np.float32)
        for v in ("view1", "view2", "view3", "view4", "view5"):
            res = p.predict_single_rep(seq, view_name=v)
            d = tuple(res["detected_defects"])
            if res["decision_state"] == "DEFECT":
                assert d in valid and len(d) > 0, d
                if "sd1" in d:
                    assert "sd2" in d, "생체역학 규칙 위반: sd1 결함인데 sd2 미동반"
            else:
                assert d == ()
            cp = res["class_probabilities"]
            assert abs(sum(cp.values()) - 1.0) < 1e-2
            seen.add(res["decision_state"])
    print(f"  -> 관측된 상태: {sorted(seen)} (모든 DEFECT 조합이 유효 5조합 내)")


def test_feature_module_and_invalid_input():
    print("\n[Test 4] 피처 모듈/입력 방어...")
    X = build_meta_features_v41([0.2, 0.8], [0.1, 0.5], [0.4, 0.9], ["view1", "view5"], np.zeros((2, 30)))
    assert X.shape == (2, len(FEATURE_NAMES)) == (2, 45) and np.isfinite(X).all()
    assert X[1, FEATURE_NAMES.index('p_sd2c')] >= 0.8  # max(p2, p1) 규칙
    Xn = build_meta_features_v41([0.5], [0.5], [0.5], ["view3"], np.full((1, 30), np.nan))
    assert np.isfinite(Xn).all()
    p = StiffIntegratedPredictor()
    for bad, view, tag in [(np.random.randn(30, 33, 3), "view3", "형상"), (np.random.randn(64, 33, 3), "top_down", "시점")]:
        try:
            p.predict(bad, view=view); assert False, f"{tag} 오류 미감지"
        except ValueError as e:
            print(f"  -> {tag} 오류 정상 감지")


if __name__ == "__main__":
    print("=" * 78); print("  StiffIntegratedPredictor v4.1 단위 테스트"); print("=" * 78)
    test_initialization(); test_dummy_inference(); test_decoding_consistency(); test_feature_module_and_invalid_input()
    print("\n" + "=" * 78); print("  [PASS] 모든 단위 테스트 통과"); print("=" * 78)

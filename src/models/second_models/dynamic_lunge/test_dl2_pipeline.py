# -*- coding: utf-8 -*-
"""
Harness Engineering: dl2_v1 가중치 번들 및 단일 시퀀스 3-State 추론 무결성 테스트
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dl2_v1 import DynamicLungeCondition2ModelV1


def test_bundle_and_inference():
    print("[*] Starting Harness Engineering Validation for dl2_v1...")

    # 1. 파일 존재 여부 검증
    bundle_path = os.path.join(CURRENT_DIR, "weights", "dl2_v1_bundle.pkl")
    assert os.path.exists(bundle_path), f"Missing dl2_v1 bundle: {bundle_path}"
    print("  [Pass] Model bundle file exists.")

    # 2. 번들 로드 검증
    m = DynamicLungeCondition2ModelV1.load_bundle(bundle_path)
    assert m.is_fitted and len(m.lgb_models) > 0 and len(m.cat_models) > 0
    assert len(m.feature_names) == 18
    print(f"  [Pass] dl2_v1 bundle loaded successfully (Folds={len(m.lgb_models)}, Features={len(m.feature_names)}).")

    # 3. 임의의 64프레임 더미 시퀀스로 단일 추론 무결성 검증
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    res = m.predict_3state(dummy_seq)
    assert res["state"] in [0, 1, 2]
    assert 0.0 <= res["defect_probability"] <= 1.0
    print(f"  [Pass] Dummy inference: state={res['state']} ({res['state_str']}), prob={res['defect_probability']:.4f}")

    # 4. 실제 전처리 데이터 1건으로 실데이터 추론 검증
    prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
    files = [f for f in os.listdir(prep_dir) if f.endswith(".npz")]
    if files:
        sample_path = os.path.join(prep_dir, files[0])
        real_data = np.load(sample_path)["kps_33_3d_64f"]
        if real_data.shape[-1] == 4:
            real_data = real_data[:, :, :3]
        real_res = m.predict_3state(real_data)
        assert real_res["state"] in [0, 1, 2]
        print(f"  [Pass] Real sequence inference ({files[0]}):")
        print(f"         state={real_res['state']} ({real_res['state_str']}), prob={real_res['defect_probability']:.4f}")
        print(f"         feedback: {real_res['feedback']}")

    print("[+] All Harness Engineering Tests Passed for dl2_v1!")


if __name__ == "__main__":
    test_bundle_and_inference()

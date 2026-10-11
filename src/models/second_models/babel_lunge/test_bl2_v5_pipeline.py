# -*- coding: utf-8 -*-
"""
Harness Engineering: bl2_v5 가중치 번들 및 단일 시퀀스 3-State 추론 무결성 테스트
"""

import os
import numpy as np
from bl2_v5 import BabelLungeCondition2ModelV5
from lunge_c2_features_v5 import extract_lunge_c2_features_v5

def test_v5_bundle_and_inference():
    print("[*] Starting Harness Engineering Validation for bl2_v5...")

    # 1. 파일 존재 여부 검증
    v5_bundle_path = r"D:\src\models\second_models\babel_lunge\weights\bl2_v5_bundle.pkl"
    assert os.path.exists(v5_bundle_path), f"Missing v5 bundle: {v5_bundle_path}"
    print("  [Pass] Model bundle file exists.")

    # 2. 번들 로드 검증
    m5 = BabelLungeCondition2ModelV5.load_bundle(v5_bundle_path)
    assert m5.is_fitted
    assert len(m5.lgb_models) == 5
    assert len(m5.cat_models) == 5
    assert len(m5.feature_names) == 16
    print(f"  [Pass] v5 bundle loaded successfully (16D Noise-Free features, 5-Fold models).")
    print(f"         tau_normal = {m5.tau_normal:.4f}, tau_defect = {m5.tau_defect:.4f}, max_retake = {m5.max_retake_rate:.2f}, min_cov = {m5.min_coverage:.2f}")

    # 3. 임의의 64프레임 더미 시퀀스로 단일 추론 검증
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feats = m5.extract_features(dummy_seq)
    assert feats.shape == (16,), f"Expected shape (16,), got {feats.shape}"

    res = m5.predict_3state(dummy_seq)
    assert res["state"] in [0, 1, 2]
    assert "feedback" in res
    print(f"  [Pass] Dummy sequence inference: state={res['state']} ({res['state_str']}), prob={res['defect_probability']:.4f}")

    # 4. 실제 전처리 데이터 1건으로 실데이터 추론 검증
    prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    sample_file = os.path.join(prep_dir, os.listdir(prep_dir)[0])
    real_data = np.load(sample_file)["kps_33_3d_64f"]
    if real_data.shape[-1] == 4:
        real_data = real_data[:, :, :3]

    real_res = m5.predict_3state(real_data)
    assert real_res["state"] in [0, 1, 2]
    print(f"  [Pass] Real sequence inference ({os.path.basename(sample_file)}):")
    print(f"         state={real_res['state']} ({real_res['state_str']}), prob={real_res['defect_probability']:.4f}")
    print(f"         feedback: {real_res['feedback']}")

    print("[+] All Harness Engineering Tests for bl2_v5 Passed Successfully!")

if __name__ == "__main__":
    test_v5_bundle_and_inference()

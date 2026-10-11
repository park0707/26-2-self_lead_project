# -*- coding: utf-8 -*-
"""
Harness Engineering: bl2_v1 및 bl2_v2 가중치 번들 및 단일 시퀀스 추론 무결성 테스트
"""

import os
import numpy as np
from bl2_v1 import BabelLungeCondition2ModelV1
from bl2_v2 import BabelLungeCondition2ModelV2

def test_bundles_and_inference():
    print("[*] Starting Harness Engineering Validation for bl2_v1 and bl2_v2...")
    
    # 1. 파일 존재 여부 검증
    v1_bundle_path = r"D:\src\models\second_models\babel_lunge\weights\bl2_v1_bundle.pkl"
    v2_bundle_path = r"D:\src\models\second_models\babel_lunge\weights\bl2_v2_bundle.pkl"
    assert os.path.exists(v1_bundle_path), f"Missing v1 bundle: {v1_bundle_path}"
    assert os.path.exists(v2_bundle_path), f"Missing v2 bundle: {v2_bundle_path}"
    print("  [Pass] Both model bundle files exist.")

    # 2. 번들 로드 검증
    m1 = BabelLungeCondition2ModelV1.load_bundle(v1_bundle_path)
    m2 = BabelLungeCondition2ModelV2.load_bundle(v2_bundle_path)
    
    assert m1.is_fitted and len(m1.tier1_lgb_models) > 0
    assert m2.is_fitted and len(m2.tier1_lgb_models) > 0
    print("  [Pass] Both model bundles successfully loaded and validated.")

    # 3. 임의의 64프레임 더미 시퀀스로 단일 추론 검증
    sample_seq = np.random.randn(64, 33, 3).astype(np.float32)
    
    res1 = m1.predict_3state(sample_seq)
    assert res1["state"] in [0, 1, 2]
    print(f"  [Pass] v1 single dummy inference: state={res1['state']} ({res1['state_str']}), prob={res1['defect_probability']:.4f}")

    res2 = m2.predict_3state(sample_seq)
    assert res2["state"] in [0, 1, 2]
    print(f"  [Pass] v2 single dummy inference: state={res2['state']} ({res2['state_str']}), prob={res2['defect_probability']:.4f}")

    # 4. 실제 전처리 데이터 1건으로 실데이터 추론 검증
    prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    sample_file = os.path.join(prep_dir, os.listdir(prep_dir)[0])
    real_data = np.load(sample_file)["kps_33_3d_64f"]
    if real_data.shape[-1] == 4:
        real_data = real_data[:, :, :3]
        
    real_res1 = m1.predict_3state(real_data)
    real_res2 = m2.predict_3state(real_data)
    print(f"  [Pass] Real sequence inference ({os.path.basename(sample_file)}):")
    print(f"         v1 -> state={real_res1['state']} ({real_res1['state_str']}), prob={real_res1['defect_probability']:.4f}")
    print(f"         v2 -> state={real_res2['state']} ({real_res2['state_str']}), prob={real_res2['defect_probability']:.4f}")

    print("[+] All Harness Engineering Tests Passed Successfully!")

if __name__ == "__main__":
    test_bundles_and_inference()

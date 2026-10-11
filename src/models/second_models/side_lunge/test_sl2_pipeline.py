# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Test: test_sl2_pipeline.py
Purpose: sl2_v1 모델 번들 로드 및 단일 시퀀스 3-State E2E 추론 무결성 검증
"""

import os
import sys
import glob
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sl2_v1 import SideLungeCondition2ModelV1

def test_pipeline():
    weight_path = os.path.join(CURRENT_DIR, "weights", "sl2_v1_bundle.pkl")
    assert os.path.exists(weight_path), f"Weight bundle not found at: {weight_path}"

    model = SideLungeCondition2ModelV1()
    model.load_bundle(weight_path)

    # 샘플 npz 로드
    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
    samples = glob.glob(os.path.join(npz_dir, "*.npz"))[:5]

    print("=" * 80)
    print("  [sl2_v1] 단일 시퀀스 3-State E2E 추론 테스트")
    print("=" * 80)

    for s_path in samples:
        fname = os.path.basename(s_path)
        data = np.load(s_path)
        kps_3d = data["kps_33_3d_64f"][:, :, :3]  # (64, 33, 3)

        res = model.predict_3state(kps_3d)
        print(f"File: {fname}")
        print(f"  -> Prob: {res['defect_probability']:.4f} | State: {res['state']} | {res['state_label']}")
        print(f"  -> Thresholds: [Clean <= {res['tau_normal']:.4f} < Retake < {res['tau_defect']:.4f} <= Defect]")
        print("-" * 80)

    print("[+] All single sequence inferences passed successfully!")

if __name__ == '__main__':
    test_pipeline()

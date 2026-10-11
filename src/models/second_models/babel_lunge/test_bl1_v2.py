# -*- coding: utf-8 -*-
"""
Unit test for bl1_v2.py and BL1V2Classifier
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_lunge.bl1_v2 import (
    BL1V2Classifier,
    BUNDLE_PKL,
    VIEW_ALIAS_MAP
)


def test_bl1_v2_classifier():
    print("=" * 60)
    print("  [Test] BL1V2Classifier Unit & Inference Test")
    print("=" * 60)

    assert os.path.exists(BUNDLE_PKL), f"번들 파일 없음: {BUNDLE_PKL}"
    clf = BL1V2Classifier(BUNDLE_PKL)
    print("[+] BL1V2Classifier 로드 완료!")
    print(f"  - tau_clean: {clf.tau_clean:.3f}, tau_defect: {clf.tau_defect:.3f}")
    print(f"  - 피처 수: {len(clf.feature_names)}개 (38D SOTA 피처셋)")

    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    assert len(npz_files) > 0, "테스트용 npz 파일 없음"

    sample_npz = npz_files[0]
    data = np.load(sample_npz)
    kps = data['kps_33_3d_64f']
    vis = data['vis_64f'] if 'vis_64f' in data else None
    print(f"[*] 테스트 샘플: {os.path.basename(sample_npz)}")

    for v_code, v_name in VIEW_ALIAS_MAP.items():
        res = clf.predict_sequence(kps, vis, view=v_code)
        assert res['state'] in [0, 1, 2]
        assert 0.0 <= res['defect_prob'] <= 1.0
        assert len(res['feedback_msg']) > 0
        state_str = ["CLEAN (정상)", "DEFECT (결함)", "RETAKE (보류)"][res['state']]
        print(f"  - {v_name} ({v_code}): State={res['state']} ({state_str}) | P={res['defect_prob']:.3f} | 각도={res['lead_knee_min_angle']:.1f}°")

    occluded_vis = np.zeros((64, 33), dtype=np.float32)
    res_occ = clf.predict_sequence(kps, occluded_vis, view='view3')
    assert res_occ['state'] == 2

    print("\n[SUCCESS] BL1V2Classifier 단위 테스트 100% 통과!")


if __name__ == "__main__":
    test_bl1_v2_classifier()

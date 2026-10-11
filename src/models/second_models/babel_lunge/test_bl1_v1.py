# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for bl1_v1.py and BL1V1Classifier
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_lunge.bl1_v1 import (
    BL1V1Classifier,
    BUNDLE_PKL,
    VIEW_ALIAS_MAP
)


def test_bl1_v1_classifier():
    print("=" * 60)
    print("  [Test] BL1V1Classifier Unit & Inference Test")
    print("=" * 60)

    assert os.path.exists(BUNDLE_PKL), f"번들 파일이 존재하지 않습니다: {BUNDLE_PKL}"
    clf = BL1V1Classifier(BUNDLE_PKL)
    print("[+] BL1V1Classifier 로드 완료!")
    print(f"  - tau_clean: {clf.tau_clean:.3f}, tau_defect: {clf.tau_defect:.3f}")
    print(f"  - 피처 수: {len(clf.feature_names)}개")

    # 샘플 파일 탐색
    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    assert len(npz_files) > 0, "테스트용 npz 파일 없음"

    sample_npz = npz_files[0]
    data = np.load(sample_npz)
    kps = data['kps_33_3d_64f']
    vis = data['vis_64f'] if 'vis_64f' in data else None
    print(f"[*] 테스트 샘플: {os.path.basename(sample_npz)}")

    # 1. 5대 시점별 추론 테스트
    print("\n[*] 5대 공식 시점별 추론 테스트 수행:")
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        res = clf.predict_sequence(kps, vis, view=v_code)
        assert res['state'] in [0, 1, 2], f"유효하지 않은 State: {res['state']}"
        assert 0.0 <= res['defect_prob'] <= 1.0, f"확률 범위 오류: {res['defect_prob']}"
        assert len(res['feedback_msg']) > 0, "피드백 메시지 공백"
        state_str = ["CLEAN (정상)", "DEFECT (결함)", "RETAKE (보류)"][res['state']]
        print(f"  - {v_name} ({v_code}): State={res['state']} ({state_str}) | P={res['defect_prob']:.3f} | 각도={res['lead_knee_min_angle']:.1f}°")

    # 2. 센서 게이트 (Tier 0 가림) 테스트
    print("\n[*] Tier 0 센서 게이트 가림 테스트:")
    occluded_vis = np.zeros((64, 33), dtype=np.float32)  # 전신 가림
    res_occ = clf.predict_sequence(kps, occluded_vis, view='view3')
    assert res_occ['state'] == 2, f"가림 시 State 2(RETAKE) 미반환: {res_occ['state']}"
    print(f"  - 센서 불량 시 반환: State={res_occ['state']} | 메시지: {res_occ['feedback_msg']}")

    print("\n[SUCCESS] BL1V1Classifier 단위 테스트를 100% 완벽히 통과하였습니다!")


if __name__ == "__main__":
    test_bl1_v1_classifier()

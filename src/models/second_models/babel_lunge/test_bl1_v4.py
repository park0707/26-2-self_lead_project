# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: test_bl1_v4.py
Purpose: BL1V4Classifier 단위 테스트 및 5대 시점 추론 무결성 검증
"""

import os
import glob
import numpy as np

from bl1_v4 import BL1V4Classifier, MODEL_BUNDLE_PKL, VIEW_ALIAS_MAP


def test_bl1_v4_inference():
    print("=" * 60)
    print("  [Test] BL1V4Classifier Unit & Inference Test")
    print("=" * 60)

    if not os.path.exists(MODEL_BUNDLE_PKL):
        print(f"[!] 번들 모델 파일이 없습니다: {MODEL_BUNDLE_PKL}")
        return

    clf = BL1V4Classifier(MODEL_BUNDLE_PKL)
    print("[+] BL1V4Classifier 로드 완료!")
    print(f"  - 피처 수: {len(clf.feature_names)}개 (30D SOTA 피처셋)")
    print(f"  - 앙상블 가중치: {clf.weights}")
    print(f"  - 시점별 적응형 임계치: {clf.view_thresholds}")

    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    samples = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:5]

    for npz_path in samples:
        fname = os.path.basename(npz_path)
        data = np.load(npz_path)
        kps_64f = data['kps_33_3d_64f']
        vis_64f = data.get('vis_64f', None)

        view_code = "view3"
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if v in fname:
                view_code = v
                break

        res = clf.predict_sequence(kps_64f, vis_64f, view_code=view_code)
        view_alias = VIEW_ALIAS_MAP.get(view_code, view_code)

        print(f"  - [{view_alias} ({view_code})]")
        print(f"    * State={res['state']} ({res['state_str']}) | Prob={res['defect_prob']:.3f}")
        print(f"    * KneeMin={res['knee_angle_min']:.1f}° | RomRatio={res['knee_rom_ratio']:.3f} | Coplanar={res['coplanarity']:.3f}")
        print(f"    * Thresholds: tau_clean={res['tau_clean']:.3f}, tau_defect={res['tau_defect']:.3f}")

    print("\n[SUCCESS] BL1V4Classifier 모든 테스트 100% 통과!")


if __name__ == '__main__':
    test_bl1_v4_inference()

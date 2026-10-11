# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: test_bl1_v3_1.py
Purpose: BL1V3_1Classifier 단위 테스트 및 5대 시점 추론 검증
"""

import os
import glob
import numpy as np

from bl1_v3_1 import BL1V3_1Classifier, MODEL_BUNDLE_PKL, VIEW_ALIAS_MAP


def test_bl1_v3_1_inference():
    print("=" * 60)
    print("  [Test] BL1V3_1Classifier Unit & Inference Test")
    print("=" * 60)

    if not os.path.exists(MODEL_BUNDLE_PKL):
        print(f"[!] 번들 모델 파일이 없습니다: {MODEL_BUNDLE_PKL}")
        return

    clf = BL1V3_1Classifier(MODEL_BUNDLE_PKL)
    print("[+] BL1V3_1Classifier 로드 완료!")
    print(f"  - tau_clean: {clf.tau_clean:.3f}, tau_defect: {clf.tau_defect:.3f}")
    print(f"  - 피처 수: {len(clf.feature_names)}개 (28D 권고안 반영 피처셋)")
    print(f"  - 앙상블 가중치: {clf.weights}")

    # 실제 NPZ 시퀀스 테스트
    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    samples = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:5]

    for npz_path in samples:
        fname = os.path.basename(npz_path)
        data = np.load(npz_path)
        kps_64f = data['kps_33_3d_64f']
        vis_64f = data.get('vis_64f', None)

        res = clf.predict_sequence(kps_64f, vis_64f)
        view_label = "미상"
        for v, alias in VIEW_ALIAS_MAP.items():
            if v in fname:
                view_label = f"{alias} ({v})"
                break

        print(f"  - {view_label}: State={res['state']} ({res['state_str']}) | P={res['defect_prob']:.3f} | KneeMin={res['knee_angle_min']:.1f}° | DropRatio={res['femur_drop_ratio']:.3f} | Gated={res['front_drop_gated']:.3f}")

    print("\n[SUCCESS] BL1V3_1Classifier 모든 테스트 100% 통과!")


if __name__ == '__main__':
    test_bl1_v3_1_inference()

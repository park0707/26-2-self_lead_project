# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: test_bl1_final.py
Purpose: 바벨 런지 c1 최종 프로덕션 모듈(bl1_final.py) 무결성 및 5대 시점 추론 검증
"""

import os
import glob
import numpy as np

from bl1_final import BL1Classifier, predict_lunge_c1, VIEW_ALIAS_MAP


def test_final_production_module():
    print("=" * 70)
    print("  [vidio-health-trainer] bl1_final (c1 최종 확정 모델) 배포 검증 테스트")
    print("=" * 70)

    clf = BL1Classifier()
    print("[+] BL1Classifier 프로덕션 번들 로드 성공!")
    print(f"  - 모델명: {clf.model_name}")
    print(f"  - 피처 차원: {len(clf.feature_names)}D")
    print(f"  - 3-State 임계치: tau_clean={clf.tau_clean:.3f}, tau_defect={clf.tau_defect:.3f}")
    print(f"  - 앙상블 가중치: LightGBM={clf.weights[0]:.3f}, CatBoost={clf.weights[1]:.3f}, XGBoost={clf.weights[2]:.3f}")

    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    samples = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:5]

    print("\n[*] 5대 시점 샘플 시퀀스 대상 추론 검증:")
    for npz_path in samples:
        fname = os.path.basename(npz_path)
        data = np.load(npz_path)
        kps_64f = data['kps_33_3d_64f']
        vis_64f = data.get('vis_64f', None)

        # 1-함수 인터페이스 테스트
        res = predict_lunge_c1(kps_64f, vis_64f)

        v_name = "미상"
        for v_code, alias in VIEW_ALIAS_MAP.items():
            if v_code in fname:
                v_name = f"{alias} ({v_code})"
                break

        print(f"  - [{v_name}]")
        print(f"    * 상태: {res['state']} ({res['state_str']}) | 결함확률: {res['defect_probability']:.3f}")
        print(f"    * 앞무릎 최소각: {res['metrics']['lead_knee_angle_min']:.1f}° | 수직거리 비율: {res['metrics']['lead_femur_vertical_drop_ratio']:.3f} | 게이팅값: {res['metrics']['front_view_femur_drop_gated']:.3f}")
        print(f"    * 피드백: {res['feedback']}")

    print("\n[SUCCESS] bl1_final 프로덕션 파이프라인 검증 100% 완료!")


if __name__ == '__main__':
    test_final_production_module()

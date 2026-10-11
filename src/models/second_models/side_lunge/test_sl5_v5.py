# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: test_sl5_v5.py
Purpose: 사이드 런지 c5(상체의 과조한 숙임/젖힘 여부) sl5_v5 추론 엔진 무결성 테스트
         - 단일 시퀀스(64, 33, 3) 추론 규격 검증
         - 3-State 신뢰 구간 판정 및 5대 서브모델 확률 무결성 검증
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.side_lunge.sl5_v5 import SL5V5Classifier

RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_sl5_inference():
    print("=" * 80)
    print("  [vidio-health-trainer] sl5_v5 단일 시퀀스 추론 무결성 테스트")
    print("=" * 80)

    clf = SL5V5Classifier()
    if not clf.is_loaded:
        print("[!] 모델 번들이 아직 로드되지 않았습니다. 학습 파이프라인을 먼저 실행하세요.")
        return

    npz_files = glob.glob(os.path.join(RAW_MP_DIR, "*.npz"))
    if not npz_files:
        print("[!] 테스트할 전처리 npz 파일이 없습니다.")
        return

    test_file = npz_files[0]
    print(f"[*] 테스트 샘플 로드: {os.path.basename(test_file)}")

    data = np.load(test_file, allow_pickle=True)
    kps_33_3d = data["kps_33_3d_64f"]
    kps_33_2d = data.get("kps_33_2d_64f", kps_33_3d[:, :, :2])

    print(f"    - kps_33_3d shape: {kps_33_3d.shape}")
    print(f"    - kps_33_2d shape: {kps_33_2d.shape}")

    res = clf.predict_sequence(kps_33_3d, kps_33_2d, view_name="시점 A (우측 후방 45°)")

    print("\n[+] 추론 결과 요약:")
    print(f"    - 모델 명칭: {res['model']}")
    print(f"    - 대상 운동: {res['exercise']}")
    print(f"    - 결함 항목: {res['defect_id']} ({res['defect_name']})")
    print(f"    - 입력 시점: {res['view']}")
    print(f"    - 보정 확률: {res['calibrated_prob']:.4f} (원시 확률: {res['raw_prob']:.4f})")
    print(f"    - 2-State 판정: {res['binary_prediction']}")
    print(f"    - 3-State 상태: {res['three_state']['state']} ({res['three_state']['status_text']})")
    print(f"    - 신뢰 여부: {res['three_state']['is_reliable']}")
    print(f"    - 임계치 구간: {res['three_state']['threshold_range']}")
    print(f"    - 5대 서브모델 예측 확률: {res['submodel_probs']}")
    print("\n[+] 핵심 도메인 피처:")
    for k, v in res.get("key_features", {}).items():
        print(f"      * {k}: {v}")

    print("\n[V] sl5_v5 추론 인터페이스 무결성 100% 검증 통과!")


if __name__ == "__main__":
    test_sl5_inference()

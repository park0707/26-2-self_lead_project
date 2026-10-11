# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_d5_v6_inference.py
Purpose: d5 v6.0 최종 확정 모델 단일 시퀀스 실서비스 추론 파이프라인 전수 무결성 검증
"""

import os
import sys
import glob
import numpy as np

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v6 import (
    infer_d5_v6_sequence,
    VIEW_OPTIMAL_THRESHOLDS_V6
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"


def main():
    print("=" * 80)
    print(" [*] [vidio-health-trainer] d5 v6.0 최종 확정 모델 단일 시퀀스 추론 무결성 테스트")
    print("=" * 80)

    # 5대 시점별 샘플 1개씩 추출하여 테스트
    view_samples = {}
    files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))

    for f in files:
        for vn in ["view1", "view2", "view3", "view4", "view5"]:
            if f"_{vn}.npz" in f and vn not in view_samples:
                view_samples[vn] = f
        if len(view_samples) == 5:
            break

    print(f"[*] 5대 시점 샘플 파일 매핑 완료: {len(view_samples)}개 시점\n")

    for vn, fpath in view_samples.items():
        raw = np.load(fpath)
        kps_64f = raw['kps_33_3d_64f']

        res = infer_d5_v6_sequence(kps_64f, view_name=vn, uncertainty_margin=0.06)

        print(f"  시점 [{vn:<5}]: 판정={res['decision_label']} | 원시확률={res['defect_prob']:.3f} | 보정확률={res['calibrated_prob']:.3f} | 확정={res['is_decided']} | 피처차원={res['features_dim']}D")
        assert res['features_dim'] == 37, f"Dimension mismatch: {res['features_dim']} != 37"
        assert res['prediction'] in [0, 1, -1], f"Invalid prediction: {res['prediction']}"
        assert isinstance(res['is_decided'], bool), "is_decided must be bool"

    print("\n[+] 5대 시점 단일 시퀀스 추론 테스트 100% 통과 (규격 완벽 일치!)")


if __name__ == "__main__":
    main()

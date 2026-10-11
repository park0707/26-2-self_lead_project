# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: finalize_and_train.py
Purpose: 77개 전수 베이스 폴더 캐시 취합, d1 v4.1 5-Fold 모델 학습, 공식 Word 보고서 생성 일괄 완수
"""

import os
import sys
import glob
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from extract_deadlift_spinepose_v4_1_64f import (
    INTERMEDIATE_DIR,
    RAW_MP33_DIR,
    OUT_CACHE_PATH,
    OFFICIAL_OUT_CACHE_PATH
)
from train_and_eval_d1_v4_1_spinepose import (
    load_dataset_and_features,
    train_and_eval_5fold_v4_1
)
from generate_d1_v4_1_official_report_docx import (
    build_v4_1_docx_report,
    OUT_DOCX_PATH
)


def main():
    print("=" * 100)
    print("  [Step 1] SpinePose v4.1 385개 멀티뷰 전수 텐서 취합 및 검증")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    all_sids = [os.path.basename(f).replace(".npz", "") for f in npz_files]
    all_base_ids = sorted(list(set(["_".join(s.split("_")[:-1]) for s in all_sids])))
    total_bases = len(all_base_ids)

    results_dict = {}
    for b in all_base_ids:
        chk_p = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if os.path.exists(chk_p):
            c = np.load(chk_p, allow_pickle=True)
            for k in c.files:
                results_dict[k] = c[k]

    N = len(all_sids)
    spine_9_tensor = np.zeros((N, 64, 9, 3), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for i, sid in enumerate(all_sids):
        if sid in results_dict:
            spine_9_tensor[i] = results_dict[sid]
            valid_mask[i] = True
        else:
            print(f"[WARN] 누락: {sid}")

    np.savez_compressed(
        OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=all_sids,
        valid_mask=valid_mask
    )
    np.savez_compressed(
        OFFICIAL_OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=all_sids,
        valid_mask=valid_mask
    )
    print(f"[OK] 최종 전수 척추 캐시 저장 완료: {OUT_CACHE_PATH} (Shape: {spine_9_tensor.shape}, 유효율: {np.mean(valid_mask)*100:.1f}%)")

    print("\n" + "=" * 100)
    print("  [Step 2] 바벨 데드리프트 d1 v4.1 5-Fold StratifiedGroupKFold 재학습 및 평가")
    print("=" * 100)
    data = load_dataset_and_features()
    summary = train_and_eval_5fold_v4_1(data)

    print("\n" + "=" * 100)
    print("  [Step 3] 공식 Word 보고서 (.docx) 최종 배포")
    print("=" * 100)
    build_v4_1_docx_report()
    print(f"[OK] 최종 보고서 배포 확인: {OUT_DOCX_PATH}")

    print("\n" + "=" * 100)
    print("  [ALL SUCCESS] v4.1 파이프라인 학습 및 공식 보고서 배포 완벽 완료!")
    print("=" * 100)


if __name__ == '__main__':
    main()

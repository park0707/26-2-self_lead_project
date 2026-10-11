# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: finish_last_17.py
Purpose: 잔여 17개 베이스 폴더를 no_grad 메모리 누수 방지 상태에서 순차 완주하고,
         385개 전수 캐시 취합 -> d1 v4.1 5-Fold 모델 학습 -> 공식 Word 보고서 (.docx) 최종 배포까지 자동 완결
"""

import os
import sys
import time
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
    process_deadlift_base_folder_v4_1,
    INTERMEDIATE_DIR,
    RAW_MP33_DIR
)
from finalize_and_train import main as finalize_main


def main():
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    all_sids = [os.path.basename(f).replace(".npz", "") for f in npz_files]
    all_base_ids = sorted(list(set(["_".join(s.split("_")[:-1]) for s in all_sids])))

    cached = set([os.path.splitext(f)[0] for f in os.listdir(INTERMEDIATE_DIR)])
    uncached = [b for b in all_base_ids if b not in cached]

    print("=" * 100)
    print(f"  [vidio-health-trainer] 잔여 {len(uncached)}개 베이스 폴더 고속 정밀 완주 (기완료: {len(cached)}/77개)")
    print("=" * 100)

    t_all = time.time()
    for i, b in enumerate(uncached):
        chk = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if os.path.exists(chk):
            print(f"  [{i+1:02d}/{len(uncached):02d}] 기완료 스킵: {b}", flush=True)
            continue
        t0 = time.time()
        res = process_deadlift_base_folder_v4_1(b)
        elapsed = time.time() - t0
        rem_sec = (len(uncached) - (i + 1)) * elapsed
        print(f"  [{i+1:02d}/{len(uncached):02d}] 완료: {b} ({elapsed:.1f}s | 예상 잔여: {rem_sec/60:.1f}분)", flush=True)

    print(f"\n[OK] 77개 전수 베이스 폴더 추출 완료! (소요 시간: {(time.time()-t_all)/60:.1f}분)")
    print("[*] 385개 멀티뷰 전수 캐시 취합, d1 v4.1 모델 재학습 및 공식 Word 보고서 배포 실행 중...")
    finalize_main()


if __name__ == '__main__':
    main()

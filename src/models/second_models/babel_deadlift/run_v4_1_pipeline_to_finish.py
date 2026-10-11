# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: run_v4_1_pipeline_to_finish.py
Purpose: 잔여 27개 베이스 폴더를 무결성 보장 배치(3 워커 + 동기 폴백)로 완수하고,
         385개 전수 캐시 취합 -> d1 v4.1 5-Fold 모델 학습 -> 공식 Word 보고서 (.docx) 최종 배포까지 일괄 완수
"""

import os
import sys
import time
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def step1_extract_remaining():
    print("=" * 100)
    print("  [Step 1/3] SpinePose v4.1 잔여 베이스 폴더 고속 정밀 추출 (3-Worker + Fallback)")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    all_sids = [os.path.basename(f).replace(".npz", "") for f in npz_files]
    all_base_ids = sorted(list(set(["_".join(s.split("_")[:-1]) for s in all_sids])))
    total_bases = len(all_base_ids)

    cached_bases = set([os.path.splitext(f)[0] for f in os.listdir(INTERMEDIATE_DIR)])
    todo_bases = [b for b in all_base_ids if b not in cached_bases]

    print(f"[*] 전체 베이스 폴더: {total_bases}개 | 기 완료: {len(cached_bases)}개 | 잔여 대상: {len(todo_bases)}개")

    t0 = time.time()
    batch_size = 3
    processed_count = len(cached_bases)

    while todo_bases:
        cur_batch = todo_bases[:batch_size]
        todo_bases = todo_bases[batch_size:]

        try:
            with ProcessPoolExecutor(max_workers=min(len(cur_batch), 3)) as executor:
                futures = {executor.submit(process_deadlift_base_folder_v4_1, b): b for b in cur_batch}
                for fut in as_completed(futures):
                    res = fut.result()
                    processed_count += 1
                    elapsed = time.time() - t0
                    speed = (processed_count - len(cached_bases)) / max(1.0, elapsed)
                    rem = (total_bases - processed_count) / max(0.01, speed)
                    print(f"    [{processed_count:02d}/{total_bases:02d}] 완료: {res.get('base_id')} | "
                          f"진행률: {processed_count/total_bases*100:5.1f}% | "
                          f"속도: {speed:.2f} f/s | 잔여: {rem/60:.1f}분", flush=True)
        except Exception as e:
            print(f"    [WARN] 배치 복구 전환 ({e}) -> 동기식 단일 처리 진행", flush=True)
            for b in cur_batch:
                chk = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
                if not os.path.exists(chk):
                    res = process_deadlift_base_folder_v4_1(b)
                    processed_count += 1
                    print(f"    [{processed_count:02d}/{total_bases:02d}] 단일 완료: {res.get('base_id')} | "
                          f"진행률: {processed_count/total_bases*100:5.1f}%", flush=True)

    print(f"\n[OK] 77개 전수 베이스 폴더 SpinePose 추출 100% 완료! (소요 시간: {(time.time()-t0)/60:.1f}분)")

    # 385개 멀티뷰 전수 캐시 취합
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
    print(f"[OK] 최종 전수 척추 캐시 저장 완료: {OUT_CACHE_PATH} (Shape: {spine_9_tensor.shape}, 유효율: {np.mean(valid_mask)*100:.1f}%)\n")


def step2_train_and_eval():
    print("=" * 100)
    print("  [Step 2/3] 바벨 데드리프트 d1 v4.1 5-Fold StratifiedGroupKFold 재학습 및 성능 평가")
    print("=" * 100)
    data = load_dataset_and_features()
    summary = train_and_eval_5fold_v4_1(data)
    return summary


def step3_generate_report():
    print("=" * 100)
    print("  [Step 3/3] 공식 Word 보고서 (.docx) 최종 배포")
    print("=" * 100)
    build_v4_1_docx_report()
    print(f"[OK] 최종 보고서 배포 확인: {OUT_DOCX_PATH}")


def main():
    step1_extract_remaining()
    step2_train_and_eval()
    step3_generate_report()
    print("\n" + "=" * 100)
    print("  [ALL COMPLETED SUCCESSFULLY] d1 v4.1 파이프라인 전수 완료 및 공식 보고서 배포 완료!")
    print("=" * 100)


if __name__ == '__main__':
    main()

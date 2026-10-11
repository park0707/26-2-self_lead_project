# -*- coding: utf-8 -*-
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from extract_deadlift_spinepose_v4_1_64f import process_deadlift_base_folder_v4_1, INTERMEDIATE_DIR

TARGETS = [
    "[정상]_D07_피험자3_Key281",
    "[정상]_D07_피험자4_Key281",
    "[정상]_D07_피험자6_Key281",
    "[정상]_D07_피험자7_Key281",
    "[정상]_D07_피험자8_Key281",
    "[정상]_D11_피험자1_Key281",
    "[정상]_D11_피험자2_Key281",
    "[정상]_D11_피험자3_Key281",
    "[정상]_D11_피험자4_Key281",
    "[정상]_D11_피험자5_Key281",
    "[정상]_D11_피험자6_Key281",
    "[정상]_D11_피험자7_Key281"
]

print(f"[*] Worker Part 1 가동 (총 {len(TARGETS)}개 폴더 처리 시작)")
for i, b in enumerate(TARGETS):
    chk = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
    if os.path.exists(chk):
        print(f"  [P1 {i+1:02d}/{len(TARGETS):02d}] 기완료 스킵: {b}", flush=True)
        continue
    t0 = time.time()
    res = process_deadlift_base_folder_v4_1(b)
    print(f"  [P1 {i+1:02d}/{len(TARGETS):02d}] 완료: {b} ({time.time()-t0:.1f}s)", flush=True)

print("[OK] Worker Part 1 전수 완료!", flush=True)

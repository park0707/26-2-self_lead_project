# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (08_크로스_런지)
Module: build_cross_lunge_reverse_mediapipe33_64f.py
Purpose: 역순 병렬 가속 풀 (남은 폴더 고속 처리)
"""

import os
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.build_cross_lunge_all_mediapipe33_64f import (
    process_folder_task, RAW_IMG_DIR, OUT_MP33_DIR
)


def main():
    all_folders = sorted(os.listdir(RAW_IMG_DIR), reverse=True)
    workers = 4
    print(f"[*] 역순 가속 프로세스 풀 가동 (Workers: {workers}개)...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_folder_task, f): f for f in all_folders}
        for future in tqdm(as_completed(futures), total=len(all_folders), desc="Reverse Acceleration"):
            try:
                future.result()
            except Exception:
                pass
    print("[+] 역순 가속 풀 완료!")


if __name__ == '__main__':
    main()

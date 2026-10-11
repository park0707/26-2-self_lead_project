# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Module: build_stiff_deadlift_full33_64f_dataset.py
Purpose: 원본 바벨 스티프 데드리프트 이미지 시퀀스로부터 MediaPipe 33개 관절 추출 및 64프레임 3차 스플라인 보간
         - 100% 순수 비전 MediaPipe 추출 (AI-Hub JSON 엄격 배제)
         - 9명 전 피험자 및 5대 전 시점(view1~view5) 전수 처리
         - AGENTS.md 공식 시점 매핑 준수:
           시점 A -> view5, 시점 B -> view2, 시점 C -> view3, 시점 D -> view4, 시점 E -> view1
"""

import os
import sys
import glob
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\03_바벨_스티프_데드리프트"
OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"

VIEW_LETTER_TO_VN = {
    'A': 'view5', # 우측 후방 45°
    'B': 'view2', # 우측 전방 45°
    'C': 'view3', # 정면 0°
    'D': 'view4', # 좌측 전방 45°
    'E': 'view1'  # 좌측 후방 45°
}


def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
    """
    (T, J, C) 관절 배열을 64프레임으로 3차 스플라인(Cubic Spline) 보간
    T < 4인 경우 linear 보간
    """
    T, J, C = kps_arr.shape
    if T == 64:
        return kps_arr.astype(np.float32)

    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'

    flat_orig = kps_arr.reshape(T, -1)
    f = interp1d(t_orig, flat_orig, kind=kind, axis=0, fill_value='extrapolate')
    flat_64 = f(t_64)
    return flat_64.reshape(64, J, C).astype(np.float32)


def process_single_folder(folder_name: str) -> Dict[str, Any]:
    """
    단일 폴더(5개 시점, 80장 이미지)에서 5개 시점별 MediaPipe 33관절 64F 시퀀스 추출
    """
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    
    # 기 처리 확인
    all_done = True
    for vl, vn in VIEW_LETTER_TO_VN.items():
        out_p = os.path.join(OUT_MP33_DIR, f"{folder_name}_{vn}.npz")
        if not os.path.exists(out_p):
            all_done = False
            break
    if all_done:
        return {'folder': folder_name, 'saved': 5, 'cached': True}

    seq_dir = os.path.join(RAW_IMG_DIR, folder_name)
    if not os.path.isdir(seq_dir):
        return {'folder': folder_name, 'saved': 0, 'error': 'directory not found'}

    all_files = sorted(os.listdir(seq_dir))
    detector = MediaPipeAdapter.get_detector()
    saved_count = 0

    for vl, vn in VIEW_LETTER_TO_VN.items():
        out_p = os.path.join(OUT_MP33_DIR, f"{folder_name}_{vn}.npz")
        if os.path.exists(out_p):
            saved_count += 1
            continue

        v_files = [f for f in all_files if (f'시점{vl}' in f or f'_{vl}_' in f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(v_files) < 4:
            continue

        raw_33_2d = []
        raw_33_3d = []
        raw_15_2d = []
        raw_15_3d = []

        for vf in v_files:
            img_p = os.path.join(seq_dir, vf)
            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
            raw_33_2d.append(res['kps_33_2d'])
            raw_33_3d.append(res['kps_33_3d'])
            raw_15_2d.append(res['kps_15_2d'])
            raw_15_3d.append(res['kps_15_3d'])

        arr_33_2d = np.array(raw_33_2d, dtype=np.float32)
        arr_33_3d = np.array(raw_33_3d, dtype=np.float32)
        arr_15_2d = np.array(raw_15_2d, dtype=np.float32)
        arr_15_3d = np.array(raw_15_3d, dtype=np.float32)

        spline_33_2d = interpolate_seq_to_64f(arr_33_2d)
        spline_33_3d = interpolate_seq_to_64f(arr_33_3d)
        spline_15_2d = interpolate_seq_to_64f(arr_15_2d)
        spline_15_3d = interpolate_seq_to_64f(arr_15_3d)

        np.savez_compressed(
            out_p,
            kps_33_3d_64f=spline_33_3d,
            kps_33_2d_64f=spline_33_2d,
            kps_15_3d_64f=spline_15_3d,
            kps_15_2d_64f=spline_15_2d
        )
        saved_count += 1

    return {'folder': folder_name, 'saved': saved_count, 'cached': False}


def main():
    start_time = time.time()
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    all_folders = sorted([f for f in os.listdir(RAW_IMG_DIR) if os.path.isdir(os.path.join(RAW_IMG_DIR, f))])
    print(f"[*] Starting MediaPipe Full33 64F extraction for Stiff Deadlift...")
    print(f"[*] Target folders: {len(all_folders)} folders (Expected sequences: {len(all_folders) * 5})")

    # Multi-processing execution with robust max_tasks_per_child
    workers = min(6, os.cpu_count() or 4)
    print(f"[*] Process pool workers: {workers} (max_tasks_per_child=5 for memory safety)")

    completed = 0
    cached = 0
    saved_total = 0

    try:
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=5) as executor:
            futures = {executor.submit(process_single_folder, f): f for f in all_folders}
            for future in tqdm(as_completed(futures), total=len(all_folders), desc="Extracting MP33 64F"):
                try:
                    res = future.result()
                    completed += 1
                    if res.get('cached'):
                        cached += 1
                    saved_total += res.get('saved', 0)
                except Exception as ex:
                    print(f"\n[WARN] Task error on folder: {ex}")
    except Exception as e:
        print(f"\n[WARN] Pool exception: {e}")

    elapsed = time.time() - start_time
    print(f"[+] Completed! Folders processed: {completed} (Cached: {cached})")
    print(f"[+] Total sequence files saved: {saved_total}")
    print(f"[+] Elapsed time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()

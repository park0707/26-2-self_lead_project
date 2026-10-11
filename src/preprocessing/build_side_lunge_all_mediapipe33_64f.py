# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Module: build_side_lunge_all_mediapipe33_64f.py
Purpose: 사이드 런지 전체 826개 폴더(4,130개 시퀀스) 대상 MediaPipe 33관절 64F 전처리 완료
         - 100% 순수 비전 MediaPipe 추출 (AI-Hub JSON 엄격 배제)
         - 9명 전 피험자 및 공식 5대 시점 전수 포함
         - 기 처리 파일(NPZ)은 즉시 스킵(캐시 재사용)
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\07_사이드_런지"
OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"

VIEW_LETTER_TO_VN = {
    'A': 'view5', # 시점 A: 우측 후방 45°
    'B': 'view2', # 시점 B: 우측 전방 45°
    'C': 'view3', # 시점 C: 정면 0°
    'D': 'view4', # 시점 D: 좌측 전방 45°
    'E': 'view1'  # 시점 E: 좌측 후방 45°
}


def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
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


def process_folder_task(folder_name: str) -> Dict[str, Any]:
    os.makedirs(OUT_MP33_DIR, exist_ok=True)

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
        return {'folder': folder_name, 'saved': 0, 'error': 'not found'}

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

        try:
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
        except Exception:
            continue

    return {'folder': folder_name, 'saved': saved_count, 'cached': False}


def main():
    start_time = time.time()
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    all_folders = sorted(os.listdir(RAW_IMG_DIR))
    print(f"[*] 원천 사이드 런지 전체 폴더 수: {len(all_folders)}개 (총 {len(all_folders)*5}개 시퀀스)")

    workers = min(6, os.cpu_count() or 4)
    print("=" * 80)
    print(f"  [vidio-health-trainer] 사이드 런지 전체 826개 폴더 MediaPipe Full33 64F 전처리")
    print(f"  [Workers: {workers}개 병렬 프로세스 풀 가동]")
    print("=" * 80)

    completed = 0
    cached = 0
    saved_total = 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_folder_task, f): f for f in all_folders}
            for future in tqdm(as_completed(futures), total=len(all_folders), desc="Extracting All Side Lunge"):
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
    print(f"\n[+] 전처리 완료! 처리된 폴더: {completed} (캐시 스킵: {cached})")
    print(f"[+] 총 시퀀스: {saved_total}개")
    print(f"[+] 총 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")


if __name__ == '__main__':
    main()

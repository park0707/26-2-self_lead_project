# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_lunge_mediapipe33_64f.py
Purpose: 원본 바벨 런지 이미지 시퀀스로부터 MediaPipe 33개 관절 추출 및 64프레임 3차 스플라인 보간
         - 100% 순수 비전 MediaPipe 추출 (AI-Hub JSON 엄격 배제)
         - 9명 전 피험자 및 5대 전 시점(view1~view5) 전수 처리
         - AGENTS.md 공식 시점 매핑 준수:
           시점 A -> view5, 시점 B -> view2, 시점 C -> view3, 시점 D -> view4, 시점 E -> view1
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

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

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"

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


def select_balanced_folders() -> List[str]:
    """
    피험자 1~9 전원 및 c2 결함/정상이 고르게 포함되도록 균형잡힌 약 220개 폴더(1,100개 시퀀스) 선별
    """
    all_raw = sorted(os.listdir(RAW_IMG_DIR))
    sub_c2_def = defaultdict(list)
    sub_c2_norm = defaultdict(list)
    pure_normals = []

    for f in all_raw:
        json_p = os.path.join(LABEL_JSON_DIR, f"{f}.json")
        if not os.path.exists(json_p):
            continue
        try:
            with open(json_p, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            conds = {c['condition']: c['value'] for c in d.get('type_info', {}).get('conditions', [])}
            is_c2_defect = (conds.get('몸통 발 무릎 방향 일치여부') is False)
            is_pure_normal = all(conds.values())
        except Exception:
            continue

        # 피험자 번호
        sub = 1
        for s in range(1, 10):
            if f"피험자{s}" in f:
                sub = s
                break

        if is_pure_normal:
            pure_normals.append(f)

        if is_c2_defect:
            sub_c2_def[sub].append(f)
        else:
            sub_c2_norm[sub].append(f)

    selected = set()
    # 1. Pure Normal 전수 추가
    for f in pure_normals:
        selected.add(f)

    # 2. 피험자 1~9별 c2 결함 균등 추가 (피험자당 12~14개씩, 총 약 110개)
    for sub in range(1, 10):
        pool = sub_c2_def[sub]
        n_take = min(len(pool), 13)
        for f in pool[:n_take]:
            selected.add(f)

    # 3. 피험자 1~9별 c2 정상 균등 추가 (피험자당 10~12개씩, 총 약 100개)
    for sub in range(1, 10):
        pool = sub_c2_norm[sub]
        n_take = min(len(pool), 11)
        for f in pool[:n_take]:
            selected.add(f)

    res = sorted(list(selected))
    print(f"[*] Balanced selection: {len(res)} folders (Expected {len(res)*5} sequences)")
    return res


def main():
    start_time = time.time()
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    target_folders = select_balanced_folders()

    workers = min(6, os.cpu_count() or 4)
    print(f"[*] Starting MediaPipe Full33 64F extraction for Lunge...")
    print(f"[*] Process pool workers: {workers} (max_tasks_per_child=5)")

    completed = 0
    cached = 0
    saved_total = 0

    try:
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=5) as executor:
            futures = {executor.submit(process_single_folder, f): f for f in target_folders}
            for future in tqdm(as_completed(futures), total=len(target_folders), desc="Extracting Lunge MP33"):
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

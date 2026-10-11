# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - Forward 05 & Backward 06)
Module: build_dynamic_lunge_mediapipe33_64f.py
Purpose: 다이나믹 런지 원본 이미지로부터 MediaPipe 33관절 64프레임 3차 스플라인 전처리 시퀀스 추출
         - 100% 순수 비전 MediaPipe 추출 (AI-Hub JSON 엄격 배제, 무누수 보장)
         - 피험자 1~9번 전원 전수 및 5대 시점(A~E, view1~view5) 전수 포함
         - 스텝 포워드(05) 108폴더 + 스텝 백워드(06) 108폴더 = 총 216폴더 (1,080 시퀀스 균형 데이터셋)
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
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIRS = {
    '05': r"D:\Data\01_raw_data\images\05_스텝_포워드_다이나믹_런지",
    '06': r"D:\Data\01_raw_data\images\06_스텝_백워드_다이나믹_런지"
}

LABEL_JSON_DIRS = {
    '05': r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    '06': r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
}

OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"

VIEW_LETTER_TO_VN = {
    'A': 'view5', # 시점 A: 우측 후방 45°
    'B': 'view2', # 시점 B: 우측 전방 45°
    'C': 'view3', # 시점 C: 정면 0°
    'D': 'view4', # 시점 D: 좌측 전방 45°
    'E': 'view1'  # 시점 E: 좌측 후방 45°
}


def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
    """(T, J, C) 관절 배열을 64프레임으로 3차 스플라인(Cubic Spline) 보간"""
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


def process_single_folder_task(task_args: Tuple[str, str, str]) -> Dict[str, Any]:
    """
    단일 폴더(5개 시점, 80장 이미지)에서 5개 시점별 MediaPipe 33관절 64F 시퀀스 추출
    task_args: (ex_code, raw_img_dir, folder_name)
    """
    ex_code, raw_img_dir, folder_name = task_args
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

    seq_dir = os.path.join(raw_img_dir, folder_name)
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


def select_balanced_folders_integrated() -> List[Tuple[str, str, str]]:
    """
    05(포워드) 및 06(백워드) 양쪽에서 각각 108개 폴더씩 총 216개 폴더 선별
    피험자 1~9 전수 및 c5 결함/정상 고른 균형 유지
    Returns: List of (ex_code, raw_img_dir, folder_name)
    """
    selected_tasks = []

    for ex_code in ['05', '06']:
        raw_dir = RAW_IMG_DIRS[ex_code]
        label_dir = LABEL_JSON_DIRS[ex_code]
        all_raw = sorted(os.listdir(raw_dir))

        sub_c5_def = defaultdict(list)
        sub_c5_norm = defaultdict(list)

        for f in all_raw:
            json_p = os.path.join(label_dir, f"{f}.json")
            if not os.path.exists(json_p):
                continue
            try:
                with open(json_p, 'r', encoding='utf-8', errors='ignore') as fp:
                    d = json.load(fp)
                conds = d.get('type_info', {}).get('conditions', [])
                c5_val = conds[4].get('value')
                is_c5_defect = (c5_val is False)
            except Exception:
                continue

            sub = 1
            for s in range(1, 10):
                if f"피험자{s}" in f:
                    sub = s
                    break

            if is_c5_defect:
                sub_c5_def[sub].append(f)
            else:
                sub_c5_norm[sub].append(f)

        ex_selected = set()
        # 피험자 1~9별로 결함 6개 + 정상 6개씩 선별 = 피험자당 12개 x 9명 = 108개 폴더
        for sub in range(1, 10):
            def_pool = sub_c5_def[sub]
            norm_pool = sub_c5_norm[sub]
            for f in def_pool[:6]:
                ex_selected.add(f)
            for f in norm_pool[:6]:
                ex_selected.add(f)

        ex_list = sorted(list(ex_selected))
        print(f"[*] {ex_code} 선별 완료: {len(ex_list)} 폴더 (예상 {len(ex_list)*5} 시퀀스)")
        for f in ex_list:
            selected_tasks.append((ex_code, raw_dir, f))

    print(f"[*] 전체 선별 완료: 총 {len(selected_tasks)} 폴더 (예상 {len(selected_tasks)*5} 시퀀스)")
    return selected_tasks


def main():
    start_time = time.time()
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    target_tasks = select_balanced_folders_integrated()

    workers = min(6, os.cpu_count() or 4)
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 (포워드+백워드) 216폴더 MediaPipe Full33 64F 전처리")
    print(f"  [Workers: {workers}개 병렬 프로세스 풀 가동]")
    print("=" * 80)

    completed = 0
    cached = 0
    saved_total = 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_folder_task, t): t[2] for t in target_tasks}
            for future in tqdm(as_completed(futures), total=len(target_tasks), desc="Extracting DL MP33"):
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
    print(f"\n[+] Completed! Folders processed: {completed} (Cached: {cached})")
    print(f"[+] Total sequence files saved: {saved_total}")
    print(f"[+] Elapsed time: {elapsed:.2f}s ({elapsed/60:.2f}min)")


if __name__ == '__main__':
    main()

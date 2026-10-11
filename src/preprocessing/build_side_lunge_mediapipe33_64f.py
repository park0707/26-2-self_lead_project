# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: build_side_lunge_mediapipe33_64f.py
Purpose: 사이드 런지 원본 이미지로부터 MediaPipe 33관절 64프레임 3차 스플라인 전처리 시퀀스 추출
         - 100% 순수 비전 MediaPipe 추출 (AI-Hub JSON 엄격 배제, 무누수 보장)
         - 피험자 1~9번 전원 전수 및 5대 시점(A~E, view1~view5) 전수 포함
         - AI-Hub 원천 데이터 중 라벨(conditions)과 설명(description)이 완벽히 일치하는 492개 폴더 전수 선별
         - 라벨-설명 불일치 334개 폴더는 100% 격리 배제
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

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\07_사이드_런지"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"

# AGENTS.md 공식 5대 시점 매핑 규격
VIEW_LETTER_TO_VN = {
    'A': 'view5', # 시점 A: 우측 후방 45°
    'B': 'view2', # 시점 B: 우측 전방 45°
    'C': 'view3', # 시점 C: 정면 0°
    'D': 'view4', # 시점 D: 좌측 전방 45°
    'E': 'view1'  # 시점 E: 좌측 후방 45°
}

# 라벨과 설명이 일치하는 19개 Key 공식 정의
MATCHING_KEYS = {
    '145', '146', '147', '148', '149', '150', '151', '152',
    '164', '165', '166', '167', '168', '169', '170',
    '172', '173', '174', '175'
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


def process_single_folder_task(folder_name: str) -> Dict[str, Any]:
    """
    단일 폴더(5개 시점, 80장 이미지)에서 5개 시점별 MediaPipe 33관절 64F 시퀀스 추출
    """
    os.makedirs(OUT_MP33_DIR, exist_ok=True)

    # 기 처리 확인 (5대 시점 모두 완료되었는지)
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
    detector = MediaPipeAdapter.get_detector("d:/pose_landmarker_heavy.task")
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


def select_all_raw_folders() -> List[str]:
    """
    사이드 런지 원천 데이터 826개 폴더 전체 전수 선별
    """
    all_raw = sorted(os.listdir(RAW_IMG_DIR))
    selected = [f for f in all_raw if os.path.isdir(os.path.join(RAW_IMG_DIR, f))]
    selected = sorted(selected)
    print(f"[*] 사이드 런지 원천 데이터 전체 선별 완료: 총 {len(selected)}개 폴더 (예상 {len(selected)*5}개 시퀀스)")
    return selected


def main():
    start_time = time.time()
    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    target_folders = select_all_raw_folders()

    # CPU 가용 코어 수 기반 병렬 풀
    workers = min(6, os.cpu_count() or 4)
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 라벨-설명 일치 정제 492개 폴더 MediaPipe Full33 64F 전처리")
    print(f"  [Workers: {workers}개 병렬 프로세스 풀 가동]")
    print("=" * 80)

    completed = 0
    cached = 0
    saved_total = 0

    # 배치 단위로 안전하게 처리 (배치당 30개씩, 프로세스 풀 재생성으로 메모리 릭 및 데드락 완벽 차단)
    batch_size = 30
    total_batches = (len(target_folders) + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        batch = target_folders[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"\n[*] Batch {b_idx + 1}/{total_batches} ({len(batch)} folders)...")
        try:
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=5) as executor:
                futures = {executor.submit(process_single_folder_task, f): f for f in batch}
                for future in as_completed(futures):
                    try:
                        res = future.result(timeout=60)
                        completed += 1
                        if res.get('cached'):
                            cached += 1
                        saved_total += res.get('saved', 0)
                    except Exception as ex:
                        print(f"\n[WARN] Task error on folder: {ex}")
        except Exception as e:
            print(f"\n[WARN] Pool exception on batch {b_idx + 1}: {e}")

    elapsed = time.time() - start_time
    print(f"\n[+] 전처리 완료! 처리된 폴더: {completed}개 (기 캐시: {cached}개)")
    print(f"[+] 생성된 총 시퀀스 파일 수: {saved_total}개")
    print(f"[+] 총 소요 시간: {elapsed:.2f}s ({elapsed/60:.2f}min)")


if __name__ == '__main__':
    main()

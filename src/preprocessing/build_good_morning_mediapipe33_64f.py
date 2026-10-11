# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_mediapipe_full33_64f_dataset.py
Purpose: 원천 바벨 스쿼트 이미지 시퀀스로부터 순수 MediaPipe 33개 관절 추출 및 무삭제 64프레임 3차 스플라인 보간 데이터셋 구축
         - AI-Hub 3D JSON 완전 배제 (100% 순수 비전 MediaPipe 추출)
         - 횟수 분할 100% 배제 (원천 시퀀스 64프레임 3차 스플라인 보존 규정 준수)
         - 5대 전 시점(view1~view5) 및 9명 전 피험자 전수 처리
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
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\09_굿모닝"
OUT_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\09_굿모닝"


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

    # (T, J * C)로 flatten 후 보간
    flat_orig = kps_arr.reshape(T, -1)
    f = interp1d(t_orig, flat_orig, kind=kind, axis=0, fill_value='extrapolate')
    flat_64 = f(t_64)
    return flat_64.reshape(64, J, C).astype(np.float32)


def process_sequence_folder(folder_name: str) -> Optional[Dict[str, Any]]:
    """
    단일 원본 폴더(5개 시점 이미지 포함)에서 5개 시점 각각의 MediaPipe 33관절 64F 시퀀스 추출 및 저장
    """
    view_letters = ['A', 'B', 'C', 'D', 'E']
    view_names = ['view1', 'view2', 'view3', 'view4', 'view5']

    # 기 추출 여부 확인 (5개 시점 모두 존재 시 스킵)
    all_exist = True
    for vn in view_names:
        out_p = os.path.join(OUT_MP33_DIR, f"{folder_name}_{vn}.npz")
        if not os.path.exists(out_p):
            all_exist = False
            break
    if all_exist:
        return {'folder': folder_name, 'saved_views': view_names, 'cached': True}

    seq_dir = os.path.join(RAW_IMG_DIR, folder_name)
    if not os.path.isdir(seq_dir):
        return None
    os.makedirs(OUT_MP33_DIR, exist_ok=True)

    all_files = sorted(os.listdir(seq_dir))
    is_correct = 1 if "[정상]" in folder_name or "정상" in folder_name else 0

    # 피험자 번호 파싱
    sub_num = 1
    for d in range(1, 10):
        if f"피험자{d}" in folder_name or f"_{d}_" in folder_name:
            sub_num = d
            break

    detector = MediaPipeAdapter.get_detector()
    saved_views = []

    for vl, vn in zip(view_letters, view_names):
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

        arr_33_2d = np.array(raw_33_2d, dtype=np.float32) # (T, 33, 2)
        arr_33_3d = np.array(raw_33_3d, dtype=np.float32) # (T, 33, 3)
        arr_15_2d = np.array(raw_15_2d, dtype=np.float32) # (T, 15, 2)
        arr_15_3d = np.array(raw_15_3d, dtype=np.float32) # (T, 15, 3)

        # 64프레임 3차 스플라인 보간
        spline_33_2d = interpolate_seq_to_64f(arr_33_2d)
        spline_33_3d = interpolate_seq_to_64f(arr_33_3d)
        spline_15_2d = interpolate_seq_to_64f(arr_15_2d)
        spline_15_3d = interpolate_seq_to_64f(arr_15_3d)

        out_fname = f"{folder_name}_{vn}.npz"
        out_p = os.path.join(OUT_MP33_DIR, out_fname)

        np.savez_compressed(
            out_p,
            kps_33_2d_64f=spline_33_2d,
            kps_33_3d_64f=spline_33_3d,
            kps_15_2d_64f=spline_15_2d,
            kps_15_3d_64f=spline_15_3d,
            kps_33_2d_raw=arr_33_2d,
            kps_33_3d_raw=arr_33_3d,
            base_sequence_id=folder_name,
            sequence_id=f"{folder_name}_{vn}",
            view=vn,
            subject=sub_num,
            label=is_correct,
            raw_frames_count=len(v_files)
        )
        saved_views.append(vn)

    return {'folder': folder_name, 'saved_views': saved_views}


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 원천 이미지 전수 MediaPipe 33관절 64프레임 스플라인 데이터셋 구축")
    print("  - AI-Hub 3D JSON 배제: 100% 순수 MediaPipe Full Pose 추출")
    print("  - 횟수 분할 100% 배제: 원천 시퀀스 64프레임 3차 스플라인 보존 규정 준수")
    print("=" * 115)

    os.makedirs(OUT_MP33_DIR, exist_ok=True)
    all_folders = sorted([f for f in os.listdir(RAW_IMG_DIR) if os.path.isdir(os.path.join(RAW_IMG_DIR, f))])
    N = len(all_folders)
    print(f"[*] 총 처리 대상 원천 시퀀스 폴더 수: {N}개")

    t0 = time.time()
    success = 0
    total_samples = 0
    workers = min(6, os.cpu_count() or 4)
    print(f"[*] 병렬 워커 수: {workers} processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sequence_folder, fn): fn for fn in all_folders}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res and len(res.get('saved_views', [])) > 0:
                    success += 1
                    total_samples += len(res['saved_views'])
            except Exception as e:
                print(f"[!] 에러 발생 ({futures[future]}): {e}")

            if i % 25 == 0 or i == N:
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                print(f"  - 진행률: {i:3d}/{N} ({i/N*100:5.1f}%) | 누적 추출 샘플: {total_samples:4d}개 | 속도: {speed:.2f}폴더/초 | 경과: {elapsed:.1f}초")

    print("\n" + "=" * 115)
    print(f"[+] MediaPipe 33관절 64F 스플라인 데이터셋 구축 완료!")
    print(f"  • 성공 시퀀스 폴더 : {success}/{N}개")
    print(f"  • 생성된 멀티뷰 샘플: {total_samples}개 (5대 전 시점)")
    print(f"  • 저장 위치         : {OUT_MP33_DIR}")
    print("=" * 115)


if __name__ == '__main__':
    main()

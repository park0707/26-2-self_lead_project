# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Module: build_stiff_deadlift_mediapipe33_64f.py
Purpose: 스티프 데드리프트 원천 이미지 시퀀스로부터 100% 순수 비전 MediaPipe 33관절 3D 추출
         및 무삭제 64프레임 3차 스플라인(Cubic Spline) 보간 캐시 데이터셋 구축
         - 피험자 1~9번 전수 (9명)
         - 공식 5대 시점 전수 (시점 A~E: view1~view5)
         - 총 207개 클립 × 5개 시점 = 1,035개 시퀀스 전수 처리
"""

import os
import sys
import glob
import time
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\03_바벨_스티프_데드리프트"
OUT_CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
os.makedirs(OUT_CACHE_DIR, exist_ok=True)

VIEW_LETTERS = ['A', 'B', 'C', 'D', 'E']
VIEW_NAMES = ['view5', 'view2', 'view3', 'view4', 'view1']  # A=view5, B=view2, C=view3, D=view4, E=view1


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


def process_stiff_clip(folder_name: str) -> Optional[int]:
    """단일 클립 폴더(5대 시점 이미지 포함)에서 5개 시점 각각의 MediaPipe 33관절 64F 시퀀스 추출 및 저장"""
    seq_dir = os.path.join(RAW_IMG_DIR, folder_name)
    if not os.path.isdir(seq_dir):
        return None

    # 기 추출 여부 확인
    all_exist = True
    for vn in VIEW_NAMES:
        out_p = os.path.join(OUT_CACHE_DIR, f"{folder_name}_{vn}.npz")
        if not os.path.exists(out_p):
            all_exist = False
            break
    if all_exist:
        return 5

    all_files = sorted(os.listdir(seq_dir))
    detector = MediaPipeAdapter.get_detector()
    saved_count = 0

    for vl, vn in zip(VIEW_LETTERS, VIEW_NAMES):
        out_p = os.path.join(OUT_CACHE_DIR, f"{folder_name}_{vn}.npz")
        if os.path.exists(out_p):
            saved_count += 1
            continue

        v_files = [f for f in all_files if (f'시점{vl}' in f or f'_{vl}_' in f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(v_files) < 4:
            continue

        raw_33_3d = []
        for vf in v_files:
            img_p = os.path.join(seq_dir, vf)
            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
            raw_33_3d.append(res['kps_33_3d'])

        arr_33_3d = np.array(raw_33_3d, dtype=np.float32)
        kps_33_3d_64f = interpolate_seq_to_64f(arr_33_3d)
        np.savez_compressed(out_p, kps_33_3d_64f=kps_33_3d_64f)
        saved_count += 1

    return saved_count


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 MediaPipe 33관절 64F 캐시 빌드 파이프라인")
    print("=" * 80)

    all_folders = sorted(os.listdir(RAW_IMG_DIR))
    print(f"[*] 원천 이미지 클립 총 {len(all_folders)}개 폴더 감지")

    # 신규 추출 대상 확인
    to_process = []
    for fn in all_folders:
        all_done = True
        for vn in VIEW_NAMES:
            if not os.path.exists(os.path.join(OUT_CACHE_DIR, f"{fn}_{vn}.npz")):
                all_done = False
                break
        if not all_done:
            to_process.append(fn)

    print(f"[*] 기 완료 클립: {len(all_folders) - len(to_process)}개 | 처리 필요 클립: {len(to_process)}개")

    if to_process:
        t0 = time.time()
        # 12코어 중 8개 워커 활용하여 안정적인 최대 처리량 확보
        workers = min(8, os.cpu_count() or 4)
        print(f"[*] 병렬 추출 시작 (워커 수: {workers})...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_stiff_clip, fn): fn for fn in to_process}
            done = 0
            for future in as_completed(futures):
                fn = futures[future]
                try:
                    res = future.result()
                    done += 1
                    if done % 10 == 0 or done == len(to_process):
                        elapsed = time.time() - t0
                        speed = done / elapsed
                        remain = (len(to_process) - done) / max(speed, 1e-4)
                        print(f"  [{done}/{len(to_process)}] {fn} 완료 ({res}개 시점 저장) | 경과: {elapsed/60:.1f}분 | 남은 예상: {remain/60:.1f}분")
                except Exception as e:
                    print(f"[!] {fn} 처리 중 오류: {e}")

        print(f"\n[+] 전처리 추출 완료! (총 소요 시간: {(time.time()-t0)/60:.1f}분)")

    # 최종 캐시 검증
    all_npz = glob.glob(os.path.join(OUT_CACHE_DIR, "*.npz"))
    print(f"[+] 최종 생성된 64F 시퀀스 npz 수: {len(all_npz)}개 (기대치: {len(all_folders)*5}개)")
    print(f"    - 저장 위치: {OUT_CACHE_DIR}")


if __name__ == "__main__":
    main()

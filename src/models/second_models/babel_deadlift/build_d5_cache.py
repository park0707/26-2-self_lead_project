# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: build_d5_cache.py
Purpose: d5 모델 학습 및 검증을 위한 50:50 클래스 균형 770개 시퀀스(154개 클립) 
         MediaPipe 33관절 64프레임 3차 스플라인 캐시 구축
         - 피험자 1~9번 전원 전수 (각 피험자별 동일한 정상/결함 비율)
         - 공식 5대 시점(시점 A~E: view1~view5) 전수 포함
"""

import os
import sys
import glob
import json
import time
import shutil
from typing import Dict, List, Tuple, Optional, Any
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

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
EXISTING_CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
OUT_D5_CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
os.makedirs(OUT_D5_CACHE_DIR, exist_ok=True)

VIEW_LETTERS = ['A', 'B', 'C', 'D', 'E']
VIEW_NAMES = ['view5', 'view2', 'view3', 'view4', 'view1']  # A=view5, B=view2, C=view3, D=view4, E=view1


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


def extract_clip_views(folder_name: str) -> Optional[int]:
    """단일 클립 폴더에 대해 5대 시점 관절 추출 및 npz 저장"""
    seq_dir = os.path.join(RAW_IMG_DIR, folder_name)
    if not os.path.isdir(seq_dir):
        return None

    all_files = sorted(os.listdir(seq_dir))
    detector = MediaPipeAdapter.get_detector()
    saved_count = 0

    for vl, vn in zip(VIEW_LETTERS, VIEW_NAMES):
        out_p = os.path.join(OUT_D5_CACHE_DIR, f"{folder_name}_{vn}.npz")
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


def select_balanced_dataset():
    """피험자 1~9번별 정상 77클립과 결함 77클립을 완벽 매칭하여 총 154클립 선별"""
    existing_npz = glob.glob(os.path.join(EXISTING_CACHE_DIR, "*.npz"))
    existing_clips = sorted(list(set(["_".join(os.path.basename(p).replace(".npz", "").split("_")[:-1]) for p in existing_npz])))
    print(f"[*] 기존 정상 캐시 클립 수: {len(existing_clips)}개")

    # 기존 정상 클립들을 d5 캐시 폴더로 복사/확인
    copied_clean = 0
    clean_by_sub = {i: [] for i in range(1, 10)}
    for p in existing_npz:
        fn = os.path.basename(p)
        clip_name = "_".join(fn.replace(".npz", "").split("_")[:-1])
        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in clip_name:
                sub = d
                break
        if clip_name not in clean_by_sub[sub]:
            clean_by_sub[sub].append(clip_name)

        dst_p = os.path.join(OUT_D5_CACHE_DIR, fn)
        if not os.path.exists(dst_p):
            shutil.copy2(p, dst_p)
            copied_clean += 1

    print(f"[+] 정상 클립 복사/동기화 완료: {len(existing_npz)}개 (신규 복사 {copied_clean}개)")
    print("[*] 피험자별 정상 클립 수:", {k: len(v) for k, v in clean_by_sub.items()})

    # 피험자별 d5 결함(False) 클립 선별
    all_labels = glob.glob(os.path.join(LABEL_JSON_DIR, "*.json"))
    defects_by_sub = {i: [] for i in range(1, 10)}
    for lp in all_labels:
        clip_name = os.path.basename(lp).replace(".json", "")
        img_p = os.path.join(RAW_IMG_DIR, clip_name)
        if not os.path.isdir(img_p):
            continue

        with open(lp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        conds = {c["condition"]: c["value"] for c in meta.get("type_info", {}).get("conditions", [])}
        if not conds.get("바벨 궤적과 몸 밀착", True):  # 결함 발생
            sub = 1
            for d in range(1, 10):
                if f"피험자{d}" in clip_name:
                    sub = d
                    break
            defects_by_sub[sub].append(clip_name)

    # 정상 클립 수와 1:1로 결함 클립 샘플링
    selected_defect_clips = []
    for sub in range(1, 10):
        target_count = len(clean_by_sub[sub])
        available = defects_by_sub[sub]
        sampled = available[:target_count]
        selected_defect_clips.extend(sampled)
        print(f"  - 피험자 {sub}: 정상 {target_count}개 vs 결함 {len(sampled)}개 (보유 결함 풀: {len(available)}개)")

    print(f"[*] 총 선정된 결함 클립 수: {len(selected_defect_clips)}개 (예상 멀티뷰 시퀀스: {len(selected_defect_clips)*5}개)")
    return selected_defect_clips


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d5 전용 50:50 균형 캐시 구축 파이프라인")
    print("=" * 80)

    selected_defects = select_balanced_dataset()

    # 결함 클립 추출 시작
    to_extract = []
    for fn in selected_defects:
        all_done = True
        for vn in VIEW_NAMES:
            if not os.path.exists(os.path.join(OUT_D5_CACHE_DIR, f"{fn}_{vn}.npz")):
                all_done = False
                break
        if not all_done:
            to_extract.append(fn)

    print(f"[*] 신규 추출 필요 결함 클립 수: {len(to_extract)}개 (이미 완료: {len(selected_defects) - len(to_extract)}개)")

    if to_extract:
        t0 = time.time()
        workers = min(6, os.cpu_count() or 4)
        print(f"[*] MediaPipe 33관절 64프레임 병렬 추출 시작 (워커 수: {workers})...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_clip_views, fn): fn for fn in to_extract}
            done = 0
            for future in as_completed(futures):
                fn = futures[future]
                try:
                    res = future.result()
                    done += 1
                    if done % 10 == 0 or done == len(to_extract):
                        print(f"  [{done}/{len(to_extract)}] {fn} 완료 ({res}개 시점 저장)")
                except Exception as e:
                    print(f"[!] {fn} 처리 중 오류 발생: {e}")

        print(f"[+] 결함 클립 전처리 추출 완료! (경과 시간: {time.time()-t0:.1f}초)")

    # 전체 캐시 무결성 검증
    all_cached_npz = glob.glob(os.path.join(OUT_D5_CACHE_DIR, "*.npz"))
    print(f"\n[+] d5 캐시 구축 최종 결과: 총 {len(all_cached_npz)}개 시퀀스 확보 완료!")
    print(f"    - 저장 위치: {OUT_D5_CACHE_DIR}")


if __name__ == "__main__":
    main()

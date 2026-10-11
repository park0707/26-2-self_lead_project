# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: extract_true_spinepose_64f.py
Purpose: 원본 바벨 스쿼트 이미지에서 SpinePose(37 Kps & 3D Lifting)를 직접 구동하여
         2,605개 전수 시퀀스에 대한 순수 비전 9대 척추 분절 (64, 9, 3) 3D 좌표를 추출하고 캐싱
- 주요 최적화:
  1) 521개 base action 폴더 단위로 묶어 멀티프로세싱 오버헤드 최소화
  2) MediaPipe 2D 관절점으로부터 바운딩 박스를 즉각 산출하여 무거운 RF-DETR 탐지기 스킵
  3) ProcessPoolExecutor 기반 병렬 가속 (CPU 코어 8개 활용)
  4) 16개 원본 프레임 추론 후 64프레임 3차 스플라인(Cubic Spline) 보간 표준 엄격 적용
  5) 중간 체크포인트 저장 지원으로 안전한 복구 보장
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

# SpinePose 공식 37개 랜드마크 중 척추 체인 9개 인덱스
# 19: S1 (hip/sacrum), 26: L5, 27: L3, 28: L1, 29: T8, 30: T3, 18: C7 (neck), 35: C4, 36: C1
SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
VIEW_LETTER_MAP = {'view1': 'A', 'view2': 'B', 'view3': 'C', 'view4': 'D', 'view5': 'E'}
VIEW_LETTER_TO_VN = {'A': 'view1', 'B': 'view2', 'C': 'view3', 'D': 'view4', 'E': 'view5'}

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\01_바벨_스쿼트"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"
INTERMEDIATE_DIR = r"D:\Data\피처_데이터\temp_spinepose_cache"


def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
    """(T, J, C) 관절 배열을 64프레임으로 3차 스플라인 보간"""
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


# 전역 추론기 인스턴스 (프로세스별 1회 지연 초기화)
_WORKER_ESTIMATOR = None

def get_worker_estimator():
    global _WORKER_ESTIMATOR
    if _WORKER_ESTIMATOR is None:
        from spinepose import SpinePoseEstimator
        _WORKER_ESTIMATOR = SpinePoseEstimator(mode="small", enable_lifting=True)
    return _WORKER_ESTIMATOR


def process_base_action_folder(base_id: str) -> dict:
    """
    단일 base action 폴더(5개 시점 이미지 포함)를 처리하여 5개 시점 각각의 SpinePose 9대 척추 64F 좌표 반환
    """
    chk_p = os.path.join(INTERMEDIATE_DIR, f"{base_id}.npz")
    if os.path.exists(chk_p):
        try:
            c = np.load(chk_p, allow_pickle=True)
            return {'base_id': base_id, 'results': {k: c[k] for k in c.files}, 'cached': True, 'success': True}
        except Exception:
            pass

    seq_img_dir = os.path.join(RAW_IMG_DIR, base_id)
    if not os.path.isdir(seq_img_dir):
        return {'base_id': base_id, 'success': False, 'error': f'Img dir not found: {seq_img_dir}'}

    all_imgs = sorted([f for f in os.listdir(seq_img_dir) if f.lower().endswith(('.jpg', '.png'))])
    if len(all_imgs) < 10:
        return {'base_id': base_id, 'success': False, 'error': f'Too few images in {base_id}: {len(all_imgs)}'}

    estimator = get_worker_estimator()
    folder_results = {}

    for vl, vn in VIEW_LETTER_TO_VN.items():
        v_imgs = [f for f in all_imgs if (f'시점{vl}' in f or f'_{vl}_' in f)]
        if len(v_imgs) < 4:
            continue

        # 대응하는 MediaPipe npz에서 bbox 계산용 2D 관절점 로드
        mp_p = os.path.join(RAW_MP33_DIR, f"{base_id}_{vn}.npz")
        kps_33_2d_raw = None
        if os.path.exists(mp_p):
            try:
                mp_d = np.load(mp_p, allow_pickle=True)
                kps_33_2d_raw = mp_d['kps_33_2d_raw']
            except Exception:
                pass

        raw_spine_9_3d = []
        for t_idx, img_f in enumerate(v_imgs):
            img_p = os.path.join(seq_img_dir, img_f)
            with open(img_p, 'rb') as f:
                buf = f.read()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]

            if kps_33_2d_raw is not None and t_idx < len(kps_33_2d_raw):
                k2d = kps_33_2d_raw[t_idx]
                x_min = max(0.0, float(np.min(k2d[:, 0]) - (np.max(k2d[:, 0]) - np.min(k2d[:, 0])) * 0.15))
                x_max = min(float(w), float(np.max(k2d[:, 0]) + (np.max(k2d[:, 0]) - np.min(k2d[:, 0])) * 0.15))
                y_min = max(0.0, float(np.min(k2d[:, 1]) - (np.max(k2d[:, 1]) - np.min(k2d[:, 1])) * 0.15))
                y_max = min(float(h), float(np.max(k2d[:, 1]) + (np.max(k2d[:, 1]) - np.min(k2d[:, 1])) * 0.10))
            else:
                x_min, y_min, x_max, y_max = w * 0.15, h * 0.05, w * 0.85, h * 0.95

            bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
            res = estimator.estimate(img, bbox)
            kps_2d, scores, kps_3d = res
            spine_9_frame = kps_3d[0, SPINEPOSE_9_INDICES] # (9, 3)
            raw_spine_9_3d.append(spine_9_frame)

        arr_raw_9_3d = np.array(raw_spine_9_3d, dtype=np.float32)
        spline_9_3d_64f = interpolate_seq_to_64f(arr_raw_9_3d) # (64, 9, 3)
        seq_id = f"{base_id}_{vn}"
        folder_results[seq_id] = spline_9_3d_64f

    # 중간 저장 (체크포인트)
    if folder_results:
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        np.savez_compressed(chk_p, **folder_results)

    return {'base_id': base_id, 'results': folder_results, 'cached': False, 'success': True}


def main():
    print("=" * 110)
    print("  [vidio-health-trainer] SpinePose 순수 비전 9대 척추 분절 64F 시계열 전수 추출 파이프라인")
    print("  - 대상: 바벨 스쿼트 521개 base action 폴더 (총 2,605개 시퀀스)")
    print("  - 모델: SpinePose (Small) + 3D Lifter")
    print("  - 가속: MediaPipe BBox 직주입 + 521개 단위 멀티프로세스 병렬화")
    print("=" * 110)

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    master_path = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
    master = np.load(master_path, allow_pickle=True)
    master_sids = master['sequence_ids']
    master_base_sids = sorted(list(set(master['base_sequence_ids'])))
    total_bases = len(master_base_sids)
    print(f"[*] 총 base action 폴더 수: {total_bases}개 (2,605 시퀀스)")

    max_workers = 3
    print(f"[*] 병렬 워커 수: {max_workers} 프로세스 가동 (max_tasks_per_child=10)")

    # 아직 처리되지 않은 폴더만 선별
    remaining_bases = []
    for b in master_base_sids:
        chk_p = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if not os.path.exists(chk_p):
            remaining_bases.append(b)

    cached_count = total_bases - len(remaining_bases)
    print(f"[*] 기존 완료된 체크포인트: {cached_count}/{total_bases}개 | 남은 폴더: {len(remaining_bases)}개")

    t0 = time.time()
    chunk_size = 30
    processed_count = cached_count

    # 청크 단위로 나누어 안전하게 실행 (메모리 완벽 회수 및 장애 격리)
    for c_idx in range(0, len(remaining_bases), chunk_size):
        chunk = remaining_bases[c_idx:c_idx + chunk_size]
        print(f"\n[+] 청크 {c_idx//chunk_size + 1}/{(len(remaining_bases)+chunk_size-1)//chunk_size} 처리 시작 ({len(chunk)}개 폴더)...", flush=True)

        retry = 0
        while retry < 3:
            try:
                with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=10) as executor:
                    futures = {executor.submit(process_base_action_folder, b): b for b in chunk}
                    for fut in as_completed(futures):
                        res = fut.result()
                        processed_count += 1
                        elapsed = time.time() - t0
                        speed = (processed_count - cached_count) / max(1.0, elapsed)
                        rem_sec = (total_bases - processed_count) / max(0.01, speed)
                        if processed_count % 5 == 0 or processed_count == total_bases:
                            print(f"    - 진행률: {processed_count}/{total_bases} ({processed_count/total_bases*100:.1f}%) | "
                                  f"속도: {speed:.2f} folder/s | 남은 시간: {rem_sec/60:.1f}분", flush=True)
                break
            except Exception as e:
                retry += 1
                print(f"[!] 청크 에러 발생 ({e}) -> 3초 후 재시도 ({retry}/3)...", flush=True)
                time.sleep(3)

    print(f"[*] 전수 폴더 추출 완료! 총 소요 시간: {(time.time()-t0)/60:.1f}분")

    # 전체 체크포인트 파일들을 로드하여 results_dict 구성
    results_dict = {}
    for b in master_base_sids:
        chk_p = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if os.path.exists(chk_p):
            try:
                c = np.load(chk_p, allow_pickle=True)
                for k in c.files:
                    results_dict[k] = c[k]
            except Exception:
                pass

    # Master 파일 순서와 동일하게 정렬하여 텐서 구성
    N_master = len(master_sids)
    spine_9_tensor = np.zeros((N_master, 64, 9, 3), dtype=np.float32)
    valid_mask = np.zeros(N_master, dtype=bool)

    for i, sid in enumerate(master_sids):
        if sid in results_dict:
            spine_9_tensor[i] = results_dict[sid]
            valid_mask[i] = True
        else:
            print(f"[WARN] 시퀀스 {sid} SpinePose 누락 -> 기본값 보정")

    os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=master_sids,
        valid_mask=valid_mask
    )
    print(f"[OK] 최종 순수 비전 척추 캐시 저장 완료: {OUT_CACHE_PATH} (Shape: {spine_9_tensor.shape}, 유효율: {np.mean(valid_mask)*100:.1f}%)")


if __name__ == '__main__':
    main()

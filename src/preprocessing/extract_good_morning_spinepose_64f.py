# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 굿모닝 (Barbell Good Morning)
Module: extract_good_morning_spinepose_64f.py
Purpose: 원본 바벨 굿모닝 이미지에서 SpinePose(37 Kps & 3D Lifting)를 직접 구동하여
         전수 시퀀스에 대한 순수 비전 9대 척추 분절 (64, 9, 3) 3D 좌표를 추출하고,
         One-Euro Filter + 생체역학적 세그먼트 클램핑(Denoising)을 적용하여 캐싱
         - 100% 순수 비전 좌표 (AI-Hub 3D JSON 엄격 배제 및 데이터 누수 영구 차단)
         - AGENTS.md 공식 5대 시점 매핑 준수:
           시점 A -> view5, 시점 B -> view2, 시점 C -> view3, 시점 D -> view4, 시점 E -> view1
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.spinepose_denoiser import apply_one_euro_filter_seq, apply_biomechanical_segment_constraint

# SpinePose 공식 37개 랜드마크 중 척추 체인 9개 인덱스
# 19: S1 (hip/sacrum), 26: L5, 27: L3, 28: L1, 29: T8, 30: T3, 18: C7 (neck), 35: C4, 36: C1
SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]

# AGENTS.md 규정: 시점 A: view5, B: view2, C: view3, D: view4, E: view1
VIEW_LETTER_TO_VN = {
    'A': 'view5', # 우측 후방 45°
    'B': 'view2', # 우측 전방 45°
    'C': 'view3', # 정면 0°
    'D': 'view4', # 좌측 전방 45°
    'E': 'view1'  # 좌측 후방 45°
}

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\09_굿모닝"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\09_굿모닝"
INTERMEDIATE_DIR = r"D:\Data\피처_데이터\temp_good_morning_spinepose_cache"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\good_morning_spinepose_denoised_64f.npz"


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


_WORKER_ESTIMATOR = None

def get_worker_estimator():
    global _WORKER_ESTIMATOR
    if _WORKER_ESTIMATOR is None:
        from spinepose import SpinePoseEstimator
        _WORKER_ESTIMATOR = SpinePoseEstimator(mode="small", enable_lifting=True)
    return _WORKER_ESTIMATOR


def process_stiff_base_folder(base_id: str) -> dict:
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

        raw_spine_9_3d = []
        for t_idx, img_f in enumerate(v_imgs):
            img_p = os.path.join(seq_img_dir, img_f)
            with open(img_p, 'rb') as f:
                buf = f.read()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            # BBox 산출 (중앙 인체 영역)
            x_min, y_min, x_max, y_max = w * 0.15, h * 0.05, w * 0.85, h * 0.95
            bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

            res = estimator.estimate(img, bbox)
            kps_2d, scores, kps_3d = res
            spine_9_frame = kps_3d[0, SPINEPOSE_9_INDICES]  # (9, 3)
            raw_spine_9_3d.append(spine_9_frame)

        if len(raw_spine_9_3d) < 4:
            continue

        arr_raw_9_3d = np.array(raw_spine_9_3d, dtype=np.float32)
        spline_9_3d_64f = interpolate_seq_to_64f(arr_raw_9_3d)  # (64, 9, 3)

        # One-Euro 필터링 및 뼈 길이 불변 제약 적용 (Denoising)
        denoised_seq = apply_one_euro_filter_seq(spline_9_3d_64f, fps=30.0)
        constrained_seq = apply_biomechanical_segment_constraint(denoised_seq)

        seq_id = f"{base_id}_{vn}"
        folder_results[seq_id] = constrained_seq

    if folder_results:
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        np.savez_compressed(chk_p, **folder_results)

    return {'base_id': base_id, 'results': folder_results, 'cached': False, 'success': True}


def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 굿모닝 SpinePose 순수 비전 9대 척추 분절 64F 전수 추출")
    print(f"  - 대상: 굿모닝 207개 base action 폴더 (총 1,035개 멀티뷰 시퀀스)")
    print("  - 모델: SpinePose (Small) + 3D Lifter + One-Euro Filter + Biomechanical Constraint")
    print("=" * 110)

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    all_base_ids = sorted([f for f in os.listdir(RAW_IMG_DIR) if os.path.isdir(os.path.join(RAW_IMG_DIR, f))])
    total_bases = len(all_base_ids)
    all_sids = []
    for b in all_base_ids:
        for vn in VIEW_LETTER_TO_VN.values():
            all_sids.append(f"{b}_{vn}")
    print(f"[*] 총 분석 대상 base 폴더: {total_bases}개 (시퀀스 {len(all_sids)}개)")

    print(f"[*] 안정적이고 신뢰도 높은 단일 프로세스 엔진 가동 (데드락 원천 차단)")

    t0 = time.time()
    processed_count = 0

    # SpinePose 인스턴스 1회 로드 후 재사용
    estimator = get_worker_estimator()

    for b in all_base_ids:
        try:
            res = process_stiff_base_folder(b)
            processed_count += 1
            elapsed = time.time() - t0
            speed = processed_count / max(1.0, elapsed)
            rem_sec = (total_bases - processed_count) / max(0.01, speed)
            if processed_count % 10 == 0 or processed_count == total_bases:
                print(f"    - 진행률: {processed_count}/{total_bases} ({processed_count/total_bases*100:.1f}%) | "
                      f"속도: {speed:.2f} folder/s | 남은 시간: {rem_sec/60:.1f}분", flush=True)
        except Exception as ex:
            print(f"\n[WARN] Error processing {b}: {ex}", flush=True)

    print(f"[*] SpinePose 추출 및 노이즈 제거 완료! (소요 시간: {(time.time()-t0)/60:.1f}분)")

    # 전체 캐시 취합
    results_dict = {}
    for b in all_base_ids:
        chk_p = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if os.path.exists(chk_p):
            c = np.load(chk_p, allow_pickle=True)
            for k in c.files:
                results_dict[k] = c[k]

    N = len(all_sids)
    spine_9_tensor = np.zeros((N, 64, 9, 3), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for i, sid in enumerate(all_sids):
        if sid in results_dict:
            spine_9_tensor[i] = results_dict[sid]
            valid_mask[i] = True
        else:
            print(f"[WARN] 누락: {sid}")

    os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=all_sids,
        valid_mask=valid_mask
    )
    print(f"\n[OK] 최종 굿모닝 SpinePose 청정 척추 캐시 저장 완료: {OUT_CACHE_PATH} (Shape: {spine_9_tensor.shape}, 유효율: {np.mean(valid_mask)*100:.1f}%)")


if __name__ == '__main__':
    main()

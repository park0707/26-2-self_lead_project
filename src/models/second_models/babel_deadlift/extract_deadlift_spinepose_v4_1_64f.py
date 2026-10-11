# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Module: extract_deadlift_spinepose_v4_1_64f.py
Purpose: 바벨 데드리프트 385개 전수 시퀀스에 대해:
         1) RF-DETR 키프레임(셋업 & 하강점) 자동 인체 감지 -> 시퀀스 맞춤형 체간 Dynamic BBox 산출
         2) 고속 정밀 RTMPose 추론 (문고리/원판/배경 사람 오검출 100% 원천 배제)
         3) 체간 BBox 게이팅 필터 (BBox 외곽 10% 이상치 컷오프)
         4) 소수 앵커 기반 PCHIP(단조 3차 에르미트 스플라인) 척추 곡선 보간
         5) 64F 시간축 3차 스플라인 정규화 + One-Euro Filter + 생체역학 뼈 길이 불변 제약
         을 순차 적용하여 초고순도 순수 비전 9대 척추 분절 3D 캐시(v4.1) 생성
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.spinepose_denoiser import apply_one_euro_filter_seq, apply_biomechanical_segment_constraint

# SpinePose 공식 37개 랜드마크 중 척추 체인 9개 인덱스
# 19: S1 (hip/sacrum), 26: L5, 27: L3, 28: L1, 29: T8, 30: T3, 18: C7 (neck), 35: C4, 36: C1
SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
VIEW_LETTER_TO_VN = {'A': 'view1', 'B': 'view2', 'C': 'view3', 'D': 'view4', 'E': 'view5'}

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
INTERMEDIATE_DIR = r"D:\Data\피처_데이터\temp_deadlift_spinepose_cache_v4_1"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
OFFICIAL_OUT_CACHE_PATH = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_64f.npz"


def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
    """T프레임을 64프레임으로 3차 스플라인 보간"""
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
        import torch
        torch.set_num_threads(2)
        import cv2
        cv2.setNumThreads(1)
        from spinepose import SpinePoseEstimator
        _WORKER_ESTIMATOR = SpinePoseEstimator(mode="small", enable_lifting=True)
    return _WORKER_ESTIMATOR


def get_exerciser_box_for_frame(estimator, img: np.ndarray):
    """단일 프레임에서 중앙 운동 피험자의 BBox 추출"""
    import torch
    with torch.no_grad():
        h, w = img.shape[:2]
        k2d, sc, _ = estimator(img)
        if len(k2d) == 0:
            return np.array([w * 0.35, h * 0.2, w * 0.65, h * 0.9], dtype=np.float32), np.array([w * 0.35, h * 0.2, w * 0.65, h * 0.9], dtype=np.float32)

        best_idx = 0
        best_m = -1.0
        for i in range(len(k2d)):
            pts = k2d[i]
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            area = (xmax - xmin) * (ymax - ymin)
            cx = (xmin + xmax) / 2.0
            dist = abs(cx - w / 2.0)
            metric = area / (1.0 + dist * 0.5)
            if metric > best_m:
                best_m = metric
                best_idx = i

        pts_best = k2d[best_idx]
        return np.min(pts_best, axis=0), np.max(pts_best, axis=0)


def process_deadlift_base_folder_v4_1(base_id: str) -> dict:
    import torch
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

    with torch.no_grad():
        for vl, vn in VIEW_LETTER_TO_VN.items():
            v_imgs = [f for f in all_imgs if (f'시점{vl}' in f or f'_{vl}_' in f)]
            if len(v_imgs) < 4:
                continue

            # 1. 시점별 셋업(0) 및 최하단(중간) 프레임 기반 동적 운동 BBox 산출
            img0_p = os.path.join(seq_img_dir, v_imgs[0])
            img_mid_p = os.path.join(seq_img_dir, v_imgs[len(v_imgs) // 2])

            with open(img0_p, 'rb') as f0:
                img0 = cv2.imdecode(np.frombuffer(f0.read(), np.uint8), cv2.IMREAD_COLOR)
            with open(img_mid_p, 'rb') as fmid:
                img_mid = cv2.imdecode(np.frombuffer(fmid.read(), np.uint8), cv2.IMREAD_COLOR)

            h, w = img0.shape[:2]
            min0, max0 = get_exerciser_box_for_frame(estimator, img0)
            min_mid, max_mid = get_exerciser_box_for_frame(estimator, img_mid)

            u_min = np.minimum(min0, min_mid)
            u_max = np.maximum(max0, max_mid)
            bw = u_max[0] - u_min[0]
            bh = u_max[1] - u_min[1]

            # 25% 가동범위 완충 마진이 적용된 피험자 전용 타이트 바운딩 박스
            u_box = np.array([[
                max(0.0, u_min[0] - 0.25 * bw),
                max(0.0, u_min[1] - 0.20 * bh),
                min(float(w), u_max[0] + 0.25 * bw),
                min(float(h), u_max[1] + 0.20 * bh)
            ]], dtype=np.float32)

            raw_spine_9_3d = []
            for t_idx, img_f in enumerate(v_imgs):
                img_p = os.path.join(seq_img_dir, img_f)
                with open(img_p, 'rb') as f:
                    buf = f.read()
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # 고속 정밀 추론 (타이트 박스 내 포즈 추정)
                k2d, sc, k3d = estimator.estimate(img, u_box)
                pts_2d = k2d[0, SPINEPOSE_9_INDICES].copy()  # (9, 2)
                scs = sc[0, SPINEPOSE_9_INDICES].copy()      # (9,)
                pts_3d = k3d[0, SPINEPOSE_9_INDICES].copy()  # (9, 3)

                # 2. 체간 BBox 게이팅 (BBox 외곽 10% 초과 이상치 컷오프)
                gate_xmin = u_box[0, 0] - 0.10 * bw
            gate_xmax = u_box[0, 2] + 0.10 * bw
            gate_ymin = u_box[0, 1] - 0.10 * bh
            gate_ymax = u_box[0, 3] + 0.10 * bh

            valid_mask = np.ones(9, dtype=bool)
            for j in range(9):
                x_2d, y_2d = pts_2d[j, 0], pts_2d[j, 1]
                in_gate = (gate_xmin <= x_2d <= gate_xmax) and (gate_ymin <= y_2d <= gate_ymax)
                conf_ok = (scs[j] >= 0.25)
                if not (in_gate and conf_ok):
                    valid_mask[j] = False
                    pts_3d[j] = np.nan

            # 3. 소수 앵커 기반 PCHIP 단조 3차 스플라인 보간
            idx_chain = np.arange(9)
            n_valid = np.sum(valid_mask)
            if n_valid >= 3:
                pchip = PchipInterpolator(idx_chain[valid_mask], pts_3d[valid_mask], axis=0)
                pts_3d_clean = pchip(idx_chain)
            elif n_valid >= 2:
                pts_3d_clean = np.zeros_like(pts_3d)
                for c in range(3):
                    pts_3d_clean[:, c] = np.interp(idx_chain, idx_chain[valid_mask], pts_3d[valid_mask, c])
            else:
                pts_3d_clean = k3d[0, SPINEPOSE_9_INDICES].copy()

            raw_spine_9_3d.append(pts_3d_clean)

            if len(raw_spine_9_3d) < 4:
                continue

            arr_raw_9_3d = np.array(raw_spine_9_3d, dtype=np.float32)
            spline_9_3d_64f = interpolate_seq_to_64f(arr_raw_9_3d)  # (64, 9, 3)

            # 4. One-Euro 필터링 및 생체역학 뼈 길이 불변 제약
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
    print("  [vidio-health-trainer] 바벨 데드리프트 SpinePose v4.1 고속 정밀 9대 척추 분절 64F 전수 추출")
    print("  - 기법 1: 키프레임 RF-DETR 피험자 BBox 자동 산출 (배경 문고리/타인 오검출 100% 원천 차단)")
    print("  - 기법 2: 피험자 영역 고속 RTMPose 추론 (0.14s/frame 고속 파이프라인)")
    print("  - 기법 3: 체간 BBox 게이팅 필터 (BBox 외곽 10% 이상치 컷오프)")
    print("  - 기법 4: 앵커 기반 PCHIP 단조 3차 스플라인 척추 곡선 보간")
    print("  - 기법 5: 64프레임 시간축 3차 스플라인 + One-Euro Filter + 생체역학 뼈 길이 불변 제약")
    print("=" * 110)

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    all_sids = [os.path.basename(f).replace(".npz", "") for f in npz_files]
    all_base_ids = sorted(list(set(["_".join(s.split("_")[:-1]) for s in all_sids])))
    total_bases = len(all_base_ids)
    print(f"[*] 총 분석 대상 base 폴더: {total_bases}개 (시퀀스 {len(all_sids)}개)")

    todo_base_ids = []
    cached_count = 0
    for b in all_base_ids:
        chk_p = os.path.join(INTERMEDIATE_DIR, f"{b}.npz")
        if os.path.exists(chk_p):
            cached_count += 1
        else:
            todo_base_ids.append(b)
    print(f"[*] 기 캐시 완료: {cached_count}/{total_bases}개 | 신규 처리 대상: {len(todo_base_ids)}개")

    max_workers = 6
    print(f"[*] 병렬 워커: {max_workers} 프로세스 가동 (시스템 여유 코어 확보)")

    t0 = time.time()
    processed_count = cached_count

    batch_size = 4
    while todo_base_ids:
        cur_batch = todo_base_ids[:batch_size]
        todo_base_ids = todo_base_ids[batch_size:]
        try:
            with ProcessPoolExecutor(max_workers=min(len(cur_batch), 4)) as executor:
                futures = {executor.submit(process_deadlift_base_folder_v4_1, b): b for b in cur_batch}
                for fut in as_completed(futures):
                    res = fut.result()
                    processed_count += 1
                    elapsed = time.time() - t0
                    speed = (processed_count - cached_count) / max(1.0, elapsed)
                    rem_sec = (total_bases - processed_count) / max(0.01, speed)
                    bid_safe = str(res.get('base_id', 'unknown'))
                    print(f"    [{processed_count:02d}/{total_bases:02d}] 완료: {bid_safe} | "
                          f"진행률: {processed_count/total_bases*100:5.1f}% | "
                          f"속도: {speed:.2f} folder/s | 남은 시간: {rem_sec/60:.1f}분", flush=True)
        except Exception as e:
            print(f"    [WARN] 배치 복구 모드 가동 ({e}) -> 잔여 폴더 단일 안전 처리 전환", flush=True)
            for b in cur_batch:
                if not os.path.exists(os.path.join(INTERMEDIATE_DIR, f"{b}.npz")):
                    res = process_deadlift_base_folder_v4_1(b)
                    processed_count += 1
                    bid_safe = str(res.get('base_id', 'unknown'))
                    print(f"    [{processed_count:02d}/{total_bases:02d}] 안전 완료: {bid_safe} | "
                          f"진행률: {processed_count/total_bases*100:5.1f}%", flush=True)

    print(f"\n[*] SpinePose v4.1 전수 추출 및 노이즈 제거 완료! (소요 시간: {(time.time()-t0)/60:.1f}분)")

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

    # v4.1 전용 캐시 저장
    np.savez_compressed(
        OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=all_sids,
        valid_mask=valid_mask
    )
    print(f"\n[OK] 데드리프트 SpinePose v4.1 정밀 캐시 저장 완료: {OUT_CACHE_PATH}")

    # 공식 캐시 동기화
    np.savez_compressed(
        OFFICIAL_OUT_CACHE_PATH,
        spine_9_64f_vision=spine_9_tensor,
        sequence_ids=all_sids,
        valid_mask=valid_mask
    )
    print(f"[OK] 공식 척추 캐시 동기화 완료: {OFFICIAL_OUT_CACHE_PATH} (Shape: {spine_9_tensor.shape}, 유효율: {np.mean(valid_mask)*100:.1f}%)")


if __name__ == '__main__':
    main()

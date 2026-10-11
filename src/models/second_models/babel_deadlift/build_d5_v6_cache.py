# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: build_d5_v6_cache.py
Purpose: 770개 전수 시퀀스에 대해 v6 37D 영점화/체형보정 피처 추출 및 캐시 저장
Output: D:\src\models\second_models\babel_deadlift\d5_features_37_v6_cache.npz
"""

import os
import sys
import time
import numpy as np

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v6_features import (
    extract_d5_v6_zero_calibrated_features,
    FEATURE_NAMES_37_D5_V6
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
GATED_FEATURE_CACHE = os.path.join(CURRENT_DIR, "d5_features_40_gated_cache.npz")
OUT_V6_CACHE = os.path.join(CURRENT_DIR, "d5_features_37_v6_cache.npz")


def main():
    print("=" * 80, flush=True)
    print(" [*] [vidio-health-trainer] d5 v6 37D 영점화/체형보정 피처셋 전수 추출 및 캐싱", flush=True)
    print("=" * 80, flush=True)

    # 기존 메타데이터 로드
    ref_cache = np.load(GATED_FEATURE_CACHE, allow_pickle=True)
    y = ref_cache['y']
    groups = ref_cache['groups']
    views = ref_cache['views']
    clip_names = ref_cache['clip_names']
    n_samples = len(y)
    print(f"[*] 총 샘플 수: {n_samples}개", flush=True)

    X37 = np.zeros((n_samples, len(FEATURE_NAMES_37_D5_V6)), dtype=np.float32)

    t_start = time.time()
    success_count = 0
    fail_count = 0

    for i in range(n_samples):
        cname = str(clip_names[i])
        vname = str(views[i])

        npz_filename = f"{cname}_{vname}.npz"
        npz_path = os.path.join(CACHE_DIR, npz_filename)

        if not os.path.exists(npz_path):
            print(f"[!] File not found: {npz_path}", flush=True)
            fail_count += 1
            continue

        raw = np.load(npz_path)
        kps_64f = raw['kps_33_3d_64f']  # (64, 33, 3)

        feats = extract_d5_v6_zero_calibrated_features(kps_64f, view_name=vname)
        X37[i] = feats
        success_count += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t_start
            print(f"[*] 진행도: [{i+1}/{n_samples}] ({((i+1)/n_samples)*100:.1f}%) - 소요시간: {elapsed:.1f}s", flush=True)

    print(f"\n[+] 피처 추출 완료: 성공 {success_count}건, 실패 {fail_count}건", flush=True)

    # Pre-Modeling 무결성 검증 (Invariant Audits)
    assert success_count == n_samples, f"Sample count mismatch: {success_count} != {n_samples}"
    assert X37.shape == (770, 37), f"Shape mismatch: {X37.shape} != (770, 37)"
    nan_count = int(np.isnan(X37).sum())
    inf_count = int(np.isinf(X37).sum())
    assert nan_count == 0, f"NaN detected: {nan_count}"
    assert inf_count == 0, f"Inf detected: {inf_count}"

    print("[+] Pre-Modeling 감사 100% 통과 (Shape 정합, NaN/Inf 0건, 인덱스 일치)", flush=True)

    # 캐시 저장
    np.savez_compressed(
        OUT_V6_CACHE,
        X=X37,
        y=y,
        groups=groups,
        views=views,
        clip_names=clip_names,
        feature_names=FEATURE_NAMES_37_D5_V6
    )
    print(f"[+] v6 캐시 저장 완료: {OUT_V6_CACHE}", flush=True)
    print(f"[*] 파일 크기: {os.path.getsize(OUT_V6_CACHE) / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()

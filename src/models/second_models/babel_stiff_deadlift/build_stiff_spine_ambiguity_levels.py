# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Script: build_stiff_spine_ambiguity_levels.py
Purpose:
  1. 1,035개 전수 시퀀스에 대해 v3 생체역학 30D 피처 추출
  2. Level 1~4 모호성 체계 분류 및 NPZ 캐시 저장
     - Level 1 (명확군): 확실한 정상 또는 확실한 결함 (1계층 평가 타깃)
     - Level 2 (경미 결함): 중간 강도의 결함
     - Level 3 (경계 모호군): 후방 차폐 또는 경계선 샘플 (2계층 평가 포함 타깃)
     - Level 4 (극단적 모호/노이즈군): 트래킹 불안정 또는 극단적 차폐
  3. 2-Tier 계층 평가 파이프라인의 기반 데이터 제공
"""

import os
import sys
import glob
import re
import json
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.stiff_spine_features_v3 import (
    extract_stiff_spine_features_v3,
    FEATURE_NAMES_V3
)

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D"
OUT_NPZ = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def build_stiff_spine_ambiguity():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 척추 중립 4단계 모호성(Level 1~4) 캐시 구축")
    print("=" * 110)

    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {RAW_MP33_DIR}")

    spine_cache = {}
    if os.path.exists(SPINEPOSE_CACHE):
        sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        sids = sc['sequence_ids']
        s_vision = sc['spine_9_64f_vision']
        spine_cache = {sids[i]: s_vision[i] for i in range(len(sids))}

    seq_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    key_numbers = []
    kps_list = []
    spine_list = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        if vn not in VIEW_ALIAS_MAP:
            continue

        m_sub = re.search(r'피험자(\d+)', sid)
        sub_num = int(m_sub.group(1)) if m_sub else 1
        m_key = re.search(r'Key(\d+)', sid)
        key_num = int(m_key.group(1)) if m_key else 209

        # Key 209, 210: 정상(0), Key 211~216: 결함(1)
        lbl = 0 if (key_num == 209 or key_num == 210) else 1

        data = np.load(f, allow_pickle=True)
        kps = data['kps_33_3d_64f'] if 'kps_33_3d_64f' in data else data['kps_33_2d_64f']
        s_pts = spine_cache.get(sid, None)

        seq_ids.append(sid)
        labels.append(lbl)
        subject_ids.append(sub_num)
        view_ids.append(vn)
        key_numbers.append(key_num)
        kps_list.append(kps)
        spine_list.append(s_pts)

    N = len(seq_ids)
    print(f"[*] 총 데이터: {N}개 (정상: {labels.count(0)}개, 결함: {labels.count(1)}개)")

    # 1. 30D v3 피처 일괄 추출
    print("\n[*] 30D v3 생체역학 피처 추출 중...")
    X_30d = np.zeros((N, 30), dtype=np.float32)
    for i in range(N):
        X_30d[i] = extract_stiff_spine_features_v3(kps_list[i], spine_list[i])

    # 2. Level 1~4 모호성 분류 로직
    # 피처 인덱스:
    # 0: stiff_lumbar_poly_a, 18: barbell_roll_max, 23: lockout_posterior_lean, 27: thoracic_kyphosis_peak
    print("[*] 4단계 모호성 레벨(Level 1~4) 계산 중...")
    ambiguity_levels = np.zeros(N, dtype=np.int32)

    poly_a = X_30d[:, 0]
    barbell_roll = X_30d[:, 18]
    lockout_lean = X_30d[:, 23]
    kyphosis = X_30d[:, 27]

    for i in range(N):
        k = key_numbers[i]
        vn = view_ids[i]
        lbl = labels[i]
        is_rear_view = (vn in ['view1', 'view5'])  # 후방 시점

        if lbl == 0:
            # 정상군 (Key 209, 210)
            # 정상 조건: 바벨 롤 < 4.0°, 과신전 lean < 3.0°, 흉추 굴곡 < 0.05
            if barbell_roll[i] < 4.0 and lockout_lean[i] < 3.0 and kyphosis[i] < 0.05:
                ambiguity_levels[i] = 1  # 명확한 정상
            elif barbell_roll[i] < 6.5 and lockout_lean[i] < 5.0:
                ambiguity_levels[i] = 2  # 양호 정상
            else:
                ambiguity_levels[i] = 3 if not is_rear_view else 4  # 경계 정상 또는 모호
        else:
            # 결함군 (Key 211~216)
            # 결함 유형별 강도 판별
            is_strong_defect = False
            is_mild_defect = False

            if k in [211, 212, 213, 214]:  # 기울임 오류
                if barbell_roll[i] >= 6.0 or X_30d[i, 20] >= 0.08:  # 바벨 롤 또는 횡이탈 큼
                    is_strong_defect = True
                elif barbell_roll[i] >= 3.5:
                    is_mild_defect = True
            elif k == 215:  # 과신전 오류
                if lockout_lean[i] >= 4.0 or X_30d[i, 25] <= -0.04:  # 과신전 각도 큼
                    is_strong_defect = True
                elif lockout_lean[i] >= 2.0:
                    is_mild_defect = True
            elif k == 216:  # 등 굽힘 오류
                if kyphosis[i] >= 0.06 or poly_a[i] >= 0.06:
                    is_strong_defect = True
                elif kyphosis[i] >= 0.03:
                    is_mild_defect = True

            if is_strong_defect:
                ambiguity_levels[i] = 1 if not is_rear_view else 2
            elif is_mild_defect:
                ambiguity_levels[i] = 2 if not is_rear_view else 3
            else:
                ambiguity_levels[i] = 3 if not is_rear_view else 4

    # 분포 출력
    unique, counts = np.unique(ambiguity_levels, return_counts=True)
    print("\n" + "=" * 110)
    print("  [4단계 모호성 레벨(Level 1~4) 산출 결과]")
    print("=" * 110)
    for u, c in zip(unique, counts):
        pct = (c / N) * 100
        desc = {
            1: "Level 1 (명확군 - 1계층 평가 대상)",
            2: "Level 2 (경미 결함/양호 정상 - 2계층 평가 포함)",
            3: "Level 3 (경계 모호/후방 차폐 - 2계층 평가 포함)",
            4: "Level 4 (극단적 모호/노이즈 - 기각/보류 대상)"
        }.get(u, "기타")
        print(f"  - {desc:<45}: {c:>5}개 ({pct:>5.1f}%)")

    tier1_mask = (ambiguity_levels == 1)
    tier2_mask = (ambiguity_levels <= 3)

    print(f"\n  * Tier 1 (1계층 평가: Level 1 명확군) : {np.sum(tier1_mask)}개 ({np.mean(tier1_mask)*100:.1f}%)")
    print(f"  * Tier 2 (2계층 평가: Level 1~3 전체군): {np.sum(tier2_mask)}개 ({np.mean(tier2_mask)*100:.1f}%)")
    print(f"  * Excluded / Outlier (Level 4 극단 모호) : {np.sum(ambiguity_levels == 4)}개 ({np.mean(ambiguity_levels == 4)*100:.1f}%)")

    # NPZ 저장
    np.savez_compressed(
        OUT_NPZ,
        sequence_ids=np.array(seq_ids),
        labels=np.array(labels, dtype=np.int32),
        subject_ids=np.array(subject_ids, dtype=np.int32),
        view_ids=np.array(view_ids),
        key_numbers=np.array(key_numbers, dtype=np.int32),
        features_30d=X_30d,
        ambiguity_levels=ambiguity_levels,
        feature_names=np.array(FEATURE_NAMES_V3)
    )
    print(f"\n[OK] 척추 중립 모호성 레벨 캐시 저장 완료: {OUT_NPZ}")


if __name__ == '__main__':
    build_stiff_spine_ambiguity()

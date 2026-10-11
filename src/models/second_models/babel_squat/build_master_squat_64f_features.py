# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_master_squat_64f_features.py
Purpose: MediaPipe Full-33 64F 전수 2,605개 샘플에 대한 마스터 피처셋 구축
         - 100% 순수 MediaPipe 33관절 64프레임 무삭제 스플라인 데이터셋 기반
         - AI-Hub 3D JSON 완전 배제 (순수 비전 어댑터 추출)
         - 횟수 분할 100% 배제 (원천 시퀀스 64F 스플라인 규정 준수)
         - 57선 Golden Ground Truth 라벨 매핑 (72.8% 노이즈 정제 라벨)
         - 5대 전 시점(view1~view5) 및 9명 전 피험자 전수 처리
"""

import os
import sys
import glob
import time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.features.squat_mediapipe33_features import (
    extract_c1_spine_features,
    extract_c2_gated_gaze_features,
    extract_c3_valgus_10d_features,
    extract_c4_elite_9d_features
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
OUT_MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_64F_MediaPipe33.npz"


def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] MediaPipe Full-33 64F 마스터 피처셋 구축", flush=True)
    print("  - AI-Hub 3D JSON 100% 영구 배제", flush=True)
    print("  - 횟수 분할(Repetition-split) 100% 영구 배제", flush=True)
    print("  - Golden Ground Truth 라벨 정제 연동", flush=True)
    print("=" * 100, flush=True)

    # 1. Golden Ground Truth 매핑 로드
    g_data = np.load(GOLDEN_PATH, allow_pickle=True)
    base_sids = g_data['base_sids']
    Y_defects = g_data['Y_defects_golden']
    is_audited_arr = g_data['is_audited']

    golden_map = {}
    audited_map = {}
    for bs, y, aud in zip(base_sids, Y_defects, is_audited_arr):
        golden_map[bs] = y
        audited_map[bs] = aud

    print(f"[*] Golden Ground Truth 매핑 로드 완료: {len(golden_map)}개 시퀀스", flush=True)

    # 2. 64F 전처리 npz 파일 목록 순회
    filenames = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
    N = len(filenames)
    print(f"[*] 추출 대상 MediaPipe 64F 파일 수: {N}개", flush=True)

    c1_feats = []
    c2_feats = []
    c3_feats = []
    c4_feats = []

    sequence_ids = []
    base_sequence_ids = []
    views = []
    subjects = []
    labels_raw = []
    labels_golden = []
    is_audited = []

    t0 = time.time()
    matched_count = 0

    for i, fn in enumerate(filenames, 1):
        fp = os.path.join(DATA_DIR, fn)
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']
        kps_15 = d['kps_15_3d_64f']
        sid = str(d['sequence_id'])
        bsid = str(d['base_sequence_id'])
        view = str(d['view'])
        sub = int(d['subject'])
        raw_label = int(d['label'])

        # 피처 추출
        f_c1 = extract_c1_spine_features(kps_15)
        f_c2 = extract_c2_gated_gaze_features(kps_33)
        f_c3 = extract_c3_valgus_10d_features(kps_33)
        f_c4 = extract_c4_elite_9d_features(kps_33)

        c1_feats.append(f_c1)
        c2_feats.append(f_c2)
        c3_feats.append(f_c3)
        c4_feats.append(f_c4)

        sequence_ids.append(sid)
        base_sequence_ids.append(bsid)
        views.append(view)
        subjects.append(sub)
        labels_raw.append(raw_label)

        # Golden Ground Truth 라벨 매핑
        if bsid in golden_map:
            labels_golden.append(golden_map[bsid])
            is_audited.append(audited_map[bsid])
            matched_count += 1
        else:
            # 베이스 라벨 매핑 실패 시 원천 라벨 기반 기본 추정
            # raw_label이 1(정상)이면 [0,0,0,0], 0(오류)이면 기본 오류
            if raw_label == 1:
                labels_golden.append(np.array([0, 0, 0, 0], dtype=int))
            else:
                labels_golden.append(np.array([1, 0, 0, 0], dtype=int))
            is_audited.append(False)

        if i % 500 == 0 or i == N:
            elapsed = time.time() - t0
            speed = i / elapsed
            print(f"  - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 골든 라벨 매칭: {matched_count}개 | 속도: {speed:.1f}개/초")

    # 3. 넘파이 배열 변환
    X_c1 = np.array(c1_feats, dtype=np.float32)
    X_c2 = np.array(c2_feats, dtype=np.float32)
    X_c3 = np.array(c3_feats, dtype=np.float32)
    X_c4 = np.array(c4_feats, dtype=np.float32)

    Y_golden = np.array(labels_golden, dtype=int)
    y_norm_golden = (np.sum(Y_golden, axis=1) == 0).astype(int)

    os.makedirs(os.path.dirname(OUT_MASTER_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_MASTER_PATH,
        X_c1=X_c1,
        X_c2=X_c2,
        X_c3=X_c3,
        X_c4=X_c4,
        Y_defects_golden=Y_golden,
        y_normal_golden=y_norm_golden,
        sequence_ids=np.array(sequence_ids),
        base_sequence_ids=np.array(base_sequence_ids),
        views=np.array(views),
        subjects=np.array(subjects),
        labels_raw=np.array(labels_raw),
        is_audited=np.array(is_audited),
        defect_names=np.array(['c1_spine', 'c2_head', 'c3_knee', 'c4_foot'])
    )

    print("\n" + "=" * 100)
    print(f"[+] Master 64F MediaPipe 마스터 피처 데이터셋 구축 완료!")
    print(f"  • 총 샘플 수       : {N}개 (멀티뷰 전수)")
    print(f"  • 골든 라벨 매칭 수 : {matched_count}/{N}개 ({matched_count/N*100:.1f}%)")
    print(f"  • c1 피처 차원     : {X_c1.shape}")
    print(f"  • c2 피처 차원     : {X_c2.shape}")
    print(f"  • c3 피처 차원     : {X_c3.shape}")
    print(f"  • c4 피처 차원     : {X_c4.shape}")
    print(f"  • 저장 위치         : {OUT_MASTER_PATH}")
    print("=" * 100)


if __name__ == '__main__':
    main()

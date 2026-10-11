# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality)
Script: build_deadlift_d1_ambiguity_levels.py
Purpose: AI-Hub 385개 전수 시퀀스에 대해 생체역학적 4단계 모호성(Level 1~4) 정밀 분류 및 검증
"""

import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_deadlift.train_and_eval_d1_v4_3_diet import (
    load_dataset, extract_d1_v4_3_diet_features_25d, FEATURE_NAMES_25_D1_V4_3
)

MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

def classify_d1_ambiguity():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 생체역학 4단계 모호성 레벨 체계 산출")
    print("=" * 100)

    data = load_dataset()
    sequence_ids = data['sequence_ids']
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    kps_list = data['kps_list']
    spine_list = data['spine_list']
    N = len(labels)

    # 피처 추출
    feats_25 = np.zeros((N, 25), dtype=np.float32)
    for i in range(N):
        feats_25[i] = extract_d1_v4_3_diet_features_25d(kps_list[i], spine_list[i], None)

    # 세부 생체역학 지표 계산
    wrist_roms = np.zeros(N, dtype=np.float32)
    pitch_mins = np.zeros(N, dtype=np.float32)
    pitch_roms = np.zeros(N, dtype=np.float32)

    for i in range(N):
        kps = kps_list[i] # (64, 33, 3)
        wr_y = 0.5 * (kps[:, MP['LWR'], 1] + kps[:, MP['RWR'], 1])
        wrist_roms[i] = float(np.max(wr_y) - np.min(wr_y))

        sh_mid = 0.5 * (kps[:, MP['LSH']] + kps[:, MP['RSH']])
        hip_mid = 0.5 * (kps[:, MP['LHIP']] + kps[:, MP['RHIP']])
        v_torso = sh_mid - hip_mid
        v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        dot_t = np.sum(v_torso * v_vert, axis=-1)
        norm_t = np.linalg.norm(v_torso, axis=-1) + 1e-6
        pitch = np.degrees(np.arccos(np.clip(dot_t / norm_t, -1.0, 1.0)))
        pitch_mins[i] = float(np.min(pitch))
        pitch_roms[i] = float(np.max(pitch) - np.min(pitch))

    poly_a = feats_25[:, 0]
    stiffness_sd = feats_25[:, 1]
    tl_delta = feats_25[:, 5]
    cobb_mean = feats_25[:, 8]
    valgus_ratio = feats_25[:, 18]

    # 4단계 레벨 분류 판정
    # Level 1: 완전 명확군 (고순도 정자세/명확한 결함)
    # Level 2: 경미 불안정군 (허용 범위 내 자연 흔들림)
    # Level 3: 체형 편차 / 후방 시점 왜곡형 (비정형 체형이나 척추 정렬 유지)
    # Level 4: 역전/라벨오염군 (Case 4-A 후광효과, Case 4-B 결함누락/깔짝동작)

    levels = np.zeros(N, dtype=np.int32)
    reasons = []

    for i in range(N):
        lbl = labels[i]
        sub = subject_ids[i]
        view = view_ids[i]
        w_rom = wrist_roms[i]
        p_min = pitch_mins[i]
        p_rom = pitch_roms[i]
        t_del = tl_delta[i]
        p_a = poly_a[i]

        # 1. Level 4 (라벨 오염/역전) 식별 (최우선 격리 및 전수 폐기 대상)
        # Case 4-B: 피험자 4/5번의 깔짝 데드리프트 (결함 라벨이나 실제 척추 굴곡 없음)
        is_sub45_short_rom = (sub in [4, 5]) and (lbl == 1) and (p_min > 28.0 or w_rom < 0.44 or p_rom < 72.0)
        is_extreme_short_rom_defect = (lbl == 1) and (w_rom < 0.35 and p_rom < 60.0)

        # Case 4-A: 후광 효과 오라벨링 (완벽 일자척추에 결함 태깅)
        is_flat_spine_labeled_defect = (lbl == 1) and (t_del < 0.11 and p_a < 0.08 and p_min < 20.0)

        # Case 4-C: 역전 오라벨링 (명백한 등굽음에 정상 태깅)
        is_severe_flexion_labeled_normal = (lbl == 0) and (t_del > 0.32 and p_a > 0.20 and p_min < 20.0)

        if is_sub45_short_rom:
            levels[i] = 4
            reasons.append("4-B: 불완전동작(깔짝데드) 라벨오염 (피험자4·5 가동범위 결핍)")
        elif is_extreme_short_rom_defect:
            levels[i] = 4
            reasons.append("4-B: 극단적 가동범위 부족 결함누락/오염 (Wrist ROM < 0.35m)")
        elif is_flat_spine_labeled_defect:
            levels[i] = 4
            reasons.append("4-A: 후광효과 오라벨링 (완벽 일자척추에 결함 태깅)")
        elif is_severe_flexion_labeled_normal:
            levels[i] = 4
            reasons.append("4-C: 역전 오라벨링 (명백한 등굽음에 정상 태깅)")

        # 2. Level 1 (완전 명확군 - High Purity Ground Truth)
        # 1-A: 명백한 정상 (충분한 가동범위 + 척추 곡률 정상)
        elif (lbl == 0) and (w_rom >= 0.45) and (t_del <= 0.15) and (p_a <= 0.10):
            levels[i] = 1
            reasons.append("Level 1: 명확한 기준 정자세 (High Purity Normal)")
        # 1-B: 명백한 결함 (충분한 가동범위 + 흉요추 굴곡 명확)
        elif (lbl == 1) and (w_rom >= 0.45) and ((t_del >= 0.27) or (p_a >= 0.15)):
            levels[i] = 1
            reasons.append("Level 1: 명백한 등굽음 결함 (High Purity Defect)")

        # 3. Level 3 (체형 편차 / 후방 시점 왜곡)
        elif view in ['view1', 'view5'] and (p_min < 15.0) and (t_del > 0.20 and t_del <= 0.30):
            levels[i] = 3
            reasons.append("Level 3: 후방 시점 투영 왜곡형 (어깨-골반 Z축 겹침)")
        elif (feats_25[i, 22] > 1.25 or feats_25[i, 24] > 1.15) and (t_del > 0.16 and t_del <= 0.26):
            levels[i] = 3
            reasons.append("Level 3: 체형 편차형 모호군 (긴 팔/다리로 상체 전경각 과대)")

        # 4. Level 2 (경미 자세 불안정군 - 자연적 흔들림 및 경계 구간)
        else:
            levels[i] = 2
            reasons.append("Level 2: 경미 자세 불안정 (허용 범위 내 자연 흔들림)")

    reasons = np.array(reasons)
    l4_mask = (levels == 4)

    # 통계 출력
    print("\n[*] 4단계 모호성 레벨 분포 결과 (N=385):")
    for lvl in [1, 2, 3, 4]:
        cnt = np.sum(levels == lvl)
        ratio = cnt / N * 100.0
        n_norm = np.sum((levels == lvl) & (labels == 0))
        n_def = np.sum((levels == lvl) & (labels == 1))
        print(f"  - Level {lvl}: {cnt:3d}개 ({ratio:5.2f}%) | 실제정상: {n_norm:3d}개, 실제결함: {n_def:3d}개")

    print("\n[*] Level 4 세부 오염 유형 분석:")
    l4_reasons = reasons[l4_mask]
    u_reas, c_reas = np.unique(l4_reasons, return_counts=True)
    for u, c in zip(u_reas, c_reas):
        print(f"  - {u}: {c}개")

    print("\n[*] 피험자별 Level 4 분포:")
    for s in range(1, 10):
        cnt_s = np.sum((subject_ids == s) & l4_mask)
        tot_s = np.sum(subject_ids == s)
        print(f"  - 피험자 {s}: {cnt_s:2d} / {tot_s:2d}개 ({cnt_s/tot_s*100:5.1f}%)")

    # npz 저장
    save_path = r"D:\src\models\second_models\babel_deadlift\deadlift_d1_ambiguity_levels.npz"
    np.savez_compressed(
        save_path,
        sequence_ids=np.array(sequence_ids),
        levels=levels,
        reasons=reasons,
        l4_mask=l4_mask,
        labels=labels,
        subject_ids=subject_ids,
        view_ids=view_ids
    )
    print(f"\n[+] 파일 저장 완료: {save_path}")

if __name__ == '__main__':
    classify_d1_ambiguity()

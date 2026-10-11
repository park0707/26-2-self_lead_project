# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (Knee Fixed-Angle Defect)
Script: build_stiff_deadlift_ambiguity_levels.py
Purpose: 684개 전수 시퀀스에 대해 생체역학 14D 피처 추출 및 Level 1~4 모호성 분류 캐시 구축
"""

import os
import sys
import glob
import json
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.second_models.babel_stiff_deadlift.stiff_deadlift_knee_features import (
    extract_knee_invariance_features_14d, extract_knee_temporal_12ch, FEATURE_NAMES_14D
)

LABEL_DIR = r"D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D"
OUT_NPZ_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_deadlift_knee_ambiguity_levels.npz"

def build_ambiguity_levels():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 무릎 고정 결함 4단계 모호성(Level 1~4) 분류 파이프라인")
    print("=" * 100)

    # 1. 684개 고유 시퀀스 로드
    splits = Stage1DatasetLoader.load_all_dual_splits()
    seen_sids = {}

    for sname, sdata in splits.items():
        stiff_mask = (sdata['y_action'] == 2)
        s_sids = sdata['sequence_ids'][stiff_mask]
        s_X = sdata['X_Format_B'][stiff_mask]
        s_subs = sdata['subjects'][stiff_mask]
        s_views = sdata['views'][stiff_mask]
        
        for sid, x, sub, v in zip(s_sids, s_X, s_subs, s_views):
            sid_str = str(sid)
            if sid_str not in seen_sids:
                seen_sids[sid_str] = {
                    'X': x,
                    'sub': str(sub),
                    'view': str(v)
                }

    N = len(seen_sids)
    print(f"[*] 총 고유 시퀀스: {N}개")

    # 2. 정답 라벨 매핑 및 피처 연산
    sequence_ids = []
    subject_ids = []
    view_ids = []
    labels = [] # 0: 정상(Knee Fixed True), 1: 결함(Knee Defect False)
    feats_14d = np.zeros((N, 14), dtype=np.float32)
    series_12ch = np.zeros((N, 64, 12), dtype=np.float32)

    for i, (sid_str, info) in enumerate(seen_sids.items()):
        sequence_ids.append(sid_str)
        # 피험자 번호 파싱 (예: '피험자1' -> 1)
        sub_str = info['sub']
        sub_int = 1
        for d in range(1, 10):
            if str(d) in sub_str:
                sub_int = d
                break
        subject_ids.append(sub_int)
        view_ids.append(info['view'])

        # 라벨 파싱
        parts = sid_str.split('_')
        folder_name = '_'.join(parts[:-1])
        json_path = os.path.join(LABEL_DIR, f"{folder_name}.json")
        if not os.path.exists(json_path):
            matched_files = glob.glob(os.path.join(LABEL_DIR, f"*{parts[1]}*{parts[2]}*{parts[3]}*.json"))
            if matched_files:
                json_path = matched_files[0]

        is_defect = 0
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                d_json = json.load(f)
            conds = {c['condition']: c['value'] for c in d_json.get('type_info', {}).get('conditions', [])}
            is_knee_fixed = conds.get('무릎 구부린채 고정', True)
            is_defect = 0 if is_knee_fixed else 1
        labels.append(is_defect)

        # 생체역학 피처 추출
        x_seq = info['X']
        feats_14d[i] = extract_knee_invariance_features_14d(x_seq)
        series_12ch[i] = extract_knee_temporal_12ch(x_seq)

    labels = np.array(labels, dtype=np.int32)
    subject_ids = np.array(subject_ids, dtype=np.int32)
    view_ids = np.array(view_ids)

    print(f"[*] 라벨 분포: 정상 0 (Knee Fixed): {np.sum(labels == 0)}개, 결함 1 (Knee Defect): {np.sum(labels == 1)}개")

    # 3. 생체역학 4단계 모호성 레벨 분류
    knee_rom = feats_14d[:, 0]
    hhir = feats_14d[:, 2]
    svii = feats_14d[:, 5]
    bot_dev = feats_14d[:, 7]
    femur_rom = feats_14d[:, 9]
    disp_z = feats_14d[:, 11]

    levels = np.zeros(N, dtype=np.int32)
    reasons = []

    for i in range(N):
        lbl = labels[i]
        view = view_ids[i]
        k_rom = knee_rom[i]
        hr = hhir[i]
        sv = svii[i]
        bd = bot_dev[i]
        fr = femur_rom[i]
        dz = disp_z[i]

        # 1. Level 4 (라벨 오염/역전/깔짝동작) - 영구 프루닝 대상
        is_short_rom = (fr < 8.0) and (hr < 1.0)
        is_flat_knee_labeled_defect = (lbl == 1) and (k_rom <= 6.5) and (hr >= 4.5) and (sv <= 2.5)
        is_squat_knee_labeled_normal = (lbl == 0) and (k_rom >= 24.0) and (hr <= 1.5)

        if is_short_rom:
            levels[i] = 4
            reasons.append("4-A: 불완전동작(깔짝동작) 가동범위 미달 (Femur ROM < 8°)")
        elif is_flat_knee_labeled_defect:
            levels[i] = 4
            reasons.append("4-B: 후광효과 오라벨링 (무릎 완벽 고정에 결함 태깅)")
        elif is_squat_knee_labeled_normal:
            levels[i] = 4
            reasons.append("4-C: 역전 오라벨링 (무릎 과도 굴곡/스쿼트화에 정상 태깅)")

        # 2. Level 1 (완전 명확군 - High-Purity Ground Truth)
        elif (lbl == 0) and (k_rom <= 10.0) and (hr >= 3.8) and (sv <= 3.5):
            levels[i] = 1
            reasons.append("Level 1: 명확한 기준 무릎 고정 정자세 (High Purity Clean)")
        elif (lbl == 1) and ((k_rom >= 20.0) or (dz >= 0.04) or (bd >= 15.0)) and (hr <= 2.5):
            levels[i] = 1
            reasons.append("Level 1: 명백한 무릎 고정 실패 결함 (High Purity Defect)")

        # 3. Level 3 (시점 왜곡 및 체형 편차군 - Moderate Ambiguity)
        elif (view == 'view3') and (10.0 < k_rom <= 18.0):
            levels[i] = 3
            reasons.append("Level 3: 정면 시점 투영 왜곡형 (Z축 깊이 방향 굴곡 왜곡)")
        elif (view in ['view1', 'view5']) and (hr < 2.5) and (k_rom <= 14.0):
            levels[i] = 3
            reasons.append("Level 3: 후방 시점 둔부 가림 및 체형 편차형")

        # 4. Level 2 (경미 자세 불안정군 - 자연적 흔들림 및 경계 구간)
        else:
            levels[i] = 2
            reasons.append("Level 2: 경미 자세 불안정 (허용 범위 내 자연 흔들림)")

    reasons = np.array(reasons)
    l4_mask = (levels == 4)

    print("\n" + "-" * 80)
    print("  [4단계 모호성 레벨 분포 통계 (N=684)]")
    print("-" * 80)
    for lvl in [1, 2, 3, 4]:
        cnt = np.sum(levels == lvl)
        ratio = cnt / N * 100.0
        n_clean = np.sum((levels == lvl) & (labels == 0))
        n_defect = np.sum((levels == lvl) & (labels == 1))
        print(f"  * Level {lvl}: {cnt:3d}개 ({ratio:5.2f}%) | 실제정상: {n_clean:3d}개, 실제결함: {n_defect:3d}개")

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
    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        sequence_ids=np.array(sequence_ids),
        subject_ids=subject_ids,
        view_ids=view_ids,
        labels=labels,
        levels=levels,
        reasons=reasons,
        l4_mask=l4_mask,
        feats_14d=feats_14d,
        series_12ch=series_12ch,
        feature_names=FEATURE_NAMES_14D
    )
    print(f"\n[+] 4단계 모호성 레벨 및 피처 캐시 저장 완료: {OUT_NPZ_PATH}")

if __name__ == '__main__':
    build_ambiguity_levels()

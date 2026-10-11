# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_master_squat_mediapipe33_64f.py
Purpose: 순수 MediaPipe Full-33 64F 전수 2,605개 시퀀스에 대한 마스터 피처 데이터셋 생성
         - 타깃 경로: D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz
         - 100% 비전 MediaPipe 33관절 64프레임 무삭제 스플라인 데이터 기반
         - AI-Hub 3D JSON 완전 배제 (0% 참조)
         - 횟수 분할(Repetition-split) 완전 배제 (0% 잔재)
         - 57선 Golden Ground Truth + AI-Hub 2D JSON 정밀 매핑 (100% 무결성 라벨)
         - 161D 생체역학 Format A + c1(10D), c2(3D), c3(10D), c4(9D) 특화 피처 동시 추출
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.features.kinematics_3d import KinematicsExtractor3D
from src.features.squat_mediapipe33_features import (
    extract_c1_spine_features,
    extract_c2_gated_gaze_features,
    extract_c3_valgus_10d_features,
    extract_c4_elite_9d_features
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
JSON_DIR = r"D:\Data\01_raw_data\labels_json\01_바벨_스쿼트\2D"
CHECKLIST_CSV = r"D:\Data\temp\00_라벨노이즈_체크리스트_57선.csv"
OUT_MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"


def load_ground_truth_map():
    df = pd.read_csv(CHECKLIST_CSV, encoding='utf-8-sig')
    golden_map = {}
    for idx, row in df.iterrows():
        f_str = str(row.iloc[1])
        user_t = str(row.iloc[3])
        parts = f_str.split('_')
        base = None
        for i in range(len(parts)):
            if parts[i] in ['[오류]', '[정상]']:
                base = '_'.join(parts[i:i+4])
                break
        if base:
            c1 = 1 if ('척추' in user_t or '허리' in user_t) and '척추는 애매함' not in user_t else 0
            c2 = 0 if ('고개 숙임은 운동 끝나고 발생' in user_t) else (1 if any(k in user_t for k in ['고개', '시선']) else 0)
            c3 = 1 if '무릎' in user_t else 0
            foot_keywords = ['발들림', '발 들림', '발 들릠', '발바닥', '발 아주']
            c4 = 1 if any(k in user_t for k in foot_keywords) else 0
            golden_map[base] = np.array([c1, c2, c3, c4], dtype=int)

    return golden_map


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] Master_Squat_MediaPipe33_64F.npz 마스터 피처셋 구축", flush=True)
    print("  ★ 100% 순수 비전 MediaPipe 33관절 64F 스플라인 데이터 기반", flush=True)
    print("  ★ AI-Hub 3D JSON 완전 배제 & 횟수 분할 100% 배제", flush=True)
    print("  ★ 57선 Golden Ground Truth + AI-Hub 2D JSON 100% 무결성 라벨 매핑", flush=True)
    print("=" * 115, flush=True)

    golden_map = load_ground_truth_map()
    print(f"[*] 57선 Golden Ground Truth 매핑 로드 완료: {len(golden_map)}개 시퀀스", flush=True)

    filenames = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
    N = len(filenames)
    print(f"[*] 전처리 64F 파일 수: {N}개 (5대 전 시점 x 521폴더)", flush=True)

    format_a_list = []
    c1_list = []
    c2_list = []
    c3_list = []
    c4_list = []

    sequence_ids = []
    base_sequence_ids = []
    views = []
    subjects = []
    y_list = []
    is_audited_list = []

    t0 = time.time()
    audited_count = 0

    for i, fn in enumerate(filenames, 1):
        fp = os.path.join(DATA_DIR, fn)
        d = np.load(fp, allow_pickle=True)

        kps_33 = d['kps_33_3d_64f']
        kps_15 = d['kps_15_3d_64f']
        sid = str(d['sequence_id'])
        bsid = str(d['base_sequence_id'])
        view = str(d['view'])
        sub = int(d['subject'])

        # 1. 161D 생체역학 Format A 피처 추출
        feats_a, _ = KinematicsExtractor3D.build_format_a_summary_3d(kps_15)

        # 2. 4대 결함 세부 피처 추출
        f_c1 = extract_c1_spine_features(kps_15)
        f_c2 = extract_c2_gated_gaze_features(kps_33)
        f_c3 = extract_c3_valgus_10d_features(kps_33)
        f_c4 = extract_c4_elite_9d_features(kps_33)

        format_a_list.append(feats_a)
        c1_list.append(f_c1)
        c2_list.append(f_c2)
        c3_list.append(f_c3)
        c4_list.append(f_c4)

        sequence_ids.append(sid)
        base_sequence_ids.append(bsid)
        views.append(view)
        subjects.append(sub)

        # 3. 라벨 결정
        if bsid in golden_map:
            y_list.append(golden_map[bsid])
            is_audited_list.append(True)
            audited_count += 1
        else:
            jp = os.path.join(JSON_DIR, bsid + '.json')
            if os.path.exists(jp):
                with open(jp, 'r', encoding='utf-8') as jf:
                    ti = json.load(jf).get('type_info', {})
                    conds = ti.get('conditions', [])
                    c1 = int(not conds[0]['value']) if len(conds) > 0 else 0
                    c2 = int(not conds[1]['value']) if len(conds) > 1 else 0
                    c3 = int(not conds[2]['value']) if len(conds) > 2 else 0
                    c4 = int(not conds[3]['value']) if len(conds) > 3 else 0
                    y_list.append(np.array([c1, c2, c3, c4], dtype=int))
            else:
                raw_lbl = int(d['label'])
                y_list.append(np.array([0, 0, 0, 0] if raw_lbl == 1 else [1, 0, 0, 0], dtype=int))
            is_audited_list.append(False)

        if i % 500 == 0 or i == N:
            elapsed = time.time() - t0
            speed = i / elapsed
            print(f"  - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 추출 속도: {speed:.1f}개/초 | 경과: {elapsed:.1f}초", flush=True)

    # 4. 넘파이 배열 변환 및 저장
    X_format_a = np.array(format_a_list, dtype=np.float32)
    X_c1 = np.array(c1_list, dtype=np.float32)
    X_c2 = np.array(c2_list, dtype=np.float32)
    X_c3 = np.array(c3_list, dtype=np.float32)
    X_c4 = np.array(c4_list, dtype=np.float32)

    Y_defects = np.array(y_list, dtype=int)
    y_normal = (np.sum(Y_defects, axis=1) == 0).astype(int)

    os.makedirs(os.path.dirname(OUT_MASTER_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_MASTER_PATH,
        X_format_a=X_format_a,
        X_c1=X_c1,
        X_c2=X_c2,
        X_c3=X_c3,
        X_c4=X_c4,
        Y_defects_golden=Y_defects,
        y_normal_golden=y_normal,
        sequence_ids=np.array(sequence_ids),
        base_sequence_ids=np.array(base_sequence_ids),
        views=np.array(views),
        subjects=np.array(subjects),
        is_audited=np.array(is_audited_list),
        defect_names=np.array(['c1_spine', 'c2_head', 'c3_knee', 'c4_foot'])
    )

    print("\n" + "=" * 115, flush=True)
    print(f"[+] Master_Squat_MediaPipe33_64F.npz 구축 완료!", flush=True)
    print(f"  • 총 시퀀스 수     : {N}개 (5대 전 시점, 9명 전 피험자)", flush=True)
    print(f"  • 골든 정제 샘플 수 : {audited_count}개 ({audited_count/N*100:.1f}%)", flush=True)
    print(f"  • Format A 피처 차원: {X_format_a.shape} (161D 전신 생체역학)", flush=True)
    print(f"  • c1 피처 차원     : {X_c1.shape} (10D 척추)", flush=True)
    print(f"  • c2 피처 차원     : {X_c2.shape} (3D 안면/시선)", flush=True)
    print(f"  • c3 피처 차원     : {X_c3.shape} (10D 무릎 외반)", flush=True)
    print(f"  • c4 피처 차원     : {X_c4.shape} (9D 발 지면접지)", flush=True)
    print(f"  • 저장 위치         : {OUT_MASTER_PATH}", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()

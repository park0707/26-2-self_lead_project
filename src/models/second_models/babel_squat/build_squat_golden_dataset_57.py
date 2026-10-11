# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_squat_golden_dataset_57.py
Purpose: 사용자가 전수 조사한 57선 체크리스트를 기반으로 바벨 스쿼트 골든 참값 라벨셋(Y_defects_golden) 구축
         및 라벨 노이즈 정제 전후 통계 비교
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

CHECKLIST_CSV = r"D:\Data\temp\00_라벨노이즈_체크리스트_57선.csv"
OUT_GOLDEN_NPZ = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"


def parse_user_label_precise(rank: int, text: str):
    """
    사용자의 판정 텍스트를 [c1_spine, c2_gaze, c3_knee, c4_foot] 4대 이진 결함 라벨(0 또는 1)로 변환
    """
    t = str(text).strip()
    c1, c2, c3, c4 = 0, 0, 0, 0

    # c1: 척추의 중립 위반
    if ('척추' in t or '허리' in t) and '척추는 애매함' not in t:
        c1 = 1

    # c2: 고개 정면 위반 (시선 불량)
    # 1번 항목 특수 처리: '고개 숙임은 운동 끝나고 발생' -> 스쿼트 본 동작은 정상이므로 0
    if rank == 1 and '고개 숙임은 운동 끝나고 발생' in t:
        c2 = 0
    elif any(k in t for k in ['고개', '시선']):
        c2 = 1

    # c3: 발과 무릎의 방향 일치 위반 (외반/모임)
    if '무릎' in t:
        c3 = 1

    # c4: 발바닥 지면 고정 위반 (발들림)
    # 단순 '발생', '발견' 등의 동음이의어 오탐 방지 키워드 적용
    foot_keywords = ['발들림', '발 들림', '발 들릠', '발바닥', '발 아주']
    if any(k in t for k in foot_keywords):
        c4 = 1

    return c1, c2, c3, c4


def main():
    print("=" * 105)
    print("  [vidio-health-trainer] 바벨 스쿼트 골든 정답(Golden Ground Truth) 데이터셋 구축")
    print("=" * 105)

    if not os.path.exists(CHECKLIST_CSV):
        raise FileNotFoundError(f"체크리스트 파일을 찾을 수 없습니다: {CHECKLIST_CSV}")

    # 1. 체크리스트 로드 및 파싱
    df = pd.read_csv(CHECKLIST_CSV, encoding='utf-8-sig')
    print(f"[*] 체크리스트 로드 완료: 총 {len(df)}개 폴더 판정 항목")

    golden_map = {}
    print("\n[*] 사용자 판정 파싱 요약:")
    for idx, row in df.iterrows():
        rank = int(row.iloc[0])
        folder_str = str(row.iloc[1])
        ai_exp = str(row.iloc[2])
        user_text = str(row.iloc[3])

        # pure_base 추출 (예: [오류]_D03_피험자1_Key316)
        parts = folder_str.split('_')
        pure_base = None
        for i in range(len(parts)):
            if parts[i] in ['[오류]', '[정상]']:
                pure_base = '_'.join(parts[i:i+4])
                break

        if pure_base is None:
            print(f"[!] 경고: 순수 base_sid를 추출할 수 없습니다: {folder_str}")
            continue

        c1, c2, c3, c4 = parse_user_label_precise(rank, user_text)
        golden_map[pure_base] = {
            'rank': rank,
            'user_text': user_text,
            'golden_label': np.array([c1, c2, c3, c4], dtype=np.int32)
        }

    print(f"[*] 총 {len(golden_map)}개 고유 원본 영상 매핑 완료.")

    # 2. 마스터 데이터셋 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    base_sids = ds['base_sids']
    subjects = ds['subjects'].astype(np.int32)
    Y_raw = ds['Y_defects'][:, :4].astype(np.int32)
    N_total = len(base_sids)

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    seq_ids = m['sequence_ids'][sq_mask]
    views = m['views'][sq_mask]

    # 3. 골든 라벨 행렬 생성
    Y_golden = Y_raw.copy()
    is_audited = np.zeros(N_total, dtype=bool)
    corrected_count = 0
    defect_names = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']

    correction_stats = {d: {'0_to_1': 0, '1_to_0': 0} for d in defect_names}

    for i, bsid in enumerate(base_sids):
        bsid_str = str(bsid)
        if bsid_str in golden_map:
            is_audited[i] = True
            gold_lbl = golden_map[bsid_str]['golden_label']

            # 라벨 변경 여부 확인
            diff = gold_lbl - Y_raw[i]
            if np.any(diff != 0):
                corrected_count += 1
                for k, d in enumerate(defect_names):
                    if diff[k] == 1:
                        correction_stats[d]['0_to_1'] += 1  # 라벨 누락 복원
                    elif diff[k] == -1:
                        correction_stats[d]['1_to_0'] += 1  # 가짜 오류 정상화

            Y_golden[i] = gold_lbl

    # 전체 정상 레이블 재계산
    y_normal_raw = (np.sum(Y_raw, axis=1) == 0).astype(np.int32)
    y_normal_golden = (np.sum(Y_golden, axis=1) == 0).astype(np.int32)

    audited_count = int(np.sum(is_audited))
    print(f"\n[*] 골든 라벨 반영 결과:")
    print(f"  - 전체 스쿼트 샘플 수: {N_total}개")
    print(f"  - 사용자 검증 완료 샘플: {audited_count}개 ({audited_count / N_total * 100:.2f}%)")
    print(f"  - 라벨 교정(수정)된 샘플: {corrected_count}개 ({corrected_count / audited_count * 100:.2f}%)")

    print("\n[*] 결함별 라벨 교정 통계 (누락 복원 vs 오탐 정상화):")
    for d in defect_names:
        s = correction_stats[d]
        print(f"  - [{d}]: 라벨 누락 복원(0->1) {s['0_to_1']:3d}건 | 허위 결함 정상화(1->0) {s['1_to_0']:3d}건 | 총 변경 {s['0_to_1'] + s['1_to_0']:3d}건")

    print("\n[*] 라벨 교정 전/후 결함 발생률 비교:")
    for k, d in enumerate(defect_names):
        raw_pct = np.mean(Y_raw[:, k]) * 100
        gold_pct = np.mean(Y_golden[:, k]) * 100
        diff_pct = gold_pct - raw_pct
        print(f"  - [{d}]: 원천 {raw_pct:5.2f}% -> 골든 {gold_pct:5.2f}% ({diff_pct:+5.2f}%p)")

    print(f"\n[*] 완전 정상(Clean Rep) 비율:")
    print(f"  - 원천 라벨 기준: {np.sum(y_normal_raw)}개 ({np.mean(y_normal_raw)*100:.2f}%)")
    print(f"  - 골든 라벨 기준: {np.sum(y_normal_golden)}개 ({np.mean(y_normal_golden)*100:.2f}%)")

    # 4. 골든 데이터셋 저장
    np.savez_compressed(
        OUT_GOLDEN_NPZ,
        Y_defects_golden=Y_golden,
        Y_defects_raw=Y_raw,
        y_normal_golden=y_normal_golden,
        y_normal_raw=y_normal_raw,
        is_audited=is_audited,
        subjects=subjects,
        sequence_ids=seq_ids,
        base_sids=base_sids,
        views=views,
        defect_names=defect_names
    )
    print(f"\n[+] 골든 데이터셋 저장 완료: {OUT_GOLDEN_NPZ}")
    print("=" * 105)


if __name__ == '__main__':
    main()

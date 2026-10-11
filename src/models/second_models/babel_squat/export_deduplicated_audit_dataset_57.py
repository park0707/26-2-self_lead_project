# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: export_deduplicated_audit_dataset_57.py
Purpose: [중복 100% 제거] 고유 원본 영상 57선 단일 폴더 추출 및 올인원 4대 결함 통합 체크리스트 생성
"""

import os
import sys
import shutil
import csv
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

TEMP_DIR = r"D:\Data\temp"
RAW_IMAGES_DIR = r"D:\Data\01_raw_data\images\01_바벨_스쿼트"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')

DEFECT_KEYS = ['c1_spine', 'c2_gaze', 'c3_knee', 'c4_foot']
DEFECT_SHORT = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']
DEFECT_NAMES = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면 고정']


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 중복 100% 제거: 고유 원본 영상 57선 올인원 검증 데이터셋 구축")
    print("=" * 115)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_true = ds['Y_defects'][:, :4].astype(int)
    subjects = ds['subjects'].astype(int)
    N_total = len(subjects)

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    seq_ids = m['sequence_ids'][sq_mask]
    base_sids = ds['base_sids']
    views = m['views'][sq_mask]

    # 2. OOF 예측 확률 로드
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    # 3. 고확신 불일치 (P >= 0.85 & Y_true == 0) 식별
    tau_high = 0.85
    defect_candidates = []
    for k in range(4):
        fp = np.where((Y_true[:, k] == 0) & (P_pred[:, k] >= tau_high))[0]
        defect_candidates.append(set(fp))

    all_union_indices = sorted(list(set.union(*defect_candidates)))
    print(f"[*] 총 고확신 불일치 샘플 수: {len(all_union_indices)}건")

    # 4. 고유 원본 영상 폴더(folder_name) 단위로 그룹화 (Deduplication)
    grouped_by_folder = {}
    for idx in all_union_indices:
        sid = str(seq_ids[idx])
        folder_name = sid.rsplit('_', 1)[0] if '_' in sid else sid

        if folder_name not in grouped_by_folder:
            grouped_by_folder[folder_name] = {
                'folder_name': folder_name,
                'sub': int(subjects[idx]),
                'views': set(),
                'sample_indices': [],
                'suspected_defects': set(),
                'max_probs': np.zeros(4, dtype=np.float32),
                'raw_labels': Y_true[idx].copy()
            }

        grouped_by_folder[folder_name]['views'].add(str(views[idx]))
        grouped_by_folder[folder_name]['sample_indices'].append(idx)

        # 의심 결함 항목 집계
        for k in range(4):
            if idx in defect_candidates[k]:
                grouped_by_folder[folder_name]['suspected_defects'].add(k)
            # 최대 확률 갱신
            if P_pred[idx, k] > grouped_by_folder[folder_name]['max_probs'][k]:
                grouped_by_folder[folder_name]['max_probs'][k] = P_pred[idx, k]

    unique_folders = sorted(list(grouped_by_folder.values()), key=lambda x: (x['sub'], x['folder_name']))
    N_unique = len(unique_folders)
    print(f"[*] 중복 완전 제거 후 고유 원본 영상 수: {N_unique}개 세트")

    # 5. D:\Data\temp 폴더 완전 초기화
    if os.path.exists(TEMP_DIR):
        print(f"[*] 기존 {TEMP_DIR} 폴더 초기화 중...")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"[!] 폴더 초기화 알림: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 6. 고유 폴더별 프레임 복사 및 올인원 안내서 작성
    csv_rows = []

    print("\n[*] 57개 고유 원본 영상 폴더 복사 및 안내서 생성 중...")
    for rank, item in enumerate(unique_folders, start=1):
        fn = item['folder_name']
        sub = item['sub']
        susp_str = "_".join([DEFECT_SHORT[k] for k in sorted(list(item['suspected_defects']))])

        dir_tag = f"{rank:02d}_피험자{sub}_{fn}_[의심_{susp_str}]"
        dest_dir = os.path.join(TEMP_DIR, dir_tag)
        os.makedirs(dest_dir, exist_ok=True)

        # 이미지 프레임 복사
        src_folder = os.path.join(RAW_IMAGES_DIR, fn)
        copied = 0
        if os.path.exists(src_folder):
            all_files = sorted(os.listdir(src_folder))
            # 정면(시점C)과 측면(시점B) 우선 복사
            priority_files = [f for f in all_files if '시점C' in f or '시점B' in f or 'C_' in f or 'B_' in f]
            files_to_copy = priority_files if len(priority_files) > 0 else all_files[:30]
            for f in files_to_copy:
                shutil.copy2(os.path.join(src_folder, f), dest_dir)
                copied += 1

        # 00_검증안내_info.txt 작성
        info_path = os.path.join(dest_dir, "00_검증안내_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 85 + "\n")
            f.write(f"  [올인원 라벨 노이즈 판독 안내서 #{rank:02d}] - {fn}\n")
            f.write("=" * 85 + "\n")
            f.write(f"• 피험자 번호 : 피험자 {sub}번\n")
            f.write(f"• 원본 영상명 : {fn}\n")
            f.write(f"• 주요 의심 결함: {', '.join([DEFECT_NAMES[k] for k in sorted(list(item['suspected_defects']))])}\n")
            f.write("-" * 85 + "\n")
            f.write("• 4대 결함별 AI 예측 확률 vs 원천 라벨:\n")
            for k in range(4):
                is_susp = "★ [라벨 누락 의심!]" if k in item['suspected_defects'] else ""
                f.write(f"   [{DEFECT_SHORT[k]}] AI 확률: {item['max_probs'][k]*100:>5.1f}% | 원천 라벨: {item['raw_labels'][k]} {is_susp}\n")
            f.write("-" * 85 + "\n")
            f.write("• 사용자 육안 판독 요령 (동봉된 사진 프레임을 보시면서):\n")
            f.write("   1) 시점C(정면) 사진: 무릎이 안쪽으로 모이는지(c3), 시선이 정면인지(c2) 확인\n")
            f.write("   2) 시점B(측면) 사진: 허리가 굽거나 상체가 과숙여지는지(c1), 뒤꿈치가 뜨는지(c4) 확인\n")
            f.write("   3) 체크리스트 CSV에 각 결함별로 '1(결함)', '0(정상)', 'M(모호)'을 한 번에 적어주세요.\n")
            f.write("=" * 85 + "\n")

        # CSV 행 추가
        csv_rows.append({
            '번호': rank,
            '폴더명': dir_tag,
            '피험자': sub,
            '주요의심항목': susp_str,
            'c1척추_원천라벨': item['raw_labels'][0],
            'c1척추_AI확률(%)': f"{item['max_probs'][0]*100:.1f}",
            'c1척추_사용자판정': '',
            'c2시선_원천라벨': item['raw_labels'][1],
            'c2시선_AI확률(%)': f"{item['max_probs'][1]*100:.1f}",
            'c2시선_사용자판정': '',
            'c3무릎_원천라벨': item['raw_labels'][2],
            'c3무릎_AI확률(%)': f"{item['max_probs'][2]*100:.1f}",
            'c3무릎_사용자판정': '',
            'c4발목_원천라벨': item['raw_labels'][3],
            'c4발목_AI확률(%)': f"{item['max_probs'][3]*100:.1f}",
            'c4발목_사용자판정': '',
            '메모': ''
        })

    # 7. 올인원 CSV 체크리스트 저장
    csv_path = os.path.join(TEMP_DIR, "00_라벨노이즈_올인원_체크리스트_57선.csv")
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = [
            '번호', '폴더명', '피험자', '주요의심항목',
            'c1척추_원천라벨', 'c1척추_AI확률(%)', 'c1척추_사용자판정',
            'c2시선_원천라벨', 'c2시선_AI확률(%)', 'c2시선_사용자판정',
            'c3무릎_원천라벨', 'c3무릎_AI확률(%)', 'c3무릎_사용자판정',
            'c4발목_원천라벨', 'c4발목_AI확률(%)', 'c4발목_사용자판정',
            '메모'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 8. README 작성
    readme_path = os.path.join(TEMP_DIR, "README_검증가이드.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  [vidio-health-trainer] 바벨 스쿼트 라벨 노이즈 '중복 100% 제거' 57선 검증 가이드\n")
        f.write("=" * 90 + "\n\n")
        f.write("1. 개요:\n")
        f.write("   - 동일한 영상이 여러 카테고리나 반복 횟수(rep)로 쪼개져 중복되던 문제를 100% 해결했습니다.\n")
        f.write("   - 순수 고유 원본 영상 57개만 딱 1번씩 확인하시면 되므로, 검사 시간이 대폭 단축됩니다.\n\n")
        f.write("2. 판독 방법:\n")
        f.write("   1) '00_라벨노이즈_올인원_체크리스트_57선.csv' 파일을 엑셀로 엽니다.\n")
        f.write("   2) 01번부터 57번 폴더를 순서대로 열어 정면(시점C) 및 측면(시점B) 사진을 봅니다.\n")
        f.write("   3) 한 행에서 4대 결함 각각에 대해 판정을 입력합니다:\n")
        f.write("      - 1 : 실제 결함 맞음! (라벨 누락 오기재)\n")
        f.write("      - 0 : 실제 정상 맞음! (모델 오탐)\n")
        f.write("      - M : 판정 불가 (애매함)\n")
        f.write("   ※ 주요 의심 항목에 표시된 결함(확률 85% 이상)을 중점적으로 보시되, 다른 부위도 한눈에 체크하시면 됩니다.\n\n")
        f.write("=" * 90 + "\n")

    print(f"\n[✔] 중복 제거 57선 올인원 데이터셋 구축 완료: {TEMP_DIR}")
    print(f"    - 고유 세션 폴더: 단 57개 폴더!")
    print(f"    - 올인원 CSV     : {csv_path}")
    print(f"    - 검증 안내서    : {readme_path}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: export_high_confidence_noise_audit_dataset.py
Purpose: [라벨 노이즈 2차 정밀 전수 검증 데이터셋 구축]
         모델 결함 확신도 85% 이상(P >= 0.85)이나 AI-Hub 라벨은 정상(0)으로 표기된
         초고확신 위양성(High-Confidence FP, 라벨 누락 의심건) 88건을 D:/Data/temp 에 추출하고,
         카테고리별 이미지, 판독 가이드 및 CSV 체크리스트 자동 생성
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
DEFECT_NAMES = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면 고정']
FOLDER_NAMES = [
    '01_c1_척추_라벨누락의심',
    '02_c2_시선_라벨누락의심',
    '03_c3_무릎_라벨누락의심',
    '04_c4_발목_라벨누락의심'
]


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 고확신 라벨 노이즈 의심 세션 정밀 추출 및 D:/Data/temp 구축")
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

    # 3. temp 폴더 초기화
    if os.path.exists(TEMP_DIR):
        print(f"[*] 기존 {TEMP_DIR} 폴더 초기화 중...")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"[!] 폴더 초기화 알림: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 4. 결함별 고확신 FP (P >= 0.85 & Y_true == 0) 식별
    tau_high = 0.85
    defect_candidates = []

    for k in range(4):
        fp_indices = np.where((Y_true[:, k] == 0) & (P_pred[:, k] >= tau_high))[0]
        # 확신도 내림차순 정렬
        fp_indices = fp_indices[np.argsort(-P_pred[fp_indices, k])]
        defect_candidates.append(fp_indices)
        print(f"  • {DEFECT_NAMES[k]}: 라벨 0이나 결함확률 {tau_high*100:.0f}% 이상인 후보 = {len(fp_indices)}건")

    # 5. 파일 복사 및 안내서 생성
    csv_rows = []
    total_copied_sessions = 0

    for k in range(4):
        target_subfolder = os.path.join(TEMP_DIR, FOLDER_NAMES[k])
        os.makedirs(target_subfolder, exist_ok=True)
        candidates = defect_candidates[k]

        print(f"\n[*] {FOLDER_NAMES[k]} 카테고리 추출 중 ({len(candidates)}건)...")

        for seq_rank, idx in enumerate(candidates, start=1):
            sid = str(seq_ids[idx])
            bsid = str(base_sids[idx])
            sub = int(subjects[idx])
            view = str(views[idx])

            # 원본 세션 폴더명 파싱
            folder_name = sid.rsplit('_', 1)[0] if '_' in sid else sid
            session_dir_tag = f"{seq_rank:02d}_피험자{sub}_{view}_{folder_name}"
            dest_dir = os.path.join(target_subfolder, session_dir_tag)
            os.makedirs(dest_dir, exist_ok=True)

            # 이미지 프레임 복사
            src_folder = os.path.join(RAW_IMAGES_DIR, folder_name)
            copied = 0
            if os.path.exists(src_folder):
                all_files = sorted(os.listdir(src_folder))
                priority_files = [f for f in all_files if '시점C' in f or '시점B' in f or 'C_' in f or 'B_' in f]
                files_to_copy = priority_files if len(priority_files) > 0 else all_files[:25]
                for f in files_to_copy:
                    shutil.copy2(os.path.join(src_folder, f), dest_dir)
                    copied += 1

            # 개별 세션 안내서 작성
            info_txt_path = os.path.join(dest_dir, "00_검증안내_info.txt")
            with open(info_txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"  [라벨 노이즈 정밀 판독 안내서] - {folder_name}\n")
                f.write("=" * 80 + "\n")
                f.write(f"1. 의심 결함 항목 : {DEFECT_NAMES[k]}\n")
                f.write(f"2. 피험자 / 시점  : 피험자 {sub}번 / 카메라 {view}\n")
                f.write(f"3. 고유 세션 ID   : {sid}\n")
                f.write(f"4. AI-Hub 원천라벨: [정상 (0)] 으로 표기되어 있음\n")
                f.write(f"5. AI 모델 결함확률: {P_pred[idx, k]*100:.1f}% (압도적 결함 확신!)\n")
                f.write("   - 4대 결함 전체 확률:\n")
                f.write(f"     * c1 (척추 중립): {P_pred[idx, 0]*100:.1f}%\n")
                f.write(f"     * c2 (고개 정면): {P_pred[idx, 1]*100:.1f}%\n")
                f.write(f"     * c3 (무릎 외반): {P_pred[idx, 2]*100:.1f}%\n")
                f.write(f"     * c4 (발바닥 고정): {P_pred[idx, 3]*100:.1f}%\n")
                f.write("-" * 80 + "\n")
                f.write("6. 사용자 판독 가이드 (동봉된 사진 프레임을 보실 때):\n")
                if k == 0:
                    f.write("   ▶ [c1 척추] 시점B(측면) 사진을 보시고 하강/최하사점에서 허리가 둥글게 말리거나(굴곡)\n")
                    f.write("      상체가 앞으로 과도하게 쏟아지는 결함이 있는지 확인해 주세요.\n")
                elif k == 1:
                    f.write("   ▶ [c2 시선] 시점C(정면) 또는 B(측면) 사진에서 바닥을 보며 고개를 푹 숙이거나\n")
                    f.write("      목을 과도하게 꺾는 결함이 있는지 확인해 주세요.\n")
                elif k == 2:
                    f.write("   ▶ [c3 무릎] 시점C(정면) 사진에서 최하사점으로 내려갈 때 양 무릎이 안쪽으로 모이는지 확인해 주세요.\n")
                else:
                    f.write("   ▶ [c4 발목] 시점B(측면) 사진에서 최하사점이나 상승 시 발뒤꿈치나 앞꿈치가 지면에서 뜨는지 확인해 주세요.\n")
                f.write("\n   ▶ 판독 체크: [ ] 원천라벨 누락(실제 결함 맞음!)  [ ] 모델 오탐(진짜 정상임)  [ ] 모호함\n")
                f.write("=" * 80 + "\n")

            total_copied_sessions += 1
            csv_rows.append({
                '번호': total_copied_sessions,
                '의심항목': DEFECT_NAMES[k],
                '폴더경로': os.path.relpath(dest_dir, TEMP_DIR),
                '피험자': sub,
                '시점': view,
                '원천라벨': 0,
                '모델결함확률(%)': f"{P_pred[idx, k]*100:.1f}",
                'c1확률': f"{P_pred[idx, 0]*100:.1f}",
                'c2확률': f"{P_pred[idx, 1]*100:.1f}",
                'c3확률': f"{P_pred[idx, 2]*100:.1f}",
                'c4확률': f"{P_pred[idx, 3]*100:.1f}",
                '사용자판정(1:실제결함, 0:실제정상, M:모호)': ''
            })

    # 6. CSV 체크리스트 저장
    csv_path = os.path.join(TEMP_DIR, "00_라벨노이즈_전수검증_체크리스트.csv")
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = [
            '번호', '의심항목', '폴더경로', '피험자', '시점', '원천라벨',
            '모델결함확률(%)', 'c1확률', 'c2확률', 'c3확률', 'c4확률',
            '사용자판정(1:실제결함, 0:실제정상, M:모호)'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 7. 최상위 README 작성
    readme_path = os.path.join(TEMP_DIR, "README_검증가이드.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  [vidio-health-trainer] 바벨 스쿼트 고확신 라벨 노이즈 2차 전수 검증 가이드\n")
        f.write("=" * 90 + "\n\n")
        f.write("1. 목적:\n")
        f.write("   - AI-Hub 원천 라벨에는 '정상(0)'으로 되어 있으나, 우리 모델이 결함 확률 85% 이상으로 확신하는\n")
        f.write("     초고확신 라벨 누락 의심 세션들을 사용자님께서 육안으로 판독하여 정제합니다.\n")
        f.write("   - 판독 완료된 정답표는 '골든 벤치마크 데이터셋'으로 승격되어 향후 모델 학습 및 공식 평가에 사용됩니다.\n\n")
        f.write("2. 추출 구성 (총 4개 카테고리, 총 92개 검증 세션 폴더):\n")
        f.write("   • 01_c1_척추_라벨누락의심 : 26건 (허리 굽힘, 상체 과숙임 의심)\n")
        f.write("   • 02_c2_시선_라벨누락의심 : 21건 (고개 숙임, 바닥 응시 의심)\n")
        f.write("   • 03_c3_무릎_라벨누락의심 : 19건 (최하사점 무릎 안쪽 붕괴 의심)\n")
        f.write("   • 04_c4_발목_라벨누락의심 : 26건 (뒤꿈치 들림, 지면 불안정 의심)\n\n")
        f.write("3. 판독 방법:\n")
        f.write("   1) 동봉된 '00_라벨노이즈_전수검증_체크리스트.csv'를 엑셀로 엽니다.\n")
        f.write("   2) 각 폴더 내의 시점C(정면) 및 시점B(측면) 사진을 확인합니다.\n")
        f.write("   3) 마지막 열 '사용자판정'에 아래 중 하나를 적어주시면 됩니다:\n")
        f.write("      - 1 : 모델이 맞음! (실제 결함인데 라벨이 누락된 것)\n")
        f.write("      - 0 : 라벨이 맞음! (모델이 오탐한 진짜 정상)\n")
        f.write("      - M : 너무 애매해서 판정 불가 (모호/경계선)\n\n")
        f.write("=" * 90 + "\n")

    print(f"\n[✔] 전수 검증 데이터셋 추출 및 구축 완료: {TEMP_DIR}")
    print(f"    - 총 추출 폴더 수: {total_copied_sessions}개 세션")
    print(f"    - CSV 체크리스트 : {csv_path}")
    print(f"    - 안내문        : {readme_path}")


if __name__ == '__main__':
    main()

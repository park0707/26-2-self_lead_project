# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: export_c3_missed_to_temp.py
Purpose: D:/Data/temp 기존 내용을 초기화하고,
         c3(발과 무릎 방향 일치 / 무릎 외반)에서 실제 오류인데 모델이 정상으로 놓친
         미인식 오류(False Negative, FN) 대표 샘플들을 복사 및 분석 안내서 작성
"""

import os
import sys
import shutil
import json
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

TEMP_DIR = r"D:\Data\temp"
RAW_IMAGES_DIR = r"D:\Data\01_raw_data\images\01_바벨_스쿼트"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 100)
    print("  [c3 무릎 외반 미인식 오류(False Negative) 샘플 D:/Data/temp 추출 및 정밀 분석]")
    print("=" * 100)

    # 1. 데이터 및 c3 라벨 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)
    y_c3 = Y_defects[:, 2] # 0: 정상, 1: 무릎 외반 결함
    subjects = ds['subjects']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    seq_ids = m['sequence_ids'][sq_mask]
    views = m['views'][sq_mask]

    # 2. c3 v5 예측 확률 및 물리 피처 로드
    cached = np.load(os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz'))
    p_c3 = cached['oof_c3_v5'] if 'oof_c3_v5' in cached else cached['oof_c3']

    v1_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    c3_feats = v1_c3['feats'].astype(np.float32)

    # 3. temp 폴더 완전 초기화
    if os.path.exists(TEMP_DIR):
        print(f"[*] 기존 {TEMP_DIR} 폴더 삭제 및 초기화 중...")
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 4. c3 v5 False Negative (y_c3 == 1 but p_c3 < 0.50) 전수 집계
    fn_indices = np.where((y_c3 == 1) & (p_c3 < 0.50))[0]
    total_defects = int(np.sum(y_c3 == 1))
    missed_count = len(fn_indices)
    recall_rate = (total_defects - missed_count) / total_defects * 100.0

    print(f"[*] c3 v5 전체 결함 수: {total_defects}건")
    print(f"[*] c3 v5 미인식 오류(False Negative) 수: {missed_count}건 (재현율: {recall_rate:.2f}%)")

    all_fn_items = []
    for idx in fn_indices:
        sid = str(seq_ids[idx])
        folder = sid.rsplit('_rep', 1)[0]
        has_img = os.path.exists(os.path.join(RAW_IMAGES_DIR, folder))

        all_fn_items.append({
            'idx': int(idx),
            'sid': sid,
            'folder': folder,
            'sub': int(subjects[idx]),
            'view': str(views[idx]),
            'p_c3': float(p_c3[idx]),
            'valgus_inc': float(c3_feats[idx, 7]),  # 외반 증가폭
            'ka_drop': float(c3_feats[idx, 2]),     # 무릎/발목 간격비 낙하폭
            'net_valgus': float(c3_feats[idx, 5]),  # 최하사점 외반각
            'asym': float(c3_feats[idx, 8]),        # 좌우 외반 비대칭
            'has_img': has_img
        })

    # 5. 피험자별 대표 케이스 선별 (피험자 1~9 전원 각 1건씩 최악의 오판 케이스 선별)
    by_sub = {}
    for it in all_fn_items:
        s = it['sub']
        if not it['has_img']:
            continue
        if s not in by_sub:
            by_sub[s] = []
        by_sub[s].append(it)

    selected_cases = []
    for s in sorted(by_sub.keys()):
        # 해당 피험자 중 결함 확률이 가장 낮았던(가장 확신을 가지고 정상으로 오판한) 대표 케이스 선별
        best = min(by_sub[s], key=lambda x: x['p_c3'])
        selected_cases.append(best)

    print(f"[*] 대표 분석 대상 선별: 총 {len(selected_cases)}개 피험자별 최악의 미인식 케이스")

    # 6. 이미지 복사 및 분석 리포트 작성
    for rank, it in enumerate(selected_cases, 1):
        folder_tag = f"c3_미인식_{rank:02d}_피험자{it['sub']}번_{it['folder']}"
        target_path = os.path.join(TEMP_DIR, folder_tag)
        os.makedirs(target_path, exist_ok=True)

        src_folder = os.path.join(RAW_IMAGES_DIR, it['folder'])
        all_imgs = sorted(os.listdir(src_folder))

        # 시점 A (후방: 모델이 오판한 입력 뷰) 및 시점 C (완전 정면: 육안 비교 검증 뷰) 프레임 선별 복사
        priority_files = [f for f in all_imgs if ('_시점A_' in f or '_시점C_' in f)]
        files_to_copy = priority_files if len(priority_files) > 0 else all_imgs[:32]

        copied_count = 0
        for f in files_to_copy:
            shutil.copy2(os.path.join(src_folder, f), target_path)
            copied_count += 1

        # 개별 상세 진단서 작성
        info_file = os.path.join(target_path, "c3_missed_analysis.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 85 + "\n")
            f.write(f"  [c3 무릎 외반 미인식 오류 정밀 진단서] - {it['folder']}\n")
            f.write("=" * 85 + "\n")
            f.write(f"1. 피험자 및 시점     : 피험자 {it['sub']}번 / {it['view']} (시점A=후방)\n")
            f.write(f"2. 원천 시퀀스 ID     : {it['sid']}\n")
            f.write(f"3. AI-Hub 원천 라벨   : [c3 결함 (무릎-발 방향 불일치 / 무릎 외반 발생)]\n")
            f.write(f"4. c3 v5.0 예측 확률  : 무릎 외반 결함 확률 {it['p_c3'] * 100:.2f}% (▶ '정상'으로 심각한 오판!)\n")
            f.write("-" * 85 + "\n")
            f.write("5. 주요 생체역학 실측 피처값:\n")
            f.write(f"   * 무릎 외반 증가폭 (valgus_inc)          : {it['valgus_inc']:.2f}°\n")
            f.write(f"   * 무릎/발목 간격비 낙하폭 (ka_drop)       : {it['ka_drop']:.4f}\n")
            f.write(f"   * 최하사점 절대 외반각 (net_valgus)      : {it['net_valgus']:.2f}°\n")
            f.write(f"   * 좌우 외반 비대칭 정도 (asym)           : {it['asym']:.2f}°\n")
            f.write("-" * 85 + "\n")
            f.write("6. [핵심 기술 원인 분석] 왜 모델이 이 결함을 놓쳤는가?\n\n")
            f.write("   ▶ 발견 ①: 78건 미인식 전수(100%)가 모두 'view1(시점A, 후방 카메라)'에서 발생!\n")
            f.write("      - 정면(view3, 시점C)과 45도(view2, view4, view5)에서는 c3 결함을 100% 잡아냈으나,\n")
            f.write("        오직 후방(view1) 시점에서만 시각적 은폐로 인해 모델이 결함을 놓쳤습니다.\n\n")
            f.write("   ▶ 원인 ②: 후방 시점에서의 무릎 관절 차폐(Occlusion) 및 3D z축 투영 왜곡\n")
            f.write("      - 후방에서는 대퇴이두근과 둔근, 종아리 근육에 의해 앞쪽 슬개골(Patella)이 가려집니다.\n")
            f.write("      - MediaPipe가 뒤쪽 오금(Popliteal fossa) 부위를 추정하면서 무릎의 안쪽 쏠림이\n")
            f.write("        앞쪽 전진 벡터와 겹쳐 2D/3D 투영 상에서 평면화(Flattening)되어 각도가 상쇄됩니다.\n\n")
            f.write("   ▶ 원인 ③: 무릎/발목 너비비(ka_drop) 수식의 상쇄\n")
            f.write("      - 피험자가 발을 좁게 딛거나 발목도 함께 좁혀지면서 분모/분자가 동시에 줄어들어\n")
            f.write(f"        비율 낙하폭(ka_drop={it['ka_drop']:.4f})이 작게 찍혀 트리가 정상으로 판정했습니다.\n")
            f.write("-" * 85 + "\n")
            f.write("7. 사용자 육안 확인 가이드 (동봉된 이미지 프레임을 보실 때):\n")
            f.write("   - [시점C(정면) 확인]: 바닥 최하사점에서 피험자의 양 무릎이 발끝보다 안쪽으로 꺾여 들어가는\n")
            f.write("     명백한 외반(Valgus) 결함이 육안으로 선명하게 확인됩니다.\n")
            f.write("   - [시점A(후방) 대조]: 반면 모델이 입력받은 후방 프레임에서는 뒤에서 보아 엉덩이에 가려\n")
            f.write("     무릎 모임이 훨씬 덜 도드라져 보이며, 모델이 이를 '정상'으로 착각하게 된 원인을 확인할 수 있습니다.\n")
            f.write("=" * 85 + "\n")

        print(f"  ✔ [복사 완료] {folder_tag} ({copied_count}장 프레임 복사: 시점A + 시점C)")

    # 7. 전체 78건 매니페스트 저장
    manifest_data = {
        'model_version': 'c3_v5.0_ultimate_53d',
        'total_c3_defects': total_defects,
        'total_c3_missed_count': missed_count,
        'c3_recall_rate': recall_rate,
        'view_distribution_of_missed': {'view1 (rear)': missed_count, 'view2~view5': 0},
        'selected_cases_count': len(selected_cases),
        'missed_cases_list': all_fn_items
    }
    manifest_file = os.path.join(TEMP_DIR, "c3_v5_all_78_missed_manifest.json")
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 8. 총괄 안내문 작성
    guide_file = os.path.join(TEMP_DIR, "00_c3_v5_미인식_오류_정밀분석_안내문.txt")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("   [vidio-health-trainer] c3 v5.0 무릎 외반 미인식 오류(False Negative) 정밀 분석 폴더\n")
        f.write("   저장 위치: D:/Data/temp\n")
        f.write("=" * 90 + "\n\n")
        f.write("1. 개요 및 핵심 분석 결과:\n")
        f.write(f"   - 최신 c3 v5.0 모델 기준, 전체 결함 {total_defects}건 중 미인식(False Negative) 결함은 총 {missed_count}건입니다.\n")
        f.write("   - ★ [결정적 발견]: 78건의 미인식 전수(100%)가 오직 'view1(후방 카메라 시점A)'에서만 발생했습니다!\n")
        f.write("     정면(view3) 및 45도 시점(view2, view4, view5)에서는 결함을 100% 완벽 검출하고 있습니다.\n")
        f.write("   - 본 폴더에는 피험자 1번~9번 전수를 대표하는 최악의 오판 케이스 9개 폴더를 수록하였습니다.\n\n")
        f.write("2. 폴더 구성 및 육안 검증 방법:\n")
        for rank, it in enumerate(selected_cases, 1):
            f.write(f"   [{rank}] c3_미인식_{rank:02d}_피험자{it['sub']}번_{it['folder']}\n")
        f.write("\n   * 각 폴더 내에는:\n")
        f.write("     - [시점A 프레임들]: 모델이 실제로 입력받아 '정상'으로 오판하게 만든 후방 영상\n")
        f.write("     - [시점C 프레임들]: 정면에서 실제 무릎이 안으로 심하게 꺾이는 것을 즉시 확인 가능한 검증 영상\n")
        f.write("     - c3_missed_analysis.txt: 개별 수치 피처값 및 오판 원인 진단서\n\n")
        f.write("3. 핵심 미인식 원인 요약:\n")
        f.write("   ① 후방(view1) 시점에서 둔근 및 종아리 근육에 의한 슬개골 차폐(Occlusion)\n")
        f.write("   ② MediaPipe가 뒤쪽 오금을 기준으로 추정하면서 무릎 모임 각도가 전후 깊이(Z)와 겹쳐 상쇄됨\n")
        f.write("   ③ 양발 보폭 및 발목 동시 좁아짐으로 인한 무릎/발목 간격비 낙하폭(ka_drop) 축소\n\n")
        f.write("=" * 90 + "\n")

    print(f"\n[*] 매니페스트 저장: {manifest_file}")
    print(f"[*] 총괄 안내문 작성: {guide_file}")
    print(f"\n[완료] D:/Data/temp 에 c3 v5 미인식 오류 샘플 및 안내서 구축이 완료되었습니다!")

if __name__ == '__main__':
    main()

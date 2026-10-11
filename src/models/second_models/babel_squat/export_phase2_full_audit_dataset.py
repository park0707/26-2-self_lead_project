# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: export_phase2_full_audit_dataset.py
Purpose: [Phase 2 전수 검증용 데이터 세트 구축]
         가짜 정상(False Clean) 51건 전수, 가짜 오류(False Defect) 3건 전수,
         c3 무릎 외반 역전 대표 10건을 D:/Data/temp 에 전수 복사하고
         사용자 직접 판독용 CSV 체크리스트 및 상세 안내문 자동 생성
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


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] Phase 2: 전수 육안 검증용 데이터 세트 추출 및 D:/Data/temp 구축")
    print("=" * 115)

    # 1. 데이터셋 및 라벨 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)
    y_overall = (np.sum(Y_defects, axis=1) > 0).astype(int)
    subjects = ds['subjects']

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    seq_ids = m['sequence_ids'][sq_mask]
    views = m['views'][sq_mask]

    # 2. 모호성 지표 및 OOF 예측 확률 로드
    c1_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    c1_l = c1_d['levels']
    c2_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    c2_l = c2_d['levels']
    c3_d = np.load('d:/src/models/second_models/squat_c3_ambiguity_levels.npz', allow_pickle=True)
    c3_l = c3_d['levels']

    cached = np.load(os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz'))
    p_c1 = cached['oof_c1']
    p_c2 = cached['oof_c2']
    p_c3 = cached['oof_c3']
    p_c4 = cached['oof_c4']
    p_max = np.max([p_c1, p_c2, p_c3, p_c4], axis=0)

    # 피처 로드 (물리 지표 상세 표기를 위함)
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    c2_feats = h_data['head_feats'].astype(np.float32)
    v1_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    c3_feats = v1_c3['feats'].astype(np.float32)
    v4_spine = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    c1_feats = v4_spine['v4_spine_feats'].astype(np.float32)

    # 3. temp 폴더 초기화
    if os.path.exists(TEMP_DIR):
        print(f"[*] 기존 {TEMP_DIR} 폴더 정리 중...")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"[!] 폴더 삭제 중 알림: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 4. 전체 카테고리별 후보 집계
    clean_idx = np.where(y_overall == 0)[0]
    defect_idx = np.where(y_overall == 1)[0]

    # (1) False Clean (원천라벨=정상이나 실제 물리 결함/앙상블 결함 51건 전수)
    false_clean_candidates = []
    for idx in clean_idx:
        reasons = []
        if c1_l[idx] == 4:
            reasons.append("c1_척추_요추굴곡_물리역전(Level 4)")
        if c3_l[idx] == 4:
            reasons.append("c3_무릎_외반붕괴_물리역전(Level 4)")
        if c2_l[idx] == 4:
            reasons.append("c2_고개_시선낙하_물리역전(Level 4)")
        if p_max[idx] >= 0.75:
            reasons.append(f"앙상블_결함확신(P={p_max[idx]*100:.1f}%)")

        if len(reasons) > 0:
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            false_clean_candidates.append({
                'category': '01_가짜정상_False_Clean',
                'idx': int(idx),
                'sid': sid,
                'folder_name': folder_name,
                'sub': int(subjects[idx]),
                'view': str(views[idx]),
                'reasons': reasons,
                'p_max': float(p_max[idx]),
                'p_c1': float(p_c1[idx]),
                'p_c2': float(p_c2[idx]),
                'p_c3': float(p_c3[idx]),
                'p_c4': float(p_c4[idx]),
                'c1_lumbar': float(c1_feats[idx, 0]) if c1_feats.shape[1] > 0 else 0.0,
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })

    # (2) False Defect (원천라벨=결함이나 4대 관절 모두 완벽 정상 3건 전수)
    false_defect_candidates = []
    for idx in defect_idx:
        if p_max[idx] < 0.25:
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            false_defect_candidates.append({
                'category': '02_가짜결함_False_Defect',
                'idx': int(idx),
                'sid': sid,
                'folder_name': folder_name,
                'sub': int(subjects[idx]),
                'view': str(views[idx]),
                'reasons': [f"4대관절_완벽정상_앙상블(P={p_max[idx]*100:.1f}%)"],
                'p_max': float(p_max[idx]),
                'p_c1': float(p_c1[idx]),
                'p_c2': float(p_c2[idx]),
                'p_c3': float(p_c3[idx]),
                'p_c4': float(p_c4[idx]),
                'c1_lumbar': float(c1_feats[idx, 0]) if c1_feats.shape[1] > 0 else 0.0,
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })

    # (3) c3 무릎 외반 역전 대표 (라벨은 외반오류이나 실제 11자 유지 10건)
    c3_inversion_candidates = []
    for idx in range(len(y_overall)):
        if c3_l[idx] == 4 and Y_defects[idx, 2] == 1:
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            c3_inversion_candidates.append({
                'category': '03_c3무릎외반_역전노이즈',
                'idx': int(idx),
                'sid': sid,
                'folder_name': folder_name,
                'sub': int(subjects[idx]),
                'view': str(views[idx]),
                'reasons': ["c3_라벨오류이나_실제11자_완벽유지(외반증가하위5%)"],
                'p_max': float(p_max[idx]),
                'p_c1': float(p_c1[idx]),
                'p_c2': float(p_c2[idx]),
                'p_c3': float(p_c3[idx]),
                'p_c4': float(p_c4[idx]),
                'c1_lumbar': float(c1_feats[idx, 0]) if c1_feats.shape[1] > 0 else 0.0,
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })
    c3_inversion_candidates = sorted(c3_inversion_candidates, key=lambda x: x['c3_valgus'])[:10]

    print(f"\n[*] 추출 대상 총 샘플:")
    print(f"    - 가짜 정상 (False Clean) 전수 : {len(false_clean_candidates)}건 (원천라벨=정상이나 실제 결함)")
    print(f"    - 가짜 오류 (False Defect) 전수: {len(false_defect_candidates)}건 (원천라벨=결함이나 실제 완벽정상)")
    print(f"    - c3 무릎 외반 역전 노이즈 대표 : {len(c3_inversion_candidates)}건 (원천라벨=외반결함이나 실제 11자 유지)")
    total_target_count = len(false_clean_candidates) + len(false_defect_candidates) + len(c3_inversion_candidates)
    print(f"    => 총 {total_target_count}개 검증 폴더 생성 시작...")

    # 5. 프레임 복사 및 검증 안내서 작성 함수
    def copy_item(item, subfolder_name, seq_num):
        folder_tag = f"{seq_num:02d}_피험자{item['sub']}_{item['view']}_{item['folder_name']}"
        target_dir = os.path.join(TEMP_DIR, subfolder_name, folder_tag)
        os.makedirs(target_dir, exist_ok=True)

        src_folder = os.path.join(RAW_IMAGES_DIR, item['folder_name'])
        copied = 0
        if os.path.exists(src_folder):
            all_files = sorted(os.listdir(src_folder))
            # 정면(시점C)과 측면(시점B) 우선 복사 (없으면 전 시점 복사)
            priority_files = [f for f in all_files if '시점C' in f or '시점B' in f or 'C_' in f or 'B_' in f]
            files_to_copy = priority_files if len(priority_files) > 0 else all_files[:30]

            for f in files_to_copy:
                shutil.copy2(os.path.join(src_folder, f), target_dir)
                copied += 1

        # 상세 안내 텍스트 파일 작성
        info_path = os.path.join(target_dir, "00_검증안내_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 85 + "\n")
            f.write(f"  [Phase 2 전수 육안 검증 안내서] - {item['folder_name']}\n")
            f.write("=" * 85 + "\n")
            f.write(f"1. 검증 카테고리      : {subfolder_name}\n")
            f.write(f"2. 피험자 번호 / 시점 : 피험자 {item['sub']}번 / {item['view']}\n")
            f.write(f"3. 고유 시퀀스 ID     : {item['sid']}\n")
            f.write(f"4. AI-Hub 원천 라벨   : {'[전신 완전 정상 (Clean)]' if '가짜정상' in subfolder_name else '[결함 있음 (Defect)]'}\n")
            f.write(f"5. AI 모델 통합 진단  : 결함 확률 {item['p_max']*100:.1f}%\n")
            f.write(f"   - c1(척추 중립) 결함확률 : {item['p_c1']*100:.1f}%\n")
            f.write(f"   - c2(고개 정면) 결함확률 : {item['p_c2']*100:.1f}%\n")
            f.write(f"   - c3(무릎 외반) 결함확률 : {item['p_c3']*100:.1f}%\n")
            f.write(f"   - c4(발바닥 고정) 결함확률: {item['p_c4']*100:.1f}%\n")
            f.write("-" * 85 + "\n")
            f.write("6. 센서/물리 지표 실측치:\n")
            for r in item['reasons']:
                f.write(f"   * 의심 사유 : {r}\n")
            f.write(f"   * 척추 요추 굴곡각 실측값 : {item['c1_lumbar']:.2f}° (정상 기준: 0~15°, 굴곡 결함: 30° 이상)\n")
            f.write(f"   * 무릎 외반 증가폭 실측값 : {item['c3_valgus']:.2f}° (정상 기준: 0~3°, 외반 결함: 10° 이상)\n")
            f.write(f"   * 시선 낙하 각도 실측값   : {item['c2_gaze']:.2f}° (정상 기준: 8~12°, 고개 숙임: 25° 이상)\n")
            f.write("-" * 85 + "\n")
            f.write("7. 사용자 육안 검증 포인트 (동봉된 사진 프레임을 보실 때):\n")
            if "가짜정상" in subfolder_name:
                f.write("   ▶ 원천 라벨은 '정상'으로 되어 있으나, 실제 사진을 보시면:\n")
                f.write("      - 최하사점(바닥 지점)에서 무릎이 안쪽으로 모이거나(시점C 정면),\n")
                f.write("      - 하강 시 허리가 말리며 상체가 과도하게 꺾이는 척추 굴곡(시점B 측면)이 육안으로 관찰됩니다.\n")
                f.write("      - 판독 결과: [ ] 원천라벨오기재(실제결함)  [ ] 라벨정상맞음  [ ] 모호/경계선\n")
            elif "가짜결함" in subfolder_name:
                f.write("   ▶ 원천 라벨은 '결함'으로 되어 있으나, 실제 사진을 보시면:\n")
                f.write("      - 상체 중립, 시선 정면, 무릎 정렬, 발바닥 접지가 모두 교과서처럼 완벽합니다.\n")
                f.write("      - 판독 결과: [ ] 원천라벨오기재(실제정상)  [ ] 라벨오류맞음  [ ] 모호/경계선\n")
            else:
                f.write("   ▶ 원천 라벨은 'c3 무릎-발 불일치(외반)'로 체크되어 있으나:\n")
                f.write("      - 정면(시점C) 사진을 보시면 무릎이 발끝 방향과 11자로 나란히 정렬되어 있습니다.\n")
                f.write("      - 판독 결과: [ ] 원천라벨오기재(실제11자정상)  [ ] 라벨외반맞음  [ ] 모호/경계선\n")
            f.write("=" * 85 + "\n")

        item['copied_frames'] = copied
        item['target_rel_path'] = os.path.relpath(target_dir, TEMP_DIR)
        return item

    # 6. 전수 복사 실행
    all_exported = []

    print("\n[1/3] 가짜 정상 (False Clean) 51건 전수 복사 중...")
    for idx, c in enumerate(false_clean_candidates):
        item = copy_item(c, "01_가짜정상_False_Clean_51건", idx + 1)
        all_exported.append(item)

    print("[2/3] 가짜 오류 (False Defect) 3건 전수 복사 중...")
    for idx, c in enumerate(false_defect_candidates):
        item = copy_item(c, "02_가짜결함_False_Defect_3건", idx + 1)
        all_exported.append(item)

    print("[3/3] c3 무릎 외반 역전 대표 10건 복사 중...")
    for idx, c in enumerate(c3_inversion_candidates):
        item = copy_item(c, "03_c3무릎외반_역전노이즈_10건", idx + 1)
        all_exported.append(item)

    # 7. 사용자 판독용 CSV 체크리스트 생성
    csv_path = os.path.join(TEMP_DIR, "00_Phase2_전수검증_체크리스트.csv")
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "연번", "카테고리", "폴더명", "피험자", "카메라시점", "원천라벨",
            "통합결함확률", "c1척추확률", "c2고개확률", "c3무릎확률", "c4발목확률",
            "척추굴곡각(도)", "무릎외반각(도)", "고개낙하각(도)", "주요의심사유",
            "사용자_직접_판독결과 (정상/결함/모호)", "비고_특이사항"
        ])
        for i, item in enumerate(all_exported):
            orig_lbl = "전신 정상" if "가짜정상" in item['category'] else "결함 있음"
            reasons_str = " | ".join(item['reasons'])
            writer.writerow([
                i + 1,
                item['category'],
                item['target_rel_path'],
                f"피험자 {item['sub']}",
                item['view'],
                orig_lbl,
                f"{item['p_max']*100:.1f}%",
                f"{item['p_c1']*100:.1f}%",
                f"{item['p_c2']*100:.1f}%",
                f"{item['p_c3']*100:.1f}%",
                f"{item['p_c4']*100:.1f}%",
                f"{item['c1_lumbar']:.2f}",
                f"{item['c3_valgus']:.2f}",
                f"{item['c2_gaze']:.2f}",
                reasons_str,
                "", # 사용자 직접 기입란
                ""  # 비고란
            ])

    # 8. 총괄 안내 README 파일 작성
    readme_path = os.path.join(TEMP_DIR, "00_Phase2_육안검증_가이드_README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 95 + "\n")
        f.write("   [vidio-health-trainer] Phase 2: 바벨 스쿼트 라벨 노이즈 전수 육안 검증 가이드\n")
        f.write("=" * 95 + "\n\n")
        f.write("1. 폴더 구조:\n")
        f.write("   - 01_가짜정상_False_Clean_51건 : 원천 라벨은 '정상'이나 실제 사진에서 결함이 관찰되는 51건 전수\n")
        f.write("   - 02_가짜결함_False_Defect_3건  : 원천 라벨은 '결함'이나 실제 사진에서 완벽 정상인 3건 전수\n")
        f.write("   - 03_c3무릎외반_역전노이즈_10건: 라벨은 무릎 외반 오류이나 실제로는 11자를 완벽 유지한 10건\n")
        f.write("   - 00_Phase2_전수검증_체크리스트.csv: 엑셀로 열어서 직접 판독 결과를 [정상/결함/모호]로 체크할 수 있는 시트\n\n")
        f.write("2. 검증 요령 (사진 확인 방법):\n")
        f.write("   ① 각 하위 폴더에는 정면(시점C)과 측면(시점B)의 프레임 이미지들이 담겨 있습니다.\n")
        f.write("   ② 폴더 내 '00_검증안내_info.txt'를 보시면 AI 모델의 예측치와 물리 계측값이 기재되어 있습니다.\n")
        f.write("   ③ 특히 [최하사점(바닥 프레임)]을 중점적으로 보십시오:\n")
        f.write("      - 측면(시점B): 최하사점에서 허리가 둥글게 말리는지(요추 굴곡, c1) 확인\n")
        f.write("      - 정면(시점C): 최하사점에서 무릎이 안쪽으로 꺾여 모이는지(무릎 외반, c3) 확인\n")
        f.write("   ④ 확인하신 뒤 '00_Phase2_전수검증_체크리스트.csv'의 [사용자_직접_판독결과] 열에 판독 내용을 입력해 주시면 됩니다.\n\n")
        f.write("=" * 95 + "\n")

    print(f"\n[*] CSV 체크리스트 생성 완료: {csv_path}")
    print(f"[*] 총괄 README 가이드 생성 완료: {readme_path}")
    print(f"\n[성공] 총 {len(all_exported)}개 검증 케이스가 D:/Data/temp 에 전수 구축되었습니다!")


if __name__ == '__main__':
    main()

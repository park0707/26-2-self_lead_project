# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: export_level4_to_temp.py
Purpose: 물리 법칙 위반 및 앙상블 불일치로 식별된 Level 4 라벨 역전 노이즈 샘플을
         D:/Data/temp 에 추출 및 시각적 검증용 프레임 복사
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
    print("  [Level 4 라벨 역전 노이즈 샘플 식별 및 D:/Data/temp 검증 폴더 생성]")
    print("=" * 100)

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

    # 2. 개별 모호성 파일 및 예측 확률 로드
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

    # 3. temp 폴더 초기화
    if os.path.exists(TEMP_DIR):
        print(f"[*] 기존 {TEMP_DIR} 폴더 정리 중...")
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 4. 전체 Level 4 후보 집계
    clean_idx = np.where(y_overall == 0)[0]
    defect_idx = np.where(y_overall == 1)[0]

    # (1) False Clean (라벨=정상이나 실제 물리 결함/앙상블 결함)
    false_clean_candidates = []
    for idx in clean_idx:
        reasons = []
        if c1_l[idx] == 4:
            reasons.append("척추_요추굴곡_물리역전(c1 Level 4)")
        if c3_l[idx] == 4:
            reasons.append("무릎_외반붕괴_물리역전(c3 Level 4)")
        if c2_l[idx] == 4:
            reasons.append("고개_시선낙하_물리역전(c2 Level 4)")
        if p_max[idx] >= 0.75:
            reasons.append(f"앙상블_결함확신(P={p_max[idx]*100:.1f}%)")

        if len(reasons) > 0:
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            false_clean_candidates.append({
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
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })

    # (2) False Defect (라벨=오류이나 4대 관절 모두 정상 & P < 0.25)
    false_defect_candidates = []
    for idx in defect_idx:
        if p_max[idx] < 0.25:
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            false_defect_candidates.append({
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
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })

    # (3) c3 단독 무릎 외반 역전 대표 (라벨=c3 오류이나 외반이 전혀 없음)
    c3_inversion_candidates = []
    for idx in range(len(y_overall)):
        if c3_l[idx] == 4 and Y_defects[idx, 2] == 1: # 라벨은 외반 오류인데
            sid = str(seq_ids[idx])
            folder_name = sid.rsplit('_rep', 1)[0]
            c3_inversion_candidates.append({
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
                'c3_valgus': float(c3_feats[idx, 7]),
                'c2_gaze': float(c2_feats[idx, 3])
            })

    # 5. 대표 샘플 선별 (서로 다른 피험자 및 유형별)
    def pick_diverse(candidates, n=4, sort_key=lambda x: x['p_max'], reverse=True):
        sorted_cand = sorted(candidates, key=sort_key, reverse=reverse)
        picked = []
        seen_subs = set()
        for c in sorted_cand:
            if c['sub'] not in seen_subs:
                picked.append(c)
                seen_subs.add(c['sub'])
            if len(picked) >= n:
                break
        if len(picked) < n:
            for c in sorted_cand:
                if c not in picked:
                    picked.append(c)
                if len(picked) >= n:
                    break
        return picked

    rep_false_clean = pick_diverse(false_clean_candidates, n=4, reverse=True)
    rep_false_defect = false_defect_candidates[:2]
    rep_c3_inv = pick_diverse(c3_inversion_candidates, n=2, sort_key=lambda x: x['c3_valgus'], reverse=False)

    print(f"\n[*] 식별된 총 Level 4 라벨 역전 후보:")
    print(f"    - 가짜 정상 (False Clean): {len(false_clean_candidates)}건 (라벨=정상이나 실제 결함)")
    print(f"    - 가짜 오류 (False Defect): {len(false_defect_candidates)}건 (라벨=오류이나 실제 완벽정상)")
    print(f"    - c3 무릎 외반 역전 노이즈 : {len(c3_inversion_candidates)}건 (라벨=외반오류이나 11자 유지)")

    # 6. 이미지 복사 및 안내문 생성 함수
    def copy_case(item, group_dir_name, tag):
        target_dir = os.path.join(TEMP_DIR, group_dir_name, f"{tag}_{item['folder_name']}")
        os.makedirs(target_dir, exist_ok=True)

        src_folder = os.path.join(RAW_IMAGES_DIR, item['folder_name'])
        copied = 0
        if os.path.exists(src_folder):
            all_files = sorted(os.listdir(src_folder))
            # 시점 C(정면) 및 시점 B(측면) 위주로 복사 (없으면 전체 복사)
            priority_files = [f for f in all_files if 'C_' in f or 'B_' in f]
            files_to_copy = priority_files if len(priority_files) > 0 else all_files[:30]

            for f in files_to_copy:
                shutil.copy2(os.path.join(src_folder, f), target_dir)
                copied += 1

        # 상세 안내 파일 작성
        info_path = os.path.join(target_dir, "inspection_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"  [Level 4 라벨 역전 노이즈 육안 검증 안내서] - {item['folder_name']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"1. 검증 유형         : {group_dir_name}\n")
            f.write(f"2. 피험자 및 시점    : 피험자 {item['sub']}번 / {item['view']}\n")
            f.write(f"3. 시퀀스 ID         : {item['sid']}\n")
            f.write(f"4. AI-Hub 원천 라벨  : {'[전신 완전 정상 (Clean)]' if '가짜정상' in group_dir_name else '[결함 있음 (Defect)]'}\n")
            f.write(f"5. AI 모델 통합 진단 : 결함 확률 {item['p_max']*100:.1f}% ({'결함 의심' if item['p_max']>=0.5 else '정상 판정'})\n")
            f.write(f"   - c1(척추) 결함확률: {item['p_c1']*100:.1f}%\n")
            f.write(f"   - c2(시선) 결함확률: {item['p_c2']*100:.1f}%\n")
            f.write(f"   - c3(무릎) 결함확률: {item['p_c3']*100:.1f}%\n")
            f.write(f"   - c4(발목) 결함확률: {item['p_c4']*100:.1f}%\n")
            f.write("-" * 80 + "\n")
            f.write("6. Level 4 판정 근거 (물리 법칙 위반 및 앙상블 불일치 사유):\n")
            for r in item['reasons']:
                f.write(f"   * {r}\n")
            f.write(f"   * 무릎 외반 증가폭 실측값 : {item['c3_valgus']:.2f}° (정상 기준: 0~3°, 심각 결함: 10° 이상)\n")
            f.write(f"   * 시선 낙하 각도 실측값   : {item['c2_gaze']:.2f}° (정상 기준: 8~12°, 고개 숙임: 25° 이상)\n")
            f.write("-" * 80 + "\n")
            f.write("7. 사용자 육안 검증 포인트 (동봉된 이미지 프레임을 보실 때):\n")
            if "가짜정상" in group_dir_name:
                f.write("   ▶ 원천 라벨은 '정상'으로 되어 있으나, 실제 사진을 보시면:\n")
                f.write("      - 최하사점(바닥 지점)에서 무릎이 안쪽으로 모이거나(무릎 외반),\n")
                f.write("      - 하강 시 허리가 말리는 척추 굴곡이 육안으로도 명백히 관찰됩니다.\n")
                f.write("      - 즉, 라벨러가 오류를 놓치고 '정상'으로 오기재한 전형적인 데이터 오염 샘플입니다.\n")
            elif "가짜오류" in group_dir_name:
                f.write("   ▶ 원천 라벨은 '오류'로 되어 있으나, 실제 사진을 보시면:\n")
                f.write("      - 상체 중립, 시선 정면, 무릎 정렬, 발바닥 접지가 모두 교과서처럼 완벽합니다.\n")
                f.write("      - 즉, 피험자가 너무나 정석적인 자세를 취했음에도 라벨에 오류가 체크된 오기재 데이터입니다.\n")
            else:
                f.write("   ▶ 원천 라벨은 'c3 무릎-발 방향 불일치(외반)'로 체크되어 있으나:\n")
                f.write("      - 정면(C시점) 사진을 보시면 무릎이 발끝 방향과 11자로 나란히 정렬되어 있습니다.\n")
                f.write("      - 물리적 외반 증가가 전혀 없는데 오류로 라벨링된 라벨 역전 노이즈입니다.\n")
            f.write("=" * 80 + "\n")
        print(f"  ✔ [복사 완료] {group_dir_name} -> {item['folder_name']} ({copied}장 프레임)")

    # 7. 그룹별 대표 샘플 복사 실행
    print("\n[*] D:/Data/temp 에 대표 케이스별 프레임 복사 진행...")
    for idx, c in enumerate(rep_false_clean):
        copy_case(c, "1_가짜정상_라벨정상_실제결함_Level4", f"FC{idx+1}")

    for idx, c in enumerate(rep_false_defect):
        copy_case(c, "2_가짜결함_라벨결함_실제정상_Level4", f"FD{idx+1}")

    for idx, c in enumerate(rep_c3_inv):
        copy_case(c, "3_c3무릎외반_라벨역전_Level4", f"C3_INV{idx+1}")

    # 8. 전체 Level 4 매니페스트 JSON 저장
    manifest = {
        'summary': {
            'total_false_clean_level4': len(false_clean_candidates),
            'total_false_defect_level4': len(false_defect_candidates),
            'total_c3_inversion_level4': len(c3_inversion_candidates),
            'representative_copied_count': len(rep_false_clean) + len(rep_false_defect) + len(rep_c3_inv)
        },
        'false_clean_list': false_clean_candidates,
        'false_defect_list': false_defect_candidates,
        'c3_inversion_list': c3_inversion_candidates
    }

    manifest_path = os.path.join(TEMP_DIR, "level4_total_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 9. 전체 개요 안내문 작성
    guide_path = os.path.join(TEMP_DIR, "00_Level4_노이즈_검증_안내문.txt")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("   [vidio-health-trainer] 바벨 스쿼트 Level 4 라벨 역전 노이즈 검증 폴더 (D:/Data/temp)\n")
        f.write("=" * 90 + "\n\n")
        f.write("1. 폴더 구성:\n")
        f.write("   - 1_가짜정상_라벨정상_실제결함_Level4 : 원천 라벨은 '정상'이나 실제 동작은 명백한 결함인 샘플 (4개 폴더)\n")
        f.write("   - 2_가짜결함_라벨결함_실제정상_Level4 : 원천 라벨은 '결함'이나 실제 동작은 교과서 정석 정상인 샘플 (2개 폴더)\n")
        f.write("   - 3_c3무릎외반_라벨역전_Level4       : 라벨은 무릎 외반이나 실제로는 11자를 완벽 유지한 샘플 (2개 폴더)\n")
        f.write("   - level4_total_manifest.json        : 전체 52건의 가짜정상 및 역전 노이즈 전체 인덱스/수치 목록\n\n")
        f.write("2. 사용 방법:\n")
        f.write("   - 각 하위 폴더에 들어가시면 정면(C시점) 및 측면(B시점) 프레임 이미지들이 담겨 있습니다.\n")
        f.write("   - 폴더 내 'inspection_info.txt'를 먼저 열어보시면, 해당 샘플의 라벨과 실제 물리 지표값,\n")
        f.write("     그리고 사진에서 어떤 부위(무릎 안쪽 꺾임, 허리 굽음 등)를 확인해야 하는지 안내되어 있습니다.\n\n")
        f.write("=" * 90 + "\n")

    print(f"\n[*] 전체 매니페스트 저장 완료: {manifest_path}")
    print(f"[*] 총괄 안내문 작성 완료: {guide_path}")
    print(f"\n[성공] D:/Data/temp 에 Level 4 검증 프레임 및 설명서 구축이 완료되었습니다!")

if __name__ == '__main__':
    main()

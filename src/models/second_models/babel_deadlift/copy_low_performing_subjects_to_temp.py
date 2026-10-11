# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: copy_low_performing_subjects_to_temp.py
Purpose: 성능이 유난히 안 좋은 피험자(피험자 5, 4, 8, 3) 및 비교군(피험자 2)의
         이미지 데이터를 D:\\Data\\temp 폴더로 체계적으로 분류 복사
"""

import os
import sys
import glob
import json
import shutil

LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TEMP_BASE = r"D:\Data\temp"

# 타깃 디렉토리 정의
TARGET_FOLDERS = {
    5: os.path.join(TEMP_BASE, "01_피험자5_최저성능(Recall_0%)_미세굴곡_결함"),
    4: os.path.join(TEMP_BASE, "02_피험자4_저성능(Recall_20%)_미세굴곡_결함"),
    8: os.path.join(TEMP_BASE, "03_피험자8_저성능(Recall_20%)_약한굴곡_결함"),
    3: os.path.join(TEMP_BASE, "04_피험자3_저성능(Recall_40%)_결함"),
    2: os.path.join(TEMP_BASE, "05_비교대조군_피험자2_최우수(Recall_87%)_결함")
}

NORM_FOLDERS = {
    5: os.path.join(TEMP_BASE, "01_피험자5_정상자세"),
    4: os.path.join(TEMP_BASE, "02_피험자4_정상자세"),
    2: os.path.join(TEMP_BASE, "05_비교대조군_피험자2_정상자세")
}

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 저성능 피험자 이미지 데이터 D:\\Data\\temp 복사")
    print("=" * 80)

    # 1. 기존 temp 정리 및 디렉토리 생성
    os.makedirs(TEMP_BASE, exist_ok=True)
    for p in TARGET_FOLDERS.values():
        os.makedirs(p, exist_ok=True)
    for p in NORM_FOLDERS.values():
        os.makedirs(p, exist_ok=True)

    # 2. 메타데이터 및 이미지 경로 수집
    json_files = sorted(glob.glob(os.path.join(LABEL_JSON_DIR, "*.json")))
    print(f"[*] 총 라벨 JSON 파일: {len(json_files)}개")

    subject_data = {2: {'defect': [], 'normal': []},
                    3: {'defect': [], 'normal': []},
                    4: {'defect': [], 'normal': []},
                    5: {'defect': [], 'normal': []},
                    8: {'defect': [], 'normal': []}}

    for jf in json_files:
        base_name = os.path.basename(jf).replace(".json", "")
        img_folder = os.path.join(RAW_IMG_DIR, base_name)
        if not os.path.exists(img_folder):
            continue

        with open(jf, "r", encoding="utf-8", errors="ignore") as f:
            d = json.load(f)

        ti = d.get("type_info", {})
        conds = {c["condition"]: c["value"] for c in ti.get("conditions", [])}
        desc = ti.get("description", "")
        is_d1_defect = (conds.get("척추의 중립", True) == False)

        sub_id = None
        for s in [2, 3, 4, 5, 8]:
            if f"피험자{s}" in base_name:
                sub_id = s
                break

        if sub_id is not None:
            item = {
                "base_name": base_name,
                "img_folder": img_folder,
                "desc": desc,
                "is_d1_defect": is_d1_defect
            }
            if is_d1_defect:
                subject_data[sub_id]['defect'].append(item)
            else:
                subject_data[sub_id]['normal'].append(item)

    # 3. 선별 복사 (결함은 전수 또는 최대 5~6개, 정상은 2~3개)
    total_copied = 0

    for sub_id in [5, 4, 8, 3, 2]:
        dest_def = TARGET_FOLDERS[sub_id]
        def_items = subject_data[sub_id]['defect']
        print(f"\n[*] [피험자 {sub_id}] 결함 샘플 {len(def_items)}개 복사 시작 -> {os.path.basename(dest_def)}")

        # 결함 폴더 복사 (최대 5개 폴더)
        for item in def_items[:5]:
            folder_label = f"{item['base_name']}"
            if item['desc']:
                clean_desc = item['desc'].replace("/", "_").replace("\\", "_")
                folder_label += f"_[{clean_desc}]"
            target_path = os.path.join(dest_def, folder_label)
            if not os.path.exists(target_path):
                shutil.copytree(item['img_folder'], target_path)
                print(f"  + 결함 복사: {folder_label}")
                total_copied += 1

        # 정상 폴더 복사 (피험자 5, 4, 2 대상 각 2개)
        if sub_id in NORM_FOLDERS:
            dest_norm = NORM_FOLDERS[sub_id]
            norm_items = subject_data[sub_id]['normal']
            print(f"[*] [피험자 {sub_id}] 정상 샘플 {min(2, len(norm_items))}개 복사 시작 -> {os.path.basename(dest_norm)}")
            for item in norm_items[:2]:
                folder_label = f"{item['base_name']}"
                target_path = os.path.join(dest_norm, folder_label)
                if not os.path.exists(target_path):
                    shutil.copytree(item['img_folder'], target_path)
                    print(f"  + 정상 복사: {folder_label}")
                    total_copied += 1

    print("\n" + "=" * 80)
    print(f"  [OK] 복사 완료! 총 {total_copied}개 액션 폴더가 D:\\Data\\temp 에 구축되었습니다.")
    print("=" * 80)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: prepare_temp_inspection_folders.py
Purpose: 피험자 4, 5번의 척추 굴곡 상태 시각적 직접 검증을 위한 4대 비교 폴더 구축
         1) D:\Data\temp\피험자4_5_결함
         2) D:\Data\temp\피험자4_5_정상
         3) D:\Data\temp\다른피험자_결함
         4) D:\Data\temp\다른피험자_정상
"""

import os
import sys
import glob
import json
import shutil

LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TEMP_BASE = r"D:\Data\temp"

target_dirs = {
    "p45_def": os.path.join(TEMP_BASE, "피험자4_5_결함"),
    "p45_norm": os.path.join(TEMP_BASE, "피험자4_5_정상"),
    "other_def": os.path.join(TEMP_BASE, "다른피험자_결함"),
    "other_norm": os.path.join(TEMP_BASE, "다른피험자_정상")
}

for d in target_dirs.values():
    os.makedirs(d, exist_ok=True)

# 시퀀스별 라벨 및 설명 매핑 수집
seq_info = []
for jf in sorted(glob.glob(os.path.join(LABEL_JSON_DIR, "*.json"))):
    base = os.path.basename(jf).replace(".json", "")
    img_dir = os.path.join(RAW_IMG_DIR, base)
    if not os.path.exists(img_dir):
        continue

    with open(jf, "r", encoding="utf-8", errors="ignore") as f:
        d = json.load(f)

    ti = d.get("type_info", {})
    conds = {c["condition"]: c["value"] for c in ti.get("conditions", [])}
    desc = ti.get("description", "")
    is_d1_defect = (conds.get("척추의 중립", True) == False)

    sub = 1
    for i in range(1, 10):
        if f"피험자{i}" in base:
            sub = i
            break

    seq_info.append({
        "base": base,
        "img_dir": img_dir,
        "sub": sub,
        "is_d1_defect": is_d1_defect,
        "desc": desc
    })

# 각 카테고리별로 12개씩 선별
p45_def_samples = [s for s in seq_info if s["sub"] in [4, 5] and s["is_d1_defect"]][:12]
p45_norm_samples = [s for s in seq_info if s["sub"] in [4, 5] and not s["is_d1_defect"]][:12]
other_def_samples = [s for s in seq_info if s["sub"] not in [4, 5] and s["is_d1_defect"]][:12]
other_norm_samples = [s for s in seq_info if s["sub"] not in [4, 5] and not s["is_d1_defect"]][:12]

categories = [
    ("피험자4_5_결함", p45_def_samples, target_dirs["p45_def"]),
    ("피험자4_5_정상", p45_norm_samples, target_dirs["p45_norm"]),
    ("다른피험자_결함", other_def_samples, target_dirs["other_def"]),
    ("다른피험자_정상", other_norm_samples, target_dirs["other_norm"])
]

print("=" * 80)
print("  [D:\\Data\\temp 시각 검증 폴더 복사 시작]")
print("=" * 80)

for cat_name, samples, dest_base in categories:
    print(f"\n[*] {cat_name} ({len(samples)}개 시퀀스 복사 중)...")
    for s in samples:
        folder_label = f"{s['base']}_[{s['desc']}]" if s["desc"] else s["base"]
        # 특수문자 정리
        clean_name = folder_label.replace("/", "_").replace("\\", "_")
        dest_folder = os.path.join(dest_base, clean_name)
        if not os.path.exists(dest_folder):
            shutil.copytree(s["img_dir"], dest_folder)
            print(f"  + 복사 완료: {clean_name}")

print("\n" + "=" * 80)
print("  [+] D:\\Data\\temp 4대 검증 폴더 구축 완료!")
print("=" * 80)

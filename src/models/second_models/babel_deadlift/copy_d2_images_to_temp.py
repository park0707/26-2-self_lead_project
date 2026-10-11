# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: copy_d2_images_to_temp.py
Purpose: 바벨 데드리프트 d2 (무릎-발 방향 일치) 정상 및 결함 영상에 대해 
         2개 핵심 시점(시점 C 정면, 시점 B 우측전방 45도)의 전체 이미지 프레임(01~16)을
         D:\Data\temp\babel_deadlift_d2_images 에 복사 배치하고 가이드 문서를 작성한다.
"""

import os
import sys
import shutil
import glob

sys.stdout.reconfigure(encoding='utf-8')

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TARGET_DIR = r"D:\Data\temp\babel_deadlift_d2_images"

# 대상 샘플 정의 (정상 2개 세트, 결함 2개 세트)
# 비교를 극대화하기 위해 피험자 1번과 피험자 6번의 정상(Key281) vs 결함(Key283: 무릎 안쪽 모임) 선택
SAMPLES = [
    {
        "category": "01_정상_데이터_Normal",
        "sub_folder": "정상_피험자1_Key281",
        "raw_folder": "[정상]_D03_피험자1_Key281",
        "subject": "피험자 1",
        "label": "정상 (0, 발과 무릎의 방향 일치)",
        "scenario": "Key281 (전 부위 정상 수행)"
    },
    {
        "category": "01_정상_데이터_Normal",
        "sub_folder": "정상_피험자6_Key281",
        "raw_folder": "[정상]_D03_피험자6_Key281",
        "subject": "피험자 6",
        "label": "정상 (0, 발과 무릎의 방향 일치)",
        "scenario": "Key281 (전 부위 정상 수행)"
    },
    {
        "category": "02_오류_데이터_Defect",
        "sub_folder": "오류_피험자1_Key283_무릎모임결함",
        "raw_folder": "[오류]_D03_피험자1_Key283",
        "subject": "피험자 1",
        "label": "결함 (1, 발과 무릎의 방향 불일치 / 무릎 외반)",
        "scenario": "Key283 (무릎 안쪽 모임 결함)"
    },
    {
        "category": "02_오류_데이터_Defect",
        "sub_folder": "오류_피험자6_Key283_무릎모임결함",
        "raw_folder": "[오류]_D03_피험자6_Key283",
        "subject": "피험자 6",
        "label": "결함 (1, 발과 무릎의 방향 불일치 / 무릎 외반)",
        "scenario": "Key283 (무릎 안쪽 모임 결함)"
    },
]

# 2대 핵심 시점
TARGET_VIEWS = [
    {"view_code": "시점C", "name": "시점C_정면_0도", "desc": "무릎 모임(외반)이 가장 정밀하게 관찰되는 최적 시점 (F1 65.45%, Recall 100%)"},
    {"view_code": "시점B", "name": "시점B_우측전방_45도", "desc": "상체 각도와 무릎 깊이를 입체적으로 함께 관찰할 수 있는 사각 시점 (Recall 94.4%)"},
]

os.makedirs(TARGET_DIR, exist_ok=True)

manifest_records = []

print(f"[*] 원천 이미지 복사 시작: {RAW_IMG_DIR} -> {TARGET_DIR}")

for s in SAMPLES:
    raw_path = os.path.join(RAW_IMG_DIR, s["raw_folder"])
    if not os.path.exists(raw_path):
        print(f"[!] 원천 폴더 없음: {raw_path}")
        continue

    sample_base_dir = os.path.join(TARGET_DIR, s["category"], s["sub_folder"])
    os.makedirs(sample_base_dir, exist_ok=True)

    for v in TARGET_VIEWS:
        v_code = v["view_code"]
        v_name = v["name"]
        view_dir = os.path.join(sample_base_dir, v_name)
        os.makedirs(view_dir, exist_ok=True)

        # 해당 시점의 프레임 검색 (프레임01 ~ 프레임16)
        all_files = os.listdir(raw_path)
        img_files = sorted([os.path.join(raw_path, f) for f in all_files if f.endswith(".jpg") and f"_{v_code}_" in f])

        copied_count = 0
        for f in img_files:
            fname = os.path.basename(f)
            dest = os.path.join(view_dir, fname)
            shutil.copy2(f, dest)
            copied_count += 1

        print(f"  [+] 복사 완료: {s['sub_folder']} / {v_name} ({copied_count} 프레임)")

        manifest_records.append({
            "category": s["category"],
            "sample": s["sub_folder"],
            "subject": s["subject"],
            "label": s["label"],
            "scenario": s["scenario"],
            "view": v_name,
            "desc": v["desc"],
            "frame_count": copied_count,
            "rel_path": os.path.join(s["category"], s["sub_folder"], v_name).replace("\\", "/")
        })

# README_이미지_가이드.md 생성
readme_path = os.path.join(TARGET_DIR, "README_이미지_가이드.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# 바벨 데드리프트 d2 (무릎-발 방향 일치) 정상 및 결함 전체 이미지 프레임 가이드\n\n")
    f.write("- **프로젝트**: `vidio-health-trainer`\n")
    f.write("- **채택 모델**: `d2 v3.0` (Cleanlab + 3D 가상정면 12D 기하 모델)\n")
    f.write("- **3-State 공식 확정 윈도우**: `[0.200, 0.250]` (보류율 9.9% <= 10.0% 준수, F1 53.98%, 정면 Recall 100.0%)\n")
    f.write(f"- **저장 위치**: `{TARGET_DIR}`\n\n")
    f.write("## 1. 개요 및 시점 선정 사유\n\n")
    f.write("바벨 데드리프트의 d2 결함 항목은 **'발과 무릎의 방향 일치 여부(무릎 안쪽 모임 / Knee Valgus)'**입니다.\n")
    f.write("육안으로 결함과 정상을 가장 뚜렷하게 식별할 수 있도록 아래 **2개 핵심 시점**의 1회 반복 전체 프레임(프레임 01~16)을 전수 배치하였습니다:\n\n")
    f.write("1. **시점 C (정면 0° / Front View)**: 무릎이 양 발끝 방향을 향해 벌어지는지(정상), 아니면 안쪽으로 꺾이며 모이는지(결함)를 정면에서 100% 명확히 판별할 수 있는 핵심 시점입니다.\n")
    f.write("2. **시점 B (우측 전방 45° / Right Front 45°)**: 무릎의 깊이 궤적과 상체 숙임 각도를 입체적으로 동시에 관찰할 수 있는 사각 시점입니다.\n\n")
    f.write("또한, **피험자 1번과 피험자 6번** 각각에 대해 **정상(Key281)과 결함(Key283)**을 1:1로 매칭하여 동일 인물 내에서의 무릎 각도 차이를 직관적으로 비교할 수 있습니다.\n\n")
    f.write("## 2. 배치된 이미지 샘플 목록 (전체 프레임)\n\n")
    f.write("| 구분 | 샘플 폴더 | 피험자 | 카메라 시점 | 시나리오 Key | 정답 라벨 | 전체 프레임 수 |\n")
    f.write("| :---: | :--- | :---: | :---: | :---: | :--- | :---: |\n")
    for m in manifest_records:
        f.write(f"| {m['category'].split('_')[1]} | `{m['rel_path']}` | {m['subject']} | {m['view']} | {m['scenario']} | {m['label']} | **{m['frame_count']}장** |\n")
    f.write("\n## 3. 데드리프트 프레임 구간별 관전 포인트 (16프레임)\n\n")
    f.write("- **프레임 01 ~ 04 (준비 및 셋업)**: 바벨을 잡고 힌지(Hinge) 자세를 잡는 구간. 발끝 너비와 무릎 각도 확인.\n")
    f.write("- **프레임 05 ~ 09 (리프팅 / 퍼스트 풀 구간, ★결함 발생 핵심)**: 지면을 밀며 바벨이 정강이를 지나 무릎을 통과하는 순간. **결함(Key283) 영상에서는 이 구간에서 무릎이 안쪽으로 심하게 모이며 화살표 모양(Valgus)으로 꺾이는 현상**이 뚜렷하게 관찰됩니다.\n")
    f.write("- **프레임 10 ~ 13 (락아웃 및 직립)**: 골반과 무릎을 완전히 펴며 직립하는 구간.\n")
    f.write("- **프레임 14 ~ 16 (하강 및 마무리)**: 바벨을 통제하며 바닥으로 원위치시키는 구간.\n\n")
    f.write("## 4. 확인 방법\n\n")
    f.write("윈도우 파일 탐색기에서 위 폴더를 열고 `프레임01.jpg`부터 방향키 `->`를 눌러 넘겨보시면, 정상 동작과 결함 동작의 무릎 궤적이 1초 만에 뚜렷하게 대조됩니다.\n")

print(f"[+] 전체 완료! 가이드 작성됨: {readme_path}")

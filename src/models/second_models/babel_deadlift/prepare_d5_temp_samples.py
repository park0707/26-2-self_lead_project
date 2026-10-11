# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: prepare_d5_temp_samples.py
Purpose: 사용자가 직접 눈으로 확인할 수 있도록 D:\Data\temp 디렉터리에
         바벨 데드리프트 d5(바벨 궤적과 몸 밀착)의 대표 오류 데이터와 정상 데이터를 
         피험자 1:1 대조 방식으로 복사 및 시각적 안내문 생성
"""

import os
import sys
import shutil
import json

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
TARGET_DIR = r"D:\Data\temp\02_바벨_데드리프트_d5_샘플_비교"

SAMPLES = [
    # (유형, 피험자라벨, 폴더명)
    ("DEFECT", "피험자1_결함", "[오류]_D03_피험자1_Key284"),
    ("DEFECT", "피험자2_결함", "[오류]_D03_피험자2_Key284"),
    ("DEFECT", "피험자5_결함", "[오류]_D03_피험자5_Key284"),
    ("CLEAN",  "피험자1_정상", "[정상]_D03_피험자1_Key281"),
    ("CLEAN",  "피험자2_정상", "[정상]_D03_피험자2_Key281"),
    ("CLEAN",  "피험자5_정상", "[정상]_D03_피험자5_Key281"),
]

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d5 바벨 밀착 오류/정상 샘플 복사 파이프라인")
    print("=" * 80)

    defect_dir = os.path.join(TARGET_DIR, "01_결함_샘플_바벨이탈")
    clean_dir = os.path.join(TARGET_DIR, "02_정상_샘플_바벨밀착")
    os.makedirs(defect_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    for stype, sub_label, folder_name in SAMPLES:
        out_sub_dir = os.path.join(defect_dir if stype == "DEFECT" else clean_dir, f"{sub_label}_{folder_name}")
        os.makedirs(out_sub_dir, exist_ok=True)

        src_folder = os.path.join(RAW_IMG_DIR, folder_name)
        if not os.path.isdir(src_folder):
            print(f"[!] 원본 폴더 없음: {src_folder}")
            continue

        # 1. 이미지 파일 복사
        img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img in img_files:
            shutil.copy2(os.path.join(src_folder, img), os.path.join(out_sub_dir, img))

        # 2. 라벨 JSON 파일 복사
        src_json = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if os.path.exists(src_json):
            shutil.copy2(src_json, os.path.join(out_sub_dir, f"{folder_name}.json"))

        print(f"[+] 복사 완료: {sub_label} -> {len(img_files)}개 이미지 복사됨 ({out_sub_dir})")

    # 3. 안내문(README.txt) 작성
    readme_path = os.path.join(TARGET_DIR, "README_바벨_데드리프트_d5_샘플_관전포인트.txt")
    readme_content = """================================================================================
vidio-health-trainer - 바벨 데드리프트 d5 (바벨 궤적과 몸 밀착) 샘플 비교 가이드
================================================================================

본 폴더는 바벨 데드리프트 d5 결함('바벨 궤적과 몸 밀착' 위반)의 동작 차이를
사용자께서 눈으로 직접 대조하여 확인하실 수 있도록 동일 피험자의 정상 vs 오류 데이터를 
1:1로 매칭하여 준비한 폴더입니다.

[폴더 위치]
D:\\Data\\temp\\02_바벨_데드리프트_d5_샘플_비교\\
├── 01_결함_샘플_바벨이탈\\   (바벨이 몸에서 멀어지는 오류 동작)
│   ├── 피험자1_결함_[오류]_D03_피험자1_Key284\\
│   ├── 피험자2_결함_[오류]_D03_피험자2_Key284\\
│   └── 피험자5_결함_[오류]_D03_피험자5_Key284\\
│
└── 02_정상_샘플_바벨밀착\\   (바벨이 몸에 밀착되어 수직 이동하는 정상 동작)
    ├── 피험자1_정상_[정상]_D03_피험자1_Key281\\
    ├── 피험자2_정상_[정상]_D03_피험자2_Key281\\
    └── 피험자5_정상_[정상]_D03_피험자5_Key281\\

--------------------------------------------------------------------------------
■ 직접 확인하실 때 가장 잘 보이는 '골든 시점' 추천
--------------------------------------------------------------------------------
데드리프트에서 바벨이 몸에서 전방으로 멀어지는 궤적은 카메라 각도에 따라 차이가 큽니다:

1. [가장 추천] 시점 D (좌측 전방 45도) & 시점 B (우측 전방 45도)
   - 파일명 예시: *_시점D_*.jpg  또는  *_시점B_*.jpg
   - 관전 포인트:
     바벨이 바닥에서 떨어져 무릎 높이를 지날 때(프레임 05~10번 구간):
     * [정상]: 바벨 봉이 정강이와 무릎을 거의 스치듯이 바짝 붙어서 수직으로 올라갑니다.
     * [오류]: 바벨 봉이 무릎 앞쪽으로 주먹 1~2개 이상 붕 떠서(전방 이탈) 앞으로 둥글게 돌아나옵니다.

2. 시점 A (우측 후방 45도) & 시점 E (좌측 후방 45도)
   - 파일명 예시: *_시점A_*.jpg  또는  *_시점E_*.jpg
   - 관전 포인트:
     어깨선과 팔의 각도를 보시면, 결함 동작에서는 팔이 수직보다 앞으로 뻗어 나가면서
     몸통(광배근)과의 결합이 풀려 있는 것을 보실 수 있습니다.

3. 시점 C (정면 0도)
   - 파일명 예시: *_시점C_*.jpg
   - 관전 포인트:
     정면에서는 바벨이 신체 전면을 가리기 때문에 전후 거리가 잘 보이지 않습니다.
     이것이 바로 앞선 보고서에서 규명된 '정면 뷰 시상면 깊이 축소(Depth Flattening)' 현상입니다.

--------------------------------------------------------------------------------
■ 1:1 대조 추천 세트
--------------------------------------------------------------------------------
1세트: 피험자 1
- 정상: 02_정상_샘플_바벨밀착\\피험자1_정상_[정상]_D03_피험자1_Key281\\..._시점D_08.jpg
- 오류: 01_결함_샘플_바벨이탈\\피험자1_결함_[오류]_D03_피험자1_Key284\\..._시점D_08.jpg
👉 두 사진을 나란히 열어보시면 바벨과 무릎 사이의 간격 차이가 확연히 드러납니다!
================================================================================
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[+] README 가이드 생성 완료: {readme_path}")
    print(f"\n[+] 모든 샘플이 준비되었습니다: {TARGET_DIR}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: prepare_d4_inspection_frames.py
Purpose: 피험자 1번의 완벽한 1:1 대조 샘플:
         1) [정상] D03_피험자1_Key281 (전체 5대 시점 전체 80프레임)
         2) [오류] D03_피험자1_Key285 (무릎 먼저 펴고 / 전체 5대 시점 전체 80프레임)
         를 D:\Data\temp\데드리프트_d4_비교검증 폴더에 공식 5대 시점별로 깔끔히 분류 복사
"""

import os
import sys
import shutil

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
DEST_BASE = r"D:\Data\temp\데드리프트_d4_비교검증"

os.makedirs(DEST_BASE, exist_ok=True)

# 시점 매핑 (시점 접두사 -> 공식 시점 명칭)
VIEW_NAME_MAP = {
    "시점A": "시점A_우측후방45도(view5)",
    "시점B": "시점B_우측전방45도(view2)",
    "시점C": "시점C_정면0도(view3)",
    "시점D": "시점D_좌측전방45도(view4)",
    "시점E": "시점E_좌측후방45도(view1)"
}

TARGET_PAIRS = [
    {
        "type": "NORMAL",
        "folder_name": "[정상]_D03_피험자1_Key281",
        "desc": "완전 정상 자세 (무릎과 골반 동시 신전)",
        "dest_sub": "01_정상_피험자1_Key281"
    },
    {
        "type": "DEFECT",
        "folder_name": "[오류]_D03_피험자1_Key285",
        "desc": "d4 오류 (무릎 먼저 펴고 / 엉덩이 조기 솟구침)",
        "dest_sub": "02_오류_무릎먼저펴짐_피험자1_Key285"
    }
]

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d4 비교 검증용 프레임 이미지 복사 시작")
    print(f"  - 대상 저장소: {DEST_BASE}")
    print("=" * 80)

    for item in TARGET_PAIRS:
        src_dir = os.path.join(RAW_IMG_DIR, item["folder_name"])
        dst_dir = os.path.join(DEST_BASE, item["dest_sub"])
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"[!] 원본 폴더 없음: {src_dir}")
            continue

        # 안내 텍스트 파일
        readme_path = os.path.join(dst_dir, "00_동작_안내.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"폴더명: {item['folder_name']}\n")
            f.write(f"구분: {item['type']}\n")
            f.write(f"동작 설명: {item['desc']}\n\n")
            f.write("■ 시점 안내 (Official 5-View Taxonomy):\n")
            f.write(" - 시점A: 우측 후방 (45도) [view5]\n")
            f.write(" - 시점B: 우측 전방 (45도) [view2]\n")
            f.write(" - 시점C: 정면 (0도) [view3]\n")
            f.write(" - 시점D: 좌측 전방 (45도) [view4]\n")
            f.write(" - 시점E: 좌측 후방 (45도) [view1]\n\n")
            f.write("※ 특히 시점B(우측전방) 및 시점D(좌측전방)를 보시면,\n")
            f.write("   오류 클립에서 바벨을 뽑을 때 무릎이 먼저 다 펴지면서 엉덩이가 위로 솟구쳐\n")
            f.write("   상체가 수평으로 꺾이는 '굿모닝/스트리퍼 데드리프트' 현상을 뚜렷이 확인하실 수 있습니다.\n")

        files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts = {}

        for fn in files:
            src_file = os.path.join(src_dir, fn)
            matched_view = None
            for vk in VIEW_NAME_MAP.keys():
                if vk in fn or f"_{vk[-1]}_" in fn:
                    matched_view = vk
                    break

            if matched_view:
                sub_v = VIEW_NAME_MAP[matched_view]
                v_dir = os.path.join(dst_dir, sub_v)
                os.makedirs(v_dir, exist_ok=True)
                dst_file = os.path.join(v_dir, fn)
                shutil.copy2(src_file, dst_file)
                counts[matched_view] = counts.get(matched_view, 0) + 1

        print(f"\n[+] {item['dest_sub']} 복사 완료 (총 {len(files)}장):")
        for vk in sorted(counts.keys()):
            print(f"    - {VIEW_NAME_MAP[vk]}: {counts[vk]}프레임")

    print("\n" + "=" * 80)
    print(f"  [성공] 모든 시점 프레임이 {DEST_BASE} 에 배치되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()

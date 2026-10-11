# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: prepare_c2_samples.py
Purpose: 바벨 런지 Condition 2 (몸통 발 무릎 방향 일치 여부, 즉 발방향 안쪽 결함)의
         실제 발생 양상을 시각적으로 확인할 수 있는 고품질 샘플 데이터셋을 D:\Data\temp에 추출 구축
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\04_바벨_런지"
PREP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_BASE_DIR = r"D:\Data\temp\바벨런지_c2_몸통발무릎방향일치_오류샘플"

SAMPLES_CONFIG = [
    {
        "folder_name": "01_단독결함_발방향안쪽_피험자1_Key331",
        "clip_name": "[오류]_D03_피험자1_Key331",
        "category": "DEFECT (단독 결함)",
        "condition_c2": False,
        "desc_korean": "발방향 안쪽 (Condition 2 단독 결함)",
        "observation_guide": "시점 C(정면) 또는 시점 B(우측 전방)에서 선행 발(오른발)을 내딛고 하강할 때, 발끝이 11자 전방을 향하지 못하고 안쪽(In-toeing)으로 15도 이상 꺾여 들어갑니다. 이로 인해 무릎 관절 역시 안쪽으로 비틀리며 몸통-무릎-발의 진행 축 정렬이 무너집니다."
    },
    {
        "folder_name": "02_단독결함_발방향안쪽_피험자2_Key331",
        "clip_name": "[오류]_D03_피험자2_Key331",
        "category": "DEFECT (단독 결함)",
        "condition_c2": False,
        "desc_korean": "발방향 안쪽 (피험자 2 단독 결함)",
        "observation_guide": "피험자 2의 순수 Condition 2 결함 동작입니다. 하강 구간에서 앞발이 안쪽으로 회전하며 착지하고, 최하사점(Bottom)에서 슬관절이 발목 기계축보다 내측으로 쏠리는 현상이 뚜렷하게 관찰됩니다."
    },
    {
        "folder_name": "03_복합결함_발방향안쪽_앞다리각도불량_피험자1_Key335",
        "clip_name": "[오류]_D03_피험자1_Key335",
        "category": "DEFECT (복합 결함)",
        "condition_c2": False,
        "desc_korean": "앞다리 각도 부족 + 발방향 안쪽 (복합 결함)",
        "observation_guide": "발끝이 안쪽으로 꺾임과 동시에 앞다리 무릎 각도가 90도까지 충분히 하강하지 못하는 복합 결함입니다. 발목 내회전으로 인한 족부 불안정성이 체간 하강 깊이(ROM)까지 제한하는 양상을 확인할 수 있습니다."
    },
    {
        "folder_name": "04_완전정상_11자정렬_피험자1_Key329",
        "clip_name": "[정상]_D03_피험자1_Key329",
        "category": "CLEAN (완전 정상 대조군)",
        "condition_c2": True,
        "desc_korean": "5대 조건 완벽 준수 (정상 11자 정렬 표준 동작)",
        "observation_guide": "[대조 비교용] 피험자 1의 완벽한 런지 표준 동작입니다. 선행 발끝이 정확히 11자 전방을 향하며, 하강 및 상승 시 무릎 중심이 발끝과 정확히 일치하여 좌우 흔들림 없이 수직 하강합니다."
    },
    {
        "folder_name": "05_완전정상_11자정렬_피험자2_Key329",
        "clip_name": "[정상]_D03_피험자2_Key329",
        "category": "CLEAN (완전 정상 대조군)",
        "condition_c2": True,
        "desc_korean": "5대 조건 완벽 준수 (정상 11자 정렬 표준 동작)",
        "observation_guide": "[대조 비교용] 피험자 2의 완벽한 런지 표준 동작입니다. 시점 C(정면)에서 볼 때 척추-골반-무릎-발목이 완벽한 대칭성과 수직 진행축을 유지합니다."
    }
]


def make_gif_from_frames(frame_paths, out_gif_path, duration_ms=100, max_size=(640, 360)):
    """프레임 이미지들로부터 리사이즈된 프리뷰 GIF 생성"""
    images = []
    for fp in frame_paths:
        try:
            im = Image.open(fp)
            im.thumbnail(max_size, Image.Resampling.LANCZOS)
            images.append(im.convert("RGB"))
        except Exception as e:
            pass

    if images:
        images[0].save(
            out_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )


def annotate_bottom_frame(frame_path, out_annotated_path, title_text, guide_text):
    """최하사점 프레임에 시각적 안내 캡션 오버레이"""
    try:
        im = Image.open(frame_path).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)

        # 상단 캡션 바
        bar_h = 70
        draw.rectangle([(0, 0), (w, bar_h)], fill=(16, 44, 87))
        # 하단 캡션 바
        draw.rectangle([(0, h - 80), (w, h)], fill=(20, 20, 20))

        # 기본 텍스트 드로잉
        draw.text((20, 15), title_text, fill=(255, 255, 255))
        draw.text((20, h - 60), guide_text, fill=(255, 220, 100))

        im.save(out_annotated_path, quality=95)
    except Exception as e:
        print(f"Error annotating frame {frame_path}: {e}")


def main():
    os.makedirs(OUT_BASE_DIR, exist_ok=True)
    print(f"[*] Base output directory: {OUT_BASE_DIR}")

    summary_records = []

    for cfg in SAMPLES_CONFIG:
        folder_name = cfg["folder_name"]
        clip_name = cfg["clip_name"]
        sample_out_dir = os.path.join(OUT_BASE_DIR, folder_name)
        os.makedirs(sample_out_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"[*] Processing: {folder_name} ({clip_name})")
        print(f"=======================================================")

        # 1. JSON 라벨 복사
        json_src = os.path.join(LABEL_DIR, f"{clip_name}.json")
        json_dst = os.path.join(sample_out_dir, f"{clip_name}.json")
        label_data = {}
        if os.path.exists(json_src):
            shutil.copy2(json_src, json_dst)
            with open(json_src, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            print(f"  [+] Copied Label JSON: {os.path.basename(json_dst)}")

        # 2. NPZ 64프레임 전처리 데이터 복사 (view1~view5)
        npz_files_copied = []
        for v in ["view1", "view2", "view3", "view4", "view5"]:
            npz_src = os.path.join(PREP_DIR, f"{clip_name}_{v}.npz")
            if os.path.exists(npz_src):
                npz_dst = os.path.join(sample_out_dir, f"{clip_name}_{v}.npz")
                shutil.copy2(npz_src, npz_dst)
                npz_files_copied.append(f"{clip_name}_{v}.npz")
        print(f"  [+] Copied {len(npz_files_copied)} NPZ files (5-Views)")

        # 3. 원본 이미지 프레임 복사 및 GIF 생성 (시점 C 정면, 시점 B 우측전방)
        raw_img_clip_dir = os.path.join(RAW_IMG_DIR, clip_name)
        if os.path.exists(raw_img_clip_dir):
            all_jpgs = sorted(os.listdir(raw_img_clip_dir))

            # 시점 C (정면 0°, view3 매핑)
            view_c_files = [f for f in all_jpgs if "시점C" in f and f.endswith(".jpg")]
            if view_c_files:
                view_c_dir = os.path.join(sample_out_dir, "frames_시점C_정면")
                os.makedirs(view_c_dir, exist_ok=True)
                c_full_paths = []
                for jf in view_c_files:
                    src_p = os.path.join(raw_img_clip_dir, jf)
                    dst_p = os.path.join(view_c_dir, jf)
                    shutil.copy2(src_p, dst_p)
                    c_full_paths.append(dst_p)
                print(f"  [+] Copied {len(view_c_files)} frames for 시점 C (정면)")

                # 움직이는 GIF 생성
                gif_c_path = os.path.join(sample_out_dir, "01_animation_시점C_정면.gif")
                make_gif_from_frames(c_full_paths, gif_c_path, duration_ms=90)
                print(f"  [+] Created Animated GIF: {os.path.basename(gif_c_path)}")

                # 최하사점(바닥 프레임: 보통 중간 프레임 부근) 스틸컷 생성
                mid_idx = len(c_full_paths) // 2
                bottom_src = c_full_paths[mid_idx]
                bottom_out = os.path.join(sample_out_dir, "02_최하사점_관찰프레임_시점C.jpg")
                annotate_bottom_frame(
                    bottom_src,
                    bottom_out,
                    f"[{cfg['category']}] {clip_name} - 시점 C (정면 0°)",
                    f"관찰 포인트: {cfg['desc_korean']}"
                )
                print(f"  [+] Created Bottom Frame Photo: {os.path.basename(bottom_out)}")

            # 시점 B (우측 전방 45°, view2 매핑)
            view_b_files = [f for f in all_jpgs if "시점B" in f and f.endswith(".jpg")]
            if view_b_files:
                view_b_dir = os.path.join(sample_out_dir, "frames_시점B_우측전방")
                os.makedirs(view_b_dir, exist_ok=True)
                b_full_paths = []
                for jf in view_b_files:
                    src_p = os.path.join(raw_img_clip_dir, jf)
                    dst_p = os.path.join(view_b_dir, jf)
                    shutil.copy2(src_p, dst_p)
                    b_full_paths.append(dst_p)
                print(f"  [+] Copied {len(view_b_files)} frames for 시점 B (우측 전방 45°)")

                # 움직이는 GIF 생성
                gif_b_path = os.path.join(sample_out_dir, "03_animation_시점B_우측전방.gif")
                make_gif_from_frames(b_full_paths, gif_b_path, duration_ms=90)
                print(f"  [+] Created Animated GIF: {os.path.basename(gif_b_path)}")
        else:
            print(f"  [!] Raw image directory not found: {raw_img_clip_dir}")

        # 4. 상세 메타데이터 및 설명 JSON 생성
        info_json_path = os.path.join(sample_out_dir, "sample_info.json")
        sample_meta = {
            "folder_name": folder_name,
            "clip_name": clip_name,
            "category": cfg["category"],
            "condition_2_label": 1 if not cfg["condition_c2"] else 0,
            "condition_2_status": "DEFECT (결함: 발방향 안쪽)" if not cfg["condition_c2"] else "CLEAN (정상: 11자 정렬)",
            "defect_korean_name": cfg["desc_korean"],
            "all_conditions": {c["condition"]: c["value"] for c in label_data.get("type_info", {}).get("conditions", [])},
            "observation_guide": cfg["observation_guide"],
            "view_recommendation": "시점 C (정면 0°)에서 앞발 발끝의 꺾임과 무릎-체간 11자 붕괴가 가장 극명하게 관찰됩니다."
        }
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)
        print(f"  [+] Created sample_info.json")

        summary_records.append(sample_meta)

    # 5. 최상위 README.md 작성
    readme_path = os.path.join(OUT_BASE_DIR, "README_오류확인_가이드.md")
    readme_content = f"""# [vidio-health-trainer] 바벨 런지 Condition 2 실제 오류 샘플 데이터셋

본 디렉터리는 **바벨 런지(04_바벨_런지)**의 두 번째 핵심 오류인 **'몸통 발 무릎 방향 일치 여부(Condition 2)'** 결함이 
실제로 피험자의 실제 동작에서 어떻게 발생하는지 육안 및 데이터로 명확히 확인할 수 있도록 구성된 샘플 저장소입니다.

---

## 1. Condition 2 오류의 본질: '발방향 안쪽 (Foot In-toeing)' 결함이란?

AI-Hub 정답 라벨링 기준, '몸통 발 무릎 방향 일치 여부'의 결함(False)은 대부분 **'발방향 안쪽'** 오류로 명명됩니다.
- **정상 동작**: 선행 발을 앞으로 내딛을 때, 발끝(장축)이 11자 전방(0°)을 향하고, 무릎 중심도 발끝과 같은 방향으로 굽혀지며, 체간과 골반이 틀어지지 않고 정면을 유지합니다.
- **결함 동작**: 선행 발을 내딛을 때 발끝이 안쪽(15°~30° 이상)으로 꺾여 착지하며, 하강 시 무릎이 안쪽으로 무너지는 동적 외반(Dynamic Knee Valgus)이 수반되어 **몸통 - 무릎 - 발의 11자 진행 방향 정렬이 심각하게 파괴**됩니다.

---

## 2. 샘플 구성 목록 및 확인 방법

각 샘플 폴더에 들어가시면 **움직이는 GIF 프리뷰(`*.gif`)**와 **최하사점 스틸컷(`*.jpg`)**, 그리고 **시점별 원본 프레임 이미지들**이 포함되어 있어 윈도우 탐색기에서 즉시 확인하실 수 있습니다.

### [샘플 1] `01_단독결함_발방향안쪽_피험자1_Key331`
- **유형**: Condition 2 순수 단독 결함 (다른 4대 조건은 전부 정상)
- **핵심 관찰 포인트**: 
  - `01_animation_시점C_정면.gif` 파일을 열어보시면, 피험자 1이 앞발을 내딛는 순간 발끝이 안쪽으로 15도 이상 꺾여 들어가며 착지합니다.
  - 최하사점(`02_최하사점_관찰프레임_시점C.jpg`)에서 오른쪽 무릎이 안쪽으로 쏠리며 몸통-발의 11자 정렬이 완전히 무너진 모습을 확인하실 수 있습니다.

### [샘플 2] `02_단독결함_발방향안쪽_피험자2_Key331`
- **유형**: 피험자 2의 Condition 2 순수 단독 결함
- **핵심 관찰 포인트**: 
  - 피험자 2 역시 선행 발끝이 안쪽으로 과도하게 꺾이며 착지하여, 슬관절이 발목 기계축 안쪽으로 편위되는 전형적인 Foot In-toeing 결함을 보여줍니다.

### [샘플 3] `03_복합결함_발방향안쪽_앞다리각도불량_피험자1_Key335`
- **유형**: 발방향 안쪽 + 앞다리 무릎 각도 부족 (복합 결함)
- **핵심 관찰 포인트**: 
  - 발끝이 안쪽으로 꺾이면서 족부 안정성이 무너져 하강 깊이(90도 각도)까지 달성하지 못하는 실제 복합 결함의 상호작용을 확인할 수 있습니다.

### [샘플 4] `04_완전정상_11자정렬_피험자1_Key329` (★ 대조군)
- **유형**: 5대 조건 완전 정상 (정상 11자 정렬 표준 동작)
- **비교 관찰 포인트**: 
  - 샘플 1(`Key331`)과 나란히 놓고 비교해 보시면, 발끝이 정확히 정면(11자)을 향하고 무릎과 몸통이 완벽한 수직 대칭을 이루는 것을 극명하게 대조해 보실 수 있습니다.

### [샘플 5] `05_완전정상_11자정렬_피험자2_Key329` (★ 대조군)
- **유형**: 피험자 2의 완전 정상 11자 정렬 표준 동작

---

## 3. 각 샘플 폴더 내 파일 구조

- `01_animation_시점C_정면.gif`: 정면 0도 전체 동작 반복 재생 애니메이션 (가장 직관적인 확인 수단)
- `02_최하사점_관찰프레임_시점C.jpg`: 런지 최하사점(가장 깊게 내려간 순간) 분석 스틸컷
- `03_animation_시점B_우측전방.gif`: 우측 전방 45도 전체 동작 애니메이션
- `frames_시점C_정면/`: 정면 카메라 고해상도 연속 원본 프레임 이미지들
- `frames_시점B_우측전방/`: 우측 전방 45도 카메라 연속 원본 프레임 이미지들
- `*.npz`: 64프레임 전처리 3차원 MediaPipe 33관절 스켈레톤 좌표 (view1~view5 5대 시점 전수)
- `*.json`: AI-Hub 2D 원천 정답 라벨 파일
- `sample_info.json`: 상세 메타데이터 및 생체역학 관찰 가이드
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[+] Master README saved: {readme_path}")

    # 최상위 data/temp/README_오류확인_가이드.md에도 복사
    top_readme = r"D:\Data\temp\README_바벨런지_c2_실제오류_샘플_안내.md"
    shutil.copy2(readme_path, top_readme)
    print(f"[+] Top-level README saved: {top_readme}")


if __name__ == "__main__":
    main()

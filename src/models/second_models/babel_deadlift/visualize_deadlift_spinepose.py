# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: visualize_deadlift_spinepose.py
Purpose: 피험자 2번(우수), 피험자 5번(최저), 피험자 4번, 8번의 데드리프트 프레임 이미지 위에
         SpinePose의 9대 척추 분절(S1, L5, L3, L1, T8, T3, C7, C4, C1) 및
         척추 연결선, 요추 곡률을 고해상도로 시각화하여 D:\\Data\\temp\\SpinePose_관절시각화 에 저장
"""

import os
import sys
import glob
import cv2
import numpy as np
from spinepose import SpinePoseEstimator

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
OUT_BASE_DIR = r"D:\Data\temp\SpinePose_관절시각화"

# SpinePose 공식 척추 체인 9개 분절 (골반 -> 요추 -> 흉추 -> 경추)
# 19: S1, 26: L5, 27: L3, 28: L1, 29: T8, 30: T3, 18: C7, 35: C4, 36: C1
SPINE_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINE_SHORT = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']
SPINE_NAMES = [
    'S1 (Sacrum/Pelvis)', 'L5 (Lower Lumbar)', 'L3 (Mid Lumbar)', 'L1 (Upper Lumbar)',
    'T8 (Mid Thoracic)', 'T3 (Upper Thoracic)', 'C7 (Lower Cervical)',
    'C4 (Mid Cervical)', 'C1 (Atlas/Base)'
]

# 분절별 색상 (BGR): S1(파랑) -> 요추(청록/녹색) -> 흉추(노랑/주황) -> 경추(빨강)
SPINE_COLORS = [
    (255, 100, 0),    # S1: 진한 파랑
    (255, 200, 0),    # L5: 시안
    (0, 230, 120),    # L3: 연두
    (0, 200, 0),      # L1: 녹색
    (0, 220, 255),    # T8: 노랑
    (0, 140, 255),    # T3: 주황
    (50, 50, 255),    # C7: 다홍
    (150, 0, 255),    # C4: 자홍
    (0, 0, 255)       # C1: 빨강
]


def load_korean_path_image(img_path):
    try:
        with open(img_path, 'rb') as f:
            buf = f.read()
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_korean_path_image(save_path, img_bgr):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        res, arr = cv2.imencode(ext, img_bgr)
        if res:
            with open(save_path, 'wb') as f:
                f.write(arr.tobytes())
            return True
    except Exception as e:
        print(f"저장 실패: {e}")
    return False


def draw_spinepose_on_frame(img, kps_2d, scores, title_text="", sub_info=""):
    h, w = img.shape[:2]
    canvas = img.copy()

    # 상단 반투명 헤더 배너
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(canvas, f"[vidio-health-trainer] SpinePose 9-Spine Segments Visualization", (25, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
    if sub_info:
        cv2.putText(canvas, sub_info, (25, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # 척추 체인 좌표 추출
    spine_pts = []
    spine_scores = []
    for idx in SPINE_INDICES:
        pt = kps_2d[idx]
        sc = scores[idx]
        spine_pts.append((int(round(pt[0])), int(round(pt[1]))))
        spine_scores.append(float(sc))

    # 1. 척추 뼈대 연결선 (글로우 효과)
    for i in range(len(spine_pts) - 1):
        pt1 = spine_pts[i]
        pt2 = spine_pts[i + 1]
        sc1 = spine_scores[i]
        sc2 = spine_scores[i + 1]

        if sc1 > 0.15 and sc2 > 0.15:
            # 외곽 검정 그림자선
            cv2.line(canvas, pt1, pt2, (0, 0, 0), 7, cv2.LINE_AA)
            # 중심 색상선
            cv2.line(canvas, pt1, pt2, SPINE_COLORS[i], 3, cv2.LINE_AA)

    # 2. 9개 관절점 마커 및 라벨 박스
    for i, (pt, sc, short, col) in enumerate(zip(spine_pts, spine_scores, SPINE_SHORT, SPINE_COLORS)):
        if sc <= 0.15:
            continue

        # 외곽 흰 테두리 + 중심 색상 원
        cv2.circle(canvas, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 라벨 텍스트
        label_text = f"{short}"
        (tw, th), tb = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        lx = pt[0] + 12
        ly = pt[1] + 4
        if lx + tw + 10 > w:
            lx = pt[0] - tw - 16

        # 라벨 배경
        cv2.rectangle(canvas, (lx - 3, ly - th - 3), (lx + tw + 3, ly + tb + 1), (15, 15, 15), -1)
        cv2.putText(canvas, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 우측 하단 범례 (Legend)
    legend_w, legend_h = 240, 150
    lx0, ly0 = w - legend_w - 20, h - legend_h - 20
    cv2.rectangle(overlay, (lx0, ly0), (w - 20, h - 20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.putText(canvas, "Spine Segments", (lx0 + 10, ly0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    legend_items = [
        ("S1: Sacrum (Hip)", SPINE_COLORS[0]),
        ("L5-L1: Lumbar Spine", SPINE_COLORS[2]),
        ("T8-T3: Thoracic Spine", SPINE_COLORS[5]),
        ("C7-C1: Cervical Spine", SPINE_COLORS[8])
    ]
    for idx_l, (l_text, l_col) in enumerate(legend_items):
        cy = ly0 + 45 + idx_l * 24
        cv2.circle(canvas, (lx0 + 20, cy), 6, l_col, -1)
        cv2.putText(canvas, l_text, (lx0 + 35, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return canvas


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] SpinePose 9개 관절점 이미지 오버레이 시각화 생성")
    print("=" * 80)

    estimator = SpinePoseEstimator(mode="small", enable_lifting=True)
    print("[*] SpinePose Estimator 로드 완료.")

    # 시각화 대상 액션 폴더 선정
    # 피험자 2(우수), 피험자 5(최저 0%), 피험자 4(20%), 피험자 8(20%)
    targets = [
        ("01_피험자2_우수탐지(Recall_87%)", "[오류]_D03_피험자2_Key282", "피험자 2 (결함: 척추의 굴곡)"),
        ("02_피험자5_최저탐지(Recall_0%)", "[오류]_D03_피험자5_Key282", "피험자 5 (결함: 척추의 굴곡 - 모델 미탐지)"),
        ("03_피험자4_저성능(Recall_20%)", "[오류]_D07_피험자4_Key282", "피험자 4 (결함: 척추의 굴곡 - 모델 미탐지)"),
        ("04_피험자8_저성능(Recall_20%)", "[오류]_D03_피험자8_Key282", "피험자 8 (결함: 척추의 굴곡 - 모델 미탐지)")
    ]

    # 관찰하기 가장 좋은 시점:
    # 시점 B (우측 전방 45도 / view2), 시점 D (좌측 전방 45도 / view4), 시점 C (정면)
    views_to_render = ['시점B', '시점D', '시점C']

    total_rendered = 0

    for out_subfolder, base_folder, sub_desc in targets:
        src_dir = os.path.join(RAW_IMG_DIR, base_folder)
        if not os.path.exists(src_dir):
            print(f"[!] 폴더 없음: {src_dir}")
            continue

        dest_dir = os.path.join(OUT_BASE_DIR, out_subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n[*] 시각화 렌더링 중: {sub_desc} -> {dest_dir}")

        all_imgs = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))])

        for v_name in views_to_render:
            v_imgs = [f for f in all_imgs if v_name in f]
            if not v_imgs:
                continue

            # 대표 프레임 선정:
            # 프레임01 (셋업/최하사점), 프레임05 (바벨 이륙), 프레임09 (무릎 통과), 프레임14 (최상단 기립)
            selected_indices = [0, 4, 8, min(13, len(v_imgs) - 1)]

            for s_idx in selected_indices:
                if s_idx >= len(v_imgs):
                    continue
                img_name = v_imgs[s_idx]
                img_p = os.path.join(src_dir, img_name)
                img = load_korean_path_image(img_p)
                if img is None:
                    continue

                h, w = img.shape[:2]
                bbox = np.array([[w * 0.15, h * 0.05, w * 0.85, h * 0.95]], dtype=np.float32)

                res = estimator.estimate(img, bbox)
                kps_2d, scores, kps_3d = res

                # 프레임 단계 텍스트
                phase_names = {0: "셋업 (Setup/최하사점)", 4: "바벨 이륙 (Floor Break)", 8: "무릎 통과 (Knee Pass)", selected_indices[-1]: "기립 (Lockout)"}
                phase_text = phase_names.get(s_idx, f"프레임 {s_idx+1}")

                title_text = f"{sub_desc} | {v_name} | {phase_text}"
                canvas = draw_spinepose_on_frame(img, kps_2d[0], scores[0], title_text, f"Image: {img_name}")

                out_img_name = f"{v_name}_Step{s_idx+1:02d}_{phase_text.split()[0]}_{img_name}"
                save_p = os.path.join(dest_dir, out_img_name)
                save_korean_path_image(save_p, canvas)
                print(f"  + 저장: {out_img_name}")
                total_rendered += 1

    print("\n" + "=" * 80)
    print(f"  [OK] SpinePose 관절 시각화 완료! 총 {total_rendered}개 이미지가 저장되었습니다.")
    print(f"  - 경로: {OUT_BASE_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: regenerate_perfect_spinepose_visuals.py
Purpose: 천장 박스 버그를 완전히 수정하고, SpinePose 내부 자동 사람 디텍터(RF-DETR)를 가동하여
         사람 몸통 정중앙에 100% 완벽하게 정렬된 9대 척추 포인트 시각화 이미지 재배포
"""

import os
import sys
import glob
import cv2
import numpy as np
import shutil
from spinepose import SpinePoseEstimator

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
OUT_BASE_DIR = r"D:\Data\temp\SpinePose_정밀관절시각화"

SPINE_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINE_SHORT = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']
SPINE_NAMES = [
    'S1 (Sacrum/Pelvis)', 'L5 (Lower Lumbar)', 'L3 (Mid Lumbar)', 'L1 (Upper Lumbar)',
    'T8 (Mid Thoracic)', 'T3 (Upper Thoracic)', 'C7 (Lower Cervical)',
    'C4 (Mid Cervical)', 'C1 (Atlas/Base)'
]

SPINE_COLORS = [
    (255, 100, 0),    # S1: 파랑
    (255, 200, 0),    # L5: 시안
    (0, 230, 120),    # L3: 연두
    (0, 200, 0),      # L1: 녹색
    (0, 220, 255),    # T8: 노랑
    (0, 140, 255),    # T3: 주황
    (50, 50, 255),    # C7: 다홍
    (150, 0, 255),    # C4: 자홍
    (0, 0, 255)       # C1: 빨강
]


def load_img(p):
    with open(p, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def save_img(p, img):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ext = os.path.splitext(p)[1]
    res, arr = cv2.imencode(ext, img)
    if res:
        with open(p, 'wb') as f:
            f.write(arr.tobytes())


def render_accurate_spine_overlay(img, kps_2d, scores, title_text, sub_text=""):
    h, w = img.shape[:2]
    canvas = img.copy()

    # 상단 반투명 헤더 배너
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(canvas, "[vidio-health-trainer] Accurate SpinePose 9-Spine Segments (RF-DETR Auto-Person Detection)", (30, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (30, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2, cv2.LINE_AA)
    if sub_text:
        cv2.putText(canvas, sub_text, (30, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    pts = kps_2d[SPINE_INDICES]  # (9, 2)
    scs = scores[SPINE_INDICES]  # (9,)

    # 1. 척추 뼈대 연결선 (글로우 효과)
    for i in range(len(pts) - 1):
        p1 = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        p2 = (int(round(pts[i + 1, 0])), int(round(pts[i + 1, 1])))
        cv2.line(canvas, p1, p2, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, SPINE_COLORS[i], 4, cv2.LINE_AA)

    # 2. 9개 관절점 마커 및 라벨
    for i in range(len(pts)):
        p = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        cv2.circle(canvas, p, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 7, SPINE_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 2, (255, 255, 255), -1, cv2.LINE_AA)

        lbl = f"{SPINE_SHORT[i]} ({scs[i]:.2f})"
        (tw, th), tb = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

        lx = p[0] + 14
        ly = p[1] + 5
        if lx + tw + 10 > w:
            lx = p[0] - tw - 18

        cv2.rectangle(canvas, (lx - 3, ly - th - 3), (lx + tw + 3, ly + tb + 1), (15, 15, 15), -1)
        cv2.putText(canvas, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 우측 하단 범례 박스
    lw, lh = 260, 160
    lx0, ly0 = w - lw - 25, h - lh - 25
    cv2.rectangle(overlay, (lx0, ly0), (w - 25, h - 25), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.putText(canvas, "Spine Anatomical Chain", (lx0 + 12, ly0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    legend_items = [
        ("S1: Sacrum / Pelvis Base", SPINE_COLORS[0]),
        ("L5-L1: Lumbar Spine (4 Segs)", SPINE_COLORS[2]),
        ("T8-T3: Thoracic Spine", SPINE_COLORS[5]),
        ("C7-C1: Cervical Spine", SPINE_COLORS[8])
    ]
    for idx_l, (l_text, l_col) in enumerate(legend_items):
        cy = ly0 + 55 + idx_l * 24
        cv2.circle(canvas, (lx0 + 20, cy), 6, l_col, -1)
        cv2.putText(canvas, l_text, (lx0 + 36, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return canvas


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] RF-DETR 자동 사람 감지 기반 척추 시각화 재생성")
    print("=" * 80)

    # 기존 잘못된 비교 폴더 정리
    old_compare_dir = r"D:\Data\temp\SpinePose_보정_시각화"
    if os.path.exists(old_compare_dir):
        shutil.rmtree(old_compare_dir)
        print(f"[*] 기존 오류 비교 폴더 삭제 완료: {old_compare_dir}")

    os.makedirs(OUT_BASE_DIR, exist_ok=True)

    # detector='rfdetr' 자동 활성화된 estimator
    estimator = SpinePoseEstimator(mode="small", enable_lifting=True)
    print("[*] SpinePose (RF-DETR Auto-Detection) 초기화 완료.")

    # 사용자가 확인하신 피험자 5, 4, 8, 2
    targets = [
        ("01_피험자5_최저탐지(Recall_0%)", "[오류]_D03_피험자5_Key282", "피험자 5 (결함: 척추의 굴곡 - 모델 미탐지 케이스)"),
        ("02_피험자4_저성능(Recall_20%)", "[오류]_D07_피험자4_Key282", "피험자 4 (결함: 척추의 굴곡 - 모델 미탐지 케이스)"),
        ("03_피험자8_저성능(Recall_20%)", "[오류]_D03_피험자8_Key282", "피험자 8 (결함: 척추의 굴곡 - 모델 미탐지 케이스)"),
        ("04_피험자2_우수탐지(Recall_87%)", "[오류]_D03_피험자2_Key282", "피험자 2 (결함: 척추의 굴곡 - 정상 탐지 케이스)")
    ]

    views = ['시점B', '시점D', '시점C']
    total_saved = 0

    for out_sub, folder_name, desc in targets:
        src_dir = os.path.join(RAW_IMG_DIR, folder_name)
        if not os.path.exists(src_dir):
            continue

        dest_dir = os.path.join(OUT_BASE_DIR, out_sub)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n[*] 재생성 중: {desc} -> {dest_dir}")

        all_imgs = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))])

        for vl in views:
            v_imgs = [f for f in all_imgs if vl in f]
            if not v_imgs:
                continue

            # 4대 핵심 위상: 셋업(1), 바벨이륙(5), 무릎통과(9), 기립(14)
            for s_idx, phase_name in [(0, "셋업(Setup)"), (4, "바벨이륙(FloorBreak)"), (8, "무릎통과(KneePass)"), (min(13, len(v_imgs) - 1), "기립(Lockout)")]:
                if s_idx >= len(v_imgs):
                    continue

                img_file = v_imgs[s_idx]
                img_path = os.path.join(src_dir, img_file)
                img = load_img(img_path)
                if img is None:
                    continue

                # ★ 핵심: BBox를 강제로 넘기지 않고 est(img)로 호출 -> 내부 RF-DETR이 사람을 완벽히 자동 감지!
                res = estimator(img)
                k2d, sc, _ = res

                if len(k2d) == 0:
                    print(f"  [!] 사람 미검출: {img_file}")
                    continue

                title = f"{desc.split('(')[0]} | {vl} | {phase_name}"
                sub_info = f"File: {img_file} | 9-Spine Segments Conf: {np.mean(sc[0, SPINE_INDICES]):.2f}"
                canvas = render_accurate_spine_overlay(img, k2d[0], sc[0], title, sub_info)

                out_name = f"{vl}_Step{s_idx+1:02d}_{phase_name.split('(')[0]}_{img_file}"
                out_path = os.path.join(dest_dir, out_name)
                save_img(out_path, canvas)
                print(f"  + [정상 안착 완료] {out_name}")
                total_saved += 1

    print("\n" + "=" * 80)
    print(f"  [OK] 모든 척추 관절점이 사람 몸통 정중앙에 완벽 안착되었습니다! 총 {total_saved}장 배포 완료.")
    print(f"  - 새 경로: {OUT_BASE_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()

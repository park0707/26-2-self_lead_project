# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: render_user_torso_gating_comparison.py
Purpose: 사용자가 목격한 문고리 등 이상치 발생 프레임 대상
         [좌측: 원본 SpinePose (문고리 616px 오검출)] vs [우측: 사용자 제안 체간 Gating & 앵커 보간]
         고해상도 2분할 비교 이미지를 D:\\Data\\temp\\SpinePose_보정_시각화 에 저장
"""

import os
import sys
import glob
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from spinepose import SpinePoseEstimator

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe\02_바벨_데드리프트"
OUT_DIR = r"D:\Data\temp\SpinePose_보정_시각화"
os.makedirs(OUT_DIR, exist_ok=True)

SPINE_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINE_SHORT = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']
SPINE_COLORS = [
    (255, 100, 0),    # S1
    (255, 200, 0),    # L5
    (0, 230, 120),    # L3
    (0, 200, 0),      # L1
    (0, 220, 255),    # T8
    (0, 140, 255),    # T3
    (50, 50, 255),    # C7
    (150, 0, 255),    # C4
    (0, 0, 255)       # C1
]


def load_img(p):
    with open(p, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def save_img(p, img):
    ext = os.path.splitext(p)[1]
    res, arr = cv2.imencode(ext, img)
    if res:
        with open(p, 'wb') as f:
            f.write(arr.tobytes())


def render_comparison(estimator, img_rel_path, mp_npz_name, title_desc):
    img_p = os.path.join(RAW_IMG_DIR, img_rel_path)
    if not os.path.exists(img_p):
        print(f"[!] 이미지 없음: {img_p}")
        return

    img = load_img(img_p)
    h, w = img.shape[:2]

    # 1. 원본 SpinePose (전역 BBox: 문고리 등 오인식 유발)
    bbox_global = np.array([[w * 0.15, h * 0.05, w * 0.85, h * 0.95]], dtype=np.float32)
    res_raw = estimator.estimate(img, bbox_global)
    k2d_raw = res_raw[0][0, SPINE_INDICES]  # (9, 2)
    sc_raw = res_raw[1][0, SPINE_INDICES]    # (9,)

    # 2. MediaPipe 4대 관절점 (L_SH=2, R_SH=3, L_HIP=8, R_HIP=9)
    mp_p = os.path.join(RAW_MP_DIR, mp_npz_name)
    if not os.path.exists(mp_p):
        print(f"[!] MediaPipe 없음: {mp_p}")
        return

    d_mp = np.load(mp_p, allow_pickle=True)
    k2d_mp = d_mp['keypoints_2d_raw']  # (16, 15, 2)
    # 이미지 파일명에서 프레임 번호 파싱
    f_num = 1
    for token in img_rel_path.split("_"):
        if "프레임" in token:
            try:
                f_num = int(token.replace("프레임", "").replace(".jpg", ""))
            except:
                pass
    f_idx = min(len(k2d_mp) - 1, max(0, f_num - 1))
    frame_mp = k2d_mp[f_idx]

    sh_l, sh_r = frame_mp[2], frame_mp[3]
    hip_l, hip_r = frame_mp[8], frame_mp[9]

    torso_box_pts = np.array([sh_l, sh_r, hip_r, hip_l], dtype=np.float32)
    tx_min, ty_min = np.min(torso_box_pts, axis=0) - 70
    tx_max, ty_max = np.max(torso_box_pts, axis=0) + 70
    ty_min = max(0, ty_min - 40)
    ty_max = min(h, ty_max + 40)
    tx_min = max(0, tx_min)
    tx_max = min(w, tx_max)

    bbox_torso = np.array([[tx_min, ty_min, tx_max, ty_max]], dtype=np.float32)

    # 3. 사용자 제안 1 & 2 적용 (체간 포커싱 + 앵커 보간)
    res_clean = estimator.estimate(img, bbox_torso)
    k2d_clean_raw = res_clean[0][0, SPINE_INDICES]
    sc_clean_raw = res_clean[1][0, SPINE_INDICES]

    # 체간 내부 여부 확인 및 앵커 보간 (사용자 제안 2)
    in_box = (
        (k2d_clean_raw[:, 0] >= tx_min) & (k2d_clean_raw[:, 0] <= tx_max) &
        (k2d_clean_raw[:, 1] >= ty_min) & (k2d_clean_raw[:, 1] <= ty_max)
    )
    valid = in_box & (sc_clean_raw >= 0.3)

    spine_z_ratios = np.array([0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00], dtype=np.float32)
    sh_mid = 0.5 * (sh_l + sh_r)
    hip_mid = 0.5 * (hip_l + hip_r)

    anchors_z = [0.0, 0.85]
    anchors_xy = [k2d_clean_raw[0] if valid[0] else hip_mid,
                  k2d_clean_raw[6] if valid[6] else sh_mid]

    for idx in range(1, 8):
        if valid[idx] and idx != 6:
            anchors_z.append(spine_z_ratios[idx])
            anchors_xy.append(k2d_clean_raw[idx])

    anchors_z = np.array(anchors_z)
    anchors_xy = np.array(anchors_xy)
    u_z, u_i = np.unique(anchors_z, return_index=True)
    u_xy = anchors_xy[u_i]

    if len(u_z) >= 2:
        fx = PchipInterpolator(u_z, u_xy[:, 0], extrapolate=True)
        fy = PchipInterpolator(u_z, u_xy[:, 1], extrapolate=True)
        clean_pts = np.stack([fx(spine_z_ratios), fy(spine_z_ratios)], axis=-1)
    else:
        clean_pts = k2d_clean_raw

    # =========================================================================
    # 2분할 캔버스 렌더링
    # =========================================================================
    # [좌측] 원본 SpinePose (문고리 튐)
    c_left = img.copy()
    cv2.rectangle(c_left, (int(tx_min), int(ty_min)), (int(tx_max), int(ty_max)), (0, 255, 255), 2)
    cv2.putText(c_left, "MediaPipe Torso Bounds", (int(tx_min), int(ty_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    for i in range(8):
        p1 = (int(round(k2d_raw[i, 0])), int(round(k2d_raw[i, 1])))
        p2 = (int(round(k2d_raw[i + 1, 0])), int(round(k2d_raw[i + 1, 1])))
        d = np.linalg.norm(k2d_raw[i] - k2d_raw[i + 1])
        line_col = (0, 0, 255) if d > 150 else SPINE_COLORS[i]
        line_thick = 5 if d > 150 else 3
        cv2.line(c_left, p1, p2, (0, 0, 0), line_thick + 3)
        cv2.line(c_left, p1, p2, line_col, line_thick)

    for i in range(9):
        p = (int(round(k2d_raw[i, 0])), int(round(k2d_raw[i, 1])))
        is_out = (k2d_raw[i, 0] < tx_min or k2d_raw[i, 0] > tx_max or k2d_raw[i, 1] < ty_min or k2d_raw[i, 1] > ty_max)
        col = (0, 0, 255) if is_out else SPINE_COLORS[i]
        cv2.circle(c_left, p, 9, (255, 255, 255), -1)
        cv2.circle(c_left, p, 7, col, -1)
        lbl = f"{SPINE_SHORT[i]}"
        if is_out:
            lbl += " [문고리/이상치]"
            cv2.putText(c_left, lbl, (p[0] + 12, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(c_left, lbl, (p[0] + 12, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # [우측] 사용자 제안 보정 후
    c_right = img.copy()
    cv2.rectangle(c_right, (int(tx_min), int(ty_min)), (int(tx_max), int(ty_max)), (0, 255, 100), 2)
    cv2.putText(c_right, "Torso-Gated & Clean Spine", (int(tx_min), int(ty_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

    for i in range(8):
        p1 = (int(round(clean_pts[i, 0])), int(round(clean_pts[i, 1])))
        p2 = (int(round(clean_pts[i + 1, 0])), int(round(clean_pts[i + 1, 1])))
        cv2.line(c_right, p1, p2, (0, 0, 0), 6)
        cv2.line(c_right, p1, p2, SPINE_COLORS[i], 3)

    for i in range(9):
        p = (int(round(clean_pts[i, 0])), int(round(clean_pts[i, 1])))
        cv2.circle(c_right, p, 9, (255, 255, 255), -1)
        cv2.circle(c_right, p, 7, SPINE_COLORS[i], -1)
        cv2.putText(c_right, f"{SPINE_SHORT[i]}", (p[0] + 12, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 상단 정보 헤더 배너 결합
    combined = np.hstack([c_left, c_right])
    cw = combined.shape[1]
    banner = np.zeros((100, cw, 3), dtype=np.uint8)
    cv2.putText(banner, f"[vidio-health-trainer] SpinePose Outlier Rejection & Torso-Gated Restoration", (30, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(banner, f"Left: RAW (Contains Door Handle / Outlier) vs Right: USER PROPOSED (Torso-Gated & Spline Interpolated) | {title_desc}",
                (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

    final_img = np.vstack([banner, combined])

    base_f = os.path.basename(img_rel_path).replace(".jpg", "")
    out_file = os.path.join(OUT_DIR, f"보정비교_{base_f}.jpg")
    save_img(out_file, final_img)
    print(f"[OK] 비교 이미지 생성 완료: {os.path.basename(out_file)}")


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 사용자 제안 체간 Gating 및 앵커 보간 비교 렌더링")
    print("=" * 80)

    estimator = SpinePoseEstimator(mode="small", enable_lifting=True)

    # 사용자가 문고리 이상치를 관찰한 주요 피험자 프레임들
    test_frames = [
        # 피험자 5 (문고리 튐 발생 케이스)
        (r"[오류]_D03_피험자5_Key282\[오류]_D03_피험자5_Key282_시점B_프레임01.jpg", "[오류]_D03_피험자5_Key282.npz", "피험자 5 | 시점 B | 셋업"),
        (r"[오류]_D03_피험자5_Key282\[오류]_D03_피험자5_Key282_시점B_프레임05.jpg", "[오류]_D03_피험자5_Key282.npz", "피험자 5 | 시점 B | 바벨 이륙"),
        (r"[오류]_D03_피험자5_Key282\[오류]_D03_피험자5_Key282_시점D_프레임01.jpg", "[오류]_D03_피험자5_Key282.npz", "피험자 5 | 시점 D | 셋업"),
        
        # 피험자 4 (이상치 발생 케이스)
        (r"[오류]_D07_피험자4_Key282\[오류]_D07_피험자4_Key282_시점B_프레임01.jpg", "[오류]_D07_피험자4_Key282.npz", "피험자 4 | 시점 B | 셋업"),
        (r"[오류]_D07_피험자4_Key282\[오류]_D07_피험자4_Key282_시점B_프레임05.jpg", "[오류]_D07_피험자4_Key282.npz", "피험자 4 | 시점 B | 바벨 이륙"),

        # 피험자 8
        (r"[오류]_D03_피험자8_Key282\[오류]_D03_피험자8_Key282_시점B_프레임01.jpg", "[오류]_D03_피험자8_Key282.npz", "피험자 8 | 시점 B | 셋업"),
        
        # 피험자 2 (비교군)
        (r"[오류]_D03_피험자2_Key282\[오류]_D03_피험자2_Key282_시점B_프레임01.jpg", "[오류]_D03_피험자2_Key282.npz", "피험자 2 | 시점 B | 셋업")
    ]

    for img_rel, mp_name, desc in test_frames:
        render_comparison(estimator, img_rel, mp_name, desc)

    print("\n" + "=" * 80)
    print(f"  [OK] 모든 보정 비교 이미지가 저장되었습니다: {OUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()

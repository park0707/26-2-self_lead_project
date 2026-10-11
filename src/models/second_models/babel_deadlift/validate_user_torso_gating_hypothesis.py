# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: validate_user_torso_gating_hypothesis.py
Purpose: 사용자 제안 2대 해결책 정밀 검증
         1) MediaPipe 4대 몸체 관절(좌우 어깨, 좌우 골반) 기반 체간 영역 이탈 필터링 (Torso Gating & Null 처리)
         2) 고신뢰도 핵심 앵커 포인트 선별 및 척추 스플라인 보간 (Key Anchor Interpolation)
         3) 문고리 등 엉뚱한 배경으로 튄 프레임 전후 비교 시각화 생성
"""

import os
import sys
import glob
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from spinepose import SpinePoseEstimator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
OUT_COMPARE_DIR = r"D:\Data\temp\SpinePose_보정_시각화"
os.makedirs(OUT_COMPARE_DIR, exist_ok=True)

SPINE_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINE_SHORT = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']

# 분절별 색상 (BGR): S1(파랑) -> 요추(청록/녹색) -> 흉추(노랑/주황) -> 경추(빨강)
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

MP_LSH, MP_RSH = 11, 12
MP_LHIP, MP_RHIP = 23, 24


def load_korean_image(p):
    with open(p, 'rb') as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def save_korean_image(p, img):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ext = os.path.splitext(p)[1]
    res, arr = cv2.imencode(ext, img)
    if res:
        with open(p, 'wb') as f:
            f.write(arr.tobytes())


def get_torso_polygon_and_bbox(kps_33_frame, img_w, img_h, margin_ratio=0.25):
    """
    MediaPipe 4대 관절점(LSH, RSH, LHIP, RHIP)으로 체간 바운딩 박스 및 영역 산출
    kps_33_frame: (33, 3) 0~1 정규화 또는 픽셀 좌표
    """
    sh_l = kps_33_frame[MP_LSH, :2]
    sh_r = kps_33_frame[MP_RSH, :2]
    hip_l = kps_33_frame[MP_LHIP, :2]
    hip_r = kps_33_frame[MP_RHIP, :2]

    pts = np.array([sh_l, sh_r, hip_r, hip_l], dtype=np.float32)

    # 픽셀 좌표화 (정규화된 경우)
    if np.max(pts) <= 1.5:
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    bw = x_max - x_min
    bh = y_max - y_min

    # 마진 적용
    x_min_m = max(0.0, x_min - bw * margin_ratio)
    x_max_m = min(float(img_w), x_max + bw * margin_ratio)
    y_min_m = max(0.0, y_min - bh * (margin_ratio + 0.1))  # 목/머리 방향 여유
    y_max_m = min(float(img_h), y_max + bh * margin_ratio)

    return (x_min_m, y_min_m, x_max_m, y_max_m), pts


def filter_and_interpolate_spine(kps_2d_raw, scores_raw, torso_bbox, kps_33_frame, img_w, img_h):
    """
    [해결책 1 & 2 통합 구현]
    1. Torso Bounding Box 바깥으로 나간 점은 Outlier로 마킹 (Null 처리)
    2. 신뢰도(Confidence > 0.35)가 높고 체간 내부에 있는 앵커 포인트들만 선택
    3. 선택된 앵커 + MediaPipe 어깨/골반 중심점을 기준으로 9개 분절 3차 PCHIP 보간 복원
    """
    x_min, y_min, x_max, y_max = torso_bbox

    raw_spine_pts = kps_2d_raw[SPINE_INDICES]  # (9, 2)
    raw_spine_sc = scores_raw[SPINE_INDICES]    # (9,)

    # 1. 체간 내부 여부 판정 (In-Torso Mask)
    in_torso = (
        (raw_spine_pts[:, 0] >= x_min) & (raw_spine_pts[:, 0] <= x_max) &
        (raw_spine_pts[:, 1] >= y_min) & (raw_spine_pts[:, 1] <= y_max)
    )
    valid_mask = in_torso & (raw_spine_sc >= 0.25)

    # MediaPipe 어깨 중심 및 골반 중심 픽셀 좌표
    sh_mid = 0.5 * (kps_33_frame[MP_LSH, :2] + kps_33_frame[MP_RSH, :2])
    hip_mid = 0.5 * (kps_33_frame[MP_LHIP, :2] + kps_33_frame[MP_RHIP, :2])
    if np.max(sh_mid) <= 1.5:
        sh_mid[0] *= img_w
        sh_mid[1] *= img_h
        hip_mid[0] *= img_w
        hip_mid[1] *= img_h

    # 척추 체인 높이 비율 (S1=0.0 -> L5=0.12 -> L3=0.25 -> L1=0.42 -> T8=0.60 -> T3=0.75 -> C7=0.85 -> C4=0.92 -> C1=1.00)
    spine_z_ratios = np.array([0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00], dtype=np.float32)

    # 유효한 앵커 포인트 수집
    anchor_z = []
    anchor_xy = []

    # 기저 앵커 (S1: 골반 중심 보완)
    if valid_mask[0]:  # S1 유효
        anchor_z.append(0.0)
        anchor_xy.append(raw_spine_pts[0])
    else:
        anchor_z.append(0.0)
        anchor_xy.append(hip_mid)

    # 중간 분절들 (L5~C4) 중 유효한 점들만 추가
    for idx in range(1, 8):
        if valid_mask[idx]:
            anchor_z.append(spine_z_ratios[idx])
            anchor_xy.append(raw_spine_pts[idx])

    # 상단 앵커 (C7/C1: 목/어깨 중심 보완)
    if valid_mask[6]:  # C7 유효
        anchor_z.append(0.85)
        anchor_xy.append(raw_spine_pts[6])
    elif valid_mask[8]:  # C1 유효
        anchor_z.append(1.00)
        anchor_xy.append(raw_spine_pts[8])
    else:
        anchor_z.append(0.85)
        anchor_xy.append(sh_mid)

    anchor_z = np.array(anchor_z)
    anchor_xy = np.array(anchor_xy)

    # 중복 z 제거 및 정렬
    unique_z, u_idx = np.unique(anchor_z, return_index=True)
    unique_xy = anchor_xy[u_idx]

    # 최소 2개 이상의 앵커로 PCHIP 보간
    if len(unique_z) >= 2:
        fx = PchipInterpolator(unique_z, unique_xy[:, 0], extrapolate=True)
        fy = PchipInterpolator(unique_z, unique_xy[:, 1], extrapolate=True)
        restored_spine = np.stack([fx(spine_z_ratios), fy(spine_z_ratios)], axis=-1)
    else:
        # 비상 대체 (선형)
        restored_spine = np.zeros((9, 2), dtype=np.float32)
        for i, r in enumerate(spine_z_ratios):
            restored_spine[i] = hip_mid + r * (sh_mid - hip_mid)

    outlier_count = int(np.sum(~valid_mask))
    return restored_spine, raw_spine_pts, valid_mask, outlier_count, torso_bbox


def draw_side_by_side_comparison(img, raw_pts, clean_pts, valid_mask, torso_bbox, title_text):
    """
    [좌: 원본 SpinePose (문고리 등 이탈)] vs [우: 체간 Gating + 앵커 보간]
    좌우 2분할 비교 이미지 렌더링
    """
    h, w = img.shape[:2]

    # 좌측 캔버스 (Raw SpinePose)
    canvas_left = img.copy()
    # 체간 BBox 표시
    x1, y1, x2, y2 = [int(v) for v in torso_bbox]
    cv2.rectangle(canvas_left, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 노란색 체간 영역
    cv2.putText(canvas_left, "MediaPipe Torso Area", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for i in range(8):
        pt1 = (int(round(raw_pts[i, 0])), int(round(raw_pts[i, 1])))
        pt2 = (int(round(raw_pts[i + 1, 0])), int(round(raw_pts[i + 1, 1])))
        cv2.line(canvas_left, pt1, pt2, (0, 0, 0), 6)
        cv2.line(canvas_left, pt1, pt2, (0, 0, 255) if not valid_mask[i] else SPINE_COLORS[i], 3)

    for i in range(9):
        pt = (int(round(raw_pts[i, 0])), int(round(raw_pts[i, 1])))
        col = (0, 0, 255) if not valid_mask[i] else SPINE_COLORS[i]
        cv2.circle(canvas_left, pt, 8, (255, 255, 255), -1)
        cv2.circle(canvas_left, pt, 6, col, -1)
        lbl = f"{SPINE_SHORT[i]}"
        if not valid_mask[i]:
            lbl += " [OUTLIER/문고리]"
            cv2.putText(canvas_left, lbl, (pt[0] + 10, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(canvas_left, lbl, (pt[0] + 10, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 우측 캔버스 (보정 후)
    canvas_right = img.copy()
    cv2.rectangle(canvas_right, (x1, y1), (x2, y2), (0, 255, 100), 2)
    cv2.putText(canvas_right, "Torso Clamped & Clean", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    for i in range(8):
        pt1 = (int(round(clean_pts[i, 0])), int(round(clean_pts[i, 1])))
        pt2 = (int(round(clean_pts[i + 1, 0])), int(round(clean_pts[i + 1, 1])))
        cv2.line(canvas_right, pt1, pt2, (0, 0, 0), 6)
        cv2.line(canvas_right, pt1, pt2, SPINE_COLORS[i], 3)

    for i in range(9):
        pt = (int(round(clean_pts[i, 0])), int(round(clean_pts[i, 1])))
        cv2.circle(canvas_right, pt, 8, (255, 255, 255), -1)
        cv2.circle(canvas_right, pt, 6, SPINE_COLORS[i], -1)
        cv2.putText(canvas_right, f"{SPINE_SHORT[i]} (Restored)", (pt[0] + 10, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 좌우 결합
    combined = np.hstack([canvas_left, canvas_right])
    cw = combined.shape[1]

    # 상단 정보 배너
    banner = np.zeros((90, cw, 3), dtype=np.uint8)
    cv2.putText(banner, f"[vidio-health-trainer] SpinePose Outlier Rejection & Torso-Gated Restoration", (30, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(banner, f"Left: RAW (Contains Door Handle / Outlier) vs Right: USER PROPOSED (Torso-Gated & Spline Interpolated)", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)

    final = np.vstack([banner, combined])
    return final


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 사용자 제안: 체간 바깥 오류(문고리) 필터링 및 앵커 보간 검증")
    print("=" * 80)

    estimator = SpinePoseEstimator(mode="small", enable_lifting=True)
    print("[*] SpinePose Estimator 준비 완료.")

    # 검증 대상: 피험자 5, 4, 8, 2의 시퀀스들
    test_cases = [
        ("[오류]_D03_피험자5_Key282", "피험자 5 (최저 성능)"),
        ("[오류]_D07_피험자4_Key282", "피험자 4 (저성능)"),
        ("[오류]_D03_피험자8_Key282", "피험자 8 (저성능)"),
        ("[오류]_D03_피험자2_Key282", "피험자 2 (우수 탐지)")
    ]

    total_outliers_detected = 0
    total_frames_checked = 0

    print("\n[*] 전수 체간 이탈(문고리 오인식) 검사 및 보정 시각화 생성 중...")

    for folder_name, sub_desc in test_cases:
        img_dir = os.path.join(RAW_IMG_DIR, folder_name)
        if not os.path.exists(img_dir):
            continue

        all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])

        # MediaPipe npz 로드
        for vn, vl in [('view2', '시점B'), ('view4', '시점D'), ('view3', '시점C')]:
            mp_p = os.path.join(RAW_MP33_DIR, f"{folder_name}_{vn}.npz")
            if not os.path.exists(mp_p):
                continue

            mp_d = np.load(mp_p, allow_pickle=True)
            kps_64f = mp_d['kps_33_3d_64f']

            v_imgs = [f for f in all_imgs if vl in f]
            if not v_imgs:
                continue

            for t_idx in [0, 4, 8, min(13, len(v_imgs) - 1)]:
                if t_idx >= len(v_imgs):
                    continue

                img_name = v_imgs[t_idx]
                img_path = os.path.join(img_dir, img_name)
                img = load_korean_image(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                kps_frame = kps_64f[t_idx]

                # MediaPipe 4대 관절점 기반 체간 BBox 산출
                torso_bbox, torso_pts = get_torso_polygon_and_bbox(kps_frame, w, h, margin_ratio=0.35)

                # 원본 SpinePose 추론 (전역 BBox)
                bbox_global = np.array([[w * 0.15, h * 0.05, w * 0.85, h * 0.95]], dtype=np.float32)
                res = estimator.estimate(img, bbox_global)
                kps_2d_raw, scores_raw, kps_3d_raw = res

                # 보정 적용 (해결책 1 & 2)
                clean_spine_2d, raw_spine_2d, valid_mask, n_outliers, t_bbox = filter_and_interpolate_spine(
                    kps_2d_raw[0], scores_raw[0], torso_bbox, kps_frame, w, h
                )

                total_frames_checked += 1
                total_outliers_detected += n_outliers

                # 이상치가 발견되었거나 대표 프레임인 경우 비교 이미지 저장
                phase_tag = f"Step{t_idx+1:02d}"
                out_name = f"비교_{folder_name}_{vl}_{phase_tag}_이탈{n_outliers}개.jpg"
                out_path = os.path.join(OUT_COMPARE_DIR, out_name)

                cmp_canvas = draw_side_by_side_comparison(
                    img, raw_spine_2d, clean_spine_2d, valid_mask, torso_bbox,
                    f"{sub_desc} | {vl} | {phase_tag}"
                )
                save_korean_image(out_path, cmp_canvas)
                print(f"  + [{vl}-{phase_tag}] {folder_name} | 체간 바깥 이탈 관절: {n_outliers}/9개 -> 저장: {out_name}")

    print("\n" + "=" * 80)
    print("  [사용자 가설 정량 검증 결과]")
    print(f"  - 총 검사 프레임 수: {total_frames_checked}개")
    print(f"  - 체간 바깥으로 튄 SpinePose 이상치 관절 수: {total_outliers_detected}건")
    print(f"  - 프레임당 평균 이상치 비율: {total_outliers_detected / max(1, total_frames_checked * 9) * 100:.1f}%")
    print(f"  - 비교 시각화 저장 경로: {OUT_COMPARE_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()

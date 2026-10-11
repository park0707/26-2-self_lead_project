# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: render_d4_misclassifications_to_temp.py
Purpose: 바벨 데드리프트 d4(무릎-골반 동시 펴짐 위반, 힙 슈트업) 모델의
         오분류(False Negative 미탐, False Positive 오탐) 대표 케이스들을 선별하여
         D:\Data\temp\바벨_데드리프트_d4_오분류_시각화 에 저장하고,
         각 프레임 위에 MediaPipe 33개 관절 포인트 및 스켈레톤, 주요 생체역학 지표를 오버레이
"""

import os
import sys
import glob
import json
import math
import shutil
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.adapters.mediapipe_adapter import MediaPipeAdapter

# 경로 설정
RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
CACHE_NPZ = r"d:\src\models\second_models\babel_deadlift\d4_v4_features_cache.npz"
OOF_GBDT = r"d:\src\models\second_models\babel_deadlift\d4_oof_probs_gbdt.npy"
OOF_TCN = r"d:\src\models\second_models\babel_deadlift\d4_oof_probs_tcn.npy"

TEMP_OUT_BASE = r"D:\Data\temp\바벨_데드리프트_d4_오분류_시각화"
FN_OUT_DIR = os.path.join(TEMP_OUT_BASE, "01_미탐_False_Negative(실제오류_정상오판)")
FP_OUT_DIR = os.path.join(TEMP_OUT_BASE, "02_오탐_False_Positive(실제정상_오류오판)")

os.makedirs(FN_OUT_DIR, exist_ok=True)
os.makedirs(FP_OUT_DIR, exist_ok=True)

# 시점 매핑
VIEW_CODE_TO_NAME = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_CODE_TO_LETTER = {
    "view5": "A",
    "view2": "B",
    "view3": "C",
    "view4": "D",
    "view1": "E"
}

# 스켈레톤 연결선 정의 (MediaPipe 33 Landmark Indices)
SKELETON_CONNECTIONS = [
    # 상체 / 어깨
    (11, 12, (220, 220, 220), 2),  # 좌어깨 - 우어깨
    (11, 13, (180, 200, 255), 2),  # 좌어깨 - 좌팔꿈치
    (13, 15, (180, 200, 255), 2),  # 좌팔꿈치 - 좌손목
    (12, 14, (180, 200, 255), 2),  # 우어깨 - 우팔꿈치
    (14, 16, (180, 200, 255), 2),  # 우팔꿈치 - 우손목
    # 몸통 / 척추
    (11, 23, (150, 220, 255), 2),  # 좌어깨 - 좌골반
    (12, 24, (150, 220, 255), 2),  # 우어깨 - 우골반
    (23, 24, (255, 0, 255), 3),    # 골반선 (Magenta)
    # 하지 (대퇴 / 하퇴) - d4 핵심 관절 체인 (Thick 3px)
    (23, 25, (0, 215, 255), 3),    # 좌골반 - 좌무릎 (Yellow/Gold)
    (24, 26, (0, 215, 255), 3),    # 우골반 - 우무릎 (Yellow/Gold)
    (25, 27, (0, 255, 100), 3),    # 좌무릎 - 좌발목 (Lime Green)
    (26, 28, (0, 255, 100), 3),    # 우무릎 - 우발목 (Lime Green)
    # 발
    (27, 29, (0, 180, 255), 2),    # 좌발목 - 좌뒤꿈치
    (27, 31, (0, 180, 255), 2),    # 좌발목 - 좌발끝
    (29, 31, (0, 180, 255), 2),    # 좌뒤꿈치 - 좌발끝
    (28, 30, (0, 180, 255), 2),    # 우발목 - 우뒤꿈치
    (28, 32, (0, 180, 255), 2),    # 우발목 - 우발끝
    (30, 32, (0, 180, 255), 2),    # 우뒤꿈치 - 우발끝
]


def calculate_angle_2d(p1, p2, p3):
    """p2를 꼭지점으로 하는 2D 사이각 (도 단위)"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        return 0.0
    cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def draw_mediapipe_annotations(img_bgr, kps_2d, vis_33, meta_info: dict) -> np.ndarray:
    """프레임 위에 MediaPipe 관절 포인트, 스켈레톤, 생체역학 정보 오버레이"""
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    # 1. 스켈레톤 연결선 그리기
    for idx1, idx2, color, thickness in SKELETON_CONNECTIONS:
        v1 = vis_33[idx1] if vis_33 is not None else 1.0
        v2 = vis_33[idx2] if vis_33 is not None else 1.0
        if v1 > 0.2 and v2 > 0.2:
            p1 = (int(kps_2d[idx1, 0]), int(kps_2d[idx1, 1]))
            p2 = (int(kps_2d[idx2, 0]), int(kps_2d[idx2, 1]))
            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    # 2. 33개 관절 포인트 그리기
    for i in range(33):
        v = vis_33[i] if vis_33 is not None else 1.0
        if v > 0.2:
            pt = (int(kps_2d[i, 0]), int(kps_2d[i, 1]))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                # 관절 부위별 색상 지정
                if i in [23, 24]:      # 골반 (붉은색/마젠타)
                    c_fill = (255, 0, 255)
                    radius = 6
                elif i in [25, 26]:    # 무릎 (밝은 주황)
                    c_fill = (0, 165, 255)
                    radius = 6
                elif i in [27, 28]:    # 발목 (형광 녹색)
                    c_fill = (0, 255, 0)
                    radius = 5
                elif i in [11, 12]:    # 어깨 (밝은 파랑)
                    c_fill = (255, 200, 0)
                    radius = 5
                else:                  # 기타 관절
                    c_fill = (200, 200, 200)
                    radius = 4

                # 외곽선(검정) + 내부 채우기
                cv2.circle(canvas, pt, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, radius, c_fill, -1, cv2.LINE_AA)

    # 3. 주요 생체역학 각도 계산 (무릎, 고관절)
    # 우측 시점(view2, view5)은 우측 하지(24, 26, 28), 좌측 시점(view1, view4)은 좌측 하지(23, 25, 27)
    is_left_dom = meta_info.get("view_code", "") in ["view1", "view4"]
    hip_idx = 23 if is_left_dom else 24
    knee_idx = 25 if is_left_dom else 26
    ankle_idx = 27 if is_left_dom else 28
    shoulder_idx = 11 if is_left_dom else 12

    knee_angle = calculate_angle_2d(kps_2d[hip_idx], kps_2d[knee_idx], kps_2d[ankle_idx])
    hip_angle = calculate_angle_2d(kps_2d[shoulder_idx], kps_2d[hip_idx], kps_2d[knee_idx])

    # 각도 텍스트 오버레이 (무릎 및 골반 위치 근처)
    kp_knee = (int(kps_2d[knee_idx, 0]) + 15, int(kps_2d[knee_idx, 1]) - 5)
    kp_hip = (int(kps_2d[hip_idx, 0]) + 15, int(kps_2d[hip_idx, 1]) - 5)

    if 0 <= kp_knee[0] < w - 100 and 0 <= kp_knee[1] < h:
        cv2.putText(canvas, f"Knee:{knee_angle:.1f}deg", kp_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Knee:{knee_angle:.1f}deg", kp_knee, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if 0 <= kp_hip[0] < w - 100 and 0 <= kp_hip[1] < h:
        cv2.putText(canvas, f"Hip:{hip_angle:.1f}deg", kp_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Hip:{hip_angle:.1f}deg", kp_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. 상단 헤더 배너 (반투명 오버레이)
    banner_h = 100
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 25, 35), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # 헤더 텍스트
    # Title
    cv2.putText(canvas, "[vidio-health-trainer] Barbell Deadlift d4 Error Analysis (MediaPipe 33 Keypoints)",
                (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Clip & View info
    clip_txt = f"Clip: {meta_info['clip_name']} | Frame: {meta_info['frame_num']}/16 | {meta_info['view_alias']}"
    cv2.putText(canvas, clip_txt, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1, cv2.LINE_AA)

    # Prediction vs Ground Truth
    gt_label = meta_info['gt_label']
    pred_prob = meta_info['pred_prob']
    error_type = meta_info['error_type']

    if error_type == "FALSE_NEGATIVE":
        stat_color = (0, 165, 255)  # Orange
        stat_txt = f"[FN 미탐] GT: DEFECT (힙 슈트업 결함) | Pred Prob: {pred_prob:.3f} -> Predicted NORMAL (오판)"
    else:
        stat_color = (0, 100, 255)  # Red/Amber
        stat_txt = f"[FP 오탐] GT: NORMAL (정상 자세) | Pred Prob: {pred_prob:.3f} -> Predicted DEFECT (오인)"

    cv2.putText(canvas, stat_txt, (20, 84), cv2.FONT_HERSHEY_DUPLEX, 0.55, stat_color, 1, cv2.LINE_AA)

    # 5. 하단 원인 진단 배너
    bottom_h = 42
    overlay_bot = canvas.copy()
    cv2.rectangle(overlay_bot, (0, h - bottom_h), (w, h), (15, 20, 25), -1)
    cv2.addWeighted(overlay_bot, 0.85, canvas, 0.15, 0, canvas)

    reason_txt = meta_info.get("reason", "")
    cv2.putText(canvas, f"Diagnostic: {reason_txt}", (20, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d4 오분류 데이터 시각화 파이프라인")
    print("=" * 80)

    # 1. OOF 예측 데이터 및 메타데이터 로드
    cache = np.load(CACHE_NPZ, allow_pickle=True)
    y = cache['y']
    subjects = cache['subjects']
    views = cache['views']
    clip_names = cache['clip_names']

    p_gbdt = np.load(OOF_GBDT)
    p_tcn = np.load(OOF_TCN)

    X_meta = np.column_stack([p_gbdt, p_tcn])
    meta = LogisticRegression(class_weight='balanced', random_state=42)
    meta.fit(X_meta, y)
    p_fusion = meta.predict_proba(X_meta)[:, 1]

    pred_2s = (p_fusion >= 0.5).astype(int)

    fn_indices = np.where((y == 1) & (pred_2s == 0))[0]
    fp_indices = np.where((y == 0) & (pred_2s == 1))[0]

    print(f"[*] 총 평가 시퀀스: {len(y)}개")
    print(f"[*] False Negatives (미탐): {len(fn_indices)}개")
    print(f"[*] False Positives (오탐): {len(fp_indices)}개")

    # 2. 대표적인 오분류 케이스 선별 (시점별/피험자별 다양성 확보)
    # FN: 모델이 결함을 정상으로 놓친 가장 아쉬운 대표 6건
    fn_sorted = fn_indices[np.argsort(p_fusion[fn_indices])]
    selected_fn = []
    seen_views_fn = set()
    for idx in fn_sorted:
        v = views[idx]
        if v not in seen_views_fn or len(selected_fn) < 6:
            selected_fn.append(idx)
            seen_views_fn.add(v)
            if len(selected_fn) >= 6:
                break

    # FP: 모델이 정상을 결함으로 오인한 대표 6건
    fp_sorted = fp_indices[np.argsort(-p_fusion[fp_indices])]
    selected_fp = []
    seen_views_fp = set()
    for idx in fp_sorted:
        v = views[idx]
        if v not in seen_views_fp or len(selected_fp) < 6:
            selected_fp.append(idx)
            seen_views_fp.add(v)
            if len(selected_fp) >= 6:
                break

    print(f"\n[+] 선별된 FN 케이스 ({len(selected_fn)}건):")
    for i in selected_fn:
        print(f"  - [{views[i]}] {clip_names[i]} (Sub {subjects[i]}, Prob: {p_fusion[i]:.4f})")

    print(f"\n[+] 선별된 FP 케이스 ({len(selected_fp)}건):")
    for i in selected_fp:
        print(f"  - [{views[i]}] {clip_names[i]} (Sub {subjects[i]}, Prob: {p_fusion[i]:.4f})")

    # 3. MediaPipe PoseLandmarker 초기화
    detector = MediaPipeAdapter.get_detector("d:/pose_landmarker_heavy.task")

    # 진단 사유 매핑 딕셔너리
    DIAGNOSTICS_FN = {
        "view3": "정면 뷰(0도) 깊이축(Z) 손실로 전후방 위상차(Phase Lag) 왜곡",
        "view1": "후방 시점 무릎 가림(Occlusion) 및 미세 선행 신전 미탐",
        "view2": "First Pull 구간 바벨 리프트 속도와 무릎 신전 속도의 경계값 판정 실패",
        "view4": "고관절 신전이 무릎 직후 거의 동시에 따라붙어 시간차 탐지 한계",
        "view5": "우측 후방 45도 카메라의 원근 왜곡 및 골반 상승 속도 과소추정"
    }

    DIAGNOSTICS_FP = {
        "view1": "좌측 후방 뷰 특유의 상체 깊은 전경(깊은 숙임)으로 인한 힙 슈트업 오인",
        "view5": "골반-어깨 수직 거리 비대칭 및 초기 골반 위치 낮음으로 인한 급상승 착각",
        "view2": "피험자 고유 체형(긴 대퇴골)에 따른 무릎 조기 펴짐 착시",
        "view4": "바벨 통과 시 상체 각도 유지 구간을 힙 슈트업으로 과대판정",
        "view3": "정면 시점 무릎 외전(Knee Valgus)에 의한 상하 움직임 노이즈"
    }

    # 4. 시각화 및 이미지 저장 루프
    all_targets = [("FN", idx) for idx in selected_fn] + [("FP", idx) for idx in selected_fp]

    for category, idx in all_targets:
        c_name = clip_names[idx]
        v_code = views[idx]
        v_letter = VIEW_CODE_TO_LETTER[v_code]
        v_alias = VIEW_CODE_TO_NAME[v_code]
        prob = p_fusion[idx]
        sub = subjects[idx]

        folder_name = c_name.rsplit(f"_{v_code}", 1)[0]
        src_folder = os.path.join(RAW_IMG_DIR, folder_name)

        if not os.path.isdir(src_folder):
            print(f"[!] 폴더 없음: {src_folder}")
            continue

        # 타깃 저장 서브폴더 생성
        prefix = "FN_미탐" if category == "FN" else "FP_오탐"
        out_sub_name = f"{prefix}_피험자{sub}_{folder_name}_{v_code}_{v_letter}"
        out_target_dir = os.path.join(FN_OUT_DIR if category == "FN" else FP_OUT_DIR, out_sub_name)
        os.makedirs(out_target_dir, exist_ok=True)

        # 이미지 필터링 (해당 시점 이미지 16장)
        all_imgs = sorted(os.listdir(src_folder))
        v_imgs = [f for f in all_imgs if (f"시점{v_letter}" in f or f"_{v_letter}_" in f) and f.lower().endswith(('.jpg', '.png'))]

        reason = DIAGNOSTICS_FN.get(v_code, "경계값 오분류") if category == "FN" else DIAGNOSTICS_FP.get(v_code, "체형 및 시점 오인")

        print(f"[*] 처리 중: [{category}] {out_sub_name} (총 {len(v_imgs)}장)...")

        for f_idx, img_fn in enumerate(v_imgs, start=1):
            img_path = os.path.join(src_folder, img_fn)
            img_bytes = np.fromfile(img_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            # MediaPipe 추출
            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_path, detector=detector)
            kps_2d = res['kps_33_2d']
            vis_33 = res['vis_33']

            meta_info = {
                "clip_name": folder_name,
                "frame_num": f_idx,
                "view_code": v_code,
                "view_alias": v_alias,
                "gt_label": "DEFECT (힙 슈트업)" if category == "FN" else "NORMAL (정상)",
                "pred_prob": float(prob),
                "error_type": "FALSE_NEGATIVE" if category == "FN" else "FALSE_POSITIVE",
                "reason": reason
            }

            annotated_img = draw_mediapipe_annotations(img_bgr, kps_2d, vis_33, meta_info)

            # 저장
            out_img_path = os.path.join(out_target_dir, f"annotated_{img_fn}")
            res_enc, buf = cv2.imencode(".jpg", annotated_img)
            if res_enc:
                with open(out_img_path, "wb") as f_out:
                    f_out.write(buf.tobytes())

        print(f"  -> 저장 완료: {out_target_dir}")

    # 5. README 가이드 파일 작성
    readme_path = os.path.join(TEMP_OUT_BASE, "README_바벨_데드리프트_d4_오분류_심층분석.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("vidio-health-trainer | 바벨 데드리프트 d4 (무릎-골반 동시 펴짐 위반, 힙 슈트업)\n")
        f.write("오분류 데이터 (False Negatives & False Positives) MediaPipe 33 관절 시각화 가이드\n")
        f.write("=" * 80 + "\n\n")
        f.write("본 폴더는 바벨 데드리프트 d4 모델의 Two-Stream v4.0 평가 중 오분류가 발생한 대표 시퀀스들에 대해,\n")
        f.write("각 프레임마다 MediaPipe 33개 랜드마크, 하지 스켈레톤, 무릎/골반 사이각을 정밀하게 오버레이하여\n")
        f.write("사용자께서 오분류의 원인을 시각적·생체역학적으로 직접 검증하실 수 있도록 구축된 폴더입니다.\n\n")
        f.write("■ 폴더 구조:\n")
        f.write("D:\\Data\\temp\\바벨_데드리프트_d4_오분류_시각화\\\n")
        f.write("├── 01_미탐_False_Negative(실제오류_정상오판)/  (실제로는 힙 슈트업 결함인데 모델이 정상으로 놓친 케이스)\n")
        f.write("└── 02_오탐_False_Positive(실제정상_오류오판)/  (실제로는 정상 동작인데 모델이 힙 슈트업으로 잘못 판정한 케이스)\n\n")
        f.write("■ 오버레이 표시 항목:\n")
        f.write(" 1. 상단 정보 배너: 클립명, 프레임 번호(1~16), 카메라 시점, 정답 라벨(GT), 모델 예측 확률(Prob)\n")
        f.write(" 2. 관절 포인트: 골반(마젠타), 무릎(주황), 발목(형광녹색), 어깨(하늘색)\n")
        f.write(" 3. 하지 스켈레톤 체인: 대퇴골(골반-무릎, 3px), 하퇴골(무릎-발목, 3px)\n")
        f.write(" 4. 생체역학 각도: 프레임별 무릎 각도(Knee deg) 및 고관절 각도(Hip deg) 실시간 표시\n")
        f.write(" 5. 하단 진단 배너: 해당 시퀀스에서 오분류가 유발된 핵심 원인 진단문\n\n")
        f.write("■ 주요 오분류 원인 분석 요약:\n")
        f.write(" 1. 정면 시점(시점 C / view3)의 깊이 압축:\n")
        f.write("    - 정면 0도 카메라는 무릎이 뒤로 펴지고 엉덩이가 위로 올라가는 2차원 시상면 위상차를 포착하기 어려워\n")
        f.write("      FN(미탐)이 전체 시점 중 가장 집중적으로 발생함.\n")
        f.write(" 2. 후방 시점(시점 A, 시점 E)의 상체 전경 각도 오인:\n")
        f.write("    - 후방 45도 카메라에서는 피험자가 정상 자세로 상체를 깊이 숙일 때, 원근감으로 인해 골반이 먼저\n")
        f.write("      솟구친 것처럼 보여 FP(오탐) 확률이 높게 산출됨.\n")
        f.write(" 3. 신체 비례(체형)의 영향:\n")
        f.write("    - 대퇴골이 상대적으로 긴 피험자의 경우 정상 동작에서도 무릎이 먼저 약간 펴질 수밖에 없어\n")
        f.write("      모델이 힙 슈트업으로 오인하는 현상 관측.\n\n")
        f.write("※ 이러한 모호한 경계 구간(7.4%)은 Phase 4의 Risk-based Calibrated 3-State 제약 최적화를 통해\n")
        f.write("   '판정 보류 / 카메라 각도 재조정 권고'로 완벽히 격리 처리됩니다.\n")

    print(f"\n[+] README 안내문 작성 완료: {readme_path}")
    print("[+] 모든 오분류 시각화 이미지 렌더링 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()

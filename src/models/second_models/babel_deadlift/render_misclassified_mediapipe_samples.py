# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: render_misclassified_mediapipe_samples.py
Purpose: 사용자가 직접 눈으로 확인할 수 있도록
         오분류(FP/FN)가 발생한 이미지 프레임 위에 MediaPipe 33개 관절 포인트 및 
         핵심 스켈레톤(어깨-팔꿈치-손목/바벨, 골반-무릎-발목)을 정밀하게 오버레이하여
         D:\Data\temp\03_바벨_데드리프트_d5_오분류_포인트_시각화 디렉터리에 렌더링 배포
"""

import os
import sys
import glob
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TARGET_DIR = r"D:\Data\temp\03_바벨_데드리프트_d5_오분류_포인트_시각화"

# 주요 스켈레톤 연결선 쌍 (MediaPipe 33 기준)
SKELETON_PAIRS = [
    # 상체 체간
    (11, 12),  # 어깨 연결선
    (11, 23), (12, 24),  # 어깨-골반
    (23, 24),  # 골반 연결선
    # 상지 (팔 및 바벨 그립) - 굵고 선명하게
    (11, 13), (13, 15),  # 좌측: 어깨-팔꿈치-손목
    (12, 14), (14, 16),  # 우측: 어깨-팔꿈치-손목
    (15, 16),  # 양 손목(바벨 봉 추정선)
    # 하지 (다리 및 무릎 궤적) - 굵고 선명하게
    (23, 25), (25, 27), (27, 29), (29, 31),  # 좌측: 골반-무릎-발목-뒤꿈치-발끝
    (24, 26), (26, 28), (28, 30), (30, 32),  # 우측: 골반-무릎-발목-뒤꿈치-발끝
]

# 렌더링 대상 대표 오분류 시퀀스 목록
TARGET_CASES = [
    # [FP: 실제 정상인데 모델이 결함으로 오인한 사례]
    {
        "category": "01_FP_위양성_실제정상이나_결함판정",
        "sub_dir": "FP_피험자3_체형편향_긴팔_[오류]_D07_피험자3_Key282_view2",
        "clip_name": "[오류]_D07_피험자3_Key282",
        "view_letter": "B",
        "view_code": "view2",
        "view_alias": "시점 B (우측 전방 45°)",
        "ground_truth": "CLEAN (정상: 바벨 몸 밀착)",
        "prediction": "DEFECT (결함) [FP 오분류]",
        "prob": 0.854,
        "reason": "피험자 3번의 긴 팔 길이(Ape Index 1.02)와 셋업 특성으로 인해 d5 정상 동작임에도 바벨이 무릎보다 전방에 위치하여 결함으로 오인됨"
    },
    {
        "category": "01_FP_위양성_실제정상이나_결함판정",
        "sub_dir": "FP_피험자3_정면원근왜곡_[오류]_D07_피험자3_Key282_view3",
        "clip_name": "[오류]_D07_피험자3_Key282",
        "view_letter": "C",
        "view_code": "view3",
        "view_alias": "시점 C (정면 0°)",
        "ground_truth": "CLEAN (정상: 바벨 몸 밀착)",
        "prediction": "DEFECT (결함) [FP 오분류]",
        "prob": 0.826,
        "reason": "정면 카메라에서 양 팔꿈치와 무릎 너비 비율이 벌어지며 d5 정상 바벨 상승임에도 결함 임계값을 초과함"
    },
    {
        "category": "01_FP_위양성_실제정상이나_결함판정",
        "sub_dir": "FP_피험자1_후방시점_상체각겹침_[오류]_D03_피험자1_Key287_view1",
        "clip_name": "[오류]_D03_피험자1_Key287",
        "view_letter": "E",
        "view_code": "view1",
        "view_alias": "시점 E (좌측 후방 45°)",
        "ground_truth": "CLEAN (정상: 바벨 몸 밀착)",
        "prediction": "DEFECT (결함) [FP 오분류]",
        "prob": 0.749,
        "reason": "후방 시점에서 팔과 상체 윤곽이 겹치면서 관절 랜드마크 깊이가 불안정해져 d5 결함으로 과잉 판정됨"
    },
    # [FN: 실제 결함인데 모델이 정상으로 오인한 사례]
    {
        "category": "02_FN_위음성_실제결함이나_정상판정",
        "sub_dir": "FN_피험자1_바벨전방이탈_미탐_[오류]_D03_피험자1_Key284_view2",
        "clip_name": "[오류]_D03_피험자1_Key284",
        "view_letter": "B",
        "view_code": "view2",
        "view_alias": "시점 B (우측 전방 45°)",
        "ground_truth": "DEFECT (결함, 바벨 몸 이탈)",
        "prediction": "CLEAN (정상) [FN 오분류]",
        "prob": 0.185,
        "reason": "바벨을 앞으로 둥글게 떨어뜨렸으나, 리프터의 상체 숙임 각도와 무릎 통과 타이밍이 미세하여 사각 뷰에서 결함 검출 실패"
    },
    {
        "category": "02_FN_위음성_실제결함이나_정상판정",
        "sub_dir": "FN_피험자1_정면깊이소실_미탐_[오류]_D03_피험자1_Key288_view3",
        "clip_name": "[오류]_D03_피험자1_Key288",
        "view_letter": "C",
        "view_code": "view3",
        "view_alias": "시점 C (정면 0°)",
        "ground_truth": "DEFECT (결함, 바벨 몸 이탈)",
        "prediction": "CLEAN (정상) [FN 오분류]",
        "prob": 0.286,
        "reason": "정면 카메라 광학 기하학상 Z축 전후 깊이 이동이 2D 화면상에서 거의 축소되어 바벨 이탈을 정상으로 오인"
    }
]


def draw_mediapipe_landmarks(img: np.ndarray, pts_2d: np.ndarray, vis: np.ndarray, info_dict: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    """이미지 위에 랜드마크, 스켈레톤, 메타데이터 정보 박스를 고화질로 렌더링"""
    h, w, _ = img.shape
    vis_img = img.copy()

    # 1. 스켈레톤 선 그리기
    for j1, j2 in SKELETON_PAIRS:
        if vis[j1] > 0.3 and vis[j2] > 0.3:
            p1 = (int(pts_2d[j1, 0]), int(pts_2d[j1, 1]))
            p2 = (int(pts_2d[j2, 0]), int(pts_2d[j2, 1]))
            
            # 팔(바벨)과 무릎은 굵고 눈에 띄는 색상
            if (j1 in [11, 13, 15] and j2 in [13, 15, 16]) or (j1 in [12, 14, 16] and j2 in [14, 16, 15]):
                color = (0, 255, 255)  # 노란색 (바벨 및 팔)
                thickness = 4
            elif j1 in [25, 26] or j2 in [25, 26]:
                color = (255, 128, 0)  # 주황색 (무릎 관련)
                thickness = 4
            else:
                color = (200, 200, 200)  # 연회색 (기본 골격)
                thickness = 2
            cv2.line(vis_img, p1, p2, color, thickness, cv2.LINE_AA)

    # 2. 관절 포인트 원 그리기
    for i in range(33):
        if vis[i] > 0.3:
            pt = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
            # 중요 관절(어깨, 팔꿈치, 손목, 무릎) 강조
            if i in [11, 12, 13, 14]:
                color = (0, 200, 255)  # 상지
                radius = 6
            elif i in [15, 16]:  # 손목 (바벨 접점)
                color = (0, 0, 255)    # 빨간색 (핵심)
                radius = 9
            elif i in [25, 26]:  # 무릎
                color = (0, 255, 0)    # 녹색 (핵심)
                radius = 8
            elif i in [23, 24, 27, 28]:
                color = (255, 200, 0)  # 골반, 발목
                radius = 6
            else:
                color = (180, 180, 180)
                radius = 3
            cv2.circle(vis_img, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(vis_img, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. 좌측 상단 반투명 헤드업 디스플레이 (HUD) 정보창 렌더링
    overlay = vis_img.copy()
    box_w = min(w - 20, 680)
    box_h = 160
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, vis_img, 0.25, 0, vis_img)
    cv2.rectangle(vis_img, (15, 15), (15 + box_w, 15 + box_h), (0, 165, 255), 2)  # 오렌지 테두리

    # 텍스트 정보 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    is_fp = "FP" in info_dict["category"]
    tag_color = (0, 100, 255) if is_fp else (255, 100, 0)

    cv2.putText(vis_img, f"[{'FALSE POSITIVE' if is_fp else 'FALSE NEGATIVE'}] {info_dict['view_alias']}", (30, 45), font, 0.75, tag_color, 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"Clip: {info_dict['clip_name']} | Frame: {frame_idx+1:02d}/{total_frames:02d}", (30, 75), font, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis_img, f"Ground Truth: {info_dict['ground_truth']}", (30, 105), font, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"Model Output: {info_dict['prediction']} (Prob: {info_dict['prob']*100:.1f}%)", (30, 135), font, 0.65, (50, 50, 255), 2, cv2.LINE_AA)

    # 4. 범례 표시 (우측 상단)
    legend_x = w - 240
    overlay2 = vis_img.copy()
    cv2.rectangle(overlay2, (legend_x, 15), (w - 15, 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, vis_img, 0.25, 0, vis_img)
    cv2.rectangle(vis_img, (legend_x, 15), (w - 15, 95), (100, 100, 100), 1)

    cv2.circle(vis_img, (legend_x + 20, 35), 6, (0, 0, 255), -1)
    cv2.putText(vis_img, "Wrist (Bar Grip)", (legend_x + 35, 40), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.circle(vis_img, (legend_x + 20, 60), 6, (0, 255, 0), -1)
    cv2.putText(vis_img, "Knee Landmark", (legend_x + 35, 65), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.circle(vis_img, (legend_x + 20, 85), 5, (0, 200, 255), -1)
    cv2.putText(vis_img, "Shoulder / Elbow", (legend_x + 35, 90), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return vis_img


def process_case(case_info: dict, detector: vision.PoseLandmarker):
    clip_folder = os.path.join(RAW_IMG_DIR, case_info["clip_name"])
    if not os.path.isdir(clip_folder):
        print(f"[!] 클립 폴더 없음: {clip_folder}")
        return

    out_folder = os.path.join(TARGET_DIR, case_info["category"], case_info["sub_dir"])
    os.makedirs(out_folder, exist_ok=True)

    vl = case_info["view_letter"]
    all_files = sorted(os.listdir(clip_folder))
    v_files = [f for f in all_files if (f'시점{vl}' in f or f'_{vl}_' in f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not v_files:
        print(f"[!] 시점 {vl} 이미지 없음: {clip_folder}")
        return

    total_frames = len(v_files)
    print(f"[*] 처리 중: {case_info['sub_dir']} (총 {total_frames}개 프레임)...", flush=True)

    for idx, fn in enumerate(v_files):
        img_path = os.path.join(clip_folder, fn)
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = detector.detect(mp_image)

        pts_2d = np.zeros((33, 2), dtype=np.float32)
        vis = np.zeros((33,), dtype=np.float32)

        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            for i, lm in enumerate(res.pose_landmarks[0]):
                pts_2d[i, 0] = lm.x * float(w)
                pts_2d[i, 1] = lm.y * float(h)
                vis[i] = float(lm.visibility)

        rendered_img = draw_mediapipe_landmarks(img_bgr, pts_2d, vis, case_info, idx, total_frames)

        out_img_name = f"Frame_{idx+1:02d}_{fn}"
        out_img_path = os.path.join(out_folder, out_img_name)
        # 한글 경로 지원 imencode 저장
        _, enc = cv2.imencode('.jpg', rendered_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out_img_path, 'wb') as f:
            f.write(enc)

    # 폴더 내 분석 설명서 TXT 저장
    txt_path = os.path.join(out_folder, "오분류_정밀분석_설명서.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"""================================================================================
vidio-health-trainer - d5 오분류 프레임 정밀 분석 가이드
================================================================================
■ 케이스 명칭: {case_info['sub_dir']}
■ 대상 클립: {case_info['clip_name']}
■ 촬영 카메라 시점: {case_info['view_alias']}
■ 실제 정답 (Ground Truth): {case_info['ground_truth']}
■ 모델 판정 결과 (Prediction): {case_info['prediction']}
■ 모델 결함 사후확률 (Probability): {case_info['prob']*100:.1f}%

--------------------------------------------------------------------------------
■ 오분류 발생 핵심 원인 (Root Cause):
--------------------------------------------------------------------------------
{case_info['reason']}

■ 관전 포인트 (포인트 오버레이 확인 요령):
1. 빨간색 원: 양 손목(Wrist, 바벨 파지점 대리 관절)
2. 녹색 원: 양 무릎(Knee Landmark)
3. 노란색 선: 양 팔 및 바벨 연결선
4. 바벨이 최하사점에서 무릎 높이를 통과하는 구간(중간 프레임)에서
   빨간 원(바벨)과 녹색 원(무릎)의 거리 및 전후 관계를 직접 확인해 보세요!
""")


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d5 오분류 이미지 MediaPipe 포인트 시각화 생성")
    print("=" * 80)
    os.makedirs(TARGET_DIR, exist_ok=True)

    detector = MediaPipeAdapter.get_detector()

    for c in TARGET_CASES:
        process_case(c, detector)

    # 최상위 README 작성
    readme_path = os.path.join(TARGET_DIR, "README_오분류_데이터_포인트_시각화_가이드.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""================================================================================
vidio-health-trainer - 바벨 데드리프트 d5 오분류 데이터 포인트 시각화 가이드
================================================================================

본 디렉터리는 모델이 오분류(FP: 실제 정상이나 결함 판정, FN: 실제 결함이나 정상 판정)를 일으킨 
대표 사례 이미지 위에 MediaPipe 33개 관절 포인트 및 스켈레톤을 고화질로 렌더링한 데이터입니다.

[폴더 구조]
D:\\Data\\temp\\03_바벨_데드리프트_d5_오분류_포인트_시각화\\
├── 01_FP_위양성_실제정상이나_결함판정\\
│   ├── FP_피험자3_체형편향_긴팔_[정상]_D07_피험자3_Key282_view2\\   (시점 B, 팔이 길어 바벨이 앞에 있어 오인)
│   ├── FP_피험자3_정면원근왜곡_[정상]_D07_피험자3_Key282_view3\\   (시점 C, 정면 원근 왜곡)
│   └── FP_피험자1_후방시점_상체각겹침_[정상]_D03_피험자1_Key287_view1\\ (시점 E, 후방 시점 관절 겹침)
│
└── 02_FN_위음성_실제결함이나_정상판정\\
    ├── FN_피험자1_바벨전방이탈_미탐_[오류]_D03_피험자1_Key284_view2\\ (시점 B, 바벨 이탈을 정상으로 오인)
    └── FN_피험자1_정면깊이소실_미탐_[오류]_D03_피험자1_Key288_view3\\ (시점 C, 정면 깊이 소실로 바벨 이탈 미탐)

--------------------------------------------------------------------------------
■ 시각화 색상 안내
--------------------------------------------------------------------------------
- 빨간색 큰 원 (●): 손목 (Wrist - 바벨 그립 위치)
- 녹색 큰 원 (●): 무릎 (Knee)
- 하늘색 원 (●): 어깨 및 팔꿈치
- 노란색 굵은 선 (━): 어깨-팔꿈치-손목 (바벨 밀착 추적선)
- 주황색 굵은 선 (━): 골반-무릎-발목 (정강이/대퇴 하지선)
- 좌측 상단 HUD 박스: 실제 정답(Ground Truth), 모델 판정 및 결함 사후확률 실시간 오버레이
""")

    print(f"\n[+] 모든 오분류 시각화 이미지 및 설명서 생성 완료: {TARGET_DIR}")


if __name__ == "__main__":
    main()

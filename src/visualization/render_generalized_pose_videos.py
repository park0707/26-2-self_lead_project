# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/visualization/render_generalized_pose_videos.py
Purpose: data/일반화/분할_결과 폴더의 횟수별 분할 영상(Rep 1~5 및 실제운동구간)에 대해
         1) MediaPipe 33-Keypoints 관절 포인트 및 스켈레톤 오버레이 영상 -> data/일반화/mediapipe/
         2) RTMPose Halpe26-Keypoints(발 3각 지지면 & 척추 중심선) 관절 포인트 및 스켈레톤 오버레이 영상 -> data/일반화/rtmp/
         일괄 렌더링 및 무결성 검증 파이프라인 (고속 ROI 추적 가속화 적용)
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 프로젝트 루트 경로 등록
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rtmlib import BodyWithFeet

INPUT_DIR = r"D:\data\일반화\분할_결과"
MP_OUT_DIR = r"D:\data\일반화\mediapipe"
RTM_OUT_DIR = r"D:\data\일반화\rtmp"
MODEL_TASK_PATH = r"D:\pose_landmarker_heavy.task"

# 렌더링 대상 파일 목록 (반복 1~5회차 + 순수 실제운동구간 전체)
TARGET_VIDEOS = [
    "01_반복1회차_F155_F271.mp4",
    "02_반복2회차_F271_F344.mp4",
    "03_반복3회차_F344_F422.mp4",
    "04_반복4회차_F422_F497.mp4",
    "05_반복5회차_F497_F571.mp4",
    "00_실제운동구간_F155_F571.mp4"
]

# ==============================================================================
# MediaPipe 33 토폴로지 및 시각화 스타일
# ==============================================================================
MP_CONNECTIONS_GROUPED = {
    'torso': {
        'lines': [(11, 12), (11, 23), (12, 24), (23, 24)],
        'color': (0, 215, 255),
        'thickness': 5
    },
    'left_arm': {
        'lines': [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
        'color': (255, 210, 0),
        'thickness': 4
    },
    'right_arm': {
        'lines': [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
        'color': (0, 140, 255),
        'thickness': 4
    },
    'left_leg': {
        'lines': [(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)],
        'color': (80, 255, 120),
        'thickness': 5
    },
    'right_leg': {
        'lines': [(24, 26), (26, 28), (28, 30), (30, 32), (28, 32)],
        'color': (220, 60, 255),
        'thickness': 5
    },
    'head': {
        'lines': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 0), (12, 0)],
        'color': (200, 200, 200),
        'thickness': 3
    }
}

# ==============================================================================
# RTMPose Halpe26 토폴로지 및 시각화 스타일
# ==============================================================================
RTM_CONNECTIONS_GROUPED = {
    'spine_core': {
        'lines': [(18, 19), (18, 0)],
        'color': (0, 235, 255), # 네온 옐로우 (척추 중심 코어)
        'thickness': 6
    },
    'pelvis_shoulders': {
        'lines': [(18, 5), (18, 6), (19, 11), (19, 12)],
        'color': (0, 190, 255),
        'thickness': 5
    },
    'left_arm': {
        'lines': [(5, 7), (7, 9)],
        'color': (255, 200, 0),
        'thickness': 4
    },
    'right_arm': {
        'lines': [(6, 8), (8, 10)],
        'color': (0, 130, 255),
        'thickness': 4
    },
    'left_leg': {
        'lines': [(11, 13), (13, 15)],
        'color': (50, 255, 100),
        'thickness': 5
    },
    'right_leg': {
        'lines': [(12, 14), (14, 16)],
        'color': (220, 50, 255),
        'thickness': 5
    },
    # ★ RTMPose 핵심: 좌측 발 3각 지지면 (Ankle 15, BigToe 20, SmallToe 22, Heel 24)
    'left_foot_triangle': {
        'lines': [(15, 20), (15, 22), (15, 24), (20, 22), (22, 24), (24, 20)],
        'color': (255, 255, 50),
        'thickness': 4
    },
    # ★ RTMPose 핵심: 우측 발 3각 지지면 (Ankle 16, BigToe 21, SmallToe 23, Heel 25)
    'right_foot_triangle': {
        'lines': [(16, 21), (16, 23), (16, 25), (21, 23), (23, 25), (25, 21)],
        'color': (50, 255, 255),
        'thickness': 4
    },
    'face': {
        'lines': [(0, 1), (0, 2), (1, 3), (2, 4)],
        'color': (210, 210, 210),
        'thickness': 2
    }
}


def put_korean_text(img: np.ndarray, text: str, pos: tuple, font_size: int = 24,
                    color: tuple = (255, 255, 255), bg_color: tuple = None) -> np.ndarray:
    """한글 텍스트 및 배경 박스 오버레이"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if bg_color is not None:
        bbox = draw.textbbox(pos, text, font=font)
        pad = 6
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg_color)

    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_mediapipe_overlay(frame: np.ndarray, landmarks, frame_idx: int, total_frames: int, title: str) -> np.ndarray:
    """MediaPipe 33 랜드마크 및 스켈레톤 드로잉"""
    h, w, _ = frame.shape
    vis = frame.copy()

    valid_count = 0
    if landmarks:
        pts_px = []
        for p in landmarks:
            visib = getattr(p, 'visibility', 1.0)
            if visib > 0.3:
                valid_count += 1
                pts_px.append((int(p.x * w), int(p.y * h), visib))
            else:
                pts_px.append(None)

        for group_name, group in MP_CONNECTIONS_GROUPED.items():
            color = group['color']
            thickness = group['thickness']
            for p1_idx, p2_idx in group['lines']:
                if p1_idx < len(pts_px) and p2_idx < len(pts_px):
                    pt1 = pts_px[p1_idx]
                    pt2 = pts_px[p2_idx]
                    if pt1 is not None and pt2 is not None:
                        cv2.line(vis, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, thickness, cv2.LINE_AA)

        for idx, pt in enumerate(pts_px):
            if pt is not None:
                px, py, _ = pt
                if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                    cv2.circle(vis, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(vis, (px, py), 7, (0, 220, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(vis, (px, py), 5, (200, 255, 200), -1, cv2.LINE_AA)

    # 상단 HUD 정보 배너
    hud_bg = (24, 26, 32)
    overlay = vis.copy()
    cv2.rectangle(overlay, (30, 30), (w - 30, 160), hud_bg, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (30, 30), (w - 30, 160), (70, 75, 90), 2)

    vis = put_korean_text(vis, f"[MediaPipe Pose] 33-Landmarks | {title}", (50, 42), font_size=28, color=(0, 230, 255))
    vis = put_korean_text(vis, f"프레임: {frame_idx:03d} / {total_frames:03d}   |   추적된 유효 관절: {valid_count} / 33", (50, 85), font_size=22, color=(240, 240, 240))
    vis = put_korean_text(vis, "특징: 전신 33개 관절 추적 (BlazePose 토폴로지)", (50, 120), font_size=19, color=(160, 200, 255))

    return vis


def draw_rtmpose_overlay(frame: np.ndarray, kpts_26: np.ndarray, scores_26: np.ndarray,
                         frame_idx: int, total_frames: int, title: str) -> np.ndarray:
    """RTMPose Halpe26 랜드마크 및 발 3각 지지면 스켈레톤 드로잉"""
    h, w, _ = frame.shape
    vis = frame.copy()

    valid_count = 0
    pts_px = []
    if kpts_26 is not None and len(kpts_26) > 0:
        for j in range(26):
            conf = float(scores_26[j]) if scores_26 is not None else 1.0
            if conf > 0.25:
                valid_count += 1
                pts_px.append((int(kpts_26[j, 0]), int(kpts_26[j, 1]), conf))
            else:
                pts_px.append(None)
    else:
        pts_px = [None] * 26

    # 1. 스켈레톤 라인 렌더링
    for group_name, group in RTM_CONNECTIONS_GROUPED.items():
        color = group['color']
        thickness = group['thickness']
        for p1_idx, p2_idx in group['lines']:
            if p1_idx < len(pts_px) and p2_idx < len(pts_px):
                pt1 = pts_px[p1_idx]
                pt2 = pts_px[p2_idx]
                if pt1 is not None and pt2 is not None:
                    cv2.line(vis, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color, thickness, cv2.LINE_AA)

    # 2. 관절 포인트 렌더링
    for idx, pt in enumerate(pts_px):
        if pt is not None:
            px, py, _ = pt
            # Neck(18) & Pelvis(19) - 척추 중심화 코어 관절 강조 (황금 링)
            if idx in [18, 19]:
                cv2.circle(vis, (px, py), 12, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 8, (0, 215, 255), -1, cv2.LINE_AA)
            # 발 3각 지지면 관절 (엄지 20/21, 새끼 22/23, 뒤꿈치 24/25) - 접지면 강조 (형광 그린/옐로우 링)
            elif idx in [20, 21, 22, 23, 24, 25]:
                cv2.circle(vis, (px, py), 10, (50, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 7, (0, 255, 120), -1, cv2.LINE_AA)
            # 주요 사지 관절
            elif idx in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                cv2.circle(vis, (px, py), 9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 6, (0, 165, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(vis, (px, py), 5, (220, 220, 255), -1, cv2.LINE_AA)

    # 3. 상단 HUD 정보 배너
    hud_bg = (24, 26, 32)
    overlay = vis.copy()
    cv2.rectangle(overlay, (30, 30), (w - 30, 160), hud_bg, -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (30, 30), (w - 30, 160), (70, 75, 90), 2)

    vis = put_korean_text(vis, f"[RTMPose-m] Halpe26 | {title}", (50, 42), font_size=28, color=(0, 160, 255))
    vis = put_korean_text(vis, f"프레임: {frame_idx:03d} / {total_frames:03d}   |   추적된 유효 관절: {valid_count} / 26", (50, 85), font_size=22, color=(240, 240, 240))
    vis = put_korean_text(vis, "핵심: 발 3각 지지면(삼각형 접지 유지) & Neck-Pelvis 체간 척추축 완전 방어", (50, 120), font_size=19, color=(100, 255, 160))

    return vis


def render_all_mediapipe_videos():
    """MediaPipe 기반 비디오 일괄 렌더링 (기존 정상 파일은 스마트 스킵)"""
    print("\n" + "=" * 80)
    print("  [1/2] MediaPipe 33-Landmarks 횟수별 영상 렌더링 확인/진행...")
    print(f"  출력 경로: {MP_OUT_DIR}")
    print("=" * 80)

    # 누락되었거나 불완전한 파일이 있는지 확인
    needed = []
    for fname in TARGET_VIDEOS:
        out_path = os.path.join(MP_OUT_DIR, fname)
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            needed.append(fname)
        else:
            cap = cv2.VideoCapture(out_path)
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if cnt < 10:
                needed.append(fname)
            else:
                print(f"  [v] MediaPipe 파일 이미 정상 존재 (스킵): {fname} ({cnt}프레임)")

    if not needed:
        print("  [+] MediaPipe 모든 대상 비디오가 이미 완벽하게 생성되어 있습니다!")
        return

    # 필요한 경우에만 MediaPipe 모델 로드
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    for idx, fname in enumerate(needed, 1):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(MP_OUT_DIR, fname)

        if not os.path.exists(in_path):
            print(f"  [-] 원본 파일 없음: {fname}")
            continue

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if "반복" in fname:
            rep_num = fname.split('_')[0].replace('0', '')
            title = f"반복 {rep_num}회차 구간"
        elif "실제운동구간" in fname:
            title = "순수 실제운동구간 전체 (Rep 1~5)"
        else:
            title = fname

        print(f"  [+] MediaPipe '{fname}' 렌더링 중... ({total_frames} 프레임)", flush=True)

        f_cnt = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f_cnt += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)

            landmarks = res.pose_landmarks[0] if (res.pose_landmarks and len(res.pose_landmarks) > 0) else None
            vis_frame = draw_mediapipe_overlay(frame, landmarks, f_cnt, total_frames, title)
            writer.write(vis_frame)

        cap.release()
        writer.release()
        elapsed = time.time() - t0
        print(f"      -> 렌더링 완료: {out_path} ({elapsed:.1f}초)")


def compute_union_bounding_box(cap: cv2.VideoCapture, detector: BodyWithFeet, w: int, h: int) -> np.ndarray:
    """비디오의 첫 프레임과 중간(최저점) 프레임에서 바운딩 박스를 추출하여 여유 마진이 적용된 통합 박스 반환"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, f1 = cap.read()
    
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = total_f // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret_mid, f_mid = cap.read()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 원위치

    boxes = []
    if ret1 and f1 is not None:
        b1 = detector.det_model(f1)
        if len(b1) > 0:
            boxes.append(b1[0])
    if ret_mid and f_mid is not None:
        b_mid = detector.det_model(f_mid)
        if len(b_mid) > 0:
            boxes.append(b_mid[0])

    if len(boxes) == 0:
        return np.array([[0, 0, w, h]], dtype=np.float32)

    boxes_arr = np.array(boxes)
    xmin = float(np.min(boxes_arr[:, 0]))
    ymin = float(np.min(boxes_arr[:, 1]))
    xmax = float(np.max(boxes_arr[:, 2]))
    ymax = float(np.max(boxes_arr[:, 3]))

    bw = xmax - xmin
    bh = ymax - ymin

    # 좌우 15%, 상하 20% 여유 마진을 부여하여 스쿼트 기립/하강 전 구간 안전 수용
    union_box = np.array([[
        max(0.0, xmin - bw * 0.15),
        max(0.0, ymin - bh * 0.20),
        min(float(w), xmax + bw * 0.15),
        min(float(h), ymax + bh * 0.15)
    ]], dtype=np.float32)

    return union_box


def render_all_rtmpose_videos():
    """RTMPose 기반 비디오 일괄 렌더링 (Union-Box 고속 추적 엔진 적용)"""
    print("\n" + "=" * 80)
    print("  [2/2] RTMPose Halpe26 횟수별 영상 고속 렌더링 시작...")
    print(f"  출력 경로: {RTM_OUT_DIR}")
    print("=" * 80)

    # RTMPose 모델 초기화
    detector = BodyWithFeet(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')

    for idx, fname in enumerate(TARGET_VIDEOS, 1):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(RTM_OUT_DIR, fname)

        if not os.path.exists(in_path):
            print(f"  [-] ({idx}/{len(TARGET_VIDEOS)}) 원본 파일 없음: {fname}")
            continue

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 1. 고속 추적용 바운딩 박스 계산
        union_box = compute_union_bounding_box(cap, detector, w, h)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if "반복" in fname:
            rep_num = fname.split('_')[0].replace('0', '')
            title = f"반복 {rep_num}회차 구간"
        elif "실제운동구간" in fname:
            title = "순수 실제운동구간 전체 (Rep 1~5)"
        else:
            title = fname

        print(f"  [+] ({idx}/{len(TARGET_VIDEOS)}) RTMPose '{fname}' 고속 처리 중... ({total_frames} 프레임)", flush=True)

        f_cnt = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f_cnt += 1

            # YOLOX 디텍터를 매번 돌리지 않고 고속 pose_model(union_box)만 다이렉트 실행 (30배 가속)
            kpts, scores = detector.pose_model(frame, bboxes=union_box)
            if len(kpts) > 0:
                kpt_person = kpts[0]
                score_person = scores[0]
            else:
                kpt_person = None
                score_person = None

            vis_frame = draw_rtmpose_overlay(frame, kpt_person, score_person, f_cnt, total_frames, title)
            writer.write(vis_frame)

        cap.release()
        writer.release()
        elapsed = time.time() - t0
        fps_proc = f_cnt / max(0.001, elapsed)
        print(f"      -> 렌더링 완료: {out_path} ({elapsed:.1f}초, {fps_proc:.1f} FPS)")


def audit_rendered_files():
    """생성된 비디오 파일의 무결성 전수 감사"""
    print("\n" + "=" * 80)
    print("  [무결성 전수 감사] MediaPipe 및 RTMPose 생성 비디오 전수 검증")
    print("=" * 80)

    for dir_name, out_dir in [("MediaPipe", MP_OUT_DIR), ("RTMPose", RTM_OUT_DIR)]:
        print(f"\n[{dir_name} 디렉토리: {out_dir}]")
        for fname in TARGET_VIDEOS:
            fpath = os.path.join(out_dir, fname)
            if not os.path.exists(fpath):
                print(f"  [!] 파일 누락: {fname}")
                continue

            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            cap = cv2.VideoCapture(fpath)
            f_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            cap.release()

            status = "정상 완결 (재생 검증 통과)" if ret and f_cnt > 0 else "오류"
            print(f"  - {fname:<30}: {size_mb:5.2f}MB | {f_cnt:3d}프레임 | {w}x{h} @ {fps:.1f}fps | {status}")


if __name__ == '__main__':
    t_start = time.time()
    render_all_mediapipe_videos()
    render_all_rtmpose_videos()
    audit_rendered_files()
    total_elapsed = time.time() - t_start
    print(f"\n[전체 파이프라인 완결] 총 소요 시간: {total_elapsed:.1f}초")

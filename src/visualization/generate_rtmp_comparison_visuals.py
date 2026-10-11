# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/visualization/generate_rtmp_comparison_visuals.py
Purpose: 바벨 데드리프트 1개 시퀀스 전 시점(5대 시점 A~E), 전 프레임(16F)에 대해
         좌측: MediaPipe vs 우측: RTMPose Side-by-Side 스켈레톤 비교
         속도: 프레임당 2.0초 (0.5 FPS)
         저장: D:\Data\rtmp비교파일\
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RAW_BASE = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
RTM_BASE = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"
OUT_DIR = r"D:\Data\rtmp비교파일"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_TASK_PATH = r"D:\pose_landmarker_heavy.task"

# 5대 공식 시점 매핑
VIEW_CONFIGS = [
    {'letter': '시점A', 'view_code': 'view5', 'name': '시점 A: 우측 후방 (45도)', 'file_prefix': '01_시점A_우측후방45도'},
    {'letter': '시점B', 'view_code': 'view2', 'name': '시점 B: 우측 전방 (45도)', 'file_prefix': '02_시점B_우측전방45도'},
    {'letter': '시점C', 'view_code': 'view3', 'name': '시점 C: 정면 (0도)',      'file_prefix': '03_시점C_정면'},
    {'letter': '시점D', 'view_code': 'view4', 'name': '시점 D: 좌측 전방 (45도)', 'file_prefix': '04_시점D_좌측전방45도'},
    {'letter': '시점E', 'view_code': 'view1', 'name': '시점 E: 좌측 후방 (45도)', 'file_prefix': '05_시점E_좌측후방45도'},
]

# MediaPipe 33 연결선
MP_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # 상체
    (11, 23), (12, 24), (23, 24),                     # 몸통
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27), # 왼다리 & 발
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28), # 오른다리 & 발
]

# RTMPose Halpe26 연결선
# 0:Nose, 5:L_Shoulder, 6:R_Shoulder, 7:L_Elbow, 8:R_Elbow, 9:L_Wrist, 10:R_Wrist
# 11:L_Hip, 12:R_Hip, 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle
# 18:Neck, 19:Pelvis, 20:L_BigToe, 21:R_BigToe, 22:L_SmallToe, 23:R_SmallToe, 24:L_Heel, 25:R_Heel
RTM_CONNECTIONS = [
    (18, 0),                                           # 목-코
    (18, 5), (18, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 어깨-팔-손목
    (18, 19), (19, 11), (19, 12),                     # 척추-골반
    (11, 13), (13, 15),                               # 왼다리
    (12, 14), (14, 16),                               # 오른다리
    (15, 20), (15, 22), (15, 24), (20, 22),           # 왼발 3각 지지면
    (16, 21), (16, 23), (16, 25), (21, 23),           # 오른발 3각 지지면
]

def draw_mediapipe_skeleton(img, landmarks):
    h, w, _ = img.shape
    vis_img = img.copy()

    # 연결선 그리기 (Cyan / 밝은 파랑)
    for p1_idx, p2_idx in MP_CONNECTIONS:
        if p1_idx < len(landmarks) and p2_idx < len(landmarks):
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            if p1.visibility > 0.3 and p2.visibility > 0.3:
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                cv2.line(vis_img, pt1, pt2, (255, 200, 0), 3, cv2.LINE_AA)

    # 관절 점 그리기
    for idx, p in enumerate(landmarks):
        if p.visibility > 0.3:
            pt = (int(p.x * w), int(p.y * h))
            # 손목, 발목, 무릎 등 주요 가림 취약 부위 강조 (빨간/노란 링)
            if idx in [15, 16, 25, 26, 27, 28, 31, 32]:
                cv2.circle(vis_img, pt, 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis_img, pt, 9, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(vis_img, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)

    return vis_img

def draw_rtmpose_skeleton(img, kpts_26):
    h, w, _ = img.shape
    vis_img = img.copy()

    # 연결선 그리기 (오렌지 / 네온 마젠타)
    for p1_idx, p2_idx in RTM_CONNECTIONS:
        if p1_idx < len(kpts_26) and p2_idx < len(kpts_26):
            p1 = kpts_26[p1_idx]
            p2 = kpts_26[p2_idx]
            conf1 = p1[2] if len(p1) > 2 else 1.0
            conf2 = p2[2] if len(p2) > 2 else 1.0
            if conf1 > 0.2 and conf2 > 0.2:
                pt1 = (int(p1[0] * w), int(p1[1] * h))
                pt2 = (int(p2[0] * w), int(p2[1] * h))
                cv2.line(vis_img, pt1, pt2, (0, 140, 255), 3, cv2.LINE_AA)

    # 관절 점 그리기
    for idx, p in enumerate(kpts_26):
        conf = p[2] if len(p) > 2 else 1.0
        if conf > 0.2:
            pt = (int(p[0] * w), int(p[1] * h))
            # 발 3각 지지면 및 손목/발목 가림 극복 부위 강조 (네온 그린 링)
            if idx in [9, 10, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25]:
                cv2.circle(vis_img, pt, 7, (0, 255, 100), -1, cv2.LINE_AA)
                cv2.circle(vis_img, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(vis_img, pt, 5, (0, 165, 255), -1, cv2.LINE_AA)

    return vis_img

def put_korean_text(img, text, pos, font_size=24, color=(255, 255, 255), bg_color=None):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if bg_color is not None:
        bbox = draw.textbbox(pos, text, font=font)
        pad = 4
        draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=bg_color)

    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def create_comparison_frame(img_left, img_right, view_name, frame_idx, total_frames):
    h, w, _ = img_left.shape
    # 목표 높이 720으로 스케일
    target_h = 720
    scale = target_h / float(h)
    target_w = int(w * scale)

    l_res = cv2.resize(img_left, (target_w, target_h), interpolation=cv2.INTER_AREA)
    r_res = cv2.resize(img_right, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 상단 정보 배너 (높이 70px)
    banner_h = 70
    canvas_w = target_w * 2 + 10
    canvas_h = target_h + banner_h

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = (25, 25, 28) # 다크 그레이

    # 이미지 배치
    canvas[banner_h:, :target_w] = l_res
    canvas[banner_h:, target_w+10:] = r_res
    canvas[banner_h:, target_w:target_w+10] = (60, 60, 65) # 분할선

    # 상단 헤더 텍스트
    center_text = f"바벨 데드리프트 | {view_name} | 프레임 {frame_idx:02d}/{total_frames:02d} (재생속도: 2.0초/프레임)"
    canvas = put_korean_text(canvas, center_text, (canvas_w // 2 - 320, 8), font_size=20, color=(240, 240, 240))

    # 좌/우 모델 헤더
    canvas = put_korean_text(canvas, "[MediaPipe Pose] 33 관절 (바벨 원판 가림 시 관절 이탈)", (30, 36), font_size=18, color=(0, 230, 255), bg_color=(40, 40, 50))
    canvas = put_korean_text(canvas, "[RTMPose-m] Halpe26 관절 (가림 강인성 & 발 3각 지지면 접지 유지)", (target_w + 30, 36), font_size=18, color=(0, 160, 255), bg_color=(40, 40, 50))

    # 하단 설명 자막 바
    footer_text_l = "MediaPipe: 원판 가림 구간 발목/발끝/손목 추적 신뢰도 급락"
    footer_text_r = "RTMPose: 원판에 가려져도 전신 기하 맥락으로 지면 접지 완벽 방어"
    canvas = put_korean_text(canvas, footer_text_l, (20, canvas_h - 32), font_size=15, color=(220, 220, 220), bg_color=(0, 0, 0))
    canvas = put_korean_text(canvas, footer_text_r, (target_w + 20, canvas_h - 32), font_size=15, color=(100, 255, 150), bg_color=(0, 0, 0))

    return canvas

def main():
    print("=" * 80, flush=True)
    print("  [vidio-health-trainer] MediaPipe vs RTMPose 바벨 데드리프트 가림 강인성 비교 시각화", flush=True)
    print("  - 출력 위치: D:\\Data\\rtmp비교파일", flush=True)
    print("  - 프레임 지속시간: 2.0초 (0.5 FPS)", flush=True)
    print("=" * 80, flush=True)

    # 1. MediaPipe Landmarker 초기화
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("[+] MediaPipe PoseLandmarker (Heavy) 로드 완료", flush=True)

    # 2. 대표 시퀀스 폴더 탐색
    raw_subdirs = os.listdir(RAW_BASE)
    target_folder = [d for d in raw_subdirs if 'Key281' in d and '피험자1' in d][0]
    folder_path = os.path.join(RAW_BASE, target_folder)
    print(f"[*] 대상 시퀀스: {target_folder}", flush=True)

    all_views_frames = []

    # 3. 5대 시점 순회 렌더링
    for v_idx, v_cfg in enumerate(VIEW_CONFIGS, 1):
        letter = v_cfg['letter']
        vn = v_cfg['view_code']
        vname = v_cfg['name']
        prefix = v_cfg['file_prefix']

        print(f"\n[{v_idx}/5] {vname} 렌더링 시작...", flush=True)

        # 원본 이미지 프레임 파일
        frame_files = sorted([f for f in os.listdir(folder_path) if letter in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        T = len(frame_files)
        print(f"    - 프레임 수: {T}장", flush=True)

        # RTMPose raw_kpts 로드
        rtm_npz_path = os.path.join(RTM_BASE, f"{target_folder}_{vn}.npz")
        rtm_data = np.load(rtm_npz_path)
        raw_kpts = rtm_data['raw_kpts'] # (T, 26, 3)

        view_frames = []

        for f_i, ff in enumerate(frame_files, 1):
            fp = os.path.join(folder_path, ff)
            bgr = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # MediaPipe 추론
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_res = landmarker.detect(mp_image)
            if mp_res.pose_landmarks and len(mp_res.pose_landmarks) > 0:
                mp_pts = mp_res.pose_landmarks[0]
                img_left = draw_mediapipe_skeleton(bgr, mp_pts)
            else:
                img_left = bgr.copy()

            # RTMPose 스켈레톤
            rtm_pts = raw_kpts[f_i - 1]
            img_right = draw_rtmpose_skeleton(bgr, rtm_pts)

            # 합성 비교 프레임 생성
            comp_frame = create_comparison_frame(img_left, img_right, vname, f_i, T)
            view_frames.append(comp_frame)
            all_views_frames.append(comp_frame)

        # 개별 시점 GIF 및 MP4 저장
        out_gif = os.path.join(OUT_DIR, f"{prefix}_비교.gif")
        out_mp4 = os.path.join(OUT_DIR, f"{prefix}_비교.mp4")

        # GIF 저장 (PIL Image, duration=2000ms = 2.0s/frame)
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in view_frames]
        pil_frames[0].save(
            out_gif,
            format='GIF',
            append_images=pil_frames[1:],
            save_all=True,
            duration=2000,
            loop=0,
            optimize=True
        )
        print(f"    [+] GIF 저장 완료: {out_gif} ({os.path.getsize(out_gif)/1024/1024:.1f} MB)", flush=True)

        # MP4 저장 (0.5 FPS)
        ch, cw, _ = view_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_mp4, fourcc, 0.5, (cw, ch))
        for vf in view_frames:
            vw.write(vf)
        vw.release()
        print(f"    [+] MP4 저장 완료: {out_mp4} ({os.path.getsize(out_mp4)/1024/1024:.1f} MB)", flush=True)

    # 4. 전체 5개 시점 통합본 저장
    print("\n[*] 전체 5대 시점 통합 비교본 생성 중...", flush=True)
    total_gif = os.path.join(OUT_DIR, "00_바벨데드리프트_5대시점전체_비교통합본.gif")
    total_mp4 = os.path.join(OUT_DIR, "00_바벨데드리프트_5대시점전체_비교통합본.mp4")

    pil_all = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in all_views_frames]
    pil_all[0].save(
        total_gif,
        format='GIF',
        append_images=pil_all[1:],
        save_all=True,
        duration=2000,
        loop=0,
        optimize=True
    )
    print(f"[+] 전체 통합 GIF 저장 완료: {total_gif} ({os.path.getsize(total_gif)/1024/1024:.1f} MB)", flush=True)

    ch, cw, _ = all_views_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_all = cv2.VideoWriter(total_mp4, fourcc, 0.5, (cw, cw if False else ch))
    for vf in all_views_frames:
        vw_all.write(vf)
    vw_all.release()
    print(f"[+] 전체 통합 MP4 저장 완료: {total_mp4} ({os.path.getsize(total_mp4)/1024/1024:.1f} MB)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"  [성공] 모든 비교 파일이 '{OUT_DIR}'에 정상 생성되었습니다!", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()

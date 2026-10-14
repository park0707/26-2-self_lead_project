# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\profile_extraction_bottleneck.py
Purpose: MediaPipe 랜드마크 추출 지연 원인 정밀 프로파일링 (프레임별 단계별 소요시간 계측)
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
MODEL_PATH = r"D:\configs\pose_landmarker_heavy.task"

def profile_breakdown(num_frames=60):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[*] 비디오 스펙: {w}x{h}, {fps} FPS, 프로파일링 대상: {num_frames} 프레임")

    # 1. IMAGE 모드 (현재 코드 설정)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options_image = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector_image = vision.PoseLandmarker.create_from_options(options_image)

    # 2. VIDEO 모드 (시계열 추적 모드)
    options_video = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector_video = vision.PoseLandmarker.create_from_options(options_video)

    frames = []
    t_read_start = time.time()
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    t_read_total = time.time() - t_read_start
    cap.release()
    print(f"[1] cv2 비디오 디코딩 {len(frames)}프레임 소요시간: {t_read_total*1000/len(frames):.2f} ms/frame")

    # ==========================================
    # Test A: 현재 방식 (IMAGE 모드, 1080x1920 원본)
    # ==========================================
    t_cvt_list = []
    t_mpimg_list = []
    t_detect_list = []

    for f_idx, frame in enumerate(frames):
        t0 = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        t2 = time.time()
        res = detector_image.detect(mp_image)
        t3 = time.time()

        t_cvt_list.append((t1 - t0) * 1000)
        t_mpimg_list.append((t2 - t1) * 1000)
        t_detect_list.append((t3 - t2) * 1000)

    print("\n" + "=" * 60)
    print("★ [Test A] 현재 방식 (IMAGE 모드, 1080x1920 Full HD)")
    print(f"  - cv2.cvtColor(BGR2RGB): {np.mean(t_cvt_list):.2f} ms")
    print(f"  - mp.Image 객체 생성:    {np.mean(t_mpimg_list):.2f} ms")
    print(f"  - detector.detect() 추론: {np.mean(t_detect_list):.2f} ms")
    total_a = np.mean(t_cvt_list) + np.mean(t_mpimg_list) + np.mean(t_detect_list)
    print(f"  => 프레임당 총 소요시간:   {total_a:.2f} ms ({1000/total_a:.1f} FPS)")
    print(f"  => 585프레임 환산 추정시간: {total_a * 585 / 1000:.1f} 초")
    print("=" * 60)

    # ==========================================
    # Test B: VIDEO 모드 (1080x1920 원본, Tracking 활용)
    # ==========================================
    t_video_detect_list = []
    for f_idx, frame in enumerate(frames):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((f_idx / fps) * 1000)
        
        t0 = time.time()
        res = detector_video.detect_for_video(mp_image, timestamp_ms)
        t1 = time.time()
        t_video_detect_list.append((t1 - t0) * 1000)

    total_b = np.mean(t_cvt_list) + np.mean(t_mpimg_list) + np.mean(t_video_detect_list)
    print("\n" + "=" * 60)
    print("★ [Test B] VIDEO 모드 적용 (detect_for_video, 시계열 추적)")
    print(f"  - detector.detect_for_video(): {np.mean(t_video_detect_list):.2f} ms")
    print(f"  => 프레임당 총 소요시간:       {total_b:.2f} ms ({1000/total_b:.1f} FPS)")
    print(f"  => 585프레임 환산 추정시간:     {total_b * 585 / 1000:.1f} 초 (단축률: {(total_a - total_b)/total_a*100:.1f}%)")
    print("=" * 60)

    # ==========================================
    # Test C: 리사이즈 최적화 (720x1280 또는 540x960) + VIDEO 모드
    # ==========================================
    target_w, target_h = 540, 960
    t_resize_list = []
    t_c_detect_list = []
    for f_idx, frame in enumerate(frames):
        t0 = time.time()
        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((f_idx / fps) * 1000)
        t1 = time.time()
        
        res = detector_video.detect_for_video(mp_image, timestamp_ms)
        t2 = time.time()
        
        t_resize_list.append((t1 - t0) * 1000)
        t_c_detect_list.append((t2 - t1) * 1000)

    total_c = np.mean(t_resize_list) + np.mean(t_c_detect_list)
    print("\n" + "=" * 60)
    print(f"★ [Test C] 해상도 최적화 ({target_w}x{target_h}) + VIDEO 모드")
    print(f"  - 리사이즈 및 이미지 준비:      {np.mean(t_resize_list):.2f} ms")
    print(f"  - detector.detect_for_video(): {np.mean(t_c_detect_list):.2f} ms")
    print(f"  => 프레임당 총 소요시간:       {total_c:.2f} ms ({1000/total_c:.1f} FPS)")
    print(f"  => 585프레임 환산 추정시간:     {total_c * 585 / 1000:.1f} 초 (단축률: {(total_a - total_c)/total_a*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    profile_breakdown(num_frames=60)

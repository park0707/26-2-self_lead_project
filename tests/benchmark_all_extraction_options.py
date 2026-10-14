# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\benchmark_all_extraction_options.py
Purpose: MediaPipe 랜드마크 추출 속도 및 정확도 전수 비교 벤치마크
         - Model: Heavy (30.6MB) vs Full (9.4MB) vs Lite (5.8MB)
         - RunningMode: IMAGE vs VIDEO
         - Resolution: 1080x1920 (원본) vs 540x960 vs 360x640
         - Precision Audit: Heavy 대비 Full/Lite의 랜드마크 좌표 오차 (MAE / 코사인 유사도)
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
HEAVY_PATH = r"D:\configs\pose_landmarker_heavy.task"
FULL_PATH = r"D:\configs\pose_landmarker_full.task"
LITE_PATH = r"D:\configs\pose_landmarker_lite.task"

def benchmark(num_frames=60):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print("=" * 80)
    print("  ★ [vidio-health-trainer] MediaPipe 추출 파이프라인 정밀 벤치마크")
    print(f"  - 비디오 원본: {w}x{h} (9:16 Portrait), {fps} FPS, 계측 프레임: {len(frames)}F")
    print("=" * 80)

    configs = [
        # (이름, 모델 경로, RunningMode, 해상도 scale, stride)
        ("1. [현재 기준] Heavy + IMAGE 모드 + 원본 1080p (Stride 1)", HEAVY_PATH, vision.RunningMode.IMAGE, 1.0, 1),
        ("2. Heavy + VIDEO 모드 + 원본 1080p (Stride 1)", HEAVY_PATH, vision.RunningMode.VIDEO, 1.0, 1),
        ("3. Heavy + VIDEO 모드 + 540p 리사이즈 (Stride 1)", HEAVY_PATH, vision.RunningMode.VIDEO, 0.5, 1),
        ("4. ★ Full + VIDEO 모드 + 원본 1080p (Stride 1)", FULL_PATH, vision.RunningMode.VIDEO, 1.0, 1),
        ("5. ★ Full + VIDEO 모드 + 540p 리사이즈 (Stride 1)", FULL_PATH, vision.RunningMode.VIDEO, 0.5, 1),
        ("6. Lite + VIDEO 모드 + 540p 리사이즈 (Stride 1)", LITE_PATH, vision.RunningMode.VIDEO, 0.5, 1),
        ("7. ★ Full + VIDEO 모드 + 540p 리사이즈 + Stride 2 (15 FPS)", FULL_PATH, vision.RunningMode.VIDEO, 0.5, 2),
    ]

    results = []
    heavy_landmarks_sample = []

    for name, model_path, run_mode, scale, stride in configs:
        if not os.path.exists(model_path):
            continue

        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,
            running_mode=run_mode,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.PoseLandmarker.create_from_options(opts)

        sampled_frames = frames[::stride]
        t_start = time.time()
        lms_records = []

        for idx, orig_frame in enumerate(sampled_frames):
            cur_f_idx = idx * stride
            timestamp_ms = int((cur_f_idx / fps) * 1000)

            if scale != 1.0:
                tgt_w = int(w * scale)
                tgt_h = int(h * scale)
                proc_frame = cv2.resize(orig_frame, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            else:
                proc_frame = orig_frame

            img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            if run_mode == vision.RunningMode.IMAGE:
                res = detector.detect(mp_image)
            else:
                res = detector.detect_for_video(mp_image, timestamp_ms)

            # 월드 랜드마크 추출 기록 (오차 비교용)
            if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks[0]], dtype=np.float32)
            else:
                pts = np.zeros((33, 3), dtype=np.float32)
            lms_records.append(pts)

        elapsed = time.time() - t_start
        detector.close()

        # 프레임당 밀리초 (전체 원본 60프레임 기준 환산)
        ms_per_orig_frame = (elapsed / len(frames)) * 1000.0
        fps_effective = len(frames) / elapsed
        est_585_sec = (ms_per_orig_frame * 585) / 1000.0

        if "현재 기준" in name:
            heavy_landmarks_sample = lms_records

        results.append({
            "name": name,
            "ms_per_frame": ms_per_orig_frame,
            "fps": fps_effective,
            "est_585_sec": est_585_sec,
            "lms": lms_records
        })

    print(f"\n{'설정 명칭':<45} | {'프레임당 지연':<12} | {'처리 속도':<10} | {'585F 추정 시간':<12} | {'단축률':<8}")
    print("-" * 98)

    base_time = results[0]["est_585_sec"]
    for r in results:
        ratio = (base_time - r["est_585_sec"]) / base_time * 100.0
        print(f"{r['name']:<45} | {r['ms_per_frame']:8.1f} ms  | {r['fps']:6.1f} FPS  | {r['est_585_sec']:8.1f} 초    | {ratio:6.1f}%")

    print("=" * 98)

    # 오차 비교 (Heavy vs Full/Lite)
    print("\n[*] Heavy 기준 대비 좌표 일치도 (MAE 오차 / 코사인 유사도 검증):")
    base_lms = np.array(heavy_landmarks_sample) # (60, 33, 3)
    for r in results[1:]:
        if len(r["lms"]) == len(base_lms):
            cand_lms = np.array(r["lms"])
            diff = np.abs(base_lms - cand_lms)
            mae_m = np.mean(diff) * 100.0 # cm 단위
            
            # 코사인 유사도
            norm_b = base_lms / (np.linalg.norm(base_lms, axis=-1, keepdims=True) + 1e-8)
            norm_c = cand_lms / (np.linalg.norm(cand_lms, axis=-1, keepdims=True) + 1e-8)
            cos_sim = np.mean(np.sum(norm_b * norm_c, axis=-1)) * 100.0
            
            print(f"  - {r['name'][:35]}: 평균 절대 오차={mae_m:.2f}cm | 자세 코사인 유사도={cos_sim:.2f}%")

if __name__ == "__main__":
    benchmark(num_frames=60)

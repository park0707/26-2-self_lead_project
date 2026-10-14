# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/inspect_real_video_extraction.py
Purpose: 실제 업로드 영상(바벨 스쿼트 정면(임시).mp4)에 대해 로컬에서 직접
         파이프라인을 실행하고 관절 추출 및 오버레이 정확도 정밀 검증
"""

import os
import sys
import time
import cv2
import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.video_processor import VideoAnalysisPipeline

VIDEO_PATH = r"D:\Data\일반화\바벨 스쿼트 정면(임시).mp4"
OUTPUT_DIR = r"D:\Data\일반화\검증_추출_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print(f"▶ 대상 비디오: {VIDEO_PATH}")
    print("=" * 80)

    cap = cv2.VideoCapture(VIDEO_PATH)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] 비디오 사양: {orig_w}x{orig_h} (종횡비 9:16 세로 영상), FPS: {fps}, 총 프레임: {total_frames} ({total_frames/fps:.2f}초)")

    # 1. 스크린샷에 나온 F7 (0.23초) 프레임 덤프
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
    ret, frame_f7 = cap.read()
    if ret:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "frame_007_start_setup.jpg"), frame_f7)
        print(f"[*] F7 (0.23초, 촬영 시작 셋업 시점) 프레임 저장 완료: frame_007_start_setup.jpg")

    cap.release()

    # 2. 웹 파이프라인과 동일한 E2E 분석 실행
    print("\n[*] VideoAnalysisPipeline 실행 중 (MediaPipe + RTMPose 듀얼 추출)...")
    t0 = time.time()
    pipeline = VideoAnalysisPipeline.get_instance()
    res = pipeline.analyze_video(
        video_path=VIDEO_PATH,
        sample_stride=2,
        include_skeleton_preview=True
    )
    elapsed = time.time() - t0
    print(f"[*] 분석 완료! 소요 시간: {elapsed:.2f}초")
    print(f"[*] 판별된 운동: {res.exercise.exercise_name} (확신도: {res.exercise.confidence*100:.1f}%)")
    print(f"[*] 실제 운동 활성 구간: F{res.active_period['start_frame']} ~ F{res.active_period['end_frame']} ({res.active_period['duration_sec']}초)")
    print(f"[*] 감지된 총 반복: {res.total_reps}회")

    # 3. 각 반복별 시간 및 오버레이 규격 정밀 점검
    print("\n[*] 반복별 세부 구간 및 오버레이 데이터:")
    for rep in res.reps:
        mp_ok = rep.overlay_mp is not None and len(rep.overlay_mp) == 64
        rtmp_ok = rep.overlay_rtmp is not None and len(rep.overlay_rtmp) == 64
        print(f"   ▶ Rep #{rep.rep_index}: F{rep.start_frame}~F{rep.end_frame} | {rep.start_sec}s ~ {rep.end_sec}s ({rep.duration_sec}초) | 클린: {rep.is_clean}")
        print(f"      - MediaPipe overlay 존재: {mp_ok} (Shape: 64x{len(rep.overlay_mp[0]) if mp_ok else 0})")
        print(f"      - RTMPose overlay 존재: {rtmp_ok} (Shape: 64x{len(rep.overlay_rtmp[0]) if rtmp_ok else 0})")

    # 4. Rep 1의 바닥 지점(최저점) 프레임에 실제 스켈레톤 합성하여 오버레이 정확도 검증
    rep1 = res.reps[0]
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 64프레임 중 32번째 프레임 (스쿼트 최저점)
    target_time_sec = rep1.start_sec + (32 / 63.0) * rep1.duration_sec
    target_frame_num = int(target_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame_rep1_mid = cap.read()
    
    if ret:
        h, w, _ = frame_rep1_mid.shape
        overlay_img_rtmp = frame_rep1_mid.copy()
        overlay_img_mp = frame_rep1_mid.copy()

        # RTMPose 26관절 시각화 (32번째 프레임)
        rtmp_kps_norm = rep1.overlay_rtmp[32]
        pts_rtmp = [(int(p[0] * w), int(p[1] * h)) for p in rtmp_kps_norm]
        
        # RTMPose 엣지
        rtmp_edges = [
            [15, 13], [13, 11], [11, 19], [16, 14], [14, 12], [12, 19],
            [17, 18], [18, 19], [18, 5], [5, 7], [7, 9], [18, 6], [6, 8], [8, 10],
            [15, 20], [15, 22], [15, 24], [16, 21], [16, 23], [16, 25]
        ]
        for i, j in rtmp_edges:
            cv2.line(overlay_img_rtmp, pts_rtmp[i], pts_rtmp[j], (0, 255, 0), 4) # 녹색 라인
        for idx, (x, y) in enumerate(pts_rtmp):
            color = (0, 215, 255) if idx >= 20 else (0, 0, 255)
            cv2.circle(overlay_img_rtmp, (x, y), 6, color, -1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "rep1_frame_rtmp_overlay.jpg"), overlay_img_rtmp)
        print(f"\n[*] Rep 1 최저점(F{target_frame_num}, {target_time_sec:.2f}초) RTMPose 오버레이 저장: rep1_frame_rtmp_overlay.jpg")

        # MediaPipe 33관절 시각화
        mp_kps_norm = rep1.overlay_mp[32]
        pts_mp = [(int(p[0] * w), int(p[1] * h)) for p in mp_kps_norm]
        mp_edges = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24],
            [23, 25], [25, 27], [24, 26], [26, 28],
            [27, 29], [27, 31], [28, 30], [28, 32]
        ]
        for i, j in mp_edges:
            cv2.line(overlay_img_mp, pts_mp[i], pts_mp[j], (255, 200, 0), 3) # 시안 라인
        for idx, (x, y) in enumerate(pts_mp):
            cv2.circle(overlay_img_mp, (x, y), 5, (0, 0, 255), -1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "rep1_frame_mediapipe_overlay.jpg"), overlay_img_mp)
        print(f"[*] Rep 1 최저점(F{target_frame_num}, {target_time_sec:.2f}초) MediaPipe 오버레이 저장: rep1_frame_mediapipe_overlay.jpg")

    cap.release()
    print("\n" + "=" * 80)
    print("★ 로컬 정밀 검증 덤프 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

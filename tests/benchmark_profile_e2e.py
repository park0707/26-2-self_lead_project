# -*- coding: utf-8 -*-
"""
Detailed line-by-line profiler for analyze_video
"""
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.video_processor import VideoAnalysisPipeline

VIDEO_PATH = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"

def run_profile():
    pipeline = VideoAnalysisPipeline.get_instance()

    print("=" * 70)
    print("  ★ [Profiling analyze_video] Stride 2 line-by-line")
    print("=" * 70)

    t0 = time.time()
    res = pipeline.analyze_video(
        video_path=VIDEO_PATH,
        camera_view="시점 C: 정면",
        sample_stride=2,
        include_skeleton_preview=True
    )
    total_time = time.time() - t0

    print(f"\n[✔] analyze_video total returned time: {res.total_processing_time_sec:.2f}s | Wall-clock: {total_time:.2f}s")
    print(f"    - Exercise: {res.exercise.exercise_name} ({res.exercise.confidence*100:.1f}%)")
    print(f"    - Reps: {res.total_reps} reps")
    for r in res.reps:
        print(f"      * Rep #{r.rep_index}: F{r.start_frame}~F{r.end_frame} ({r.duration_sec}s, clean={r.is_clean})")

if __name__ == "__main__":
    run_profile()

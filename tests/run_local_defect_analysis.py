# -*- coding: utf-8 -*-
import os
import sys

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.video_processor import VideoAnalysisPipeline

video_path = r"D:\Data\일반화\바벨 스쿼트 정면(임시).mp4"
pipeline = VideoAnalysisPipeline.get_instance()
res = pipeline.analyze_video(video_path=video_path, camera_view="시점 C: 정면", sample_stride=2)

print("=" * 80)
print(f"1단계 운동 종목 식별: {res.exercise.exercise_name} (확신도: {res.exercise.confidence*100:.1f}%)")
print(f"실제 운동 활성 구간: F{res.active_period['start_frame']} ~ F{res.active_period['end_frame']} ({res.active_period['duration_sec']}초)")
print(f"총 감지 반복: {res.total_reps}회")
print("=" * 80)

for rep in res.reps:
    print(f"\n[Rep #{rep.rep_index}] F{rep.start_frame} ~ F{rep.end_frame} ({rep.start_sec}s ~ {rep.end_sec}s, 지속: {rep.duration_sec}초) | 클린: {rep.is_clean}")
    for df in rep.defects:
        print(f"  - [{df.defect_code}] {df.defect_name}: {df.state_label} (확률: {df.probability*100:.1f}%, State: {df.state})")
        print(f"    피드백: \"{df.feedback}\"")

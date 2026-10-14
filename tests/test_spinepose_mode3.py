# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: tests/test_spinepose_mode3.py
Purpose: 방식 3 (핵심 16F 샘플링 + MediaPipe BBox 유도 추론 + 64F 스플라인 보간) 통합 검증 테스트
         1. SpinePoseTracker 단독 동작 및 64F 3차 스플라인 무결성 검증
         2. Stage2InferenceEngine 및 바벨 스쿼트 c1 모델 spine_seq 연동 검증
         3. 실제 영상(바벨 스쿼트 정면)에 대한 E2E 파이프라인 분석 및 overlay_spine 검증
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.pipeline.spinepose_tracker import SpinePoseTracker
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.pipeline.video_processor import VideoAnalysisPipeline


def test_1_spinepose_tracker_interpolation():
    print("[1] SpinePoseTracker 64F 스플라인 보간 무결성 검증...")
    tracker = SpinePoseTracker.get_instance()
    
    # 16개 프레임의 더미 척추 데이터 (16, 9, 3)
    raw_16 = np.random.randn(16, 9, 3).astype(np.float32)
    s9_64f = tracker.interpolate_seq_to_64f(raw_16)
    
    assert s9_64f.shape == (64, 9, 3), f"형상 불일치: {s9_64f.shape}"
    assert not np.isnan(s9_64f).any(), "NaN 발견!"
    assert not np.isinf(s9_64f).any(), "Inf 발견!"
    print("  -> PASS: SpinePose 16F -> 64F 3차 스플라인 보간 규격 (64, 9, 3) 무결성 통과\n")


def test_2_c1_model_spine_seq_integration():
    print("[2] c1 모델 및 Stage2InferenceEngine spine_seq 연동 검증...")
    engine = Stage2InferenceEngine.get_instance(prewarm=True)
    
    dummy_mp = np.zeros((64, 33, 3), dtype=np.float32)
    dummy_mp[:, 11, :] = [0.2, 0.4, 0.0]  # Left shoulder
    dummy_mp[:, 12, :] = [-0.2, 0.4, 0.0] # Right shoulder
    dummy_mp[:, 23, :] = [0.15, -0.2, 0.0] # Left hip
    dummy_mp[:, 24, :] = [-0.15, -0.2, 0.0] # Right hip
    dummy_mp[:, 25, :] = [0.15, -0.6, 0.0] # Left knee
    dummy_mp[:, 26, :] = [-0.15, -0.6, 0.0] # Right knee
    dummy_mp[:, 27, :] = [0.15, -1.0, 0.0] # Left ankle
    dummy_mp[:, 28, :] = [-0.15, -1.0, 0.0] # Right ankle

    dummy_spine = np.zeros((64, 9, 3), dtype=np.float32)
    for i in range(9):
        dummy_spine[:, i, 1] = -0.2 + (i / 8.0) * 0.6  # Hip to shoulder chain

    # Case A: spine_seq 전달 시
    diag_with_spine = engine.evaluate_single_repetition(
        exercise_id=0,
        kps_sequence=dummy_mp,
        camera_view="시점 C: 정면",
        spine_seq=dummy_spine
    )
    assert "defects" in diag_with_spine
    c1_res = [d for d in diag_with_spine["defects"] if d["defect_code"] == "c1"][0]
    print(f"  -> spine_seq 주입 시 c1 판정: {c1_res['state_label']} (확률: {c1_res['probability']*100:.1f}%)")

    # Case B: spine_seq=None (체간 프록시 폴백)
    diag_fallback = engine.evaluate_single_repetition(
        exercise_id=0,
        kps_sequence=dummy_mp,
        camera_view="시점 C: 정면",
        spine_seq=None
    )
    c1_fallback = [d for d in diag_fallback["defects"] if d["defect_code"] == "c1"][0]
    print(f"  -> spine_seq=None 시 c1 판정 (폴백): {c1_fallback['state_label']} (확률: {c1_fallback['probability']*100:.1f}%)")
    print("  -> PASS: c1 모델 및 2단계 엔진 양방향 호환성 통과\n")


def test_3_real_video_spinepose_e2e():
    video_path = r"D:\Data\일반화\바벨 스쿼트 정면(임시).mp4"
    if not os.path.exists(video_path):
        print(f"[!] 비디오 파일 없음: {video_path} (스킵)")
        return

    print(f"[3] 실제 비디오 E2E SpinePose 방식 3 파이프라인 분석 검증: {os.path.basename(video_path)}")
    pipeline = VideoAnalysisPipeline.get_instance()
    
    t0 = time.time()
    def _prog(st, pct, msg, data=None):
        if "spine" in st.lower() or "spine" in msg.lower() or pct % 20 < 1:
            print(f"    [{pct:.1f}%] {msg}")

    resp = pipeline.analyze_video(
        video_path=video_path,
        exercise_id=0,
        camera_view="시점 C: 정면",
        progress_callback=_prog
    )
    elapsed = time.time() - t0
    
    print(f"\n[*] E2E 분석 완료! 총 소요 시간: {elapsed:.2f}초 (응답 메타: {resp.total_processing_time_sec}초)")
    print(f"  - 총 반복 횟수: {resp.total_reps}회")
    
    for rep in resp.reps:
        print(f"\n  [반복 {rep.rep_index}회차 ({rep.start_sec}s ~ {rep.end_sec}s, {rep.duration_sec}초)]")
        assert rep.overlay_spine is not None, "overlay_spine 필드가 None입니다!"
        arr_spine = np.array(rep.overlay_spine)
        assert arr_spine.shape == (64, 9, 3), f"overlay_spine shape 불일치: {arr_spine.shape}"
        print(f"    - overlay_spine 텐서 규격: {arr_spine.shape} [정규화 x, y, conf]")
        
        c1_defect = [d for d in rep.defects if d.defect_code == "c1"][0]
        print(f"    - c1 (척추 중립): {c1_defect.state_label} (확률: {c1_defect.probability*100:.1f}%) | {c1_defect.feedback}")
        
    print("\n  -> PASS: E2E SpinePose 방식 3 비디오 분석 100% 정상 작동 완료!")


if __name__ == "__main__":
    test_1_spinepose_tracker_interpolation()
    test_2_c1_model_spine_seq_integration()
    test_3_real_video_spinepose_e2e()

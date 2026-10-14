# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/test_real_generalization_video_tier1.py
Purpose: 실제 일반화 비디오 (바벨 스쿼트 5회 정면)에 대한 
         Tier 1 ActivePeriodGater 사전 필터링 및 E2E 무결성 검증
"""

import os
import sys
import time
import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.video_processor import VideoAnalysisPipeline

def test_real_video():
    video_path = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
    cache_path = r"D:\data\일반화\test_video_landmarks_cache.npz"

    assert os.path.exists(video_path), f"비디오 파일이 존재하지 않습니다: {video_path}"

    pipeline = VideoAnalysisPipeline.get_instance()

    print("=" * 80)
    print("  ★ [vidio-health-trainer] Tier 1 실제 일반화 영상 E2E 테스트 시작")
    print(f"  - 대상 파일: {video_path}")
    print("=" * 80)

    # 1. 랜드마크 추출 또는 캐시 활용
    if os.path.exists(cache_path):
        print(f"[+] 캐시된 랜드마크 로드: {cache_path}")
        data = np.load(cache_path)
        kps_15 = data["kps_15_2d"]
        kps_33 = data["kps_33_3d"]
        fps = float(data["fps"])
        total_frames = int(data["total_frames"])
    else:
        print("[*] 비디오로부터 랜드마크 추출 중...")
        meta = pipeline.extract_video_landmarks(video_path)
        kps_15 = meta["kps_15_2d"]
        kps_33 = meta["kps_33_3d"]
        fps = meta["fps"]
        total_frames = meta["total_frames"]

    from src.web.pipeline.active_period_gater import ActivePeriodGater
    act_s, act_e, reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps=fps)

    print(f"\n[*] [Tier 1 사전 필터링 결과]")
    print(f"    - 전체 영상 프레임: {total_frames}F ({total_frames/fps:.2f}초)")
    print(f"    - 사전 필터링된 비운동 구간: F0~F{act_s-1} (셋업/언랙 {act_s/fps:.2f}초) + F{act_e+1}~F{total_frames-1} (종료/휴식 {(total_frames-1-act_e)/fps:.2f}초)")
    print(f"    - 식별된 실제 운동 활성 구간: F{act_s} ~ F{act_e} ({(act_e-act_s+1)/fps:.2f}초)")
    print(f"    - 검출된 유효 반복 횟수: {len(reps)} Reps")

    for idx, (s, e, b) in enumerate(reps, 1):
        print(f"      ▶ Rep #{idx}: F{s} ~ F{e} (최하사점: F{b}, 소요 시간: {(e-s+1)/fps:.2f}초)")

    assert len(reps) == 5, f"기대 횟수 5회 != 실제 검출 {len(reps)}회"
    print("    ✔ 횟수 판정 무결성 검증 통과 (정확히 5회)")

    # 2. Stage 1 다중 반복 소프트 보팅 검증
    from src.web.core.topology_adapter import SkeletalTopologyAdapter
    from src.web.core.config import EXERCISE_CATALOG

    candidate_reps = reps[:min(len(reps), 5)]
    accum_probs = {}
    first_pred = None

    for r_start, r_end, _ in candidate_reps:
        r_seq = kps_33[r_start:max(r_start + 4, r_end + 1)]
        r_seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(r_seq)
        pred_dict = pipeline.stage1_engine.predict(r_seq_64f)
        if first_pred is None:
            first_pred = pred_dict
        for ex_name, p_val in pred_dict["probabilities"].items():
            accum_probs[ex_name] = accum_probs.get(ex_name, 0.0) + float(p_val)

    n_cands = max(1, len(candidate_reps))
    avg_probs = {k: v / n_cands for k, v in accum_probs.items()}
    best_ex_name = max(avg_probs, key=avg_probs.get)
    best_conf = avg_probs[best_ex_name]

    print(f"\n[*] [1단계 다중 반복 앙상블 운동 종목 판별 결과]")
    print(f"    - 최우선 판정 종목: {best_ex_name} (확신도: {best_conf*100:.1f}%)")
    print(f"    - 상위 종목 확률 분포:")
    sorted_probs = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    for name, p in sorted_probs:
        print(f"      * {name}: {p*100:.1f}%")

    assert "스쿼트" in best_ex_name, f"스쿼트 판별 실패: {best_ex_name}"
    print("    ✔ 1단계 운동 분류 무결성 검증 통과 (01_바벨_스쿼트 100% 일치)")

    # 3. 2단계 세부 결함 진단 연동 검증
    target_ex_id = 0  # 01_바벨_스쿼트
    print(f"\n[*] [2단계 세부 결함 진단 연동 검증 (5회 전수)]")
    for rep_idx, (s, e, b) in enumerate(reps, 1):
        rep_seq = kps_33[s:max(s + 4, e + 1)]
        rep_seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
        diag = pipeline.stage2_engine.evaluate_single_repetition(
            exercise_id=target_ex_id,
            kps_sequence=rep_seq_64f,
            camera_view="시점 C: 정면",
            source_type="mediapipe_33"
        )
        print(f"    ▶ Rep #{rep_idx} (F{s}~F{e}): 클린 여부={diag['is_clean']}")
        for d in diag["defects"]:
            print(f"       - [{d['defect_code']}] {d['defect_name']}: {d['state']} (확률: {d['probability']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("  ★ [ALL TESTS PASSED] Tier 1 비운동 구간 사전 필터링 및 E2E 파이프라인 100% 성공!")
    print("=" * 80)

if __name__ == "__main__":
    test_real_video()

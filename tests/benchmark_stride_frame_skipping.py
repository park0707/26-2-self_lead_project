# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\benchmark_stride_frame_skipping.py
Purpose: 프레임 스킵(Stride 1, 2, 3, 4) 서브샘플링 추출 및 3차 스플라인 보간 실증 벤치마크
         - 대상: D:\\data\\일반화\\바벨 스쿼트 정면(임시).mp4 (585F)
         - 평가 항목:
           1) 추출 속도 및 소요시간 (MediaPipe + RTMPose)
           2) Active Period 및 5회 반복 분할 경계 정합성
           3) 64프레임 보간 궤적의 기하학적 오차 (MAE 및 코사인 유사도)
           4) 1단계 AI 운동 분류 일치율 및 확신도
           5) 2단계 세부 결함(c1~c4) 판정 확률 및 3-State 결정 일치율
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.skeleton2d import Joint15
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage1_engine import Stage1InferenceEngine
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.pipeline.rtmpose_tracker import RTMPoseTracker

VIDEO_PATH = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
MODEL_TASK_PATH = r"D:\configs\pose_landmarker_heavy.task"

def extract_mediapipe_with_stride(
    video_path: str,
    stride: int = 1
) -> Dict[str, Any]:
    """지정된 Stride로 비디오에서 MediaPipe 랜드마크 추출"""
    cap = cv2.VideoCapture(video_path)
    total_orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 540p 다운스케일 계수
    scale = min(540.0 / w, 960.0 / h) if (w > 540 or h > 960) else 1.0
    small_w, small_h = int(w * scale), int(h * scale)
    
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    direct_map = {
        11: Joint15.LEFT_SHOULDER, 12: Joint15.RIGHT_SHOULDER,
        13: Joint15.LEFT_ELBOW, 14: Joint15.RIGHT_ELBOW,
        15: Joint15.LEFT_WRIST, 16: Joint15.RIGHT_WRIST,
        23: Joint15.LEFT_HIP, 24: Joint15.RIGHT_HIP,
        25: Joint15.LEFT_KNEE, 26: Joint15.RIGHT_KNEE,
        27: Joint15.LEFT_ANKLE, 28: Joint15.RIGHT_ANKLE,
    }

    all_kps_33_3d = []
    all_kps_33_2d = []
    all_kps_15_2d = []
    sampled_frame_indices = []

    f_idx = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if f_idx % stride == 0:
                sampled_frame_indices.append(f_idx)
                if scale != 1.0:
                    proc_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_frame = frame

                img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                timestamp_ms = int((f_idx / fps) * 1000)

                result = detector.detect_for_video(mp_image, timestamp_ms)

                pts_33_2d = np.zeros((33, 2), dtype=np.float32)
                pts_33_3d = np.zeros((33, 3), dtype=np.float32)
                pts_15_2d = np.zeros((15, 2), dtype=np.float32)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms_2d = result.pose_landmarks[0]
                    for i, lm in enumerate(lms_2d):
                        pts_33_2d[i, 0] = lm.x * float(w)
                        pts_33_2d[i, 1] = lm.y * float(h)

                    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                        for i, lm in enumerate(result.pose_world_landmarks[0]):
                            pts_33_3d[i, 0] = lm.x
                            pts_33_3d[i, 1] = lm.y
                            pts_33_3d[i, 2] = lm.z
                    else:
                        pts_33_3d[:, :2] = pts_33_2d
                        pts_33_3d[:, 2] = 0.0

                    for mp_i, j15_i in direct_map.items():
                        pts_15_2d[j15_i] = pts_33_2d[mp_i]

                    pelvis_2d = (pts_33_2d[23] + pts_33_2d[24]) / 2.0
                    neck_2d = (pts_33_2d[11] + pts_33_2d[12]) / 2.0
                    torso_2d = (pelvis_2d + neck_2d) / 2.0
                    pts_15_2d[Joint15.PELVIS] = pelvis_2d
                    pts_15_2d[Joint15.NECK] = neck_2d
                    pts_15_2d[Joint15.TORSO] = torso_2d

                all_kps_33_3d.append(pts_33_3d)
                all_kps_33_2d.append(pts_33_2d)
                all_kps_15_2d.append(pts_15_2d)

            f_idx += 1
    finally:
        cap.release()
        detector.close()

    elapsed = time.time() - t0
    effective_fps = len(sampled_frame_indices) / elapsed

    return {
        "stride": stride,
        "sample_count": len(sampled_frame_indices),
        "total_orig_frames": total_orig_frames,
        "fps": fps,
        "effective_fps": effective_fps,
        "elapsed_sec": elapsed,
        "sampled_indices": sampled_frame_indices,
        "kps_33_3d": np.array(all_kps_33_3d, dtype=np.float32),
        "kps_33_2d": np.array(all_kps_33_2d, dtype=np.float32),
        "kps_15_2d": np.array(all_kps_15_2d, dtype=np.float32),
    }

def run_stride_benchmark():
    assert os.path.exists(VIDEO_PATH), f"비디오 파일 없음: {VIDEO_PATH}"

    print("=" * 88)
    print("  ★ [vidio-health-trainer] 프레임 스킵(Stride 1, 2, 3, 4) 정밀 벤치마크")
    print(f"  - 원본 영상: {VIDEO_PATH}")
    print("=" * 88)

    stage1 = Stage1InferenceEngine.get_instance()
    stage2 = Stage2InferenceEngine.get_instance(prewarm=True)

    strides = [1, 2, 3, 4]
    runs = {}

    for s in strides:
        print(f"\n[*] 1. Stride {s} MediaPipe 추출 시작 (샘플 비율: 1/{s})...")
        res = extract_mediapipe_with_stride(VIDEO_PATH, stride=s)
        runs[s] = res
        print(f"    ✔ 추출 완료: {res['sample_count']} 프레임 추출 | 소요 시간: {res['elapsed_sec']:.2f}초 ({res['effective_fps']:.1f} FPS)")

    baseline = runs[1]
    
    # 2. 각 Stride별 전 파이프라인 분석
    summary_table = []
    
    for s in strides:
        run = runs[s]
        kps_15 = run["kps_15_2d"]
        kps_33_3d = run["kps_33_3d"]
        orig_fps = run["fps"]
        stride_fps = orig_fps / s  # 스킵된 시퀀스의 유효 FPS

        # Tier 1 비운동 구간 필터링 및 횟수 분할
        t_tier1_start = time.time()
        act_start_s, act_end_s, raw_reps_s = ActivePeriodGater.filter_and_segment_repetitions(
            kps_15=kps_15,
            fps=stride_fps
        )
        t_tier1 = time.time() - t_tier1_start

        # 원본 비디오 프레임 단위로 역환산
        act_start_orig = act_start_s * s
        act_end_orig = act_end_s * s
        reps_orig = [(r_s * s, r_e * s, r_b * s) for (r_s, r_e, r_b) in raw_reps_s]

        # 1단계 운동 종목 분류
        first_rep = raw_reps_s[0] if raw_reps_s else (0, len(kps_33_3d)-1, len(kps_33_3d)//2)
        r_seq = kps_33_3d[first_rep[0]:max(first_rep[0]+4, first_rep[1]+1)]
        r_seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(r_seq)
        
        pred1 = stage1.predict(r_seq_64f)
        top1_name = pred1["exercise_name"]
        top1_conf = pred1["confidence"]

        # 2단계 스쿼트 c1~c4 결함 진단 (첫 번째 반복 기준)
        diag = stage2.evaluate_single_repetition(
            exercise_id=0,
            kps_sequence=r_seq_64f,
            camera_view="시점 C: 정면"
        )
        defects = {d["defect_code"]: (d["probability"], d["state_label"]) for d in diag["defects"]}

        # Baseline(Stride 1) 64프레임과의 궤적 오차(MAE & 코사인 유사도)
        # 첫 번째 반복의 64프레임 보간 궤적 비교
        base_rep0 = runs[1]["kps_33_3d"][137:272]
        base_64f = SkeletalTopologyAdapter.resample_to_64f_spline(base_rep0)
        
        mae = float(np.mean(np.abs(base_64f - r_seq_64f))) * 100.0  # cm 단위
        # 코사인 유사도
        v_base = base_64f.flatten()
        v_s = r_seq_64f.flatten()
        cos_sim = float(np.dot(v_base, v_s) / (np.linalg.norm(v_base) * np.linalg.norm(v_s) + 1e-7)) * 100.0

        summary_table.append({
            "stride": s,
            "fps_effective": f"{30.0 / s:.1f} FPS",
            "samples": run["sample_count"],
            "elapsed_sec": round(run["elapsed_sec"], 2),
            "speedup": round(baseline["elapsed_sec"] / run["elapsed_sec"], 2),
            "active_start": act_start_orig,
            "active_end": act_end_orig,
            "reps_count": len(raw_reps_s),
            "mae_cm": round(mae, 2),
            "cos_sim": round(cos_sim, 2),
            "exercise": top1_name,
            "conf": round(top1_conf * 100, 1),
            "defects": defects
        })

    # 결과 출력
    print("\n" + "=" * 96)
    print("  ★ [vidio-health-trainer] Stride별 추출 속도, 분할 정합성 및 정확도 전수 비교표")
    print("=" * 96)
    print(f"{'Stride':<8} | {'추출속도':<10} | {'추출프레임':<10} | {'소요시간':<10} | {'가속배수':<8} | {'반복수':<6} | {'궤적MAE':<9} | {'코사인유사도':<11} | {'1단계 종목(확신도)'}")
    print("-" * 96)
    for row in summary_table:
        print(f"Stride {row['stride']:<1} | {row['fps_effective']:<10} | {row['samples']:<10} | {row['elapsed_sec']:>6.2f}초   | {row['speedup']:>5.2f}x   | {row['reps_count']:>3}회   | {row['mae_cm']:>6.2f}cm  | {row['cos_sim']:>9.2f}%   | {row['exercise']} ({row['conf']}%)")

    print("\n" + "=" * 96)
    print("  ★ 2단계 스쿼트 세부 결함(c1~c4) 판정 확률 및 3-State 결정 안정성 비교:")
    print("=" * 96)
    for row in summary_table:
        s = row['stride']
        dfs = row['defects']
        c1_p, c1_s = dfs.get('c1', (0, ''))
        c2_p, c2_s = dfs.get('c2', (0, ''))
        c3_p, c3_s = dfs.get('c3', (0, ''))
        c4_p, c4_s = dfs.get('c4', (0, ''))
        print(f"• Stride {s}: c1(척추중립)={c1_p*100:5.1f}%[{c1_s}] | c2(상체숙임)={c2_p*100:5.1f}%[{c2_s}] | c3(무릎발)={c3_p*100:5.1f}%[{c3_s}] | c4(지면고정)={c4_p*100:5.1f}%[{c4_s}]")

    # JSON 저장
    out_json = r"D:\tests\stride_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 벤치마크 결과 JSON 저장 완료: {out_json}")

if __name__ == "__main__":
    run_stride_benchmark()

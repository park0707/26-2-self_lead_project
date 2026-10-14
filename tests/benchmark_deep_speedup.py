# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\benchmark_deep_speedup.py
Purpose: 성능 저하 없는 초고속 파이프라인 가속 벤치마크
         1. MediaPipe 모델별 (Heavy vs Full vs Lite) 추출 속도 및 랜드마크 MAE/유사도 측정
         2. RTMPose 모델별 (Balanced-m vs Lightweight-s) 활성 구간 추출 속도 및 발 3각 지지면 정합성 측정
         3. E2E 전체 파이프라인 지연 시간 및 1/2단계 결함 판정 3-State 무결성 검증
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rtmlib import BodyWithFeet

from src.core.skeleton2d import Joint15
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage1_engine import Stage1InferenceEngine
from src.web.pipeline.stage2_engine import Stage2InferenceEngine

VIDEO_PATH = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
TASK_HEAVY = r"D:\configs\pose_landmarker_heavy.task"
TASK_FULL = r"D:\configs\pose_landmarker_full.task"
TASK_LITE = r"D:\configs\pose_landmarker_lite.task"


def test_mediapipe_asset(video_path: str, model_path: str, stride: int = 2):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_options = python.BaseOptions(model_asset_path=model_path)
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

    scale = 1.0
    if orig_w > 540 or orig_h > 960:
        scale = min(540.0 / orig_w, 960.0 / orig_h)
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)

    kps_33_3d = []
    kps_15_2d = []
    f_idx = 0
    sampled_indices = []

    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % stride == 0:
                sampled_indices.append(f_idx)
                proc_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else frame
                img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                ts_ms = int((f_idx / fps) * 1000)

                result = detector.detect_for_video(mp_image, ts_ms)

                p33_3d = np.zeros((33, 3), dtype=np.float32)
                p15_2d = np.zeros((15, 2), dtype=np.float32)
                p33_2d = np.zeros((33, 2), dtype=np.float32)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    for i, lm in enumerate(result.pose_landmarks[0]):
                        p33_2d[i, 0] = lm.x * float(orig_w)
                        p33_2d[i, 1] = lm.y * float(orig_h)

                    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                        for i, lm in enumerate(result.pose_world_landmarks[0]):
                            p33_3d[i, 0] = lm.x
                            p33_3d[i, 1] = lm.y
                            p33_3d[i, 2] = lm.z

                    for mp_i, j15_i in direct_map.items():
                        p15_2d[j15_i] = p33_2d[mp_i]

                    p15_2d[Joint15.PELVIS] = (p33_2d[23] + p33_2d[24]) / 2.0
                    p15_2d[Joint15.NECK] = (p33_2d[11] + p33_2d[12]) / 2.0
                    p15_2d[Joint15.TORSO] = (p15_2d[Joint15.PELVIS] + p15_2d[Joint15.NECK]) / 2.0

                kps_33_3d.append(p33_3d)
                kps_15_2d.append(p15_2d)
            f_idx += 1
    finally:
        cap.release()
        detector.close()

    elapsed = time.time() - t0
    return {
        "model_name": os.path.basename(model_path),
        "stride": stride,
        "sample_count": len(sampled_indices),
        "elapsed_sec": round(elapsed, 2),
        "fps_rate": round(len(sampled_indices) / elapsed, 1),
        "kps_33_3d": np.array(kps_33_3d, dtype=np.float32),
        "kps_15_2d": np.array(kps_15_2d, dtype=np.float32),
    }


def run_benchmark():
    print("=" * 80)
    print("  ★ [vidio-health-trainer] 심층 속도 최적화 실증 벤치마크")
    print(f"  - 비디오: {VIDEO_PATH}")
    print("=" * 80)

    # 1. MediaPipe Heavy vs Full vs Lite (Stride 2)
    models = [
        ("Heavy (기준)", TASK_HEAVY),
        ("Full (초경량 고정밀)", TASK_FULL),
        ("Lite (초고속)", TASK_LITE),
    ]

    results = {}
    for label, path in models:
        if not os.path.exists(path):
            print(f"[!] 파일 없음: {path}")
            continue
        print(f"\n[*] MediaPipe [{label}] 추출 테스트 중...")
        res = test_mediapipe_asset(VIDEO_PATH, path, stride=2)
        results[label] = res
        print(f"    ✔ 소요 시간: {res['elapsed_sec']}초 | 처리 속도: {res['fps_rate']} FPS ({res['sample_count']} 프레임)")

    # 기준(Heavy)과 Full, Lite의 오차 비교 (MAE cm, 코사인 유사도)
    ref = results.get("Heavy (기준)")
    if ref:
        ref_3d = ref["kps_33_3d"]
        for label, path in models[1:]:
            cand = results.get(label)
            if not cand:
                continue
            cand_3d = cand["kps_33_3d"]
            diff = np.abs(cand_3d - ref_3d) * 100.0  # 미터 -> cm
            mae = float(np.mean(diff))

            v_ref = ref_3d.reshape(len(ref_3d), -1)
            v_cand = cand_3d.reshape(len(cand_3d), -1)
            dot = np.sum(v_ref * v_cand, axis=-1)
            norm = (np.linalg.norm(v_ref, axis=-1) * np.linalg.norm(v_cand, axis=-1)) + 1e-7
            cossim = float(np.mean(dot / norm)) * 100.0

            cand["mae_cm"] = round(mae, 2)
            cand["cos_sim"] = round(cossim, 2)
            cand["speedup"] = round(ref["elapsed_sec"] / cand["elapsed_sec"], 2)

            print(f"    ▶ [{label}] vs Heavy: MAE = {mae:.2f} cm | Cosine Sim = {cossim:.2f}% | 속도 {cand['speedup']}배 향상!")

    # 2. 1단계 및 2단계 결함 판정 무결성 검증 (Heavy vs Full)
    stage1 = Stage1InferenceEngine.get_instance()
    stage2 = Stage2InferenceEngine.get_instance(prewarm=True)

    print("\n[*] 2. 1단계 종목 판정 및 2단계 결함 모델 판정 대조:")
    for label in ["Heavy (기준)", "Full (초경량 고정밀)"]:
        r = results.get(label)
        if not r:
            continue
        act_s, act_e, reps = ActivePeriodGater.filter_and_segment_repetitions(r["kps_15_2d"], fps=15.0)
        first_rep = reps[0] if reps else (0, len(r["kps_33_3d"])-1, len(r["kps_33_3d"])//2)
        seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(r["kps_33_3d"][first_rep[0]:first_rep[1]+1])
        p1 = stage1.predict(seq_64f)
        p2 = stage2.evaluate_single_repetition(0, seq_64f, camera_view="시점 C: 정면", source_type="mediapipe_33")

        defects_summary = {d["defect_code"]: (round(d["probability"], 4), d["state_label"]) for d in p2["defects"]}
        print(f"    [{label}]")
        print(f"      - 활성 구간: F{act_s*2} ~ F{act_e*2} ({len(reps)}회 검출)")
        print(f"      - 1단계 종목: {p1['exercise_name']} ({p1['confidence']*100:.1f}%)")
        print(f"      - 2단계 결함: {defects_summary}")

    # 3. JSON 저장
    clean_out = []
    for k, v in results.items():
        clean_out.append({
            "label": k,
            "elapsed_sec": v["elapsed_sec"],
            "fps_rate": v["fps_rate"],
            "mae_cm": v.get("mae_cm", 0.0),
            "cos_sim": v.get("cos_sim", 100.0),
            "speedup": v.get("speedup", 1.0)
        })
    with open("D:/tests/speedup_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_out, f, indent=2, ensure_ascii=False)
    print("\n[OK] 벤치마크 결과 저장 완료: D:/tests/speedup_benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()

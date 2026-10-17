# -*- coding: utf-8 -*-
"""
vidio-health-trainer
실험: 전체 세트 압축(5회 반복 통짜) vs 단일 횟수 분할(1회 반복) 추론 비교
대상: D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4
"""

import os
import sys
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter

def run_experiment():
    vid_path = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    task_path = r"D:\pose_landmarker_heavy.task"

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(vid_path)
    raw_kps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            kps = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks[0]]
        else:
            kps = [[0.0, 0.0, 0.0]] * 33
        raw_kps.append(kps)
    cap.release()
    detector.close()

    raw_kps = np.array(raw_kps, dtype=np.float32) # (T, 33, 3)
    T = len(raw_kps)
    print(f"Extracted {T} frames.")

    # 1. 엉덩이 깊이 궤적 분석
    hip_y = (raw_kps[:, 23, 1] + raw_kps[:, 24, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=2.5)

    # 극대점(최하사점: y값이 클수록 아래로 깊게 앉은 상태)
    peaks, _ = find_peaks(hip_y_s, distance=40, prominence=0.02)
    print(f"Detected Repetition Nadirs (Bottom frames): {peaks} (총 {len(peaks)}회 감지)")

    engine = Stage2InferenceEngine.get_instance()

    # Case A: 전체 세트(531프레임 통째로 64F 압축) - 기존 벤치마크 방식
    kps_full_64f = SkeletalTopologyAdapter.resample_to_64_frames(raw_kps)
    res_full = engine.diagnose(exercise="스쿼트", keypoints_seq=kps_full_64f, camera_view="시점 B")
    print("\n[Case A: 전체 세트(5회 반복 통짜) 64F 압축 추론 결과]")
    for d in res_full["defects"]:
        print(f"  - {d['defect_code']} ({d['defect_name']}): prob={d['probability']*100:.1f}%, state={d['state_label']}")

    # Case B: 단일 1회 반복 분할 (예: 1회차 동작 구간 [0 ~ peaks[0] + (peaks[1]-peaks[0])//2])
    # 각 반복의 시작과 끝 추정
    # valleys (기립 정점: y값이 작을수록 일어선 상태)
    valleys, _ = find_peaks(-hip_y_s, distance=40, prominence=0.02)
    print(f"Detected Standing valleys (top frames): {valleys}")

    if len(valleys) >= 2 and len(peaks) >= 1:
        # 1회차: valleys[0] ~ valleys[1]
        rep1_start = max(0, valleys[0] - 10)
        rep1_end = min(T, valleys[1] + 5)
        rep1_kps = raw_kps[rep1_start:rep1_end]
        print(f"\nExtracted Repetition 1: frames {rep1_start} ~ {rep1_end} ({len(rep1_kps)} frames)")

        kps_rep1_64f = SkeletalTopologyAdapter.resample_to_64_frames(rep1_kps)
        res_rep1 = engine.diagnose(exercise="스쿼트", keypoints_seq=kps_rep1_64f, camera_view="시점 B")
        print("\n[Case B: 단일 1회차(1 Repetition) 정상 분할 64F 추론 결과]")
        for d in res_rep1["defects"]:
            print(f"  - {d['defect_code']} ({d['defect_name']}): prob={d['probability']*100:.1f}%, state={d['state_label']}")

if __name__ == "__main__":
    run_experiment()

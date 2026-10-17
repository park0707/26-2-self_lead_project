# -*- coding: utf-8 -*-
"""
vidio-health-trainer
[Test 1] 방법 1: 등방성 픽셀 좌표계 통일 (Isotropic Normalization) 테스트 스크립트
개념:
  - 영상의 가로(W), 세로(H) 비등방성 왜곡을 제거하고 1:1 정사각형 픽셀 비율(Isotropic)로 정규화
  - 원본 픽셀 좌표 (X_px, Y_px)를 영상 높이(H) 기준으로 통일 정규화하여 기하학적 종횡비 왜곡 원천 차단
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter

def apply_isotropic_normalization(kps_64f: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """
    방법 1: 등방성 픽셀 좌표계 통일 변환
    - x좌표의 스케일을 높이 H 기준으로 통일: x_iso = x * (W / H)
    - 인체 중심을 기준으로 중심 정렬하여 시야각 왜곡 방지
    """
    kps_corr = kps_64f.copy()
    ar = float(orig_w) / float(orig_h) # 예: 1080 / 1920 = 0.5625
    
    # 3D 월드 좌표계 또는 정규화 좌표계의 x축에 등방성 보정
    # AI-Hub 기준 종횡비 16:9 (1.7778) 대비 스마트폰 9:16 (0.5625) 보정
    # scale factor: sqrt(ar_phone / ar_ai)
    scale_x = np.sqrt(ar / (16.0 / 9.0)) # ~ 0.5625
    
    x_center = np.mean(kps_corr[:, :, 0])
    kps_corr[:, :, 0] = (kps_corr[:, :, 0] - x_center) * scale_x + x_center
    return kps_corr

def test_single_clip(vid_path: str, exercise_id: int, camera_view: str):
    pipeline = VideoAnalysisPipeline.get_instance()
    stage2 = Stage2InferenceEngine.get_instance()
    
    vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
    orig_w = vmeta.get("orig_w", 1080)
    orig_h = vmeta.get("orig_h", 1920)
    kps_15 = vmeta["kps_15_2d"]
    fps = vmeta["fps"]
    kps_33 = vmeta["kps_33_3d"]
    
    act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
    if not raw_reps:
        raw_reps = [(0, len(kps_15) - 1, len(kps_15) // 2)]
        
    print(f"\n[Test 1] Video: {os.path.basename(vid_path)} (W={orig_w}, H={orig_h}, Reps={len(raw_reps)})")
    
    results = []
    for r_idx, (sf, ef, bf) in enumerate(raw_reps, start=1):
        rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
        rep_64f_base = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
        rep_64f_iso = apply_isotropic_normalization(rep_64f_base, orig_w, orig_h)
        
        diag_base = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_base, camera_view=camera_view)
        diag_test1 = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_iso, camera_view=camera_view)
        
        results.append({
            "rep_index": r_idx,
            "baseline": diag_base["defects"],
            "test1": diag_test1["defects"]
        })
        
        print(f"  --- Repetition {r_idx} ---")
        for db, dt in zip(diag_base["defects"], diag_test1["defects"]):
            diff = dt["probability"] - db["probability"]
            sign = "+" if diff >= 0 else ""
            print(f"    [{db['defect_code']}] Base: {db['probability']*100:5.1f}% ({db['state_label']}) -> Test1: {dt['probability']*100:5.1f}% ({dt['state_label']}) ({sign}{diff*100:4.1f}%p)")
            
    return results

if __name__ == "__main__":
    sample_vid = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    test_single_clip(sample_vid, exercise_id=0, camera_view="시점 B")

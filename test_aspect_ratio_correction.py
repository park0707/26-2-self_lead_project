# -*- coding: utf-8 -*-
"""
vidio-health-trainer
카메라 종횡비(Aspect Ratio) 역보정 A/B 테스트 하네스 스크립트
대상: 스쿼트 클립 1 (01_세트1_정상_전체세트5회_F0380_F0910.mp4)
비교:
  - Baseline (No correction): 현재 방식 (세로 9:16 정규화 그대로 입력)
  - Method A (X-axis compression by 1/3.1605): 가로축을 AI-Hub 16:9 비율에 맞춰 압축
  - Method B (Y-axis expansion by 3.1605): 세로축을 AI-Hub 비율에 맞춰 확장
  - Method C (Isotropic Bounding Box Normalization): 인체 바운딩박스/키 기준 등방성 정규화
"""

import os
import sys
import json
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter

def test_aspect_ratio_correction():
    vid_path = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    print(f"Target Video: {vid_path}")
    
    pipeline = VideoAnalysisPipeline.get_instance()
    stage2 = Stage2InferenceEngine.get_instance()
    
    # 1. 랜드마크 추출
    vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
    kps_15 = vmeta["kps_15_2d"]
    fps = vmeta["fps"]
    kps_33 = vmeta["kps_33_3d"] # (T, 33, 3)
    
    act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
    print(f"Detected {len(raw_reps)} reps: {raw_reps}")
    
    # 종횡비 비율 계수
    # W_phone = 1080, H_phone = 1920 (AR = 0.5625)
    # W_ai = 1920, H_ai = 1080 (AR = 1.7778)
    # scale_factor = (1920/1080) / (1080/1920) = (16/9)^2 = 3.16049
    AR_phone = 1080.0 / 1920.0
    AR_ai = 1920.0 / 1080.0
    ratio_gamma = AR_ai / AR_phone # 3.16049
    
    # 첫 2개 반복에 대해 테스트
    for r_idx, (sf, ef, bf) in enumerate(raw_reps[:2], start=1):
        rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
        rep_64f_base = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
        
        print(f"\n=======================================================")
        print(f" [Repetition {r_idx}] 구간: {sf}~{ef} 프레임")
        print(f"=======================================================")
        
        # --- 1. Baseline (No correction) ---
        diag_base = stage2.diagnose(exercise=0, keypoints_seq=rep_64f_base, camera_view="시점 B")
        print("[1. Baseline: 현재 상태 (보정 없음)]")
        for d in diag_base["defects"]:
            print(f"   - {d['defect_code']:3s} ({d['defect_name'][:10]}): prob={d['probability']*100:5.1f}% | state={d['state']} ({d['state_label']})")
            
        # --- 2. Method A: X-axis compression (x_new = (x - x_mid) / ratio_gamma + x_mid) ---
        rep_64f_mA = rep_64f_base.copy()
        x_mid = np.mean(rep_64f_mA[:, :, 0])
        rep_64f_mA[:, :, 0] = (rep_64f_mA[:, :, 0] - x_mid) / np.sqrt(ratio_gamma) + x_mid
        diag_mA = stage2.diagnose(exercise=0, keypoints_seq=rep_64f_mA, camera_view="시점 B")
        print("\n[2. Method A: X축 sqrt(3.16)=1.78배 압축 (가로 비율 복원)]")
        for d in diag_mA["defects"]:
            print(f"   - {d['defect_code']:3s} ({d['defect_name'][:10]}): prob={d['probability']*100:5.1f}% | state={d['state']} ({d['state_label']})")
            
        # --- 3. Method B: Full X-axis compression (x_new = (x - x_mid) / 3.1605 + x_mid) ---
        rep_64f_mB = rep_64f_base.copy()
        rep_64f_mB[:, :, 0] = (rep_64f_mB[:, :, 0] - x_mid) / ratio_gamma + x_mid
        diag_mB = stage2.diagnose(exercise=0, keypoints_seq=rep_64f_mB, camera_view="시점 B")
        print("\n[3. Method B: X축 3.16배 전면 압축]")
        for d in diag_mB["defects"]:
            print(f"   - {d['defect_code']:3s} ({d['defect_name'][:10]}): prob={d['probability']*100:5.1f}% | state={d['state']} ({d['state_label']})")

        # --- 4. Method C: Geometric Angle Re-scaling in Feature Space ---
        # AI-Hub 픽셀 매핑: X_pixel = x * 1080, Y_pixel = y * 1920
        # AI-Hub 화면에 올리면: x_ai = X_pixel / 1920 = x * (1080/1920) = x * 0.5625
        #                      y_ai = Y_pixel / 1080 = y * (1920/1080) = y * 1.7778
        # 하지만 [0, 1] 범위 유지를 위해 인체 중심 기준 스케일링:
        rep_64f_mC = rep_64f_base.copy()
        x_c = np.mean(rep_64f_mC[:, :, 0])
        y_c = np.mean(rep_64f_mC[:, :, 1])
        # x축 스케일: (1080/1920) = 0.5625
        rep_64f_mC[:, :, 0] = (rep_64f_mC[:, :, 0] - x_c) * (1080.0 / 1920.0) + x_c
        diag_mC = stage2.diagnose(exercise=0, keypoints_seq=rep_64f_mC, camera_view="시점 B")
        print("\n[4. Method C: 픽셀 가상투영 (X축 * 0.5625 = 1/1.778배 축소)]")
        for d in diag_mC["defects"]:
            print(f"   - {d['defect_code']:3s} ({d['defect_name'][:10]}): prob={d['probability']*100:5.1f}% | state={d['state']} ({d['state_label']})")

if __name__ == "__main__":
    test_aspect_ratio_correction()

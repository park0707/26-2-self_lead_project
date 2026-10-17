# -*- coding: utf-8 -*-
"""
vidio-health-trainer
[Test 3] 방법 3: 기립 구간 카메라 앙각 가상 수평 정렬 (Virtual Leveling via Gravity Alignment) 테스트 스크립트
개념:
  - 스마트폰을 바닥에 두고 위를 올려다보는 앙각(Tilt-up 20~30도)으로 인한 발뒤꿈치 들림(c4) 및 고개 숙임(c2) 착시 원천 제거
  - 운동 시작 전 기립 구간(Stand Phase: 0~8프레임)에서 중력 수직축(발목 -> 골반)을 측정하여 카메라 피치 앙각(theta_tilt) 추정
  - 3차원 회전 행렬 R_x(-theta_tilt)을 적용하여 '삼각대를 수평(Pitch 0도)으로 놓고 촬영한 상태'로 가상 정렬
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

MP_L_HIP = 23
MP_R_HIP = 24
MP_L_ANKLE = 27
MP_R_ANKLE = 28
MP_L_HEEL = 29
MP_R_HEEL = 30
MP_L_TOE = 31
MP_R_TOE = 32

def apply_virtual_leveling(kps_64f: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    방법 3: 기립 구간 카메라 앙각 자동 수평 정렬
    - 초기 기립 구간(0~8프레임)에서 발목 -> 골반 수직 벡터의 전후(Z-Y) 기울기 앙각 추정
    - R_x(-theta_tilt) 회전 적용
    """
    T, J, C = kps_64f.shape
    kps_leveled = kps_64f.copy()
    
    # 1. 기립 구간 발목 및 골반 중심 측정
    stand_ank_l = np.median(kps_64f[:8, MP_L_ANKLE], axis=0)
    stand_ank_r = np.median(kps_64f[:8, MP_R_ANKLE], axis=0)
    stand_hip_l = np.median(kps_64f[:8, MP_L_HIP], axis=0)
    stand_hip_r = np.median(kps_64f[:8, MP_R_HIP], axis=0)
    
    ank_mid = (stand_ank_l + stand_ank_r) / 2.0
    hip_mid = (stand_hip_l + stand_hip_r) / 2.0
    
    # 수직 기립 벡터 V (발목 -> 골반)
    # MediaPipe 월드 좌표계: Y는 아래쪽, Z는 전방(카메라 쪽이 음수 또는 양수)
    # 기립 시 Y는 음의 방향(골반이 발목보다 위쪽), Z는 전후
    v_stand = hip_mid - ank_mid
    
    # 카메라 시선 광축(Z)과 수직축(Y) 사이의 전후 앙각 theta_tilt (라디안)
    # 수평 촬영일 경우 v_stand의 z성분은 0에 가까워야 함
    # 위로 올려다보고 있으면(Tilt-up), z축 기울기가 발생
    theta_tilt = np.arctan2(v_stand[2], -v_stand[1]) # 음수 Y가 위쪽
    
    # 각도가 너무 극단적(생체역학적 범위를 벗어난 경우) 클리핑 (-35도 ~ +35도)
    theta_tilt = np.clip(theta_tilt, np.radians(-35.0), np.radians(35.0))
    theta_deg = float(np.degrees(theta_tilt))
    
    # 2. X축 중심 3D 회전 행렬 R_x(-theta_tilt) (수평 정렬)
    cos_t = np.cos(-theta_tilt)
    sin_t = np.sin(-theta_tilt)
    R_x = np.array([
        [1.0,   0.0,    0.0],
        [0.0, cos_t, -sin_t],
        [0.0, sin_t,  cos_t]
    ], dtype=np.float32)
    
    # 3. 발목 중심 기준 회전 적용
    for t in range(T):
        pts_rel = kps_64f[t] - ank_mid
        kps_leveled[t] = np.dot(pts_rel, R_x.T) + ank_mid
        
    return kps_leveled, theta_deg

def test_single_clip(vid_path: str, exercise_id: int, camera_view: str):
    pipeline = VideoAnalysisPipeline.get_instance()
    stage2 = Stage2InferenceEngine.get_instance()
    
    vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
    kps_15 = vmeta["kps_15_2d"]
    fps = vmeta["fps"]
    kps_33 = vmeta["kps_33_3d"]
    
    act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
    if not raw_reps:
        raw_reps = [(0, len(kps_15) - 1, len(kps_15) // 2)]
        
    print(f"\n[Test 3] Video: {os.path.basename(vid_path)} (Reps={len(raw_reps)})")
    
    results = []
    for r_idx, (sf, ef, bf) in enumerate(raw_reps, start=1):
        rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
        rep_64f_base = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
        rep_64f_level, est_angle = apply_virtual_leveling(rep_64f_base)
        
        diag_base = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_base, camera_view=camera_view)
        diag_test3 = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_level, camera_view=camera_view)
        
        results.append({
            "rep_index": r_idx,
            "estimated_tilt_deg": est_angle,
            "baseline": diag_base["defects"],
            "test3": diag_test3["defects"]
        })
        
        print(f"  --- Repetition {r_idx} (추정 카메라 앙각: {est_angle:+.1f}도) ---")
        for db, dt in zip(diag_base["defects"], diag_test3["defects"]):
            diff = dt["probability"] - db["probability"]
            sign = "+" if diff >= 0 else ""
            print(f"    [{db['defect_code']}] Base: {db['probability']*100:5.1f}% ({db['state_label']}) -> Test3: {dt['probability']*100:5.1f}% ({dt['state_label']}) ({sign}{diff*100:4.1f}%p)")
            
    return results

if __name__ == "__main__":
    sample_vid = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    test_single_clip(sample_vid, exercise_id=0, camera_view="시점 B")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
[Test 2] 방법 2: 인체 로컬 골격 좌표계 정규화 (Body-Centric Normalization) 테스트 스크립트
개념:
  - 카메라의 위치, 거리, 앙각, 종횡비에 100% 무관하도록 인체 골격 자체를 기준 직교 좌표계로 변환
  - 원점: 양 골반 중심 (Pelvis / Hips Center)
  - Y축: 골반 중심 -> 어깨 중심 벡터 (척추 체간 축)
  - X축: 좌측 골반 -> 우측 골반 벡터 (골반 횡축, Gram-Schmidt 직교화)
  - Z축: X축과 Y축의 외적 (신체 정면 법선 벡터)
  - 스케일: 체간 길이(Torso Length)를 1.0으로 정규화한 뒤 AI-Hub 표준 스케일(0.5m)로 복원
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

# MediaPipe 랜드마크 인덱스
MP_L_SHOULDER = 11
MP_R_SHOULDER = 12
MP_L_HIP = 23
MP_R_HIP = 24
MP_L_ANKLE = 27
MP_R_ANKLE = 28

def apply_body_centric_normalization(kps_64f: np.ndarray, target_torso_len: float = 0.50) -> np.ndarray:
    """
    방법 2: 인체 로컬 골격 좌표계 정규화
    - (64, 33, 3) 시퀀스에 대해 프레임별 로컬 회전 및 체간 스케일 정규화
    """
    T, J, C = kps_64f.shape
    kps_corr = np.zeros_like(kps_64f)
    
    # 기립 상태(Stand Slice: 0~8프레임)에서 전역 신체 좌표계 정의하여 프레임간 떨림 방지
    stand_l_hip = np.mean(kps_64f[:8, MP_L_HIP], axis=0)
    stand_r_hip = np.mean(kps_64f[:8, MP_R_HIP], axis=0)
    stand_l_sh = np.mean(kps_64f[:8, MP_L_SHOULDER], axis=0)
    stand_r_sh = np.mean(kps_64f[:8, MP_R_SHOULDER], axis=0)
    
    origin = (stand_l_hip + stand_r_hip) / 2.0
    sh_mid = (stand_l_sh + stand_r_sh) / 2.0
    
    # Y축: 골반 -> 어깨 (체간 방향)
    v_y = sh_mid - origin
    torso_len = np.linalg.norm(v_y) + 1e-7
    u_y = v_y / torso_len
    
    # X축: 좌측 골반 -> 우측 골반
    v_x = stand_r_hip - stand_l_hip
    # Gram-Schmidt 직교화 (Y축과 수직인 성분만 추출)
    v_x_ortho = v_x - np.dot(v_x, u_y) * u_y
    u_x = v_x_ortho / (np.linalg.norm(v_x_ortho) + 1e-7)
    
    # Z축: 외적 (정면 전방)
    u_z = np.cross(u_x, u_y)
    u_z = u_z / (np.linalg.norm(u_z) + 1e-7)
    
    # 로컬 직교 변환 행렬 R = [u_x, u_y, u_z]^T
    R_body = np.stack([u_x, u_y, u_z], axis=0) # (3, 3)
    
    # 스케일 보정 계수: 현재 체간 길이를 AI-Hub 표준(0.50m)으로 리스케일링
    scale_factor = target_torso_len / torso_len
    
    # 모든 프레임 및 관절에 일괄 적용
    for t in range(T):
        # 골반 원점 기준 상대 좌표
        pts_rel = kps_64f[t] - origin
        # 회전 변환 및 스케일링
        pts_rot = np.dot(pts_rel, R_body.T) * scale_factor
        # 모델 입력 좌표계(Y축이 아래로 증가하는 이미지 관례 보정: Y 반전 복원)
        # MediaPipe 월드 좌표계는 Y가 아래쪽이므로 방향 정합 유지
        kps_corr[t] = pts_rot + origin
        
    return kps_corr

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
        
    print(f"\n[Test 2] Video: {os.path.basename(vid_path)} (Reps={len(raw_reps)})")
    
    results = []
    for r_idx, (sf, ef, bf) in enumerate(raw_reps, start=1):
        rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
        rep_64f_base = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
        rep_64f_body = apply_body_centric_normalization(rep_64f_base)
        
        diag_base = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_base, camera_view=camera_view)
        diag_test2 = stage2.diagnose(exercise=exercise_id, keypoints_seq=rep_64f_body, camera_view=camera_view)
        
        results.append({
            "rep_index": r_idx,
            "baseline": diag_base["defects"],
            "test2": diag_test2["defects"]
        })
        
        print(f"  --- Repetition {r_idx} ---")
        for db, dt in zip(diag_base["defects"], diag_test2["defects"]):
            diff = dt["probability"] - db["probability"]
            sign = "+" if diff >= 0 else ""
            print(f"    [{db['defect_code']}] Base: {db['probability']*100:5.1f}% ({db['state_label']}) -> Test2: {dt['probability']*100:5.1f}% ({dt['state_label']}) ({sign}{diff*100:4.1f}%p)")
            
    return results

if __name__ == "__main__":
    sample_vid = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    test_single_clip(sample_vid, exercise_id=0, camera_view="시점 B")

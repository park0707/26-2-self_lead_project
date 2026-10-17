# -*- coding: utf-8 -*-
"""
vidio-health-trainer
c4 (발바닥 지면 고정) 25D 피처값 정밀 역추적 디버거
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.models.second_models.babel_squat.c4_model_v5_1 import extract_c4_v5_features_from_seq, FEATURE_NAMES_25_C4
from src.models.second_models.babel_squat.c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP

def debug_c4():
    vid_path = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"
    pipeline = VideoAnalysisPipeline.get_instance()
    vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
    kps_15 = vmeta["kps_15_2d"]
    fps = vmeta["fps"]
    kps_33 = vmeta["kps_33_3d"]
    
    act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
    sf, ef, bf = raw_reps[0] # 1회차
    rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
    rep_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
    
    # 25D 피처 추출
    feat_25 = extract_c4_v5_features_from_seq(rep_64f)
    
    c4_model = BabelSquatC4ModelV6RTMP()
    prob = c4_model.predict_sequence(rep_64f, view_name='view2')
    
    print(f"\n[스쿼트 01번 1회차 c4 진단 결과]")
    print(f"Prob: {prob['calibrated_probability']*100:.1f}%, State: {prob['state_code']} ({prob['state_label']})")
    print("\n[c4 25D 세부 피처값 목록]")
    print("-" * 65)
    for i, (name, val) in enumerate(zip(FEATURE_NAMES_25_C4, feat_25)):
        print(f"[{i:02d}] {name:38s}: {val:10.4f}")

if __name__ == "__main__":
    debug_c4()

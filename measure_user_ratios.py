# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline

pipeline = VideoAnalysisPipeline.get_instance()
vid_path = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"

vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
kps = vmeta["kps_33_3d"]

# 기립 상태 (첫 10프레임 평균)
kps_stand = np.mean(kps[:10], axis=0)

sh_w = np.linalg.norm(kps_stand[11] - kps_stand[12])
thigh_l = np.linalg.norm(kps_stand[23] - kps_stand[25])
thigh_r = np.linalg.norm(kps_stand[24] - kps_stand[26])
thigh = (thigh_l + thigh_r) / 2.0
torso = np.linalg.norm((kps_stand[11] + kps_stand[12])/2.0 - (kps_stand[23] + kps_stand[24])/2.0)

print(f"User Smartphone Body Metrics (Stand Phase):")
print(f"  - Shoulder Width : {sh_w:.4f} m (약 {sh_w*100:.1f} cm)")
print(f"  - Thigh Length   : {thigh:.4f} m (약 {thigh*100:.1f} cm)")
print(f"  - Torso Length   : {torso:.4f} m (약 {torso*100:.1f} cm)")
print(f"  - Shoulder / Thigh Ratio : {sh_w / thigh:.4f}")
print(f"  - Torso / Thigh Ratio    : {torso / thigh:.4f}")

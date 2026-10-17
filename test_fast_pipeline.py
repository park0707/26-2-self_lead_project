# -*- coding: utf-8 -*-
import time
import numpy as np
from src.preprocessing.rep_segmenter import RepetitionSegmenter
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.pipeline.video_processor import VideoAnalysisPipeline

pipeline = VideoAnalysisPipeline.get_instance()
vid_path = r"D:\data\일반화\분할_결과\스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4"

t0 = time.time()
# 1. MediaPipe extraction (stride 2)
vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
t1 = time.time()
print(f"Landmark extraction: {t1-t0:.2f}s ({vmeta['total_frames']} frames)")

# 2. Dynamic rep segmentation
kps_15 = vmeta['kps_15_2d']
fps = vmeta['fps']
act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
t2 = time.time()
print(f"Segmentation: {t2-t1:.4f}s ({len(raw_reps)} reps detected: {raw_reps})")

# 3. Stage 2 diagnose per repetition
engine = Stage2InferenceEngine.get_instance()
kps_33 = vmeta['kps_33_3d']
for i, (sf, ef, bf) in enumerate(raw_reps, 1):
    rep_seq = kps_33[sf:ef+1]
    rep_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
    diag = engine.diagnose(exercise="스쿼트", keypoints_seq=rep_64f, camera_view="시점 B")
    defect_strs = [f"{d['defect_code']}:{d['probability']*100:.1f}%({d['state_label']})" for d in diag['defects']]
    print(f"Rep {i}: is_clean={diag['is_clean']}, defects={defect_strs}")
t3 = time.time()
print(f"Diagnosis: {t3-t2:.4f}s")
print(f"TOTAL TIME: {t3-t0:.2f}s")

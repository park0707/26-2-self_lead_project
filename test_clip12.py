# -*- coding: utf-8 -*-
import time
from src.preprocessing.rep_segmenter import RepetitionSegmenter
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.pipeline.video_processor import VideoAnalysisPipeline

pipeline = VideoAnalysisPipeline.get_instance()
vid_path = r"D:\data\일반화\분할_결과\바벨 데드\우후방\06_결함5_전체세트3회_F1950_F2340.mp4"

t0 = time.time()
vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
print(f"Landmarks: {time.time()-t0:.2f}s")

t1 = time.time()
kps_15 = vmeta['kps_15_2d']
fps = vmeta['fps']
act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
print(f"Segmented {len(raw_reps)} reps in {time.time()-t1:.4f}s: {raw_reps}")

t2 = time.time()
stage2 = Stage2InferenceEngine.get_instance()
kps_33 = vmeta['kps_33_3d']
for i, (sf, ef, bf) in enumerate(raw_reps, 1):
    rep_seq = kps_33[sf:max(sf+4, ef+1)]
    rep_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
    diag = stage2.diagnose(exercise=1, keypoints_seq=rep_64f, camera_view="시점 A")
    print(f"  Rep {i} diag: is_clean={diag['is_clean']}")
print(f"Diagnosed in {time.time()-t2:.2f}s")

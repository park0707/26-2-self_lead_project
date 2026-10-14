import os
import sys
sys.path.insert(0, r"d:")
import time
import cv2
import numpy as np
from src.web.pipeline.rtmpose_tracker import RTMPoseTracker

def benchmark_zero_redecode():
    video_path = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
    tracker = RTMPoseTracker()

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # Pre-decode first 60 frames and resize to 540p
    scale = min(540.0 / orig_w, 960.0 / orig_h)
    sw, sh = int(orig_w * scale), int(orig_h * scale)
    for _ in range(60):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (sw, sh)))
    cap.release()

    dummy_mp_2d = np.zeros((60, 33, 2), dtype=np.float32)
    dummy_mp_2d[:, 11] = [orig_w*0.3, orig_h*0.2]
    dummy_mp_2d[:, 12] = [orig_w*0.7, orig_h*0.2]
    dummy_mp_2d[:, 27] = [orig_w*0.3, orig_h*0.8]
    dummy_mp_2d[:, 28] = [orig_w*0.7, orig_h*0.8]

    # Warmup
    tracker._get_detector()

    # Test 1: Disk-based VideoCapture read with cap.set
    t0 = time.perf_counter()
    kps_disk = tracker.extract_active_period_rtmpose(
        video_path=video_path,
        active_start_frame=10,
        active_end_frame=50,
        mp_kps_33_2d=dummy_mp_2d,
        sample_stride=1
    )
    t_disk = time.perf_counter() - t0
    print(f"1. Disk read with cap.set (41 frames): {t_disk:.3f}s ({41/t_disk:.1f} FPS)")

    # Test 2: In-memory cached frames
    t0 = time.perf_counter()
    # Direct inference on memory frames
    det = tracker._get_detector()
    kps_mem = []
    smooth_box = None
    for idx, f in enumerate(frames[10:51]):
        mp_2d_cur = dummy_mp_2d[10 + idx]
        raw_bbox = tracker._compute_bbox_from_mp_landmarks(mp_2d_cur, orig_w, orig_h, padding=1.25)
        raw_bbox_scaled = raw_bbox * scale
        if smooth_box is None:
            smooth_box = raw_bbox_scaled
        else:
            smooth_box = 0.75 * smooth_box + 0.25 * raw_bbox_scaled
        kpts_list, scores_list = det.pose_model(f, bboxes=np.array([smooth_box], dtype=np.float32))
        kps_mem.append(kpts_list[0])
    t_mem = time.perf_counter() - t0
    print(f"2. Zero-Redecode Memory Buffer (41 frames): {t_mem:.3f}s ({41/t_mem:.1f} FPS)")
    print(f"Latency reduction from disk I/O elimination: {t_disk - t_mem:.3f}s saved!")

if __name__ == '__main__':
    benchmark_zero_redecode()

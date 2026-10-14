import os
import time
import numpy as np
import cv2
from rtmlib import BodyWithFeet

def test_rtmpose_comparison():
    video_path = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
    if not os.path.exists(video_path):
        print("Video not found")
        return

    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = min(540.0 / w, 960.0 / h)
        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(frame_resized)
    cap.release()

    # MediaPipe pose to get realistic bbox
    from rtmlib import BodyWithFeet
    tracker_m = BodyWithFeet(mode='balanced', backend='onnxruntime', device='cpu')
    bboxes = tracker_m.det_model(frames[0]) # Get realistic person bbox from frame 0
    print("Detected person bbox:", bboxes)
    if len(bboxes) == 0:
        bboxes = np.array([[50, 20, frames[0].shape[1]-50, frames[0].shape[0]-20]], dtype=np.float32)

    dummy_bbox = bboxes[:1]
    # Warmup
    tracker_m.pose_model(frames[0], bboxes=dummy_bbox)

    t0 = time.perf_counter()
    kpts_m_list = []
    for f in frames:
        kpts, scores = tracker_m.pose_model(f, bboxes=dummy_bbox)
        kpts_m_list.append(kpts[0])
    time_m = time.perf_counter() - t0
    fps_m = len(frames) / time_m
    print(f"Balanced: {time_m:.3f}s ({fps_m:.1f} FPS, {time_m*1000/len(frames):.1f} ms/frame)")

    # 2. Lightweight (RTMPose-s)
    print("Testing Lightweight (rtmpose-s)...")
    tracker_s = BodyWithFeet(mode='lightweight', backend='onnxruntime', device='cpu')
    # Warmup
    tracker_s.pose_model(frames[0], bboxes=dummy_bbox)

    t0 = time.perf_counter()
    kpts_s_list = []
    for f in frames:
        kpts, scores = tracker_s.pose_model(f, bboxes=dummy_bbox)
        kpts_s_list.append(kpts[0])
    time_s = time.perf_counter() - t0
    fps_s = len(frames) / time_s
    print(f"Lightweight: {time_s:.3f}s ({fps_s:.1f} FPS, {time_s*1000/len(frames):.1f} ms/frame)")

    # Compare keypoint distance (MAE in pixels)
    arr_m = np.array(kpts_m_list) # (50, 26, 2)
    arr_s = np.array(kpts_s_list) # (50, 26, 2)
    diff = np.abs(arr_m - arr_s)
    mae_px = np.mean(diff)
    print(f"Mean Absolute Difference between m and s: {mae_px:.2f} pixels on 540p frame!")
    print(f"Speedup: {fps_s / fps_m:.2f}x faster!")

if __name__ == '__main__':
    test_rtmpose_comparison()

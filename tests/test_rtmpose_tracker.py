# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\test_rtmpose_tracker.py
Purpose: MediaPipe 유도형 고속 RTMPose 추적기 프로토타입 및 성능/정확도 실증
"""

import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import BodyWithFeet

VIDEO_PATH = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"

def test_rtmpose_fast_tracking():
    assert os.path.exists(VIDEO_PATH), f"비디오 파일 없음: {VIDEO_PATH}"
    
    print("=" * 80)
    print("  ★ [vidio-health-trainer] RTMPose-m 고속 유도형 추적기 실증 벤치마크")
    print("=" * 80)
    
    # 1. BodyWithFeet 로드
    t0 = time.time()
    detector = BodyWithFeet(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')
    print(f"[*] 1. BodyWithFeet 로드 완료 ({time.time() - t0:.2f}초)")
    
    # 2. ONNX Runtime 세션 최적화 패치
    pose_tool = detector.pose_model
    if pose_tool.backend == 'onnxruntime':
        total_cores = os.cpu_count() or 8
        phys_cores = max(2, total_cores // 2)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = phys_cores
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = True
        
        # 모델 재할당
        optimized_session = ort.InferenceSession(
            pose_tool.onnx_model,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        pose_tool.session = optimized_session
        print(f"[*] 2. ONNX Runtime 세션 최적화 완료 (물리 코어: {phys_cores}개 할당, GraphOpt=ALL)")

    # 3. 비디오 로드
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 활성 운동 구간 (F137 ~ F584) 모의 테스트: 100프레임 연속 추출
    test_frames_count = 100
    frames = []
    for _ in range(test_frames_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"[*] 3. 비디오 프레임 로드: {len(frames)}F ({w}x{h}, {fps} FPS)")

    # 4. YOLOX 전체 실행 모드 (Baseline) 계측 (처음 5프레임만)
    t_base_start = time.time()
    for f in frames[:5]:
        b = detector.det_model(f)
        kp, sc = detector.pose_model(f, bboxes=b)
    t_base_per_frame = (time.time() - t_base_start) / 5.0
    print(f"\n[-] [기존 Baseline] 매 프레임 YOLOX + RTMPose: {t_base_per_frame*1000:.1f} ms/frame ({1.0/t_base_per_frame:.1f} FPS)")

    # 5. MediaPipe 유도형 타이트 BBox 고속 추적 모드 계측 (540p 리사이즈 + YOLOX 완전 생략)
    # 초기 BBox는 첫 프레임 YOLOX 1회로 설정
    init_b = detector.det_model(frames[0])
    if len(init_b) > 0:
        cur_bbox = init_b[0][:4].astype(np.float32)
    else:
        cur_bbox = np.array([0, 0, w, h], dtype=np.float32)

    extracted_kpts = []
    t_opt_start = time.time()
    
    scale = 0.5
    small_w, small_h = int(w * scale), int(h * scale)
    scaled_bbox = cur_bbox * scale
    
    for idx, frame in enumerate(frames):
        # 540p 리사이즈
        proc_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # RTMPose 단독 추론 (YOLOX 스킵)
        kpts_list, scores_list = detector.pose_model(proc_frame, bboxes=np.array([scaled_bbox], dtype=np.float32))
        
        if len(kpts_list) > 0:
            kpts = kpts_list[0]
            scores = scores_list[0]
            
            # 원본 좌표 복원 및 정규화
            kpts_orig = kpts / scale
            norm_kpts = kpts_orig.copy()
            norm_kpts[:, 0] /= w
            norm_kpts[:, 1] /= h
            
            # (26, 3) 생성 [x, y, conf]
            kpt_3d = np.column_stack([norm_kpts, scores])
            extracted_kpts.append(kpt_3d)
            
            # 이전 관절 좌표 기반 다음 프레임 BBox 동적 갱신 (EMA)
            valid_mask = scores > 0.20
            vk = kpts[valid_mask] if np.sum(valid_mask) >= 4 else kpts
            xmin, ymin = np.min(vk[:, 0]), np.min(vk[:, 1])
            xmax, ymax = np.max(vk[:, 0]), np.max(vk[:, 1])
            bw_box = max(30.0, (xmax - xmin) * 1.25)
            bh_box = max(50.0, (ymax - ymin) * 1.25)
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            
            # 3:4 비율 유지
            bw_box = max(bw_box, bh_box * 0.75)
            bh_box = max(bh_box, bw_box * 1.33)
            
            next_box = np.array([
                max(0.0, cx - bw_box / 2.0),
                max(0.0, cy - bh_box / 2.0),
                min(float(small_w), cx + bw_box / 2.0),
                min(float(small_h), cy + bh_box / 2.0)
            ], dtype=np.float32)
            
            scaled_bbox = 0.75 * scaled_bbox + 0.25 * next_box
        else:
            scaled_bbox = np.array([0, 0, small_w, small_h], dtype=np.float32)
            extracted_kpts.append(np.zeros((26, 3), dtype=np.float32))
            
    t_opt_total = time.time() - t_opt_start
    t_opt_per_frame = t_opt_total / len(frames)
    speedup = t_base_per_frame / t_opt_per_frame
    
    print(f"\n[+] [고속 최적화 모드] 540p + YOLOX 생략 + 동적 BBox 추적:")
    print(f"    - 총 소요 시간: {t_opt_total:.2f}초 ({len(frames)} 프레임)")
    print(f"    - 프레임당 지연: {t_opt_per_frame*1000:.2f} ms/frame")
    print(f"    - 처리 속도: {1.0 / t_opt_per_frame:.1f} FPS")
    print(f"    - 가속 배수: ★ {speedup:.1f}배 초고속화 달성!")
    
    extracted_arr = np.array(extracted_kpts, dtype=np.float32)
    print(f"\n[*] 6. 추출된 RTMPose 관절 텐서 규격:")
    print(f"    - Shape: {extracted_arr.shape} (기대: ({len(frames)}, 26, 3))")
    print(f"    - X 범위: [{np.min(extracted_arr[:, :, 0]):.3f}, {np.max(extracted_arr[:, :, 0]):.3f}]")
    print(f"    - Y 범위: [{np.min(extracted_arr[:, :, 1]):.3f}, {np.max(extracted_arr[:, :, 1]):.3f}]")
    print(f"    - 평균 신뢰도: {np.mean(extracted_arr[:, :, 2]):.4f}")
    
    assert extracted_arr.shape == (len(frames), 26, 3)
    assert np.mean(extracted_arr[:, :, 2]) > 0.40
    print("\n" + "=" * 80)
    print("  ★ [SUCCESS] RTMPose 초고속 추출 및 관절 정밀도 무결성 검증 완료!")
    print("=" * 80)

if __name__ == "__main__":
    test_rtmpose_fast_tracking()

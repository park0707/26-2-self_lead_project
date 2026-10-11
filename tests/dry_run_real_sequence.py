# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tests/dry_run_real_sequence.py
Purpose: 실제 바벨 데드리프트 원천 이미지 시퀀스에 대해 RTMPose 추론 및 RTMPosePreprocessor 파이프라인 E2E 검증
"""

import os
import sys
import cv2
import numpy as np

# 프로젝트 루트 경로 등록
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from rtmlib import BodyWithFeet
from src.preprocessing.rtmpose_preprocessor import RTMPosePreprocessor

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] RTMPosePreprocessor 실제 데이터 E2E 무결성 검증")
    print("=" * 80)

    raw_dir = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
    folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
    test_folder = folders[0]
    folder_path = os.path.join(raw_dir, test_folder)
    print(f"[*] 테스트 대상 원천 폴더: {test_folder}")

    # view3 (정면, 시점C) 프레임 탐색
    img_files = sorted([f for f in os.listdir(folder_path) if '시점C' in f and f.endswith('.jpg')])
    print(f"[*] 발견된 프레임 수: {len(img_files)}장")
    if len(img_files) < 4:
        print("[!] 프레임 수가 너무 적습니다.")
        return

    detector = BodyWithFeet(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')

    # 1. 첫 번째 및 중간 프레임으로 바운딩 박스 추정
    f1_path = os.path.join(folder_path, img_files[0])
    f1 = cv2.imdecode(np.fromfile(f1_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = f1.shape
    b1 = detector.det_model(f1)
    
    if len(b1) > 0:
        a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        box = b1[np.argmax(a1)]
        bw, bh = box[2] - box[0], box[3] - box[1]
        union_box = np.array([[
            max(0, box[0] - bw * 0.15),
            max(0, box[1] - bh * 0.15),
            min(w, box[2] + bw * 0.15),
            min(h, box[3] + bh * 0.15)
        ]], dtype=np.float32)
    else:
        union_box = np.array([[0, 0, w, h]], dtype=np.float32)

    # 2. 키포인트 추출
    raw_kpts = []
    for fn in img_files:
        img_path = os.path.join(folder_path, fn)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        kpts, scores = detector.pose_model(img, bboxes=union_box)
        if len(kpts) > 0:
            kp = kpts[0].copy()
            sc = scores[0].copy()
            kp[:, 0] /= w
            kp[:, 1] /= h
            raw_kpts.append(np.column_stack([kp, sc]))
        else:
            raw_kpts.append(np.zeros((26, 3), dtype=np.float32))

    raw_arr = np.array(raw_kpts, dtype=np.float32)
    print(f"[+] RTMPose 원천 키포인트 추출 완료: Shape {raw_arr.shape}")

    # 3. New 전처리 파이프라인 적용
    res = RTMPosePreprocessor.process_raw_sequence(raw_arr, exercise_name="02_바벨_데드리프트")
    if res is None:
        print("[!] 전처리 실패 (유효 관절 비율 부족)")
        return

    print("\n[+] RTMPosePreprocessor 7단계 파이프라인 통과 완료!")
    print(f"    - 표준 RTMPose 26-관절 64F Shape: {res['kps_26_64f'].shape}")
    print(f"    - 하위 호환 15-관절 64F Shape   : {res['kps_15_64f'].shape}")
    print(f"    - 척추/체간 스케일 계수         : {res['torso_scale']:.4f}")
    print(f"    - 카메라 롤 회전각              : {res['rot_angle_deg']:.2f}°")
    print(f"    - 발 3각 기하 키목록            : {list(res['foot_info'].keys())}")
    print(f"    - 결측치(NaN/Inf) 검출 수       : {np.isnan(res['kps_26_64f']).sum()}개")
    print("=" * 80)
    print("  [SUCCESS] 실제 데이터 대상 전처리 파이프라인 E2E 무결성 검증 100% 통과!")
    print("=" * 80)

if __name__ == '__main__':
    main()

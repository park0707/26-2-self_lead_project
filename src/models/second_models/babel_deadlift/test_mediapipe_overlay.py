# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_mediapipe_overlay.py
Purpose: MediaPipe 관절 추출 및 이미지 오버레이 테스트
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, 'd:/')
from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TEST_FOLDER = "[오류]_D03_피험자1_Key282"
folder_path = os.path.join(RAW_IMG_DIR, TEST_FOLDER)

print(f"[*] 테스트 폴더 확인: {folder_path}")
print(f"[*] 폴더 존재 여부: {os.path.isdir(folder_path)}")

files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
print(f"[*] 이미지 개수: {len(files)}개")

test_img_path = os.path.join(folder_path, files[0])
print(f"[*] 테스트 이미지: {files[0]}")

detector = MediaPipeAdapter.get_detector("d:/pose_landmarker_heavy.task")
res = MediaPipeAdapter.extract_frame_landmarks_full33(test_img_path, detector=detector)

kps_2d = res['kps_33_2d']
vis = res['vis_33']
print(f"[+] 추출 성공!")
print(f"  - 2D 랜드마크 Shape: {kps_2d.shape}")
print(f"  - 골반 좌(23): {kps_2d[23]}, 우(24): {kps_2d[24]}")
print(f"  - 무릎 좌(25): {kps_2d[25]}, 우(26): {kps_2d[26]}")
print(f"  - 발목 좌(27): {kps_2d[27]}, 우(28): {kps_2d[28]}")

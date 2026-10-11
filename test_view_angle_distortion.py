# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

PROJECT_ROOT = 'd:/'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

seq_dir = r"D:\Data\01_raw_data\images\07_사이드_런지\[정상]_D02_피험자1_Key145"
detector = MediaPipeAdapter.get_detector()

def calc_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

views = {'A': 'view5', 'B': 'view2', 'C': 'view3', 'D': 'view4', 'E': 'view1'}
alias_map = {
    'A': '시점 A (우측 후방 45°)',
    'B': '시점 B (우측 전방 45°)',
    'C': '시점 C (완전 정면 0°)',
    'D': '시점 D (좌측 전방 45°)',
    'E': '시점 E (좌측 후방 45°)'
}

print("=" * 90)
print(" [정밀 검증] Frame 03 (오른쪽 런지 최저점) 5대 시점 2D 투영 각도 vs 3D 공간 각도 대조")
print("=" * 90)
print(f"{'시점 코드':<10} | {'시점 명칭':<24} | {'2D 투영 각도':<14} | {'3D 공간 각도':<14} | {'2D vs 3D 차이'}")
print("-" * 90)

for vl, vn in views.items():
    f_name = f"[정상]_D02_피험자1_Key145_시점{vl}_프레임03.jpg"
    img_p = os.path.join(seq_dir, f_name)
    res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
    k2d = res['kps_33_2d']
    k3d = res['kps_33_3d']
    
    r_2d = calc_angle(k2d[24], k2d[26], k2d[28])
    r_3d = calc_angle(k3d[24], k3d[26], k3d[28])
    diff = r_2d - r_3d
    
    print(f"{vn:<10} | {alias_map[vl]:<24} | {r_2d:>8.1f}°        | {r_3d:>8.1f}°        | {diff:>+8.1f}°")

print("\n" + "=" * 90)
print(" [정밀 검증] Frame 06 (왼쪽 런지 최저점) 5대 시점 2D 투영 각도 vs 3D 공간 각도 대조")
print("=" * 90)
print(f"{'시점 코드':<10} | {'시점 명칭':<24} | {'2D 투영 각도':<14} | {'3D 공간 각도':<14} | {'2D vs 3D 차이'}")
print("-" * 90)

for vl, vn in views.items():
    f_name = f"[정상]_D02_피험자1_Key145_시점{vl}_프레임06.jpg"
    img_p = os.path.join(seq_dir, f_name)
    res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
    k2d = res['kps_33_2d']
    k3d = res['kps_33_3d']
    
    l_2d = calc_angle(k2d[23], k2d[25], k2d[27])
    l_3d = calc_angle(k3d[23], k3d[25], k3d[27])
    diff = l_2d - l_3d
    
    print(f"{vn:<10} | {alias_map[vl]:<24} | {l_2d:>8.1f}°        | {l_3d:>8.1f}°        | {diff:>+8.1f}°")

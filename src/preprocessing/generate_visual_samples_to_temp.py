# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_visual_samples_to_temp.py
Purpose: data/temp 폴더에 피험자 1의 5가지 시점 프레임 위 MediaPipe 포인트/스켈레톤 오버레이 샘플 생성
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMAGE_ROOT = r"D:\Data\01_raw_data\images\07_사이드_런지"
OUT_TEMP_DIR = r"D:\Data\temp"

# MediaPipe 33 주요 뼈대 연결 쌍
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # 체간 (어깨-골반)
    (11, 13), (13, 15),                     # 왼팔
    (12, 14), (14, 16),                     # 오른팔
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # 왼다리 (고관절-무릎-발목-발)
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), # 오른다리
]

VIEW_LETTER_MAP = {
    'A': ('view5', '시점 A (우측 후방 45°)'),
    'B': ('view2', '시점 B (우측 전방 45°)'),
    'C': ('view3', '시점 C (완전 정면 0°)'),
    'D': ('view4', '시점 D (좌측 전방 45°)'),
    'E': ('view1', '시점 E (좌측 후방 45°)'),
}

def calc_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 180.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def draw_skeleton(img, kps_2d, vis, lead_knee_idx=None):
    h, w = img.shape[:2]
    vis_img = img.copy()

    # 1. 연결선 (스켈레톤 본) 그리기
    for j1, j2 in POSE_CONNECTIONS:
        v1, v2 = vis[j1], vis[j2]
        if v1 < 0.3 or v2 < 0.3:
            continue
        pt1 = (int(kps_2d[j1, 0]), int(kps_2d[j1, 1]))
        pt2 = (int(kps_2d[j2, 0]), int(kps_2d[j2, 1]))

        # 다리 색상 구분 (좌다리: 청록/파랑, 우다리: 주황/빨강, 체간: 초록)
        if j1 in [23, 25, 27, 29, 31] and j2 in [23, 25, 27, 29, 31]:
            color = (255, 180, 0) # BGR (Cyan/Blue)
        elif j1 in [24, 26, 28, 30, 32] and j2 in [24, 26, 28, 30, 32]:
            color = (0, 120, 255) # BGR (Orange/Red)
        else:
            color = (0, 255, 120) # BGR (Green)
        cv2.line(vis_img, pt1, pt2, color, 3, cv2.LINE_AA)

    # 2. 관절 포인트(랜드마크 점) 그리기
    for i in range(len(kps_2d)):
        if vis[i] < 0.3:
            continue
        pt = (int(kps_2d[i, 0]), int(kps_2d[i, 1]))
        
        # 기본 관절점
        if i in [23, 25, 27]: # 왼하지
            cv2.circle(vis_img, pt, 6, (255, 200, 50), -1, cv2.LINE_AA)
            cv2.circle(vis_img, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
        elif i in [24, 26, 28]: # 오른하지
            cv2.circle(vis_img, pt, 6, (30, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_img, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(vis_img, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # 3. 앞다리(Lead Knee) 무릎 강조
    if lead_knee_idx is not None and vis[lead_knee_idx] >= 0.3:
        lead_pt = (int(kps_2d[lead_knee_idx, 0]), int(kps_2d[lead_knee_idx, 1]))
        cv2.circle(vis_img, lead_pt, 16, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(vis_img, lead_pt, 22, (0, 0, 255), 2, cv2.LINE_AA)

    return vis_img

def process_action_folder(folder_name: str, label_desc: str):
    folder_path = os.path.join(RAW_IMAGE_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"[!] 폴더 없음: {folder_path}")
        return

    all_files = sorted(os.listdir(folder_path))
    detector = MediaPipeAdapter.get_detector()
    print(f"\n[*] 처리 시작: {folder_name} (총 {len(all_files)}개 이미지)")

    for v_letter, (v_code, v_alias) in VIEW_LETTER_MAP.items():
        v_files = [f for f in all_files if f"시점{v_letter}" in f and f.endswith('.jpg')]
        if not v_files:
            continue
        v_files = sorted(v_files)

        # 1. 16개 프레임에 대해 랜드마크 추출 및 최저점(골반 최하사점) 프레임 탐색
        frame_data = []
        for f_name in v_files:
            img_path = os.path.join(folder_path, f_name)
            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_path, detector=detector)
            kps_2d = res['kps_33_2d']
            vis = res['vis_33']
            # 골반 중심 Y (23: left hip, 24: right hip)
            mid_hip_y = (kps_2d[23, 1] + kps_2d[24, 1]) / 2.0
            frame_data.append({
                'file': f_name,
                'path': img_path,
                'kps_2d': kps_2d,
                'kps_3d': res['kps_33_3d'],
                'vis': vis,
                'hip_y': mid_hip_y
            })

        # 골반 Y가 가장 큰(화면상 가장 아래로 내려간) 최저점 인덱스
        lowest_idx = max(range(len(frame_data)), key=lambda i: frame_data[i]['hip_y'])
        low_item = frame_data[lowest_idx]

        # 2. 무릎 각도 계산 (최저점)
        k2d = low_item['kps_2d']
        l_angle = calc_angle_2d(k2d[23], k2d[25], k2d[27]) # Left Knee
        r_angle = calc_angle_2d(k2d[24], k2d[26], k2d[28]) # Right Knee

        if l_angle < r_angle:
            lead_idx = 25 # Left Knee
            lead_str = f"Left Knee (Lead): {l_angle:.1f} deg"
            rear_str = f"Right Knee (Rear): {r_angle:.1f} deg"
        else:
            lead_idx = 26 # Right Knee
            lead_str = f"Right Knee (Lead): {r_angle:.1f} deg"
            rear_str = f"Left Knee (Rear): {l_angle:.1f} deg"

        # 3. 원본 이미지 로드 및 오버레이 렌더링
        img_bytes = np.fromfile(low_item['path'], dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]

        vis_img = draw_skeleton(img_bgr, k2d, low_item['vis'], lead_knee_idx=lead_idx)

        # 4. 상단 정보 배너 렌더링
        banner_h = 105
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.78, vis_img, 0.22, 0, vis_img)

        # 텍스트 출력
        title_txt = f"[{label_desc}] {v_alias} - Frame {lowest_idx+1:02d} (Lowest Inflection)"
        cv2.putText(vis_img, title_txt, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        
        stat_txt = f"{lead_str}   |   {rear_str}"
        cv2.putText(vis_img, stat_txt, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (50, 255, 120), 2, cv2.LINE_AA)

        # 5. 저장 (Windows 한글 경로 호환 imencode)
        out_fname = f"Sample_{label_desc}_시점{v_letter}_{v_code}_frame{lowest_idx+1:02d}.jpg"
        out_path = os.path.join(OUT_TEMP_DIR, out_fname)
        _, enc = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enc.tofile(out_path)
        print(f"  [+] 최저점 저장 완료: {out_fname} ({lead_str})")

        # 시점 C(정면)인 경우, 런지 동작 과정 4단계 프레임 추가 저장
        if v_letter == 'C':
            step_indices = [0, max(1, lowest_idx // 2), lowest_idx, min(len(v_files)-1, lowest_idx + 4)]
            step_names = ['01_Start_Stand', '02_Descending', '03_Bottom_Peak', '04_Ascending']
            for s_idx, s_name in zip(step_indices, step_names):
                s_item = frame_data[s_idx]
                s_bytes = np.fromfile(s_item['path'], dtype=np.uint8)
                s_bgr = cv2.imdecode(s_bytes, cv2.IMREAD_COLOR)
                s_vis = draw_skeleton(s_bgr, s_item['kps_2d'], s_item['vis'], lead_knee_idx=lead_idx)

                s_over = s_vis.copy()
                cv2.rectangle(s_over, (0, 0), (w, 85), (20, 20, 25), -1)
                cv2.addWeighted(s_over, 0.78, s_vis, 0.22, 0, s_vis)
                cv2.putText(s_vis, f"[{label_desc}] Front View (Step: {s_name}) - Frame {s_idx+1:02d}", 
                            (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 230, 80), 2, cv2.LINE_AA)
                
                s_fname = f"Sequence_{label_desc}_FrontView_{s_name}_frame{s_idx+1:02d}.jpg"
                _, s_enc = cv2.imencode('.jpg', s_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                s_enc.tofile(os.path.join(OUT_TEMP_DIR, s_fname))

def main():
    os.makedirs(OUT_TEMP_DIR, exist_ok=True)
    print(f"[*] 출력 디렉토리: {OUT_TEMP_DIR}")

    # 1. 피험자 1 정상 동작 (Key 145) 5대 시점 처리
    process_action_folder("[정상]_D02_피험자1_Key145", "정상_Key145")

    # 2. 피험자 1 결함 동작 (Key 146 - 앞다리 조금 접힘) 5대 시점 처리
    process_action_folder("[오류]_D02_피험자1_Key146", "결함_Key146_앞다리미달")

    print("\n[+] 모든 시각화 샘플 생성 완료!")
    out_files = sorted(os.listdir(OUT_TEMP_DIR))
    print(f"[+] 총 생성된 파일 수: {len(out_files)}개")
    for f in out_files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()

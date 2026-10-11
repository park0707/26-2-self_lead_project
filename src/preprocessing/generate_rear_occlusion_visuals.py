# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_rear_occlusion_visuals.py
Purpose: 후방 시점(시점 A, 시점 E)에서 사이드 런지 최하사점 시
         정강이/종아리에 의한 발끝 자기 폐색(Self-Occlusion) 및
         MediaPipe 추정 관절 포인트를 원본 이미지 위에 정밀 오버레이하여 data/temp에 저장
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
os.makedirs(OUT_TEMP_DIR, exist_ok=True)

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


def draw_occlusion_visual(img_bgr, kps_2d, vis, lead_knee_idx, lead_foot_idx, lead_heel_idx,
                          clip_desc, view_name, frame_num, is_lowest=True):
    h, w = img_bgr.shape[:2]
    vis_img = img_bgr.copy()

    # 1. 스켈레톤 본 연결선 그리기
    for j1, j2 in POSE_CONNECTIONS:
        v1, v2 = vis[j1], vis[j2]
        if v1 < 0.25 or v2 < 0.25:
            continue
        pt1 = (int(kps_2d[j1, 0]), int(kps_2d[j1, 1]))
        pt2 = (int(kps_2d[j2, 0]), int(kps_2d[j2, 1]))

        # 하지 색상 구분
        if j1 in [23, 25, 27, 29, 31] and j2 in [23, 25, 27, 29, 31]:
            color = (255, 180, 0) # Cyan/Blue (왼다리)
        elif j1 in [24, 26, 28, 30, 32] and j2 in [24, 26, 28, 30, 32]:
            color = (0, 120, 255) # Orange/Red (오른다리)
        else:
            color = (0, 255, 120) # Green (체간)
        cv2.line(vis_img, pt1, pt2, color, 3, cv2.LINE_AA)

    # 2. 일반 관절점 그리기
    for i in range(len(kps_2d)):
        if vis[i] < 0.25:
            continue
        pt = (int(kps_2d[i, 0]), int(kps_2d[i, 1]))
        if i in [23, 25, 27]: # 왼다리 관절
            cv2.circle(vis_img, pt, 6, (255, 200, 50), -1, cv2.LINE_AA)
            cv2.circle(vis_img, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
        elif i in [24, 26, 28]: # 오른다리 관절
            cv2.circle(vis_img, pt, 6, (30, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_img, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(vis_img, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # 3. 앞다리 무릎 및 발끝(Toe) 강력 강조
    # 앞다리 무릎
    lead_k_pt = (int(kps_2d[lead_knee_idx, 0]), int(kps_2d[lead_knee_idx, 1]))
    cv2.circle(vis_img, lead_k_pt, 14, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis_img, "Lead Knee", (lead_k_pt[0] + 18, lead_k_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 앞다리 뒤꿈치
    heel_pt = (int(kps_2d[lead_heel_idx, 0]), int(kps_2d[lead_heel_idx, 1]))
    cv2.circle(vis_img, heel_pt, 8, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(vis_img, "Heel", (heel_pt[0] - 50, heel_pt[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

    # 앞다리 발끝 (Foot Index / Toe) - 가림 현상 확인의 핵심 포인트
    toe_pt = (int(kps_2d[lead_foot_idx, 0]), int(kps_2d[lead_foot_idx, 1]))
    toe_vis = vis[lead_foot_idx]
    
    # 발끝 강조 원 & 십자선 (Magenta/Yellow)
    cv2.circle(vis_img, toe_pt, 14, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(vis_img, toe_pt, 20, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(vis_img, (toe_pt[0] - 25, toe_pt[1]), (toe_pt[0] + 25, toe_pt[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(vis_img, (toe_pt[0], toe_pt[1] - 25), (toe_pt[0], toe_pt[1] + 25), (0, 0, 255), 2, cv2.LINE_AA)

    toe_label = f"Toe (MP Est) Vis:{toe_vis:.2f}"
    cv2.putText(vis_img, toe_label, (toe_pt[0] + 22, toe_pt[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 뒤꿈치 -> 발끝 벡터 화살표
    cv2.arrowedLine(vis_img, heel_pt, toe_pt, (0, 0, 255), 3, tipLength=0.3)

    # 4. 좌측 하단에 발 부위 2.2배 확대 Inset (Picture-in-Picture) 렌더링
    crop_size = 130
    cx, cy = toe_pt[0], toe_pt[1]
    x1 = max(0, cx - crop_size)
    y1 = max(0, cy - crop_size)
    x2 = min(w, cx + crop_size)
    y2 = min(h, cy + crop_size)

    if (x2 - x1) > 40 and (y2 - y1) > 40:
        foot_crop = img_bgr[y1:y2, x1:x2].copy()
        # crop 위에 toe와 heel 랜드마크 찍기
        crop_toe = (int(toe_pt[0] - x1), int(toe_pt[1] - y1))
        crop_heel = (int(heel_pt[0] - x1), int(heel_pt[1] - y1))
        cv2.circle(foot_crop, crop_toe, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(foot_crop, crop_toe, 12, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(foot_crop, crop_heel, 6, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(foot_crop, crop_heel, crop_toe, (0, 0, 255), 2, tipLength=0.3)

        # 확대
        inset_w, inset_h = 320, 320
        inset = cv2.resize(foot_crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(inset, (0, 0), (inset_w, inset_h), (0, 255, 255), 3)
        cv2.putText(inset, "[Foot Crop 2.2x Inset]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(inset, "Red: Estimated Toe", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(inset, "Magenta: Heel", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

        # 오버레이 배치 (우측 하단)
        ix1 = w - inset_w - 25
        iy1 = h - inset_h - 25
        vis_img[iy1:iy1+inset_h, ix1:ix1+inset_w] = inset

    # 5. 상단 정보 배너 렌더링
    banner_h = 110
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 28), -1)
    cv2.addWeighted(overlay, 0.82, vis_img, 0.18, 0, vis_img)

    status_tag = "[최하사점: 정강이 가림 발생 구간]" if is_lowest else "[하강 진행 구간]"
    cv2.putText(vis_img, f"{status_tag} {view_name} - Frame {frame_num:02d}",
                (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(vis_img, f"Clip: {clip_desc}  |  Toe Confidence: {toe_vis*100:.1f}%",
                (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 255, 120), 2, cv2.LINE_AA)

    cv2.putText(vis_img, "Notice: Rear view shows Shank/Calf occluding Toe. MediaPipe hallucinates 3D toe position.",
                (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 200, 255), 1, cv2.LINE_AA)

    return vis_img


def process_selected_clips():
    detector = MediaPipeAdapter.get_detector()

    # 대표 분석 대상 4개 폴더 (피험자 1, 2의 정상 및 결함)
    target_folders = [
        ("[정상]_D02_피험자1_Key145", "피험자1_정상자세"),
        ("[오류]_D02_피험자1_Key147", "피험자1_발방향바깥쪽_결함"),
        ("[정상]_D02_피험자2_Key145", "피험자2_정상자세"),
        ("[오류]_D02_피험자2_Key147", "피험자2_발방향바깥쪽_결함"),
    ]

    saved_files = []

    for folder_name, desc in target_folders:
        folder_path = os.path.join(RAW_IMAGE_ROOT, folder_name)
        if not os.path.exists(folder_path):
            print(f"[!] 폴더 없음: {folder_path}")
            continue

        all_files = sorted(os.listdir(folder_path))
        print(f"\n[*] 후방 시점 분석 처리: {folder_name} ({desc})")

        for v_letter in ['A', 'E']:
            v_code, v_alias = VIEW_LETTER_MAP[v_letter]
            v_files = sorted([f for f in all_files if f"시점{v_letter}" in f and f.endswith('.jpg')])
            if not v_files:
                continue

            # 모든 프레임 랜드마크 추출 후 골반 최하사점 프레임 찾기
            frame_records = []
            for f_name in v_files:
                img_path = os.path.join(folder_path, f_name)
                res = MediaPipeAdapter.extract_frame_landmarks_full33(img_path, detector=detector)
                kps_2d = res['kps_33_2d']
                vis = res['vis_33']
                mid_hip_y = (kps_2d[23, 1] + kps_2d[24, 1]) / 2.0
                frame_records.append({
                    'file': f_name,
                    'path': img_path,
                    'kps_2d': kps_2d,
                    'vis': vis,
                    'hip_y': mid_hip_y
                })

            lowest_idx = int(np.argmax([fr['hip_y'] for fr in frame_records]))
            low_rec = frame_records[lowest_idx]

            # 앞다리 판별 (더 많이 굽혀진 쪽이 lead leg)
            k2d = low_rec['kps_2d']
            l_knee_ang = calc_angle_2d(k2d[23], k2d[25], k2d[27])
            r_knee_ang = calc_angle_2d(k2d[24], k2d[26], k2d[28])

            if l_knee_ang < r_knee_ang:
                lead_k_idx, lead_f_idx, lead_h_idx = 25, 31, 29 # 왼다리
                lead_side = "Left"
            else:
                lead_k_idx, lead_f_idx, lead_h_idx = 26, 32, 30 # 오른다리
                lead_side = "Right"

            # 원본 이미지 읽기
            img_bytes = np.fromfile(low_rec['path'], dtype=np.uint8)
            img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            vis_out = draw_occlusion_visual(
                img_bgr, k2d, low_rec['vis'],
                lead_k_idx, lead_f_idx, lead_h_idx,
                clip_desc=f"{desc} (Lead: {lead_side})",
                view_name=v_alias,
                frame_num=lowest_idx + 1,
                is_lowest=True
            )

            out_name = f"후방가림검증_{desc}_시점{v_letter}_{v_code}_최하사점_frame{lowest_idx+1:02d}.jpg"
            out_path = os.path.join(OUT_TEMP_DIR, out_name)
            _, enc = cv2.imencode('.jpg', vis_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            enc.tofile(out_path)
            saved_files.append((out_name, out_path, v_alias, desc))
            print(f"  [+] 후방 최하사점 저장 완료: {out_name}")

            # 피험자1 결함 클립의 경우, 하강 중간 프레임도 추가로 1개 저장하여 비교 관찰 지원
            if "피험자1_발방향바깥쪽" in desc and v_letter == 'A':
                mid_idx = max(0, lowest_idx - 3)
                mid_rec = frame_records[mid_idx]
                img_m_bytes = np.fromfile(mid_rec['path'], dtype=np.uint8)
                img_m_bgr = cv2.imdecode(img_m_bytes, cv2.IMREAD_COLOR)
                vis_m = draw_occlusion_visual(
                    img_m_bgr, mid_rec['kps_2d'], mid_rec['vis'],
                    lead_k_idx, lead_f_idx, lead_h_idx,
                    clip_desc=f"{desc} (하강중 비교)",
                    view_name=v_alias,
                    frame_num=mid_idx + 1,
                    is_lowest=False
                )
                m_out_name = f"후방가림검증_{desc}_시점{v_letter}_{v_code}_하강중_frame{mid_idx+1:02d}.jpg"
                m_out_path = os.path.join(OUT_TEMP_DIR, m_out_name)
                _, enc = cv2.imencode('.jpg', vis_m, [cv2.IMWRITE_JPEG_QUALITY, 95])
                enc.tofile(m_out_path)
                saved_files.append((m_out_name, m_out_path, v_alias, desc + " (하강중)"))
                print(f"  [+] 후방 하강중 프레임 저장 완료: {m_out_name}")

    print(f"\n[+] 총 {len(saved_files)}개 후방 가림 검증 이미지 생성 완료! 위치: {OUT_TEMP_DIR}")
    return saved_files


if __name__ == '__main__':
    process_selected_clips()

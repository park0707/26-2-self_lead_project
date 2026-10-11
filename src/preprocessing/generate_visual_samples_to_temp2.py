# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_visual_samples_to_temp2.py
Purpose: data/temp2 폴더에 사이드 런지 상체 젖힘/숙임 결함(c5)의 다양한 유형별 및 정상 샘플 생성
         - MediaPipe 스켈레톤 및 체간/척추 라인 하이라이트
         - 5대 공식 시점(시점 A~E) 최하사점 오버레이 이미지
         - 런지 진행 단계별 4단계 시퀀스(시작 -> 하강 -> 최저점 -> 상승)
         - 체간 기울기 각도(Torso Angle) 및 결함 진단 배너
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
OUT_TEMP2_DIR = r"D:\Data\temp2"

# MediaPipe 33 주요 뼈대 연결 쌍
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # 체간 (어깨-골반)
    (11, 13), (13, 15),                     # 왼팔
    (12, 14), (14, 16),                     # 오른팔
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # 왼다리
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), # 오른다리
]

VIEW_LETTER_MAP = {
    'A': ('view5', '시점 A (우측 후방 45°)'),
    'B': ('view2', '시점 B (우측 전방 45°)'),
    'C': ('view3', '시점 C (완전 정면 0°)'),
    'D': ('view4', '시점 D (좌측 전방 45°)'),
    'E': ('view1', '시점 E (좌측 후방 45°)'),
}

# 분석 대상 6대 대표 동작 유형
TARGET_ACTIONS = [
    {
        'folder': '[정상]_D02_피험자1_Key145',
        'sub_dir': '00_정상_Key145_상체중립',
        'key_name': 'Key145 (정상)',
        'type_desc': '표준 정상 (상체 및 척추 중립 유지)',
        'is_defect': False,
        'defect_detail': '상체 중립 각도 유지 (결함 없음)'
    },
    {
        'folder': '[오류]_D02_피험자1_Key150',
        'sub_dir': '01_결함_Key150_순수상체뒤로젖힘',
        'key_name': 'Key150 (결함)',
        'type_desc': '순수 상체 뒤로 젖힘 (Torso Hyperextension)',
        'is_defect': True,
        'defect_detail': '골반 대비 체간이 뒤로 과도하게 젖혀짐'
    },
    {
        'folder': '[오류]_D02_피험자1_Key165',
        'sub_dir': '02_결함_Key165_하늘보고상체젖힘',
        'key_name': 'Key165 (결함)',
        'type_desc': '하늘보고 상체 젖힘 (극단적 젖힘 + 경추 과신전)',
        'is_defect': True,
        'defect_detail': '시선이 위를 향하며 흉추/요추가 심하게 후방 신전됨'
    },
    {
        'folder': '[오류]_D02_피험자1_Key166',
        'sub_dir': '03_결함_Key166_바닥보고상체젖힘_숙임',
        'key_name': 'Key166 (결함)',
        'type_desc': '바닥보고 상체 젖힘 (체간 젖힘 + 경추 굴곡)',
        'is_defect': True,
        'defect_detail': '목은 숙이고 상체는 뒤로 젖혀 척추 곡선 왜곡'
    },
    {
        'folder': '[오류]_D02_피험자1_Key160',
        'sub_dir': '04_결함_Key160_몸통갸우뚱_상체젖힘',
        'key_name': 'Key160 (결함)',
        'type_desc': '몸통 옆으로 갸우뚱 + 상체 젖힘 (측면 비틀림 복합)',
        'is_defect': True,
        'defect_detail': '체간이 횡방향으로 기울면서 후방으로 동시에 무너짐'
    },
    {
        'folder': '[오류]_D02_피험자1_Key154',
        'sub_dir': '05_결함_Key154_앞다리미달_상체젖힘',
        'key_name': 'Key154 (결함)',
        'type_desc': '앞다리 조금접고 + 상체 뒤로 젖힘 (하지/상체 복합 결함)',
        'is_defect': True,
        'defect_detail': '앞다리를 충분히 낮추지 못하고 상체를 뒤로 보상'
    },
]

def calc_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 180.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def calc_torso_metrics(kps_2d, kps_3d):
    """체간(어깨 중심 ~ 골반 중심) 기울기 및 척추 각도 계산"""
    mid_sh_2d = (kps_2d[11] + kps_2d[12]) * 0.5
    mid_hip_2d = (kps_2d[23] + kps_2d[24]) * 0.5
    v_torso_2d = mid_sh_2d - mid_hip_2d
    
    # 수직 기준선(위쪽 방향: [0, -1])과의 2D 사잇각
    v_up = np.array([0.0, -1.0])
    norm_t = np.linalg.norm(v_torso_2d) + 1e-6
    cos_2d = np.dot(v_torso_2d, v_up) / norm_t
    lean_deg_2d = float(np.degrees(np.arccos(np.clip(cos_2d, -1.0, 1.0))))
    
    # 3D 체간 벡터
    mid_sh_3d = (kps_3d[11] + kps_3d[12]) * 0.5
    mid_hip_3d = (kps_3d[23] + kps_3d[24]) * 0.5
    v_torso_3d = mid_sh_3d - mid_hip_3d
    # Z-Y 평면에서의 전후 기울기 각도 (Pitch)
    # MediaPipe 3D: Y는 아래, Z는 카메라 방향(전방 음수, 후방 양수 또는 반대)
    pitch_3d = float(np.degrees(np.arctan2(v_torso_3d[2], -v_torso_3d[1] + 1e-6)))

    return {
        'mid_sh_2d': (int(mid_sh_2d[0]), int(mid_sh_2d[1])),
        'mid_hip_2d': (int(mid_hip_2d[0]), int(mid_hip_2d[1])),
        'lean_deg_2d': lean_deg_2d,
        'pitch_3d': pitch_3d
    }

def draw_skeleton_with_torso(img, kps_2d, vis, is_defect=False, lead_knee_idx=None):
    vis_img = img.copy()

    # 1. 일반 스켈레톤 본
    for j1, j2 in POSE_CONNECTIONS:
        v1, v2 = vis[j1], vis[j2]
        if v1 < 0.3 or v2 < 0.3:
            continue
        pt1 = (int(kps_2d[j1, 0]), int(kps_2d[j1, 1]))
        pt2 = (int(kps_2d[j2, 0]), int(kps_2d[j2, 1]))

        # 다리 색상
        if j1 in [23, 25, 27, 29, 31] and j2 in [23, 25, 27, 29, 31]:
            color = (255, 180, 0) # Cyan/Blue
        elif j1 in [24, 26, 28, 30, 32] and j2 in [24, 26, 28, 30, 32]:
            color = (0, 140, 255) # Orange
        else:
            color = (200, 200, 200) # Gray
        cv2.line(vis_img, pt1, pt2, color, 2, cv2.LINE_AA)

    # 2. 체간 중심선(Spine Line: 골반 중심 -> 어깨 중심) 강조 렌더링
    mid_sh = ((kps_2d[11, 0] + kps_2d[12, 0]) * 0.5, (kps_2d[11, 1] + kps_2d[12, 1]) * 0.5)
    mid_hip = ((kps_2d[23, 0] + kps_2d[24, 0]) * 0.5, (kps_2d[23, 1] + kps_2d[24, 1]) * 0.5)
    pt_sh = (int(mid_sh[0]), int(mid_sh[1]))
    pt_hip = (int(mid_hip[0]), int(mid_hip[1]))

    # 결함 여부에 따른 체간 중심선 색상: 정상은 청록/초록, 결함은 굵은 진홍/빨강
    torso_color = (40, 50, 255) if is_defect else (50, 230, 50)
    cv2.line(vis_img, pt_hip, pt_sh, (255, 255, 255), 7, cv2.LINE_AA) # 흰색 외곽선
    cv2.line(vis_img, pt_hip, pt_sh, torso_color, 4, cv2.LINE_AA)     # 주 색상선

    # 어깨 중심점과 골반 중심점 원 표시
    cv2.circle(vis_img, pt_sh, 8, torso_color, -1, cv2.LINE_AA)
    cv2.circle(vis_img, pt_sh, 11, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(vis_img, pt_hip, 8, torso_color, -1, cv2.LINE_AA)
    cv2.circle(vis_img, pt_hip, 11, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. 주요 랜드마크 점
    for i in range(len(kps_2d)):
        if vis[i] < 0.3:
            continue
        pt = (int(kps_2d[i, 0]), int(kps_2d[i, 1]))
        if i in [11, 12, 23, 24]: # 체간 코어
            cv2.circle(vis_img, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
        elif i in [25, 26]: # 무릎
            cv2.circle(vis_img, pt, 5, (255, 100, 50), -1, cv2.LINE_AA)
        else:
            cv2.circle(vis_img, pt, 3, (220, 220, 220), -1, cv2.LINE_AA)

    # 4. 앞다리 무릎 강조
    if lead_knee_idx is not None and vis[lead_knee_idx] >= 0.3:
        lead_pt = (int(kps_2d[lead_knee_idx, 0]), int(kps_2d[lead_knee_idx, 1]))
        cv2.circle(vis_img, lead_pt, 14, (0, 255, 255), 2, cv2.LINE_AA)

    return vis_img

def process_single_action(action_cfg: dict, detector):
    folder_name = action_cfg['folder']
    sub_dir_name = action_cfg['sub_dir']
    type_desc = action_cfg['type_desc']
    is_defect = action_cfg['is_defect']
    defect_detail = action_cfg['defect_detail']
    key_name = action_cfg['key_name']

    folder_path = os.path.join(RAW_IMAGE_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"[!] 폴더 없음: {folder_path}")
        return

    # 대상 저장 하위 디렉토리 생성
    target_dir = os.path.join(OUT_TEMP2_DIR, sub_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    all_files = sorted(os.listdir(folder_path))
    print(f"\n[*] 동작 처리 시작: {key_name} - {type_desc}")
    print(f"    - 저장 경로: {target_dir}")

    for v_letter, (v_code, v_alias) in VIEW_LETTER_MAP.items():
        v_files = [f for f in all_files if f"시점{v_letter}" in f and f.endswith('.jpg')]
        if not v_files:
            continue
        v_files = sorted(v_files)

        # 1. 프레임별 랜드마크 추출 및 최저점(골반 최하사점) 프레임 탐색
        frame_data = []
        for f_name in v_files:
            img_path = os.path.join(folder_path, f_name)
            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_path, detector=detector)
            kps_2d = res['kps_33_2d']
            kps_3d = res['kps_33_3d']
            vis = res['vis_33']
            mid_hip_y = (kps_2d[23, 1] + kps_2d[24, 1]) / 2.0
            frame_data.append({
                'file': f_name,
                'path': img_path,
                'kps_2d': kps_2d,
                'kps_3d': kps_3d,
                'vis': vis,
                'hip_y': mid_hip_y
            })

        # 골반 Y가 가장 큰(화면상 가장 아래) 최저점 인덱스
        lowest_idx = max(range(len(frame_data)), key=lambda i: frame_data[i]['hip_y'])
        low_item = frame_data[lowest_idx]

        # 2. 메트릭 계산 (무릎 각도 및 체간 기울기)
        k2d = low_item['kps_2d']
        k3d = low_item['kps_3d']
        l_knee_ang = calc_angle_2d(k2d[23], k2d[25], k2d[27])
        r_knee_ang = calc_angle_2d(k2d[24], k2d[26], k2d[28])
        lead_idx = 25 if l_knee_ang < r_knee_ang else 26
        lead_knee_ang = min(l_knee_ang, r_knee_ang)

        torso_info = calc_torso_metrics(k2d, k3d)
        lean_2d = torso_info['lean_deg_2d']
        pitch_3d = torso_info['pitch_3d']

        # 3. 원본 이미지 로드 및 렌더링
        img_bytes = np.fromfile(low_item['path'], dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        h, w = img_bgr.shape[:2]

        vis_img = draw_skeleton_with_torso(img_bgr, k2d, low_item['vis'], is_defect=is_defect, lead_knee_idx=lead_idx)

        # 4. 상단 정보 배너 렌더링
        banner_h = 115
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.82, vis_img, 0.18, 0, vis_img)

        # 상태 텍스트 및 색상
        tag_color = (80, 80, 255) if is_defect else (50, 230, 80)
        status_tag = "[결함: 상체 젖힘/숙임]" if is_defect else "[정상: 상체 중립]"
        title_txt = f"{status_tag} {key_name} | {v_alias} - Frame {lowest_idx+1:02d} (최하사점)"
        cv2.putText(vis_img, title_txt, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.88, tag_color, 2, cv2.LINE_AA)

        detail_txt = f"동작 유형: {type_desc}"
        cv2.putText(vis_img, detail_txt, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

        metric_txt = f"체간 기울기: {lean_2d:.1f} deg | 3D Pitch: {pitch_3d:+.1f} deg | 앞무릎각도: {lead_knee_ang:.1f} deg | {defect_detail}"
        cv2.putText(vis_img, metric_txt, (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 200), 1, cv2.LINE_AA)

        # 5. 최저점 이미지 저장 (하위 폴더 및 temp2 루트 양쪽 모두 저장하여 편의성 극대화)
        fname_sub = f"Lowest_{v_code}_{v_letter}_frame{lowest_idx+1:02d}.jpg"
        path_sub = os.path.join(target_dir, fname_sub)
        _, enc = cv2.imencode('.jpg', vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enc.tofile(path_sub)

        # 루트 경로에도 보기 쉽게 저장
        fname_root = f"{sub_dir_name}_{v_code}_frame{lowest_idx+1:02d}.jpg"
        path_root = os.path.join(OUT_TEMP2_DIR, fname_root)
        enc.tofile(path_root)
        print(f"  [+] 시점 {v_letter} ({v_alias}) 최저점 저장 완료: 체간 기울기 {lean_2d:.1f}°")

        # 시점 C(정면) 또는 시점 E(후방-상체젖힘이 가장 잘 보임)에 대해 4단계 진행 시퀀스 생성
        if v_letter in ['C', 'E']:
            step_indices = [0, max(1, lowest_idx // 2), lowest_idx, min(len(v_files)-1, lowest_idx + 4)]
            step_names = ['01_Start_준비', '02_Descending_하강', '03_Bottom_최저점', '04_Ascending_상승']
            for s_idx, s_name in zip(step_indices, step_names):
                s_item = frame_data[s_idx]
                s_bytes = np.fromfile(s_item['path'], dtype=np.uint8)
                s_bgr = cv2.imdecode(s_bytes, cv2.IMREAD_COLOR)
                s_vis = draw_skeleton_with_torso(s_bgr, s_item['kps_2d'], s_item['vis'], is_defect=is_defect, lead_knee_idx=lead_idx)

                s_tinfo = calc_torso_metrics(s_item['kps_2d'], s_item['kps_3d'])
                s_over = s_vis.copy()
                cv2.rectangle(s_over, (0, 0), (w, 90), (20, 20, 25), -1)
                cv2.addWeighted(s_over, 0.82, s_vis, 0.18, 0, s_vis)

                s_title = f"{status_tag} {key_name} [{v_alias}] ({s_name}) - Frame {s_idx+1:02d}"
                cv2.putText(s_vis, s_title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, tag_color, 2, cv2.LINE_AA)
                s_stat = f"체간 기울기: {s_tinfo['lean_deg_2d']:.1f} deg | 유형: {type_desc}"
                cv2.putText(s_vis, s_stat, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 1, cv2.LINE_AA)

                seq_fname = f"Sequence_{v_code}_{s_name}_frame{s_idx+1:02d}.jpg"
                seq_path = os.path.join(target_dir, seq_fname)
                _, s_enc = cv2.imencode('.jpg', s_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                s_enc.tofile(seq_path)

def create_readme_guide():
    readme_path = os.path.join(OUT_TEMP2_DIR, "README_상체결함_유형별_비교안내.md")
    content = """# vidio-health-trainer: 사이드 런지 상체 젖힘/숙임(c5) 결함 유형별 분석 데이터셋

이 폴더(`D:\\Data\\temp2`)는 사이드 런지 동작 중 발생하는 **상체 과도한 젖힘/숙임 결함(c5)**의 다양한 발현 유형과 **표준 정상(Normal)** 동작을 시각적으로 직접 비교하여 확인할 수 있도록 구성된 데이터입니다.

---

## 1. 수록된 6대 대표 동작 유형

| 폴더명 | 동작 Key | 구분 | 결함 발현 유형 및 기구학적 특성 |
| :--- | :---: | :---: | :--- |
| **00_정상_Key145_상체중립** | Key 145 | **정상** | 골반 위에 척추와 체간이 꼿꼿이 수직 중립을 유지하며, 좌/우/전/후 젖힘 없음. |
| **01_결함_Key150_순수상체뒤로젖힘** | Key 150 | **결함** | 하지는 정상이나 상체 체간만 뒤쪽으로 젖혀지는 순수 체간 과신전(Hyperextension). |
| **02_결함_Key165_하늘보고상체젖힘** | Key 165 | **결함** | 시선이 위(하늘)를 향하며 경추 과신전과 함께 흉추·요추 전체가 뒤로 크게 젖혀짐. |
| **03_결함_Key166_바닥보고상체젖힘_숙임** | Key 166 | **결함** | 시선과 목(경추)은 바닥을 보며 숙이고 있으나 체간은 뒤로 젖혀져 척추 곡선이 심하게 왜곡됨. |
| **04_결함_Key160_몸통갸우뚱_상체젖힘** | Key 160 | **결함** | 체간이 횡방향(외측)으로 기울어지면서(갸우뚱) 후방으로 동시에 무너지는 3차원 복합 결함. |
| **05_결함_Key154_앞다리미달_상체젖힘** | Key 154 | **결함** | 앞무릎을 90도까지 충분히 굽히지 못하고 체중 이동을 상체 젖힘으로 보상하려는 보상 패턴. |

---

## 2. 시각화 지표 설명

1. **체간 중심선 (Spine Core Line)**:
   - **어깨 중심(Mid-Shoulder) $\leftrightarrow$ 골반 중심(Mid-Hip)**을 잇는 척추 중심선
   - **정상 동작**: 녹색(Green) 실선으로 안정적인 중립 축 표시
   - **결함 동작**: 적색/진홍색(Red) 굵은 실선으로 뒤로 젖혀지거나 비틀린 척추 곡선 표시
2. **체간 기울기 (Torso Lean Angle)**:
   - 중력 수직축 대비 체간이 기울어진 2D 각도 (°). 정상에 비해 결함 동작에서 각도가 크게 치솟거나 후방 각도가 왜곡됨.
3. **5대 공식 시점 전수 수록**:
   - `view5` (시점 A: 우측 후방 45°)
   - `view2` (시점 B: 우측 전방 45°)
   - `view3` (시점 C: 완전 정면 0°)
   - `view4` (시점 D: 좌측 전방 45°)
   - `view1` (시점 E: 좌측 후방 45°)
   * **관측 추천 시점**: 상체 뒤로 젖힘은 후방 시점(**시점 A, 시점 E**)에서 척추 후만/전만 곡선이 가장 뚜렷하게 관측됩니다.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] 가이드 마크다운 생성 완료: {readme_path}")

def main():
    os.makedirs(OUT_TEMP2_DIR, exist_ok=True)
    print(f"[*] 출력 루트 디렉토리: {OUT_TEMP2_DIR}")

    detector = MediaPipeAdapter.get_detector()
    for act in TARGET_ACTIONS:
        process_single_action(act, detector)

    create_readme_guide()
    print("\n[+] 모든 상체 젖힘/숙임 결함 유형별 시각화 데이터 생성 완료!")

if __name__ == '__main__':
    main()

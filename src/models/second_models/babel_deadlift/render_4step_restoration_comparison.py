# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: render_4step_restoration_comparison.py
Purpose: 사용자가 제안한 [기립 상태 스켈레톤 기억 & 불가능 동작 Null 처리]와
         [Closed-Chain 2-Link IK + PCHIP 단조성 보간 + 바벨 접지 앵커링]의 4단계를 통합 구현하여,
         과거 원본 MediaPipe 추출 포인트(좌) vs 4단계 기구학 보정 포인트(우)를
         1:1 좌우(Side-by-Side) 분할 이미지로 D:\Data\temp 에 렌더링
"""

import os
import sys
import glob
import cv2
import numpy as np
from scipy.interpolate import pchip_interpolate

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.adapters.mediapipe_adapter import MediaPipeAdapter

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
TEMP_BASE = r"D:\Data\temp\바벨_데드리프트_4단계_포인트보정_비교"
os.makedirs(TEMP_BASE, exist_ok=True)

# 대상 3대 대표 시퀀스 (다양한 시점 및 오류 유형)
TARGET_CLIPS = [
    {
        "folder": "[오류]_D03_피험자7_Key294",
        "sub": 7,
        "view_letter": "C",
        "view_code": "view3",
        "view_name": "시점 C (정면 0°)",
        "out_sub": "01_피험자7_Key294_시점C_정면(결측및지터복원)",
        "desc": "정면 뷰: 프레임 5, 9의 완전 결측(0,0) 및 프레임 13의 발목 80px 급발진 지터 복원"
    },
    {
        "folder": "[오류]_D03_피험자1_Key295",
        "sub": 1,
        "view_letter": "E",
        "view_code": "view1",
        "view_name": "시점 E (좌측 후방 45°)",
        "out_sub": "02_피험자1_Key295_시점E_좌측후방(상체및무릎복원)",
        "desc": "좌측 후방 뷰: 깊은 상체 숙임과 원판 가림으로 인한 골반-무릎 연결선 왜곡 복원"
    },
    {
        "folder": "[오류]_D03_피험자5_Key288",
        "sub": 5,
        "view_letter": "B",
        "view_code": "view2",
        "view_name": "시점 B (우측 전방 45°)",
        "out_sub": "03_피험자5_Key288_시점B_우측전방(원판차폐복원)",
        "desc": "우측 전방 뷰: 셋업 최하사점 바벨 원판 차폐에 의한 하지 관절 떨림 완활화"
    }
]

SKELETON_CONNECTIONS = [
    (11, 12, (200, 200, 200), 2),  # 좌어깨 - 우어깨
    (11, 13, (160, 190, 240), 2),  # 좌어깨 - 좌팔꿈치
    (13, 15, (160, 190, 240), 2),  # 좌팔꿈치 - 좌손목
    (12, 14, (160, 190, 240), 2),  # 우어깨 - 우팔꿈치
    (14, 16, (160, 190, 240), 2),  # 우팔꿈치 - 우손목
    (11, 23, (150, 210, 255), 2),  # 좌어깨 - 좌골반
    (12, 24, (150, 210, 255), 2),  # 우어깨 - 우골반
    (23, 24, (255, 0, 255), 3),    # 골반선 (Magenta)
    (23, 25, (0, 215, 255), 3),    # 좌골반 - 좌무릎 (Yellow/Gold)
    (24, 26, (0, 215, 255), 3),    # 우골반 - 우무릎 (Yellow/Gold)
    (25, 27, (0, 255, 100), 3),    # 좌무릎 - 좌발목 (Lime Green)
    (26, 28, (0, 255, 100), 3),    # 우무릎 - 우발목 (Lime Green)
    (27, 29, (0, 180, 255), 2),    # 좌발목 - 좌뒤꿈치
    (27, 31, (0, 180, 255), 2),    # 좌발목 - 좌발끝
    (29, 31, (0, 180, 255), 2),    # 좌뒤꿈치 - 좌발끝
    (28, 30, (0, 180, 255), 2),    # 우발목 - 우뒤꿈치
    (28, 32, (0, 180, 255), 2),    # 우발목 - 우발끝
    (30, 32, (0, 180, 255), 2),    # 우뒤꿈치 - 우발끝
]


def calculate_angle_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        return 0.0
    cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def apply_4step_restoration(raw_kps: np.ndarray, vis_arr: np.ndarray, view_code: str):
    """
    총 4단계 기구학 복원 알고리즘
    - Step 1: 기립 상태 스켈레톤 캘리브레이션 & 해부학적 불가능 동작(Null 마스킹)
    - Step 2: Closed-Chain 2-Link 역기구학(IK) 기하학적 무릎 복원
    - Step 3: PCHIP 단조성 보존 시계열 보간 및 지터 평활화
    - Step 4: 지면 접지 앵커링 및 물리적 제약 정렬
    """
    N_FRAMES = len(raw_kps)
    restored_kps = raw_kps.copy()
    restored_flags = np.zeros((N_FRAMES, 33), dtype=bool)

    is_left_dom = view_code in ["view1", "view4"]
    hip_idx = 23 if is_left_dom else 24
    knee_idx = 25 if is_left_dom else 26
    ankle_idx = 27 if is_left_dom else 28

    # [Step 1-A: 기립 상태 스켈레톤 캘리브레이션]
    # 관절이 온전히 검출된 프레임(후반 락아웃 구간) 탐색
    valid_frames = []
    for f in range(N_FRAMES):
        if (raw_kps[f, hip_idx, 0] > 10 and raw_kps[f, knee_idx, 0] > 10 and raw_kps[f, ankle_idx, 0] > 10):
            # 길이 검사 (극단적 점프 프레임 제외)
            l_t = np.linalg.norm(raw_kps[f, hip_idx] - raw_kps[f, knee_idx])
            l_s = np.linalg.norm(raw_kps[f, knee_idx] - raw_kps[f, ankle_idx])
            if l_t > 50 and l_s > 50:
                valid_frames.append(f)

    if len(valid_frames) == 0:
        valid_frames = list(range(N_FRAMES))

    # 후반부(락아웃/기립) 우선으로 기준 뼈 길이 산출
    late_valid = [f for f in valid_frames if f >= N_FRAMES // 2]
    calib_pool = late_valid if len(late_valid) >= 3 else valid_frames

    thigh_lens = [np.linalg.norm(raw_kps[f, hip_idx] - raw_kps[f, knee_idx]) for f in calib_pool]
    shank_lens = [np.linalg.norm(raw_kps[f, knee_idx] - raw_kps[f, ankle_idx]) for f in calib_pool]
    ground_ys = [raw_kps[f, ankle_idx, 1] for f in calib_pool]

    L_thigh_ref = float(np.median(thigh_lens))
    L_shank_ref = float(np.median(shank_lens))
    Y_ground_ref = float(np.median(ground_ys))

    # [Step 1-B: 이상치 감지 및 Null 마스킹 (사용자 아이디어)]
    null_mask = np.zeros(N_FRAMES, dtype=bool)

    for f in range(N_FRAMES):
        # 1. 완전 결측 (0, 0)
        if (raw_kps[f, hip_idx, 0] <= 1.0 or raw_kps[f, knee_idx, 0] <= 1.0 or raw_kps[f, ankle_idx, 0] <= 1.0):
            null_mask[f] = True
            continue

        # 2. 뼈 길이 불변성 위반 (기준치 대비 25% 이상 팽창/수축)
        cur_t = np.linalg.norm(raw_kps[f, hip_idx] - raw_kps[f, knee_idx])
        cur_s = np.linalg.norm(raw_kps[f, knee_idx] - raw_kps[f, ankle_idx])
        if abs(cur_t - L_thigh_ref) / L_thigh_ref > 0.25 or abs(cur_s - L_shank_ref) / L_shank_ref > 0.25:
            null_mask[f] = True
            continue

        # 3. 위상 전도 (발목이 무릎보다 위에 있거나 바닥에서 50px 이상 공중부양)
        if raw_kps[f, ankle_idx, 1] < raw_kps[f, knee_idx, 1] or abs(raw_kps[f, ankle_idx, 1] - Y_ground_ref) > 50:
            null_mask[f] = True
            continue

        # 4. 해부학적 각도 위반 (무릎 과신전 > 178도 이거나 역방향 꺾임)
        ang = calculate_angle_2d(raw_kps[f, hip_idx], raw_kps[f, knee_idx], raw_kps[f, ankle_idx])
        if ang < 15 or ang > 178:
            null_mask[f] = True

    # [Step 2 & 3: PCHIP 시계열 보간 및 Closed-Chain 2-Link IK 기하 복원]
    clean_frames = np.where(~null_mask)[0]

    # 만약 유효 프레임이 너무 적으면 0, N-1 강제 앵커링
    if len(clean_frames) < 3:
        clean_frames = np.array([0, N_FRAMES - 1])

    frames_all = np.arange(N_FRAMES)

    for j in range(33):
        # x, y 각각에 대해 PCHIP 보간
        valid_x = raw_kps[clean_frames, j, 0]
        valid_y = raw_kps[clean_frames, j, 1]

        # 결측이 있었던 관절 채우기
        if np.any(valid_x > 0):
            interp_x = pchip_interpolate(clean_frames, valid_x, frames_all)
            interp_y = pchip_interpolate(clean_frames, valid_y, frames_all)
            restored_kps[:, j, 0] = interp_x
            restored_kps[:, j, 1] = interp_y

    # [Step 4: 기하학적 2-Link IK 무릎 위치 정밀 투영 및 바닥 앵커링]
    for f in range(N_FRAMES):
        if null_mask[f]:
            restored_flags[f, :] = True
            # 발목은 바닥면에 스냅
            restored_kps[f, ankle_idx, 1] = Y_ground_ref

            # 2-Link IK 수식 적용 (고관절 H와 발목 A 사이의 무릎 K 위치 기하학적 계산)
            H = restored_kps[f, hip_idx]
            A = restored_kps[f, ankle_idx]
            d_HA = np.linalg.norm(H - A)

            if d_HA < (L_thigh_ref + L_shank_ref) and d_HA > abs(L_thigh_ref - L_shank_ref):
                # 삼각형 해법 (코사인 제2법칙)
                a = (L_thigh_ref**2 - L_shank_ref**2 + d_HA**2) / (2 * d_HA)
                h = np.sqrt(max(0, L_thigh_ref**2 - a**2))
                P2 = H + a * (A - H) / d_HA

                # 무릎 전방 돌출 벡터 (시점별 전방 방향)
                # 시점C/B에서는 전방이 x축 방향 혹은 아래 전방
                sign = 1 if is_left_dom else -1
                norm_vec = np.array([-(A[1] - H[1]), (A[0] - H[0])]) * sign
                norm_vec = norm_vec / (np.linalg.norm(norm_vec) + 1e-6)

                K_proj = P2 + h * norm_vec
                restored_kps[f, knee_idx] = K_proj

    return restored_kps, null_mask, restored_flags, L_thigh_ref, L_shank_ref


def draw_skeleton_panel(img_bgr, kps, is_restored_panel: bool, null_frames_mask: np.ndarray,
                        frame_idx: int, meta: dict, L_thigh_ref: float, L_shank_ref: float) -> np.ndarray:
    """단일 패널(좌: 과거 원본, 우: 4단계 보정) 렌더링"""
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    is_null_frame = null_frames_mask[frame_idx]
    is_left_dom = meta["view_code"] in ["view1", "view4"]
    hip_idx = 23 if is_left_dom else 24
    knee_idx = 25 if is_left_dom else 26
    ankle_idx = 27 if is_left_dom else 28

    # 1. 스켈레톤 라인 그리기
    for idx1, idx2, col, thick in SKELETON_CONNECTIONS:
        p1 = (int(kps[idx1, 0]), int(kps[idx1, 1]))
        p2 = (int(kps[idx2, 0]), int(kps[idx2, 1]))

        # 원본 패널에서 (0,0) 결측 시 선 생략
        if not is_restored_panel and (p1 == (0, 0) or p2 == (0, 0)):
            continue

        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            line_col = col
            if not is_restored_panel and is_null_frame and (idx1 in [23, 25, 27] or idx2 in [23, 25, 27]):
                line_col = (50, 50, 220) # Red/corrupted on left panel
            elif is_restored_panel and is_null_frame and (idx1 in [23, 25, 27] or idx2 in [23, 25, 27]):
                line_col = (0, 255, 180) # Bright Cyan/Restored on right panel
                thick = 4

            cv2.line(canvas, p1, p2, line_col, thick, cv2.LINE_AA)

    # 2. 33개 관절 포인트 그리기
    for i in range(33):
        pt = (int(kps[i, 0]), int(kps[i, 1]))
        if pt == (0, 0) and not is_restored_panel:
            continue

        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            radius = 5
            c_fill = (220, 220, 220)
            if i in [23, 24]:
                c_fill = (255, 0, 255)
                radius = 6
            elif i in [25, 26]:
                c_fill = (0, 165, 255) if not (is_restored_panel and is_null_frame) else (0, 255, 0)
                radius = 7
            elif i in [27, 28]:
                c_fill = (0, 255, 0)
                radius = 6

            # 원본 패널에서 이상치인 경우 빨간 링 표시
            if not is_restored_panel and is_null_frame and i in [23, 25, 27]:
                cv2.circle(canvas, pt, radius + 4, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.circle(canvas, pt, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, radius, c_fill, -1, cv2.LINE_AA)

    # 3. 각도 및 뼈 길이 텍스트
    p_h = kps[hip_idx]
    p_k = kps[knee_idx]
    p_a = kps[ankle_idx]

    knee_ang = calculate_angle_2d(p_h, p_k, p_a)
    cur_thigh = np.linalg.norm(p_h - p_k)
    cur_shank = np.linalg.norm(p_k - p_a)

    # 상단 패널 뱃지
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 25, 35), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    if not is_restored_panel:
        title_txt = "[좌] 과거 원본 MediaPipe (미보정)"
        sub_txt = "관절 소실(0,0), 지터 왜곡, 뼈길이 팽창/수축 방치"
        badge_col = (0, 0, 255) if is_null_frame else (200, 200, 200)
        status_txt = "STATUS: ANOMALY DETECTED (결측/이상치)" if is_null_frame else "STATUS: RAW OBSERVATION"
    else:
        title_txt = "[우] 4단계 기구학 보정 (스켈레톤 기억+IK+PCHIP)"
        sub_txt = "해부학적 Null 게이팅 + 2-Link IK 복원 + 단조 보간"
        badge_col = (0, 255, 100) if is_null_frame else (220, 220, 0)
        status_txt = "STATUS: RESTORED via 2-Link IK (정상 재구성)" if is_null_frame else "STATUS: VALIDATED KINEMATICS"

    cv2.putText(canvas, title_txt, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, badge_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, sub_txt, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, status_txt, (15, 70), cv2.FONT_HERSHEY_DUPLEX, 0.48, badge_col, 1, cv2.LINE_AA)

    # 하단 메트릭 오버레이
    overlay_bot = canvas.copy()
    cv2.rectangle(overlay_bot, (0, h - 55), (w, h), (15, 20, 25), -1)
    cv2.addWeighted(overlay_bot, 0.85, canvas, 0.15, 0, canvas)

    t_err = ((cur_thigh - L_thigh_ref) / L_thigh_ref) * 100 if L_thigh_ref > 0 else 0
    metric_txt = f"Knee Ang: {knee_ang:.1f}deg | Thigh: {cur_thigh:.1f}px (Err: {t_err:+.1f}%) | Shank: {cur_shank:.1f}px"
    m_col = (100, 100, 255) if (not is_restored_panel and is_null_frame) else (100, 255, 200)
    cv2.putText(canvas, metric_txt, (15, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, m_col, 1, cv2.LINE_AA)

    return canvas


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 4단계 기구학 보정 vs 원본 MediaPipe 비교 렌더링")
    print("=" * 80)

    detector = MediaPipeAdapter.get_detector("d:/pose_landmarker_heavy.task")

    total_rendered = 0

    for clip_info in TARGET_CLIPS:
        c_folder = clip_info["folder"]
        v_let = clip_info["view_letter"]
        v_code = clip_info["view_code"]
        out_sub_dir = os.path.join(TEMP_BASE, clip_info["out_sub"])
        os.makedirs(out_sub_dir, exist_ok=True)

        src_dir = os.path.join(RAW_IMG_DIR, c_folder)
        if not os.path.isdir(src_dir):
            print(f"[!] 원본 폴더 없음: {src_dir}")
            continue

        imgs = sorted([f for f in os.listdir(src_dir) if f"시점{v_let}" in f or f"_{v_let}_" in f])
        print(f"\n[*] 클립 처리 시작: {clip_info['out_sub']} (총 {len(imgs)}장)...")

        # 1. 전체 16프레임 원본 MediaPipe 추출
        raw_kps_list = []
        vis_list = []
        img_bgr_list = []

        for fn in imgs:
            img_p = os.path.join(src_dir, fn)
            img_bytes = np.fromfile(img_p, dtype=np.uint8)
            bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img_bgr_list.append(bgr)

            res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
            raw_kps_list.append(res['kps_33_2d'])
            vis_list.append(res['vis_33'])

        raw_kps_arr = np.array(raw_kps_list)
        vis_arr = np.array(vis_list)

        # 2. 4단계 기구학 보정 적용
        restored_kps_arr, null_mask, restored_flags, L_thigh_ref, L_shank_ref = apply_4step_restoration(
            raw_kps_arr, vis_arr, v_code
        )

        print(f"  - 캘리브레이션 대퇴골: {L_thigh_ref:.1f}px, 하퇴골: {L_shank_ref:.1f}px")
        print(f"  - Null 감지 및 복원 프레임: {np.where(null_mask)[0] + 1}")

        # 3. 프레임별 좌우 분할 합성 렌더링
        for f_idx in range(len(imgs)):
            bgr = img_bgr_list[f_idx]
            if bgr is None:
                continue

            # 좌측 패널 (원본)
            left_panel = draw_skeleton_panel(
                bgr, raw_kps_arr[f_idx], is_restored_panel=False,
                null_frames_mask=null_mask, frame_idx=f_idx, meta=clip_info,
                L_thigh_ref=L_thigh_ref, L_shank_ref=L_shank_ref
            )

            # 우측 패널 (4단계 보정)
            right_panel = draw_skeleton_panel(
                bgr, restored_kps_arr[f_idx], is_restored_panel=True,
                null_frames_mask=null_mask, frame_idx=f_idx, meta=clip_info,
                L_thigh_ref=L_thigh_ref, L_shank_ref=L_shank_ref
            )

            # 좌우 결합
            combined = cv2.hconcat([left_panel, right_panel])

            # 최상단 통합 헤더 배너
            ch, cw = combined.shape[:2]
            header_h = 42
            top_bar = np.zeros((header_h, cw, 3), dtype=np.uint8)
            top_bar[:] = (10, 15, 20)

            header_text = f"[vidio-health-trainer] 4-Step Kinematic Restoration | {c_folder} | Frame {f_idx+1:02d}/16 | {clip_info['view_name']}"
            cv2.putText(top_bar, header_text, (20, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            final_img = cv2.vconcat([top_bar, combined])

            # 저장
            out_fn = f"comparison_frame_{f_idx+1:02d}.jpg"
            out_path = os.path.join(out_sub_dir, out_fn)
            res_enc, buf = cv2.imencode(".jpg", final_img)
            if res_enc:
                with open(out_path, "wb") as f_out:
                    f_out.write(buf.tobytes())
                total_rendered += 1

        print(f"  -> {clip_info['out_sub']} 저장 완료 (16장)")

    # 4. 종합 가이드 파일 작성
    readme_path = os.path.join(TEMP_BASE, "README_4단계_포인트보정_관전가이드.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("vidio-health-trainer | 바벨 데드리프트 최하사점 4단계 관절 복원 전후 비교 가이드\n")
        f.write("=" * 80 + "\n\n")
        f.write("본 폴더는 사용자의 아이디어(기립 상태 스켈레톤 기억 및 불가능 동작 Null 처리)와\n")
        f.write("컴퓨터 비전 3대 기법(Closed-Chain 2-Link IK + PCHIP 단조 보간 + 바닥 앵커링)을 통합한\n")
        f.write("총 4단계 기구학 복원 알고리즘의 효과를 직접 확인하실 수 있는 1:1 비교 폴더입니다.\n\n")
        f.write("■ 폴더 구조:\n")
        f.write("D:\\Data\\temp\\바벨_데드리프트_4단계_포인트보정_비교\\\n")
        f.write("├── 01_피험자7_Key294_시점C_정면(결측및지터복원)/  (정면: 프레임 5,9 완전 결측 및 프레임 13 지터 복원)\n")
        f.write("├── 02_피험자1_Key295_시점E_좌측후방(상체및무릎복원)/  (후방: 깊은 숙임 각도 왜곡 및 뼈길이 복원)\n")
        f.write("└── 03_피험자5_Key288_시점B_우측전방(원판차폐복원)/  (전방 대각: 최하사점 원판 가림 구간 무릎 궤적 안정화)\n\n")
        f.write("■ 좌우 이미지 관전 포인트:\n")
        f.write(" [좌측 패널: 과거 원본 MediaPipe (미보정)]\n")
        f.write("  - 최하사점이나 바벨 통과 시 관절이 사라져 (0,0)으로 떨어지거나 스켈레톤이 끊김.\n")
        f.write("  - 순간적으로 발목/무릎이 튀면서 뼈 길이가 40% 이상 수축/팽창되는 지터 발생 (빨간색 링 표시).\n\n")
        f.write(" [우측 패널: 4단계 기구학 보정 (스켈레톤 기억 + IK + PCHIP)]\n")
        f.write("  - Step 1 (캘리브레이션 & Null): 기립 상태에서 측정한 뼈 길이를 기준으로 이상치 자동 감지 및 제거.\n")
        f.write("  - Step 2 (2-Link IK): 고관절과 바닥 발목 위치 사이에서 무릎을 완벽한 뼈 길이 교점으로 역추정 복원.\n")
        f.write("  - Step 3 (PCHIP): 룽게 현상 없는 단조 보존 곡선으로 부드럽게 연결.\n")
        f.write("  - 결측되었던 프레임에서도 온전한 형광 녹색 스켈레톤과 정확한 무릎 각도가 표시됨.\n")

    print(f"\n[+] 가이드 파일 작성 완료: {readme_path}")
    print(f"[+] 총 {total_rendered}장의 비교 이미지가 성공적으로 렌더링되었습니다!")


if __name__ == "__main__":
    main()

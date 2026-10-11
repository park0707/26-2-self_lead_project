# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: visualize_spinepose_on_images.py
Purpose: D:/Data/temp를 비우고, 실제 바벨 스쿼트 프레임 이미지 위에
         SpinePose의 9대 척추 분절(S1, L5, L3, L1, T8, T3, C7, C4, C1) 및
         척추 정렬 곡선을 고해상도로 시각화하여 저장
"""

import os
import sys
import glob
import cv2
import numpy as np
from spinepose import SpinePoseEstimator

TEMP_DIR = 'd:/Data/temp'
IMG_BASE_DIR = 'd:/Data/01_raw_data/images'

# SpinePose 공식 척추 체인 9개 분절 (골반 -> 요추 -> 흉추 -> 경추)
SPINE_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINE_NAMES = ['S1 (Sacrum)', 'L5 (Low Lumbar)', 'L3 (Mid Lumbar)', 'L1 (Upper Lumbar)',
               'T8 (Mid Thoracic)', 'T3 (Upper Thoracic)', 'C7 (Lower Cervical)',
               'C4 (Mid Cervical)', 'C1 (Atlas/Base)']
SPINE_SHORT = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']

# 분절별 색상 (BGR): S1(파랑) -> 요추(녹색/노랑) -> 흉추(주황) -> 경추(빨강)
SPINE_COLORS = [
    (255, 100, 0),    # S1: 진한 파랑
    (255, 200, 0),    # L5: 시안
    (0, 230, 120),    # L3: 연두
    (0, 200, 0),      # L1: 녹색
    (0, 220, 255),    # T8: 노랑
    (0, 140, 255),    # T3: 주황
    (50, 50, 255),    # C7: 다홍
    (150, 0, 255),    # C4: 자홍
    (0, 0, 255)       # C1: 빨강
]


def clean_temp_directory():
    """temp 디렉토리 비우기"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    existing_files = os.listdir(TEMP_DIR)
    for f in existing_files:
        fp = os.path.join(TEMP_DIR, f)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                import shutil
                shutil.rmtree(fp)
        except Exception as e:
            print(f"  - 파일 삭제 실패 ({f}): {e}")
    print(f"[*] {TEMP_DIR} 디렉토리 정리 완료: 기존 {len(existing_files)}개 파일 삭제됨.")


def load_korean_path_image(img_path):
    """Windows 한글 경로 이미지 안전 로드"""
    try:
        arr = np.fromfile(img_path, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_korean_path_image(save_path, img_bgr):
    """Windows 한글 경로 이미지 안전 저장"""
    try:
        ext = os.path.splitext(save_path)[1]
        res, arr = cv2.imencode(ext, img_bgr)
        if res:
            with open(save_path, 'wb') as f:
                f.write(arr.tobytes())
            return True
    except Exception as e:
        print(f"저장 실패: {e}")
    return False


def draw_spinepose_skeleton(img, kps_2d, scores, title_text=""):
    """이미지 위에 9대 척추 분절 및 척추선 시각화"""
    h, w = img.shape[:2]
    canvas = img.copy()

    # 1. 상단 정보 배너 렌더링
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(canvas, f"[vidio-health-trainer] SpinePose 9-Spine Chain Verification", (30, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)

    # 척추 체인 좌표 추출
    spine_pts = []
    spine_scores = []
    for idx in SPINE_INDICES:
        pt = kps_2d[idx]
        sc = scores[idx]
        spine_pts.append((int(round(pt[0])), int(round(pt[1]))))
        spine_scores.append(float(sc))

    # 2. 척추 뼈대 연결선 (Spine Alignment Backbone Line)
    for i in range(len(spine_pts) - 1):
        pt1 = spine_pts[i]
        pt2 = spine_pts[i + 1]
        sc1 = spine_scores[i]
        sc2 = spine_scores[i + 1]

        if sc1 > 0.2 and sc2 > 0.2:
            # 척추선 글로우 효과 (외부 그림자 + 중심선)
            cv2.line(canvas, pt1, pt2, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, SPINE_COLORS[i], 4, cv2.LINE_AA)

    # 3. 척추 9개 관절점 및 라벨 렌더링
    for i, (pt, sc, name, short, col) in enumerate(zip(spine_pts, spine_scores, SPINE_NAMES, SPINE_SHORT, SPINE_COLORS)):
        if sc <= 0.2:
            continue

        # 원형 관절 마커 (외곽 흰색 테두리 + 중심 색상 원)
        cv2.circle(canvas, pt, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 7, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 관절 라벨 박스 (시인성 극대화)
        label_text = f"{short} ({sc:.2f})"
        (tw, th), tb = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # 라벨 위치: 포인트 우측 또는 좌측
        lx = pt[0] + 16
        ly = pt[1] + 5

        # 화면 밖 이탈 방지
        if lx + tw + 10 > w:
            lx = pt[0] - tw - 20

        # 반투명 배경 박스
        cv2.rectangle(canvas, (lx - 4, ly - th - 4), (lx + tw + 4, ly + tb + 2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (lx - 4, ly - th - 4), (lx + tw + 4, ly + tb + 2), col, 1)
        cv2.putText(canvas, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. 범례 박스 (우측 하단)
    legend_w, legend_h = 320, 260
    lx0, ly0 = w - legend_w - 30, h - legend_h - 30
    cv2.rectangle(canvas, (lx0, ly0), (lx0 + legend_w, ly0 + legend_h), (25, 25, 25), -1)
    cv2.rectangle(canvas, (lx0, ly0), (lx0 + legend_w, ly0 + legend_h), (180, 180, 180), 1)
    cv2.putText(canvas, "SpinePose Segments", (lx0 + 15, ly0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    for idx, (short, full, col) in enumerate(zip(SPINE_SHORT, SPINE_NAMES, SPINE_COLORS)):
        y_pos = ly0 + 55 + idx * 22
        cv2.circle(canvas, (lx0 + 22, y_pos - 4), 6, col, -1)
        cv2.putText(canvas, f"{short}: {full}", (lx0 + 38, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    return canvas


def main():
    print("=" * 100)
    print("  [Step 1] temp 디렉토리 비우기")
    print("=" * 100)
    clean_temp_directory()

    print("\n" + "=" * 100)
    print("  [Step 2] SpinePose 모델 로드 및 다양한 시점/위상 스쿼트 시퀀스 선별")
    print("=" * 100)
    estimator = SpinePoseEstimator(mode='small', enable_lifting=False)
    print("[*] SpinePoseEstimator 초기화 완료.")

    # 스쿼트 이미지 폴더들 탐색
    sq_dirs = glob.glob(os.path.join(IMG_BASE_DIR, '*01*/*'))
    print(f"[*] 발견된 스쿼트 시퀀스 폴더 수: {len(sq_dirs)}")

    # 다양한 시퀀스(다른 피험자/각도) 4개 선택: 0번, 30번, 70번, 120번
    target_dir_indices = [0, 30, 70, 120]
    selected_targets = []

    for idx in target_dir_indices:
        if idx >= len(sq_dirs):
            continue
        sdir = sq_dirs[idx]
        raw_files = [f for f in os.listdir(sdir) if f.lower().endswith('.jpg')]
        raw_files.sort()
        img_files = [os.path.join(sdir, f) for f in raw_files]
        if len(img_files) < 20:
            continue

        bname = os.path.basename(sdir)
        n = len(img_files)
        # 스쿼트 4대 위상 프레임 선택
        f_standing = img_files[0]
        f_descent = img_files[int(n * 0.30)]
        f_nadir = img_files[int(n * 0.50)]
        f_ascent = img_files[int(n * 0.70)]

        selected_targets.append({
            'seq_name': bname,
            'seq_idx': idx,
            'frames': [
                ('01_standing', f_standing, 'Standing (Start)'),
                ('02_descent', f_descent, 'Descent (Hip-Hinge Flexion)'),
                ('03_nadir', f_nadir, 'Nadir (Bottom / Deep Squat)'),
                ('04_ascent', f_ascent, 'Ascent (Sticking Point / Ascent)')
            ]
        })

    print(f"[*] 시각화 대상 시퀀스 {len(selected_targets)}개 선정 완료:")
    for tgt in selected_targets:
        print(f"    - Seq #{tgt['seq_idx']}: {tgt['seq_name']}")

    print("\n" + "=" * 100)
    print("  [Step 3] SpinePose 추론 및 척추 9분절 시각화 이미지 생성")
    print("=" * 100)

    saved_count = 0
    img_idx = 1

    for tgt in selected_targets:
        seq_name = tgt['seq_name']
        seq_idx = tgt['seq_idx']
        
        for phase_code, img_path, phase_desc in tgt['frames']:
            img_bgr = load_korean_path_image(img_path)
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # SpinePose 검출
            bboxes = estimator.detect(img_rgb)
            if len(bboxes) == 0:
                # 바운딩 박스 검출 실패 시 전체 화면
                bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

            kps = estimator.estimate(img_rgb, bboxes)
            if len(kps) < 2 or len(kps[0]) == 0:
                continue

            kps_2d_all = kps[0] # (num_people, 37, 2)
            scores_all = kps[1] # (num_people, 37)

            # 주 피험자 선정: 화면 중심(x=0.5, y=0.5)에 가장 가깝고 신뢰도가 높은 사람
            best_person = 0
            best_score = -1.0
            for p_idx in range(len(kps_2d_all)):
                mean_sc = np.mean(scores_all[p_idx, SPINE_INDICES])
                center_x = np.mean(kps_2d_all[p_idx, SPINE_INDICES, 0])
                dist_center = abs(center_x - w / 2.0)
                score_weight = mean_sc * (1.0 / (dist_center + 100.0))
                if score_weight > best_score:
                    best_score = score_weight
                    best_person = p_idx

            primary_kps = kps_2d_all[best_person]
            primary_scores = scores_all[best_person]

            # 시각화 텍스트
            title_text = f"Sequence: {seq_name} | {phase_desc}"
            vis_img = draw_spinepose_skeleton(img_bgr, primary_kps, primary_scores, title_text=title_text)

            # 파일 저장
            out_filename = f"spinepose_{img_idx:02d}_seq{seq_idx:03d}_{phase_code}.jpg"
            out_path = os.path.join(TEMP_DIR, out_filename)
            save_success = save_korean_path_image(out_path, vis_img)

            if save_success:
                print(f"  [{img_idx:02d}/16] 저장 완료: {out_filename} ({phase_desc})")
                saved_count += 1
            img_idx += 1

    print("\n" + "=" * 100)
    print(f"[*] 총 {saved_count}장의 SpinePose 척추 검증 이미지가 {TEMP_DIR}에 성공적으로 저장되었습니다!")
    print("=" * 100)


if __name__ == '__main__':
    main()

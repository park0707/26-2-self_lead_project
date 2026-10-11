# -*- coding: utf-8 -*-
"""
오분류 대표 케이스(Top FP, Top FN)의 실제 이미지 프레임 및 스켈레톤 추출 확인
"""
import os
import glob
import cv2
import numpy as np

IMG_BASE = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
TEMP_VIS_DIR = r"D:\Data\temp\d1_misclass_visuals"

os.makedirs(TEMP_VIS_DIR, exist_ok=True)

VIEW_TAG_MAP = {
    'view5': 'A_',
    'view2': 'B_',
    'view3': 'C_',
    'view4': 'D_',
    'view1': 'E_'
}

cases = [
    {
        "type": "FP_피험자7_과도숙임",
        "key": "Key286",
        "sub": "7",
        "view": "view5",
        "sid": "[오류]_D03_피험자7_Key286_view5",
        "desc": "정상(척추중립) 라벨이나 상체가 거의 수평(Pitch<10)으로 깊게 숙여져 모델이 결함으로 오판"
    },
    {
        "type": "FP_피험자1_정상_자연만곡",
        "key": "Key281",
        "sub": "1",
        "view": "view5",
        "sid": "[정상]_D03_피험자1_Key281_view5",
        "desc": "정상 라벨이나 후방 시점(시점A)에서 체간 깊이 압축으로 인해 자연 곡률이 결함으로 오판"
    },
    {
        "type": "FN_피험자5_불완전ROM",
        "key": "Key287",
        "sub": "5",
        "view": "view5",
        "sid": "[오류]_D03_피험자5_Key287_view5",
        "desc": "결함 라벨이나 상체가 92도 이하로 숙여지지 않아 척추 굴곡이 물리적으로 발생하지 않은 라벨 노이즈"
    },
    {
        "type": "FN_피험자4_불완전ROM",
        "key": "Key282",
        "sub": "4",
        "view": "view2",
        "sid": "[오류]_D03_피험자4_Key282_view2",
        "desc": "결함 라벨이나 가동범위 부족으로 실제 척추 만곡이 거의 없는 미세 결함"
    }
]

spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

all_dirs = [d for d in os.listdir(IMG_BASE) if os.path.isdir(os.path.join(IMG_BASE, d))]

for c in cases:
    # 폴더 매칭
    target_dir = None
    for d in all_dirs:
        if c["key"] in d and c["sub"] in d:
            target_dir = os.path.join(IMG_BASE, d)
            break

    if not target_dir:
        print(f"[!] 폴더를 찾지 못함: {c['key']}, 피험자 {c['sub']}")
        continue

    v_tag = VIEW_TAG_MAP[c["view"]]
    all_files = sorted(os.listdir(target_dir))
    v_files = [f for f in all_files if v_tag in f and f.endswith(".jpg")]
    if not v_files:
        print(f"[!] 이미지 파일 없음: {target_dir} ({v_tag})")
        continue

    # 16개 프레임 중 최하사점(보통 8~10번째) 이미지 선택
    target_idx = min(8, len(v_files) - 1)
    img_p = os.path.join(target_dir, v_files[target_idx])
    img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    h, w = img.shape[:2]

    # SpinePose 32번째(보간 64F의 중간) 척추 좌표
    sid = c["sid"]
    s9 = spine_dict.get(sid, None)

    out_img = img.copy()
    if s9 is not None:
        pts = s9[32]  # (9, 3) normalized [0, 1]
        for j in range(9):
            px = int(np.clip(pts[j, 0] * w, 0, w - 1))
            py = int(np.clip(pts[j, 1] * h, 0, h - 1))
            cv2.circle(out_img, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(out_img, (px, py), 8, (255, 255, 255), 1)

        # 척추선 연결
        for j in range(8):
            p1 = (int(np.clip(pts[j, 0] * w, 0, w - 1)), int(np.clip(pts[j, 1] * h, 0, h - 1)))
            p2 = (int(np.clip(pts[j+1, 0] * w, 0, w - 1)), int(np.clip(pts[j+1, 1] * h, 0, h - 1)))
            cv2.line(out_img, p1, p2, (0, 255, 0), 3)

    # 텍스트 박스
    cv2.rectangle(out_img, (10, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.putText(out_img, f"[{c['type']}] {sid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out_img, f"Deepest Point Frame {target_idx+1}/{len(v_files)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out_p = os.path.join(TEMP_VIS_DIR, f"{c['type']}.jpg")
    cv2.imencode('.jpg', out_img)[1].tofile(out_p)
    print(f"[+] 시각화 렌더링 완료: {out_p}")

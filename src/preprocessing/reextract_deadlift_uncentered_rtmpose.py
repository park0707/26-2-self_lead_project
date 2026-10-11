# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/preprocessing/reextract_deadlift_uncentered_rtmpose.py
Purpose: 02_바벨_데드리프트 구버전 파일 385건 (정상 160건 + 기타 225건)을
         단일 프로세스 순차 루프로 안전하고 신속하게 신규 7단계 전처리(골반 중심화)로 재추출 및 덮어쓰기
"""

import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from rtmlib import BodyWithFeet
from src.preprocessing.rtmpose_preprocessor import RTMPosePreprocessor

RAW_BASE_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
OUT_BASE_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"

VIEW_TO_LETTER = {
    'view5': '시점A',
    'view2': '시점B',
    'view3': '시점C',
    'view4': '시점D',
    'view1': '시점E'
}

def main():
    print("=" * 80, flush=True)
    print("  [vidio-health-trainer] 02_바벨_데드리프트 구버전 파일 전수 재추출 (단일 프로세스 안전 모드)", flush=True)
    print("=" * 80, flush=True)

    # 1. 대상 파일 필터링 (크기 < 25,000 바이트)
    all_files = os.listdir(OUT_BASE_DIR)
    target_files = [f for f in all_files if f.endswith('.npz') and os.path.getsize(os.path.join(OUT_BASE_DIR, f)) < 25000]
    print(f"[*] 재추출 대상 구버전 파일 수: {len(target_files)}건", flush=True)

    tasks = []
    view_names = ['view1', 'view2', 'view3', 'view4', 'view5']
    target_folders = sorted(list(set([f.rsplit('_view', 1)[0] for f in target_files])))
    norm_cnt = len([f for f in target_folders if '정상' in f])
    abn_cnt = len([f for f in target_folders if '정상' not in f])
    print(f"[*] 대상 폴더 수: {len(target_folders)}개 (정상: {norm_cnt}개=160건, 기타: {abn_cnt}개=225건)", flush=True)

    for fld in target_folders:
        for vn in view_names:
            tasks.append({
                'folder_name': fld,
                'view_name': vn
            })

    total = len(tasks)
    print(f"[*] 총 실행 시퀀스 수: {total}개", flush=True)

    # 2. RTMPose 검출기 1회 로드
    t_load = time.time()
    detector = BodyWithFeet(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')
    print(f"[+] RTMPose 검출기 로드 완료 ({time.time()-t_load:.2f}초)", flush=True)

    t0 = time.time()
    success = 0
    fail = 0

    for idx, item in enumerate(tasks, 1):
        folder_name = item['folder_name']
        vn = item['view_name']
        letter = VIEW_TO_LETTER.get(vn, '')
        out_path = os.path.join(OUT_BASE_DIR, f"{folder_name}_{vn}.npz")
        img_folder = os.path.join(RAW_BASE_DIR, folder_name)

        if not os.path.isdir(img_folder):
            fail += 1
            continue

        frame_files = sorted([f for f in os.listdir(img_folder) if letter in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(frame_files) < 4:
            fail += 1
            continue

        try:
            f1_path = os.path.join(img_folder, frame_files[0])
            f_mid_path = os.path.join(img_folder, frame_files[len(frame_files)//2])

            img1 = cv2.imdecode(np.fromfile(f1_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_mid = cv2.imdecode(np.fromfile(f_mid_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img1 is None or img_mid is None:
                fail += 1
                continue

            h, w, _ = img1.shape
            b1 = detector.det_model(img1)
            b_mid = detector.det_model(img_mid)

            if len(b1) == 0 and len(b_mid) == 0:
                union_box = np.array([[0, 0, w, h]], dtype=np.float32)
            else:
                boxes = []
                if len(b1) > 0:
                    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
                    boxes.append(b1[np.argmax(a1)])
                if len(b_mid) > 0:
                    a_mid = (b_mid[:, 2] - b_mid[:, 0]) * (b_mid[:, 3] - b_mid[:, 1])
                    boxes.append(b_mid[np.argmax(a_mid)])
                boxes = np.array(boxes)
                union_box = np.array([[
                    np.min(boxes[:, 0]),
                    np.min(boxes[:, 1]),
                    np.max(boxes[:, 2]),
                    np.max(boxes[:, 3])
                ]], dtype=np.float32)

            bw = union_box[0, 2] - union_box[0, 0]
            bh = union_box[0, 3] - union_box[0, 1]
            union_box[0, 0] = max(0, union_box[0, 0] - bw * 0.15)
            union_box[0, 1] = max(0, union_box[0, 1] - bh * 0.15)
            union_box[0, 2] = min(w, union_box[0, 2] + bw * 0.15)
            union_box[0, 3] = min(h, union_box[0, 3] + bh * 0.15)

            seq_kpts = []
            for ff in frame_files:
                fp = os.path.join(img_folder, ff)
                fimg = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
                if fimg is None:
                    continue
                kpts, scores = detector.pose_model(fimg, bboxes=union_box)
                if len(kpts) > 0:
                    kp = kpts[0].copy()
                    sc = scores[0].copy()
                    kp[:, 0] /= w
                    kp[:, 1] /= h
                    seq_kpts.append(np.column_stack([kp, sc]))
                else:
                    if len(seq_kpts) > 0:
                        seq_kpts.append(seq_kpts[-1].copy())
                    else:
                        seq_kpts.append(np.zeros((26, 3), dtype=np.float32))

            if len(seq_kpts) < 4:
                fail += 1
                continue

            raw_kpts_arr = np.array(seq_kpts, dtype=np.float32)

            prep_res = RTMPosePreprocessor.process_raw_sequence(raw_kpts_arr, exercise_name="02_바벨_데드리프트")
            if prep_res is None:
                fail += 1
                continue

            sub_num = 1
            for d in range(1, 10):
                if f"피험자{d}" in folder_name or f"_{d}_" in folder_name:
                    sub_num = d
                    break
            is_correct = 1 if "[정상]" in folder_name or "정상" in folder_name else 0

            # 덮어쓰기 저장
            tmp_out = out_path + ".tmp.npz"
            np.savez_compressed(
                tmp_out,
                kps_26_64f=prep_res['kps_26_64f'],     # 표준 (64, 26, 3) - 골반 중심화 완료
                kps_15_64f=prep_res['kps_15_64f'],     # 1단계 호환 (64, 15, 3)
                kps_clean=prep_res['kps_clean'],
                raw_kpts=raw_kpts_arr,
                torso_scale=prep_res['torso_scale'],
                rot_angle_deg=prep_res['rot_angle_deg'],
                is_flipped=prep_res['is_flipped'],
                exercise="02_바벨_데드리프트",
                base_sequence_id=folder_name,
                sequence_id=f"{folder_name}_{vn}",
                view=vn,
                subject=sub_num,
                label=is_correct,
                raw_frames_count=len(frame_files)
            )
            if os.path.exists(out_path):
                os.remove(out_path)
            os.rename(tmp_out, out_path)

            success += 1

        except Exception as e:
            fail += 1

        if idx % 10 == 0 or idx == total:
            el = time.time() - t0
            fps = idx / max(el, 0.001)
            eta = (total - idx) / max(fps, 0.001)
            print(f"[{idx:3d}/{total:3d}] ({idx/total*100:5.1f}%) | 성공: {success} | 속도: {fps:.1f} seq/s | 경과: {el:.1f}s | 남은시간: {eta:.0f}s", flush=True)

    print(f"\n[+] 전수 재추출 완료! (소요 시간: {time.time() - t0:.1f}초, 성공: {success}/{total})", flush=True)

if __name__ == '__main__':
    main()

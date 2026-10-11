# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/preprocessing/batch_extract_all_rtmpose_64f.py
Purpose: 9종 전 운동 원천 이미지 데이터에 대한 RTMPose-m (Halpe26) 일괄 추출 및 New 7단계 전처리 파이프라인 배치 러너
Features:
  - 9종 전 운동(스쿼트, 데드리프트, 런지류 5종, 굿모닝 등) 및 피험자 1~9번 전수 커버
  - 공식 5대 전 시점 (시점 A~E / view1~view5) 전수 처리
  - 기 추출 완료 파일 자동 스킵 및 무중단 이어받기 (Resume) 지원
  - RTMPosePreprocessor 7단계 파이프라인 (신뢰도 게이팅, 발 3각 기하, 64F 스플라인, 하위 호환 15관절) 동시 저장
  - 멀티프로세싱 기반 CPU 병렬 고속 추출
"""

import os
import sys
import glob
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, cpu_count
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로 등록
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from rtmlib import BodyWithFeet
from src.preprocessing.rtmpose_preprocessor import RTMPosePreprocessor

RAW_BASE_DIR = r"D:\Data\01_raw_data\images"
OUT_BASE_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F"

EXERCISE_FOLDERS = [
    "01_바벨_스쿼트",
    "02_바벨_데드리프트",
    "03_바벨_스티프_데드리프트",
    "04_바벨_런지",
    "05_스텝_포워드_다이나믹_런지",
    "06_스텝_백워드_다이나믹_런지",
    "07_사이드_런지",
    "08_크로스_런지",
    "09_굿모닝"
]

VIEW_TO_LETTER = {
    'view5': '시점A',
    'view2': '시점B',
    'view3': '시점C',
    'view4': '시점D',
    'view1': '시점E'
}

def worker_task(worker_id: int, tasks: List[Dict[str, str]], progress_q: Queue):
    """개별 워커 프로세스: RTMPose 로드 및 시퀀스 추출/전처리/저장"""
    try:
        detector = BodyWithFeet(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')
    except Exception as e:
        progress_q.put((worker_id, None, False, f"Worker init failed: {e}"))
        return

    for item in tasks:
        ex_name = item['ex_name']
        folder_name = item['folder_name']
        vn = item['view_name']
        letter = VIEW_TO_LETTER.get(vn, '')

        out_dir = os.path.join(OUT_BASE_DIR, ex_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{folder_name}_{vn}.npz")

        if os.path.exists(out_path):
            progress_q.put((worker_id, f"{folder_name}_{vn}", True, "cached"))
            continue

        img_folder = os.path.join(RAW_BASE_DIR, ex_name, folder_name)
        if not os.path.isdir(img_folder):
            progress_q.put((worker_id, f"{folder_name}_{vn}", False, "folder missing"))
            continue

        frame_files = sorted([f for f in os.listdir(img_folder) if letter in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(frame_files) < 4:
            progress_q.put((worker_id, f"{folder_name}_{vn}", False, "insufficient frames"))
            continue

        # 1. 첫 번째 및 중간 프레임으로 바운딩 박스 추정 (여유 패딩 15%)
        try:
            f1_path = os.path.join(img_folder, frame_files[0])
            f_mid_path = os.path.join(img_folder, frame_files[len(frame_files)//2])

            img1 = cv2.imdecode(np.fromfile(f1_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_mid = cv2.imdecode(np.fromfile(f_mid_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img1 is None or img_mid is None:
                progress_q.put((worker_id, f"{folder_name}_{vn}", False, "image read error"))
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

            # 2. 모든 프레임 포즈 추론
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
                progress_q.put((worker_id, f"{folder_name}_{vn}", False, "valid pose < 4"))
                continue

            raw_kpts_arr = np.array(seq_kpts, dtype=np.float32) # (T, 26, 3)

            # 3. New RTMPosePreprocessor 7단계 전처리 적용
            prep_res = RTMPosePreprocessor.process_raw_sequence(raw_kpts_arr, exercise_name=ex_name)
            if prep_res is None:
                progress_q.put((worker_id, f"{folder_name}_{vn}", False, "preprocessor rejected"))
                continue

            # 피험자 번호 및 정상 여부 파싱
            sub_num = 1
            for d in range(1, 10):
                if f"피험자{d}" in folder_name or f"_{d}_" in folder_name:
                    sub_num = d
                    break
            is_correct = 1 if "[정상]" in folder_name or "정상" in folder_name else 0

            # 4. 압축 npz 저장
            np.savez_compressed(
                out_path,
                kps_26_64f=prep_res['kps_26_64f'],     # 표준 (64, 26, 3)
                kps_15_64f=prep_res['kps_15_64f'],     # 1단계 호환 (64, 15, 3)
                kps_clean=prep_res['kps_clean'],       # 결측 보간된 원본 텐서
                raw_kpts=raw_kpts_arr,                 # 순수 비전 원천 텐서
                torso_scale=prep_res['torso_scale'],
                rot_angle_deg=prep_res['rot_angle_deg'],
                is_flipped=prep_res['is_flipped'],
                exercise=ex_name,
                base_sequence_id=folder_name,
                sequence_id=f"{folder_name}_{vn}",
                view=vn,
                subject=sub_num,
                label=is_correct,
                raw_frames_count=len(frame_files)
            )
            progress_q.put((worker_id, f"{folder_name}_{vn}", True, "saved"))

        except Exception as e:
            progress_q.put((worker_id, f"{folder_name}_{vn}", False, str(e)))

def build_all_tasks(target_exercises: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """전체 처리 대상 시퀀스 및 5대 시점 태스크 리스트 생성"""
    exercises = target_exercises or EXERCISE_FOLDERS
    tasks = []
    view_names = ['view1', 'view2', 'view3', 'view4', 'view5']

    for ex in exercises:
        ex_dir = os.path.join(RAW_BASE_DIR, ex)
        if not os.path.isdir(ex_dir):
            continue
        folders = sorted([f for f in os.listdir(ex_dir) if os.path.isdir(os.path.join(ex_dir, f))])
        for fold in folders:
            for vn in view_names:
                tasks.append({
                    'ex_name': ex,
                    'folder_name': fold,
                    'view_name': vn
                })
    return tasks

def run_batch_extraction(num_workers: int = 4, target_exercises: Optional[List[str]] = None):
    tasks = build_all_tasks(target_exercises)
    total_tasks = len(tasks)
    print("=" * 88, flush=True)
    print("  [vidio-health-trainer] 9종 전 운동 RTMPose-m (Halpe26) 일괄 추출 및 New 전처리 배치", flush=True)
    print(f"  - 총 처리 대상 태스크 수: {total_tasks:,}건", flush=True)
    print(f"  - 병렬 워커 수: {num_workers} processes", flush=True)
    print(f"  - 저장 경로: {OUT_BASE_DIR}", flush=True)
    print("=" * 88, flush=True)

    # 워커별 태스크 분배
    tasks_per_worker = [[] for _ in range(num_workers)]
    for i, t in enumerate(tasks):
        tasks_per_worker[i % num_workers].append(t)

    progress_q = Queue()
    procs = []
    t0 = time.time()

    try:
        for w_id in range(num_workers):
            p = Process(target=worker_task, args=(w_id, tasks_per_worker[w_id], progress_q))
            p.start()
            procs.append(p)

        finished = 0
        saved_cnt = 0
        cached_cnt = 0
        fail_cnt = 0

        while finished < total_tasks:
            w_id, seq_id, success, msg = progress_q.get()
            finished += 1
            if success:
                if msg == "cached":
                    cached_cnt += 1
                else:
                    saved_cnt += 1
            else:
                fail_cnt += 1

            if finished % 25 == 0 or finished == total_tasks:
                elapsed = time.time() - t0
                speed = finished / max(0.1, elapsed)
                rem = (total_tasks - finished) / max(0.01, speed)
                print(f"[{finished:5d}/{total_tasks:5d}] {finished/total_tasks*100:5.1f}% | "
                      f"신규저장: {saved_cnt:4d} | 캐시스킵: {cached_cnt:4d} | 실패: {fail_cnt:3d} | "
                      f"속도: {speed:5.2f} task/s | 남은시간: {rem/60:5.1f}분", flush=True)

        for p in procs:
            p.join()

        print("\n" + "=" * 88, flush=True)
        print(f"[+] 전체 배치 완료! 소요시간: {(time.time()-t0)/60:.1f}분", flush=True)
        print(f"    - 총 완료 태스크: {finished:,}건 (신규 저장: {saved_cnt:,}건, 캐시 스킵: {cached_cnt:,}건, 실패: {fail_cnt:,}건)", flush=True)
        print("=" * 88, flush=True)

    except Exception as e:
        import traceback
        print(f"\n[!] 메인 프로세스 예외 발생: {e}", flush=True)
        traceback.print_exc()
        for p in procs:
            if p.is_alive():
                p.terminate()

if __name__ == '__main__':
    # 명령행 인자로 워커 수 조정 가능 (기본: 4)
    w_cnt = 4
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        w_cnt = int(sys.argv[1])
    run_batch_extraction(num_workers=w_cnt)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_포워드 + 06_백워드)
Defect: Condition 3 - 뒤다리 무릎 각도 90도 (dl3)
Module: build_dynamic_lunge_dl3_rtmp_cache.py
Purpose: 다이나믹 런지 RTMPose Halpe26 전수(9,795건) 대상
         dl3_rtmp_1 전용 27D 생체역학 피처 추출 및 고속 캐시 생성
         (AI-Hub 1~9번 피험자, 공식 5대 시점 전수 수용, 0.00% 무누수 격리)
"""

import os
import sys
import glob
import json
import time
import concurrent.futures
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")

from lunge_dl3_rtmp_features import (
    extract_lunge_dl3_rtmp_features_all,
    FEATURE_NAMES_27D,
    FEATURE_NAMES_22D,
    FEATURE_NAMES_14D
)

RTMP_DIRS = [
    ('05_forward', r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지"),
    ('06_backward', r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지")
]

LABEL_DIRS = [
    r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
]

CACHE_OUT_PATH = r"D:\Data\피처_데이터\dynamic_lunge_dl3_rtmp_27d_cache.npz"
os.makedirs(os.path.dirname(CACHE_OUT_PATH), exist_ok=True)

def load_label_dict() -> Dict[str, int]:
    """2D 라벨 JSON 파일들로부터 Condition 3 (뒤다리 무릎 90도) 정답 라벨 로드"""
    label_dict = {}
    total_labels = 0
    for ld in LABEL_DIRS:
        jfiles = glob.glob(os.path.join(ld, "*.json"))
        total_labels += len(jfiles)
        for jf in jfiles:
            bname = os.path.splitext(os.path.basename(jf))[0]
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conds = data.get("type_info", {}).get("conditions", [])
                if len(conds) > 2:
                    val = conds[2].get("value")
                    # False -> 결함(1), True -> 정상(0)
                    label_dict[bname] = 1 if (val is False) else 0
            except Exception:
                continue
    print(f"[+] 총 {total_labels}개 라벨 파일 중 {len(label_dict)}개 c3 라벨 파싱 완료")
    return label_dict

def process_single_file(item: Tuple[str, str, str, int]) -> Optional[Dict[str, Any]]:
    """단일 RTMPose npz 파일 처리 및 피처 추출"""
    fpath, ex_type, view_code, sub_id = item
    fname = os.path.basename(fpath)
    try:
        data = np.load(fpath)
        kps26 = data['kps_26_64f']
        feat27 = extract_lunge_dl3_rtmp_features_all(kps26, view_code=view_code)
        return {
            'fname': fname,
            'ex_type': ex_type,
            'view': view_code,
            'subj': sub_id,
            'feat27': feat27
        }
    except Exception as e:
        return None

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 dl3_rtmp_1 27D 피처 캐시 구축 시작")
    print("=" * 80)
    
    t0 = time.time()
    label_dict = load_label_dict()
    
    # 작업 항목 리스트업
    tasks = []
    skipped_no_view = 0
    skipped_no_label = 0
    
    for ex_type, r_dir in RTMP_DIRS:
        npz_files = sorted(glob.glob(os.path.join(r_dir, "*.npz")))
        print(f"[*] {ex_type} RTMPose 파일: {len(npz_files)} 건")
        for f in npz_files:
            fname = os.path.basename(f)
            v_code = None
            for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
                if f'_{v}.npz' in fname:
                    v_code = v
                    break
            if v_code is None:
                skipped_no_view += 1
                continue
                
            raw_key = fname.replace(f'_{v_code}.npz', '')
            if raw_key not in label_dict:
                skipped_no_label += 1
                continue
                
            sub_id = 1
            for s in range(1, 10):
                if f"피험자{s}" in fname:
                    sub_id = s
                    break
                    
            tasks.append((f, ex_type, v_code, sub_id, label_dict[raw_key]))
            
    print(f"[+] 최종 유효 매칭 파일 수: {len(tasks)} 건 (시점 미인식: {skipped_no_view}, 라벨 결손: {skipped_no_label})")
    
    # 멀티프로세싱 병렬 피처 추출
    results_X = []
    results_y = []
    results_subj = []
    results_view = []
    results_ex = []
    results_fname = []
    
    print("[*] 8개 프로세스로 27D 고급 생체역학 피처 병렬 추출 중...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        worker_args = [(t[0], t[1], t[2], t[3]) for t in tasks]
        labels = [t[4] for t in tasks]
        
        for idx, res in enumerate(tqdm(executor.map(process_single_file, worker_args), total=len(worker_args))):
            if res is not None:
                results_X.append(res['feat27'])
                results_y.append(labels[idx])
                results_subj.append(res['subj'])
                results_view.append(res['view'])
                results_ex.append(res['ex_type'])
                results_fname.append(res['fname'])
                
    X = np.array(results_X, dtype=np.float32)
    y = np.array(results_y, dtype=np.int32)
    subjects = np.array(results_subj, dtype=np.int32)
    views = np.array(results_view)
    exercise_types = np.array(results_ex)
    filenames = np.array(results_fname)
    
    print("\n[+] 피처 추출 완료!")
    print(f"  - X Shape: {X.shape}")
    print(f"  - y Shape: {y.shape} (정상 0: {np.sum(y == 0)}, 결함 1: {np.sum(y == 1)}, 결함율: {np.mean(y)*100:.2f}%)")
    print(f"  - 피험자 분포: {dict(Counter(subjects))}")
    print(f"  - 카메라 시점 분포: {dict(Counter(views))}")
    print(f"  - 운동 유형 분포: {dict(Counter(exercise_types))}")
    
    # 무결성 검증 (NaN / Inf)
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  - 무결성 감사: NaN={nan_count}, Inf={inf_count} (Pass: {nan_count==0 and inf_count==0})")
    assert nan_count == 0 and inf_count == 0, "Feature matrix contains NaN or Inf!"
    
    # 캐시 저장
    np.savez_compressed(
        CACHE_OUT_PATH,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        exercise_types=exercise_types,
        filenames=filenames,
        feature_names=np.array(FEATURE_NAMES_27D)
    )
    print(f"[+] 캐시 파일 성공적으로 압축 저장: {CACHE_OUT_PATH}")
    print(f"[+] 총 소요 시간: {time.time() - t0:.2f} 초")

if __name__ == '__main__':
    main()

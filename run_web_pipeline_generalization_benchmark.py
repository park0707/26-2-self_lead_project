# -*- coding: utf-8 -*-
"""
vidio-health-trainer
웹 서비스 실배포 E2E 파이프라인(ActivePeriodGater + 횟수 동적 분할 + 64F 스플라인 보간 + Stage 2 모델 전수 진단)
일반화 3대 운동 72개 클립 전수 벤치마크 고속 실행 스크립트
"""

import os
import sys
import json
import time
import glob
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter

BENCHMARK_OLD_JSON = r"D:\data\일반화\generalization_benchmark_results.json"
OUTPUT_JSON = r"D:\data\일반화\generalization_web_pipeline_benchmark_results.json"

EXERCISE_MAP = {
    "바벨 데드": 1,
    "스쿼트": 0,
    "스티프 데드": 2
}

def load_old_metadata():
    with open(BENCHMARK_OLD_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark():
    items = load_old_metadata()
    print(f"Loaded {len(items)} clips to benchmark with 1-Rep dynamic segmentation.")

    pipeline = VideoAnalysisPipeline.get_instance()
    stage2 = Stage2InferenceEngine.get_instance()

    results = []
    t_total_start = time.time()

    for idx, item in enumerate(items, start=1):
        ex_name = item["exercise_name"]
        ex_id = EXERCISE_MAP[ex_name]
        view_alias = item["view_alias"] # e.g. "시점 B"
        fn = item["filename"]
        
        # 실제 비디오 파일 경로
        sub_folder = item["rel_path"].split("\\")[1]
        vid_path = os.path.join(r"D:\data\일반화\분할_결과", ex_name, sub_folder, fn)
        
        if not os.path.exists(vid_path):
            print(f"[{idx:02d}/{len(items)}] ERROR: File not found: {vid_path}")
            continue

        print(f"[{idx:02d}/{len(items)}] {ex_name} | {view_alias} | {fn} ... ", end="", flush=True)
        t0 = time.time()

        # 1. 랜드마크 고속 추출
        vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
        t_ext = time.time() - t0

        # 2. 동적 횟수 구간 분할 (ActivePeriodGater)
        kps_15 = vmeta["kps_15_2d"]
        fps = vmeta["fps"]
        stride = vmeta["sample_stride"]
        orig_fps = vmeta["orig_fps"]
        total_orig_frames = vmeta["total_frames"]

        act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
        if not raw_reps:
            # 피크 미감지 시 전체를 단일 1회로 폴백
            raw_reps = [(0, len(kps_15) - 1, len(kps_15) // 2)]

        num_reps = len(raw_reps)

        # 3. 각 1회 반복별 64프레임 3차 스플라인 보간 및 Stage 2 진단
        kps_33 = vmeta["kps_33_3d"]
        rep_records = []

        for r_idx, (sf, ef, bf) in enumerate(raw_reps, start=1):
            rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
            rep_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)

            diag = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_64f, camera_view=view_alias)

            start_sec = round((sf * stride) / orig_fps, 2)
            end_sec = round(min(total_orig_frames - 1, ef * stride) / orig_fps, 2)

            rep_records.append({
                "rep_index": r_idx,
                "start_frame": int(sf * stride),
                "end_frame": int(min(total_orig_frames - 1, ef * stride)),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(max(0.1, end_sec - start_sec), 2),
                "is_clean": diag["is_clean"],
                "defects": [{
                    "defect_code": d["defect_code"],
                    "defect_name": d["defect_name"],
                    "probability": round(float(d["probability"]), 4),
                    "state": d["state"],
                    "state_label": d["state_label"]
                } for d in diag["defects"]]
            })

        t_diag = time.time() - t0

        # 4. 세트 수준 종합 집계
        defect_keys = list(item["defects"].keys())
        defect_summary = {}

        for d_key in defect_keys:
            true_label = item["defects"][d_key]["true"]
            probs = []
            states = []
            for r in rep_records:
                for d in r["defects"]:
                    if d["defect_code"] == d_key:
                        probs.append(d["probability"])
                        states.append(d["state"])

            if probs:
                max_prob = max(probs)
                avg_prob = sum(probs) / len(probs)
                # 세트 기준 판정: 1회라도 결함(state=1) 발생 시 세트 결함
                has_defect = any(s == 1 for s in states)
                has_warning = any(s == 2 for s in states)
                pred_binary = 1 if has_defect else 0
                final_state = 1 if has_defect else (2 if has_warning else 0)
            else:
                max_prob = 0.0
                avg_prob = 0.0
                pred_binary = 0
                final_state = 0

            defect_summary[d_key] = {
                "true": true_label,
                "pred": pred_binary,
                "max_prob": round(float(max_prob), 4),
                "avg_prob": round(float(avg_prob), 4),
                "state": final_state,
                "match": (true_label == pred_binary)
            }

        exact_match = all(v["match"] for v in defect_summary.values())

        # 로그 출력
        detected_defects = [k for k, v in defect_summary.items() if v["pred"] == 1]
        print(f"Done in {t_diag:.2f}s ({num_reps} reps | Defs: {detected_defects} | ExactMatch: {exact_match})")

        res_item = {
            "index": idx,
            "exercise_name": ex_name,
            "exercise_id": ex_id,
            "view_alias": view_alias,
            "view_code": item["view_code"],
            "filename": fn,
            "rel_path": os.path.relpath(vid_path, r"D:\data\일반화"),
            "total_frames": total_orig_frames,
            "inference_time_sec": round(t_diag, 2),
            "total_reps_detected": num_reps,
            "exact_match": exact_match,
            "defects_set_aggregated": defect_summary,
            "repetitions": rep_records
        }
        results.append(res_item)

        # 5개마다 중간 체크포인트 저장
        if idx % 5 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 최종 결과 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_total_start
    print(f"\n=======================================================")
    print(f"전수 72개 클립 1회 반복 분할 벤치마크 완료!")
    print(f"총 소요시간: {total_time:.1f}초 (평균 {total_time/len(items):.2f}초/클립)")
    print(f"결과 저장 위치: {OUTPUT_JSON}")
    print(f"=======================================================")

if __name__ == "__main__":
    run_benchmark()

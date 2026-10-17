# -*- coding: utf-8 -*-
"""
vidio-health-trainer
사용자 제안 신체 분할 단위 좌표계 vs 기존 기법 통합 실증 벤치마크
- Baseline: 무보정
- Test 1: 등방성 해상도 보정 (Isotropic)
- Test 3: 가상 수평 정렬 (Virtual Leveling)
- User Prop A: 사용자 제안 인체 단위 좌표계 (어깨 N등분 + 대퇴골 M등분)
- User Prop B: 사용자 제안 인체 단위 좌표계 (어깨 N등분 + 체간 M등분)
- User Prop A + Test 3: 인체 단위 좌표계 + 가상 수평 정렬 융합
"""

import os
import sys
import json
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.core.topology_adapter import SkeletalTopologyAdapter

from test1_isotropic_normalization import apply_isotropic_normalization
from test3_virtual_leveling import apply_virtual_leveling
from test_anatomical_unit_normalization import apply_anatomical_unit_normalization

OUTPUT_JSON = r"D:\data\일반화\anatomical_benchmark_results.json"

TARGET_CLIPS = [
    # 스쿼트 (시점 B - 우측 전방)
    {"ex_name": "스쿼트", "ex_id": 0, "view": "시점 B", "rel_path": r"스쿼트\우전방_단독\01_세트1_정상_전체세트5회_F0380_F0910.mp4", "intent": "정상"},
    {"ex_name": "스쿼트", "ex_id": 0, "view": "시점 B", "rel_path": r"스쿼트\우전방_단독\02_세트2_결함1_전체세트5회_F1000_F1500.mp4", "intent": "결함 1: 척추 중립"},
    {"ex_name": "스쿼트", "ex_id": 0, "view": "시점 B", "rel_path": r"스쿼트\우전방_단독\04_세트4_결함3_전체세트5회_F3750_F4250.mp4", "intent": "결함 3: 무릎-발 일치"},
    {"ex_name": "스쿼트", "ex_id": 0, "view": "시점 B", "rel_path": r"스쿼트\우전방_단독\05_세트5_결함4_전체세트5회_F4980_F5520.mp4", "intent": "결함 4: 발바닥 지면 고정"},
    
    # 바벨 데드리프트 (시점 B - 우측 전방)
    {"ex_name": "바벨 데드", "ex_id": 1, "view": "시점 B", "rel_path": r"바벨 데드\우정면\01_정상_전체세트3회_F0220_F0680.mp4", "intent": "정상"},
    {"ex_name": "바벨 데드", "ex_id": 1, "view": "시점 B", "rel_path": r"바벨 데드\우정면\02_결함1_전체세트3회_F0725_F1090.mp4", "intent": "결함 1: 척추 중립"},
    {"ex_name": "바벨 데드", "ex_id": 1, "view": "시점 B", "rel_path": r"바벨 데드\우정면\04_결함3_전체세트3회_F1400_F1690.mp4", "intent": "결함 3: 바벨 밀착"},
    # 바벨 데드리프트 (결함 5 - 우측 후방 / 시점 A)
    {"ex_name": "바벨 데드", "ex_id": 1, "view": "시점 A", "rel_path": r"바벨 데드\우후방\06_결함5_전체세트3회_F1950_F2340.mp4", "intent": "결함 5: 발바닥 고정"},
    
    # 스티프 데드리프트 (시점 A - 우측 후방)
    {"ex_name": "스티프 데드", "ex_id": 2, "view": "시점 A", "rel_path": r"스티프 데드\우후방\01_세트1_전체세트3회_F0400_F0680.mp4", "intent": "정상"},
    {"ex_name": "스티프 데드", "ex_id": 2, "view": "시점 A", "rel_path": r"스티프 데드\우후방\02_세트2_전체세트3회_F0700_F0950.mp4", "intent": "결함 1: 바벨 밀착"}
]

def run_benchmark():
    pipeline = VideoAnalysisPipeline.get_instance()
    stage2 = Stage2InferenceEngine.get_instance()
    base_dir = r"D:\data\일반화\분할_결과"
    
    all_results = []
    print("=" * 80)
    print(" 사용자 제안 인체 단위 정규화 vs 기존 기법 정밀 비교 벤치마크")
    print("=" * 80)
    
    for c_idx, item in enumerate(TARGET_CLIPS, 1):
        vid_path = os.path.join(base_dir, item["rel_path"])
        if not os.path.exists(vid_path):
            print(f"[{c_idx}/{len(TARGET_CLIPS)}] File not found: {vid_path}")
            continue
            
        ex_name = item["ex_name"]
        ex_id = item["ex_id"]
        view = item["view"]
        intent = item["intent"]
        fn = os.path.basename(vid_path)
        
        print(f"\n[{c_idx}/{len(TARGET_CLIPS)}] {ex_name} | {view} | {fn} ({intent}) ... ", end="", flush=True)
        t0 = time.time()
        
        vmeta = pipeline.extract_video_landmarks(vid_path, sample_stride=2)
        orig_w = vmeta.get("orig_w", 1080)
        orig_h = vmeta.get("orig_h", 1920)
        kps_15 = vmeta["kps_15_2d"]
        fps = vmeta["fps"]
        kps_33 = vmeta["kps_33_3d"]
        
        act_start, act_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(kps_15, fps)
        if not raw_reps:
            raw_reps = [(0, len(kps_15) - 1, len(kps_15) // 2)]
            
        print(f"Extracted ({len(raw_reps)} reps) in {time.time()-t0:.2f}s")
        
        rep_comparisons = []
        for r_idx, (sf, ef, bf) in enumerate(raw_reps, 1):
            rep_seq = kps_33[sf:max(sf + 4, ef + 1)]
            
            # 1. Baseline
            rep_base = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)
            diag_base = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_base, camera_view=view)
            
            # 2. Test 1 (등방성 해상도 보정)
            rep_t1 = apply_isotropic_normalization(rep_base, orig_w, orig_h)
            diag_t1 = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_t1, camera_view=view)
            
            # 3. Test 3 (가상 수평 정렬)
            rep_t3, est_tilt = apply_virtual_leveling(rep_base)
            diag_t3 = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_t3, camera_view=view)
            
            # 4. User Prop A (어깨 N등분 + 대퇴골 M등분)
            rep_user_a, scale_info_a = apply_anatomical_unit_normalization(rep_base, vertical_anchor="thigh")
            diag_user_a = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_user_a, camera_view=view)
            
            # 5. User Prop B (어깨 N등분 + 체간 M등분)
            rep_user_b, scale_info_b = apply_anatomical_unit_normalization(rep_base, vertical_anchor="torso")
            diag_user_b = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_user_b, camera_view=view)
            
            # 6. User Prop A + Test 3 (인체 단위 정규화 + 가상 수평 정렬)
            rep_user_hyb, _ = apply_virtual_leveling(rep_user_a)
            diag_user_hyb = stage2.diagnose(exercise=ex_id, keypoints_seq=rep_user_hyb, camera_view=view)
            
            rep_comparisons.append({
                "rep_index": r_idx,
                "scale_info_a": scale_info_a,
                "scale_info_b": scale_info_b,
                "estimated_tilt_deg": est_tilt,
                "baseline": diag_base["defects"],
                "test1_isotropic": diag_t1["defects"],
                "test3_virtual_leveling": diag_t3["defects"],
                "user_prop_a_thigh": diag_user_a["defects"],
                "user_prop_b_torso": diag_user_b["defects"],
                "user_prop_a_plus_leveling": diag_user_hyb["defects"]
            })
            
        clip_record = {
            "index": c_idx,
            "exercise_name": ex_name,
            "exercise_id": ex_id,
            "view": view,
            "intent": intent,
            "filename": fn,
            "reps_count": len(raw_reps),
            "reps": rep_comparisons
        }
        all_results.append(clip_record)
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n[OK] All comparisons saved to {OUTPUT_JSON}")
    return all_results

if __name__ == "__main__":
    run_benchmark()

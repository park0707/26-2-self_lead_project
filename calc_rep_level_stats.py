# -*- coding: utf-8 -*-
"""
vidio-health-trainer
총 256회 개별 반복(1-Repetition) 단위의 운동별 맞춘 개수 및 틀린 개수 정밀 집계 스크립트
"""

import json
import numpy as np

def run_analysis():
    with open(r"D:\data\일반화\generalization_web_pipeline_benchmark_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_clips = len(data)
    total_reps = sum(len(it["repetitions"]) for it in data)
    print(f"Total clips: {total_clips}, Total reps: {total_reps}")

    stats = {}

    for it in data:
        ex = it["exercise_name"]
        if ex not in stats:
            stats[ex] = {
                "clips": 0,
                "reps": 0,
                "slot_total": 0,
                "slot_correct": 0,
                "slot_wrong": 0,
                "exact_rep_match": 0,
                "exact_rep_mismatch": 0,
                "target_hit": 0,
                "target_miss": 0,
                "defect_slots": {}
            }
        stats[ex]["clips"] += 1
        
        true_dict = {k: v["true"] for k, v in it["defects_set_aggregated"].items()}
        true_target_keys = [k for k, v in true_dict.items() if v == 1]
        
        for r in it["repetitions"]:
            stats[ex]["reps"] += 1
            
            # 각 결함별 판정 결과 (state 1이면 결함)
            rep_pred_dict = {d["defect_code"]: (1 if d["state"] == 1 else 0) for d in r["defects"]}
            
            # 1. 개별 결함 슬롯별 채점
            rep_all_correct = True
            for dk, y_true in true_dict.items():
                if dk not in stats[ex]["defect_slots"]:
                    stats[ex]["defect_slots"][dk] = {"total": 0, "correct": 0, "wrong": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
                
                stats[ex]["slot_total"] += 1
                stats[ex]["defect_slots"][dk]["total"] += 1
                
                y_pred = rep_pred_dict.get(dk, 0)
                if y_true == y_pred:
                    stats[ex]["slot_correct"] += 1
                    stats[ex]["defect_slots"][dk]["correct"] += 1
                    if y_true == 1:
                        stats[ex]["defect_slots"][dk]["tp"] += 1
                    else:
                        stats[ex]["defect_slots"][dk]["tn"] += 1
                else:
                    stats[ex]["slot_wrong"] += 1
                    stats[ex]["defect_slots"][dk]["wrong"] += 1
                    rep_all_correct = False
                    if y_true == 1 and y_pred == 0:
                        stats[ex]["defect_slots"][dk]["fn"] += 1
                    else:
                        stats[ex]["defect_slots"][dk]["fp"] += 1
                    
            # 2. 1회 반복 전체 완벽 일치 (Exact Match)
            if rep_all_correct:
                stats[ex]["exact_rep_match"] += 1
            else:
                stats[ex]["exact_rep_mismatch"] += 1
                
            # 3. 당초 의도 결함 적중 (Target Hit)
            if not true_target_keys:
                # 정상 의도 세트: 아무런 결함도 감지되지 않아야 적중
                if all(v == 0 for v in rep_pred_dict.values()):
                    stats[ex]["target_hit"] += 1
                else:
                    stats[ex]["target_miss"] += 1
            else:
                # 결함 의도 세트: 의도한 결함이 실제로 감지되었으면 적중 (동반 결함 발생 여부와 무관)
                if all(rep_pred_dict.get(tk, 0) == 1 for tk in true_target_keys):
                    stats[ex]["target_hit"] += 1
                else:
                    stats[ex]["target_miss"] += 1

    print("\n" + "=" * 90)
    print(" [1] 256회 반복 단위 3대 운동별 정밀 채점 집계")
    print("=" * 90)

    tot_reps_all = 0
    tot_slot_corr = 0
    tot_slot_wrong = 0
    tot_slot_all = 0
    tot_exact_match = 0
    tot_target_hit = 0

    for ex in ["바벨 데드", "스쿼트", "스티프 데드"]:
        s = stats[ex]
        tot_reps_all += s["reps"]
        tot_slot_corr += s["slot_correct"]
        tot_slot_wrong += s["slot_wrong"]
        tot_slot_all += s["slot_total"]
        tot_exact_match += s["exact_rep_match"]
        tot_target_hit += s["target_hit"]
        
        print(f"\n▶ [{ex}] (총 {s['reps']}회 반복, 클립 {s['clips']}개)")
        print(f"   ① 결함 판정 슬롯 기준 (전체 {s['slot_total']}개 슬롯):")
        print(f"      - 맞춘 개수: {s['slot_correct']}개 ({s['slot_correct']/s['slot_total']*100:.1f}%)")
        print(f"      - 틀린 개수: {s['slot_wrong']}개 ({s['slot_wrong']/s['slot_total']*100:.1f}%)")
        print(f"   ② 1회 반복 전체 완벽 일치 (Exact Match, {s['reps']}회 중):")
        print(f"      - 맞춘 횟수: {s['exact_rep_match']}회 ({s['exact_rep_match']/s['reps']*100:.1f}%)")
        print(f"      - 틀린 횟수: {s['exact_rep_mismatch']}회 ({s['exact_rep_mismatch']/s['reps']*100:.1f}%)")
        print(f"   ③ 의도 결함 적중 (Target Hit - 동반 결함 허용, {s['reps']}회 중):")
        print(f"      - 적중 횟수: {s['target_hit']}회 ({s['target_hit']/s['reps']*100:.1f}%)")
        print(f"      - 미적중 횟수: {s['target_miss']}회 ({s['target_miss']/s['reps']*100:.1f}%)")

    print("\n" + "=" * 90)
    print(" [2] 전체 256회 총합")
    print("=" * 90)
    print(f"총 반복 횟수: {tot_reps_all}회")
    print(f"1) 결함 슬롯 기준 (총 {tot_slot_all}개): 맞춘 개수 {tot_slot_corr}개 ({tot_slot_corr/tot_slot_all*100:.1f}%) / 틀린 개수 {tot_slot_wrong}개 ({tot_slot_wrong/tot_slot_all*100:.1f}%)")
    print(f"2) 1회 완벽 일치 기준 (총 {tot_reps_all}회): 맞춘 횟수 {tot_exact_match}회 ({tot_exact_match/tot_reps_all*100:.1f}%) / 틀린 횟수 {tot_reps_all-tot_exact_match}회")
    print(f"3) 의도 결함 적중 기준 (총 {tot_reps_all}회): 적중 횟수 {tot_target_hit}회 ({tot_target_hit/tot_reps_all*100:.1f}%) / 미적중 횟수 {tot_reps_all-tot_target_hit}회")

    print("\n" + "=" * 90)
    print(" [3] 세부 결함 항목별 256회 반복 슬롯 채점표")
    print("=" * 90)
    for ex in ["바벨 데드", "스쿼트", "스티프 데드"]:
        print(f"\n--- {ex} ---")
        for dk, dv in sorted(stats[ex]["defect_slots"].items()):
            acc = dv["correct"] / dv["total"] * 100
            print(f"  * {dk:4s} (총 {dv['total']:3d}회): 맞춤 {dv['correct']:3d}개 ({acc:5.1f}%) [TP={dv['tp']:2d}, TN={dv['tn']:2d}] | 틀림 {dv['wrong']:3d}개 [FP={dv['fp']:2d}, FN={dv['fn']:2d}]")

if __name__ == "__main__":
    run_analysis()

# -*- coding: utf-8 -*-
import json

with open(r"D:\data\일반화\anatomical_benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for c in data:
    c_idx = c['index']
    ex = c['exercise_name']
    view = c['view']
    intent = c['intent']
    reps = c['reps']
    print(f"\n=== Clip {c_idx}: {ex} | {view} | {intent} ({len(reps)} reps) ===")
    for r in reps:
        d_base = {d['defect_code']: d['probability'] for d in r['baseline']}
        d_t1 = {d['defect_code']: d['probability'] for d in r['test1_isotropic']}
        d_t3 = {d['defect_code']: d['probability'] for d in r['test3_virtual_leveling']}
        d_prop = {d['defect_code']: d['probability'] for d in r['user_prop_a_thigh']}
        d_hyb = {d['defect_code']: d['probability'] for d in r['user_prop_a_plus_leveling']}
        print(f"  [Rep {r['rep_index']}]")
        for code in d_base:
            p_b = d_base[code] * 100
            p_t1 = d_t1[code] * 100
            p_t3 = d_t3[code] * 100
            p_pr = d_prop[code] * 100
            p_hy = d_hyb[code] * 100
            print(f"    {code:<4}: Base={p_b:5.1f}% | Test1={p_t1:5.1f}% | Test3={p_t3:5.1f}% | PropA={p_pr:5.1f}% | PropA+3={p_hy:5.1f}%")

# -*- coding: utf-8 -*-
import json
import sys

with open(r'D:\data\일반화\generalization_web_pipeline_benchmark_results.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

view_stats = {}
for it in new_data:
    v = it['view_alias']
    if v not in view_stats:
        view_stats[v] = {'clips': 0, 'slots': 0, 'correct_any': 0, 'correct_maj': 0}
    view_stats[v]['clips'] += 1
    reps = it['repetitions']
    num_reps = len(reps)
    for dk, dv in it['defects_set_aggregated'].items():
        view_stats[v]['slots'] += 1
        yt = dv['true']
        ya = dv['pred']
        if yt == ya:
            view_stats[v]['correct_any'] += 1
        rep_states = [rd['state'] for r in reps for rd in r['defects'] if rd['defect_code'] == dk]
        num_def = sum(1 for s in rep_states if s == 1)
        ym = 1 if (num_reps > 0 and num_def >= (num_reps / 2.0)) else 0
        if yt == ym:
            view_stats[v]['correct_maj'] += 1

print("=== 카메라 시점별 정확도 ===")
for v, st in sorted(view_stats.items()):
    c_any = st['correct_any']
    c_maj = st['correct_maj']
    tot = st['slots']
    print(f"{v:8s} ({st['clips']:2d}클립, {tot:3d}슬롯) | 단일반복: {c_any:2d}맞춤/{tot-c_any:2d}틀림 ({c_any/tot*100:4.1f}%) | 과반수: {c_maj:2d}맞춤/{tot-c_maj:2d}틀림 ({c_maj/tot*100:4.1f}%)")

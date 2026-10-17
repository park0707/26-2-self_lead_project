# -*- coding: utf-8 -*-
import json
import sys

with open(r'D:\data\일반화\generalization_web_pipeline_benchmark_results.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

print(f"{'View':7s} | {'Filename':32s} | True | Pred | AvgP  | MaxP  | Reps detail")
print("-" * 105)

for it in new_data:
    if it['exercise_name'] != '스쿼트':
        continue
    fn = it['filename']
    view = it['view_alias']
    c4_agg = it['defects_set_aggregated']['c4']
    rep_parts = []
    for r in it['repetitions']:
        for d in r['defects']:
            if d['defect_code'] == 'c4':
                rep_parts.append(f"R{r['rep_index']}:{d['probability']*100:.1f}%(S{d['state']})")
    
    r_str = " ".join(rep_parts)
    print(f"{view:7s} | {fn[:32]:32s} |  {c4_agg['true']}   |  {c4_agg['pred']}   | {c4_agg['avg_prob']*100:4.1f}% | {c4_agg['max_prob']*100:4.1f}% | {r_str}")

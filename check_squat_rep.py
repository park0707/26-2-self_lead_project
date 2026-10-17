# -*- coding: utf-8 -*-
import json

with open(r'D:\data\일반화\generalization_web_pipeline_benchmark_results.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

c = new_data[28]
print(c['exercise_name'], c['filename'], c['view_alias'])
print('Total reps:', len(c['repetitions']))
for r in c['repetitions']:
    c4 = [d for d in r['defects'] if d['defect_code']=='c4'][0]
    c1 = [d for d in r['defects'] if d['defect_code']=='c1'][0]
    c2 = [d for d in r['defects'] if d['defect_code']=='c2'][0]
    c3 = [d for d in r['defects'] if d['defect_code']=='c3'][0]
    print(f"Rep {r['rep_index']}: [{r['start_sec']}s ~ {r['end_sec']}s] len={r['duration_sec']}s | c1:{c1['probability']*100:.1f}% c2:{c2['probability']*100:.1f}% c3:{c3['probability']*100:.1f}% c4:{c4['probability']*100:.1f}%")

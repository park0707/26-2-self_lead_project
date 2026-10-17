# -*- coding: utf-8 -*-
import json
import sys
import numpy as np

def get_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp, fp, fn, tn, acc, prec, rec, f1

with open(r'D:\data\일반화\generalization_web_pipeline_benchmark_results.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open(r'D:\data\일반화\generalization_benchmark_results.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

old_map = {it['filename']: it for it in old_data}

defect_map = {}
for it in new_data:
    ex = it['exercise_name']
    fn = it['filename']
    oit = old_map[fn]
    reps = it['repetitions']
    num_reps = len(reps)
    for dk, dv in it['defects_set_aggregated'].items():
        full_k = f"{ex}_{dk}"
        if full_k not in defect_map:
            defect_map[full_k] = {'true': [], 'old': [], 'new_any': [], 'new_maj': [], 'new_avg': []}
        yt = dv['true']
        yo = oit['defects'][dk]['pred']
        yna = dv['pred']
        rep_states = [rd['state'] for r in reps for rd in r['defects'] if rd['defect_code'] == dk]
        num_def = sum(1 for s in rep_states if s == 1)
        ynm = 1 if (num_reps > 0 and num_def >= (num_reps / 2.0)) else 0
        avg_p = dv['avg_prob']
        yav = 1 if avg_p >= 0.50 else 0
        defect_map[full_k]['true'].append(yt)
        defect_map[full_k]['old'].append(yo)
        defect_map[full_k]['new_any'].append(yna)
        defect_map[full_k]['new_maj'].append(ynm)
        defect_map[full_k]['new_avg'].append(yav)

print(f"{'Defect Item':25s} | {'Pos':3s} | {'Case A (Old)':25s} | {'Case B (Any-Rep)':25s} | {'Case B (Majority)':25s}")
print("-" * 115)
for k, v in sorted(defect_map.items()):
    tp_o, fp_o, fn_o, tn_o, _, _, rec_o, f1_o = get_metrics(v['true'], v['old'])
    tp_a, fp_a, fn_a, tn_a, _, _, rec_a, f1_a = get_metrics(v['true'], v['new_any'])
    tp_m, fp_m, fn_m, tn_m, _, _, rec_m, f1_m = get_metrics(v['true'], v['new_maj'])
    pos = tp_o + fn_o
    old_str = f"Rec={rec_o*100:3.0f}% FP={fp_o:2d} F1={f1_o*100:4.1f}%"
    any_str = f"Rec={rec_a*100:3.0f}% FP={fp_a:2d} F1={f1_a*100:4.1f}%"
    maj_str = f"Rec={rec_m*100:3.0f}% FP={fp_m:2d} F1={f1_m*100:4.1f}%"
    print(f"{k:25s} | {pos:3d} | {old_str:25s} | {any_str:25s} | {maj_str:25s}")

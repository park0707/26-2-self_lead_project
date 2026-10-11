# -*- coding: utf-8 -*-
import json

with open(r'D:\src\models\second_models\babel_deadlift\weights\d2_pruning_ablation_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for k, v in d.items():
    s = v['3state']
    cm = v['cm']
    print(f"=== {k} ===")
    print(f"Features ({v['n_features']}개): {v['features']}")
    print(f"AUC: {v['auc']:.4f} | F1: {s['f1']:.2f}% | Prec: {s['prec']:.2f}% | Rec: {s['rec']:.2f}% | Acc: {s['acc']:.2f}% | Cov: {s['cov']:.2f}%")
    print(f"CM: TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']} | Win: {s['win']}")
    view_strs = [f"{vk}: F1={vv['f1']:.2f}% (Prec={vv['prec']:.1f}%, Rec={vv['rec']:.1f}%, Cov={vv['cov']:.1f}%)" for vk, vv in v['views'].items()]
    print("Views: " + " | ".join(view_strs))
    print()

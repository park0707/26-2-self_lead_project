# -*- coding: utf-8 -*-
import json

for ver in ['v4', 'v5', 'v6']:
    path = f'bl2_{ver}_benchmark_results.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    o = data['overall_3state']
    macro_f1 = (o['selective_defect_f1'] + o['selective_clean_f1']) / 2.0
    print(f"=== {ver.upper()} ===")
    print(f"Thresholds: tau_normal={data.get('tau_normal', 'N/A')}, tau_defect={data.get('tau_defect', 'N/A')}")
    print(f"Coverage: {o['coverage']*100:.2f}%, Retake: {o['retake_rate']*100:.2f}%")
    print(f"Selective Defect F1: {o['selective_defect_f1']*100:.2f}%, Clean F1: {o['selective_clean_f1']*100:.2f}%, Macro F1: {macro_f1*100:.2f}%")
    print(f"Selective Acc: {o['selective_accuracy']*100:.2f}%, Prec: {o['selective_precision']*100:.2f}%, Recall: {o['selective_recall']*100:.2f}%")
    print(f"Full Coverage Defect F1: {o['full_coverage_defect_f1']*100:.2f}%, AUC: {o['roc_auc']*100:.2f}%")
    print("Views Selective Defect F1:")
    for v, info in data['view_benchmarks'].items():
        vm = info['metrics']
        print(f"  - {v} ({info.get('alias', v)}): F1={vm['selective_defect_f1']*100:.2f}%, Acc={vm['selective_accuracy']*100:.2f}%, Rec={vm['selective_recall']*100:.2f}%")
    print()

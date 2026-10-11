# -*- coding: utf-8 -*-
import json
import sys

files = {
    'c1_v1': r'D:\src\models\second_models\side_lunge\sl1_v1_evaluation_results.json',
    'c1_v2': r'D:\src\models\second_models\side_lunge\sl1_v2_evaluation_results.json',
    'c1_v3': r'D:\src\models\second_models\side_lunge\sl1_v3_evaluation_results.json',
    'c2_v1': r'D:\src\models\second_models\side_lunge\sl2_v1_benchmark_results.json',
    'c2_v2': r'D:\src\models\second_models\side_lunge\sl2_v2_evaluation_results.json',
    'c2_v3': r'D:\src\models\second_models\side_lunge\sl2_v3_evaluation_results.json',
    'c3_v1': r'D:\src\models\second_models\side_lunge\sl3_v1_evaluation_results.json',
    'c3_v2': r'D:\src\models\second_models\side_lunge\sl3_v2_evaluation_results.json',
    'c4_v1': r'D:\src\models\second_models\side_lunge\sl4_evaluation_results.json',
    'c4_v2': r'D:\src\models\second_models\side_lunge\sl4_v2_evaluation_results.json',
    'c4_v3': r'D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json',
    'c5_v5': r'D:\src\models\second_models\side_lunge\sl5_v5_all_relabel_benchmark_results.json',
    'c5_v6': r'D:\src\models\second_models\side_lunge\sl5_v6_benchmark_results.json',
}

for k, path in files.items():
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f'=== {k} ({path}) ===')
            for key in ['best_3state', 'optimal_3state', 'full_coverage', 'view_audit', 'view_metrics']:
                if key in data:
                    val = data[key]
                    if isinstance(val, dict):
                        summary = {subk: round(subv, 4) if isinstance(subv, float) else subv for subk, subv in val.items() if subk not in ['cm', 'confusion_matrix']}
                        print(f'  {key}: {summary}')
            if 'auc' in data or 'f1' in data or 'accuracy' in data:
                top_metrics = {m: round(data[m], 4) for m in ['f1', 'auc', 'accuracy', 'precision', 'recall'] if m in data}
                print(f'  top: {top_metrics}')
    except Exception as e:
        print(f'Error reading {k}: {e}')

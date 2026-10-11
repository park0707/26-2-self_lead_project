# -*- coding: utf-8 -*-
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def parse_metrics(name, p):
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    res = {'name': name}
    if 'two_state_metrics' in d:
        m2 = d['two_state_metrics']
        m3 = d['three_state_metrics']
        res['f1_2s'] = m2.get('f1')
        res['auc'] = m2.get('auc')
        res['acc_2s'] = m2.get('acc') or m2.get('accuracy')
        res['f1_3s'] = m3.get('f1')
        res['cov_3s'] = m3.get('coverage')
        res['rej_3s'] = m3.get('rejection_rate')
        res['acc_3s'] = m3.get('accuracy')
    elif 'gbdt_2state_metrics' in d:
        m2 = d['gbdt_2state_metrics']
        m3 = d['adaptive_3state_metrics']
        res['f1_2s'] = m2.get('f1')
        res['auc'] = m2.get('auc')
        res['acc_2s'] = m2.get('accuracy')
        res['f1_3s'] = m3.get('f1')
        res['cov_3s'] = m3.get('coverage')
        res['rej_3s'] = m3.get('rejection_rate')
        res['acc_3s'] = m3.get('accuracy')
    elif 'ensemble_2state' in d:
        m2 = d['ensemble_2state']
        m3 = d['ensemble_3state']
        res['f1_2s'] = m2.get('f1')
        res['auc'] = m2.get('auc')
        res['acc_2s'] = m2.get('acc') or m2.get('accuracy')
        res['f1_3s'] = m3.get('f1')
        res['cov_3s'] = m3.get('coverage')
        res['rej_3s'] = m3.get('abstention_rate') or m3.get('rejection_rate')
        res['acc_3s'] = m3.get('acc') or m3.get('accuracy')
    elif 'full_coverage' in d:
        m2 = d['full_coverage']
        m3 = d['best_3state']
        res['f1_2s'] = m2.get('f1')
        res['auc'] = m2.get('auc')
        res['acc_2s'] = m2.get('accuracy')
        res['f1_3s'] = m3.get('f1')
        res['cov_3s'] = m3.get('coverage')
        res['rej_3s'] = m3.get('rejection_rate')
        res['acc_3s'] = m3.get('accuracy')
    return res

targets = [
    ('sl1_v2 (이전)', r'D:\src\models\second_models\side_lunge\sl1_v2_evaluation_results.json'),
    ('sl1_v3 (최신)', r'D:\src\models\second_models\side_lunge\sl1_v3_evaluation_results.json'),
    ('sl2_v2 (이전)', r'D:\src\models\second_models\side_lunge\sl2_v2_evaluation_results.json'),
    ('sl2_v3 (최신)', r'D:\src\models\second_models\side_lunge\sl2_v3_evaluation_results.json'),
    ('sl3_v1 (이전)', r'D:\src\models\second_models\side_lunge\sl3_v1_evaluation_results.json'),
    ('sl3_v2 (최신)', r'D:\src\models\second_models\side_lunge\sl3_v2_evaluation_results.json'),
    ('sl4_v2 (이전)', r'D:\src\models\second_models\side_lunge\sl4_v2_evaluation_results.json'),
    ('sl4_v3 (최신)', r'D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json'),
    ('sl5_v5 (이전)', r'D:\src\models\second_models\side_lunge\sl5_v5_all_relabel_benchmark_results.json'),
    ('sl5_v6 (최신)', r'D:\src\models\second_models\side_lunge\sl5_v6_benchmark_results.json'),
]

print(f"{'서브모델 (버전)':<20} | {'2S F1':<8} | {'AUC':<8} | {'3S F1':<8} | {'Coverage':<9} | {'3S ACC':<8}")
print("-" * 75)
for label, p in targets:
    m = parse_metrics(label, p)
    f1_2s = f"{m.get('f1_2s', 0)*100:6.2f}%" if m.get('f1_2s') is not None else "   N/A "
    auc = f"{m.get('auc', 0)*100:6.2f}%" if m.get('auc') is not None else "   N/A "
    f1_3s = f"{m.get('f1_3s', 0)*100:6.2f}%" if m.get('f1_3s') is not None else "   N/A "
    cov_3s = f"{m.get('cov_3s', 0)*100:6.2f}%" if m.get('cov_3s') is not None else "   N/A  "
    acc_3s = f"{m.get('acc_3s', 0)*100:6.2f}%" if m.get('acc_3s') is not None else "   N/A "
    print(f"{m['name']:<20} | {f1_2s:<8} | {auc:<8} | {f1_3s:<8} | {cov_3s:<9} | {acc_3s:<8}")

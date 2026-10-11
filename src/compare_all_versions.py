# -*- coding: utf-8 -*-
import json

def get_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    # 3-state f1, cov, auc, full f1
    f1_3s = None
    cov_3s = None
    f1_2s = None
    auc = None
    acc_3s = None
    
    # check keys
    if 'best_3state' in d:
        b = d['best_3state']
        f1_3s = b.get('f1')
        cov_3s = b.get('coverage')
        acc_3s = b.get('accuracy')
    elif 'optimal_3state' in d:
        b = d['optimal_3state']
        f1_3s = b.get('f1_score') or b.get('f1')
        cov_3s = b.get('coverage')
        acc_3s = b.get('accuracy')
    
    if 'full_coverage' in d:
        fc = d['full_coverage']
        f1_2s = fc.get('f1')
        auc = fc.get('auc')
    elif 'auc' in d:
        auc = d.get('auc')
        f1_2s = d.get('f1') or d.get('2state_f1')
    
    return {
        'f1_3s': f1_3s,
        'cov_3s': cov_3s,
        'acc_3s': acc_3s,
        'f1_2s': f1_2s,
        'auc': auc
    }

models = [
    ('c1 (앞무릎 90도)', 'sl1_v1', r'D:\src\models\second_models\side_lunge\sl1_v1_evaluation_results.json'),
    ('c1 (앞무릎 90도)', 'sl1_v2', r'D:\src\models\second_models\side_lunge\sl1_v2_evaluation_results.json'),
    ('c1 (앞무릎 90도)', 'sl1_v3 (c3공간통합)', r'D:\src\models\second_models\side_lunge\sl1_v3_evaluation_results.json'),
    ('c2 (무릎-발 방향)', 'sl2_v1', r'D:\src\models\second_models\side_lunge\sl2_v1_benchmark_results.json'),
    ('c2 (무릎-발 방향)', 'sl2_v2', r'D:\src\models\second_models\side_lunge\sl2_v2_evaluation_results.json'),
    ('c2 (무릎-발 방향)', 'sl2_v3 (c3공간통합)', r'D:\src\models\second_models\side_lunge\sl2_v3_evaluation_results.json'),
    ('c3 (안굽히는 다리)', 'sl3_v1 (구버전)', r'D:\src\models\second_models\side_lunge\sl3_v1_evaluation_results.json'),
    ('c3 (안굽히는 다리)', 'sl3_v2 (공간거리 챔피언)', r'D:\src\models\second_models\side_lunge\sl3_v2_evaluation_results.json'),
    ('c4 (척추 중립)', 'sl4_v1', r'D:\src\models\second_models\side_lunge\sl4_evaluation_results.json'),
    ('c4 (척추 중립)', 'sl4_v2', r'D:\src\models\second_models\side_lunge\sl4_v2_evaluation_results.json'),
    ('c4 (척추 중립)', 'sl4_v3 (c3공간통합)', r'D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json'),
    ('c5 (상체 숙임/젖힘)', 'sl5_v5 (재라벨링)', r'D:\src\models\second_models\side_lunge\sl5_v5_all_relabel_benchmark_results.json'),
    ('c5 (상체 숙임/젖힘)', 'sl5_v6 (c3공간통합)', r'D:\src\models\second_models\side_lunge\sl5_v6_benchmark_results.json'),
]

print(f"{'서브모델':<20} | {'버전':<22} | {'2S F1':<8} | {'AUC':<8} | {'3S F1':<8} | {'Coverage':<8}")
print("-" * 85)
for name, ver, p in models:
    s = get_summary(p)
    f2 = f"{s['f1_2s']*100:.2f}%" if s['f1_2s'] is not None else "N/A"
    auc = f"{s['auc']*100:.2f}%" if s['auc'] is not None else "N/A"
    f3 = f"{s['f1_3s']*100:.2f}%" if s['f1_3s'] is not None else "N/A"
    cov = f"{s['cov_3s']*100:.2f}%" if s['cov_3s'] is not None else "N/A"
    print(f"{name:<20} | {ver:<22} | {f2:<8} | {auc:<8} | {f3:<8} | {cov:<8}")

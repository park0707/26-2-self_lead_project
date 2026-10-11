import json
with open(r'D:\src\models\second_models\babel_stiff_deadlift\sd1_rtmp_2_evaluation_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for vk, vr in d['view_results'].items():
    print(f"{vr['alias']}: F1 = {vr['f1_3s']}% (AUC = {vr['auc']}%, Coverage = {vr['coverage']}%, Retake = {vr['retake']}%)")

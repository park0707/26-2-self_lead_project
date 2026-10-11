import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'D:\src\models\second_models\babel_stiff_deadlift\sd1_hybrid_1_benchmark_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

print(f"{'No':<3} | {'모델 구성':<48} | {'피처수':<6} | {'Binary F1':<10} | {'ROC-AUC':<8} | {'3-State F1':<11} | {'Coverage':<9} | {'Precision':<9} | {'Recall':<8}")
print("-" * 130)

for idx, (k, v) in enumerate(res.items(), 1):
    name = v['config_name']
    dim = v['features_dim']
    bf1 = f"{v['binary_f1']:.2f}%"
    auc = f"{v['roc_auc']:.2f}%"
    sf1 = f"{v['selective_defect_f1']:.2f}%"
    cov = f"{v['coverage']:.2f}%"
    pr = f"{v['precision']:.2f}%"
    rc = f"{v['recall']:.2f}%"
    print(f"{idx:<3} | {name:<48} | {dim:<6} | {bf1:<10} | {auc:<8} | {sf1:<11} | {cov:<9} | {pr:<9} | {rc:<8}")

print("\n" + "=" * 130)
print("시점별 Defect F1 상세 비교:")
print("=" * 130)

view_keys = ['view5', 'view2', 'view3', 'view4', 'view1']
header_views = " | ".join([f"{res['mediapipe_24d']['view_benchmarks'][vk]['alias']:<22}" for vk in view_keys])
print(f"{'No':<3} | {'모델 구성':<36} | {header_views}")
print("-" * 155)

for idx, (k, v) in enumerate(res.items(), 1):
    name = v['config_name'][:35]
    row_views = []
    for vk in view_keys:
        vb = v['view_benchmarks'][vk]
        f1_str = f"{vb['selective_defect_f1']:.1f}% (AUC {vb['roc_auc']:.1f}%)"
        row_views.append(f"{f1_str:<22}")
    print(f"{idx:<3} | {name:<36} | {' | '.join(row_views)}")

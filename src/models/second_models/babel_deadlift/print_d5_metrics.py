import json

with open('d5_model_v6_evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== d5 원천 데이터 3-State 성능 (v6.0 기준) ===")
opt = data.get('optimal_3state')
if opt:
    print(f"F1-Score: {opt['f1_score']*100:.2f}%")
    print(f"Accuracy: {opt['accuracy']*100:.2f}%")
    print(f"Precision: {opt['precision']*100:.2f}%")
    print(f"Recall: {opt['recall']*100:.2f}%")
    print(f"Coverage: {opt['coverage']*100:.2f}%")

print("\n=== 시점 별 성능 (view_breakdown) ===")
for view, m in data.get('view_breakdown', {}).items():
    alias = m.get('alias', view)
    f1 = m.get('f1_score', 0) * 100
    acc = m.get('accuracy', 0) * 100
    prec = m.get('precision', 0) * 100
    rec = m.get('recall', 0) * 100
    cov = m.get('coverage', 0)
    
    # 3-state coverage가 있는지 확인, 없으면 그냥 Binary
    cov_str = f"Cov {cov*100:.2f}%" if cov else "Binary/Two-State"
    print(f"{alias}: F1 {f1:.2f}%, Acc {acc:.2f}%, Prec {prec:.2f}%, Rec {rec:.2f}%, {cov_str}")

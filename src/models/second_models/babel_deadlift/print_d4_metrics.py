import json

with open('d4_v6_3state_final_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for view, m in data['views_3state'].items():
    alias = m['alias']
    samples = m['samples']
    opt = m.get('optimal_3state', m.get('standard_3state'))
    f1 = opt['f1'] * 100
    acc = opt['accuracy'] * 100
    cov = opt['coverage'] * 100
    prec = opt['precision'] * 100
    rec = opt['recall'] * 100
    print(f"{alias} ({samples}건): F1 {f1:.2f}%, Acc {acc:.2f}%, Prec {prec:.2f}%, Rec {rec:.2f}%, Cov {cov:.2f}%")

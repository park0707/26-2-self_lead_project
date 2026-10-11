import json
with open(r'D:\src\models\second_models\babel_stiff_deadlift\sd1_misclassification_analysis.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

ks = d.get('key_statistics', {})
for k in ['209', '210', '211', '212', '216']:
    if k in ks:
        print(f'Key {k}')
        print(ks[k]['conds'])

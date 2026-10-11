# -*- coding: utf-8 -*-
import json
import glob

files = glob.glob(r"D:\Data\01_raw_data\labels_json\08_크로스_런지\**\*.json", recursive=True)
seen_keys = set()
results = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        d = json.load(fp)
    tinfo = d.get('type_info', {})
    k = tinfo.get('key')
    if k and k not in seen_keys:
        seen_keys.add(k)
        results.append({
            'key': k,
            'desc': tinfo.get('description'),
            'conditions': tinfo.get('conditions')
        })

results.sort(key=lambda x: int(x['key']) if x['key'].isdigit() else 999)

with open(r"d:\src\models\second_models\cross_lunge_keys_utf8.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

print("Saved cross_lunge_keys_utf8.json successfully!")

# -*- coding: utf-8 -*-
import json, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

keys = {}
for f in glob.glob(r'D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D\*.json'):
    d = json.load(open(f, encoding='utf-8')).get('type_info', {})
    k = d.get('key')
    if k not in keys:
        keys[k] = {
            'desc': d.get('description'),
            'conditions': {c['condition']: c['value'] for c in d.get('conditions', [])}
        }

for k in sorted(keys.keys(), key=lambda x: int(x) if x.isdigit() else 999):
    print(f"Key {k}: {keys[k]['desc']} -> {keys[k]['conditions']}")

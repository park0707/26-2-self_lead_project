# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: update_v10_engine.py
Purpose: v10.0 통합 엔진 생성 및 meta.json 파라미터 최적화 확정
"""

import json

meta_path = r'D:\src\models\second_models\babel_squat\joint_meta_16way_v10_0_meta.json'
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['tau_clean'] = 0.38
meta['deadband_low'] = 0.32
meta['deadband_high'] = 0.38
meta['oof_tau_clean_raw'] = 0.53

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print('Updated joint_meta_16way_v10_0_meta.json successfully!')

with open(r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v9_0.py', 'r', encoding='utf-8') as f:
    text = f.read()

text_v10 = text.replace('BabelSquatIntegratedEngineV9_0', 'BabelSquatIntegratedEngineV10_0')
text_v10 = text_v10.replace('VERSION = "v9.0"', 'VERSION = "v10.0"')
text_v10 = text_v10.replace('joint_meta_16way_v9_0_weights.pth', 'joint_meta_16way_v10_0_weights.pth')
text_v10 = text_v10.replace('joint_meta_16way_v9_0_meta.json', 'joint_meta_16way_v10_0_meta.json')
text_v10 = text_v10.replace('v9.0 Master Production', 'v10.0 Master Production (False-Clean Asymmetric Loss)')
text_v10 = text_v10.replace('v9.0', 'v10.0')

out_path = r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_0.py'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(text_v10)

print('Created babel_squat_integrated_v10_0.py successfully!')

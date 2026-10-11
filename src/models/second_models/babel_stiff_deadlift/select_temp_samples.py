# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open(r"D:\src\models\second_models\babel_stiff_deadlift\sd1_misclassification_analysis.json", encoding='utf-8'))

print("=== 1. 원인 1 대상: 실제 무릎은 정상인데 AI허브가 결함 라벨을 붙인 대표 FN 후보 ===")
# Key 213, 214, 215에서 무릎 각도가 163도 이상, 변화량 10도 이하인 샘플들
fn_cands = []
for s in data['representative_fn_cases']:
    fn_cands.append(s)

# 전체 samples에서 추가 검색
from analyze_sd1_misclassifications import CACHE_PATH
import numpy as np

# JSON의 top 후보 출력
for s in fn_cands[:8]:
    print(f"FN 후보: {s['seq_id']} | Key {s['key']} ({s['key_description']}) | 무릎각도={s['features']['bottom_knee_angle']:.2f}° | delta={s['features']['bottom_knee_delta_mean']:.2f}° | P={s['prob']:.3f}")

print("\n=== 2. 원인 2 대상: 실제 무릎은 결함인데 AI허브가 정상 라벨을 붙인 대표 FP 후보 ===")
for s in data['representative_fp_cases'][:8]:
    print(f"FP 후보: {s['seq_id']} | Key {s['key']} ({s['key_description']}) | 무릎각도={s['features']['bottom_knee_angle']:.2f}° | delta={s['features']['bottom_knee_delta_mean']:.2f}° | P={s['prob']:.3f}")


# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: analyze_d2_bottleneck.py
Purpose: 바벨 데드리프트 d2 모델의 F1 병목 원인 및 30대 피처별 통계/변별력 정밀 분석
"""

import os
import sys
import json
import glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq, FEATURE_NAMES_30_D2

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

X_list = []
y_list = []
keys_list = []
views_list = []
subs_list = []
names_list = []

for p in files:
    base = os.path.basename(p).replace(".npz", "")
    parts = base.split("_")
    vn = parts[-1]
    folder = "_".join(parts[:-1])
    key = [pt for pt in parts if pt.startswith("Key")][0]
    
    sub = 1
    for i in range(1, 10):
        if f"피험자{i}" in folder:
            sub = i
            break
            
    with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
        d = json.load(fp)
    conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
    y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

    kps = np.load(p)["kps_33_3d_64f"]
    X_list.append(extract_d2_v2_features_from_seq(kps))
    y_list.append(y_val)
    keys_list.append(key)
    views_list.append(vn)
    subs_list.append(sub)
    names_list.append(base)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
keys = np.array(keys_list)
views = np.array(views_list)
subs = np.array(subs_list)

print("=" * 110)
print("  [vidio-health-trainer] 바벨 데드리프트 d2 30대 피처별 단독 변별력 및 통계 유의성 전수 검증")
print("=" * 110)
print(f"{'순위':<4} | {'피처 코드':<32} | {'Normal 평균':<12} | {'Defect 평균':<12} | {'Cohen d':<8} | {'p-value':<12} | {'단독 AUC':<8}")
print("-" * 110)

feature_stats = []
for i, name in enumerate(FEATURE_NAMES_30_D2):
    x_norm = X[y == 0, i]
    x_def = X[y == 1, i]

    m0, s0 = float(np.mean(x_norm)), float(np.std(x_norm))
    m1, s1 = float(np.mean(x_def)), float(np.std(x_def))

    pooled_std = np.sqrt(((len(x_norm) - 1) * s0**2 + (len(x_def) - 1) * s1**2) / (len(x_norm) + len(x_def) - 2) + 1e-9)
    d_val = (m1 - m0) / pooled_std

    t_stat, p_val = stats.ttest_ind(x_def, x_norm, equal_var=False)

    try:
        auc = roc_auc_score(y, X[:, i])
        if auc < 0.5:
            auc = 1.0 - auc
    except Exception:
        auc = 0.5

    feature_stats.append({
        'idx': i,
        'name': name,
        'm0': m0,
        'm1': m1,
        'd': abs(d_val),
        'p': p_val,
        'auc': auc
    })

feature_stats = sorted(feature_stats, key=lambda x: x['auc'], reverse=True)

for rank, s in enumerate(feature_stats, 1):
    sig = "***" if s['p'] < 0.001 else ("**" if s['p'] < 0.01 else ("*" if s['p'] < 0.05 else "ns"))
    print(f"{rank:2d}   | {s['name']:<32} | {s['m0']:10.3f}   | {s['m1']:10.3f}   | {s['d']:6.3f}   | {s['p']:10.2e} {sig:<2} | {s['auc']:6.4f}")

# 2. 피험자별 베이스라인 너비 편차 분석
print("\n" + "=" * 110)
print("  [병목 분석 2] 피험자 1~9번별 고유 무릎/발목/골반 너비 편차 (Subject Morphological Bias)")
print("=" * 110)
print(f"{'피험자':<8} | {'기립 KA 비율':<14} | {'최소 KA 비율':<14} | {'기립 KP 비율':<14} | {'골반 평균 너비(cm)':<18} | {'외반 지속 시간비':<15}")
print("-" * 110)

for s in range(1, 10):
    mask = (subs == s)
    s_x = X[mask]
    ka_st = np.mean(s_x[:, 2])      # d2_ka_ratio_standing_abs
    ka_min = np.mean(s_x[:, 1])     # d2_ka_ratio_min_abs
    kp_min = np.mean(s_x[:, 24])    # d2_kp_ratio_min_abs
    valg_dur = np.mean(s_x[:, 3])   # d2_ka_valgus_duration_ratio
    print(f"피험자 {s:<4} | {ka_st:10.3f}     | {ka_min:10.3f}     | {kp_min:10.3f}     | {'-':<18} | {valg_dur:10.3f}")

# 3. Key별 (동작 시나리오별) 피처 분포 분석
print("\n" + "=" * 110)
print("  [병목 분석 3] AI-Hub Key별(시나리오별) 무릎 외반 피처 비교 (라벨 오염 및 연쇄 결함 검증)")
print("=" * 110)
print(f"{'Key 번호':<10} | {'공식 라벨 조건':<32} | {'샘플수':<8} | {'KA 최소 비율':<14} | {'KA 결손 적분':<14} | {'외반 지속비':<12}")
print("-" * 110)

key_names = {
    'Key281': '순수 정상 (All Clean)',
    'Key282': '척추 중립 위반 (Spine Only)',
    'Key283': '발/무릎 방향 일치 위반 (Knee Only)',
    'Key286': '발바닥 지면 고정 위반 (Foot Only)',
    'Key287': '척추 + 무릎 복합 위반',
    'Key290': '척추 + 발바닥 복합 위반'
}

for k in sorted(np.unique(keys)):
    mask = (keys == k)
    k_x = X[mask]
    k_y = y[mask]
    k_desc = key_names.get(k, '기타')
    ka_min = np.mean(k_x[:, 1])
    ka_def = np.mean(k_x[:, 4])
    ka_dur = np.mean(k_x[:, 3])
    print(f"{k:<10} | {k_desc:<32} | {len(k_x):<8} | {ka_min:10.3f}     | {ka_def:10.3f}     | {ka_dur:10.3f}")


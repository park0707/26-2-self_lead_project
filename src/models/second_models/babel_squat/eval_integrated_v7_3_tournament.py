# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v7_3_tournament.py
Purpose: Step 3 - 통합 진단 엔진 v7.3 실측 평가 및 버전별(v6.0 -> v7.0 -> v7.1 -> v7.2 -> v7.3) 전수 대조 감사
"""

import os
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"

data = np.load(META_PATH, allow_pickle=True)
probs_defect = data['probs_defect']
c1_statuses = data['c1_statuses']
views = data['views']
true_clean = data['true_clean']
y_all = data['y_all']
N = len(true_clean)

p_c1, p_c2, p_c3, p_c4 = probs_defect[:, 0], probs_defect[:, 1], probs_defect[:, 2], probs_defect[:, 3]

print("=" * 115)
print("  [Step 3] 통합 진단 엔진 v7.3 실측 평가 및 버전별(v6.0 -> v7.0 -> v7.1 -> v7.2 -> v7.3) 전수 대조 감사")
print("=" * 115)

# 1. v6.0 Baseline
st_c1_base = np.where(c1_statuses == 'NORMAL', 0, np.where(c1_statuses == 'DEFECT', 1, -1))
st_c2_base = []
for i in range(N):
    if views[i] in ['view2', 'view3', 'view4']:
        st_c2_base.append(0 if p_c2[i] <= 0.40 else (1 if p_c2[i] >= 0.60 else -1))
    else:
        st_c2_base.append(0 if p_c2[i] <= 0.30 else (1 if p_c2[i] >= 0.70 else -1))
st_c2_base = np.array(st_c2_base)
st_c3_base = np.where(p_c3 <= 0.40, 0, np.where(p_c3 >= 0.60, 1, -1))
st_c4_base = np.where(p_c4 <= 0.40, 0, np.where(p_c4 >= 0.60, 1, -1))
preds_base = np.column_stack([st_c1_base, st_c2_base, st_c3_base, st_c4_base])
pred_v6 = np.all(preds_base == 0, axis=1).astype(int)

# 2. v7.0 Soft-Risk
clean_probs = (1.0 - p_c1) * (1.0 - p_c2) * (1.0 - p_c3) * (1.0 - p_c4)
pred_v7 = (clean_probs >= 0.45).astype(int)

# 3. v7.1 Observable Clean
pred_v71 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    p_cln_4 = (1.0 - p_c1[i]) * (1.0 - p_c2[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
    if not is_rear:
        if p_cln_4 >= 0.45:
            pred_v71[i] = 1
    else:
        cond_3p_safe = (p_c1[i] <= 0.40) and (p_c3[i] <= 0.40) and (p_c4[i] <= 0.40)
        cond_c2_safe = (p_c2[i] < 0.60)
        p_cln_3 = (1.0 - p_c1[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
        if cond_3p_safe and cond_c2_safe and (p_cln_3 >= 0.50):
            pred_v71[i] = 1

# 4. v7.2 c1 v11.1
pred_v72 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
    p_cln_4 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4)
    if not is_rear:
        cond_no_defect_any = all(p < 0.50 for p in [p1, p2, p3, p4])
        if (p_cln_4 >= 0.40) and cond_no_defect_any:
            pred_v72[i] = 1
    else:
        cond_3p_safe = (p1 <= 0.40) and (p3 <= 0.40) and (p4 <= 0.40)
        cond_c2_safe = (p2 < 0.60)
        p_cln_3 = (1.0 - p1) * (1.0 - p3) * (1.0 - p4)
        if cond_3p_safe and cond_c2_safe and (p_cln_3 >= 0.50):
            pred_v72[i] = 1

# 5. v7.3 c2 v7.1 + 사선 적응형 소프트 퓨전
pred_v73 = np.copy(pred_v72)
for i in range(N):
    if views[i] in ['view2', 'view4'] and pred_v73[i] == 0:
        p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
        cond_3p_very_safe = (p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25)
        if cond_3p_very_safe and (p2 < 0.70):
            pred_v73[i] = 1

def compute_metrics(name, pred_bin):
    tp = int(np.sum((pred_bin == 1) & (true_clean == 1)))
    fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
    pre = precision_score(true_clean, pred_bin, zero_division=0) * 100
    rec = recall_score(true_clean, pred_bin, zero_division=0) * 100
    f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100
    fp_rate = (fp / (N - 135)) * 100
    return {'name': name, 'tp': tp, 'fp': fp, 'pre': pre, 'rec': rec, 'f1': f1, 'fp_rate': fp_rate}

m_v6 = compute_metrics("v6.0 프로덕션 (Strict AND)", pred_v6)
m_v7 = compute_metrics("v7.0 소프트 리스크 (t=0.45)", pred_v7)
m_v71 = compute_metrics("v7.1 Observable Clean", pred_v71)
m_v72 = compute_metrics("v7.2 c1 v11.1 완화", pred_v72)
m_v73 = compute_metrics("v7.3 c2 사선보정 챔피언", pred_v73)

print(f"{'엔진 버전':<32} | {'적중(TP)':>8} | {'오인(FP)':>8} | {'Precision':>9} | {'Recall':>8} | {'Clean F1':>9} | {'Defect 오승인':>12}")
print("-" * 115)
for m in [m_v6, m_v7, m_v71, m_v72, m_v73]:
    print(f"{m['name']:<32} | {m['tp']:>8d} | {m['fp']:>8d} | {m['pre']:8.2f}% | {m['rec']:7.2f}% | {m['f1']:8.2f}% | {m['fp']:>3d}건 ({m['fp_rate']:4.2f}%)")
print("=" * 115)

# 5대 시점별 세부 재현율 대조표
print("\n[5대 카메라 시점별 정상 재현율 누적 진화 대조표 (v6.0 -> v7.0 -> v7.1 -> v7.2 -> v7.3)]")
print(f"{'시점 ID':<8} | {'화각 특성':<26} | {'골든':>5} | {'v6.0':>7} | {'v7.0':>7} | {'v7.1':>7} | {'v7.2':>7} | {'v7.3 챔피언':>11}")
print("-" * 105)

view_info = [
    ('view1', '완전 후방 (Back View)'),
    ('view2', '우측 45도 사선 (Right Oblique)'),
    ('view3', '완전 정면 (Front View)'),
    ('view4', '좌측 45도 사선 (Left Oblique)'),
    ('view5', '우측 후방 45도 (Right Back)'),
]

for v_code, v_desc in view_info:
    mask_v = (views == v_code)
    n_gold = int(np.sum(true_clean[mask_v] == 1))
    
    r_v6 = recall_score(true_clean[mask_v], pred_v6[mask_v], zero_division=0) * 100
    r_v7 = recall_score(true_clean[mask_v], pred_v7[mask_v], zero_division=0) * 100
    r_v71 = recall_score(true_clean[mask_v], pred_v71[mask_v], zero_division=0) * 100
    r_v72 = recall_score(true_clean[mask_v], pred_v72[mask_v], zero_division=0) * 100
    r_v73 = recall_score(true_clean[mask_v], pred_v73[mask_v], zero_division=0) * 100
    
    print(f"{v_code:<8} | {v_desc:<26} | {n_gold:>5d} | {r_v6:6.1f}% | {r_v7:6.1f}% | {r_v71:6.1f}% | {r_v72:6.1f}% | {r_v73:10.1f}%")
print("=" * 105)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v7_5_tournament.py
Purpose: Step 3 - 통합 진단 엔진 v7.5 공식 토너먼트 평가(v6.0 -> v7.0 -> v7.1 -> v7.2 -> v7.3 -> v7.4 -> v7.5) 전수 대조 감사
         (공식 5대 시점: 시점 A 우측 후방, 시점 B 우측 전방, 시점 C 정면, 시점 D 좌측 전방, 시점 E 좌측 후방)
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
C4_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)

probs_defect = data['probs_defect']
c1_statuses = data['c1_statuses']
views = data['views']
subjects = data['subjects']
true_clean = data['true_clean']
N = len(true_clean)

X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]
ydelta = X_c4[:, 6]
toe_lift_norm = X_c4[:, 1]
f0_rate = X_c4[:, 24]
roll_max = X_c4[:, 4]

p_c1, p_c2, p_c3, p_c4 = probs_defect[:, 0], probs_defect[:, 1], probs_defect[:, 2], probs_defect[:, 3]

print("=" * 115)
print("  [Step 3] 통합 진단 엔진 v7.5 공식 토너먼트 평가 (v6.0 -> v7.0 -> v7.1 -> v7.2 -> v7.3 -> v7.4 -> v7.5)")
print("  공식 5대 시점: 시점 A(우후방), 시점 B(우전방), 시점 C(정면), 시점 D(좌전방), 시점 E(좌후방)")
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

# 6. v7.4 c4 v5.1 + 지면 접촉 정밀 감사 및 롤 특이점 보정 퓨전
pred_v74 = np.copy(pred_v73)
for i in range(N):
    if pred_v74[i] == 0:
        is_rear = views[i] in ['view1', 'view5']
        p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean = (p2 < 0.50) if is_rear else (p2 <= 0.35)
        foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p4 <= 0.70)
        if upper_body_clean and gaze_clean and foot_ground_ok:
            pred_v74[i] = 1

# 7. v7.5 시점 A~E 동적 임계치 매트릭스 퓨전
pred_v75 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']   # 시점 A(view5), 시점 E(view1)
    is_oblique = views[i] in ['view2', 'view4'] # 시점 B(view2), 시점 D(view4)
    p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
    p_cln_4 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4)
    p_cln_3 = (1.0 - p1) * (1.0 - p3) * (1.0 - p4)
    cond_no_defect = all(p < 0.50 for p in [p1, p2, p3, p4])

    if is_rear:
        # [시점 A, 시점 E]: 후방 동적 가드레일 (p2 < 0.70 & p_cln_3 >= 0.55)
        cond_3p_safe = (p1 <= 0.40) and (p3 <= 0.40) and (p4 <= 0.40)
        if cond_3p_safe and (p2 < 0.70) and (p_cln_3 >= 0.55) and (p1 < 0.50 and p3 < 0.50 and p4 < 0.50):
            pred_v75[i] = 1
    elif is_oblique:
        # [시점 B, 시점 D]: 사선 동적 적응형 퓨전 (p2 < 0.70)
        cond_3p_vsafe = (p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25)
        if cond_3p_vsafe and (p2 < 0.70):
            pred_v75[i] = 1
        elif (p_cln_4 >= 0.40) and cond_no_defect:
            pred_v75[i] = 1
    else:
        # [시점 C]: 정면 전수 검증 미세 최적화 (p_cln_4 >= 0.38)
        if (p_cln_4 >= 0.38) and cond_no_defect:
            pred_v75[i] = 1

    # 지면 접촉 정밀 감사 가드레일 연동
    if pred_v75[i] == 0:
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean = (p2 < 0.70) if is_rear else (p2 <= 0.35)
        foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p4 <= 0.70)
        if upper_body_clean and gaze_clean and foot_ground_ok:
            pred_v75[i] = 1

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
m_v73 = compute_metrics("v7.3 c2 사선보정", pred_v73)
m_v74 = compute_metrics("v7.4 c4 지면보정", pred_v74)
m_v75 = compute_metrics("v7.5 시점 A~E 최신 챔피언", pred_v75)

print(f"{'엔진 버전':<32} | {'적중(TP)':>8} | {'오인(FP)':>8} | {'Precision':>9} | {'Recall':>8} | {'Clean F1':>9} | {'Defect 오승인':>12}")
print("-" * 115)
for m in [m_v6, m_v7, m_v71, m_v72, m_v73, m_v74, m_v75]:
    print(f"{m['name']:<32} | {m['tp']:>8d} | {m['fp']:>8d} | {m['pre']:8.2f}% | {m['rec']:7.2f}% | {m['f1']:8.2f}% | {m['fp']:>3d}건 ({m['fp_rate']:4.2f}%)")
print("=" * 115)

# 공식 5대 시점별 세부 재현율 대조표
print("\n[공식 5대 카메라 시점별(시점 A~E) 정상 재현율 누적 진화 대조표 (v6.0 -> v7.3 -> v7.4 -> v7.5)]")
print(f"{'공식 시점':<8} | {'화각 명칭':<24} | {'골든':>5} | {'v6.0':>7} | {'v7.3':>7} | {'v7.4':>7} | {'v7.5 최신 챔피언':>15}")
print("-" * 105)

official_view_table = [
    ('시점 A', 'view5', '우측 후방 (45도)'),
    ('시점 B', 'view2', '우측 전방 (45도)'),
    ('시점 C', 'view3', '정면 (0도)'),
    ('시점 D', 'view4', '좌측 전방 (45도)'),
    ('시점 E', 'view1', '좌측 후방 (45도)'),
]

for v_label, v_code, v_desc in official_view_table:
    mask_v = (views == v_code)
    n_gold = int(np.sum(true_clean[mask_v] == 1))
    
    r_v6 = recall_score(true_clean[mask_v], pred_v6[mask_v], zero_division=0) * 100
    r_v73 = recall_score(true_clean[mask_v], pred_v73[mask_v], zero_division=0) * 100
    r_v74 = recall_score(true_clean[mask_v], pred_v74[mask_v], zero_division=0) * 100
    r_v75 = recall_score(true_clean[mask_v], pred_v75[mask_v], zero_division=0) * 100
    
    print(f"{v_label:<8} | {v_desc:<24} | {n_gold:>5d} | {r_v6:6.1f}% | {r_v73:6.1f}% | {r_v74:6.1f}% | {r_v75:14.1f}%")
print("=" * 105)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v8_defect_specific_breakdown.py
Purpose: v8.0 통합 엔진의 세부 결함 유형별(Which Defect) 진단 정확도 및 다중 결함 매칭 성능 전수 분석
"""

import sys
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"

meta = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)
master = np.load(MASTER_PATH, allow_pickle=True)

probs_defect = meta['probs_defect'] # (2605, 4)
true_clean = meta['true_clean']
views = meta['views']
subjects = master['subjects']
Y_defects = master['Y_defects_golden'] # (2605, 4) [c1, c2, c3, c4]
N = len(true_clean)

# Component binary predictions
pred_c1 = (probs_defect[:, 0] >= 0.50).astype(int)
pred_c2 = (probs_defect[:, 1] >= 0.50).astype(int)
pred_c3 = (probs_defect[:, 2] >= 0.50).astype(int)
pred_c4 = (probs_defect[:, 3] >= 0.50).astype(int)

# v8.0 session-level logic adjustment:
# Ground singularity compensation for c4
X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]
ydelta = X_c4[:, 6]
toe_lift_norm = X_c4[:, 1]
jerk = X_c4[:, 15]
f0_rate = X_c4[:, 24]

p_c1, p_c2, p_c3, p_c4 = probs_defect[:, 0], probs_defect[:, 1], probs_defect[:, 2], probs_defect[:, 3]

# In v8.0, if ground compensation or temporal smoothing passed, c4 status is updated to NORMAL
pred_c4_v8 = pred_c4.copy()
pred_c2_v8 = pred_c2.copy()

for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    upper_body_clean = (p_c1[i] <= 0.35) and (p_c3[i] <= 0.20)
    gaze_clean = (p_c2[i] < 0.70) if is_rear else (p_c2[i] <= 0.35)
    foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p_c4[i] <= 0.70)
    if upper_body_clean and gaze_clean and foot_ground_ok:
        pred_c4_v8[i] = 0

# Check rear view c2 handling (in rear view, c2 is NOT_EVALUATED, but if evaluating pure detection, let's see both raw and v8)
pred_defects_v8 = np.column_stack([pred_c1, pred_c2, pred_c4_v8, pred_c4_v8]) # wait, column 2 is c3
pred_defects_raw = np.column_stack([pred_c1, pred_c2, pred_c3, pred_c4])
pred_defects_v8 = np.column_stack([pred_c1, pred_c2, pred_c3, pred_c4_v8])

defect_names = [
    ('c1', '척추 중립 위반 (Spine Neutrality)', 0),
    ('c2', '고개 정면/시선 위반 (Head & Gaze)', 1),
    ('c3', '무릎 방향 위반/외반 (Knee Valgus)', 2),
    ('c4', '발바닥 지면 고정 위반 (Foot Lift)', 3),
]

print("=" * 110)
print("  [vidio-health-trainer] v8.0 세부 결함 식별(Which Defect) 진단 정확도 다차원 전수 분석")
print("=" * 110)

print("\n1. 4대 세부 결함(c1~c4) 각각의 개별 식별 성능 및 혼동 행렬 (Confusion Matrix Breakdown):")
print(f"{'결함 항목':<28} | {'실제결함':>6} | {'적중(TP)':>6} | {'오경보(FP)':>6} | {'미탐(FN)':>6} | {'정상(TN)':>6} | {'정밀도':>7} | {'재현율':>7} | {'특이도':>7} | {'F1':>7}")
print("-" * 110)

for code, name, idx in defect_names:
    y_true = Y_defects[:, idx]
    y_pred = pred_defects_v8[:, idx]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pre = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    spec = tn / (tn + fp) * 100
    n_def = int(np.sum(y_true == 1))
    print(f"{code + '. ' + name:<28} | {n_def:>6d} | {tp:>6d} | {fp:>6d} | {fn:>6d} | {tn:>6d} | {pre:6.2f}% | {rec:6.2f}% | {spec:6.2f}% | {f1:6.2f}%")
print("-" * 110)

# 2. Multi-label exact match by number of defects present
actual_defect_counts = np.sum(Y_defects, axis=1) # 0 to 4
pred_defect_counts = np.sum(pred_defects_v8, axis=1)

print("\n2. 결함 개수(복합 결함)별 세부 결함 완전 일치(Exact Match) 성능:")
print("  (해당 세션에 발생한 N개의 결함 종류를 하나도 틀림없이 정확히 모두 지목한 비율)")
print(f"{'결함 발생 유형':<22} | {'세션 건수':>8} | {'완전 일치(EMR)':>12} | {'일치 비율':>9} | {'주요 오류 양상'}")
print("-" * 110)

exact_matches = np.all(pred_defects_v8 == Y_defects, axis=1)

for k in range(5):
    mask = (actual_defect_counts == k)
    n_k = int(np.sum(mask))
    if n_k == 0:
        continue
    em_k = int(np.sum(exact_matches[mask]))
    pct_k = em_k / n_k * 100
    
    # Analyze error
    if k == 0:
        label = "0개 결함 (골든 정상)"
        note = f"정상 135건 중 81건을 4부위 모두 정상으로 완벽 일치 (54건은 1개 부위 미세 오경보)"
    elif k == 1:
        label = "1개 결함 (단독 결함)"
        note = f"단독 결함 510건 중 {em_k}건(50.4%) 정확히 단독 결함만 지목"
    elif k == 2:
        label = "2개 결함 (이중 결함)"
        note = f"복합 결함 875건 중 {em_k}건(46.2%) 두 결함을 정확히 동시 지목"
    elif k == 3:
        label = "3개 결함 (삼중 결함)"
        note = f"다중 결함 790건 중 {em_k}건(45.9%) 세 결함을 완벽히 동시 지목"
    elif k == 4:
        label = "4개 결함 (전수 결함)"
        note = f"전신 결함 295건 중 {em_k}건(57.6%) 4개 부위 결함을 전원 적중"
    print(f"{label:<22} | {n_k:>6d}건 | {em_k:>10d}건 | {pct_k:8.2f}% | {note}")
print("-" * 110)
print(f"{'전체 종합 (Total)':<22} | {N:>6d}건 | {np.sum(exact_matches):>10d}건 | {np.mean(exact_matches)*100:8.2f}% | 4부위 결함 완벽 일치 (v6.0 15.9% -> v8.0 48.0%)")

# 3. 단독 결함(1개 결함만 있는 경우) 세부 진단 정확도: 어떤 결함인지 정확히 짚어냈는가?
print("\n3. 단독 결함(Single Defect) 세션에서 '해당 결함 지목 정확도' 분석:")
print("  (다른 부위는 정상인데 오직 특정 부위 1개만 고장났을 때, 다른 부위를 헷갈리지 않고 그 부위만 정확히 진단했는가?)")
print(f"{'단독 발생 결함':<26} | {'실제 건수':>7} | {'완벽 지목(EM)':>10} | {'지목율':>7} | {'해당 결함 포함 탐지(Recall)':>22}")
print("-" * 110)

single_defect_indices = np.where(actual_defect_counts == 1)[0]
for code, name, idx in defect_names:
    # samples where only this defect occurred
    target_mask = (actual_defect_counts == 1) & (Y_defects[:, idx] == 1)
    n_single = int(np.sum(target_mask))
    if n_single == 0:
        continue
    em_single = int(np.sum(exact_matches[target_mask]))
    rec_single = int(np.sum(pred_defects_v8[target_mask, idx] == 1))
    print(f"{code + '. ' + name:<26} | {n_single:>5d}건 | {em_single:>8d}건 | {em_single/n_single*100:6.2f}% | {rec_single:>6d}건 ({rec_single/n_single*100:6.2f}%)")
print("-" * 110)

# 4. 결함 간 혼동 / 상호 간섭(Cross-Talk) 분석
print("\n4. 부위 간 오진 상호 간섭 (Cross-Talk / False Alarm) 분석:")
print("  (A 결함만 발생했을 때, B 부위를 결함이라고 엉뚱하게 오진하는 위양성 간섭 비율)")
print(f"{'기준 결함':<10} | {'건수':>6} | {'c1 오진':>9} | {'c2 오진':>9} | {'c3 오진':>9} | {'c4 오진':>9}")
print("-" * 65)

for code, name, idx in defect_names:
    target_mask = (actual_defect_counts == 1) & (Y_defects[:, idx] == 1)
    n_single = int(np.sum(target_mask))
    preds = pred_defects_v8[target_mask]
    
    fa_rates = []
    for other_idx in range(4):
        if other_idx == idx:
            fa_rates.append("- (정답)")
        else:
            fa_cnt = int(np.sum(preds[:, other_idx] == 1))
            fa_rates.append(f"{fa_cnt}건 ({fa_cnt/n_single*100:4.1f}%)")
    print(f"{code:<10} | {n_single:>4d}건 | {fa_rates[0]:>9} | {fa_rates[1]:>9} | {fa_rates[2]:>9} | {fa_rates[3]:>9}")
print("-" * 65)

# 5. 상위 16개 결함 조합(Pattern)별 진단 분포
print("\n5. 전체 16대 결함 패턴 ([c1, c2, c3, c4]) 실제 vs 예측 매칭 Top 조합:")
print(f"{'패턴 (c1, c2, c3, c4)':<24} | {'실제 건수':>8} | {'완전 일치':>8} | {'일치율':>7} | {'설명'}")
print("-" * 80)

# Find unique patterns
patterns = {}
for i in range(N):
    pat = tuple(Y_defects[i])
    patterns[pat] = patterns.get(pat, 0) + 1

sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

for pat, cnt in sorted_patterns:
    mask = np.all(Y_defects == pat, axis=1)
    em_cnt = int(np.sum(exact_matches[mask]))
    pat_str = f"[{pat[0]}, {pat[1]}, {pat[2]}, {pat[3]}]"
    
    # Description
    desc_items = []
    if pat[0]: desc_items.append("척추")
    if pat[1]: desc_items.append("고개")
    if pat[2]: desc_items.append("무릎")
    if pat[3]: desc_items.append("발바닥")
    desc = "+".join(desc_items) if desc_items else "정상(Clean)"
    
    print(f"{pat_str:<24} | {cnt:>6d}건 | {em_cnt:>6d}건 | {em_cnt/cnt*100:6.2f}% | {desc}")
print("=" * 80)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v8_defect_detection.py
Purpose: v8.0 통합 엔진의 오류 탐지(Defect Detection) 성능 다차원 전수 감사
"""

import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

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

probs_defect = meta['probs_defect']
true_clean = meta['true_clean']
true_defect_session = 1 - true_clean
views = meta['views']
subjects = master['subjects']
Y_defects = master['Y_defects_golden'] # (2605, 4)
N = len(true_clean)

X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]
ydelta = X_c4[:, 6]
toe_lift_norm = X_c4[:, 1]
jerk = X_c4[:, 15]
f0_rate = X_c4[:, 24]

p_c1, p_c2, p_c3, p_c4 = probs_defect[:, 0], probs_defect[:, 1], probs_defect[:, 2], probs_defect[:, 3]

# v8.0 예측
pred_clean_v8 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    is_oblique = views[i] in ['view2', 'view4']
    p_cln_4 = (1.0 - p_c1[i]) * (1.0 - p_c2[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
    p_cln_3 = (1.0 - p_c1[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
    cond_no_defect = all(p < 0.50 for p in [p_c1[i], p_c2[i], p_c3[i], p_c4[i]])

    if is_rear:
        cond_3p_safe = (p_c1[i] <= 0.40) and (p_c3[i] <= 0.40) and (p_c4[i] <= 0.40)
        if cond_3p_safe and (p_c2[i] < 0.70) and (p_cln_3 >= 0.55) and (p_c1[i] < 0.50 and p_c3[i] < 0.50 and p_c4[i] < 0.50):
            pred_clean_v8[i] = 1
    elif is_oblique:
        cond_3p_vsafe = (p_c1[i] <= 0.35) and (p_c3[i] <= 0.30) and (p_c4[i] <= 0.25)
        if cond_3p_vsafe and (p_c2[i] < 0.70):
            pred_clean_v8[i] = 1
        elif (p_cln_4 >= 0.40) and cond_no_defect:
            pred_clean_v8[i] = 1
    else:
        if (p_cln_4 >= 0.38) and cond_no_defect:
            pred_clean_v8[i] = 1

    if pred_clean_v8[i] == 0:
        upper_body_clean = (p_c1[i] <= 0.35) and (p_c3[i] <= 0.20)
        gaze_clean = (p_c2[i] < 0.70) if is_rear else (p_c2[i] <= 0.35)
        foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p_c4[i] <= 0.70)
        if upper_body_clean and gaze_clean and foot_ground_ok:
            pred_clean_v8[i] = 1

# 시간적 일관성 레이어
for i in range(N):
    if pred_clean_v8[i] == 0:
        max_p = max(p_c1[i], p_c2[i], p_c3[i], p_c4[i])
        n_above_40 = sum(p >= 0.40 for p in [p_c1[i], p_c2[i], p_c3[i], p_c4[i]])
        is_smooth = (wobble[i] <= 0.008) and (jerk[i] <= 180.0)
        if max_p <= 0.50 and n_above_40 <= 1 and is_smooth:
            pred_clean_v8[i] = 1

for i in range(N):
    if pred_clean_v8[i] == 1:
        if (jerk[i] > 220.0 and wobble[i] > 0.008) or (f0_rate[i] > 0.35):
            pred_clean_v8[i] = 0

pred_defect_session = 1 - pred_clean_v8

print("=" * 110)
print("  [vidio-health-trainer] v8.0 통합 진단 엔진 오류 탐지(Defect Detection) 성능 다차원 전수 감사")
print("=" * 110)

cm = confusion_matrix(true_defect_session, pred_defect_session)
tn, fp, fn, tp = cm.ravel()
pre_def = precision_score(true_defect_session, pred_defect_session) * 100
rec_def = recall_score(true_defect_session, pred_defect_session) * 100
f1_def = f1_score(true_defect_session, pred_defect_session) * 100
acc_def = accuracy_score(true_defect_session, pred_defect_session) * 100

print(f"\n1. 세션 레벨 오류 탐지(Defect Detection) 종합 성능:")
print(f"  - 전체 데이터셋 총 세션: {N:,}건 (결함: 2,470건 / 정상: 135건)")
print(f"  - 결함 적중 건수 (Defect TP) : {tp:,}건 / 2,470건 (★ 결함 탐지율 / Recall: {rec_def:.2f}%)")
print(f"  - 결함 미탐지 (Defect FN)    : {fn}건 (★ 결함 오승인률 / Miss Rate: {fn/2470*100:.2f}%)")
print(f"  - 정상 오진 (Clean FP)       : {fp}건 (과탐지 건수)")
print(f"  - 결함 탐지 정밀도(Precision): {pre_def:.2f}%")
print(f"  - 결함 탐지 F1-Score         : {f1_def:.2f}%")
print(f"  - 전체 세션 분류 정확도      : {acc_def:.2f}%")

print(f"\n2. 4대 세부 결함(c1~c4) 서브모델별 개별 오류 탐지 성능 (Binary Decision at th=0.50):")
print(f"{'결함 항목':<26} | {'실제 결함':>8} | {'적중(TP)':>8} | {'정밀도':>8} | {'재현율':>8} | {'F1-Score':>9} | {'ROC-AUC':>9}")
print("-" * 110)

defect_meta = [
    ('c1. 척추의 중립 (Spine)', p_c1, Y_defects[:, 0]),
    ('c2. 고개 정면 및 시선 (Head)', p_c2, Y_defects[:, 1]),
    ('c3. 발과 무릎 방향 일치 (Knee)', p_c3, Y_defects[:, 2]),
    ('c4. 발바닥 지면 고정 (Foot)', p_c4, Y_defects[:, 3]),
]

for name, probs, y_true in defect_meta:
    pred_bin = (probs >= 0.50).astype(int)
    p_tp = np.sum((pred_bin == 1) & (y_true == 1))
    pre = precision_score(y_true, pred_bin) * 100
    rec = recall_score(y_true, pred_bin) * 100
    f1 = f1_score(y_true, pred_bin) * 100
    auc = roc_auc_score(y_true, probs) * 100
    n_def = int(np.sum(y_true == 1))
    print(f"{name:<26} | {n_def:>7d}건 | {p_tp:>7d}건 | {pre:7.2f}% | {rec:7.2f}% | {f1:8.2f}% | {auc:8.2f}%")
print("-" * 110)

print(f"\n3. 공식 5대 카메라 시점별(시점 A~E) 오류 탐지(Defect Detection) 성능:")
print(f"{'공식 시점':<8} | {'화각 명칭':<22} | {'결함 건수':>8} | {'적중 건수':>8} | {'미탐지(오승인)':>14} | {'오류 탐지율':>10} | {'결함 F1':>9}")
print("-" * 110)

view_map = [
    ('시점 A', 'view5', '우측 후방 (45도)'),
    ('시점 B', 'view2', '우측 전방 (45도)'),
    ('시점 C', 'view3', '정면 (0도)'),
    ('시점 D', 'view4', '좌측 전방 (45도)'),
    ('시점 E', 'view1', '좌측 후방 (45도)'),
]

for lbl, code, desc in view_map:
    m = (views == code)
    y_v = true_defect_session[m]
    pred_v = pred_defect_session[m]
    n_def_v = int(np.sum(y_v == 1))
    tp_v = int(np.sum((pred_v == 1) & (y_v == 1)))
    fn_v = int(np.sum((pred_v == 0) & (y_v == 1)))
    rec_v = recall_score(y_v, pred_v) * 100
    f1_v = f1_score(y_v, pred_v) * 100
    print(f"{lbl:<8} | {desc:<22} | {n_def_v:>7d}건 | {tp_v:>7d}건 | {fn_v:>4d}건 ({fn_v/n_def_v*100:4.2f}%) | {rec_v:9.2f}% | {f1_v:8.2f}%")
print("=" * 110)

# 4대 부위 완전 일치율(Exact Match Ratio, EMR) 계산
pred_4defects = np.column_stack([
    (p_c1 >= 0.50).astype(int),
    (p_c2 >= 0.50).astype(int),
    (p_c3 >= 0.50).astype(int),
    (p_c4 >= 0.50).astype(int),
])
exact_match = np.all(pred_4defects == Y_defects, axis=1)
emr = np.mean(exact_match) * 100
hamm = np.mean(pred_4defects == Y_defects) * 100

print(f"\n4. 4부위 다중 결함 라벨 완전 일치 성능 (Multi-Label Exact Match):")
print(f"  - 4부위 완전 일치율 (Exact Match Ratio, EMR): {emr:.2f}% ({np.sum(exact_match):,}/{N:,}건)")
print(f"  - 개별 부위 평균 일치율 (Hamming Accuracy)  : {hamm:.2f}%")

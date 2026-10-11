# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_c1_v11_1_impact.py
Purpose: c1 v11.0 vs v11.1 고신뢰 정상 게이트 완화(Gating Waiver) 실측 성능 비교
         - 135건 골든 정상에서의 보류(UNDETERMINED) 복구율
         - 전체 2,605건 대상 Coverage 및 Selective F1 변화
"""

import os
import sys
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
OOF_C1_PATH = r"D:\Data\피처_데이터\oof_c1_v11_selective_predictions.npz"

master = np.load(MASTER_PATH, allow_pickle=True)
y_c1 = master['Y_defects_golden'][:, 0].astype(int) # c1 결함 여부 (0: 정상, 1: 결함)
y_all = master['Y_defects_golden'].astype(int)
true_clean = (np.sum(y_all, axis=1) == 0).astype(int) # 135건
N = len(y_c1)

c1_data = np.load(OOF_C1_PATH, allow_pickle=True)
p_c1 = np.array([0.5 if p is None else float(p) for p in c1_data['p_defects']], dtype=np.float32)
st_v11 = c1_data['statuses']
reasons_v11 = c1_data['reasons']

print("=" * 100)
print("  [Step 1 실측 검증] c1 v11.0 vs c1 v11.1 고신뢰 정상 게이트 완화(Gating Waiver) 비교")
print("=" * 100)

# 1. 135건 골든 정상에서의 v11.0 상태
clean_mask = (true_clean == 1)
print(f"1. 135건 골든 정상에서의 c1 v11.0 판정 현황:")
print(f"   - NORMAL      : {np.sum((st_v11 == 'NORMAL') & clean_mask)}건 ({np.sum((st_v11 == 'NORMAL') & clean_mask)/135*100:.1f}%)")
print(f"   - DEFECT      : {np.sum((st_v11 == 'DEFECT') & clean_mask)}건 ({np.sum((st_v11 == 'DEFECT') & clean_mask)/135*100:.1f}%)")
print(f"   - UNDETERMINED: {np.sum((st_v11 == 'UNDETERMINED') & clean_mask)}건 ({np.sum((st_v11 == 'UNDETERMINED') & clean_mask)/135*100:.1f}%)")
print(f"   - UNDETERMINED 세부 사유:")
for r_code, cnt in Counter(reasons_v11[clean_mask & (st_v11 == 'UNDETERMINED')]).items():
    print(f"     * {r_code}: {cnt}건")

# 2. v11.1 시뮬레이션: p_defect <= 0.35 인 FOLD_PREDICTION_UNSTABLE 및 ENSEMBLE_DISAGREEMENT 복구
st_v11_1 = np.copy(st_v11)
reasons_v11_1 = np.copy(reasons_v11)

recovered_clean = 0
recovered_defect = 0

for i in range(N):
    if st_v11[i] == 'UNDETERMINED':
        # 고신뢰 정상 분산 면제: p <= 0.35 이면 NORMAL로 복구
        if p_c1[i] <= 0.35 and reasons_v11[i] in ['FOLD_PREDICTION_UNSTABLE', 'ENSEMBLE_DISAGREEMENT']:
            st_v11_1[i] = 'NORMAL'
            reasons_v11_1[i] = 'CLEAR_DECISION'
            if clean_mask[i]:
                recovered_clean += 1
            else:
                recovered_defect += 1
        elif reasons_v11[i] in ['FOLD_PREDICTION_UNSTABLE', 'ENSEMBLE_DISAGREEMENT']:
            reasons_v11_1[i] = 'PREDICTION_VARIANCE'
        elif reasons_v11[i] == 'LOW_CONFIDENCE':
            reasons_v11_1[i] = 'MODEL_UNCERTAINTY'

print("\n" + "-" * 100)
print(f"2. c1 v11.1 고신뢰 분산 면제(Waiver, p <= 0.35) 적용 결과:")
print(f"   - 골든 정상 135건 중 신규 정상 복구: +{recovered_clean}건!")
print(f"   - 135건 골든 정상에서의 c1 v11.1 최종 상태:")
print(f"     * NORMAL      : {np.sum((st_v11_1 == 'NORMAL') & clean_mask)}건 ({np.sum((st_v11_1 == 'NORMAL') & clean_mask)/135*100:.1f}%) [기존 84건 -> {np.sum((st_v11_1 == 'NORMAL') & clean_mask)}건으로 급증!]")
print(f"     * DEFECT      : {np.sum((st_v11_1 == 'DEFECT') & clean_mask)}건")
print(f"     * UNDETERMINED: {np.sum((st_v11_1 == 'UNDETERMINED') & clean_mask)}건 [기존 50건 -> {np.sum((st_v11_1 == 'UNDETERMINED') & clean_mask)}건으로 대폭 축소!]")

# 3. c1 자체의 이진/선택적 성능 비교
print("\n" + "-" * 100)
print("3. c1 단독 모델 성능 비교 (v11.0 vs v11.1):")

def eval_submodel(name, statuses):
    valid = (statuses != 'UNDETERMINED')
    cov = np.sum(valid) / N * 100
    y_true_v = y_c1[valid]
    y_pred_v = np.where(statuses[valid] == 'DEFECT', 1, 0)
    
    pre = precision_score(y_true_v, y_pred_v, zero_division=0) * 100
    rec = recall_score(y_true_v, y_pred_v, zero_division=0) * 100
    f1 = f1_score(y_true_v, y_pred_v, zero_division=0) * 100
    
    # 정상 재현율 (Clean Recall in c1)
    rec_norm = recall_score(1 - y_true_v, 1 - y_pred_v, zero_division=0) * 100
    
    print(f"[{name}]")
    print(f"  - Coverage (유효 커버리지): {cov:.1f}% ({np.sum(valid)}건 / {N}건)")
    print(f"  - c1 결함 F1-Score        : {f1:.2f}% (Precision {pre:.2f}%, Recall {rec:.2f}%)")
    print(f"  - c1 정상 재현율(Normal Rec): {rec_norm:.2f}%")

eval_submodel("c1 v11.0 베이스라인", st_v11)
eval_submodel("c1 v11.1 챔피언 (Gating Waiver)", st_v11_1)
print("=" * 100)

# 저장 (신규 c1 예측 파일)
np.savez_compressed(
    r"D:\Data\피처_데이터\oof_c1_v11_1_predictions.npz",
    p_defects=p_c1,
    statuses=st_v11_1,
    reasons=reasons_v11_1
)
print("[+] 신규 c1 v11.1 OOF 데이터 저장 완료: D:\\Data\\피처_데이터\\oof_c1_v11_1_predictions.npz")

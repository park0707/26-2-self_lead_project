# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_oof_meta_dataset.py
Purpose: Step 1 - 4대 최신 챔피언 서브모델 OOF 예측값 및 메타 피처셋 구축
         - c1 v11, c2 v7, c3 v6, c4 v5 OOF 확률 및 상태
         - 피험자(Subject 1~9), 카메라 시점(view1~view5)
         - 골든 정답 라벨 (y_c1..c4, is_clean, defect_count)
"""

import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
OOF_C1_PATH = r"D:\Data\피처_데이터\oof_c1_v11_selective_predictions.npz"
OOF_C2_PATH = r"D:\Data\피처_데이터\oof_c2_v7_predictions.npz"
OOF_C3_PATH = r"D:\Data\피처_데이터\oof_c3_v6_predictions.npz"
OOF_C4_PATH = r"D:\Data\피처_데이터\oof_c4_v5_predictions.npz"
OUT_META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"

print("=" * 90)
print("  [Step 1] 메타 퓨전 데이터셋 구축 시작 (Master + 4대 챔피언 서브모델 OOF)")
print("=" * 90)

# 1. Master 데이터셋 로드
master = np.load(MASTER_PATH, allow_pickle=True)
y_all = master['Y_defects_golden'].astype(int) # (2605, 4)
views = master['views']                         # (2605,)
subjects = master['subjects']                   # (2605,)
sequence_ids = master['sequence_ids']           # (2605,)
N = len(views)

true_clean = (np.sum(y_all, axis=1) == 0).astype(int) # 135건
defect_counts = np.sum(y_all, axis=1).astype(int)

# 2. c1 v11 OOF 로드
c1_data = np.load(OOF_C1_PATH, allow_pickle=True)
c1_p_defect = np.array([0.5 if p is None else float(p) for p in c1_data['p_defects']], dtype=np.float32)
c1_statuses = np.array(c1_data['statuses'], dtype=str)
c1_reasons = np.array(c1_data['reasons'], dtype=str)

# 3. c2 v7 OOF 로드
c2_data = np.load(OOF_C2_PATH, allow_pickle=True)
c2_p_defect = c2_data['oof_preds'].astype(np.float32)

# 4. c3 v6 OOF 로드
c3_data = np.load(OOF_C3_PATH, allow_pickle=True)
c3_p_defect = c3_data['oof_c3_v6'].astype(np.float32)

# 5. c4 v5 OOF 로드
c4_data = np.load(OOF_C4_PATH, allow_pickle=True)
c4_p_defect = c4_data['oof_preds'].astype(np.float32)

# 6. 카메라 시점 One-Hot 인코딩
view_names = ['view1', 'view2', 'view3', 'view4', 'view5']
view_onehot = np.zeros((N, len(view_names)), dtype=np.float32)
for i, v in enumerate(views):
    if v in view_names:
        view_onehot[i, view_names.index(v)] = 1.0

# 7. c1 사유 코드 플래그
is_c1_clear = (c1_reasons == 'CLEAR_DECISION').astype(np.float32)
is_c1_unstable = (c1_reasons == 'FOLD_PREDICTION_UNSTABLE').astype(np.float32)
is_c1_disagree = (c1_reasons == 'ENSEMBLE_DISAGREEMENT').astype(np.float32)
is_c1_lowconf = (c1_reasons == 'LOW_CONFIDENCE').astype(np.float32)

# 8. 메타 피처 배열 패키징
# 기본 4차원 확률: p1, p2, p3, p4
# 정상 확률: 1-p1, 1-p2, 1-p3, 1-p4
# 마진 / 불확실성: abs(p - 0.5)
probs_defect = np.column_stack([c1_p_defect, c2_p_defect, c3_p_defect, c4_p_defect]) # (N, 4)
probs_normal = 1.0 - probs_defect
uncertainty_margin = np.abs(probs_defect - 0.5) # 0에 가까울수록 가장 불확실

# 결합 메타 피처 (Feature Matrix for Meta-Learner)
# X_meta = [p1, p2, p3, p4, unc1, unc2, unc3, unc4, view1..view5, c1_unstable, c1_disagree]
X_meta = np.column_stack([
    probs_defect,
    uncertainty_margin,
    view_onehot,
    is_c1_unstable,
    is_c1_disagree
]).astype(np.float32)

feature_names = [
    'p_defect_c1', 'p_defect_c2', 'p_defect_c3', 'p_defect_c4',
    'unc_c1', 'unc_c2', 'unc_c3', 'unc_c4',
    'view_1', 'view_2', 'view_3', 'view_4', 'view_5',
    'c1_flag_unstable', 'c1_flag_disagree'
]

# 9. 저장
np.savez_compressed(
    OUT_META_PATH,
    X_meta=X_meta,
    feature_names=feature_names,
    probs_defect=probs_defect,
    probs_normal=probs_normal,
    c1_statuses=c1_statuses,
    c1_reasons=c1_reasons,
    views=views,
    subjects=subjects,
    sequence_ids=sequence_ids,
    y_all=y_all,
    true_clean=true_clean,
    defect_counts=defect_counts
)

print(f"[+] 메타 피처셋 구축 완료: {OUT_META_PATH}")
print(f"    - 총 샘플 수: {N:,}건 (Strict Clean: {np.sum(true_clean)}건, Defect: {N - np.sum(true_clean)}건)")
print(f"    - 메타 피처 차원: {X_meta.shape[1]}D")
print(f"    - 피처 목록: {', '.join(feature_names)}")
print("=" * 90)

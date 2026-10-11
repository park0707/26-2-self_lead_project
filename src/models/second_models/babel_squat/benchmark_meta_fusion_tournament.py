# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: benchmark_meta_fusion_tournament.py
Purpose: Step 2 - 바벨 스쿼트 세션 메타 결합기(Meta-Classifier) 3종 토너먼트 실측 비교
         - 베이스라인: 현행 v6.0 (Strict AND)
         - 후보 A: Weighted Noisy-OR 소프트 리스크 퓨전
         - 후보 B: L2 규제 로지스틱 회귀 메타 분류기 (5-Fold GroupKFold)
         - 후보 C: Shallow CatBoost 메타 분류기 (5-Fold GroupKFold)
         - 안전성 제약: Defect-to-Clean 오승인 최소화 및 Clean F1 극대화
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"

print("=" * 110)
print("  [Step 2] 세션 메타 결합기(Meta-Classifier) 3종 토너먼트 실측 비교")
print("  - 데이터: AI-Hub 피험자 1~9 전수 2,605건 (Clean: 135건, Defect: 2,470건)")
print("  - 평가 원칙: 5-Fold StratifiedGroupKFold 피험자 완전 격리")
print("=" * 110)

data = np.load(META_PATH, allow_pickle=True)
X_meta = data['X_meta']
feature_names = list(data['feature_names'])
probs_defect = data['probs_defect'] # [p1, p2, p3, p4]
c1_statuses = data['c1_statuses']
c1_reasons = data['c1_reasons']
views = data['views']
subjects = data['subjects']
y_all = data['y_all']               # (2605, 4)
true_clean = data['true_clean']     # (2605,) 135건
N = len(true_clean)

# -------------------------------------------------------------
# 1. 베이스라인: 현행 v6.0 Strict AND 계산
# -------------------------------------------------------------
p_c1 = probs_defect[:, 0]
p_c2 = probs_defect[:, 1]
p_c3 = probs_defect[:, 2]
p_c4 = probs_defect[:, 3]

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
pred_clean_base = np.all(preds_base == 0, axis=1).astype(int)

def evaluate_predictions(name, pred_clean_binary, pred_clean_score=None):
    tp = int(np.sum((pred_clean_binary == 1) & (true_clean == 1)))
    fp = int(np.sum((pred_clean_binary == 1) & (true_clean == 0)))
    fn = int(np.sum((pred_clean_binary == 0) & (true_clean == 1)))
    tn = int(np.sum((pred_clean_binary == 0) & (true_clean == 0)))
    
    pre = precision_score(true_clean, pred_clean_binary, zero_division=0) * 100
    rec = recall_score(true_clean, pred_clean_binary, zero_division=0) * 100
    f1 = f1_score(true_clean, pred_clean_binary, zero_division=0) * 100
    auc = roc_auc_score(true_clean, pred_clean_score) * 100 if pred_clean_score is not None else 0.0
    
    # 안전 지표: 전체 결함(2,470건) 중 Clean으로 오인 승인된 비율
    defect_total = int(np.sum(true_clean == 0))
    false_approval_rate = (fp / defect_total) * 100
    
    print(f"[{name}]")
    print(f"  - 예측 Clean 건수 : {np.sum(pred_clean_binary):>4d}건 (TP: {tp:>3d}, FP: {fp:>3d}, FN: {fn:>3d})")
    print(f"  - Clean Precision: {pre:6.2f}%")
    print(f"  - Clean Recall   : {rec:6.2f}%")
    print(f"  - Clean F1-Score : {f1:6.2f}%")
    if pred_clean_score is not None:
        print(f"  - Clean PR/ROC-AUC: {auc:6.2f}%")
    print(f"  - [안전성] Defect-to-Clean 오승인: {fp:>3d}건 / {defect_total}건 ({false_approval_rate:4.2f}%)")
    
    # 5대 시점별 Clean Recall 출력
    view_recs = []
    for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
        v_idx = np.where(views == v)[0]
        v_true = true_clean[v_idx]
        v_pred = pred_clean_binary[v_idx]
        v_rec = recall_score(v_true, v_pred, zero_division=0) * 100
        view_recs.append(f"{v}:{v_rec:4.1f}%")
    print(f"  - 시점별 정상 재현율 : {' | '.join(view_recs)}")
    print("-" * 110)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'pre': pre, 'rec': rec, 'f1': f1, 'auc': auc, 'fp_rate': false_approval_rate}

# 베이스라인 평가
res_base = evaluate_predictions("Baseline: 현행 v6.0 (Strict AND)", pred_clean_base)


# -------------------------------------------------------------
# 2. 후보 A: Weighted Noisy-OR 소프트 리스크 퓨전
# -------------------------------------------------------------
# 원리:
# 각 부위의 결함 확률 p_j 에 가중치 w_j 적용
# 후방 시점(view1, view5)에서 c2는 얼굴 가림이 발생하므로 가중치 감쇄 w_c2_rear 적용
# p_session_defect = 1 - prod(1 - w_j * p_j)
# p_session_clean = prod(1 - w_j * p_j)

# 그리드 서치로 최적의 w_c2_rear 및 t_clean 탐색 (안전 오승인율 0.5% 이하 제약)
best_noisy_f1 = 0.0
best_noisy_pred = None
best_noisy_score = None
best_noisy_params = None

w_candidates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
t_candidates = np.linspace(0.50, 0.95, 46)

for w_rear in w_candidates:
    # 각 샘플별 p_session_clean 계산
    clean_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        w2 = w_rear if views[i] in ['view1', 'view5'] else 1.0
        # c1, c3, c4는 1.0 가중치
        # 단, c1에서 FOLD_UNSTABLE 보류였으나 p<=0.40인 경우는 실측 p1 그대로 인정
        p1 = p_c1[i]
        p2 = p_c2[i] * w2
        p3 = p_c3[i]
        p4 = p_c4[i]
        
        # 각 부위가 정상일 확률: (1 - p)
        # 세션 전체가 정상일 결합 확률: (1-p1)*(1-p2)*(1-p3)*(1-p4)
        clean_scores[i] = (1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4)
        
    for t in t_candidates:
        pred_bin = (clean_scores >= t).astype(int)
        fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
        # 안전 제약: FP <= 10건 (Defect 오승인 0.4% 이하)
        if fp <= 10:
            f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100
            if f1 > best_noisy_f1:
                best_noisy_f1 = f1
                best_noisy_pred = pred_bin
                best_noisy_score = clean_scores
                best_noisy_params = (w_rear, t)

print(f"[후보 A 파라미터 튜닝 결과] w_rear={best_noisy_params[0]}, t_clean={best_noisy_params[1]:.2f}")
res_noisy = evaluate_predictions("후보 A: Weighted Noisy-OR 소프트 리스크 퓨전", best_noisy_pred, best_noisy_score)


# -------------------------------------------------------------
# 3. 후보 B: L2 규제 로지스틱 회귀 메타 분류기 (5-Fold GroupKFold)
# -------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5)
oof_pred_lr = np.zeros(N, dtype=np.float32)

# 타깃: 1 if true_clean else 0
y_target = true_clean

for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_meta, y_target, groups=subjects)):
    X_trn, y_trn = X_meta[trn_idx], y_target[trn_idx]
    X_val, y_val = X_meta[val_idx], y_target[val_idx]
    
    # L2 규제 로지스틱 회귀 (C=0.5, class_weight='balanced')
    lr = LogisticRegression(C=0.5, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
    lr.fit(X_trn, y_trn)
    
    # 검증셋 예측 확률 (Clean일 확률)
    oof_pred_lr[val_idx] = lr.predict_proba(X_val)[:, 1]

# 안전 제약(FP <= 10) 하에서 최적 threshold 탐색
best_lr_f1 = 0.0
best_lr_bin = None
best_lr_t = 0.5

for t in np.linspace(0.3, 0.95, 66):
    pred_bin = (oof_pred_lr >= t).astype(int)
    fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
    if fp <= 10:
        f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100
        if f1 > best_lr_f1:
            best_lr_f1 = f1
            best_lr_bin = pred_bin
            best_lr_t = t

print(f"[후보 B 최적 임계치] t_clean={best_lr_t:.3f}")
res_lr = evaluate_predictions("후보 B: L2 규제 로지스틱 회귀 메타 분류기 (OOF)", best_lr_bin, oof_pred_lr)


# -------------------------------------------------------------
# 4. 후보 C: Shallow CatBoost Meta-Learner (5-Fold GroupKFold)
# -------------------------------------------------------------
oof_pred_cb = np.zeros(N, dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_meta, y_target, groups=subjects)):
    X_trn, y_trn = X_meta[trn_idx], y_target[trn_idx]
    X_val, y_val = X_meta[val_idx], y_target[val_idx]
    
    cb = CatBoostClassifier(
        depth=3,
        iterations=80,
        learning_rate=0.08,
        l2_leaf_reg=5.0,
        auto_class_weights='Balanced',
        verbose=0,
        random_seed=42
    )
    cb.fit(X_trn, y_trn)
    oof_pred_cb[val_idx] = cb.predict_proba(X_val)[:, 1]

best_cb_f1 = 0.0
best_cb_bin = None
best_cb_t = 0.5

for t in np.linspace(0.3, 0.95, 66):
    pred_bin = (oof_pred_cb >= t).astype(int)
    fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
    if fp <= 10:
        f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100
        if f1 > best_cb_f1:
            best_cb_f1 = f1
            best_cb_bin = pred_bin
            best_cb_t = t

print(f"[후보 C 최적 임계치] t_clean={best_cb_t:.3f}")
res_cb = evaluate_predictions("후보 C: Shallow CatBoost 메타 분류기 (OOF)", best_cb_bin, oof_pred_cb)


# -------------------------------------------------------------
# 5. 최종 종합 비교 요약표 출력
# -------------------------------------------------------------
print("=" * 110)
print(f"{'모델 결합 방식':<38} | {'Clean Pre':>9} | {'Clean Rec':>9} | {'Clean F1':>9} | {'ROC-AUC':>9} | {'Defect 오승인':>12}")
print("-" * 110)
print(f"{'현행 v6.0 (Strict AND)':<38} | {res_base['pre']:8.2f}% | {res_base['rec']:8.2f}% | {res_base['f1']:8.2f}% | {'-':>9} | {res_base['fp']:>3d}건 ({res_base['fp_rate']:4.2f}%)")
print(f"{'후보 A (Weighted Noisy-OR Soft-Risk)':<38} | {res_noisy['pre']:8.2f}% | {res_noisy['rec']:8.2f}% | {res_noisy['f1']:8.2f}% | {res_noisy['auc']:8.2f}% | {res_noisy['fp']:>3d}건 ({res_noisy['fp_rate']:4.2f}%)")
print(f"{'후보 B (L2 Logistic Meta-Learner OOF)':<38} | {res_lr['pre']:8.2f}% | {res_lr['rec']:8.2f}% | {res_lr['f1']:8.2f}% | {res_lr['auc']:8.2f}% | {res_lr['fp']:>3d}건 ({res_lr['fp_rate']:4.2f}%)")
print(f"{'후보 C (Shallow CatBoost Meta-Learner OOF)':<38} | {res_cb['pre']:8.2f}% | {res_cb['rec']:8.2f}% | {res_cb['f1']:8.2f}% | {res_cb['auc']:8.2f}% | {res_cb['fp']:>3d}건 ({res_cb['fp_rate']:4.2f}%)")
print("=" * 110)

# 결과 저장
np.savez_compressed(
    r"D:\Data\피처_데이터\meta_tournament_results.npz",
    oof_pred_lr=oof_pred_lr,
    oof_pred_cb=oof_pred_cb,
    best_noisy_score=best_noisy_score,
    true_clean=true_clean
)

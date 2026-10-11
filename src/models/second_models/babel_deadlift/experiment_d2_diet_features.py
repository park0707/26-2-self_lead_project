# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: experiment_d2_diet_features.py
Purpose: 데드리프트에 적합하지 않은 노이즈 피처 선별 및 최적 정예 피처셋(v2.0) 탐색 실험
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq, FEATURE_NAMES_30_D2

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

print(f"[+] 총 데이터셋 파일 로드: {len(files)}개")

X_list = []
y_list = []
keys_list = []
views_list = []
subs_list = []

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

X_all = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
groups = np.array(subs_list)
views = np.array(views_list)

# 피처 목록 확인
print(f"X_all shape: {X_all.shape}, Defect 비율: {np.mean(y)*100:.2f}%")

# 후보군 정의
# 1) 기존 30D (전체)
# 2) v1.0 22D (스쿼트 계승 원본)
# 3) Top 9 (통계적 유의성 p < 0.05인 핵심 9대 피처)
# 4) Top 12 (p < 0.10인 11대 피처 + 정강이 경사)
# 5) 데드리프트 맞춤 정예 14D (스쿼트 전용 발끝 회전 4개 + 3D 관상각 4개 + 수직 가동범위/시점 2개 + 2차 미분 급변도 4개 등 총 16개 노이즈 피처 완전 제거)
# 6) 데드리프트 초정예 10D

candidate_sets = {
    "1. Baseline 30D (기존 v2.0)": list(range(30)),
    "2. Baseline 22D (v1.0 스쿼트 원본)": list(range(22)),
    "3. Top 9 (p < 0.05 통계적 유의 피처)": [
        24, # d2_kp_ratio_min_abs (p=3.69e-3, AUC 0.6140)
        3,  # d2_ka_valgus_duration_ratio (p=1.30e-3, AUC 0.6104)
        4,  # d2_ka_absolute_deficit_integ (p=5.40e-4, AUC 0.6099)
        1,  # d2_ka_ratio_min_abs (p=1.79e-2, AUC 0.6004)
        8,  # d2_valgus_concurrence_max (p=4.61e-3, AUC 0.6001)
        7,  # d2_valgus_abs_sum_max (p=2.40e-3, AUC 0.5974)
        25, # d2_kp_ratio_p10 (p=1.23e-2, AUC 0.5936)
        26, # d2_valgus_norm_hip_max (p=2.31e-2, AUC 0.5829)
        6,  # d2_valgus_abs_sum_nadir (p=2.90e-2, AUC 0.5704)
    ],
    "4. Top 12 (p < 0.10 + 정강이 관상 경사)": [
        24, 3, 4, 1, 8, 7, 25, 26, 6,
        2,  # d2_ka_ratio_standing_abs (p=0.078)
        5,  # d2_ka_chronic_standing_valgus (p=0.080)
        28, # d2_shank_inward_tilt_p90 (AUC 0.5651)
    ],
    "5. 정예 14D (데드리프트 최적화 생체역학 정예셋)": [
        # [무릎-발목 관상면 비율 4D]
        1,  # d2_ka_ratio_min_abs
        2,  # d2_ka_ratio_standing_abs
        3,  # d2_ka_valgus_duration_ratio
        4,  # d2_ka_absolute_deficit_integ
        # [골반 관상면 투영 외반합 3D]
        6,  # d2_valgus_abs_sum_nadir
        7,  # d2_valgus_abs_sum_max
        8,  # d2_valgus_concurrence_max
        # [골반 대비 무릎 너비 비율 및 정규화 4D]
        24, # d2_kp_ratio_min_abs
        25, # d2_kp_ratio_p10
        26, # d2_valgus_norm_hip_max
        28, # d2_shank_inward_tilt_p90
        # [보조 역학 3D]
        0,  # d2_ka_ratio_nadir_abs
        5,  # d2_ka_chronic_standing_valgus
        29, # d2_bilateral_inward_sync
    ],
    "6. 초정예 10D (고변별력 단독 최우수)": [
        24, 3, 4, 1, 8, 7, 25, 26, 6, 28
    ]
}

def evaluate_feature_subset(feat_indices, name):
    X = X_all[:, feat_indices]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y), dtype=np.float32)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.8,
        'min_child_samples': 10,
        'is_unbalance': False,
        'scale_pos_weight': 2.0,
        'verbose': -1,
        'random_state': 42
    }

    cb_params = {
        'iterations': 150,
        'learning_rate': 0.03,
        'depth': 4,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'scale_pos_weight': 2.0,
        'verbose': 0,
        'random_seed': 42
    }

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=120,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        lgb_prob = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
        cb_prob = cb_model.predict_proba(X_val)[:, 1]

        oof_preds[val_idx] = 0.5 * lgb_prob + 0.5 * cb_prob

    auc = roc_auc_score(y, oof_preds)
    f1_default = f1_score(y, (oof_preds >= 0.5).astype(int), zero_division=0)
    acc_default = accuracy_score(y, (oof_preds >= 0.5).astype(int))

    # 최적 F1 임계값 탐색
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        score = f1_score(y, (oof_preds >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th

    # 3-State [0.28, 0.30] 평가
    low_th, high_th = 0.28, 0.30
    det_mask = (oof_preds < low_th) | (oof_preds >= high_th)
    coverage = np.mean(det_mask) * 100
    y_3state = (oof_preds[det_mask] >= high_th).astype(int)
    f1_3state = f1_score(y[det_mask], y_3state, zero_division=0)
    acc_3state = accuracy_score(y[det_mask], y_3state)
    rec_3state = recall_score(y[det_mask], y_3state, zero_division=0)
    prec_3state = precision_score(y[det_mask], y_3state, zero_division=0)

    return {
        'name': name,
        'n_feats': len(feat_indices),
        'auc': auc,
        'f1_default': f1_default,
        'acc_default': acc_default,
        'best_th': best_th,
        'best_f1': best_f1,
        'coverage': coverage,
        'f1_3state': f1_3state,
        'acc_3state': acc_3state,
        'prec_3state': prec_3state,
        'rec_3state': rec_3state
    }

print("\n" + "=" * 115)
print("  [데드리프트 d2 피처 다이어트 후보군 5-Fold StratifiedGroupKFold OOF 성능 비교]")
print("=" * 115)
print(f"{'후보군 명칭':<36} | {'피처수':<5} | {'AUC':<7} | {'기본 F1':<8} | {'Best F1(th)':<14} | {'3-State F1':<10} | {'3-State Acc':<11} | {'Coverage':<8}")
print("-" * 115)

results = []
for name, indices in candidate_sets.items():
    res = evaluate_feature_subset(indices, name)
    results.append(res)
    print(f"{res['name']:<36} | {res['n_feats']:<5} | {res['auc']:.4f}  | {res['f1_default']*100:.2f}%   | {res['best_f1']*100:.2f}%({res['best_th']:.2f}) | {res['f1_3state']*100:.2f}%     | {res['acc_3state']*100:.2f}%      | {res['coverage']:.1f}%")

print("=" * 115)

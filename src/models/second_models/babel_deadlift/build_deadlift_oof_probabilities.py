# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_deadlift_oof_probabilities.py
Description:
  AI-Hub 770개 데드리프트 시퀀스 전수에 대해 5대 공식 세부 모델(d1~d5)의
  5-Fold StratifiedGroupKFold OOF 예측 확률 및 핵심 도메인 피처를 통합 추출하여 캐싱
"""

import os
import sys
import glob
import json
import time
import pickle
import numpy as np

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_OOF_CACHE = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")

from d4_model_v5 import extract_d4_v5_features_from_seq, DIET_CORE_INDICES_15_D4
from d5_model_v5_1 import extract_d5_v4_gated_features_from_seq, KEPT_FEATURE_INDICES_D5

print("=" * 80)
print("  [vidio-health-trainer] 5대 세부 모델 OOF 확률 및 메타 피처 추출 파이프라인")
print("=" * 80)

# 1. 파일 목록 및 라벨 추출
raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.npz")))
print(f"[*] 총 시퀀스 수: {len(raw_files)}")

y_dict = {f'd{i}': [] for i in range(1, 6)}
subjects = []
views = []
clip_names = []
states_32 = []

for f in raw_files:
    bname = os.path.basename(f).replace('.npz', '')
    clip_names.append(bname)
    parts = bname.split('_')
    vn = parts[-1]
    views.append(vn)
    folder_name = '_'.join(parts[:-1])

    sub = 1
    for d in range(1, 10):
        if f"피험자{d}" in folder_name:
            sub = d
            break
    subjects.append(sub)

    lp = os.path.join(LABEL_DIR, f'{folder_name}.json')
    with open(lp, 'r', encoding='utf-8', errors='ignore') as jf:
        data = json.load(jf)
    conds = {c['condition']: c['value'] for c in data.get('type_info', {}).get('conditions', [])}

    d1 = 0 if conds.get('척추의 중립', True) else 1
    d2 = 0 if conds.get('발과 무릎의 방향 일치', True) else 1
    d3 = 0 if conds.get('발바닥 지면 고정', True) else 1
    d4 = 0 if conds.get('무릎과 골반이 동시에 펴짐', True) else 1
    d5 = 0 if conds.get('바벨 궤적과 몸 밀착', True) else 1

    y_dict['d1'].append(d1)
    y_dict['d2'].append(d2)
    y_dict['d3'].append(d3)
    y_dict['d4'].append(d4)
    y_dict['d5'].append(d5)

    s = (d1 << 4) | (d2 << 3) | (d3 << 2) | (d4 << 1) | d5
    states_32.append(s)

subjects = np.array(subjects, dtype=np.int64)
views = np.array(views)
y_all = np.column_stack([y_dict[f'd{i}'] for i in range(1, 6)])  # (770, 5)
states_32 = np.array(states_32, dtype=np.int64)

# 2. d4 피처 및 OOF (기존 캐시 활용)
cache_v4 = np.load(os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz"), allow_pickle=True)
meta_d4 = np.load(os.path.join(CURRENT_DIR, "d4_v5_1_diet_champion.npz"), allow_pickle=True)
X_d4_15 = cache_v4["X_gbdt"][:, meta_d4["indices"]]

# d4 OOF 산출 (5-Fold Tri-Ensemble)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_p_d4 = np.zeros(len(subjects), dtype=np.float32)
for train_idx, val_idx in sgkf.split(X_d4_15, y_all[:, 3], groups=subjects):
    m_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    m_cat = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                               auto_class_weights='Balanced', random_seed=42, verbose=0)
    m_rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    m_lgb.fit(X_d4_15[train_idx], y_all[train_idx, 3])
    m_cat.fit(X_d4_15[train_idx], y_all[train_idx, 3])
    m_rf.fit(X_d4_15[train_idx], y_all[train_idx, 3])
    oof_p_d4[val_idx] = 0.45 * m_lgb.predict_proba(X_d4_15[val_idx])[:, 1] + \
                       0.40 * m_cat.predict_proba(X_d4_15[val_idx])[:, 1] + \
                       0.15 * m_rf.predict_proba(X_d4_15[val_idx])[:, 1]
print(f"[+] d4 OOF 산출 완료! AUC: {roc_auc_score(y_all[:, 3], oof_p_d4):.4f}")

# 3. d5 피처 및 OOF
print("[*] d5 (바벨 밀착) 피처 추출 및 OOF 산출 중...")
feats_d5_list = []
for f in raw_files:
    kps = np.load(f)["kps_33_3d_64f"]
    feats_40 = extract_d5_v4_gated_features_from_seq(kps)
    feats_32 = feats_40[KEPT_FEATURE_INDICES_D5]
    feats_d5_list.append(feats_32)
X_d5_32 = np.array(feats_d5_list, dtype=np.float32)

oof_p_d5 = np.zeros(len(subjects), dtype=np.float32)
for train_idx, val_idx in sgkf.split(X_d5_32, y_all[:, 4], groups=subjects):
    m_lgb = lgb.LGBMClassifier(n_estimators=120, max_depth=5, num_leaves=20, learning_rate=0.04,
                               class_weight='balanced', random_state=42, verbose=-1)
    m_cat = CatBoostClassifier(iterations=120, depth=5, learning_rate=0.04,
                               auto_class_weights='Balanced', random_seed=42, verbose=0)
    m_lgb.fit(X_d5_32[train_idx], y_all[train_idx, 4])
    m_cat.fit(X_d5_32[train_idx], y_all[train_idx, 4])
    oof_p_d5[val_idx] = 0.50 * m_lgb.predict_proba(X_d5_32[val_idx])[:, 1] + \
                       0.50 * m_cat.predict_proba(X_d5_32[val_idx])[:, 1]
print(f"[+] d5 OOF 산출 완료! AUC: {roc_auc_score(y_all[:, 4], oof_p_d5):.4f}")

# 4. d1, d2, d3 피처 추출 및 OOF
# d1 (척추)
print("[*] d1 (척추 중립) 피처 및 OOF 산출 중...")
from d1_model_v4_3_diet import extract_d1_v4_3_diet_features_25d
feats_d1_list = []
for f in raw_files:
    kps = np.load(f)["kps_33_3d_64f"]
    feats_25 = extract_d1_v4_3_diet_features_25d(kps)
    feats_d1_list.append(feats_25)
X_d1_25 = np.array(feats_d1_list, dtype=np.float32)

oof_p_d1 = np.zeros(len(subjects), dtype=np.float32)
for train_idx, val_idx in sgkf.split(X_d1_25, y_all[:, 0], groups=subjects):
    m_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
                               class_weight='balanced', random_state=42, verbose=-1)
    m_cat = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                               auto_class_weights='Balanced', random_seed=42, verbose=0)
    m_lgb.fit(X_d1_25[train_idx], y_all[train_idx, 0])
    m_cat.fit(X_d1_25[train_idx], y_all[train_idx, 0])
    oof_p_d1[val_idx] = 0.50 * m_lgb.predict_proba(X_d1_25[val_idx])[:, 1] + \
                       0.50 * m_cat.predict_proba(X_d1_25[val_idx])[:, 1]
print(f"[+] d1 OOF 산출 완료! AUC: {roc_auc_score(y_all[:, 0], oof_p_d1):.4f}")

# d2 (무릎방향)
print("[*] d2 (발-무릎 방향) 피처 및 OOF 산출 중...")
from d2_model_v5_final import extract_d2_v5_final_features
feats_d2_list = []
for f in raw_files:
    kps = np.load(f)["kps_33_3d_64f"]
    res_d2 = extract_d2_v5_final_features(kps)
    feats_10 = res_d2[0] if isinstance(res_d2, tuple) else res_d2
    feats_d2_list.append(feats_10)
X_d2_10 = np.array(feats_d2_list, dtype=np.float32)

oof_p_d2 = np.zeros(len(subjects), dtype=np.float32)
for train_idx, val_idx in sgkf.split(X_d2_10, y_all[:, 1], groups=subjects):
    m_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.04,
                               class_weight='balanced', random_state=42, verbose=-1)
    m_cat = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.04,
                               auto_class_weights='Balanced', random_seed=42, verbose=0)
    m_lgb.fit(X_d2_10[train_idx], y_all[train_idx, 1])
    m_cat.fit(X_d2_10[train_idx], y_all[train_idx, 1])
    oof_p_d2[val_idx] = 0.50 * m_lgb.predict_proba(X_d2_10[val_idx])[:, 1] + \
                       0.50 * m_cat.predict_proba(X_d2_10[val_idx])[:, 1]
print(f"[+] d2 OOF 산출 완료! AUC: {roc_auc_score(y_all[:, 1], oof_p_d2):.4f}")

# d3 (발바닥 고정)
print("[*] d3 (발바닥 지면 고정) 피처 및 OOF 산출 중...")
from d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq
feats_d3_list = []
for f in raw_files:
    kps = np.load(f)["kps_33_3d_64f"]
    feats_17 = extract_d3_v5_3_opt_features_from_seq(kps)
    feats_d3_list.append(feats_17)
X_d3_17 = np.array(feats_d3_list, dtype=np.float32)

oof_p_d3 = np.zeros(len(subjects), dtype=np.float32)
for train_idx, val_idx in sgkf.split(X_d3_17, y_all[:, 2], groups=subjects):
    m_lgb = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.04,
                               class_weight='balanced', random_state=42, verbose=-1)
    m_cat = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.04,
                               auto_class_weights='Balanced', random_seed=42, verbose=0)
    m_lgb.fit(X_d3_17[train_idx], y_all[train_idx, 2])
    m_cat.fit(X_d3_17[train_idx], y_all[train_idx, 2])
    oof_p_d3[val_idx] = 0.50 * m_lgb.predict_proba(X_d3_17[val_idx])[:, 1] + \
                       0.50 * m_cat.predict_proba(X_d3_17[val_idx])[:, 1]
print(f"[+] d3 OOF 산출 완료! AUC: {roc_auc_score(y_all[:, 2], oof_p_d3):.4f}")

# 5. 메타 특성 벡터(48D) 합성
print("\n[*] 48D 메타 특성 텐서 합성 중...")
# 1) 확률 5D
P_5D = np.column_stack([oof_p_d1, oof_p_d2, oof_p_d3, oof_p_d4, oof_p_d5])
# 2) 마진 5D (|p - 0.5|)
M_5D = np.abs(P_5D - 0.5)
# 3) 시점 원핫 5D
# 시점 순서: view5(A), view2(B), view3(C), view4(D), view1(E)
view_codes = ['view5', 'view2', 'view3', 'view4', 'view1']
V_5D = np.zeros((len(views), 5), dtype=np.float32)
for vi, vc in enumerate(view_codes):
    V_5D[:, vi] = (views == vc).astype(np.float32)

# 4) 도메인 대표 피처 (각 모델별 5개씩 25D)
dom_d1 = X_d1_25[:, :5]
dom_d2 = X_d2_10[:, :5]
dom_d3 = X_d3_17[:, :5]
dom_d4 = X_d4_15[:, :5]
dom_d5 = X_d5_32[:, :5]
DOM_25D = np.column_stack([dom_d1, dom_d2, dom_d3, dom_d4, dom_d5])

# 5) 시계열 물리/안정성 지표 8D
# mean_prob, max_prob, min_margin, mean_margin, max_margin, entropy, std_prob, var_margin
mean_p = np.mean(P_5D, axis=1, keepdims=True)
max_p = np.max(P_5D, axis=1, keepdims=True)
min_m = np.min(M_5D, axis=1, keepdims=True)
mean_m = np.mean(M_5D, axis=1, keepdims=True)
max_m = np.max(M_5D, axis=1, keepdims=True)
entropy = -np.sum(P_5D * np.log2(np.clip(P_5D, 1e-6, 1.0)) + (1 - P_5D) * np.log2(np.clip(1 - P_5D, 1e-6, 1.0)), axis=1, keepdims=True)
std_p = np.std(P_5D, axis=1, keepdims=True)
var_m = np.var(M_5D, axis=1, keepdims=True)
PHY_8D = np.column_stack([mean_p, max_p, min_m, mean_m, max_m, entropy, std_p, var_m])

# 전체 결합 (5 + 5 + 5 + 25 + 8 = 48D)
X_meta_48 = np.column_stack([P_5D, M_5D, V_5D, DOM_25D, PHY_8D]).astype(np.float32)
# 결측치 무결성 검증
assert not np.isnan(X_meta_48).any(), "NaN found in X_meta_48"
assert not np.isinf(X_meta_48).any(), "Inf found in X_meta_48"
print(f"[+] 48D 메타 특성 텐서 검증 완료! Shape: {X_meta_48.shape}, NaN=0, Inf=0")

# 6. 캐싱 저장
np.savez_compressed(
    OUT_OOF_CACHE,
    X_meta=X_meta_48,
    y=y_all,
    states_32=states_32,
    oof_probs=P_5D,
    subjects=subjects,
    views=views,
    clip_names=np.array(clip_names)
)
print(f"[+] 메타 데이터셋 캐싱 완료: {OUT_OOF_CACHE}")

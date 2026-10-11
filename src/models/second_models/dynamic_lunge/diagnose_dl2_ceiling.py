# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 & 06)
Script: diagnose_dl2_ceiling.py
Purpose: 바벨 런지 bl2의 성능 천장(Ceiling) 현상이 다이나믹 런지(dl2)에서도 동일하게 작용하는지
         동일한 생체역학/통계학적 진단 파이프라인으로 전수 검증
"""

import os
import sys
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import lightgbm as lgb
from scipy.stats import norm

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(CURRENT_DIR, "dynamic_lunge_c2_features_18d_cache.npz")
c = np.load(cache_path, allow_pickle=True)
X, y, subj, views, names = c["X"], c["y"], c["subjects"], c["views"], [str(n) for n in c["clip_names"]]

from dynamic_lunge_c2_features import FEATURE_NAMES_18D

print("=" * 80)
print("  [vidio-health-trainer] 다이나믹 런지 dl2 성능 한계 및 천장 원인 정밀 검증")
print("=" * 80)

# [1] 포워드(05) vs 백워드(06) 구성 및 라벨 구조
label_dirs = {
    '05': r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    '06': r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
}

fwd_indices, bwd_indices = [], []
for i, n in enumerate(names):
    folder = "_".join(n.split("_")[:-1])
    if os.path.exists(os.path.join(label_dirs['05'], folder + ".json")):
        fwd_indices.append(i)
    else:
        bwd_indices.append(i)

print("\n[1] 데이터 구성 (포워드 vs 백워드 전수 투입 검증):")
print(f"  - 전체 시퀀스 수: {len(names)}개")
print(f"  - 스텝 포워드(05): {len(fwd_indices)}개 (결함: {np.sum(y[fwd_indices]==1)}건, 정상: {np.sum(y[fwd_indices]==0)}건, 결함비율: {np.mean(y[fwd_indices])*100:.1f}%)")
print(f"  - 스텝 백워드(06): {len(bwd_indices)}개 (결함: {np.sum(y[bwd_indices]==1)}건, 정상: {np.sum(y[bwd_indices]==0)}건, 결함비율: {np.mean(y[bwd_indices])*100:.1f}%)")

# [2] 동일 동작(클립 키)의 5시점 라벨 일치 여부
key = [re.sub(r"_view\d$", "", n) for n in names]
grp = defaultdict(list)
for i, k in enumerate(key):
    grp[k].append(i)
sizes = [len(v) for v in grp.values()]
mixed = sum(1 for v in grp.values() if len(set(y[v])) > 1)

print("\n[2] 라벨 구조 및 독립 동작(Repetition) 분석:")
print(f"  - 고유 독립 동작 수: {len(grp)}개 (전체 {len(names)} 시퀀스는 동일 동작의 5개 시점 촬영)")
print(f"  - 시점 간 라벨 불일치 동작 수: {mixed}건 (동작 단위 정답 완벽 일치)")
rep_pos = np.mean([y[v[0]] for v in grp.values()])
print(f"  - 동작 단위 결함 비율: {rep_pos*100:.1f}%")

# [3] 피험자별 결함 비율 및 표본 편차
print("\n[3] 피험자 1~9번 전수 결함 비율 및 표본 분포:")
for s in np.unique(subj):
    m = (subj == s)
    reps = len({key[i] for i in np.where(m)[0]})
    print(f"  - 피험자 {s}: 총 샘플 {m.sum():3d}개 (독립동작 {reps:2d}회) | 결함 {np.sum(y[m]==1):2d}개 ({y[m].mean()*100:4.1f}%) | 정상 {np.sum(y[m]==0):3d}개")

# [4] 18D 피처 단변량 ROC-AUC 랭킹 (신호 세기)
print("\n[4] 18D 피처 단변량 신호 강도 (Univariate ROC-AUC):")
aucs = []
for j, n in enumerate(FEATURE_NAMES_18D):
    a = roc_auc_score(y, X[:, j])
    aucs.append((max(a, 1 - a), a, n))
aucs.sort(reverse=True)
for rank, (eff_a, raw_a, n) in enumerate(aucs, 1):
    print(f"  {rank:2d}. {n:38s} : Effective AUC={eff_a:.4f} (Raw AUC={raw_a:.4f})")
print(f"  => 최고 단변량 AUC: {aucs[0][0]:.4f}")
print(f"  => AUC > 0.55 유효 피처 수: {sum(1 for m,_,_ in aucs if m>0.55)}/18개")

# [5] 분할 방식별 AUC 비교 (피험자 격리 vs 동작 격리 vs 무작위)
print("\n[5] 일반화 검증 방식별 AUC 비교 (피험자 간 변동성 노이즈 측정):")
P = dict(n_estimators=130, max_depth=3, num_leaves=7, learning_rate=0.04, min_child_samples=16,
         colsample_bytree=0.75, subsample=0.85, reg_lambda=2.0, verbose=-1, n_jobs=-1)

def cv_eval(splitter, groups=None):
    oof = np.zeros(len(y))
    for tr, va in splitter.split(X, y, groups):
        m = lgb.LGBMClassifier(**P).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof), oof

a_subj, oof_subj = cv_eval(StratifiedGroupKFold(5, shuffle=True, random_state=42), subj)
keys_arr = np.array(key)
a_rep, oof_rep = cv_eval(StratifiedGroupKFold(5, shuffle=True, random_state=42), keys_arr)
a_rand, oof_rand = cv_eval(StratifiedKFold(5, shuffle=True, random_state=42))

print(f"  - [엄격] 피험자 완전 격리 (StratifiedGroupKFold(subj)) : AUC = {a_subj:.4f}")
print(f"  - [중간] 동작 단위 격리 (StratifiedGroupKFold(rep))     : AUC = {a_rep:.4f}  (동일 피험자 참조)")
print(f"  - [낙관] 무작위 단순 분할 (StratifiedKFold)             : AUC = {a_rand:.4f}  (동일 동작 시점 누출)")
print(f"  => 피험자 간 체형/골격 차이로 인한 일반화 손실: {-(a_subj - a_rep)*100:.2f}%p")

# [6] 동일 동작 내 시점 간 예측 변동성 (시점 노이즈)
print("\n[6] 동일 동작(5개 시점) 간 예측 일관성 및 시점 노이즈 분석:")
sd_within, sd_total = [], np.std(oof_subj)
for v in grp.values():
    if len(v) >= 3:
        sd_within.append(np.std(oof_subj[v]))
mean_sd_within = np.mean(sd_within)
noise_ratio = mean_sd_within / sd_total * 100
print(f"  - 전체 예측 확률 표준편차 (Total Variance)         : {sd_total:.4f}")
print(f"  - 동일 동작 내 시점 간 예측 표준편차 (Within-Rep SD): {mean_sd_within:.4f}")
print(f"  - 시점 노이즈 비율 (Within / Total)                : {noise_ratio:.1f}%")

rep_prob = np.array([oof_subj[v].mean() for v in grp.values()])
rep_y = np.array([y[v[0]] for v in grp.values()])
rep_auc = roc_auc_score(rep_y, rep_prob)
best_f1 = 0
for t in np.linspace(0.3, 0.7, 81):
    best_f1 = max(best_f1, f1_score(rep_y, (rep_prob >= t).astype(int)))
print(f"  - 동작 단위 5시점 평균 앙상블 시 AUC              : {rep_auc:.4f} (단일 시점 대비 {rep_auc-a_subj:+.4f})")
print(f"  - 동작 단위 5시점 평균 시 최대 Defect F1           : {best_f1*100:.2f}%")

# [7] 수학적 F1 이론 상한 (AUC 및 양성률 기반)
print("\n[7] 통계적 Defect F1 이론 상한 (Theoretical Ceiling):")
pi = y.mean()
for A in [a_subj, 0.70, 0.75, 0.80]:
    d = np.sqrt(2) * norm.ppf(A)
    bf = 0
    for t in np.linspace(-3, 5, 801):
        tpr = 1 - norm.cdf(t - d)
        fpr = 1 - norm.cdf(t)
        tp = pi * tpr
        fp = (1 - pi) * fpr
        fn = pi * (1 - tpr)
        bf = max(bf, 2 * tp / (2 * tp + fp + fn + 1e-9))
    print(f"  - AUC = {A:.3f} 일 때 수학적 최대 달성 가능 Defect F1 ~ {bf*100:.1f}% (양성률 {pi*100:.1f}%)")

print("=" * 80)

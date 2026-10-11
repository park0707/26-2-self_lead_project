# -*- coding: utf-8 -*-
"""bl2 성능 천장(45%) 원인 진단 스크립트 (분석 전용, 모델/피처 수정 없음)"""
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import lightgbm as lgb

c = np.load("lunge_c2_features_v6_18d_delta_cache.npz", allow_pickle=True)
X, y, subj, views, names = c["X"], c["y"], c["subjects"], c["views"], [str(n) for n in c["clip_names"]]
from lunge_c2_features_v6 import FEATURE_NAMES_18D

print("=" * 70)
print("[1] 라벨 구조: 동일 동작(클립 키)의 5시점 라벨 일치 여부")
key = [re.sub(r"_view\d$", "", n) for n in names]
grp = defaultdict(list)
for i, k in enumerate(key):
    grp[k].append(i)
sizes = [len(v) for v in grp.values()]
mixed = sum(1 for v in grp.values() if len(set(y[v])) > 1)
print(f"  고유 동작(rep) 수={len(grp)}, 시점 수 분포={np.bincount(sizes)}")
print(f"  시점 간 라벨 불일치 동작 수={mixed}")
print(f"  => 1081 샘플의 실질 독립 샘플 수는 약 {len(grp)}개 (시점은 같은 동작의 재촬영)")
rep_pos = np.mean([y[v[0]] for v in grp.values()])
print(f"  동작 단위 결함 비율={rep_pos*100:.1f}%")

print("=" * 70)
print("[2] 피험자별 결함 비율 / 표본")
for s in np.unique(subj):
    m = subj == s
    reps = len({key[i] for i in np.where(m)[0]})
    print(f"  피험자{s}: 샘플={m.sum():4d}, 독립동작={reps:3d}, 결함비율={y[m].mean()*100:5.1f}%")

print("=" * 70)
print("[3] 피처 단변량 AUC (상위 10 / 하위)")
aucs = []
for j, n in enumerate(FEATURE_NAMES_18D):
    a = roc_auc_score(y, X[:, j])
    aucs.append((max(a, 1 - a), a, n))
aucs.sort(reverse=True)
for m, a, n in aucs[:10]:
    print(f"  {n:42s} AUC={a:.4f}")
print(f"  최고 단변량 AUC={aucs[0][0]:.4f}, 0.55 초과 피처 수={sum(1 for m,_,_ in aucs if m>0.55)}/18")

print("=" * 70)
print("[4] 분할 방식별 AUC (피험자 격리 vs 무작위 vs 동작 격리)")
P = dict(n_estimators=130, max_depth=3, num_leaves=7, learning_rate=0.04, min_child_samples=18,
         colsample_bytree=0.75, subsample=0.85, reg_lambda=2.0, verbose=-1, n_jobs=-1)

def cv_auc(splitter, groups=None):
    oof = np.zeros(len(y))
    for tr, va in splitter.split(X, y, groups):
        m = lgb.LGBMClassifier(**P).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof), oof

a_subj, oof_subj = cv_auc(StratifiedGroupKFold(5, shuffle=True, random_state=42), subj)
a_rand, _ = cv_auc(StratifiedKFold(5, shuffle=True, random_state=42))
keys_arr = np.array(key)
a_rep, _ = cv_auc(StratifiedGroupKFold(5, shuffle=True, random_state=42), keys_arr)
print(f"  피험자 격리 AUC={a_subj:.4f}")
print(f"  동작(rep) 격리 AUC={a_rep:.4f}  (동일 피험자 다른 동작은 학습에 포함)")
print(f"  무작위 분할 AUC={a_rand:.4f}  (같은 동작의 다른 시점이 누출됨 => 낙관적)")

print("=" * 70)
print("[5] 같은 동작의 5시점 예측 일관성 (측정 노이즈 지표)")
sd_within, sd_total = [], np.std(oof_subj)
corr_pairs = []
for v in grp.values():
    if len(v) >= 3:
        sd_within.append(np.std(oof_subj[v]))
print(f"  전체 예측확률 표준편차={sd_total:.4f}, 동일동작 내 시점간 표준편차 평균={np.mean(sd_within):.4f}")
print(f"  => 시점 간 변동 비율={np.mean(sd_within)/sd_total*100:.0f}% (높을수록 시점/관측 노이즈가 지배)")
# 동작 단위로 5시점 평균 예측했을 때
rep_prob = np.array([oof_subj[v].mean() for v in grp.values()])
rep_y = np.array([y[v[0]] for v in grp.values()])
print(f"  동작 단위 시점 평균 앙상블 AUC={roc_auc_score(rep_y, rep_prob):.4f} (개별 시점 AUC={a_subj:.4f})")
best = 0
for t in np.linspace(0.3, 0.7, 81):
    best = max(best, f1_score(rep_y, (rep_prob >= t).astype(int)))
print(f"  동작 단위 5시점 평균 시 최대 Defect F1={best*100:.2f}%")

print("=" * 70)
print("[6] F1 이론 상한 (AUC 기반, binormal 가정)")
from scipy.stats import norm
pi = y.mean()
for A in [a_subj, 0.65, 0.70, 0.80]:
    d = np.sqrt(2) * norm.ppf(A)
    bf = 0
    for t in np.linspace(-3, 5, 801):
        tpr = 1 - norm.cdf(t - d); fpr = 1 - norm.cdf(t)
        tp = pi * tpr; fp = (1 - pi) * fpr; fn = pi * (1 - tpr)
        bf = max(bf, 2 * tp / (2 * tp + fp + fn))
    print(f"  AUC={A:.3f} -> 달성 가능한 최대 Defect F1 ≈ {bf*100:.1f}% (양성률 {pi*100:.1f}%)")
print(f"  무작위 분류기 F1(전부 결함 예측)={2*pi/(1+pi)*100:.1f}%")

# -*- coding: utf-8 -*-
"""v7 Stage 1: v6 기준선 측정 잡음(Noise Floor) 및 평가 프로토콜 편향 진단"""
import sys, os, json
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from v7_ablation_core import *

D = load_dataset()
print(f"유효 {len(D['y_orig'])}개 / 고유 텐서 {int(D['uniq'].sum())}개 / 결함(원본) {int(D['y_orig'].sum())}", flush=True)
y_clean = clean_labels(D, "v6")
print(f"v6 정제: {int((D['y_orig']==1).sum()-(y_clean==1).sum())}건", flush=True)
ev = {'orig': D['y_orig'], 'clean': y_clean}

SEEDS = list(range(10))
print("\n[A] v6 기준선 (14D, v6 정제, CatBoost) - 분할 시드 10회 반복 / 모델 시드 42 고정", flush=True)
base = run_cv(D, V6_IDX, y_clean, SEEDS, cb_factory(), ev)
for k in ev:
    print(f"  평가라벨={k:5s} " + " | ".join(
        f"{m}={summarize(base[k], m)[0]:.2f}±{summarize(base[k], m)[1]:.2f}"
        for m in ["f1d", "prec", "spec", "auc", "abst", "f1d_u"]), flush=True)
print("  분할 시드별 clean f1d:", [round(r['f1d'], 1) for r in base['clean']], flush=True)

print("\n[B] 모델 시드 민감도 (분할 시드 42 고정)", flush=True)
vals = []
for ms in range(10):
    r = run_cv(D, V6_IDX, y_clean, [42], cb_factory(seed=ms), ev)
    vals.append((r['clean'][0]['f1d'], r['orig'][0]['f1d']))
v = np.array(vals)
print(f"  clean f1d {v[:,0].mean():.2f}±{v[:,0].std():.2f} (min {v[:,0].min():.1f}, max {v[:,0].max():.1f})", flush=True)
print(f"  orig  f1d {v[:,1].mean():.2f}±{v[:,1].std():.2f} (min {v[:,1].min():.1f}, max {v[:,1].max():.1f})", flush=True)

print("\n[C] 피처 상관 (|r|>=0.85, 19D)", flush=True)
F = D['feats']
C = np.corrcoef(F.T)
for i in range(19):
    for j in range(i+1, 19):
        if np.isfinite(C[i, j]) and abs(C[i, j]) >= 0.85:
            print(f"  {FEATURE_NAMES_19D[i]:28s} ~ {FEATURE_NAMES_19D[j]:28s} r={C[i,j]:+.3f}", flush=True)

print("\n[D] 피처 단독 AUC (원본 라벨, 방향 무관)", flush=True)
from sklearn.metrics import roc_auc_score
for i in range(19):
    a = roc_auc_score(D['y_orig'], F[:, i]) * 100
    print(f"  {i:2d} {FEATURE_NAMES_19D[i]:28s} AUC={max(a,100-a):.2f}", flush=True)

json.dump({'baseline_clean': base['clean'], 'baseline_orig': base['orig']},
          open(os.path.join(CURRENT_DIR, 'v7_stage1_noise_floor.json'), 'w'), ensure_ascii=False, indent=1, default=float)

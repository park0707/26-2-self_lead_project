# -*- coding: utf-8 -*-
"""sd1~sd3 모호성 제거 없는 전수(1,035) 피험자 격리 OOF 성능 (읽기 전용 집계)."""
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score

c = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrated_stiff_oof_cache_v2.npz"), allow_pickle=True)
v = c['views']
print("N =", len(v))
for n in (1, 2, 3):
    y = c[f'y_sd{n}'].astype(int); s = c[f'oof_sd{n}']
    print(f"\nsd{n}: 결함비율={y.mean():.3f}  자명기준선(항상 결함) F1={2*y.mean()/(1+y.mean())*100:.1f} Acc={max(y.mean(),1-y.mean())*100:.1f}")
    for name, t in (("thr0.5", 0.5), ("결함비율 매칭 임계", np.quantile(s, 1 - y.mean()))):
        p = (s >= t).astype(int)
        print(f"  [{name}] AUC {roc_auc_score(y,s)*100:.2f} | 결함F1 {f1_score(y,p)*100:.2f} | macroF1 {f1_score(y,p,average='macro')*100:.2f} | BalAcc {balanced_accuracy_score(y,p)*100:.2f} | Acc {accuracy_score(y,p)*100:.2f}")
    print("  시점별 AUC:", {k: round(roc_auc_score(y[v == k], s[v == k]) * 100, 1) for k in sorted(set(v))})

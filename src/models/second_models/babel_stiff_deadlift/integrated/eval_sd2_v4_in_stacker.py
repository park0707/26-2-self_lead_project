# -*- coding: utf-8 -*-
"""sd2 v4(MiniRocket 앙상블) OOF 생성 + v4.1 스태커 중첩 CV 재평가 (기존 sd2 대비)."""
import os, sys, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_sd2_v4 as E
import train_and_package_integrated_v4_1 as T
from stiff_meta_v41 import build_meta_features_v41, COMBO_ID

HERE = E.HERE
S, ids = E.load_all()
c = np.load(T.CACHE, allow_pickle=True)
sid = list(c['seq_ids']); pos = {s: i for i, s in enumerate(ids)}
S = S[[pos[s] for s in sid]]
f30 = np.load(E.os.path.join(E.BASE, "stiff_spine_ambiguity_levels.npz"), allow_pickle=True)
fp = {s: i for i, s in enumerate(f30['sequence_ids'])}
X30 = f30['features_30d'][[fp[s] for s in sid]].astype(np.float32)
y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
views, subj = c['views'], c['subjects']

# 비지도 로켓 변환 (라벨 미사용) 
mr = E.MiniRocketLite().fit(S); R = mr.transform(S)

# sd2 신규 OOF: 3 seeds 평균 (피험자 격리)
new = np.zeros(len(y2))
for seed in (0, 1, 2):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X30, y2, subj):
        cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4, auto_class_weights="Balanced", verbose=0, random_seed=seed).fit(X30[tr], y2[tr])
        lg = LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16, class_weight="balanced", verbose=-1, random_state=seed).fit(X30[tr], y2[tr])
        pg = 0.5 * cb.predict_proba(X30[te])[:, 1] + 0.5 * lg.predict_proba(X30[te])[:, 1]
        mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6
        rc = RidgeClassifierCV(alphas=np.logspace(0, 4, 9), class_weight="balanced").fit((R[tr] - mu) / sd, y2[tr])
        pr = 1 / (1 + np.exp(-rc.decision_function((R[te] - mu) / sd)))
        new[te] += (0.5 * pg + 0.5 * pr) / 3
np.savez(os.path.join(HERE, "sd2_v4_oof.npz"), seq_ids=np.array(sid), oof_sd2_v4=new)
print("sd2 AUC old %.2f  new %.2f" % (roc_auc_score(y2, c['oof_sd2']) * 100, roc_auc_score(y2, new) * 100))

y5 = np.array([COMBO_ID[(int(a), int(b), int(d))] for a, b, d in zip(y1, y2, y3)])
for name, s2 in [("v4.1 (기존 sd2)", c['oof_sd2']), ("v4.2 (sd2 MiniRocket 앙상블)", new)]:
    X = build_meta_features_v41(c['oof_sd1'], s2, c['oof_sd3'], views, X30)
    assert np.isfinite(X).all()
    r2, r3, pv = [], [], {v: [] for v in T.VIEW_ALIAS}
    for seed in (42, 7, 123):
        pred, acc, conf = T.nested_eval(X, y5, yc, subj, seed)
        r2.append(T.metrics(y5, yc, pred)); r3.append(T.metrics(y5, yc, pred, acc))
        for v in T.VIEW_ALIAS:
            ii = views == v; pv[v].append(T.metrics(y5[ii], yc[ii], pred[ii]))
    a = lambda L, k: (np.mean([d[k] for d in L]), np.std([d[k] for d in L]))
    print(f"\n[{name}] 2S EMR {a(r2,'emr')[0]:.2f}±{a(r2,'emr')[1]:.2f}  CleanF1 {a(r2,'clean_f1')[0]:.2f}±{a(r2,'clean_f1')[1]:.2f}  TP {a(r2,'clean_tp')[0]:.0f}"
          f" | 3S EMR {a(r3,'emr')[0]:.2f}  CleanF1 {a(r3,'clean_f1')[0]:.2f}  cov {a(r3,'coverage')[0]:.1f}")
    print("  per-view:", {v: (round(a(pv[v], 'emr')[0], 2), round(a(pv[v], 'clean_f1')[0], 2)) for v in sorted(pv)})

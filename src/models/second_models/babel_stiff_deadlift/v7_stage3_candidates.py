# -*- coding: utf-8 -*-
"""v7 Stage 3: 사전 지정(Pre-specified) 후보 소수 정밀 검증 - 신규 모델 시드 10개 + 피험자 단위 + 대체 분할(5-Fold) 교차 확인"""
import sys, os, json, time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from v7_ablation_core import *
from sklearn.metrics import roc_auc_score

SEEDS = list(range(100, 110))     # Stage 2(0~2)와 겹치지 않는 새 시드
D = load_dataset()
ycl = {r: clean_labels(D, r) for r in ["v6", "strict"]}
ev = {'orig': D['y_orig'], 'clean': ycl['v6']}
N = FEATURE_NAMES_19D
DROP_REDUNDANT = [6, 7]                         # r>=0.9 중복 쌍 (rel_bottom_knee_delta, norm_knee_disp_y)
V7A = [j for j in V6_IDX if j not in DROP_REDUNDANT]

cands = [   # (이름, 컬럼, 정제규칙, 모델팩토리, 구분)
    ("v6 (기준)",                        V6_IDX, "v6",     lambda s: cb_factory(seed=s), "기준"),
    ("v7a: 중복2개 제거(12D)",            V7A,    "v6",     lambda s: cb_factory(seed=s), "사전지정"),
    ("v7b: v7a + 정제 strict",           V7A,    "strict", lambda s: cb_factory(seed=s), "사전지정"),
    ("v7c: v7a + depth5",                V7A,    "v6",     lambda s: cb_factory(depth=5, seed=s), "사전지정"),
    ("v7d: v7a + strict + depth5",       V7A,    "strict", lambda s: cb_factory(depth=5, seed=s), "사전지정"),
    ("탐색: v7a - kadr_max",              [j for j in V7A if j != 10], "v6", lambda s: cb_factory(seed=s), "탐색(편향주의)"),
    ("탐색: v7a - address_knee_angle",    [j for j in V7A if j != 8],  "v6", lambda s: cb_factory(seed=s), "탐색(편향주의)"),
    ("탐색: v7a - bottom_knee_delta_mean", [j for j in V7A if j != 1], "v6", lambda s: cb_factory(seed=s), "탐색(편향주의)"),
]

def loso_oof(cols, y_train, mf, seed):
    X = D['feats'][:, cols]; oof = np.zeros(len(X))
    for s in np.unique(D['subs']):
        va = D['subs'] == s
        m = mf(seed)(); m.fit(X[~va], y_train[~va]); oof[va] = m.predict_proba(X[va])[:, 1]
    return oof

def subj_auc(oof, y):
    out = {}
    for s in np.unique(D['subs']):
        m = D['subs'] == s
        if len(np.unique(y[m])) > 1:
            out[int(s)] = roc_auc_score(y[m], oof[m]) * 100
    return out

results, base = {}, None
t0 = time.time()
print(f"후보 {len(cands)}개 / 새 모델시드 {SEEDS[0]}~{SEEDS[-1]} (10개) / LOSO 9-Fold", flush=True)
for name, cols, rule, mf, grp in cands:
    per_seed = {k: [] for k in ev}; oofs = []
    for sd in SEEDS:
        oof = loso_oof(cols, ycl[rule], mf, sd); oofs.append(oof)
        for k, yl in ev.items():
            per_seed[k].append(three_state(oof, yl, D['uniq']))
    oof_mean = np.mean(oofs, axis=0)                 # 시드 앙상블 (배포 번들의 Fold 평균과 동일한 개념)
    ens = {k: three_state(oof_mean, yl, D['uniq']) for k, yl in ev.items()}
    sa = {k: subj_auc(oof_mean, yl) for k, yl in ev.items()}
    results[name] = dict(group=grp, cols=cols, rule=rule, per_seed=per_seed, ens=ens, subj_auc=sa)
    if base is None:
        base = results[name]
    print(f"\n■ [{grp}] {name} (d={len(cols)}) {time.time()-t0:.0f}s", flush=True)
    for k in ['orig', 'clean']:
        s = []
        for m in ['f1d', 'prec', 'spec', 'auc']:
            mean, std = summarize(per_seed[k], m)
            dm, ds, win = paired(base['per_seed'][k], per_seed[k], m)
            s.append(f"{m}={mean:5.1f}±{std:3.1f}(Δ{dm:+.1f},승{win*100:3.0f}%)")
        print(f"   {k:5s} " + " ".join(s) + f" | 시드앙상블 f1d={ens[k]['f1d']:.1f} auc={ens[k]['auc']:.1f} abst={ens[k]['abst']:.1f}", flush=True)
    wins = sum(1 for s_, a in sa['orig'].items() if a > base['subj_auc']['orig'].get(s_, -1))
    print(f"   피험자별 AUC(원본라벨) 기준 대비 개선 피험자: {wins}/{len(sa['orig'])}", flush=True)
    json.dump(results, open(os.path.join(CURRENT_DIR, 'v7_stage3_candidates.json'), 'w'), ensure_ascii=False, default=float)

# 대체 분할(5-Fold, 분할 시드 10회) 교차 확인: LOSO와 다른 분할에서도 같은 방향인지
print("\n[대체 분할 교차확인] 반복 StratifiedGroupKFold(분할시드 0~9, 모델시드 42)", flush=True)
cv_base = None
for name, cols, rule, mf, grp in cands[:5]:
    r = run_cv(D, cols, ycl[rule], list(range(10)), mf(42), ev)
    if cv_base is None:
        cv_base = r
    s = []
    for k in ['orig', 'clean']:
        for m in ['f1d', 'auc']:
            dm, ds, win = paired(cv_base[k], r[k], m)
            s.append(f"{k[0]}{m}={summarize(r[k], m)[0]:5.1f}(Δ{dm:+.1f},승{win*100:3.0f}%)")
    print(f"  {name:30s} " + " ".join(s) + f" spec={summarize(r['orig'],'spec')[0]:.1f}", flush=True)
print("완료", flush=True)

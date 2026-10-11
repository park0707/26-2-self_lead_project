# -*- coding: utf-8 -*-
"""v7 Stage 2: v6 구성요소 전수 절제(Ablation) - LOSO 9-Fold, 모델 시드 반복, 이중 채점(원본/정제 라벨)"""
import sys, os, json, time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from v7_ablation_core import *

SEEDS = [0, 1, 2]
D = load_dataset()
ycl = {r: clean_labels(D, r) for r in ["none", "v6", "strict", "loose", "excess"]}
ev = {'orig': D['y_orig'], 'clean': ycl['v6']}   # 채점 라벨은 v6 정제 라벨로 고정(정제 규칙 간 공정 비교)
N = FEATURE_NAMES_19D

def nm(idx): return ",".join(N[i] for i in idx)

configs = []  # (name, cols, train_label_rule, model_fn_of_seed, group)
def add(name, cols, rule="v6", mf=None, group="misc"):
    configs.append((name, list(cols), rule, mf or (lambda s: cb_factory(seed=s)), group))

add("BASE v6 (14D/v6정제/CB d4)", V6_IDX, group="base")
for i in V6_IDX:                       # 1) Leave-One-Feature-Out
    add(f"-{N[i]}", [j for j in V6_IDX if j != i], group="LOFO")
for i in [2, 5, 9, 12, 18]:            # 2) v6 선별에서 제외된 피처 재투입
    add(f"+{N[i]}", V6_IDX + [i], group="ADD")
groups = {                              # 3) 피처 그룹 제거
    "-신규v6(kadr,excess,zrom)": [10, 11, 13, 14, 15],
    "-kadr(2)": [10, 11], "-excess(2)": [13, 14], "-context(sd3,coronal)": [16, 17],
    "-정규화중복(rel,norm_y)": [6, 7], "-address_knee_angle": [8],
    "-무릎각도계(0,1,6,8)": [0, 1, 6, 8], "-수직변위계(3,7)": [3, 7],
}
for k, rm in groups.items():
    add(k, [j for j in V6_IDX if j not in rm], group="GROUP")
for r in ["none", "strict", "loose", "excess"]:   # 4) 라벨 정제 규칙
    add(f"정제={r}", V6_IDX, rule=r, group="CLEAN")
for d in [2, 3, 5, 6]:                  # 5) 모델 복잡도
    add(f"CB depth={d}", V6_IDX, mf=(lambda s, d=d: cb_factory(depth=d, seed=s)), group="MODEL")
for it, lr in [(100, 0.04), (400, 0.04), (200, 0.02)]:
    add(f"CB iters={it},lr={lr}", V6_IDX, mf=(lambda s, it=it, lr=lr: cb_factory(iters=it, lr=lr, seed=s)), group="MODEL")
for l2 in [1, 15, 30]:
    add(f"CB l2={l2}", V6_IDX, mf=(lambda s, l2=l2: cb_factory(l2=l2, seed=s)), group="MODEL")
add("CB class_weight=None", V6_IDX, mf=(lambda s: cb_factory(cw=None, seed=s)), group="MODEL")
add("19D 전체", list(range(19)), group="ADD")

out, base = {}, None
t0 = time.time()
print(f"설정 {len(configs)}개 / 모델시드 {SEEDS} / LOSO 9-Fold", flush=True)
for name, cols, rule, mf, grp in configs:
    r = run_loso(D, cols, ycl[rule], SEEDS, mf, ev)
    out[name] = {'group': grp, 'cols': cols, 'rule': rule, 'res': r}
    if base is None:
        base = r
    row = []
    for k in ['orig', 'clean']:
        for m in ['f1d', 'auc']:
            mean, std = summarize(r[k], m)
            dm, ds, win = paired(base[k], r[k], m)
            row.append(f"{k[0]}{m}={mean:5.1f}({dm:+.1f})")
    pm = summarize(r['orig'], 'prec')[0]
    sp = summarize(r['orig'], 'spec')[0]
    print(f"[{grp:5s}] {name:34s} d={len(cols):2d} | " + " ".join(row) + f" | prec={pm:4.1f} spec={sp:4.1f} | {time.time()-t0:5.0f}s", flush=True)
    json.dump(out, open(os.path.join(CURRENT_DIR, 'v7_stage2_ablation.json'), 'w'), ensure_ascii=False, default=float)
print("완료", flush=True)

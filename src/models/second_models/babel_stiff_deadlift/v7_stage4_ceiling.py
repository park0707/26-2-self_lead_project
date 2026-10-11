# -*- coding: utf-8 -*-
"""v7 Stage 4: '더 이상의 성능 향상은 어려운가' 전수 진단 (천장 분석)
 E1 라벨-물리 불일치 상한 / E2 비무릎 피처만으로 라벨 예측 / E3 Key별 예측력 /
 E4 학습곡선 / E5 모델·피처 고용량 탐색 / E6 통계적 검정력(피험자 부트스트랩)"""
import sys, os, json, time, re, warnings
import numpy as np
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from v7_ablation_core import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

D = load_dataset()
N = FEATURE_NAMES_19D
y_o = D['y_orig']; y_c = clean_labels(D, "v6"); F = D['feats']; subs = D['subs']; uniq = D['uniq']
V7 = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15, 16, 17]
KNEE = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15]      # 무릎/하지 관련
NONKNEE = [16, 17, 18]                           # sd3 확률, 체간 관상/시상 ROM
OUT = {}
t0 = time.time()

def auc(p, y): return roc_auc_score(y, p) * 100

def loso(X, y_train, mf, seeds=(0,)):
    oofs = []
    for sd in seeds:
        oof = np.zeros(len(X))
        for s in np.unique(subs):
            va = subs == s
            m = mf(sd); m.fit(X[~va], y_train[~va]); oof[va] = m.predict_proba(X[va])[:, 1]
        oofs.append(oof)
    return np.mean(oofs, axis=0)

cbf = lambda sd: cb_factory(seed=sd)()

# ---------------------------------------------------------------- E1
print("=" * 90 + "\n[E1] 라벨-물리 불일치 상한 (고유 텐서 기준)\n" + "=" * 90, flush=True)
bl = F[:, 1]; ex = F[:, 13]; ang = F[:, 0]
halo = np.array([any(k in s for k in HALO_KEYS) for s in D['sids']])
straight = (bl <= 9.0) & (ex <= 1.0)             # 물리적으로 무릎 고정(굴곡 변화 작음)
m = uniq
pos, neg = (y_o == 1) & m, (y_o == 0) & m
p_str_pos, p_str_neg = straight[pos].mean(), straight[neg].mean()
print(f"  결함(원본) 중 무릎이 물리적으로 고정된 비율: {p_str_pos*100:.1f}% (N={pos.sum()})")
print(f"  정상(원본) 중 무릎이 물리적으로 고정된 비율: {p_str_neg*100:.1f}% (N={neg.sum()})")
ceil = 0.5 + 0.5 * (1 - p_str_pos)
print(f"  → '무릎 물리'만 보는 완벽한 검출기의 AUC 상한 ≈ 0.5+0.5×(1−{p_str_pos:.2f}) = {ceil*100:.1f}% (정상에도 굴곡이 있으면 더 낮음)")
print(f"  현재 v7 원본라벨 AUC ≈ 57%, 정제라벨 AUC ≈ 82% (정제 라벨은 같은 피처로 라벨을 바꿔 순환 포함)")
OUT['E1'] = dict(p_straight_pos=p_str_pos, p_straight_neg=p_str_neg, auc_ceiling=ceil * 100)

# ---------------------------------------------------------------- E2
print("\n" + "=" * 90 + "\n[E2] 무릎 피처 vs 비무릎(체간·바벨) 피처의 원본 라벨 예측력 (LOSO, 시드3 평균, 원본라벨 학습)\n" + "=" * 90, flush=True)
res = {}
for name, cols in [("무릎/하지 피처 10D", KNEE), ("비무릎 3D(sd3,체간ROM)", NONKNEE), ("sd3 확률 단독", [16]),
                   ("체간 관상 ROM 단독", [17]), ("v7 12D 전체", V7), ("19D 전체", list(range(19)))]:
    oof = loso(F[:, cols], y_o, cbf, seeds=(0, 1, 2))
    res[name] = auc(oof, y_o); print(f"  {name:26s} AUC(원본)={res[name]:.1f}", flush=True)
OUT['E2'] = res

# ---------------------------------------------------------------- E3
print("\n" + "=" * 90 + "\n[E3] v7(12D, 정제학습) OOF의 Key별 판별력 (원본 정상 대비)\n" + "=" * 90, flush=True)
oof7 = loso(F[:, V7], y_c, cbf, seeds=tuple(range(5)))
key = np.array([re.search(r'Key(\d+)', s).group(1) if re.search(r'Key(\d+)', s) else '?' for s in D['sids']])
nm = {'209': '정상', '210': '바벨이격', '211': '우기울', '212': '좌기울', '213': '우기울+이격*', '214': '좌기울+이격*', '215': '과도하게뒤로*', '216': '어깨말림'}
normal_mask = (y_o == 0)
rows = []
for k in sorted(set(key)):
    mk = key == k
    lbl = int(y_o[mk].max())
    a = auc(np.concatenate([oof7[mk], oof7[normal_mask]]), np.concatenate([np.ones(mk.sum()), np.zeros(normal_mask.sum())])) if lbl == 1 else float('nan')
    rows.append((k, nm.get(k, '?'), lbl, oof7[mk].mean(), a))
    print(f"  Key{k} {nm.get(k,'?'):12s} 원본라벨={lbl} 평균확률={oof7[mk].mean():.3f}  vs정상 AUC={a:.1f}" if lbl == 1 else
          f"  Key{k} {nm.get(k,'?'):12s} 원본라벨={lbl} 평균확률={oof7[mk].mean():.3f}", flush=True)
print("  (* = 라벨상 '무릎 구부린채 고정=False'로 자동 부여되는 복합 오류 키)")
OUT['E3'] = [dict(key=r[0], name=r[1], label=r[2], mean_p=r[3], auc=r[4]) for r in rows]

# ---------------------------------------------------------------- E4
print("\n" + "=" * 90 + "\n[E4] 학습곡선: 학습 피험자 수 k에 따른 AUC (v7, 정제라벨 학습, 원본/정제 채점, 반복 3)\n" + "=" * 90, flush=True)
rng = np.random.RandomState(0); X7 = F[:, V7]; lc = {}
for k in [2, 4, 6, 8]:
    oo = []
    for rep in range(3):
        oof = np.zeros(len(X7))
        for s in np.unique(subs):
            others = [u for u in np.unique(subs) if u != s]
            tr_s = rng.choice(others, size=k, replace=False)
            tr = np.isin(subs, tr_s); va = subs == s
            if len(np.unique(y_c[tr])) < 2: oof[va] = 0.5; continue
            mm = cb_factory(seed=rep)(); mm.fit(X7[tr], y_c[tr]); oof[va] = mm.predict_proba(X7[va])[:, 1]
        oo.append((auc(oof, y_o), auc(oof, y_c)))
    a = np.array(oo); lc[k] = (a[:, 0].mean(), a[:, 1].mean())
    print(f"  학습 피험자 {k}명: AUC 원본={a[:,0].mean():.1f}±{a[:,0].std():.1f} / 정제={a[:,1].mean():.1f}±{a[:,1].std():.1f}", flush=True)
full = (auc(oof7, y_o), auc(oof7, y_c)); print(f"  학습 피험자 8명(전체 LOSO, 시드5평균): AUC 원본={full[0]:.1f} / 정제={full[1]:.1f}")
OUT['E4'] = {str(k): v for k, v in lc.items()}; OUT['E4']['8_full'] = full

# ---------------------------------------------------------------- E5
print("\n" + "=" * 90 + "\n[E5] 모델·피처 고용량 탐색 (LOSO, 정제라벨 학습 / 원본라벨 학습 두 가지, AUC·F1d)\n" + "=" * 90, flush=True)
cache = os.path.join(CURRENT_DIR, 'v7_rawstats_cache.npz')
if os.path.exists(cache):
    RAW = np.load(cache)['raw']
else:
    from src.models.first_model.dataset import Stage1DatasetLoader
    splits = Stage1DatasetLoader.load_all_dual_splits(); seen = {}
    for sd in splits.values():
        mm = sd['y_action'] == 2
        for sid, x in zip(sd['sequence_ids'][mm], sd['X_Format_B'][mm]): seen[str(sid)] = x
    RAW = np.array([np.concatenate([seen[s].mean(0), seen[s].std(0), seen[s].min(0), seen[s].max(0), seen[s].max(0) - seen[s].min(0)]) for s in D['sids']], dtype=np.float32)
    RAW = np.nan_to_num(RAW); np.savez(cache, raw=RAW)
print(f"  원시 텐서 통계 피처 {RAW.shape[1]}D 준비", flush=True)

models = {
    "CatBoost": cbf,
    "LightGBM": lambda sd: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, max_depth=3, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, class_weight='balanced', random_state=sd, verbose=-1),
    "RandomForest": lambda sd: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight='balanced', random_state=sd, n_jobs=-1),
    "ExtraTrees": lambda sd: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight='balanced', random_state=sd, n_jobs=-1),
    "LogReg(L2)": lambda sd: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)),
    "kNN(k=15)": lambda sd: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
}
sets = {"v7 12D": F[:, V7], "19D": F, "원시통계 %dD" % RAW.shape[1]: RAW, "v7+원시통계": np.hstack([F[:, V7], RAW])}
e5 = []
for sname, X in sets.items():
    for mname, mf in models.items():
        if X.shape[1] > 100 and mname == "kNN(k=15)": continue
        seeds = (0, 1, 2) if mname in ("CatBoost", "LightGBM", "RandomForest", "ExtraTrees") else (0,)
        if X.shape[1] > 100 and mname == "CatBoost": seeds = (0,)
        line = []
        for tag, ytr in (("정제학습", y_c), ("원본학습", y_o)):
            oof = loso(X, ytr, mf, seeds)
            ts_o, ts_c = three_state(oof, y_o, uniq), three_state(oof, y_c, uniq)
            line.append((tag, auc(oof, y_o), auc(oof, y_c), ts_o['f1d'], ts_c['f1d'], ts_o['spec']))
            e5.append(dict(set=sname, model=mname, train=tag, auc_o=line[-1][1], auc_c=line[-1][2], f1_o=line[-1][3], f1_c=line[-1][4], spec_o=line[-1][5]))
        print(f"  [{sname:14s}] {mname:13s} " + " | ".join(f"{t}: AUCo={a:.1f} AUCc={b:.1f} F1o={c:.1f} F1c={d:.1f} spec={e:.0f}" for t, a, b, c, d, e in line) + f" ({time.time()-t0:.0f}s)", flush=True)
OUT['E5'] = e5

# ---------------------------------------------------------------- E6
print("\n" + "=" * 90 + "\n[E6] 통계적 검정력: 피험자 단위 부트스트랩 (v7 OOF, 시드5평균, 2000회)\n" + "=" * 90, flush=True)
oof6 = loso(F[:, V6_IDX], y_c, cbf, seeds=tuple(range(5)))
us = np.unique(subs); rng = np.random.RandomState(1); a7, a6 = [], []
for _ in range(2000):
    pick = rng.choice(us, size=len(us), replace=True)
    idx = np.concatenate([np.where(subs == s)[0] for s in pick])
    if len(np.unique(y_o[idx])) < 2: continue
    a7.append(auc(oof7[idx], y_o[idx])); a6.append(auc(oof6[idx], y_o[idx]))
a7, a6 = np.array(a7), np.array(a6); dlt = a7 - a6
ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
print(f"  v7 AUC(원본) = {auc(oof7,y_o):.1f}  95% CI [{ci(a7)[0]:.1f}, {ci(a7)[1]:.1f}]  (폭 {ci(a7)[1]-ci(a7)[0]:.1f}%p)")
print(f"  v6 AUC(원본) = {auc(oof6,y_o):.1f}  95% CI [{ci(a6)[0]:.1f}, {ci(a6)[1]:.1f}]")
print(f"  Δ(v7−v6) = {np.mean(dlt):+.2f}  95% CI [{ci(dlt)[0]:+.2f}, {ci(dlt)[1]:+.2f}]  P(Δ>0)={np.mean(dlt>0)*100:.0f}%")
sw = ci(a7)[1] - ci(a7)[0]
print(f"  → 피험자 9명 부트스트랩 CI 폭이 약 {sw:.0f}%p: 1~2%p 수준 개선은 통계적으로 구분 불가(검출 가능 최소 개선폭 ≈ {sw/2:.0f}%p 이상)")
OUT['E6'] = dict(auc_v7=auc(oof7, y_o), ci_v7=ci(a7), auc_v6=auc(oof6, y_o), delta=float(np.mean(dlt)), ci_delta=ci(dlt), p_pos=float(np.mean(dlt > 0)))
json.dump(OUT, open(os.path.join(CURRENT_DIR, 'v7_stage4_ceiling.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(f"\n완료 {time.time()-t0:.0f}s", flush=True)

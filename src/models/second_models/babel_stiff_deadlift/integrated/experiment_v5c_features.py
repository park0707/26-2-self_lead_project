"""
experiment_v5c_features.py
(1) v4 방식 Clean F1의 정직한 값(LOSO 임계치 선택) 측정
(2) 5-class LR 스태커에 sd2 30D 원본 생체역학 피처(순수 비전 유래, 라벨 아님) 추가 효과
    - 모든 w0 선택은 중첩 CV. 단일 시점 / 다시점 평균 피처 비교.
"""
import os, re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

os.environ["OMP_NUM_THREADS"] = "1"
D = r"D:\src\models\second_models\babel_stiff_deadlift"
c = np.load(D + r"\integrated\integrated_stiff_oof_cache_v2.npz", allow_pickle=True)
s1, s2, s3 = c['oof_sd1'], c['oof_sd2'], c['oof_sd3']
y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
views, subj, sid = c['views'], c['subjects'], list(c['seq_ids'])
N = len(s1); p2 = np.maximum(s2, s1)
sd2 = np.load(D + r"\stiff_spine_ambiguity_levels.npz", allow_pickle=True)
mp = {s: i for i, s in enumerate(sd2['sequence_ids'])}
B = sd2['features_30d'][[mp[s] for s in sid]].astype(np.float32)   # 순수 비전 피처
combos = [(0,0,0),(0,0,1),(0,1,0),(1,1,0),(1,1,1)]
cid = {k: i for i, k in enumerate(combos)}
y5 = np.array([cid[(int(a), int(b), int(d))] for a, b, d in zip(y1, y2, y3)])
key = np.array([re.sub(r'_view\d$', '', s) for s in sid])
uk = {k: i for i, k in enumerate(np.unique(key))}; kidx = np.array([uk[k] for k in key])

def ent(p):
    e = 1e-7; return -(p*np.log2(p+e)+(1-p)*np.log2(1-p+e))
def kmean(M):
    out = np.zeros_like(M)
    for k in range(len(uk)):
        ii = np.where(kidx == k)[0]; out[ii] = M[ii].mean(0)
    return out
def build(multi, bio):
    P_ = np.column_stack([s1, p2, s3])
    vm = np.zeros((N, 5))
    for i, v in enumerate(views): vm[i, int(str(v)[-1])-1] = 1
    f = [P_, ent(P_), np.abs(P_-.5)*2, (1-P_).prod(1, keepdims=True), vm]
    if multi: f += [kmean(P_), ent(kmean(P_))]
    if bio:
        f.append(B)
        if multi: f.append(kmean(B))
    return np.column_stack(f).astype(np.float32)

def fit_prob(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=4000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))
W0 = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
def dec(pr, w): q = pr.copy(); q[:, 0] *= w; return q.argmax(1)
def sc_(y, p, ycl): return (p == y).mean()*100 + f1_score(ycl, (p == 0).astype(int), zero_division=0)*100

def nested(X, seed, C):
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(N, int)
    for tr, te in sg.split(X, y5, groups=subj):
        ig = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        oo = np.zeros((len(tr), 5))
        for a, b in ig.split(X[tr], y5[tr], groups=subj[tr]):
            oo[b] = fit_prob(X[tr][a], y5[tr][a], X[tr][b], C)
        w = max(W0, key=lambda w: sc_(y5[tr], dec(oo, w), yc[tr]))
        pred[te] = dec(fit_prob(X[tr], y5[tr], X[te], C), w)
    return pred

print("== (1) v4 방식(독립 이진+시점별 임계치) 정직한 값: LOSO ==")
grid = [np.linspace(0.45, 0.65, 9), np.linspace(0.50, 0.70, 9), np.linspace(0.45, 0.65, 9)]
def sc3(idx, t):
    a = [(s1[idx] >= t[0]).astype(int), (p2[idx] >= t[1]).astype(int), (s3[idx] >= t[2]).astype(int)]
    e = ((a[0]==y1[idx]) & (a[1]==y2[idx]) & (a[2]==y3[idx])).mean()*100
    f = f1_score(yc[idx], ((a[0]==0)&(a[1]==0)&(a[2]==0)).astype(int), zero_division=0)*100
    return e + f
def pick(idx):
    best, bt = -1, None
    for t1 in grid[0]:
        for t2 in grid[1]:
            for t3 in grid[2]:
                s = sc3(idx, (t1, t2, t3))
                if s > best: best, bt = s, (t1, t2, t3)
    return bt
pc = np.zeros(N, int); pe = np.zeros(N, int)
for s in np.unique(subj):
    te = np.where(subj == s)[0]; tr = np.where(subj != s)[0]
    for v in np.unique(views):
        trv = tr[views[tr] == v]; tev = te[views[te] == v]
        if len(tev) == 0: continue
        t = pick(trv)
        a = [(s1[tev] >= t[0]), (p2[tev] >= t[1]), (s3[tev] >= t[2])]
        pc[tev] = ((~a[0]) & (~a[1]) & (~a[2])).astype(int)
        pe[tev] = ((a[0]==y1[tev]) & (a[1]==y2[tev]) & (a[2]==y3[tev])).astype(int)
print(f"  LOSO 시점별 임계치: EMR={pe.mean()*100:.2f}%  CleanF1={f1_score(yc, pc, zero_division=0)*100:.2f}%")

print("\n== (2) 5-class LR 스태커 피처 비교 (중첩 CV, 3 seeds, 2-State) ==")
for multi in (False, True):
    for bio in (False, True):
        X = build(multi, bio)
        for C in (0.05, 0.5):
            r = []
            for sd in (42, 7, 123):
                p = nested(X, sd, C)
                r.append(((p == y5).mean()*100, f1_score(yc, (p == 0).astype(int), zero_division=0)*100))
            r = np.array(r)
            print(f"  다시점={int(multi)} sd2생체피처={int(bio)} C={C:<4} EMR={r[:,0].mean():5.2f}±{r[:,0].std():.2f}%  CleanF1={r[:,1].mean():5.2f}±{r[:,1].std():.2f}%  (피처 {X.shape[1]}D)")

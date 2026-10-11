"""
experiment_v5b_nested.py
5-class LR 스태커 + Clean 클래스 prior-shift(w0) + 신뢰도 기반 선택적 보류(커버리지 90%)
모든 하이퍼파라미터(w0, 보류 임계)는 바깥 fold의 학습 데이터 안쪽 CV로만 선택(중첩 CV, 누수 없음).
"""
import os, re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

os.environ["OMP_NUM_THREADS"] = "1"
P = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"
c = np.load(P, allow_pickle=True)
s1, s2, s3 = c['oof_sd1'], c['oof_sd2'], c['oof_sd3']
y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
views, subj, sid = c['views'], c['subjects'], list(c['seq_ids'])
N = len(s1); p2 = np.maximum(s2, s1)
combos = [(0,0,0),(0,0,1),(0,1,0),(1,1,0),(1,1,1)]
cid = {k: i for i, k in enumerate(combos)}
y5 = np.array([cid[(int(a), int(b), int(d))] for a, b, d in zip(y1, y2, y3)])
key = np.array([re.sub(r'_view\d$', '', s) for s in sid])
uk = {k: i for i, k in enumerate(np.unique(key))}; kidx = np.array([uk[k] for k in key])

def ent(p):
    e = 1e-7; return -(p*np.log2(p+e)+(1-p)*np.log2(1-p+e))
def build(single):
    P_ = np.column_stack([s1, p2, s3])
    f = [P_, ent(P_), np.abs(P_-.5)*2, (1-P_).prod(1, keepdims=True)]
    vm = np.zeros((N, 5))
    for i, v in enumerate(views): vm[i, int(str(v)[-1])-1] = 1
    f.append(vm)
    if not single:
        m = np.zeros_like(P_); sd = np.zeros_like(P_)
        for k in range(len(uk)):
            ii = np.where(kidx == k)[0]; m[ii] = P_[ii].mean(0); sd[ii] = P_[ii].std(0)
        f += [m, sd, ent(m)]
    return np.column_stack(f).astype(np.float32)

def fit_prob(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))

W0 = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
def decide(prob, w0):
    q = prob.copy(); q[:, 0] *= w0
    return q.argmax(1)
def score(y, pred, yclean):
    return (pred == y).mean()*100 + f1_score(yclean, (pred == 0).astype(int), zero_division=0)*100

def inner_oof(X, y, grp, seed):
    sg = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 5))
    for a, b in sg.split(X, y, groups=grp):
        oof[b] = fit_prob(X[a], y[a], X[b])
    return oof

def run(single, seed, alpha):
    X = build(single)
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.full(N, -1); accept = np.zeros(N, bool); chosen = []
    for tr, te in sg.split(X, y5, groups=subj):
        oo = inner_oof(X[tr], y5[tr], subj[tr], seed)
        best = max(W0, key=lambda w: score(y5[tr], decide(oo, w), yc[tr]) )
        chosen.append(best)
        pr = fit_prob(X[tr], y5[tr], X[te])
        q = pr.copy(); q[:, 0] *= best; q /= q.sum(1, keepdims=True)
        pred[te] = q.argmax(1)
        # 선택적 보류: 안쪽 OOF 신뢰도의 하위 (1-cov) 분위수를 임계로
        qo = oo.copy(); qo[:, 0] *= best; qo /= qo.sum(1, keepdims=True)
        thr = np.quantile(qo.max(1), alpha)
        accept[te] = q.max(1) >= thr
    return pred, accept, chosen

def rep(tag, pred, accept=None):
    m = np.ones(N, bool) if accept is None else accept
    emr = (pred[m] == y5[m]).mean()*100
    cf = f1_score(yc[m], (pred[m] == 0).astype(int), zero_division=0)*100
    tp = int(((pred[m] == 0) & (yc[m] == 1)).sum())
    return emr, cf, m.mean()*100, tp

for single, name in [(True, "단일 시점 입력"), (False, "다시점 융합 입력(다중 카메라 필요)")]:
    print(f"\n===== {name} : 중첩 CV (3 seeds) =====")
    rows = {"2-State": [], "3-State(보류10%)": []}
    for seed in (42, 7, 123):
        pred, acc, ch = run(single, seed, alpha=0.10)
        rows["2-State"].append(rep("", pred))
        rows["3-State(보류10%)"].append(rep("", pred, acc))
        print(f"  seed {seed}: 선택된 w0 = {ch}")
    for k, v in rows.items():
        a = np.array(v)
        print(f"  {k:16s} EMR={a[:,0].mean():.2f}±{a[:,0].std():.2f}%  CleanF1={a[:,1].mean():.2f}±{a[:,1].std():.2f}%  "
              f"커버리지={a[:,2].mean():.1f}%  정상TP={a[:,3].mean():.0f}/130")
    pred, acc, _ = run(single, 42, 0.10)
    print("  [시점별(seed 42, 2-State)]")
    for v in sorted(np.unique(views)):
        ii = views == v
        print(f"    {v}: EMR={(pred[ii]==y5[ii]).mean()*100:.2f}% CleanF1={f1_score(yc[ii], (pred[ii]==0).astype(int), zero_division=0)*100:.2f}%")

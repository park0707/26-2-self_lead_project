"""
experiment_v5_powerset.py
5-class label-powerset 스태커 (유효 조합만 디코딩) + 동작(Key) 단위 다시점 융합 검증.
평가는 피험자 격리 StratifiedGroupKFold(정직한 평가, 임계치 사후 튜닝 없음).
"""
import os, re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier

os.environ["OMP_NUM_THREADS"] = "1"
P = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"
c = np.load(P, allow_pickle=True)
s1, s2, s3 = c['oof_sd1'], c['oof_sd2'], c['oof_sd3']
y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
views, subj, sid = c['views'], c['subjects'], list(c['seq_ids'])
N = len(s1)
p2 = np.maximum(s2, s1)

combos = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 1)]
cid = {k: i for i, k in enumerate(combos)}
y5 = np.array([cid[(int(a), int(b), int(d))] for a, b, d in zip(y1, y2, y3)])

# 동작 키 (view 접미사 제거)
key = np.array([re.sub(r'_view\d$', '', s) for s in sid])
uk = {k: i for i, k in enumerate(np.unique(key))}
kidx = np.array([uk[k] for k in key])
print("고유 동작 수:", len(uk), "| 동작당 시점 수 분포:",
      np.unique(np.bincount(kidx), return_counts=True))

def ent(p):
    e = 1e-7
    return -(p * np.log2(p + e) + (1 - p) * np.log2(1 - p + e))

def build(single=True):
    P_ = np.column_stack([s1, p2, s3])
    f = [P_, ent(P_), np.abs(P_ - .5) * 2, (1 - P_).prod(1, keepdims=True)]
    vm = np.zeros((N, 5)); 
    for i, v in enumerate(views): vm[i, int(str(v)[-1]) - 1] = 1
    f.append(vm)
    if not single:
        m = np.zeros_like(P_); sd = np.zeros_like(P_)
        for k in range(len(uk)):
            ii = np.where(kidx == k)[0]
            m[ii] = P_[ii].mean(0); sd[ii] = P_[ii].std(0)
        f += [m, sd, ent(m)]
    return np.column_stack(f).astype(np.float32)

def report(name, pred):
    emr = (pred == y5).mean() * 100
    cf = f1_score(yc, (pred == 0).astype(int), zero_division=0) * 100
    rec = [((pred == k) & (y5 == k)).sum() / max(1, (y5 == k).sum()) * 100 for k in range(5)]
    print(f"{name:42s} EMR={emr:5.2f}%  CleanF1={cf:5.2f}%  클래스별 recall={['%.0f' % r for r in rec]}")
    return emr, cf

print("\n[베이스라인]")
report("최빈 클래스 고정 예측", np.full(N, 2))
base = np.column_stack([(s1 >= .5), (p2 >= .5), (s3 >= .5)]).astype(int)
def to5(b):
    out = np.zeros(len(b), dtype=int)
    for i, r in enumerate(b):
        t = tuple(int(x) for x in r)
        out[i] = cid.get(t, 2)  # 무효 조합은 최빈으로 대체
    return out
report("v2식 독립 이진 + 0.5 (무효조합->최빈)", to5(base))

def cv_eval(X, model_fn, tag, seeds=(42, 7, 123)):
    emrs, cfs, last = [], [], None
    for sd in seeds:
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
        pred = np.zeros(N, dtype=int); prob = np.zeros((N, 5))
        for tr, te in sg.split(X, y5, groups=subj):
            sc = StandardScaler().fit(X[tr])
            m = model_fn(); m.fit(sc.transform(X[tr]), y5[tr])
            pr = m.predict_proba(sc.transform(X[te])); prob[te] = pr
            pred[te] = pr.argmax(1)
        e = (pred == y5).mean() * 100
        cf = f1_score(yc, (pred == 0).astype(int), zero_division=0) * 100
        emrs.append(e); cfs.append(cf); last = (pred, prob)
    print(f"{tag:42s} EMR={np.mean(emrs):5.2f}±{np.std(emrs):.2f}%  CleanF1={np.mean(cfs):5.2f}±{np.std(cfs):.2f}%  (3 seeds)")
    return last

lr = lambda: LogisticRegression(C=0.5, max_iter=2000)
lrb = lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
cb = lambda: CatBoostClassifier(iterations=250, learning_rate=0.04, depth=4, l2_leaf_reg=6,
                                loss_function='MultiClass', random_seed=42, verbose=0, thread_count=1)
cbb = lambda: CatBoostClassifier(iterations=250, learning_rate=0.04, depth=4, l2_leaf_reg=6,
                                 loss_function='MultiClass', auto_class_weights='Balanced',
                                 random_seed=42, verbose=0, thread_count=1)

print("\n[단일 시점 입력 (실서비스와 동일 조건)]")
X1 = build(True)
cv_eval(X1, lr, "5-class LR")
cv_eval(X1, lrb, "5-class LR balanced")
pred_s, prob_s = cv_eval(X1, cb, "5-class CatBoost")
cv_eval(X1, cbb, "5-class CatBoost balanced")

print("\n[다시점 융합 입력 (동일 동작의 5개 시점 확률 평균/분산 사용; 다중 카메라 필요)]")
Xm = build(False)
cv_eval(Xm, lr, "5-class LR + multiview")
cv_eval(Xm, lrb, "5-class LR balanced + multiview")
pred_m, prob_m = cv_eval(Xm, cb, "5-class CatBoost + multiview")

print("\n[혼동행렬: 단일시점 CatBoost]  행=정답, 열=예측 (0:정상 1:(0,0,1) 2:(0,1,0) 3:(1,1,0) 4:(1,1,1))")
cm = np.zeros((5, 5), int)
for a, b in zip(y5, pred_s): cm[a, b] += 1
print(cm)
print("\n[시점별 EMR: 단일시점 CatBoost]")
for v in sorted(np.unique(views)):
    ii = views == v
    print(f"  {v}: EMR={(pred_s[ii]==y5[ii]).mean()*100:.2f}%  CleanF1={f1_score(yc[ii], (pred_s[ii]==0).astype(int), zero_division=0)*100:.2f}%")

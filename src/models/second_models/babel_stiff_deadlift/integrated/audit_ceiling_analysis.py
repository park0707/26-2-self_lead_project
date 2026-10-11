"""
audit_ceiling_analysis.py
v4 성능 감사: (1) 라벨 구조, (2) 서브모델별 오류 기여, (3) Oracle 상한,
(4) 임계치 튜닝의 낙관 편향(Leave-One-Subject-Out 임계치 선택)
"""
import os
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score

os.environ["OMP_NUM_THREADS"] = "1"
P = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"
c = np.load(P, allow_pickle=True)
s1, s2, s3 = c['oof_sd1'], c['oof_sd2'], c['oof_sd3']
y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
views, subj, sid = c['views'], c['subjects'], c['seq_ids']
N = len(s1)
p2 = np.maximum(s2, s1)

print("== 1. 라벨 구조 ==")
combo = Counter(zip(y1, y2, y3))
for k, v in sorted(combo.items(), key=lambda x: -x[1]):
    print(f"  (sd1,sd2,sd3)={k}: {v}건 ({v/N*100:.1f}%)")
print(f"  최빈 조합 단순 예측 EMR(베이스라인): {max(combo.values())/N*100:.2f}%")
print(f"  sd1=1 이면서 sd2=0 인 건수(생체역학 규칙 위반 라벨): {int(((y1==1)&(y2==0)).sum())}")
print("  seq_id 예시:", list(sid[:3]))

print("\n== 2. 라벨별 성능(임계 0.5) ==")
for n, s, y in [("sd1", s1, y1), ("sd2", p2, y2), ("sd3", s3, y3)]:
    pr = (s >= 0.5).astype(int)
    print(f"  {n}: 양성비율={y.mean()*100:.1f}% AUC={roc_auc_score(y, s)*100:.2f}% "
          f"Acc={(pr==y).mean()*100:.2f}% F1={f1_score(y, pr)*100:.2f}%")

print("\n== 3. Oracle 상한 (한 서브모델을 정답으로 교체) ==")
th = (0.55, 0.62, 0.55)
a1, a2, a3 = (s1 >= th[0]).astype(int), (p2 >= th[1]).astype(int), (s3 >= th[2]).astype(int)
def emr(a, b, c_): return ((a == y1) & (b == y2) & (c_ == y3)).mean() * 100
print(f"  현재 EMR(전역 임계치): {emr(a1,a2,a3):.2f}%")
print(f"  sd1 완벽 시: {emr(y1,a2,a3):.2f}%")
print(f"  sd2 완벽 시: {emr(a1,y2,a3):.2f}%")
print(f"  sd3 완벽 시: {emr(a1,a2,y3):.2f}%")
print(f"  sd1+sd2 완벽: {emr(y1,y2,a3):.2f}%  sd2+sd3 완벽: {emr(a1,y2,y3):.2f}%")
print("  라벨별 정확도 곱(독립 가정 EMR 기대치):",
      f"{(a1==y1).mean()*(a2==y2).mean()*(a3==y3).mean()*100:.2f}%")

print("\n== 4. 낙관 편향 점검: Leave-One-Subject-Out 임계치 선택 ==")
grid1 = np.linspace(0.45, 0.65, 9); grid2 = np.linspace(0.50, 0.70, 9); grid3 = np.linspace(0.45, 0.65, 9)
def score_sub(idx, t):
    p = [(s1[idx] >= t[0]).astype(int), (p2[idx] >= t[1]).astype(int), (s3[idx] >= t[2]).astype(int)]
    e = ((p[0]==y1[idx]) & (p[1]==y2[idx]) & (p[2]==y3[idx])).mean()*100
    f = f1_score(yc[idx], ((p[0]==0)&(p[1]==0)&(p[2]==0)).astype(int), zero_division=0)*100
    return e, f, e + f
def pick(idx):
    best, bt = -1, None
    for t1 in grid1:
        for t2 in grid2:
            for t3 in grid3:
                sc = score_sub(idx, (t1, t2, t3))[2]
                if sc > best: best, bt = sc, (t1, t2, t3)
    return bt
# (a) 전역 임계치 (낙관: 전체에서 선택) vs (b) LOSO 전역 vs (c) LOSO 시점별
pred = {k: np.zeros(N, dtype=int) for k in ("g", "v")}
for s in np.unique(subj):
    te = np.where(subj == s)[0]; tr = np.where(subj != s)[0]
    tg = pick(tr)
    pg = [(s1[te] >= tg[0]), (p2[te] >= tg[1]), (s3[te] >= tg[2])]
    pred["g"][te] = ((pg[0]==y1[te]) & (pg[1]==y2[te]) & (pg[2]==y3[te])).astype(int)
    for v in np.unique(views):
        trv = tr[views[tr] == v]; tev = te[views[te] == v]
        if len(tev) == 0: continue
        tv = pick(trv)
        pv = [(s1[tev] >= tv[0]), (p2[tev] >= tv[1]), (s3[tev] >= tv[2])]
        pred["v"][tev] = ((pv[0]==y1[tev]) & (pv[1]==y2[tev]) & (pv[2]==y3[tev])).astype(int)
print(f"  전체데이터로 선택한 전역 임계치 EMR (낙관): {emr(a1,a2,a3):.2f}%")
print(f"  LOSO 전역 임계치 EMR (정직):             {pred['g'].mean()*100:.2f}%")
print(f"  LOSO 시점별 임계치 EMR (정직):           {pred['v'].mean()*100:.2f}%")
print(f"  임계치 0.50 고정 EMR (튜닝 없음):        {emr((s1>=.5).astype(int),(p2>=.5).astype(int),(s3>=.5).astype(int)):.2f}%")

print("\n== 5. Clean 소표본 점검 ==")
for v in sorted(np.unique(views)):
    print(f"  {v}: 정상 {int(yc[views==v].sum())}건 / {int((views==v).sum())}건")

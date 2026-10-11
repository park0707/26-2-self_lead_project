# 분석 전용: Condition3(진짜 시점별 3D)에서 시점 앙상블/원본라벨 학습/물리 불일치 재확인. 파일 수정 없음.
import numpy as np
from sklearn.metrics import roc_auc_score
from v7_ablation_core import load_dataset, cb_factory, clean_labels
D = load_dataset(); c = np.load('v7_cond3_feature_cache.npz', allow_pickle=True); F3 = c['feats19']
V7 = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15, 16, 17]
y = D['y_orig']; subs = D['subs']; sids = D['sids']
base = np.array([s.rsplit('_view', 1)[0] for s in sids])
print('sequences(base)', len(set(base)))

# 물리: 결함/정상 중 무릎각(idx0) >= 160 비율 (C3 vs C4)
for nm, F in (('C4', D['feats']), ('C3', F3)):
    st = F[:, 0] >= 160
    print(f'{nm}: 결함 중 직선무릎 {st[y==1].mean()*100:.1f}% / 정상 중 {st[y==0].mean()*100:.1f}%  | 무릎각 단독 AUC(낮을수록 결함→-각도) {roc_auc_score(y, -F[:,0])*100:.1f}')

def loso(X, ytr, seed):
    oof = np.zeros(len(X))
    for s in np.unique(subs):
        va = subs == s
        m = cb_factory(seed=seed)(); m.fit(X[~va], ytr[~va]); oof[va] = m.predict_proba(X[va])[:, 1]
    return oof
def seq_mean(oof):
    out = np.zeros(len(oof))
    for b in set(base):
        m = base == b; out[m] = oof[m].mean()
    return out
for nm, cols, F in (('C3 v7 12D', V7, F3), ('C3 19D', list(range(19)), F3), ('C4 v7 12D', V7, D['feats'])):
    for lab, ytr in (('원본학습', y), ('정제학습', clean_labels(dict(D, feats=F), 'v6'))):
        a1, a2 = [], []
        for sd in (1, 2, 3):
            o = loso(F[:, cols], ytr, sd); a1.append(roc_auc_score(y, o) * 100)
            a2.append(roc_auc_score(y, seq_mean(o)) * 100)
        print(f'{nm} {lab}: 원본AUC 시점별 {np.mean(a1):.1f}±{np.std(a1):.1f} | 시퀀스(시점평균) {np.mean(a2):.1f}±{np.std(a2):.1f}')

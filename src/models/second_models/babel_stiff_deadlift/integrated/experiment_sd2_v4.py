# -*- coding: utf-8 -*-
"""
vidio-health-trainer | sd2(척추 중립) 고도화 실험 (v4.1 sd2 -> sd2 v4)
선행연구 반영: 스켈레톤 시계열 소표본 -> MiniRocket(랜덤 컨볼루션 PPV) + Ridge/GBDT 앙상블.
- 입력은 100% MediaPipe 64프레임 (AI-Hub JSON/라벨은 y 타깃으로만 사용)
- 피험자 격리 StratifiedGroupKFold (5-fold x 3 seeds)
- 기준선: 기존 sd2 (CatBoost+LGBM 0.5/0.5, 30D 피처) 를 동일 fold 에서 재측정
"""
import os, sys, re, itertools, warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import RidgeClassifierCV
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
sys.path.insert(0, "d:/")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RAW = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPN = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
OUT = os.path.join(HERE, "sd2_v4_cache.npz")

MP = dict(NOSE=0, LEAR=7, REAR=8, LSH=11, RSH=12, LWR=15, RWR=16, LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28)


def _ang(a, b, c):
    v1, v2 = a - b, c - b
    cs = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def extract_sd2_series(kps, spine=None):
    """(64,33,3) -> (C,64) 척추 관련 시계열 채널 (스케일/위치 불변)."""
    g = lambda i: kps[:, MP[i], :]
    msh = (g("LSH") + g("RSH")) / 2; mhip = (g("LHIP") + g("RHIP")) / 2
    mkn = (g("LKNE") + g("RKNE")) / 2; mank = (g("LANK") + g("RANK")) / 2
    mwr = (g("LWR") + g("RWR")) / 2; mear = (g("LEAR") + g("REAR")) / 2
    tl = np.linalg.norm(msh - mhip, axis=1, keepdims=True) + 1e-7
    ch = []
    ch.append(_ang(msh, mhip, mkn))          # 힙 힌지 각
    ch.append(_ang(mhip, mkn, mank))         # 무릎 각
    ch.append(_ang(mear, msh, mhip))         # 경추-체간
    ch.append(_ang(g("NOSE"), msh, mhip))
    for k in range(3):
        ch.append(((msh - mhip) / tl)[:, k])
        ch.append(((mwr - mhip) / tl)[:, k])
        ch.append(((mhip - mank) / tl)[:, k])
    dw = g("LWR") - g("RWR"); ds = g("LSH") - g("RSH"); dh = g("LHIP") - g("RHIP")
    for d in (dw, ds, dh):
        ch.append(np.degrees(np.arctan2(d[:, 1], d[:, 0] + 1e-7)))
    if spine is not None:  # 요추 곡률 poly_a 시계열
        z = np.cross((msh - mhip) / tl, ds)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-7)
        zc = np.stack([((spine[:, j] - mhip) * z).sum(-1) / tl[:, 0] for j in range(5)], 1)
        x = np.array([0, .25, .5, .75, 1.0])
        ch.append(np.array([np.polyfit(x, zc[t], 2)[0] for t in range(len(zc))]))
    else:
        ch.append(np.zeros(len(kps)))
    return np.stack(ch).astype(np.float32)


# ---------------- MiniRocket-lite (numpy) ----------------
def _kernels():
    idx = list(itertools.combinations(range(9), 3))
    ks = []
    for c in idx:
        w = -np.ones(9, np.float32); w[list(c)] = 2.0
        ks.append(w)
    return np.stack(ks)  # (84,9)


KER = _kernels()
DIL = [1, 2, 4, 8, 12]
QS = np.array([0.25, 0.5, 0.75])


def _conv(x, w, d):
    # x: (N,T) -> (N,T) 'same' zero padding, dilation d
    T = x.shape[1]; pad = 4 * d
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros_like(x)
    for j in range(9):
        out += w[j] * xp[:, j * d: j * d + T]
    return out


class MiniRocketLite:
    def fit(self, X):  # X (N,C,T) -> 바이어스 분위수 (비지도)
        self.mu = X.mean((0, 2), keepdims=True); self.sd = X.std((0, 2), keepdims=True) + 1e-6
        Xn = (X - self.mu) / self.sd
        self.bias = {}
        for c in range(Xn.shape[1]):
            for d in DIL:
                for k in range(len(KER)):
                    o = _conv(Xn[:, c], KER[k], d)
                    self.bias[(c, d, k)] = np.quantile(o.ravel(), QS)
        return self

    def transform(self, X):
        Xn = (X - self.mu) / self.sd
        feats = []
        for c in range(Xn.shape[1]):
            for d in DIL:
                for k in range(len(KER)):
                    o = _conv(Xn[:, c], KER[k], d)
                    b = self.bias[(c, d, k)]
                    feats.append(np.stack([(o > bb).mean(1) for bb in b], 1))
        return np.concatenate(feats, 1).astype(np.float32)


def load_all():
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        return z["S"], z["ids"]
    f30 = np.load(os.path.join(BASE, "stiff_spine_ambiguity_levels.npz"), allow_pickle=True)
    ids = f30["sequence_ids"]
    sp = np.load(SPN, allow_pickle=True) if os.path.exists(SPN) else None
    spd = {s: v for s, v in zip(sp["sequence_ids"], sp["spine_9_64f_vision"])} if sp is not None else {}
    S = []
    for sid in ids:
        d = np.load(os.path.join(RAW, f"{sid}.npz"), allow_pickle=True)
        k = d["kps_33_3d_64f"] if "kps_33_3d_64f" in d else d["kps_33_2d_64f"]
        S.append(extract_sd2_series(k, spd.get(sid)))
    S = np.stack(S)
    np.savez_compressed(OUT, S=S, ids=ids)
    return S, ids


def metrics(y, s, thr=None):
    auc = roc_auc_score(y, s)
    # sd2 라벨 1=결함. 양성 F1 은 순위 기반 임계(결함 비율 맞춤)
    t = np.quantile(s, 1 - y.mean()) if thr is None else thr
    return auc, f1_score(y, (s >= t).astype(int))


if __name__ == "__main__":
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    S, ids = load_all()
    print("series", S.shape, "NaN", int(np.isnan(S).sum()), "Inf", int(np.isinf(S).sum()))
    oof = np.load(os.path.join(HERE, "integrated_stiff_oof_cache_v2.npz"), allow_pickle=True)
    f30 = np.load(os.path.join(BASE, "stiff_spine_ambiguity_levels.npz"), allow_pickle=True)
    pos = {s: i for i, s in enumerate(f30["sequence_ids"])}
    order = np.array([pos[s] for s in oof["seq_ids"]])
    X30 = f30["features_30d"][order]; S = S[order]
    y = oof["y_sd2"].astype(int); subj = oof["subjects"]; views = oof["views"]
    assert len(y) == len(S) == len(X30) and np.isfinite(X30).all()
    print("y_sd2 pos rate", y.mean(), "N", len(y))

    mr = MiniRocketLite().fit(S)       # 비지도 바이어스 (라벨 미사용)
    R = mr.transform(S)
    print("rocket feats", R.shape)
    res = {k: [] for k in ["gbdt", "ridge", "blend", "gbdt+ridge_rank"]}
    perview = {k: {} for k in res}
    for seed in (0, 1, 2):
        sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        o = {k: np.zeros(len(y)) for k in res}
        for tr, te in sg.split(X30, y, subj):
            cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4, auto_class_weights="Balanced", verbose=0, random_seed=seed)
            lg = LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16, class_weight="balanced", verbose=-1, random_state=seed)
            cb.fit(X30[tr], y[tr]); lg.fit(X30[tr], y[tr])
            pg = 0.5 * cb.predict_proba(X30[te])[:, 1] + 0.5 * lg.predict_proba(X30[te])[:, 1]
            mu, sd = R[tr].mean(0), R[tr].std(0) + 1e-6
            rc = RidgeClassifierCV(alphas=np.logspace(0, 4, 9), class_weight="balanced").fit((R[tr] - mu) / sd, y[tr])
            dr = rc.decision_function((R[te] - mu) / sd)
            pr = 1 / (1 + np.exp(-dr))
            o["gbdt"][te] = pg; o["ridge"][te] = pr
            o["blend"][te] = 0.5 * pg + 0.5 * pr
            o["gbdt+ridge_rank"][te] = (rankdata(pg) + rankdata(pr)) / (2 * len(te))
        for k in res:
            res[k].append(metrics(y, o[k]))
            for v in np.unique(views):
                m = views == v
                perview[k].setdefault(v, []).append(roc_auc_score(y[m], o[k][m]))
        print("seed", seed, {k: round(res[k][-1][0] * 100, 2) for k in res}, flush=True)
    print("\n=== sd2 AUC / F1 (mean over 3 seeds, 피험자 격리) ===")
    for k in res:
        a = np.array(res[k]) * 100
        print(f"{k:18s} AUC {a[:,0].mean():.2f}±{a[:,0].std():.2f}  F1 {a[:,1].mean():.2f}")
    print("\n시점별 AUC")
    for k in res:
        print(k, {v: round(float(np.mean(perview[k][v])) * 100, 2) for v in sorted(perview[k])})

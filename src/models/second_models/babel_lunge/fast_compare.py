import os
import sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

D = r"D:\src\models\second_models\babel_lunge"
c = np.load(r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz", allow_pickle=True)
y, views, subj = c['y'], c['views'], c['subjects']
VIEWS = [('view5', '시점A'), ('view2', '시점B'), ('view3', '시점C'), ('view4', '시점D'), ('view1', '시점E')]
W = (0.25, 0.25, 0.25, 0.25)

def ens(o, w=W):
    return w[0] * o['tcn'] + w[1] * o['lgb'] + w[2] * o['xgb'] + w[3] * o['cat']

def fast_best_at_coverage(p, y, min_cov):
    best = None
    tcs = np.linspace(0.20, 0.50, 61)
    tds = np.linspace(0.50, 0.80, 61)
    
    # Pre-calculate boolean arrays for speed
    # We want max f1 where cov >= min_cov
    for tc in tcs:
        pred_0 = p <= tc
        for td in tds:
            pred_1 = p >= td
            m = pred_0 | pred_1
            cov = m.mean()
            if cov < min_cov:
                continue
                
            y_m = y[m]
            pred_m = pred_1[m].astype(int) # Since m is only True where it's 0 or 1, and pred_1 means 1
            
            # fast f1
            tp = (y_m * pred_m).sum()
            fp = ((1 - y_m) * pred_m).sum()
            fn = (y_m * (1 - pred_m)).sum()
            
            if tp + fp + fn == 0:
                continue
            
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if best is None or f1 > best[0]['f1']:
                prec = tp / (tp + fp + 1e-9)
                rec = tp / (tp + fn + 1e-9)
                acc = (y_m == pred_m).mean()
                best = (dict(cov=cov, f1=f1, prec=prec, rec=rec, acc=acc), tc, td)
    return best

def oof_path(mode):
    pre = "bl4_v5_oof_" if mode.startswith("sp_") else "bl4_v4b_oof_"
    return os.path.join(D, f"{pre}{mode}.npz")

def three_state(p, tc, td):
    pred = np.where(p <= tc, 0, np.where(p >= td, 1, -1))
    m = pred != -1
    if m.sum() == 0:
        return dict(cov=0, f1=0, prec=0, rec=0, acc=0)
    return dict(cov=m.mean(), f1=f1_score(y[m], pred[m], zero_division=0),
                prec=precision_score(y[m], pred[m], zero_division=0),
                rec=recall_score(y[m], pred[m], zero_division=0), acc=accuracy_score(y[m], pred[m]))

def evaluate(mode):
    print(f"Evaluating {mode}...")
    o = np.load(oof_path(mode))
    p = ens(o)
    res = dict(mode=mode, auc=roc_auc_score(y, p), f1_2s=f1_score(y, p >= 0.5),
               auc_tcn=roc_auc_score(y, o['tcn']), auc_lgb=roc_auc_score(y, o['lgb']),
               auc_xgb=roc_auc_score(y, o['xgb']), auc_cat=roc_auc_score(y, o['cat']))
    res['fixed'] = three_state(p, 0.44, 0.51)  # v3 공식 임계치 그대로
    for cv in (0.95, 0.90):
        r, tc, td = fast_best_at_coverage(p, y, cv)
        res[f'cov{int(cv*100)}'] = (r, tc, td)
    res['views'] = {}
    for v, name in VIEWS:
        m = views == v
        res['views'][name] = (roc_auc_score(y[m], p[m]), f1_score(y[m], p[m] >= 0.5))
    res['subj_auc'] = {s: roc_auc_score(y[subj == s], p[subj == s]) for s in sorted(set(subj))
                       if len(set(y[subj == s])) > 1}
    return res

if __name__ == '__main__':
    modes = [m for m in ('ref', 'cl', 'sp_ref', 'sp_cl') if os.path.exists(oof_path(m))]
    R = {m: evaluate(m) for m in modes}
    print("=== 전체 (OOF, N=787, 피험자 격리 5-Fold) ===")
    print(f"{'mode':6s} | AUC    | 2sF1   | 고정(0.44/0.51) cov/F1  | cov>=95% F1 (tc,td)      | cov>=90% F1 (tc,td)")
    for m, r in R.items():
        f = r['fixed']; a, atc, atd = r['cov95']; b, btc, btd = r['cov90']
        print(f"{m:6s} | {r['auc']*100:5.2f}% | {r['f1_2s']*100:5.2f}% | {f['cov']*100:5.1f}% / {f['f1']*100:5.2f}%     | "
              f"{a['f1']*100:5.2f}% cov{a['cov']*100:5.1f} ({atc:.2f},{atd:.2f}) | {b['f1']*100:5.2f}% cov{b['cov']*100:5.1f} ({btc:.2f},{btd:.2f})")
    print("\n=== 서브모델 AUC (c1~c4 대응: TCN/LGB/XGB/CAT) ===")
    for m, r in R.items():
        print(f"{m:6s} TCN {r['auc_tcn']*100:5.2f} LGB {r['auc_lgb']*100:5.2f} XGB {r['auc_xgb']*100:5.2f} CAT {r['auc_cat']*100:5.2f}")
    print("\n=== 5대 시점 AUC / 2-State F1 ===")
    for m, r in R.items():
        print(m, ' '.join(f"{n}:{v[0]*100:5.1f}/{v[1]*100:5.1f}" for n, v in r['views'].items()))
    print("\n=== 피험자별 AUC ===")
    for m, r in R.items():
        print(m, ' '.join(f"{s[-1]}:{v*100:4.0f}" for s, v in r['subj_auc'].items()))

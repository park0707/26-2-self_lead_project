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
    for tc in tcs:
        pred_0 = p <= tc
        for td in tds:
            pred_1 = p >= td
            m = pred_0 | pred_1
            cov = m.mean()
            if cov < min_cov:
                continue
            y_m = y[m]
            pred_m = pred_1[m].astype(int)
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
    o = np.load(os.path.join(D, f"{mode}.npz"))
    p = ens(o)
    res = dict(mode=mode, auc=roc_auc_score(y, p), f1_2s=f1_score(y, p >= 0.5),
               auc_tcn=roc_auc_score(y, o['tcn']), auc_lgb=roc_auc_score(y, o['lgb']),
               auc_xgb=roc_auc_score(y, o['xgb']), auc_cat=roc_auc_score(y, o['cat']))
    res['fixed'] = three_state(p, 0.44, 0.51)
    for cv in (0.95, 0.90):
        r, tc, td = fast_best_at_coverage(p, y, cv)
        res[f'cov{int(cv*100)}'] = (r, tc, td)
    res['views'] = {}
    for v, name in VIEWS:
        m = views == v
        res['views'][name] = (roc_auc_score(y[m], p[m]), f1_score(y[m], p[m] >= 0.5))
    res['subj_auc'] = {s: roc_auc_score(y[subj == s], p[subj == s]) for s in sorted(set(subj)) if len(set(y[subj == s])) > 1}
    return res

if __name__ == '__main__':
    modes = ['bl4_v4b_oof_cl', 'bl4_v6_flip_oof_cl']
    R = {m: evaluate(m) for m in modes}
    print("=== OOF ===")
    print(f"{'mode':20s} | AUC    | 2sF1   | cov>=95% F1 (tc,td)      | cov>=90% F1 (tc,td)")
    for m, r in R.items():
        a, atc, atd = r['cov95']; b, btc, btd = r['cov90']
        print(f"{m:20s} | {r['auc']*100:5.2f}% | {r['f1_2s']*100:5.2f}% | "
              f"{a['f1']*100:5.2f}% cov{a['cov']*100:5.1f} ({atc:.2f},{atd:.2f}) | {b['f1']*100:5.2f}% cov{b['cov']*100:5.1f} ({btc:.2f},{btd:.2f})")
    print("\n=== Submodels AUC ===")
    for m, r in R.items():
        print(f"{m:20s} TCN {r['auc_tcn']*100:5.2f} LGB {r['auc_lgb']*100:5.2f} XGB {r['auc_xgb']*100:5.2f} CAT {r['auc_cat']*100:5.2f}")
    print("\n=== Views AUC / 2sF1 ===")
    for m, r in R.items():
        print(m, ' '.join(f"{n}:{v[0]*100:5.1f}/{v[1]*100:5.1f}" for n, v in r['views'].items()))

"""
train_and_package_integrated_v4_1.py
v4.1: 5-class label-powerset 스태커 (유효 조합만 디코딩) + Clean prior-shift(w0) + 신뢰도 기반 3-State 보류.
- 베이스라인 계승: v4 번들의 서브모델(sd1_v8/sd2/sd3_v4) 가중치를 그대로 복사하고 메타 단계만 교체.
- 단일 카메라 입력 기준(다시점 융합 미사용).
- 보고 성능은 피험자 격리 중첩 CV(바깥 5-fold, 안쪽 4-fold에서 w0 선택, 3 seeds)의 정직한 값.
- 모델 구성 전 검증(shape/NaN/누수 차단) 및 구성 후 검증(서브모델별 F1, 시점별 성능) 포함.
"""
import os, sys, pickle, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

os.environ["OMP_NUM_THREADS"] = "1"
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR)
from stiff_meta_v41 import build_meta_features_v41, COMBOS, COMBO_ID, FEATURE_NAMES

SD_DIR = os.path.abspath(os.path.join(CUR, '..'))
CACHE = os.path.join(CUR, "integrated_stiff_oof_cache_v2.npz")
SD2_NPZ = os.path.join(SD_DIR, "stiff_spine_ambiguity_levels.npz")
V4_BUNDLE = os.path.join(CUR, "stiff_integrated_v4_bundle.pkl")
OUT_BUNDLE = os.path.join(CUR, "stiff_integrated_v4_1_bundle.pkl")
OUT_JSON = os.path.join(CUR, "stiff_integrated_v4_1_evaluation_results.json")
VIEW_ALIAS = {"view5": "시점 A (우측 후방 45°)", "view2": "시점 B (우측 전방 45°)", "view3": "시점 C (정면 0°)",
              "view4": "시점 D (좌측 전방 45°)", "view1": "시점 E (좌측 후방 45°)"}
C_REG = 0.05
W0_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
REJECT_RATE = 0.09  # 정직한 평가에서 평균 보류율이 10% 이하가 되도록 (0.10은 평균 커버리지 89.9%였음)


def fit_prob(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C_REG, max_iter=4000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))


def shift(pr, w0):
    q = pr.copy(); q[:, 0] *= w0
    return q / q.sum(1, keepdims=True)


def obj(y5, yc, pr, w0):
    pred = shift(pr, w0).argmax(1)
    return (pred == y5).mean() * 100 + f1_score(yc, (pred == 0).astype(int), zero_division=0) * 100


def inner_oof(X, y, grp, seed):
    oof = np.zeros((len(y), 5))
    for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(X, y, groups=grp):
        oof[b] = fit_prob(X[a], y[a], X[b])
    return oof


def nested_eval(X, y5, yc, subj, seed):
    N = len(y5); pred = np.zeros(N, int); accept = np.zeros(N, bool); conf = np.zeros(N)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y5, groups=subj):
        oo = inner_oof(X[tr], y5[tr], subj[tr], seed)
        w0 = max(W0_GRID, key=lambda w: obj(y5[tr], yc[tr], oo, w))
        q = shift(fit_prob(X[tr], y5[tr], X[te]), w0)
        pred[te] = q.argmax(1); conf[te] = q.max(1)
        thr = np.quantile(shift(oo, w0).max(1), REJECT_RATE)
        accept[te] = q.max(1) >= thr
    return pred, accept, conf


def metrics(y5, yc, pred, mask=None):
    m = np.ones(len(y5), bool) if mask is None else mask
    return {'emr': float((pred[m] == y5[m]).mean() * 100),
            'clean_f1': float(f1_score(yc[m], (pred[m] == 0).astype(int), zero_division=0) * 100),
            'clean_tp': int(((pred[m] == 0) & (yc[m] == 1)).sum()),
            'coverage': float(m.mean() * 100)}


def main():
    print("=" * 78)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 통합 진단 v4.1 (5-class 스태커, 단일 카메라)")
    print("=" * 78)
    c = np.load(CACHE, allow_pickle=True)
    s1, s2, s3 = c['oof_sd1'], c['oof_sd2'], c['oof_sd3']
    y1, y2, y3, yc = c['y_sd1'], c['y_sd2'], c['y_sd3'], c['y_clean']
    views, subj, sid = c['views'], c['subjects'], list(c['seq_ids'])
    sd2 = np.load(SD2_NPZ, allow_pickle=True)
    mp = {s: i for i, s in enumerate(sd2['sequence_ids'])}
    bio = sd2['features_30d'][[mp[s] for s in sid]].astype(np.float32)
    N = len(s1)

    # ---- 모델 구성 전 검증 (Pre-Modeling) ----
    X = build_meta_features_v41(s1, s2, s3, views, bio)
    assert X.shape == (N, len(FEATURE_NAMES)), X.shape
    assert np.isfinite(X).all(), "NaN/Inf 존재"
    assert len(np.unique(subj)) == 9 and set(np.unique(views)) == {'view1', 'view2', 'view3', 'view4', 'view5'}
    y5 = np.array([COMBO_ID[(int(a), int(b), int(d))] for a, b, d in zip(y1, y2, y3)])
    assert ((y1 == 1) <= (y2 == 1)).all(), "생체역학 규칙(sd1->sd2) 위반 라벨 존재"
    print(f"[Pre] X={X.shape}, NaN/Inf=0, 피험자 9명/5시점 전수, 입력=순수 비전 피처+서브모델 OOF 확률(라벨 미투입)")

    # ---- 정직한 성능: 중첩 CV 3 seeds ----
    res2, res3, per_view, preds = [], [], {v: [] for v in VIEW_ALIAS}, []
    for seed in (42, 7, 123):
        pred, acc, conf = nested_eval(X, y5, yc, subj, seed)
        preds.append(pred)
        res2.append(metrics(y5, yc, pred)); res3.append(metrics(y5, yc, pred, acc))
        for v in VIEW_ALIAS:
            ii = views == v
            per_view[v].append(metrics(y5[ii], yc[ii], pred[ii]))
    avg = lambda L: {k: float(np.mean([d[k] for d in L])) for k in L[0]}
    sd = lambda L, k: float(np.std([d[k] for d in L]))
    m2, m3 = avg(res2), avg(res3)
    print(f"\n[정직한 성능 | 중첩 CV 3 seeds]")
    print(f"  2-State : EMR={m2['emr']:.2f}±{sd(res2,'emr'):.2f}%  CleanF1={m2['clean_f1']:.2f}±{sd(res2,'clean_f1'):.2f}%  정상TP={m2['clean_tp']:.0f}/130")
    print(f"  3-State : EMR={m3['emr']:.2f}±{sd(res3,'emr'):.2f}%  CleanF1={m3['clean_f1']:.2f}±{sd(res3,'clean_f1'):.2f}%  커버리지={m3['coverage']:.1f}%")

    # ---- 모델 구성 후 검증 (Post-Modeling): 서브모델(c1~c3)별 + 시점별 ----
    pred0 = preds[0]
    dec = np.array([COMBOS[k] for k in pred0])
    print("\n[Post] 서브모델별(5-class 디코딩 결과) F1 / 원 서브모델 AUC")
    sub = {}
    for j, (n, y, s) in enumerate([("sd1", y1, s1), ("sd2", y2, np.maximum(s2, s1)), ("sd3", y3, s3)]):
        f = f1_score(y, dec[:, j], zero_division=0) * 100; a = roc_auc_score(y, s) * 100
        sub[n] = {'f1_decoded': float(f), 'auc_submodel_oof': float(a)}
        print(f"  {n}: F1={f:.2f}%  AUC(서브모델 OOF)={a:.2f}%")
    print("[Post] 5대 시점별 (2-State)")
    vres = {}
    for v, a in VIEW_ALIAS.items():
        r = avg(per_view[v]); vres[v] = {'alias': a, **r}
        print(f"  {a}: EMR={r['emr']:.2f}%  CleanF1={r['clean_f1']:.2f}%")

    # ---- 최종 배포 모델: 전수 학습 + w0/보류 임계치 (그룹 CV OOF 기준) ----
    oo = inner_oof(X, y5, subj, 42)
    w0 = max(W0_GRID, key=lambda w: obj(y5, yc, oo, w))
    thr = float(np.quantile(shift(oo, w0).max(1), REJECT_RATE))
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C_REG, max_iter=4000).fit(sc.transform(X), y5)
    print(f"\n[배포] w0={w0}, 보류 임계(신뢰도)={thr:.4f}")

    with open(V4_BUNDLE, 'rb') as fp:
        b = pickle.load(fp)
    keep = [k for k in b if k.startswith(('sd1_', 'sd2_', 'sd3_'))]
    bundle = {k: b[k] for k in keep}
    bundle.update({
        'version': 'v4.1', 'architecture': 'C5_LabelPowerset5_LR_SingleCamera_v41',
        'exercise': '03_바벨_스티프_데드리프트',
        'meta_scaler41': sc, 'meta_lr41': lr, 'w0': float(w0), 'reject_threshold': thr,
        'combos': COMBOS, 'feature_names': FEATURE_NAMES,
        'metrics': {'emr_2state': m2['emr'], 'clean_f1_2state': m2['clean_f1'], 'emr_3state': m3['emr'],
                    'clean_f1_3state': m3['clean_f1'], 'coverage': m3['coverage'],
                    'evaluation': '피험자 격리 중첩 CV, 3 seeds (정직한 값)'},
        'view_results': vres, 'submodel_audit': sub})
    with open(OUT_BUNDLE, 'wb') as fp:
        pickle.dump(bundle, fp)
    with open(OUT_JSON, 'w', encoding='utf-8') as fp:
        json.dump({'metrics': bundle['metrics'], 'view_results': vres, 'submodel_audit': sub,
                   'std': {'emr_2state': sd(res2, 'emr'), 'clean_f1_2state': sd(res2, 'clean_f1'),
                           'emr_3state': sd(res3, 'emr'), 'clean_f1_3state': sd(res3, 'clean_f1')}},
                  fp, indent=2, ensure_ascii=False)
    print(f"[+] 번들: {OUT_BUNDLE}\n[+] JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()

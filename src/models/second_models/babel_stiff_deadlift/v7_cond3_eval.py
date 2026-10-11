# -*- coding: utf-8 -*-
"""분석 전용: Condition3_MediaPipe_3D(시점별 고유 텐서)로 v7 재평가. 기존 파일/번들 수정 없음.
산출: v7_cond3_eval.json (utf-8), 콘솔 요약. 피처 캐시: v7_cond3_feature_cache.npz"""
import os, json, hashlib
import numpy as np
from v7_ablation_core import (load_dataset, clean_labels, run_loso, cb_factory, summarize,
                              three_state, CURRENT_DIR)

C3_DIR = 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'
C3_CACHE = os.path.join(CURRENT_DIR, 'v7_cond3_feature_cache.npz')
V7 = [0, 1, 3, 4, 8, 10, 11, 13, 14, 15, 16, 17]

D = load_dataset()  # Condition4 기반 (기존)
sids = D['sids']

if os.path.exists(C3_CACHE):
    c = np.load(C3_CACHE, allow_pickle=True)
    F3, h3 = c['feats19'], c['hash']
else:
    from stiff_deadlift_knee_features_v6 import extract_knee_invariance_features_v6
    from sd3_v2_classifier import SD3V2Classifier
    seen = {}
    for fn in ['Dual_Train.npz', 'Dual_Valid.npz', 'Dual_Test_Unseen_View.npz', 'Dual_Test_Strict.npz', 'Dual_Test_Unseen_Subj.npz']:
        d = np.load(f'{C3_DIR}/{fn}', allow_pickle=True)
        for s, x in zip(d['sequence_ids'], d['X_Format_B']):
            seen[str(s)] = x
    sd3 = SD3V2Classifier()
    F3 = np.zeros((len(sids), 19), dtype=np.float32); h3 = []
    for i, s in enumerate(sids):
        x = seen[s]
        F3[i] = extract_knee_invariance_features_v6(x, p_sd3=float(sd3.predict_proba(x)))
        h3.append(hashlib.md5(x.tobytes()).hexdigest())
    h3 = np.array(h3)
    np.savez(C3_CACHE, feats19=F3, hash=h3)

print('고유 텐서: C4 =', len(set(D['thash'])), ' C3 =', len(set(h3)), ' / N =', len(sids))
F4 = D['feats']
print('피처 평균 비교 (C4 vs C3), v7 12D')
for j in V7:
    print(f'  idx{j:2d}  C4 mean={F4[:, j].mean():8.3f} sd={F4[:, j].std():7.3f} | C3 mean={F3[:, j].mean():8.3f} sd={F3[:, j].std():7.3f}')
print('NaN/Inf C3:', int(np.sum(~np.isfinite(F3))))

D3 = dict(D); D3['feats'] = F3; D3['uniq'] = np.ones(len(sids), bool); D3['thash'] = h3
y_orig = D['y_orig']
SEEDS = [11, 12, 13, 14, 15]
out = {}
for name, DD in (('C4(기존, 시점중복)', D), ('C3(시점별 고유)', D3)):
    y_cl = clean_labels(DD, 'v6')
    ev = {'orig': y_orig, 'clean': y_cl}
    print(f'\n== {name}: 정제로 바뀐 라벨 {int((y_cl != y_orig).sum())}건')
    r_cl = run_loso(DD, V7, y_cl, SEEDS, lambda s: cb_factory(seed=s), ev)
    r_or = run_loso(DD, V7, y_orig, SEEDS, lambda s: cb_factory(seed=s), ev)
    row = {}
    for tl, r in (('정제학습', r_cl), ('원본학습', r_or)):
        for el in ('orig', 'clean'):
            s = {k: summarize(r[el], k) for k in ('auc', 'f1d', 'prec', 'spec', 'abst')}
            row[f'{tl}/{el}'] = {k: [round(v[0], 2), round(v[1], 2)] for k, v in s.items()}
            print(f'  {tl}→{el}채점: AUC={s["auc"][0]:.1f}±{s["auc"][1]:.1f} F1d={s["f1d"][0]:.1f}±{s["f1d"][1]:.1f} '
                  f'정밀도={s["prec"][0]:.1f} 특이도={s["spec"][0]:.1f} 보류={s["abst"][0]:.1f}')
    out[name] = row

# 시점별 AUC (C3, 정제학습 OOF 1시드, 원본/정제 채점)
from sklearn.metrics import roc_auc_score
y_cl = clean_labels(D3, 'v6'); X = F3[:, V7]; subs = D['subs']; oof = np.zeros(len(X))
for s in np.unique(subs):
    va = subs == s
    m = cb_factory(seed=42)(); m.fit(X[~va], y_cl[~va]); oof[va] = m.predict_proba(X[va])[:, 1]
print('\n시점별 AUC (C3, 시드42, 정제학습)')
vres = {}
for v in sorted(set(D['views'])):
    m = D['views'] == v
    if len(np.unique(y_orig[m])) > 1:
        a_o, a_c = roc_auc_score(y_orig[m], oof[m]) * 100, roc_auc_score(y_cl[m], oof[m]) * 100
        vres[str(v)] = [round(a_o, 1), round(a_c, 1), int(m.sum())]
        print(f'  {v}: 원본 {a_o:.1f} / 정제 {a_c:.1f} (N={int(m.sum())})')
out['per_view_C3'] = vres
json.dump(out, open(os.path.join(CURRENT_DIR, 'v7_cond3_eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('완료')




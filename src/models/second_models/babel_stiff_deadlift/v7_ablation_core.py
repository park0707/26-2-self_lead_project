# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 / Defect: sd1 (무릎 구부린 채 고정 위반)
Module: v7_ablation_core.py
Purpose: v6 구성요소 절제(Ablation) 실험 공용 코어
  - 피처 캐시 (19D, v6 추출기 계승), 라벨/피험자/시점/중복 마스크
  - 반복 StratifiedGroupKFold (피험자 격리) + 다중 시드 평가
  - 이중 채점: 원본(AI-Hub) 라벨 / 정제 라벨
  - 3-State(보류율 <= 9.9%) 결함 F1, 정밀도, 특이도, AUC
"""

import os
import sys
import hashlib
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from stiff_deadlift_knee_features_v6 import extract_knee_invariance_features_v6, FEATURE_NAMES_19D

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
FEAT_CACHE = os.path.join(CURRENT_DIR, "v7_feature_cache.npz")

V6_IDX = [0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17]
HALO_KEYS = ["Key213", "Key214", "Key215"]
ABST_LIMIT = 9.9


def load_dataset():
    """v6와 동일 데이터 로드. 반환: dict(feats19, y_orig, subs, views, sids, uniq_mask, valid_idx)"""
    data = np.load(CACHE_PATH, allow_pickle=True)
    sids = np.array([str(s) for s in data['sequence_ids']])
    if os.path.exists(FEAT_CACHE):
        c = np.load(FEAT_CACHE, allow_pickle=True)
        feats = c['feats19']
        tensor_hash = c['tensor_hash']
    else:
        from src.models.first_model.dataset import Stage1DatasetLoader
        from sd3_v2_classifier import SD3V2Classifier
        splits = Stage1DatasetLoader.load_all_dual_splits()
        seen = {}
        for sd in splits.values():
            m = sd['y_action'] == 2
            for sid, x in zip(sd['sequence_ids'][m], sd['X_Format_B'][m]):
                seen[str(sid)] = x
        sd3 = SD3V2Classifier()
        feats = np.zeros((len(sids), 19), dtype=np.float32)
        tensor_hash = []
        for i, sid in enumerate(sids):
            x = seen[sid]
            feats[i] = extract_knee_invariance_features_v6(x, p_sd3=float(sd3.predict_proba(x)))
            tensor_hash.append(hashlib.md5(x.tobytes()).hexdigest())
        tensor_hash = np.array(tensor_hash)
        np.savez(FEAT_CACHE, feats19=feats, tensor_hash=tensor_hash)

    valid = ~data['l4_mask']
    vi = np.where(valid)[0]
    # 중복 텐서 대표 1개만 True (실효 표본 평가용)
    seen_h, uniq = set(), np.zeros(len(vi), dtype=bool)
    for k, i in enumerate(vi):
        if tensor_hash[i] not in seen_h:
            seen_h.add(tensor_hash[i])
            uniq[k] = True
    return dict(
        feats=feats[vi], y_orig=data['labels'][vi].astype(int), subs=data['subject_ids'][vi],
        views=data['view_ids'][vi], sids=sids[vi], uniq=uniq, thash=tensor_hash[vi],
    )


def clean_labels(D, rule="v6"):
    """결함(1) 중 Key213/214/215 오라벨 후보를 정상(0)으로 정제. rule: none/v6/loose/strict"""
    y = D['y_orig'].copy()
    if rule == "none":
        return y
    F = D['feats']
    for i, sid in enumerate(D['sids']):
        if y[i] == 1 and any(k in sid for k in HALO_KEYS):
            ang, dl, dy, exf = F[i, 0], F[i, 1], F[i, 3], F[i, 13]
            if rule == "v6":
                ok = ang >= 160.0 and dl <= 10.0 and dy <= 0.30
            elif rule == "strict":
                ok = ang >= 162.0 and dl <= 8.0 and dy <= 0.30
            elif rule == "loose":
                ok = ang >= 158.0 and dl <= 12.0
            elif rule == "excess":
                ok = ang >= 158.0 and dl <= 10.0 and exf <= 1.5
            else:
                raise ValueError(rule)
            if ok:
                y[i] = 0
    return y


def three_state(p, y, uniq=None):
    """보류율 <= 9.9%를 만족하는 최대 대칭 밴드로 3-State 평가. 반환 dict"""
    best = 0.005
    for d in np.linspace(0.005, 0.25, 400):
        if np.mean((p >= 0.5 - d) & (p < 0.5 + d)) * 100 <= ABST_LIMIT:
            best = d
    st = np.where(p >= 0.5 + best, 1, np.where(p < 0.5 - best, 0, 2))
    out = {}
    for tag, m in (("", np.ones(len(y), bool)), ("_u", uniq if uniq is not None else np.ones(len(y), bool))):
        pm, ym, sm = p[m], y[m], st[m]
        dec = sm != 2
        if dec.sum() == 0 or len(np.unique(ym[dec])) < 2:
            out.update({f"f1d{tag}": 0.0, f"prec{tag}": 0.0, f"spec{tag}": 0.0, f"auc{tag}": 0.5, f"abst{tag}": 100.0})
            continue
        tn, fp, fn, tp = confusion_matrix(ym[dec], sm[dec], labels=[0, 1]).ravel()
        out[f"f1d{tag}"] = f1_score(ym[dec], sm[dec], pos_label=1, zero_division=0) * 100
        out[f"prec{tag}"] = tp / max(tp + fp, 1) * 100
        out[f"spec{tag}"] = tn / max(tn + fp, 1) * 100
        out[f"auc{tag}"] = roc_auc_score(ym, pm) * 100
        out[f"abst{tag}"] = float(np.mean(sm == 2) * 100)
    return out


def cb_factory(depth=4, iters=200, lr=0.04, l2=5, cw='Balanced', seed=42):
    def f():
        return CatBoostClassifier(iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
                                  auto_class_weights=cw, random_seed=seed, verbose=0)
    return f


def run_loso(D, cols, y_train, model_seeds, model_fn_of_seed, eval_labels):
    """Leave-One-Subject-Out(9 folds, 결정적 분할). 잡음원은 모델 시드뿐. 반환: {eval_name: [metrics per seed]}"""
    X = D['feats'][:, cols]
    subs = D['subs']
    res = {k: [] for k in eval_labels}
    for ms in model_seeds:
        oof = np.zeros(len(X))
        for s in np.unique(subs):
            va = subs == s
            tr = ~va
            if len(np.unique(y_train[tr])) < 2:
                oof[va] = 0.5
                continue
            m = model_fn_of_seed(ms)()
            m.fit(X[tr], y_train[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        for k, yl in eval_labels.items():
            res[k].append(three_state(oof, yl, D['uniq']))
    return res


def run_cv(D, cols, y_train, split_seeds, model_fn, eval_labels):
    """반복 그룹 CV. eval_labels: {'orig': y, 'clean': y}. 반환: {eval_name: [metrics per repeat]}"""
    X = D['feats'][:, cols]
    res = {k: [] for k in eval_labels}
    for ss in split_seeds:
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=ss)
        oof = np.zeros(len(X))
        for tr, va in sg.split(X, y_train, D['subs']):
            m = model_fn()
            m.fit(X[tr], y_train[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        for k, yl in eval_labels.items():
            res[k].append(three_state(oof, yl, D['uniq']))
    return res


def summarize(res_list, key):
    arr = np.array([r[key] for r in res_list])
    return arr.mean(), arr.std()


def paired(res_a, res_b, key):
    """b - a 쌍체 차이: (평균, 표준편차, b가 이긴 반복 비율)"""
    a = np.array([r[key] for r in res_a]); b = np.array([r[key] for r in res_b])
    d = b - a
    return d.mean(), d.std(), float(np.mean(d > 0))

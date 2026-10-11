# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: audit_bl_overfit_leakage.py
Purpose: bl1~bl5 세부 오류 모델의 과적합 / 데이터 누수 정량 감사 (기존 모델/캐시 수정 없음, 읽기 전용)

감사 항목 (모델별 대리 앙상블 LightGBM+CatBoost 사용 - 최종 번들 재학습 아님):
  A. Train vs OOF 성능 격차 (과적합 지표)
  B. 라벨 순열(Permutation) 테스트: y를 섞으면 AUC가 0.5여야 함 (누수/파이프라인 결함 탐지)
  C. 피험자 격리 강도: 5-Fold SGKF vs Leave-One-Subject-Out(LOSO), 3개 시드 변동성
  D. 임계치 낙관 편향: 전체 OOF에 튜닝한 3-State F1 vs 피험자 단위 중첩(nested) 임계치 F1
  E. 시점별(view) AUC (AGENTS.md 다차원 감사)
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BL_DIR = r"D:\src\models\second_models\babel_lunge"
FEAT_DIR = r"D:\Data\피처_데이터"
OUT_JSON = os.path.join(BL_DIR, "bl_overfit_leakage_audit.json")


def _s(a):
    return np.array([int("".join(ch for ch in str(x) if ch.isdigit())) for x in a])


def load_all():
    ds = {}
    with open(os.path.join(FEAT_DIR, "bl1_v4_dataset_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    ds["bl1_v4(30D)"] = (d["X"], d["y"], _s(d["groups"]), np.array(d["views"]).astype(str))

    d = np.load(os.path.join(BL_DIR, "lunge_c2_features_v6_18d_delta_cache.npz"), allow_pickle=True)
    ds["bl2_v6(18D)"] = (d["X"], d["y"], _s(d["subjects"]), d["views"].astype(str))

    sys.path.insert(0, "d:/")
    from src.features.lunge_bl3_features_v5 import FEATURE_INDICES_15D
    d = np.load(os.path.join(FEAT_DIR, "lunge_bl3_dataset_19d_adv.npz"), allow_pickle=True)
    ds["bl3_v5(15D)"] = (d["X"][:, FEATURE_INDICES_15D], d["y"], d["groups"].astype(int), None)

    d = np.load(os.path.join(FEAT_DIR, "lunge_c4_dataset_cache_v3.npz"), allow_pickle=True)
    ds["bl4_v4b(24D)"] = (d["X_24d"], d["y"], _s(d["subjects"]), d["views"].astype(str))

    d = np.load(os.path.join(FEAT_DIR, "lunge_c5_v5_36d_features.npz"), allow_pickle=True)
    ds["bl5_v5(36D)"] = (d["X"], d["y"], d["subjects"].astype(int), d["views"].astype(str))
    return ds


def fit_predict(Xtr, ytr, Xva, seed=42):
    """대리 앙상블: LightGBM + CatBoost (고정 라운드, early stopping 미사용). Returns (p_val, p_train)."""
    spw = float(np.sum(ytr == 0)) / max(1, np.sum(ytr == 1))
    m1 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.04, max_depth=4, num_leaves=15, subsample=0.85,
                            colsample_bytree=0.8, scale_pos_weight=spw, random_state=seed, verbose=-1, n_jobs=4)
    m2 = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=4, auto_class_weights="Balanced",
                            random_seed=seed, verbose=0, thread_count=4)
    m1.fit(Xtr, ytr)
    m2.fit(Xtr, ytr)
    pv = 0.5 * m1.predict_proba(Xva)[:, 1] + 0.5 * m2.predict_proba(Xva)[:, 1]
    pt = 0.5 * m1.predict_proba(Xtr)[:, 1] + 0.5 * m2.predict_proba(Xtr)[:, 1]
    return pv, pt


def oof_sgkf(X, y, g, seed=42, shuffle_label=False):
    rng = np.random.RandomState(seed)
    yy = rng.permutation(y) if shuffle_label else y
    oof = np.zeros(len(y))
    tr_aucs = []
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in sg.split(X, yy, g):
        pv, pt = fit_predict(X[tr], yy[tr], X[va], seed)
        oof[va] = pv
        tr_aucs.append(roc_auc_score(yy[tr], pt))
    return oof, yy, float(np.mean(tr_aucs))


def oof_loso(X, y, g):
    oof = np.zeros(len(y))
    for tr, va in LeaveOneGroupOut().split(X, y, g):
        oof[va], _ = fit_predict(X[tr], y[tr], X[va])
    return oof


def best_thr(y, p, max_rej=0.10):
    """3-State 임계치: 보류율<=10% 제약 하 확정 샘플 결함 F1 최대화."""
    best, cfg = -1.0, (0.45, 0.55)
    for lo in np.linspace(0.20, 0.60, 41):
        for hi in np.linspace(0.30, 0.80, 51):
            if hi <= lo:
                continue
            dec = (p < lo) | (p >= hi)
            if 1 - dec.mean() > max_rej or dec.sum() == 0:
                continue
            f = f1_score(y[dec], (p[dec] >= hi).astype(int), zero_division=0)
            if f > best:
                best, cfg = f, (lo, hi)
    return best, cfg


def apply_thr(y, p, cfg):
    lo, hi = cfg
    dec = (p < lo) | (p >= hi)
    return dec, (p[dec] >= hi).astype(int)


def nested_threshold_f1(y, p, g):
    """피험자 단위: 나머지 8명 OOF로 임계치 선택 -> 해당 피험자에 적용 -> 전체 pooled F1."""
    dy, dp, tot_unc = [], [], 0
    for s in np.unique(g):
        m = g == s
        _, cfg = best_thr(y[~m], p[~m])
        dec, pred = apply_thr(y[m], p[m], cfg)
        dy.extend(y[m][dec])
        dp.extend(pred)
        tot_unc += int((~dec).sum())
    return float(f1_score(dy, dp, zero_division=0)), tot_unc / len(y)


def main():
    ds = load_all()
    report = {}
    for name, (X, y, g, v) in ds.items():
        print("=" * 78)
        print(f"[{name}] N={len(y)} D={X.shape[1]} defect={y.mean()*100:.1f}% subjects={np.unique(g).tolist()}")
        assert np.isfinite(X).all(), "NaN/Inf in X"
        r = {"N": int(len(y)), "D": int(X.shape[1])}

        # A + C(seed variability)
        aucs, f1s, tr_aucs = [], [], []
        for seed in (42, 7, 2024):
            oof, _, tra = oof_sgkf(X, y, g, seed)
            aucs.append(roc_auc_score(y, oof))
            f1s.append(f1_score(y, (oof >= 0.5).astype(int)))
            tr_aucs.append(tra)
            if seed == 42:
                oof42 = oof
        r["oof_auc_seeds"] = [float(a) for a in aucs]
        r["oof_auc_mean"] = float(np.mean(aucs))
        r["oof_auc_std"] = float(np.std(aucs))
        r["oof_f1_mean"] = float(np.mean(f1s))
        r["train_auc_mean"] = float(np.mean(tr_aucs))
        r["train_oof_auc_gap"] = float(np.mean(tr_aucs) - np.mean(aucs))
        print(f"  A) Train AUC {r['train_auc_mean']:.4f} vs OOF AUC {r['oof_auc_mean']:.4f} (+-{r['oof_auc_std']:.4f}) "
              f"-> gap {r['train_oof_auc_gap']:.4f} | OOF F1@0.5 {r['oof_f1_mean']:.4f}")

        # B permutation
        oofp, yp, _ = oof_sgkf(X, y, g, 42, shuffle_label=True)
        r["permuted_label_auc"] = float(roc_auc_score(yp, oofp))
        print(f"  B) Label-permutation AUC = {r['permuted_label_auc']:.4f} (기대값 ~0.50)")

        # C LOSO
        oofl = oof_loso(X, y, g)
        r["loso_auc"] = float(roc_auc_score(y, oofl))
        r["loso_f1"] = float(f1_score(y, (oofl >= 0.5).astype(int)))
        per_subj = {}
        for s in np.unique(g):
            m = g == s
            if len(np.unique(y[m])) > 1:
                per_subj[int(s)] = float(roc_auc_score(y[m], oofl[m]))
        r["loso_per_subject_auc"] = per_subj
        print(f"  C) LOSO AUC {r['loso_auc']:.4f} (5-Fold {r['oof_auc_mean']:.4f}) | 피험자별 AUC min/max "
              f"{min(per_subj.values()):.3f}/{max(per_subj.values()):.3f}")

        # D nested thresholds (on LOSO OOF and 5-fold OOF)
        opt_f1, cfg = best_thr(y, oof42)
        nest_f1, nest_unc = nested_threshold_f1(y, oof42, g)
        r["threshold_in_sample_f1"] = float(opt_f1)
        r["threshold_nested_f1"] = float(nest_f1)
        r["threshold_optimism"] = float(opt_f1 - nest_f1)
        r["nested_reject_rate"] = float(nest_unc)
        print(f"  D) 3-State F1 in-sample {opt_f1:.4f} vs nested {nest_f1:.4f} -> 낙관편향 {opt_f1-nest_f1:+.4f} "
              f"(nested 보류율 {nest_unc*100:.1f}%)")

        # E view AUC
        if v is not None:
            r["view_auc"] = {}
            for vv in sorted(np.unique(v)):
                m = v == vv
                a = roc_auc_score(y[m], oof42[m]) if len(np.unique(y[m])) > 1 else float("nan")
                r["view_auc"][vv] = float(a)
            print("  E) 시점별 AUC:", {k: round(x, 3) for k, x in r["view_auc"].items()})
        else:
            print("  E) 시점 정보 없음 (캐시에 views 미저장)")
        report[name] = r

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] saved {OUT_JSON}")


if __name__ == "__main__":
    main()

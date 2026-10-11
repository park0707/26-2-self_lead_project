# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Fine-tune v3 Feature Set & Architecture
목표: 3-State F1 극대화, 보류율 <= 10.0%, ROC-AUC 극대화
"""

import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

CACHE_PATH = r"D:\Data\피처_데이터\bl1_v2_dataset_cache.pkl"

with open(CACHE_PATH, "rb") as f:
    cache_data = pickle.load(f)

X = np.array(cache_data["X"], dtype=np.float32)
y = np.array(cache_data["y"], dtype=np.int32)
groups = np.array(cache_data.get("groups", cache_data.get("subjects")))
views = np.array(cache_data["views"])
all_names = cache_data.get("feature_names")

# 기본 필수 제거 (완전 중복 및 극단적 공선성/무관성)
MUST_REMOVE = [
    "view_invariant_flexion_3d",     # r=1.0000 with lead_knee_angle_min
    "hip_knee_flexion_coupling",     # Ablation F1 +3.58%p
    "kot_normalized_displacement",   # Ablation F1 +2.90%p
    "camera_view_azimuth_est",       # Ablation F1 +2.56%p
    "lead_knee_valgus_varus",        # Ablation F1 +2.49%p
    "lead_knee_rebound_jitter",      # r=0.9250 공선성
]

# 선택적 제거 후보들
OPTIONAL_REMOVE = [
    "descent_ascent_tempo_ratio",
    "pelvic_axial_rotation",
    "bottom_phase_dwell_ratio",
    "lead_knee_under_90_ratio",
    "pelvic_drop_angle",
    "kinetic_energy_inflection",
    "jerk_metric_lead_leg",
    "step_width_normalized"
]

print("=== Sequential Selection across Optional Candidates ===")

current_removed = list(MUST_REMOVE)
best_overall_f1 = 0
best_overall_pack = None

spw = float(np.sum(y == 0) / np.sum(y == 1))
sgkf = StratifiedGroupKFold(n_splits=5)

for opt_feat in [None] + OPTIONAL_REMOVE:
    test_removed = list(current_removed)
    if opt_feat is not None:
        test_removed.append(opt_feat)

    idx = [i for i, n in enumerate(all_names) if n not in test_removed]
    X_sub = X[:, idx]
    n_feats = X_sub.shape[1]

    oof_lgb = np.zeros(len(y), dtype=np.float32)
    oof_cat = np.zeros(len(y), dtype=np.float32)

    for tr, val in sgkf.split(X_sub, y, groups=groups):
        td = lgb.Dataset(X_sub[tr], label=y[tr])
        vd = lgb.Dataset(X_sub[val], label=y[val], reference=td)
        m_lgb = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.04,
                "max_depth": 5,
                "num_leaves": 23,
                "scale_pos_weight": spw,
                "colsample_bytree": 0.7,
                "reg_lambda": 3.0,
                "reg_alpha": 0.5,
                "min_child_samples": 20,
                "verbose": -1,
                "n_jobs": -1
            },
            td, num_boost_round=350, valid_sets=[vd],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        oof_lgb[val] = m_lgb.predict(X_sub[val], num_iteration=m_lgb.best_iteration)

        m_cat = CatBoostClassifier(
            iterations=400, learning_rate=0.04, depth=5,
            scale_pos_weight=spw, l2_leaf_reg=3.0, rsm=0.7,
            random_seed=42, verbose=False, early_stopping_rounds=40
        )
        m_cat.fit(X_sub[tr], y[tr], eval_set=(X_sub[val], y[val]), verbose=False)
        oof_cat[val] = m_cat.predict_proba(X_sub[val])[:, 1]

    oof_ens = 0.4 * oof_lgb + 0.6 * oof_cat
    auc = roc_auc_score(y, oof_ens)

    # 3-state
    best_f1, best_acc, best_unc = 0, 0, 0
    best_tc, best_td = 0, 0
    for tc in np.linspace(0.25, 0.45, 41):
        for td in np.linspace(tc + 0.05, 0.65, 41):
            unc = np.mean((oof_ens >= tc) & (oof_ens < td))
            if unc > 0.10:
                continue
            dec = (oof_ens < tc) | (oof_ens >= td)
            yp = (oof_ens[dec] >= td).astype(int)
            f1 = f1_score(y[dec], yp)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = accuracy_score(y[dec], yp)
                best_unc = unc
                best_tc, best_td = tc, td

    tag = f"Base + {opt_feat}" if opt_feat else "Base (6 removed)"
    print(f"[{n_feats}D] {tag:<35} -> 3-State F1: {best_f1*100:.2f}% | ACC: {best_acc*100:.2f}% | AUC: {auc*100:.2f}% | Unc: {best_unc*100:.2f}%")

    if best_f1 > best_overall_f1:
        best_overall_f1 = best_f1
        best_overall_pack = {
            "n_feats": n_feats,
            "removed": test_removed,
            "f1": best_f1,
            "acc": best_acc,
            "auc": auc,
            "unc": best_unc,
            "tc": best_tc,
            "td": best_td
        }

print("\n" + "=" * 60)
print(f"BEST FOUND: {best_overall_pack['n_feats']}D with F1={best_overall_pack['f1']*100:.2f}%, ACC={best_overall_pack['acc']*100:.2f}%, AUC={best_overall_pack['auc']*100:.2f}%")
print(f"Removed features: {best_overall_pack['removed']}")

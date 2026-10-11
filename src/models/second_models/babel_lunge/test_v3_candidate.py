# -*- coding: utf-8 -*-
"""
검증 스크립트: 30D 정밀 최적 피처셋 + 최적화 모델 구조 평가
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
all_feature_names = cache_data.get("feature_names")

# 8개 제거 대상:
# 1. view_invariant_flexion_3d (r=1.0000 with lead_knee_angle_min)
# 2. hip_knee_flexion_coupling (Ablation F1 +3.58%p 저해)
# 3. kot_normalized_displacement (Ablation F1 +2.90%p 저해)
# 4. camera_view_azimuth_est (Ablation F1 +2.56%p 저해)
# 5. lead_knee_valgus_varus (Ablation F1 +2.49%p 저해)
# 6. lead_knee_rebound_jitter (r=0.9250 극단 공선성)
# 7. descent_ascent_tempo_ratio (Ablation F1 +1.34%p, AUC +0.15%p 저해)
# 8. pelvic_axial_rotation (Ablation F1 +1.35%p 저해)

REMOVED_FEATURES = [
    "view_invariant_flexion_3d",
    "hip_knee_flexion_coupling",
    "kot_normalized_displacement",
    "camera_view_azimuth_est",
    "lead_knee_valgus_varus",
    "lead_knee_rebound_jitter",
    "descent_ascent_tempo_ratio",
    "pelvic_axial_rotation"
]

v3_indices = [i for i, name in enumerate(all_feature_names) if name not in REMOVED_FEATURES]
v3_feature_names = [all_feature_names[i] for i in v3_indices]
print(f"[*] v3 최적 피처 차원: {len(v3_feature_names)}D (제거된 피처: {len(REMOVED_FEATURES)}개)")

X_v3 = X[:, v3_indices]
n_samples = len(y)
sgkf = StratifiedGroupKFold(n_splits=5)

neg_cnt = np.sum(y == 0)
pos_cnt = np.sum(y == 1)
spw = float(neg_cnt / pos_cnt)

# 최적화된 모델 구조 하이퍼파라미터
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 23,
    "max_depth": 5,
    "scale_pos_weight": spw,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.04,
    "max_depth": 5,
    "scale_pos_weight": spw,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "min_child_weight": 2,
    "random_state": 42,
    "n_jobs": -1
}

cat_params = {
    "iterations": 400,
    "learning_rate": 0.04,
    "depth": 5,
    "scale_pos_weight": spw,
    "l2_leaf_reg": 3.0,
    "rsm": 0.7,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 40
}

oof_lgb = np.zeros(n_samples, dtype=np.float32)
oof_cat = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_v3, y, groups=groups)):
    X_train, y_train = X_v3[train_idx], y[train_idx]
    X_val, y_val = X_v3[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model_lgb = lgb.train(
        lgb_params, train_data, num_boost_round=350, valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)

    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model_xgb = xgb.train(
        xgb_params, dtrain, num_boost_round=350, evals=[(dval, "val")],
        early_stopping_rounds=35, verbose_eval=False
    )
    oof_xgb[val_idx] = model_xgb.predict(dval)

# 앙상블
def loss_fn(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    if np.sum(w) == 0:
        return 10.0
    w /= np.sum(w)
    pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return -roc_auc_score(y, pred)

res = minimize(loss_fn, [0.35, 0.45, 0.20], method="Nelder-Mead")
opt_w = np.maximum(res.x, 0)
opt_w /= np.sum(opt_w)

oof_ens = opt_w[0] * oof_lgb + opt_w[1] * oof_cat + opt_w[2] * oof_xgb
auc_ens = roc_auc_score(y, oof_ens)
f1_bin = f1_score(y, (oof_ens >= 0.5).astype(int))

print(f"\n[+] v3 베이스라인 이진 F1: {f1_bin*100:.2f}% | ROC-AUC: {auc_ens*100:.2f}%")
print(f"[+] 최적 앙상블 가중치: LGB={opt_w[0]:.3f}, CAT={opt_w[1]:.3f}, XGB={opt_w[2]:.3f}")

# 3-State 최적화
best_3s = None
best_f1_3s = -1.0
for tau_c in np.linspace(0.25, 0.45, 41):
    for tau_d in np.linspace(tau_c + 0.05, 0.65, 41):
        mask_clean = oof_ens < tau_c
        mask_defect = oof_ens >= tau_d
        mask_unc = (~mask_clean) & (~mask_defect)
        unc_rate = np.mean(mask_unc)
        if unc_rate > 0.10:
            continue
        dec_mask = ~mask_unc
        if np.sum(dec_mask) == 0:
            continue
        y_true_dec = y[dec_mask]
        y_pred_dec = np.where(oof_ens[dec_mask] >= tau_d, 1, 0)
        f1_3s = f1_score(y_true_dec, y_pred_dec, zero_division=0)
        acc_3s = accuracy_score(y_true_dec, y_pred_dec)
        if f1_3s > best_f1_3s:
            best_f1_3s = f1_3s
            best_3s = {
                "3state_f1": f1_3s,
                "3state_acc": acc_3s,
                "uncertainty_rate": unc_rate,
                "tau_clean": tau_c,
                "tau_defect": tau_d,
                "prec": precision_score(y_true_dec, y_pred_dec, zero_division=0),
                "recall": recall_score(y_true_dec, y_pred_dec, zero_division=0)
            }

print(f"[+] v3 3-State F1: {best_3s['3state_f1']*100:.2f}% | 정확도: {best_3s['3state_acc']*100:.2f}% | 보류율: {best_3s['uncertainty_rate']*100:.2f}% (tau_clean={best_3s['tau_clean']:.3f}, tau_defect={best_3s['tau_defect']:.3f})")

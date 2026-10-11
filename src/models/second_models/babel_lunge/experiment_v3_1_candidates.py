# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Experiment: v3.1 후보 구조 및 권고안 적용 토너먼트
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from scipy.optimize import minimize

NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
V3_CACHE_PATH = r"D:\Data\피처_데이터\bl1_v3_dataset_cache.pkl"

sys.path.append(r"D:\src\features")
from lunge_c1_features_v3 import (
    detect_lead_leg_robust,
    find_dual_anchor_inflection_point,
    safe_savgol_filter,
    vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)

# 캐시 로드
with open(V3_CACHE_PATH, "rb") as f:
    v3_cache = pickle.load(f)

X_v3 = v3_cache["X"]
y = v3_cache["y"]
groups = v3_cache.get("groups", v3_cache.get("subjects"))
views = v3_cache["views"]
clip_names = v3_cache["clip_names"]
feature_names_v3 = v3_cache["feature_names"]

n_samples = len(y)
print(f"[*] 캐시 로드 완료: N={n_samples}, D={X_v3.shape[1]}")

feat_femur_dy = np.zeros(n_samples, dtype=np.float32)
feat_front_geo = np.zeros(n_samples, dtype=np.float32)
feat_is_view3 = np.zeros(n_samples, dtype=np.float32)

print("[*] 1,081개 시퀀스에서 권고 피처 및 정면 기하 계산 중...")
for idx, cname in enumerate(clip_names):
    view_code = views[idx]
    npz_path = os.path.join(NPZ_DIR, f"{cname}_{view_code}.npz")
    data = np.load(npz_path)
    kps_64f = data["kps_33_3d_64f"]

    lead_side = detect_lead_leg_robust(kps_64f)
    hip_idx = MP_LEFT_HIP if lead_side == "left" else MP_RIGHT_HIP
    knee_idx = MP_LEFT_KNEE if lead_side == "left" else MP_RIGHT_KNEE
    ankle_idx = MP_LEFT_ANKLE if lead_side == "left" else MP_RIGHT_ANKLE

    lead_angles = np.zeros(64, dtype=np.float32)
    for f in range(64):
        v_fem = kps_64f[f, hip_idx] - kps_64f[f, knee_idx]
        v_tib = kps_64f[f, ankle_idx] - kps_64f[f, knee_idx]
        lead_angles[f] = vector_angle_3d(v_fem, v_tib)
    angles_sg = safe_savgol_filter(lead_angles, window=7, poly=2)
    inflection_idx, _, _ = find_dual_anchor_inflection_point(kps_64f, angles_sg)

    hip_pt = kps_64f[inflection_idx, hip_idx]
    knee_pt = kps_64f[inflection_idx, knee_idx]

    femur_len = np.linalg.norm(hip_pt - knee_pt)
    if femur_len < 1e-4:
        femur_len = 1.0

    delta_y = abs(knee_pt[1] - hip_pt[1])
    drop_femur = float(delta_y / femur_len)
    feat_femur_dy[idx] = drop_femur

    # 관상면 정면성 지표 (순수 3D 키포인트 기반: 골반 회전각 Yaw)
    pelvis_vec = kps_64f[inflection_idx, MP_RIGHT_HIP] - kps_64f[inflection_idx, MP_LEFT_HIP]
    # xz 평면 상 yaw 각도: 정면(view3)일 때 yaw ≈ 0 -> cos(yaw) ≈ 1
    yaw = np.arctan2(abs(pelvis_vec[2]), abs(pelvis_vec[0]) + 1e-6)
    front_confidence = float(np.cos(yaw))  # 0.0 ~ 1.0
    feat_front_geo[idx] = drop_femur * front_confidence

    feat_is_view3[idx] = 1.0 if view_code == 'view3' else 0.0

feat_femur_dy = feat_femur_dy.reshape(-1, 1)
feat_front_geo = feat_front_geo.reshape(-1, 1)
feat_front_tag = (feat_femur_dy * feat_is_view3.reshape(-1, 1)).astype(np.float32)

print("[*] 피처 엔지니어링 완료!")


def run_5fold(X_mat, name, tune_reg=False):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(n_samples, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_mat, y, groups=groups)):
        X_tr, y_tr = X_mat[train_idx], y[train_idx]
        X_va, y_va = X_mat[val_idx], y[val_idx]

        scale_pos = float(np.sum(y_tr == 0)) / max(1, np.sum(y_tr == 1))

        colsample = 0.65 if tune_reg else 0.70
        subsample = 0.80 if tune_reg else 0.85
        reg_l1 = 0.8 if tune_reg else 0.5
        reg_l2 = 4.0 if tune_reg else 3.0

        m_lgb = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=24,
            subsample=subsample, colsample_bytree=colsample,
            reg_alpha=reg_l1, reg_lambda=reg_l2, scale_pos_weight=scale_pos,
            random_state=42 + fold, verbose=-1
        )
        m_cb = CatBoostClassifier(
            iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=reg_l2,
            scale_pos_weight=scale_pos, random_seed=42 + fold, verbose=0
        )
        m_xgb = xgb.XGBClassifier(
            n_estimators=160, learning_rate=0.035, max_depth=5,
            subsample=subsample, colsample_bytree=colsample,
            reg_alpha=reg_l1, reg_lambda=reg_l2, scale_pos_weight=scale_pos,
            random_state=42 + fold, eval_metric="logloss"
        )

        m_lgb.fit(X_tr, y_tr)
        m_cb.fit(X_tr, y_tr)
        m_xgb.fit(X_tr, y_tr)

        p_lgb = m_lgb.predict_proba(X_va)[:, 1]
        p_cb = m_cb.predict_proba(X_va)[:, 1]
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        p_ens = 0.40 * p_lgb + 0.35 * p_cb + 0.25 * p_xgb
        oof_preds[val_idx] = p_ens

    auc = roc_auc_score(y, oof_preds)

    best_f1, best_acc, best_tau1, best_tau2, best_unc = 0.0, 0.0, 0.4, 0.6, 0.0
    for tau1 in np.linspace(0.25, 0.45, 41):
        for tau2 in np.linspace(tau1 + 0.05, 0.75, 41):
            unc_mask = (oof_preds >= tau1) & (oof_preds < tau2)
            unc_rate = np.mean(unc_mask)
            if unc_rate > 0.10:
                continue
            det_mask = ~unc_mask
            if np.sum(det_mask) == 0:
                continue
            y_det = y[det_mask]
            p_det = (oof_preds[det_mask] >= tau2).astype(int)
            f1 = f1_score(y_det, p_det, average='macro', zero_division=0)
            acc = accuracy_score(y_det, p_det)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_tau1 = tau1
                best_tau2 = tau2
                best_unc = unc_rate

    # view3 전용 AUC
    v3_mask = np.array([v == 'view3' for v in views])
    auc_v3 = roc_auc_score(y[v3_mask], oof_preds[v3_mask])

    print(f"=== {name} ===")
    print(f"  전체 AUC: {auc*100:.2f}% | 정면(view3) AUC: {auc_v3*100:.2f}%")
    print(f"  3-State F1: {best_f1*100:.2f}% | 정확도: {best_acc*100:.2f}% | 보류율: {best_unc*100:.2f}% (tau1={best_tau1:.3f}, tau2={best_tau2:.3f})")
    return {
        "name": name,
        "auc": auc,
        "auc_v3": auc_v3,
        "f1": best_f1,
        "acc": best_acc,
        "unc": best_unc,
        "tau1": best_tau1,
        "tau2": best_tau2,
        "oof_preds": oof_preds
    }


# 1. Baseline v3.0 (26D)
r_v30 = run_5fold(X_v3, "Baseline v3.0 (26D)")

# 2. 권고 수식 단독 추가: 대퇴골 정규화 수직거리 (27D)
X_27D = np.hstack([X_v3, feat_femur_dy])
r_27D = run_5fold(X_27D, "후보 1: v3 + 대퇴골 정규화 수직거리 (27D)")

# 3. 권고 수식 + 관상면 정면 지향도 곱연산 게이팅 (28D)
X_28D_geo = np.hstack([X_v3, feat_femur_dy, feat_front_geo])
r_28D_geo = run_5fold(X_28D_geo, "후보 2: v3 + 대퇴골 정규화 + 관상면 정면 게이팅 (28D)")

# 4. 권고 수식 + 시점 태그 상호작용 (28D)
X_28D_tag = np.hstack([X_v3, feat_femur_dy, feat_front_tag])
r_28D_tag = run_5fold(X_28D_tag, "후보 3: v3 + 대퇴골 정규화 + 정면 시점 태그 상호작용 (28D)")

# 5. 후보 2에 규제화 튜닝 적용 (28D)
r_28D_geo_reg = run_5fold(X_28D_geo, "후보 4: 후보 2 + L1/L2 복합 정밀 규제화 (28D)", tune_reg=True)

# 6. 후보 3에 규제화 튜닝 적용 (28D)
r_28D_tag_reg = run_5fold(X_28D_tag, "후보 5: 후보 3 + L1/L2 복합 정밀 규제화 (28D)", tune_reg=True)

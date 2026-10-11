# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tournament_v4_features.py
Purpose: v4 고도화 후보 피처(PACS, BLC-DR, FFR, DPD, BCR, morph_ratio 등) 추출 및 5-Fold 실측 성능 토너먼트
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from scipy.optimize import minimize

NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
V3_1_CACHE = r"D:\Data\피처_데이터\bl1_v3_1_dataset_cache.pkl"

sys.path.append(r"D:\src\features")
from lunge_c1_features_v3_1 import (
    detect_lead_leg_robust,
    find_dual_anchor_inflection_point,
    safe_savgol_filter,
    vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)

# 1. v3.1 캐시 로드
with open(V3_1_CACHE, "rb") as f:
    v3_1_data = pickle.load(f)

X_v3_1 = v3_1_data["X"]
y = v3_1_data["y"]
groups = v3_1_data["groups"]
views = v3_1_data["views"]
clip_names = v3_1_data["clip_names"]
v3_1_feature_names = v3_1_data["feature_names"]

n_samples = len(y)
print(f"[*] v3.1 데이터 로드 완료: N={n_samples}, D={X_v3_1.shape[1]}")

# 2. 신규 연구 피처 7종 추출
# Feat 1: ffr (Femur Foreshortening Ratio: 2D 대퇴골 투영 압축비)
# Feat 2: angle_blc (Bone-length Constrained 3D 무릎 굴곡각)
# Feat 3: angle_pacs (PACS 골반 정규직교 기저계 시상면 2D 굴곡각)
# Feat 4: dpd (Dimensionless Pelvis Drop: 초기 직립 대비 골반 무차원 강하량)
# Feat 5: bcr (Biomechanical Consistency Residual: 이론적 강하각 대비 실제 각도 잔차)
# Feat 6: coplanarity (하지 평면 법선과 골반 횡축 직교 정합도)
# Feat 7: morph_ratio (대퇴골-정강이 골격비: L_femur / L_tibia)

cand_ffr = np.zeros(n_samples, dtype=np.float32)
cand_blc = np.zeros(n_samples, dtype=np.float32)
cand_pacs = np.zeros(n_samples, dtype=np.float32)
cand_dpd = np.zeros(n_samples, dtype=np.float32)
cand_bcr = np.zeros(n_samples, dtype=np.float32)
cand_coplanar = np.zeros(n_samples, dtype=np.float32)
cand_morph = np.zeros(n_samples, dtype=np.float32)

print("[*] 1,081개 시퀀스에서 선행연구 기반 7대 고도화 피처 추출 중...")
eps = 1e-6

for idx, cname in enumerate(clip_names):
    v_code = views[idx]
    npz_path = os.path.join(NPZ_DIR, f"{cname}_{v_code}.npz")
    data = np.load(npz_path)
    kps = data["kps_33_3d_64f"]  # (64, 33, 3)

    lead_side = detect_lead_leg_robust(kps)
    hip_idx = MP_LEFT_HIP if lead_side == "left" else MP_RIGHT_HIP
    knee_idx = MP_LEFT_KNEE if lead_side == "left" else MP_RIGHT_KNEE
    ankle_idx = MP_LEFT_ANKLE if lead_side == "left" else MP_RIGHT_ANKLE

    # 초기 5프레임 캘리브레이션
    init_f = kps[:5]
    f_lens = np.linalg.norm(init_f[:, knee_idx] - init_f[:, hip_idx], axis=-1)
    t_lens = np.linalg.norm(init_f[:, ankle_idx] - init_f[:, knee_idx], axis=-1)
    L_f = float(np.median(f_lens))
    L_t = float(np.median(t_lens))
    L_f = max(L_f, 0.1)
    L_t = max(L_t, 0.1)
    L_leg = L_f + L_t
    morph = float(L_f / (L_t + eps))
    cand_morph[idx] = morph

    # 골반 초기 Y (직립 기준)
    pelvis_init_y = float(np.median((init_f[:, MP_LEFT_HIP, 1] + init_f[:, MP_RIGHT_HIP, 1]) / 2.0))

    # 변곡점 (최저점)
    pelvis_y_traj = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    lead_angles = np.zeros(64, dtype=np.float32)
    for f in range(64):
        vf = kps[f, hip_idx] - kps[f, knee_idx]
        vt = kps[f, ankle_idx] - kps[f, knee_idx]
        lead_angles[f] = vector_angle_3d(vf, vt)
    lead_sg = safe_savgol_filter(lead_angles, 7, 2)
    inflect_idx, _, _ = find_dual_anchor_inflection_point(kps, lead_sg)

    inf_kps = kps[inflect_idx]

    # 1. FFR: 대퇴골 2D 투영 압축비
    femur_2d_len = float(np.linalg.norm(inf_kps[knee_idx, :2] - inf_kps[hip_idx, :2]))
    ffr = float(femur_2d_len / (L_f + eps))
    cand_ffr[idx] = ffr

    # 2. BLC-DR: 강체 구속 피타고라스 역투영 무릎 각도
    dz_sq = max(0.0, L_f**2 - femur_2d_len**2)
    dz_rec = -np.sqrt(dz_sq)
    dxy = inf_kps[knee_idx, :2] - inf_kps[hip_idx, :2]
    rec_femur_vec = np.array([dxy[0], dxy[1], dz_rec])
    tib_vec = inf_kps[ankle_idx] - inf_kps[knee_idx]
    cos_blc = np.dot(-rec_femur_vec, tib_vec) / (np.linalg.norm(rec_femur_vec) * np.linalg.norm(tib_vec) + eps)
    cand_blc[idx] = float(np.degrees(np.arccos(np.clip(cos_blc, -1.0, 1.0))))

    # 3. PACS 시상면 2D 각도
    l_hip = inf_kps[MP_LEFT_HIP]
    r_hip = inf_kps[MP_RIGHT_HIP]
    origin = (l_hip + r_hip) / 2.0
    mid_sho = (inf_kps[MP_LEFT_SHOULDER] + inf_kps[MP_RIGHT_SHOULDER]) / 2.0
    u_x = (r_hip - l_hip) / (np.linalg.norm(r_hip - l_hip) + eps)
    v_spine = mid_sho - origin
    proj_x = np.dot(v_spine, u_x) * u_x
    v_spine_orth = v_spine - proj_x
    u_y = v_spine_orth / (np.linalg.norm(v_spine_orth) + eps)
    u_z = np.cross(u_x, u_y)
    u_z /= (np.linalg.norm(u_z) + eps)
    R = np.stack([u_x, u_y, u_z], axis=1)

    pacs_kps = np.dot(inf_kps - origin, R)
    pacs_vf = pacs_kps[knee_idx] - pacs_kps[hip_idx]
    pacs_vt = pacs_kps[ankle_idx] - pacs_kps[knee_idx]
    # 시상면 (Y, Z 평면) 투영
    pacs_vf_sag = pacs_vf[1:3]
    pacs_vt_sag = pacs_vt[1:3]
    cos_pacs = np.dot(-pacs_vf_sag, pacs_vt_sag) / (np.linalg.norm(pacs_vf_sag) * np.linalg.norm(pacs_vt_sag) + eps)
    cand_pacs[idx] = float(np.degrees(np.arccos(np.clip(cos_pacs, -1.0, 1.0))))

    # 4. DPD: 무차원 골반 강하량
    dpd = float((pelvis_y_traj[inflect_idx] - pelvis_init_y) / (L_f + eps))
    cand_dpd[idx] = dpd

    # 5. BCR: 생체역학 각도-깊이 잔차
    # 골반이 1.0 * L_f 만큼 내려갈 때 이상적인 90도 각도 형성
    ideal_angle = 180.0 - 90.0 * np.clip(dpd, 0.0, 1.2)
    cand_bcr[idx] = float(cand_blc[idx] - ideal_angle)

    # 6. Coplanarity: 하지 평면 법선과 골반 횡축 정합도
    v_f = inf_kps[knee_idx] - inf_kps[hip_idx]
    v_t = inf_kps[ankle_idx] - inf_kps[knee_idx]
    leg_norm = np.cross(v_f, v_t)
    leg_norm /= (np.linalg.norm(leg_norm) + eps)
    u_hip = (r_hip - l_hip) / (np.linalg.norm(r_hip - l_hip) + eps)
    cand_coplanar[idx] = float(abs(np.dot(leg_norm, u_hip)))

print("[+] 7대 신규 피처 추출 완료!")

cand_features_dict = {
    "ffr": cand_ffr.reshape(-1, 1),
    "angle_blc": cand_blc.reshape(-1, 1),
    "angle_pacs": cand_pacs.reshape(-1, 1),
    "dpd": cand_dpd.reshape(-1, 1),
    "bcr": cand_bcr.reshape(-1, 1),
    "coplanar": cand_coplanar.reshape(-1, 1),
    "morph_ratio": cand_morph.reshape(-1, 1),
}


def evaluate_feature_combination(X_input, name):
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ens = np.zeros(n_samples, dtype=np.float32)

    spw = float(np.sum(y == 0) / np.sum(y == 1))

    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'learning_rate': 0.04, 'num_leaves': 23, 'max_depth': 5, 'scale_pos_weight': spw,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 3.0, 'reg_alpha': 0.5,
        'min_child_samples': 20, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
    }
    cat_params = {
        'iterations': 400, 'learning_rate': 0.04, 'depth': 5, 'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0, 'rsm': 0.7, 'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 40
    }
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'learning_rate': 0.04,
        'max_depth': 5, 'scale_pos_weight': spw, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_lambda': 3.0, 'reg_alpha': 0.5, 'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
    }

    w = [0.343, 0.450, 0.207]

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_input, y, groups=groups)):
        X_tr, y_tr = X_input[train_idx], y[train_idx]
        X_va, y_va = X_input[val_idx], y[val_idx]

        m_lgb = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=350, valid_sets=[lgb.Dataset(X_va, label=y_va)], callbacks=[lgb.early_stopping(35, verbose=False)])
        p_lgb = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
        p_xgb = m_xgb.predict(dval)

        oof_ens[val_idx] = w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb

    auc = roc_auc_score(y, oof_ens)

    # 3-State 그리드 서치
    best_f1, best_acc, best_tau1, best_tau2, best_unc = 0.0, 0.0, 0.4, 0.6, 0.0
    for tau1 in np.linspace(0.25, 0.45, 41):
        for tau2 in np.linspace(0.45, 0.65, 41):
            if tau2 <= tau1:
                continue
            unc = (oof_ens >= tau1) & (oof_ens < tau2)
            unc_rate = np.mean(unc)
            if unc_rate > 0.10:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            f1 = f1_score(y[dec], (oof_ens[dec] >= tau2).astype(int), zero_division=0)
            acc = accuracy_score(y[dec], (oof_ens[dec] >= tau2).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_tau1 = tau1
                best_tau2 = tau2
                best_unc = unc_rate

    # 피험자2 특화 성능
    p2_mask = (groups == '피험자2')
    auc_p2 = roc_auc_score(y[p2_mask], oof_ens[p2_mask])
    unc_p2 = (oof_ens[p2_mask] >= best_tau1) & (oof_ens[p2_mask] < best_tau2)
    dec_p2 = ~unc_p2
    f1_p2 = f1_score(y[p2_mask][dec_p2], (oof_ens[p2_mask][dec_p2] >= best_tau2).astype(int), zero_division=0)

    # 정면 view3 특화 성능
    v3_mask = (views == 'view3')
    auc_v3 = roc_auc_score(y[v3_mask], oof_ens[v3_mask])
    unc_v3 = (oof_ens[v3_mask] >= best_tau1) & (oof_ens[v3_mask] < best_tau2)
    dec_v3 = ~unc_v3
    f1_v3 = f1_score(y[v3_mask][dec_v3], (oof_ens[v3_mask][dec_v3] >= best_tau2).astype(int), zero_division=0)

    print(f"[{name}]")
    print(f"  전체 AUC: {auc*100:.2f}% | 3-State F1: {best_f1*100:.2f}% | 정확도: {best_acc*100:.2f}% | 보류율: {best_unc*100:.2f}%")
    print(f"  피험자2: AUC={auc_p2*100:.2f}%, F1={f1_p2*100:.2f}% | 정면(view3): AUC={auc_v3*100:.2f}%, F1={f1_v3*100:.2f}%")

    return {
        "name": name, "auc": auc, "f1": best_f1, "acc": best_acc, "unc": best_unc,
        "auc_p2": auc_p2, "f1_p2": f1_p2, "auc_v3": auc_v3, "f1_v3": f1_v3
    }


# 1. Baseline v3.1 (28D)
res_base = evaluate_feature_combination(X_v3_1, "Baseline v3.1 (28D)")

# 2. 토너먼트: 각 피처 개별 추가 (29D)
individual_results = []
for fname, fvec in cand_features_dict.items():
    X_cand = np.hstack([X_v3_1, fvec])
    res = evaluate_feature_combination(X_cand, f"v3.1 + {fname} (29D)")
    individual_results.append(res)

# 3. 우수 피처 전수 결합 (v4 Candidate)
# PACS(시상면 회전각) + FFR(대퇴골 투영비) + DPD(무차원 강하량) + BCR(일관성 잔차) + morph_ratio(체형비)
top_feats = [cand_features_dict["ffr"], cand_features_dict["angle_pacs"], cand_features_dict["dpd"], cand_features_dict["bcr"], cand_features_dict["morph_ratio"]]
X_top = np.hstack([X_v3_1] + top_feats)
res_top = evaluate_feature_combination(X_top, "v4 후보: v3.1 + Top 5 (33D)")

# 4. 전체 7개 신규 피처 결합 (35D)
X_all = np.hstack([X_v3_1] + list(cand_features_dict.values()))
res_all = evaluate_feature_combination(X_all, "v4 후보: v3.1 + All 7 (35D)")

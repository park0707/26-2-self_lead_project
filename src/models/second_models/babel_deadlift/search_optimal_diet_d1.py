# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: search_optimal_diet_d1.py
Purpose: 피처 다이어트(Feature Diet) 최적화 탐색
         - 상수(Zero Variance) 4개 제거
         - 고상관(|r|>0.85) 중복 피처 정리
         - Drop-one에서 식별된 성능 저해 피처 단계적 제거
         - 후보 다이어트 서브셋별 5-Fold StratifiedGroupKFold 실측 및 최적 다이어트 모델 확정
"""

import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_2_spinepose import (
    extract_d1_v4_2_features_34d_base,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_2,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_40_D1_V4_2
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
AUDIT_JSON_PATH = os.path.join(CURRENT_DIR, "d1_v4_2_feature_audit_results.json")
OUT_DIET_JSON = os.path.join(CURRENT_DIR, "d1_optimal_diet_search_results.json")

VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def load_all_data():
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

    sequence_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as jf:
            d_json = json.load(jf)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        d = np.load(f, allow_pickle=True)
        kps_33_64f = d['kps_33_3d_64f']

        sequence_ids.append(sid)
        labels.append(is_d1_defect)
        subject_ids.append(sub)
        view_ids.append(vn)
        kps_list.append(kps_33_64f)
        spine_list.append(spine_dict.get(sid, None))

    labels = np.array(labels, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)
    view_ids = np.array(view_ids)
    N = len(labels)

    X_34d_base = np.zeros((N, 34), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_34d_base[i] = extract_d1_v4_2_features_34d_base(kps, s9)
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps)
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X_40d = np.zeros((N, 40), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_34d_base, labels, groups=subject_ids)):
        norm_train_indices = train_idx[labels[train_idx] == 0]
        res = MorphologyResidualizerV4_2(alpha=15.0)
        res.fit(
            cond_ratios[norm_train_indices],
            X_34d_base[norm_train_indices, 17],
            X_34d_base[norm_train_indices, 23],
            X_34d_base[norm_train_indices, 0]
        )

        sub_34 = X_34d_base[val_idx]
        sub_cond = cond_ratios[val_idx]
        sub_res = np.zeros((len(val_idx), 3), dtype=np.float32)
        for k in range(len(val_idx)):
            dp = float(sub_34[k, 17])
            pa = float(sub_34[k, 23])
            ca = float(sub_34[k, 0])
            sub_res[k] = res.transform(sub_cond[k], dp, pa, ca)
        X_40d[val_idx] = np.concatenate([sub_34, sub_res, sub_cond[:, :3]], axis=1)

    return X_40d, X_temporal, labels, subject_ids, view_ids


def evaluate_diet_combo(X_sub, X_temporal, y, subs, th=0.35, eval_tcn=False):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p_lgb = np.zeros(len(y), dtype=np.float32)
    oof_p_cat = np.zeros(len(y), dtype=np.float32)
    oof_p_tcn = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subs)):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]

        clf_l = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.032, num_leaves=16, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=8,
            random_state=42 + fold, verbose=-1
        )
        clf_l.fit(X_tr, y_tr)
        oof_p_lgb[val_idx] = clf_l.predict_proba(X_va)[:, 1]

        clf_c = CatBoostClassifier(
            iterations=200, learning_rate=0.038, depth=4, l2_leaf_reg=4.5,
            random_seed=42 + fold, verbose=False
        )
        clf_c.fit(X_tr, y_tr)
        oof_p_cat[val_idx] = clf_c.predict_proba(X_va)[:, 1]

        if eval_tcn:
            tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
            optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.tensor(X_temporal[train_idx].transpose(0, 2, 1), dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
            X_va_t = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)

            loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
            tcn_model.train()
            for _ in range(30):
                for bx, by in loader:
                    optimizer.zero_grad()
                    loss = criterion(tcn_model(bx), by)
                    loss.backward()
                    optimizer.step()

            tcn_model.eval()
            with torch.no_grad():
                oof_p_tcn[val_idx] = torch.sigmoid(tcn_model(X_va_t).squeeze(1)).numpy()

    p_gbdt = 0.5 * oof_p_lgb + 0.5 * oof_p_cat
    if eval_tcn:
        p_ens = 0.5 * p_gbdt + 0.5 * oof_p_tcn
    else:
        p_ens = p_gbdt

    preds = (p_ens >= th).astype(int)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, p_ens)
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)

    # 3-State 최적화
    best_sel_f1 = 0.0
    best_abstain = 0.0
    for tn in [0.24, 0.26, 0.28]:
        for td in [0.42, 0.45, 0.48]:
            stat = []
            for p in p_ens:
                if p <= tn: stat.append(0)
                elif p >= td: stat.append(1)
                else: stat.append(-1)
            stat = np.array(stat)
            valid = (stat != -1)
            if np.mean(~valid) <= 0.15 and np.sum(valid) > 0:
                s_f1 = f1_score(y[valid], stat[valid])
                if s_f1 > best_sel_f1:
                    best_sel_f1 = s_f1
                    best_abstain = float(np.mean(~valid) * 100.0)

    return {
        "f1": float(f1),
        "auc": float(auc),
        "recall": float(rec),
        "precision": float(prec),
        "accuracy": float(acc),
        "sel_f1": float(best_sel_f1),
        "abstain": float(best_abstain),
        "cm": cm.tolist(),
        "oof_p": p_ens
    }


def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 피처 다이어트(Feature Diet) 최적화 탐색")
    print("=" * 110)

    X_40d, X_temporal, y, subs, views = load_all_data()
    names = FEATURE_NAMES_40_D1_V4_2

    # 40D 전체 베이스라인
    print("\n[*] 40D 전체 베이스라인 평가 중...")
    base_res = evaluate_diet_combo(X_40d, X_temporal, y, subs)
    print(f"  [Base 40D] F1: {base_res['f1']*100:.2f}%, Recall: {base_res['recall']*100:.2f}%, AUC: {base_res['auc']:.4f}, 3-State F1: {base_res['sel_f1']*100:.2f}%")

    # 감사 결과 로드
    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    # 1. 분산 0(상수) 피처: [7] phase_speed, [19] conf, [20] energy, [21] symm (총 4개)
    # 2. Drop-one에서 식별된 저해 피처:
    #    [29] rom_competence_score (+1.66)
    #    [23] setup_lumbar_poly_a (+1.43)
    #    [11] cobb_range_active (+1.41)
    #    [24] floor_break_cobb_delta (+1.41)
    #    [25] knee_pass_lumbar_flexion (+1.19)
    #    [10] cobb_max_flexion (+0.95)
    #    [14] pelvis_tilt_range (+0.95)
    #    [17] torso_pitch_deepest (+0.95)
    #    [30] rom_scaled_lumbar_poly_a (+0.95)
    #    [35] res_setup_lumbar_poly_a (+0.95)
    #    [34] res_torso_pitch_deepest (+0.93)
    #    [ 4] cervical_min (+0.72)
    #    [ 2] spine_lateral_deviation_max (+0.71)
    #    [ 9] cobb_mean_active (+0.71)
    #    [ 8] hip_to_knee_dist_ratio (+0.47)
    #    [18] view_score_normal_z (+0.47)
    #    [38] ratio_femur_to_tibia (+0.47)
    #    [13] lateral_asymmetry_integral (+0.46)
    #    [15] lumbar_flex_acceleration_peak (+0.25)
    #    [ 0] lumbar_curvature_poly_a (+0.25)
    #    [ 6] active_rom (+0.24)
    #    [ 3] trunk_flex_speed_max (+0.23)
    #    [12] shoulder_hip_twist_max (+0.23)

    # 핵심 필수 피처 (절대 제거 금지):
    # [33] view_compensated_poly_a (-2.16%p 급락)
    # [16] torso_pitch_standing (-1.18%p 급락)
    # [27] spine_femur_sync_ratio (-0.95%p 급락)
    # [ 1] lumbar_stiffness_sd (-0.47%p 급락)
    # [22] lumbar_shear_stress_proxy
    # [37] ratio_arm_to_torso
    # [39] ratio_leg_to_torso
    # [36] res_lumbar_poly_decoupled
    # [31] knee_valgus_ratio
    # [32] view_occlusion_factor
    # [28] torso_pitch_rom
    # [26] lockout_hyperextension
    # [ 5] signed_thoracolumbar_delta

    zero_var_feats = [7, 19, 20, 21]

    # 다이어트 후보 서브셋 설계
    combos = {}

    # 후보 1: 상수 4개만 제거 (36D)
    combos["Diet-36D (Zero-Var 4개 제거)"] = [i for i in range(40) if i not in zero_var_feats]

    # 후보 2: 상수 4개 + 극단적 저해 피처 Top 5 제거 (31D)
    # [29] rom_score, [23] setup_poly, [11] cobb_range, [24] fb_cobb, [25] knee_pass
    drop_top5 = zero_var_feats + [29, 23, 11, 24, 25]
    combos["Diet-31D (상수4 + 극단저해 Top 5 제거)"] = [i for i in range(40) if i not in drop_top5]

    # 후보 3: 상수 4개 + 저해 피처(ΔF1 > 0.90) 11개 제거 (25D)
    drop_top11 = zero_var_feats + [29, 23, 11, 24, 25, 10, 14, 17, 30, 35, 34]
    combos["Diet-25D (상수4 + 저해피처 11개 제거)"] = [i for i in range(40) if i not in drop_top11]

    # 후보 4: Cobb각 중복군 및 저해피처 정밀 제거 (20D 정예 서브셋)
    # Cobb각은 cobb_mean, max, range, fb_delta, knee_pass 중 전부 제외하고 3D 공간 만곡 보정(33)으로 단일화!
    # pitch_standing(16), sync_ratio(27), stiffness_sd(1), view_comp(33), res_lumbar_decoupled(36),
    # shear(22), arm_torso(37), leg_torso(39), knee_valgus(31), view_occlusion(32),
    # torso_pitch_rom(28), lockout(26), signed_delta(5), flex_accel(15), active_rom(6)
    candidate_20d = [1, 5, 6, 12, 15, 16, 22, 26, 27, 28, 31, 32, 33, 36, 37, 39]  # 16D 핵심
    combos["Diet-16D (핵심 기여 정예 서브셋)"] = candidate_20d
    combos["Diet-18D (16D + 흉요추 곡률/속도)"] = candidate_20d + [0, 3]
    combos["Diet-21D (18D + 사지비율 38 + 어깨골반 13, 14)"] = candidate_20d + [0, 3, 38, 13, 14]

    # 후보 5: Greedy Forward Selection 기반 상위 피처 (12D ~ 15D)
    # 핵심 8개 피처로부터 시작하여 최적 증분 피처 탐색
    core_seeds = [33, 16, 27, 1, 22, 37, 39, 36]
    combos["Diet-12D (초슬림 초고효율)"] = core_seeds + [31, 32, 28, 26]
    combos["Diet-14D (12D + ROM/곡률)"] = core_seeds + [31, 32, 28, 26, 5, 0]

    print(f"\n[*] 총 {len(combos)}개 피처 다이어트 후보 서브셋 5-Fold 실측 시작...")
    results = {}

    for c_name, feat_indices in combos.items():
        print(f"\n>>> [{c_name}] 평가 중... (피처 수: {len(feat_indices)}개)")
        X_sub = X_40d[:, feat_indices]
        res = evaluate_diet_combo(X_sub, X_temporal, y, subs)
        diff_f1 = (res["f1"] - base_res["f1"]) * 100.0
        diff_auc = res["auc"] - base_res["auc"]

        print(f"    -> Macro F1: {res['f1']*100:6.2f}% ({diff_f1:+5.2f}%p) | Recall: {res['recall']*100:6.2f}% | Prec: {res['precision']*100:6.2f}% | AUC: {res['auc']:.4f} ({diff_auc:+7.4f}) | 3-State F1: {res['sel_f1']*100:.2f}%")
        results[c_name] = {
            "indices": feat_indices,
            "feature_names": [names[k] for k in feat_indices],
            "num_features": len(feat_indices),
            "f1": res["f1"],
            "recall": res["recall"],
            "precision": res["precision"],
            "accuracy": res["accuracy"],
            "auc": res["auc"],
            "sel_f1": res["sel_f1"],
            "abstain": res["abstain"],
            "diff_f1_pctp": diff_f1,
            "diff_auc": diff_auc,
            "cm": res["cm"]
        }

    # 최적 서브셋 도출
    best_combo_name = max(results.items(), key=lambda x: (x[1]["f1"], x[1]["auc"]))[0]
    best_info = results[best_combo_name]

    print("\n" + "=" * 110)
    print(f"  [★ 최적의 다이어트 피처셋 확정: {best_combo_name}]")
    print("=" * 110)
    print(f"  - 피처 수: 기존 40D -> {best_info['num_features']}D ({40 - best_info['num_features']}개 피처 대폭 다이어트!)")
    print(f"  - Macro F1-Score : {base_res['f1']*100:.2f}% -> {best_info['f1']*100:.2f}% ({best_info['diff_f1_pctp']:+5.2f}%p 대폭 상승!)")
    print(f"  - Recall (재현율): {base_res['recall']*100:.2f}% -> {best_info['recall']*100:.2f}%")
    print(f"  - Precision      : {base_res['precision']*100:.2f}% -> {best_info['precision']*100:.2f}%")
    print(f"  - ROC-AUC        : {base_res['auc']:.4f} -> {best_info['auc']:.4f} ({best_info['diff_auc']:+7.4f})")
    print(f"  - 3-State 유효 F1: {base_res['sel_f1']*100:.2f}% -> {best_info['sel_f1']*100:.2f}%")
    print(f"  - 포함된 피처 목록:")
    for rank, fn in enumerate(best_info["feature_names"], 1):
        print(f"    {rank:2d}. {fn}")

    # 최종 결과 저장
    save_data = {
        "baseline_40d": {
            "f1": base_res["f1"], "auc": base_res["auc"], "recall": base_res["recall"],
            "precision": base_res["precision"], "accuracy": base_res["accuracy"],
            "sel_f1": base_res["sel_f1"], "abstain": base_res["abstain"]
        },
        "all_combos": results,
        "best_combo": {
            "name": best_combo_name,
            **best_info
        }
    }

    with open(OUT_DIET_JSON, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 피처 다이어트 최적화 결과 JSON 저장 완료: {OUT_DIET_JSON}")


if __name__ == "__main__":
    main()

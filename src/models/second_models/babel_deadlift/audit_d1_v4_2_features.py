# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: audit_d1_v4_2_features.py
Purpose: v4.2 모델 40D 피처 전수 통계 및 성능 기여도 감사
         1) 분산 0 또는 상수 피처 탐지 (Zero Variance)
         2) 피처 간 다중공선성(상관계수 |r| > 0.85) 상관 행렬 분석
         3) 피처별 단변량 ROC-AUC 및 정보 획득량(Mutual Information)
         4) 5-Fold OOF 기반 개별 피처 제거(Drop-one Ablation) 시 F1 변화량 측정 (저해 피처 식별)
         5) GBDT(LightGBM + CatBoost) Feature Importance (Gain / Split)
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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d1_model_v4_2_spinepose import (
    extract_d1_v4_2_features_34d_base,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_2,
    FEATURE_NAMES_40_D1_V4_2
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_AUDIT_JSON = os.path.join(CURRENT_DIR, "d1_v4_2_feature_audit_results.json")


def load_dataset_40d():
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

    # 40D 피처셋 추출
    X_34d_base = np.zeros((N, 34), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_34d_base[i] = extract_d1_v4_2_features_34d_base(kps, s9)
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps)

    # 5-Fold OOF 기반으로 40D 완성
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

    return X_40d, labels, subject_ids, view_ids


def evaluate_oof_f1(X_subset, y, subs, th=0.35):
    """지정된 피처 서브셋으로 5-Fold GBDT 앙상블 OOF F1 및 AUC 산출"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_subset, y, groups=subs)):
        X_tr, y_tr = X_subset[train_idx], y[train_idx]
        X_va, y_va = X_subset[val_idx], y[val_idx]

        clf_l = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.032, num_leaves=16, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=8,
            random_state=42 + fold, verbose=-1
        )
        clf_l.fit(X_tr, y_tr)
        p_l = clf_l.predict_proba(X_va)[:, 1]

        clf_c = CatBoostClassifier(
            iterations=200, learning_rate=0.038, depth=4, l2_leaf_reg=4.5,
            random_seed=42 + fold, verbose=False
        )
        clf_c.fit(X_tr, y_tr)
        p_c = clf_c.predict_proba(X_va)[:, 1]

        oof_p[val_idx] = 0.5 * p_l + 0.5 * p_c

    preds = (oof_p >= th).astype(int)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, oof_p)
    rec = recall_score(y, preds)
    prec = precision_score(y, preds)
    acc = accuracy_score(y, preds)
    return f1, auc, rec, prec, acc, oof_p


def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v4.2 40D 피처 전수 통계 및 성능 기여도 감사")
    print("=" * 110)

    X_40d, y, subs, views = load_dataset_40d()
    N, D = X_40d.shape
    names = FEATURE_NAMES_40_D1_V4_2

    # 1. 40D 전체 베이스라인 실측
    print(f"\n[*] 40D 전체 베이스라인 5-Fold OOF 측정 중...")
    base_f1, base_auc, base_rec, base_prec, base_acc, base_oof_p = evaluate_oof_f1(X_40d, y, subs)
    print(f"    - 40D Baseline | F1: {base_f1*100:.2f}%, AUC: {base_auc:.4f}, Recall: {base_rec*100:.2f}%, Prec: {base_prec*100:.2f}%, Acc: {base_acc*100:.2f}%")

    # 2. 분산 0(상수) 및 저분산 피처 식별
    stds = np.std(X_40d, axis=0)
    zero_var_indices = np.where(stds < 1e-6)[0]
    print(f"\n[*] [분석 1] 분산 0(Zero Variance / 상수) 피처 탐지:")
    for idx in zero_var_indices:
        print(f"    [상수 피처] Idx {idx:2d}: {names[idx]:35s} (std={stds[idx]:.6f}, value={X_40d[0, idx]})")

    # 3. 다중공선성(상관계수 |r| > 0.85) 쌍 탐색
    print(f"\n[*] [분석 2] 고상관 다중공선성(|r| > 0.85) 피처 쌍 탐지:")
    df_feats = pd.DataFrame(X_40d, columns=names)
    corr_matrix = df_feats.corr().abs()
    high_corr_pairs = []

    for i in range(D):
        for j in range(i + 1, D):
            r_val = corr_matrix.iloc[i, j]
            if r_val > 0.85:
                high_corr_pairs.append((names[i], names[j], i, j, float(r_val)))
                print(f"    - |r| = {r_val:.3f} : [{i:2d}] {names[i]:32s} <--> [{j:2d}] {names[j]}")

    # 4. 단변량 ROC-AUC 및 Mutual Information
    print(f"\n[*] [분석 3] 단변량 변별력 (Univariate ROC-AUC & Mutual Information):")
    uni_metrics = []
    mi_scores = mutual_info_classif(X_40d, y, random_state=42)

    for i in range(D):
        feat_col = X_40d[:, i]
        # ROC-AUC (방향성 고려)
        if stds[i] < 1e-6:
            u_auc = 0.5
        else:
            try:
                raw_auc = roc_auc_score(y, feat_col)
                u_auc = max(raw_auc, 1.0 - raw_auc)
            except Exception:
                u_auc = 0.5
        uni_metrics.append({
            "idx": i,
            "name": names[i],
            "std": float(stds[i]),
            "auc": float(u_auc),
            "mi": float(mi_scores[i])
        })

    # 5. Drop-One Ablation Study (피처 하나씩 제거 시 F1 변화량)
    print(f"\n[*] [분석 4] Drop-One Ablation Study 가동 (개별 피처 제거 시 F1 변화):")
    print(f"    (제거 시 F1이 오르면 -> f1-score를 오히려 저해시키는 노이즈/간섭 피처!)")
    drop_results = []

    for i in range(D):
        # i번째 피처 제외 서브셋
        sub_indices = [k for k in range(D) if k != i]
        d_f1, d_auc, d_rec, d_prec, d_acc, _ = evaluate_oof_f1(X_40d[:, sub_indices], y, subs)
        diff_f1 = (d_f1 - base_f1) * 100.0  # %p
        diff_auc = d_auc - base_auc

        is_harmful = (diff_f1 > 0.05)  # 제거했을 때 F1이 0.05%p 이상 오르면 저해 피처
        drop_results.append({
            "idx": i,
            "name": names[i],
            "f1_after_drop": float(d_f1),
            "f1_diff_pctp": float(diff_f1),
            "auc_after_drop": float(d_auc),
            "auc_diff": float(diff_auc),
            "is_harmful": is_harmful
        })

        status = "★ 저해 피처 (제거 시 성능 향상!)" if is_harmful else ("유지 필요 (핵심)" if diff_f1 < -0.3 else "중립")
        print(f"  [{i:2d}] Drop {names[i]:32s} -> F1: {d_f1*100:5.2f}% ({diff_f1:+5.2f}%p), AUC: {d_auc:.4f} | {status}")

    # 저해 피처 리스트 정리
    harmful_feats = [r for r in drop_results if r["is_harmful"]]
    harmful_feats = sorted(harmful_feats, key=lambda x: x["f1_diff_pctp"], reverse=True)

    print("\n" + "=" * 90)
    print(f"  [식별된 성능 저해 피처 목록 (총 {len(harmful_feats)}개 - 제거 시 F1 상승)]")
    print("=" * 90)
    for hf in harmful_feats:
        print(f"  - [{hf['idx']:2d}] {hf['name']:35s}: 제거 시 F1 {hf['f1_diff_pctp']:+5.2f}%p 상승")

    # 결과 JSON 저장
    audit_summary = {
        "baseline_40d": {
            "f1": float(base_f1),
            "auc": float(base_auc),
            "recall": float(base_rec),
            "precision": float(base_prec),
            "accuracy": float(base_acc)
        },
        "zero_var_indices": [int(x) for x in zero_var_indices],
        "high_corr_pairs": high_corr_pairs,
        "univariate_metrics": uni_metrics,
        "drop_results": drop_results,
        "harmful_features": harmful_feats
    }

    with open(OUT_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 피처 감사 결과 JSON 저장 완료: {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    main()

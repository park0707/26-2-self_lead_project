# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Version: v5.0 (2-Tier Ambiguity Pruned Architecture)
Script: train_and_eval_d1_v5_ambiguity_pruned.py
Purpose: 바벨 데드리프트 d1 최신 v4.3 Diet 파이프라인을 100% 계승하고,
         생체역학적 4단계 모호성 레벨 체계 기반 Level 4 오염 데이터(7.27%)를 폐기한
         2계층 구조(1계층: Level 1 명확군 평가, 2계층: Level 1~3 실무군 평가)를 도입하여
         원천 라벨 학습 v4.3 이전 버전 대비 종합 성능 도약 실측 비교
"""

import os
import sys
import time
import glob
import json
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import (
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_25_D1_V4_3
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
AMBIGUITY_NPZ = os.path.join(CURRENT_DIR, "deadlift_d1_ambiguity_levels.npz")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_d1_v5")
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "d1_v5_ambiguity_comparison_summary.json")

VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def load_dataset_with_ambiguity():
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

    amb_data = np.load(AMBIGUITY_NPZ, allow_pickle=True)
    amb_levels = amb_data['levels']
    amb_reasons = amb_data['reasons']
    amb_dict = {sid: (amb_levels[i], amb_reasons[i]) for i, sid in enumerate(amb_data['sequence_ids'])}

    sequence_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []
    levels = []
    reasons = []

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

        lvl, rsn = amb_dict.get(sid, (1, "Level 1: 기본"))

        sequence_ids.append(sid)
        labels.append(is_d1_defect)
        subject_ids.append(sub)
        view_ids.append(vn)
        kps_list.append(kps_33_64f)
        spine_list.append(spine_dict.get(sid, None))
        levels.append(lvl)
        reasons.append(rsn)

    labels = np.array(labels, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)
    view_ids = np.array(view_ids)
    levels = np.array(levels, dtype=int)
    reasons = np.array(reasons)
    N = len(labels)

    print(f"[*] 총 원천 데이터: {N}개 시퀀스 | 정상: {np.sum(labels == 0)}개, 결함: {np.sum(labels == 1)}개")
    print(f"[*] 4단계 모호성 레벨 분포:")
    for lvl in [1, 2, 3, 4]:
        cnt = np.sum(levels == lvl)
        print(f"    - Level {lvl}: {cnt:3d}개 ({cnt/N*100:5.2f}%)")

    # 시계열 텐서 (64, 24)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps)

    return {
        'sequence_ids': sequence_ids,
        'labels': labels,
        'subject_ids': subject_ids,
        'view_ids': view_ids,
        'kps_list': kps_list,
        'spine_list': spine_list,
        'cond_ratios': cond_ratios,
        'X_temporal': X_temporal,
        'levels': levels,
        'reasons': reasons
    }


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    return {
        'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc, 'roc_auc': auc,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'sample_count': len(y_true)
    }


def run_experiment():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v5.0 2계층 구조 (모호성 정제) 5-Fold 피험자 격리 실험")
    print("=" * 115)

    data = load_dataset_with_ambiguity()
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    kps_list = data['kps_list']
    spine_list = data['spine_list']
    cond_ratios = data['cond_ratios']
    X_temporal = data['X_temporal']
    levels = data['levels']
    N = len(labels)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. 5-Fold 분할 (전체 데이터 피험자 격리 기준 동일 적용)
    # -------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(N), labels, groups=subject_ids))

    # OOF 확률 저장 배열
    # [Model A] 원천 라벨 전체 학습 (v4.3 베이스라인)
    oof_p_raw_ens = np.zeros(N, dtype=np.float32)

    # [Model B] Level 4 폐기 정제 학습 (v5.0 2계층 모델)
    oof_p_pruned_ens = np.zeros(N, dtype=np.float32)

    lgb_params = dict(
        n_estimators=180, learning_rate=0.032, num_leaves=16, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=8, verbose=-1
    )
    cat_params = dict(
        iterations=200, learning_rate=0.038, depth=4, l2_leaf_reg=4.5, verbose=False
    )

    for fold, (train_idx, val_idx) in enumerate(splits):
        val_subs = sorted(list(set(subject_ids[val_idx])))
        print(f"\n>>> [Fold {fold + 1}/5] 검증 피험자: {val_subs} (검증 샘플: {len(val_idx)}개)")

        # =========================================================
        # [A] 원천 라벨 모델 학습 (v4.3 Baseline: Train 전체 100% 사용)
        # =========================================================
        norm_train_raw = train_idx[labels[train_idx] == 0]
        train_norm_curvs_raw = []
        for ti in norm_train_raw:
            f_temp = extract_d1_v4_3_diet_features_25d(kps_list[ti], spine_list[ti], None)
            train_norm_curvs_raw.append(float(f_temp[0]))

        res_raw = MorphologyResidualizerV4_3(alpha=15.0)
        res_raw.fit(cond_ratios[norm_train_raw], np.array(train_norm_curvs_raw, dtype=np.float32))

        X_train_raw_25 = np.zeros((len(train_idx), 25), dtype=np.float32)
        for k, idx in enumerate(train_idx):
            X_train_raw_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res_raw)

        X_val_raw_25 = np.zeros((len(val_idx), 25), dtype=np.float32)
        for k, idx in enumerate(val_idx):
            X_val_raw_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res_raw)

        y_train_raw = labels[train_idx]

        clf_lgb_raw = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold)
        clf_lgb_raw.fit(X_train_raw_25, y_train_raw)
        p_lgb_raw = clf_lgb_raw.predict_proba(X_val_raw_25)[:, 1]

        clf_cat_raw = CatBoostClassifier(**cat_params, random_seed=42 + fold)
        clf_cat_raw.fit(X_train_raw_25, y_train_raw)
        p_cat_raw = clf_cat_raw.predict_proba(X_val_raw_25)[:, 1]

        # TCN 학습 (Raw)
        tcn_raw = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        opt_raw = torch.optim.AdamW(tcn_raw.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()

        X_tr_t_raw = torch.tensor(X_temporal[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t_raw = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)

        loader_raw = DataLoader(TensorDataset(X_tr_t_raw, y_tr_t_raw), batch_size=32, shuffle=True)
        tcn_raw.train()
        for epoch in range(30):
            for bx, by in loader_raw:
                opt_raw.zero_grad()
                crit(tcn_raw(bx), by).backward()
                opt_raw.step()
        tcn_raw.eval()
        with torch.no_grad():
            p_tcn_raw = torch.sigmoid(tcn_raw(X_va_t).squeeze(1)).numpy()

        p_gbdt_raw = 0.5 * p_lgb_raw + 0.5 * p_cat_raw
        oof_p_raw_ens[val_idx] = 0.5 * p_gbdt_raw + 0.5 * p_tcn_raw

        # =========================================================
        # [B] Level 4 폐기 정제 모델 학습 (v5.0: Train에서 Level 4 제거)
        # =========================================================
        train_clean_mask = (levels[train_idx] != 4)
        train_clean_idx = train_idx[train_clean_mask]

        norm_train_clean = train_clean_idx[labels[train_clean_idx] == 0]
        train_norm_curvs_clean = []
        for ti in norm_train_clean:
            f_temp = extract_d1_v4_3_diet_features_25d(kps_list[ti], spine_list[ti], None)
            train_norm_curvs_clean.append(float(f_temp[0]))

        res_clean = MorphologyResidualizerV4_3(alpha=15.0)
        res_clean.fit(cond_ratios[norm_train_clean], np.array(train_norm_curvs_clean, dtype=np.float32))

        X_train_clean_25 = np.zeros((len(train_clean_idx), 25), dtype=np.float32)
        for k, idx in enumerate(train_clean_idx):
            X_train_clean_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res_clean)

        X_val_clean_25 = np.zeros((len(val_idx), 25), dtype=np.float32)
        for k, idx in enumerate(val_idx):
            X_val_clean_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res_clean)

        y_train_clean = labels[train_clean_idx]

        clf_lgb_clean = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold)
        clf_lgb_clean.fit(X_train_clean_25, y_train_clean)
        p_lgb_clean = clf_lgb_clean.predict_proba(X_val_clean_25)[:, 1]

        clf_cat_clean = CatBoostClassifier(**cat_params, random_seed=42 + fold)
        clf_cat_clean.fit(X_train_clean_25, y_train_clean)
        p_cat_clean = clf_cat_clean.predict_proba(X_val_clean_25)[:, 1]

        # TCN 학습 (Clean)
        tcn_clean = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        opt_clean = torch.optim.AdamW(tcn_clean.parameters(), lr=1e-3, weight_decay=1e-4)

        X_tr_t_clean = torch.tensor(X_temporal[train_clean_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t_clean = torch.tensor(y_train_clean, dtype=torch.float32).unsqueeze(1)

        loader_clean = DataLoader(TensorDataset(X_tr_t_clean, y_tr_t_clean), batch_size=32, shuffle=True)
        tcn_clean.train()
        for epoch in range(30):
            for bx, by in loader_clean:
                opt_clean.zero_grad()
                crit(tcn_clean(bx), by).backward()
                opt_clean.step()
        tcn_clean.eval()
        with torch.no_grad():
            p_tcn_clean = torch.sigmoid(tcn_clean(X_va_t).squeeze(1)).numpy()

        p_gbdt_clean = 0.5 * p_lgb_clean + 0.5 * p_cat_clean
        oof_p_pruned_ens[val_idx] = 0.5 * p_gbdt_clean + 0.5 * p_tcn_clean

        # 폴드 가중치 저장 (Fold 1 대표)
        if fold == 0:
            with open(os.path.join(WEIGHTS_DIR, "d1_v5_clean_gbdt_models.pkl"), "wb") as f:
                pickle.dump({
                    "clf_lgb": clf_lgb_clean,
                    "clf_cat": clf_cat_clean,
                    "residualizer": res_clean,
                    "decision_threshold": 0.35
                }, f)
            torch.save(tcn_clean.state_dict(), os.path.join(WEIGHTS_DIR, "d1_v5_clean_tcn.pth"))

    th = 0.35
    preds_raw = (oof_p_raw_ens >= th).astype(int)
    preds_clean = (oof_p_pruned_ens >= th).astype(int)

    # -------------------------------------------------------------
    # 2. 성능 평가: 2계층 구조 대조
    #    - 1계층 평가: Level 1 (완전 명확군)
    #    - 2계층 평가: Level 1 ~ Level 3 (실무 전체 보존군, Level 4 폐기)
    #    - 전체 평가: Level 1 ~ Level 4 (원천 385개 전체)
    # -------------------------------------------------------------
    m_tier1_eval = (levels == 1)
    m_tier2_eval = (levels >= 1) & (levels <= 3) # Level 4 폐기

    # [1계층 평가: Level 1만]
    metrics_tier1_raw = compute_metrics(labels[m_tier1_eval], preds_raw[m_tier1_eval], oof_p_raw_ens[m_tier1_eval])
    metrics_tier1_clean = compute_metrics(labels[m_tier1_eval], preds_clean[m_tier1_eval], oof_p_pruned_ens[m_tier1_eval])

    # [2계층 평가: Level 1~3 전체]
    metrics_tier2_raw = compute_metrics(labels[m_tier2_eval], preds_raw[m_tier2_eval], oof_p_raw_ens[m_tier2_eval])
    metrics_tier2_clean = compute_metrics(labels[m_tier2_eval], preds_clean[m_tier2_eval], oof_p_pruned_ens[m_tier2_eval])

    # [원천 385개 전체 평가]
    metrics_full_raw = compute_metrics(labels, preds_raw, oof_p_raw_ens)
    metrics_full_clean = compute_metrics(labels, preds_clean, oof_p_pruned_ens)

    # -------------------------------------------------------------
    # 3. 콘솔 결과 출력 및 비교표
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [종합 비교] 이전 버전(v4.3 원천학습) vs 신규 버전(v5.0 Level 4 정제학습)")
    print("=" * 115)

    headers = f"{'평가 계층 (데이터셋)':<28} | {'버전':<15} | {'F1-Score':>10} | {'정밀도':>10} | {'재현율':>10} | {'정확도':>10} | {'ROC-AUC':>10} | {'FP':>5} | {'FN':>5}"
    print(headers)
    print("-" * 115)

    def print_row(name, ver, m):
        print(f"{name:<28} | {ver:<15} | {m['f1']*100:>9.2f}% | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['accuracy']*100:>9.2f}% | {m['roc_auc']*100:>9.2f}% | {m['fp']:>5d} | {m['fn']:>5d}")

    print_row("1계층 (Level 1 명확군, N=136)", "v4.3 원천학습", metrics_tier1_raw)
    print_row("1계층 (Level 1 명확군, N=136)", "v5.0 모호성정제", metrics_tier1_clean)
    diff_t1_f1 = (metrics_tier1_clean['f1'] - metrics_tier1_raw['f1']) * 100
    print(f"  └─>> 1계층 성능 변화폭: F1 {diff_t1_f1:+6.2f}%p | FP {metrics_tier1_clean['fp'] - metrics_tier1_raw['fp']:+3d} | FN {metrics_tier1_clean['fn'] - metrics_tier1_raw['fn']:+3d}")
    print("-" * 115)

    print_row("2계층 (Level 1~3 실무군, N=357)", "v4.3 원천학습", metrics_tier2_raw)
    print_row("2계층 (Level 1~3 실무군, N=357)", "v5.0 모호성정제", metrics_tier2_clean)
    diff_t2_f1 = (metrics_tier2_clean['f1'] - metrics_tier2_raw['f1']) * 100
    print(f"  └─>> 2계층 성능 변화폭: F1 {diff_t2_f1:+6.2f}%p | FP {metrics_tier2_clean['fp'] - metrics_tier2_raw['fp']:+3d} | FN {metrics_tier2_clean['fn'] - metrics_tier2_raw['fn']:+3d}")
    print("-" * 115)

    print_row("원천전체 (Level 1~4 전체, N=385)", "v4.3 원천학습", metrics_full_raw)
    print_row("원천전체 (Level 1~4 전체, N=385)", "v5.0 모호성정제", metrics_full_clean)
    diff_f_f1 = (metrics_full_clean['f1'] - metrics_full_raw['f1']) * 100
    print(f"  └─>> 전체 성능 변화폭: F1 {diff_f_f1:+6.2f}%p | FP {metrics_full_clean['fp'] - metrics_full_raw['fp']:+3d} | FN {metrics_full_clean['fn'] - metrics_full_raw['fn']:+3d}")
    print("-" * 115)

    # -------------------------------------------------------------
    # 4. 5대 공식 카메라 시점별 성능 비교 (2계층 Level 1~3 기준)
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [5대 공식 카메라 시점별 2계층(Level 1~3) 성능 비교]")
    print("=" * 115)
    print(f"{'시점 명칭':<25} | {'코드':<6} | {'샘플수':>6} | {'v4.3 F1':>10} | {'v5.0 F1':>10} | {'F1 변화':>10} | {'v5.0 ROC-AUC':>14}")
    print("-" * 115)

    view_results = {}
    for vc in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = VIEW_ALIAS_MAP[vc]
        m_view = m_tier2_eval & (view_ids == vc)
        cnt_v = np.sum(m_view)
        m_v_raw = compute_metrics(labels[m_view], preds_raw[m_view], oof_p_raw_ens[m_view])
        m_v_clean = compute_metrics(labels[m_view], preds_clean[m_view], oof_p_pruned_ens[m_view])
        diff_v = (m_v_clean['f1'] - m_v_raw['f1']) * 100
        print(f"{va:<25} | {vc:<6} | {cnt_v:6d} | {m_v_raw['f1']*100:>9.2f}% | {m_v_clean['f1']*100:>9.2f}% | {diff_v:>+9.2f}%p | {m_v_clean['roc_auc']*100:>13.2f}%")
        view_results[va] = {
            'view_code': vc,
            'sample_count': int(cnt_v),
            'v4_3_f1': m_v_raw['f1'],
            'v5_0_f1': m_v_clean['f1'],
            'diff_f1': diff_v,
            'v5_0_roc_auc': m_v_clean['roc_auc'],
            'v5_0_accuracy': m_v_clean['accuracy']
        }

    # -------------------------------------------------------------
    # 5. 피험자 1~9번별 2계층 성능 비교
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [피험자 1~9번별 2계층(Level 1~3) 성능 비교]")
    print("=" * 115)
    print(f"{'피험자':<12} | {'샘플수':>6} | {'정상/결함':>12} | {'v4.3 F1':>10} | {'v5.0 F1':>10} | {'F1 변화':>10} | {'v5.0 정확도':>12}")
    print("-" * 115)

    sub_results = {}
    for s in range(1, 10):
        m_s = m_tier2_eval & (subject_ids == s)
        cnt_s = np.sum(m_s)
        n_norm = np.sum(m_s & (labels == 0))
        n_def = np.sum(m_s & (labels == 1))
        m_s_raw = compute_metrics(labels[m_s], preds_raw[m_s], oof_p_raw_ens[m_s])
        m_s_clean = compute_metrics(labels[m_s], preds_clean[m_s], oof_p_pruned_ens[m_s])
        diff_s = (m_s_clean['f1'] - m_s_raw['f1']) * 100
        print(f"피험자 {s}번     | {cnt_s:6d} | {f'{n_norm}/{n_def}':>12} | {m_s_raw['f1']*100:>9.2f}% | {m_s_clean['f1']*100:>9.2f}% | {diff_s:>+9.2f}%p | {m_s_clean['accuracy']*100:>11.2f}%")
        sub_results[f"피험자_{s}"] = {
            'sample_count': int(cnt_s),
            'normal_count': int(n_norm),
            'defect_count': int(n_def),
            'v4_3_f1': m_s_raw['f1'],
            'v5_0_f1': m_s_clean['f1'],
            'diff_f1': diff_s,
            'v5_0_accuracy': m_s_clean['accuracy']
        }

    # 결과 JSON 저장
    final_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_name': 'Barbell Deadlift d1 2-Tier Ambiguity Pruned Architecture (v5.0)',
        'pruned_level4_count': int(np.sum(levels == 4)),
        'tier1_level1_metrics': {
            'v4_3_raw': metrics_tier1_raw,
            'v5_0_clean': metrics_tier1_clean,
            'f1_diff_pct': diff_t1_f1
        },
        'tier2_level1_to_3_metrics': {
            'v4_3_raw': metrics_tier2_raw,
            'v5_0_clean': metrics_tier2_clean,
            'f1_diff_pct': diff_t2_f1
        },
        'full_dataset_metrics': {
            'v4_3_raw': metrics_full_raw,
            'v5_0_clean': metrics_full_clean,
            'f1_diff_pct': diff_f_f1
        },
        'view_results': view_results,
        'subject_results': sub_results
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as jf:
        json.dump(final_summary, jf, indent=2, ensure_ascii=False)
    print(f"\n[+] 벤치마크 요약 JSON 저장 완료: {OUTPUT_SUMMARY_JSON}")


if __name__ == '__main__':
    run_experiment()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_babel_squat_integrated_v1_vs_v2_benchmark.py
Purpose: 바벨 스쿼트 정상 오류 분류 모델 v1 (c1 v1 탑재) vs v2 (c1 v6-A 탑재) 5-Fold 실측 비교 벤치마크
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, hamming_loss
)
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v1 import BabelSquatC1ModelV1
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES
from src.models.second_models.babel_squat.c2_model_v3 import V2_TO_V3_INDICES as C2_INDICES
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES
from src.models.second_models.babel_squat.c4_model_v3 import V3_INDICES_FROM_26D as C4_INDICES
from src.models.second_models.babel_squat.babel_squat_integrated_v1 import DEFAULT_RISK_WEIGHTS

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/squat_integrated_v1_vs_v2_results.json'

def compute_binary_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'accuracy': round(float(acc), 2),
        'specificity': round(float(spec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def compute_soft_defect_scores(probs_matrix, risk_weights=DEFAULT_RISK_WEIGHTS):
    # probs_matrix: (N, 4) [c1, c2, c3, c4]
    w = np.array([risk_weights['c1'], risk_weights['c2'], risk_weights['c3'], risk_weights['c4']], dtype=np.float32)
    weighted = np.clip(probs_matrix * w, 0.0, 1.0)
    prod_clean = np.prod(1.0 - weighted, axis=1)
    return 1.0 - prod_clean

def evaluate_integrated_pipeline(oof_probs, Y_defects, y_overall_true, mask=None, desc=""):
    if mask is not None:
        oof_p = {k: v[mask] for k, v in oof_probs.items()}
        Y_sub = Y_defects[mask]
        y_ov_sub = y_overall_true[mask]
    else:
        oof_p = oof_probs
        Y_sub = Y_defects
        y_ov_sub = y_overall_true

    N = len(y_ov_sub)
    keys = ['c1', 'c2', 'c3', 'c4']
    prob_mat = np.column_stack([oof_p[k] for k in keys])
    pred_multi = (prob_mat >= 0.50).astype(int)

    # 1. 개별 결함별 지표
    defect_stats = {}
    for i, k in enumerate(keys):
        m = compute_binary_metrics(Y_sub[:, i], prob_mat[:, i], threshold=0.50)
        defect_stats[k] = m

    # 2. 다중 라벨 종합 지표
    exact_matches = np.all(pred_multi == Y_sub, axis=1)
    exact_match_ratio = round(float(np.mean(exact_matches) * 100.0), 2)
    ham_loss = round(float(hamming_loss(Y_sub, pred_multi) * 100.0), 2)
    macro_f1 = round(float(np.mean([defect_stats[k]['f1'] for k in keys])), 2)
    micro_f1 = round(float(f1_score(Y_sub.ravel(), pred_multi.ravel(), zero_division=0) * 100.0), 2)

    # 3. Soft Risk Fusion 종합 점수
    soft_scores = compute_soft_defect_scores(prob_mat)

    # 4. 결합 방식별 성능 (Hard OR vs Soft Fusion @0.79 vs Soft Fusion Opt)
    hard_or_preds = (np.sum(pred_multi, axis=1) > 0).astype(int)
    hard_or_metrics = {
        'f1': round(float(f1_score(y_ov_sub, hard_or_preds, zero_division=0) * 100.0), 2),
        'precision': round(float(precision_score(y_ov_sub, hard_or_preds, zero_division=0) * 100.0), 2),
        'recall': round(float(recall_score(y_ov_sub, hard_or_preds, zero_division=0) * 100.0), 2),
        'accuracy': round(float(accuracy_score(y_ov_sub, hard_or_preds) * 100.0), 2),
        'specificity': round(float(np.sum((hard_or_preds == 0) & (y_ov_sub == 0)) / max(1, np.sum(y_ov_sub == 0)) * 100.0), 2),
        'clean_fps': int(np.sum((hard_or_preds == 1) & (y_ov_sub == 0)))
    }

    # Soft Fusion @0.79
    soft_079_preds = (soft_scores >= 0.79).astype(int)
    soft_079_metrics = {
        'f1': round(float(f1_score(y_ov_sub, soft_079_preds, zero_division=0) * 100.0), 2),
        'precision': round(float(precision_score(y_ov_sub, soft_079_preds, zero_division=0) * 100.0), 2),
        'recall': round(float(recall_score(y_ov_sub, soft_079_preds, zero_division=0) * 100.0), 2),
        'accuracy': round(float(accuracy_score(y_ov_sub, soft_079_preds) * 100.0), 2),
        'specificity': round(float(np.sum((soft_079_preds == 0) & (y_ov_sub == 0)) / max(1, np.sum(y_ov_sub == 0)) * 100.0), 2),
        'clean_fps': int(np.sum((soft_079_preds == 1) & (y_ov_sub == 0))),
        'roc_auc': round(float(roc_auc_score(y_ov_sub, soft_scores) * 100.0), 2)
    }

    # Soft Fusion 최적 임계값 탐색
    best_th, best_f1 = 0.50, 0.0
    for th in np.linspace(0.40, 0.90, 51):
        f = f1_score(y_ov_sub, (soft_scores >= th).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, th
    
    soft_opt_preds = (soft_scores >= best_th).astype(int)
    soft_opt_metrics = {
        'opt_th': round(float(best_th), 2),
        'f1': round(float(best_f1), 2),
        'precision': round(float(precision_score(y_ov_sub, soft_opt_preds, zero_division=0) * 100.0), 2),
        'recall': round(float(recall_score(y_ov_sub, soft_opt_preds, zero_division=0) * 100.0), 2),
        'accuracy': round(float(accuracy_score(y_ov_sub, soft_opt_preds) * 100.0), 2),
        'specificity': round(float(np.sum((soft_opt_preds == 0) & (y_ov_sub == 0)) / max(1, np.sum(y_ov_sub == 0)) * 100.0), 2),
        'clean_fps': int(np.sum((soft_opt_preds == 1) & (y_ov_sub == 0))),
        'roc_auc': round(float(roc_auc_score(y_ov_sub, soft_scores) * 100.0), 2)
    }

    # 5. 결함 개수별 (0개~4개) 정밀 분석
    num_defects = np.sum(Y_sub, axis=1)
    counts_analysis = {}
    for cnt in range(5):
        c_mask = (num_defects == cnt)
        cnt_samples = int(np.sum(c_mask))
        if cnt_samples > 0:
            c_exact = round(float(np.mean(exact_matches[c_mask]) * 100.0), 2)
            c_exact_count = int(np.sum(exact_matches[c_mask]))
            c_ov_acc = round(float(np.mean(soft_079_preds[c_mask] == y_ov_sub[c_mask]) * 100.0), 2)
            c_slot_acc = round(float((1.0 - hamming_loss(Y_sub[c_mask], pred_multi[c_mask])) * 100.0), 2)
            counts_analysis[str(cnt)] = {
                'samples': cnt_samples,
                'overall_acc': c_ov_acc,
                'exact_match_ratio': c_exact,
                'exact_match_count': c_exact_count,
                'slot_accuracy': c_slot_acc
            }

    return {
        'defect_metrics': defect_stats,
        'exact_match_ratio': exact_match_ratio,
        'exact_match_count': int(np.sum(exact_matches)),
        'hamming_loss': ham_loss,
        'slot_accuracy': round(100.0 - ham_loss, 2),
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'hard_or': hard_or_metrics,
        'soft_fusion_079': soft_079_metrics,
        'soft_fusion_opt': soft_opt_metrics,
        'defect_count_breakdown': counts_analysis
    }

def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 정상 오류 분류 모델 v1 (c1 v1) vs v2 (c1 v6-A) 실측 비교")
    print("  ★ 9명 전수 5-Fold StratifiedGroupKFold 완전 격리 환경 OOF 정밀 벤치마크")
    print("=" * 115)

    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)
    subjects = ds['subjects']
    N_total = len(Y_defects)
    y_overall_true = (np.sum(Y_defects, axis=1) > 0).astype(int)

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    camera_views = m['views'][sq_mask]

    amb_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    lvl4_mask = ~amb_d['l4_mask']

    # 1. 공통 피처 준비
    print("\n[*] 64프레임 스플라인 보간 및 피처셋 로드 중...")
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    # c1 v1 22D 피처
    feats_c1_v1_22d = BabelSquatC1ModelV1.extract_features(X_b_64, use_pruned=True)

    # c1 v6-A 29D 피처
    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz', allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v5_spinepose_feats.npz', allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    X_34d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d]).astype(np.float32)
    feats_c1_v6a_29d = X_34d[:, V6A_KEEP_INDICES]

    # c2 13D 피처
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names_c2 = list(h_data['feature_names'])
    elite_idx_c2 = [i for i, n in enumerate(feat_names_c2) if n != 'c2_gaze_hysteresis_rmse']
    X_18d_c2 = X_19d[:, elite_idx_c2]
    g_data_c2 = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats_c2 = g_data_c2['gated_feats'].astype(np.float32)[:, :2]
    feats_c2_20d = np.concatenate([X_18d_c2, gated_feats_c2], axis=1)
    feats_c2_13d = feats_c2_20d[:, C2_INDICES]

    # c3 20D 피처
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d_c3 = v1_data_c3['feats'].astype(np.float32)
    foot_data_c3 = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d_c3 = foot_data_c3['foot_feats'].astype(np.float32)
    feats_c3_24d = np.concatenate([X_16d_c3, X_foot_8d_c3], axis=1)
    feats_c3_20d = feats_c3_24d[:, C3_INDICES]

    # c4 9D 피처
    v1_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d_c4 = v1_data_c4['feats'].astype(np.float32)
    foot_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d_c4 = foot_data_c4['foot_feats'].astype(np.float32)
    feats_c4_26d = np.concatenate([X_16d_c4, X_foot_10d_c4], axis=1)
    feats_c4_9d = feats_c4_26d[:, C4_INDICES]

    print(f"    - c1 v1 피처: {feats_c1_v1_22d.shape}")
    print(f"    - c1 v6-A 피처: {feats_c1_v6a_29d.shape}")
    print(f"    - c2 v3 피처: {feats_c2_13d.shape}")
    print(f"    - c3 v3 피처: {feats_c3_20d.shape}")
    print(f"    - c4 v3 피처: {feats_c4_9d.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. 5-Fold OOF 확률 생성
    print("\n[*] 5-Fold StratifiedGroupKFold OOF 예측 진행 중...")
    oof_v1 = {'c1': np.zeros(N_total, dtype=np.float32), 'c2': np.zeros(N_total, dtype=np.float32), 'c3': np.zeros(N_total, dtype=np.float32), 'c4': np.zeros(N_total, dtype=np.float32)}
    oof_v2 = {'c1': np.zeros(N_total, dtype=np.float32), 'c2': np.zeros(N_total, dtype=np.float32), 'c3': np.zeros(N_total, dtype=np.float32), 'c4': np.zeros(N_total, dtype=np.float32)}

    # c2, c3, c4 공통 OOF 산출 (Cascade: 0.60 CatBoost + 0.40 LightGBM)
    for code, X_col, y_col in [
        ('c2', feats_c2_13d, Y_defects[:, 1]),
        ('c3', feats_c3_20d, Y_defects[:, 2]),
        ('c4', feats_c4_9d, Y_defects[:, 3])
    ]:
        for trn, val in sgkf.split(X_col, y_col, groups=subjects):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(X_col[trn], y_col[trn])
            p_cb = cb.predict_proba(X_col[val])[:, 1]

            clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf_l.fit(X_col[trn], y_col[trn])
            p_lgb = clf_l.predict_proba(X_col[val])[:, 1]

            p_casc = 0.60 * p_cb + 0.40 * p_lgb
            oof_v1[code][val] = p_casc
            oof_v2[code][val] = p_casc

    # c1 v1 OOF
    y_c1 = Y_defects[:, 0]
    for trn, val in sgkf.split(feats_c1_v1_22d, y_c1, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_c1_v1_22d[trn], y_c1[trn])
        p_cb = cb.predict_proba(feats_c1_v1_22d[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_c1_v1_22d[trn], y_c1[trn])
        p_lgb = clf_l.predict_proba(feats_c1_v1_22d[val])[:, 1]

        oof_v1['c1'][val] = 0.60 * p_cb + 0.40 * p_lgb

    # c1 v6-A OOF
    for trn, val in sgkf.split(feats_c1_v6a_29d, y_c1, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_c1_v6a_29d[trn], y_c1[trn])
        p_cb = cb.predict_proba(feats_c1_v6a_29d[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_c1_v6a_29d[trn], y_c1[trn])
        p_lgb = clf_l.predict_proba(feats_c1_v6a_29d[val])[:, 1]

        oof_v2['c1'][val] = 0.60 * p_cb + 0.40 * p_lgb

    print("[*] OOF 확률 산출 완료. 통합 평가 시작...")

    # 3. 원천 1,906건 전체 대상 평가
    res_v1_all = evaluate_integrated_pipeline(oof_v1, Y_defects, y_overall_true, mask=None, desc="v1_all")
    res_v2_all = evaluate_integrated_pipeline(oof_v2, Y_defects, y_overall_true, mask=None, desc="v2_all")

    # 4. Level 1~3 실무군 1,577건 대상 평가
    res_v1_l123 = evaluate_integrated_pipeline(oof_v1, Y_defects, y_overall_true, mask=lvl4_mask, desc="v1_l123")
    res_v2_l123 = evaluate_integrated_pipeline(oof_v2, Y_defects, y_overall_true, mask=lvl4_mask, desc="v2_l123")

    # 콘솔 출력
    print("\n" + "=" * 115)
    print("  [원천 1,906건 전체] 바벨 스쿼트 정상 오류 분류 모델 v1 vs v2 실측 비교")
    print("=" * 115)
    print(f"{'평가 지표':<32} | {'통합 모델 v1 (기존)':<20} | {'통합 모델 v2 (c1 v6-A)':<22} | {'변화량 (Δ)'}")
    print("-" * 115)
    print(f"{'c1 (척추 중립) F1 Score':<32} | {res_v1_all['defect_metrics']['c1']['f1']:6.2f}%              | {res_v2_all['defect_metrics']['c1']['f1']:6.2f}%                | {res_v2_all['defect_metrics']['c1']['f1'] - res_v1_all['defect_metrics']['c1']['f1']:+5.2f}%p")
    print(f"{'c1 (척추 중립) ROC-AUC':<32} | {res_v1_all['defect_metrics']['c1']['roc_auc']:6.2f}%              | {res_v2_all['defect_metrics']['c1']['roc_auc']:6.2f}%                | {res_v2_all['defect_metrics']['c1']['roc_auc'] - res_v1_all['defect_metrics']['c1']['roc_auc']:+5.2f}%p")
    print(f"{'다중라벨 Exact Match (4개 100%)':<32} | {res_v1_all['exact_match_ratio']:6.2f}% ({res_v1_all['exact_match_count']}건)       | {res_v2_all['exact_match_ratio']:6.2f}% ({res_v2_all['exact_match_count']}건)         | {res_v2_all['exact_match_ratio'] - res_v1_all['exact_match_ratio']:+5.2f}%p")
    print(f"{'Hamming Loss (개별 오류율)':<32} | {res_v1_all['hamming_loss']:6.2f}%              | {res_v2_all['hamming_loss']:6.2f}%                | {res_v2_all['hamming_loss'] - res_v1_all['hamming_loss']:+5.2f}%p")
    print(f"{'개별 슬롯 정답률 (Slot Acc)':<32} | {res_v1_all['slot_accuracy']:6.2f}%              | {res_v2_all['slot_accuracy']:6.2f}%                | {res_v2_all['slot_accuracy'] - res_v1_all['slot_accuracy']:+5.2f}%p")
    print(f"{'Macro-averaged F1':<32} | {res_v1_all['macro_f1']:6.2f}%              | {res_v2_all['macro_f1']:6.2f}%                | {res_v2_all['macro_f1'] - res_v1_all['macro_f1']:+5.2f}%p")
    print(f"{'Micro-averaged F1':<32} | {res_v1_all['micro_f1']:6.2f}%              | {res_v2_all['micro_f1']:6.2f}%                | {res_v2_all['micro_f1'] - res_v1_all['micro_f1']:+5.2f}%p")
    print("-" * 115)
    print(f"{'Soft Fusion (θ=0.79) F1':<32} | {res_v1_all['soft_fusion_079']['f1']:6.2f}%              | {res_v2_all['soft_fusion_079']['f1']:6.2f}%                | {res_v2_all['soft_fusion_079']['f1'] - res_v1_all['soft_fusion_079']['f1']:+5.2f}%p")
    print(f"{'Soft Fusion (θ=0.79) 정답률':<32} | {res_v1_all['soft_fusion_079']['accuracy']:6.2f}%              | {res_v2_all['soft_fusion_079']['accuracy']:6.2f}%                | {res_v2_all['soft_fusion_079']['accuracy'] - res_v1_all['soft_fusion_079']['accuracy']:+5.2f}%p")
    print(f"{'Soft Fusion 정상맞춤률 (Spec)':<32} | {res_v1_all['soft_fusion_079']['specificity']:6.2f}%              | {res_v2_all['soft_fusion_079']['specificity']:6.2f}%                | {res_v2_all['soft_fusion_079']['specificity'] - res_v1_all['soft_fusion_079']['specificity']:+5.2f}%p")
    print(f"{'정상 오탐 건수 (Clean FP / 125건)':<32} | {res_v1_all['soft_fusion_079']['clean_fps']}건                    | {res_v2_all['soft_fusion_079']['clean_fps']}건                      | {res_v2_all['soft_fusion_079']['clean_fps'] - res_v1_all['soft_fusion_079']['clean_fps']:+2d}건")
    print(f"{'종합 판정 ROC-AUC':<32} | {res_v1_all['soft_fusion_079']['roc_auc']:6.2f}%              | {res_v2_all['soft_fusion_079']['roc_auc']:6.2f}%                | {res_v2_all['soft_fusion_079']['roc_auc'] - res_v1_all['soft_fusion_079']['roc_auc']:+5.2f}%p")

    print("\n" + "=" * 115)
    print("  [Level 1~3 실무군 1,577건] 바벨 스쿼트 정상 오류 분류 모델 v1 vs v2 실측 비교")
    print("=" * 115)
    print(f"{'평가 지표':<32} | {'통합 모델 v1 (기존)':<20} | {'통합 모델 v2 (c1 v6-A)':<22} | {'변화량 (Δ)'}")
    print("-" * 115)
    print(f"{'c1 (척추 중립) F1 Score':<32} | {res_v1_l123['defect_metrics']['c1']['f1']:6.2f}%              | {res_v2_l123['defect_metrics']['c1']['f1']:6.2f}%                | {res_v2_l123['defect_metrics']['c1']['f1'] - res_v1_l123['defect_metrics']['c1']['f1']:+5.2f}%p")
    print(f"{'다중라벨 Exact Match (4개 100%)':<32} | {res_v1_l123['exact_match_ratio']:6.2f}% ({res_v1_l123['exact_match_count']}건)       | {res_v2_l123['exact_match_ratio']:6.2f}% ({res_v2_l123['exact_match_count']}건)         | {res_v2_l123['exact_match_ratio'] - res_v1_l123['exact_match_ratio']:+5.2f}%p")
    print(f"{'Hamming Loss (개별 오류율)':<32} | {res_v1_l123['hamming_loss']:6.2f}%              | {res_v2_l123['hamming_loss']:6.2f}%                | {res_v2_l123['hamming_loss'] - res_v1_l123['hamming_loss']:+5.2f}%p")
    print(f"{'개별 슬롯 정답률 (Slot Acc)':<32} | {res_v1_l123['slot_accuracy']:6.2f}%              | {res_v2_l123['slot_accuracy']:6.2f}%                | {res_v2_l123['slot_accuracy'] - res_v1_l123['slot_accuracy']:+5.2f}%p")
    print(f"{'Macro-averaged F1':<32} | {res_v1_l123['macro_f1']:6.2f}%              | {res_v2_l123['macro_f1']:6.2f}%                | {res_v2_l123['macro_f1'] - res_v1_l123['macro_f1']:+5.2f}%p")
    print(f"{'Micro-averaged F1':<32} | {res_v1_l123['micro_f1']:6.2f}%              | {res_v2_l123['micro_f1']:6.2f}%                | {res_v2_l123['micro_f1'] - res_v1_l123['micro_f1']:+5.2f}%p")
    print(f"{'정상 오탐 건수 (Clean FP / 115건)':<32} | {res_v1_l123['soft_fusion_079']['clean_fps']}건                    | {res_v2_l123['soft_fusion_079']['clean_fps']}건                      | {res_v2_l123['soft_fusion_079']['clean_fps'] - res_v1_l123['soft_fusion_079']['clean_fps']:+2d}건")

    out_data = {
        'full1906': {
            'v1': res_v1_all,
            'v2': res_v2_all
        },
        'level123': {
            'v1': res_v1_l123,
            'v2': res_v2_l123
        }
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 통합 모델 v1 vs v2 벤치마크 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()

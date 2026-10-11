# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_babel_squat_integrated_benchmark.py
Purpose: 바벨 스쿼트 4대 세부 결함(c1~c4) 통합 추론 및 실서비스 다중 라벨 종합 평가 벤치마크
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 OOF 평가
  3) Hard OR vs Soft Risk-Fusion 비교 및 최적 임계값(θ_clean) 정밀 튜닝
  4) Exact Match Ratio, Hamming Loss, 결함 복합 발생별 적중률, 카메라 5대 시점별 편차 전수 측정
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
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v1 import BabelSquatC1ModelV1
from src.models.second_models.babel_squat.c2_model_v3 import V2_TO_V3_INDICES as C2_INDICES
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES
from src.models.second_models.babel_squat.c4_model_v3 import V3_INDICES_FROM_26D as C4_INDICES
from src.models.second_models.babel_squat.babel_squat_integrated_v1 import BabelSquatIntegratedEngineV1, DEFAULT_RISK_WEIGHTS

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/squat_integrated_benchmark_results.json'

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

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 4대 세부 결함(c1~c4) 실서비스 통합 평가 벤치마크")
    print("  ★ 5-Fold StratifiedGroupKFold 완전 격리 환경 OOF 종합 검증")
    print("  ★ Hard OR vs Soft Risk-Fusion 비교 + 최적 임계값 튜닝 + 다중 라벨 진단 지표 산출")
    print("=" * 110)

    # 1. 원천 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int) # (1906, 4) [c1, c2, c3, c4]
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    camera_views = m['views'][sq_mask]
    subjects = ds['subjects']
    N_total = len(Y_defects)

    # 실제 종합 라벨: 4개 중 1개라도 결함이면 1, 4개 모두 0이면 0 (Clean)
    y_overall_true = (np.sum(Y_defects, axis=1) > 0).astype(int)

    print(f"[*] 총 스쿼트 시퀀스: {N_total}건")
    print(f"    - 실제 완전 정상 (Clean, 0 defects) : {np.sum(y_overall_true == 0):4d}건 ({np.sum(y_overall_true == 0)/N_total*100:.1f}%)")
    print(f"    - 실제 결함 포함 (Defect, ≥1 defect): {np.sum(y_overall_true == 1):4d}건 ({np.sum(y_overall_true == 1)/N_total*100:.1f}%)")

    # 2. 각 모델별 피처셋 로드 및 정렬
    print("\n[*] 4대 모델별 검증된 피처셋 로드 중...")

    # c1 피처 (22D)
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)
    feats_c1_22d = BabelSquatC1ModelV1.extract_features(X_b_64, use_pruned=True)

    # c2 피처 (13D)
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names_c2 = list(h_data['feature_names'])
    elite_idx_c2 = [i for i, n in enumerate(feat_names_c2) if n != 'c2_gaze_hysteresis_rmse']
    X_18d_c2 = X_19d[:, elite_idx_c2]
    g_data_c2 = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats_c2 = g_data_c2['gated_feats'].astype(np.float32)[:, :2]
    feats_c2_20d = np.concatenate([X_18d_c2, gated_feats_c2], axis=1)
    feats_c2_13d = feats_c2_20d[:, C2_INDICES]

    # c3 피처 (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d_c3 = v1_data_c3['feats'].astype(np.float32)
    foot_data_c3 = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d_c3 = foot_data_c3['foot_feats'].astype(np.float32)
    feats_c3_24d = np.concatenate([X_16d_c3, X_foot_8d_c3], axis=1)
    feats_c3_20d = feats_c3_24d[:, C3_INDICES]

    # c4 피처 (9D)
    v1_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d_c4 = v1_data_c4['feats'].astype(np.float32)
    foot_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d_c4 = foot_data_c4['foot_feats'].astype(np.float32)
    feats_c4_26d = np.concatenate([X_16d_c4, X_foot_10d_c4], axis=1)
    feats_c4_9d = feats_c4_26d[:, C4_INDICES]

    print(f"    - c1 척추 피처: shape = {feats_c1_22d.shape}")
    print(f"    - c2 고개 피처: shape = {feats_c2_13d.shape}")
    print(f"    - c3 무릎 피처: shape = {feats_c3_20d.shape}")
    print(f"    - c4 발 피처  : shape = {feats_c4_9d.shape}")

    # 3. 5-Fold StratifiedGroupKFold 완전 격리 OOF 예측 확률 산출
    print("\n[*] 5-Fold StratifiedGroupKFold 완전 격리 환경 OOF 확률 산출 중...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = {
        'c1': np.zeros(N_total, dtype=np.float32),
        'c2': np.zeros(N_total, dtype=np.float32),
        'c3': np.zeros(N_total, dtype=np.float32),
        'c4': np.zeros(N_total, dtype=np.float32)
    }

    feature_dict = {
        'c1': feats_c1_22d,
        'c2': feats_c2_13d,
        'c3': feats_c3_20d,
        'c4': feats_c4_9d
    }

    t0 = time.time()
    for col_idx, code in enumerate(['c1', 'c2', 'c3', 'c4']):
        y_col = Y_defects[:, col_idx]
        X_col = feature_dict[code]

        for fold, (trn, val) in enumerate(sgkf.split(X_col, y_col, groups=subjects)):
            cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
            cb.fit(X_col[trn], y_col[trn])
            p_cb = cb.predict_proba(X_col[val])[:, 1]

            lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_col[trn], y_col[trn])
            p_lgb = lgb_m.predict_proba(X_col[val])[:, 1]

            rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42+fold, n_jobs=-1)
            rf.fit(X_col[trn], y_col[trn])
            p_rf = rf.predict_proba(X_col[val])[:, 1]

            # 앙상블 캐스케이드 (CatBoost 50% + LightGBM 30% + RF 20%)
            oof_probs[code][val] = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

    print(f"[*] 4대 모델 OOF 확률 산출 완료 ({time.time() - t0:.1f}초)")

    # 4. 개별 모델의 OOF 성능 평가
    print("\n" + "=" * 80)
    print("  [개별 모델 성능 확인] 4대 결함별 5-Fold OOF 개별 성능 (Threshold = 0.50)")
    print("=" * 80)
    print(f"{'결함 코드':<8} | {'결함 명칭':<24} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<10}")
    print("-" * 80)
    single_metrics = {}
    for col_idx, code in enumerate(['c1', 'c2', 'c3', 'c4']):
        m_res = compute_binary_metrics(Y_defects[:, col_idx], oof_probs[code], threshold=0.50)
        single_metrics[code] = m_res
        cname = ['척추의 중립', '고개 정면', '발-무릎 일치', '발바닥 지면'][col_idx]
        print(f"{code:<8} | {cname:<24} | {m_res['f1']:6.2f}%    | {m_res['precision']:6.2f}%    | {m_res['recall']:6.2f}%    | {m_res['roc_auc']:6.2f}%")
    print("=" * 80)

    # 5. [비교 실험] Hard OR 결합 vs Soft Risk Fusion 결합
    print("\n" + "=" * 90)
    print("  [비교 분석 1] Hard OR 결합 vs Soft Risk Fusion 결합 비교 (오탐 누적 문제 검증)")
    print("=" * 90)

    # (1) Hard OR 결합: c1~c4 중 하나라도 >= 0.50 이면 결함 (1)
    hard_or_preds = (
        (oof_probs['c1'] >= 0.50) |
        (oof_probs['c2'] >= 0.50) |
        (oof_probs['c3'] >= 0.50) |
        (oof_probs['c4'] >= 0.50)
    ).astype(int)

    hard_or_f1 = f1_score(y_overall_true, hard_or_preds) * 100.0
    hard_or_prec = precision_score(y_overall_true, hard_or_preds) * 100.0
    hard_or_rec = recall_score(y_overall_true, hard_or_preds) * 100.0
    hard_or_acc = accuracy_score(y_overall_true, hard_or_preds) * 100.0
    tn_or, fp_or, fn_or, tp_or = confusion_matrix(y_overall_true, hard_or_preds).ravel()
    spec_or = (tn_or / (tn_or + fp_or)) * 100.0

    print(f"[*] (A) 단순 Hard OR 결합 방식:")
    print(f"    - Overall F1       : {hard_or_f1:.2f}%")
    print(f"    - Precision        : {hard_or_prec:.2f}%")
    print(f"    - Recall           : {hard_or_rec:.2f}% (결함 탐지율은 매우 높음)")
    print(f"    - Specificity (정상맞춤률): {spec_or:.2f}% (★ 치명적: 정상인 중 {fp_or}건 오탐, 오탐률 {100-spec_or:.1f}%)")
    print(f"    - 혼동 행렬        : TN={tn_or}, FP={fp_or} (오탐), FN={fn_or}, TP={tp_or}")

    # (2) Soft Risk-Fusion 종합 점수 산출
    # Defect_Score = 1 - ∏ (1 - clip(w_k * P_k, 0, 1))
    weights = DEFAULT_RISK_WEIGHTS
    prod_clean = (
        (1.0 - np.clip(weights['c1'] * oof_probs['c1'], 0.0, 1.0)) *
        (1.0 - np.clip(weights['c2'] * oof_probs['c2'], 0.0, 1.0)) *
        (1.0 - np.clip(weights['c3'] * oof_probs['c3'], 0.0, 1.0)) *
        (1.0 - np.clip(weights['c4'] * oof_probs['c4'], 0.0, 1.0))
    )
    defect_scores = 1.0 - prod_clean

    # 6. 최적 임계값(θ_clean) 튜닝
    print("\n" + "=" * 90)
    print("  [최적 임계값 튜닝] Soft Risk-Fusion 종합 점수 임계값(θ_clean) 스윕")
    print("=" * 90)

    best_th = 0.50
    best_f1 = 0.0
    best_bacc = 0.0
    sweep_results = []

    # 임계값 0.20 ~ 0.80 스윕
    for th in np.linspace(0.20, 0.80, 61):
        preds = (defect_scores >= th).astype(int)
        f = f1_score(y_overall_true, preds, zero_division=0) * 100.0
        bacc = balanced_accuracy_score(y_overall_true, preds) * 100.0
        acc = accuracy_score(y_overall_true, preds) * 100.0
        tn, fp, fn, tp = confusion_matrix(y_overall_true, preds).ravel()
        spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

        sweep_results.append({
            'threshold': round(float(th), 2),
            'f1': round(float(f), 2),
            'bacc': round(float(bacc), 2),
            'accuracy': round(float(acc), 2),
            'specificity': round(float(spec), 2),
            'fp': int(fp), 'fn': int(fn)
        })

        # Balanced Accuracy와 F1을 모두 고려하여 최적 지점 선정 (특히 불균형 데이터셋에서 정상 맞춤률 Specificity 확보)
        score_combined = 0.6 * f + 0.4 * bacc
        if score_combined > best_bacc:
            best_bacc = score_combined
            best_th = th
            best_f1 = f

    print(f"[*] 최적 임계값 도출: θ_clean = {best_th:.2f} (Overall F1 = {best_f1:.2f}%)")

    # 최적 임계값 적용 결과
    soft_fusion_metrics = compute_binary_metrics(y_overall_true, defect_scores, threshold=best_th)
    print(f"[*] (B) Soft Risk-Fusion (θ = {best_th:.2f}) 적용 결과:")
    print(f"    - Overall F1       : {soft_fusion_metrics['f1']:.2f}%")
    print(f"    - Precision        : {soft_fusion_metrics['precision']:.2f}%")
    print(f"    - Recall           : {soft_fusion_metrics['recall']:.2f}%")
    print(f"    - Specificity (정상맞춤률): {soft_fusion_metrics['specificity']:.2f}% (정상 정답 TN {soft_fusion_metrics['tn']}건, 오탐 {soft_fusion_metrics['fp']}건)")
    print(f"    - Balanced Accuracy: {soft_fusion_metrics['bacc']:.2f}%")
    print(f"    - ROC-AUC          : {soft_fusion_metrics['roc_auc']:.2f}%")
    print(f"    - 혼동 행렬        : TN={soft_fusion_metrics['tn']}, FP={soft_fusion_metrics['fp']}, FN={soft_fusion_metrics['fn']}, TP={soft_fusion_metrics['tp']}")

    # 7. 다중 라벨(Multi-label) 세부 결함 진단 지표 산출
    print("\n" + "=" * 90)
    print("  [다중 라벨 진단 평가] Exact Match Ratio, Hamming Loss, Macro/Micro F1")
    print("=" * 90)

    # 4개 결함 각각의 이진 판정 (N, 4)
    preds_multi = np.column_stack([
        (oof_probs['c1'] >= 0.50).astype(int),
        (oof_probs['c2'] >= 0.50).astype(int),
        (oof_probs['c3'] >= 0.50).astype(int),
        (oof_probs['c4'] >= 0.50).astype(int)
    ])

    # (1) Exact Match Ratio (Subset Accuracy): 4개 슬롯이 100% 모두 일치한 비율
    exact_matches = np.all(preds_multi == Y_defects, axis=1)
    exact_match_ratio = np.mean(exact_matches) * 100.0

    # (2) Hamming Loss (개별 슬롯 오류율)
    h_loss = hamming_loss(Y_defects, preds_multi) * 100.0

    # (3) Macro F1 & Micro F1
    macro_f1 = np.mean([single_metrics[k]['f1'] for k in ['c1', 'c2', 'c3', 'c4']])
    micro_f1 = f1_score(Y_defects.ravel(), preds_multi.ravel(), average='micro') * 100.0

    print(f"[*] Exact Match Ratio (4개 결함 100% 완전 일치율) : {exact_match_ratio:.2f}% (총 {np.sum(exact_matches)}/{N_total}건)")
    print(f"[*] Hamming Loss (개별 결함 슬롯 오류율)          : {h_loss:.2f}% (슬롯 정답률 {100-h_loss:.2f}%)")
    print(f"[*] Macro-averaged F1 (4개 결함 평균 F1)          : {macro_f1:.2f}%")
    print(f"[*] Micro-averaged F1 (전체 라벨 풀링 F1)         : {micro_f1:.2f}%")

    # 8. 결함 복합 발생 빈도(0개~4개)별 적중률 분석
    print("\n" + "=" * 90)
    print("  [복합 결함 빈도별 분석] 0개(Clean), 1개, 2개, 3개, 4개 결함 동시 발생별 성능")
    print("=" * 90)

    defect_counts_true = np.sum(Y_defects, axis=1)
    counts_analysis = {}

    print(f"{'결함 개수':<10} | {'샘플 수':<10} | {'종합 판정 정확도':<16} | {'4개 완전 일치(Exact Match)':<22} | {'평균 슬롯 정확도'}")
    print("-" * 80)
    for k in range(5):
        mask_k = (defect_counts_true == k)
        n_k = np.sum(mask_k)
        if n_k == 0:
            continue

        # 종합 판정(정상:0, 결함:1) 정확도
        overall_acc_k = np.mean(y_overall_true[mask_k] == (defect_scores[mask_k] >= best_th).astype(int)) * 100.0
        # 4개 완전 일치율
        exact_k = np.mean(exact_matches[mask_k]) * 100.0
        # 슬롯 일치율
        slot_acc_k = (1.0 - hamming_loss(Y_defects[mask_k], preds_multi[mask_k])) * 100.0

        counts_analysis[str(k)] = {
            'count': int(n_k),
            'overall_accuracy': round(float(overall_acc_k), 2),
            'exact_match_ratio': round(float(exact_k), 2),
            'slot_accuracy': round(float(slot_acc_k), 2)
        }
        k_desc = f"{k}개 발생" + (" (Clean)" if k==0 else "")
        print(f"{k_desc:<10} | {n_k:4d}건     | {overall_acc_k:6.2f}%          | {exact_k:6.2f}% ({np.sum(exact_matches[mask_k])}/{n_k})       | {slot_acc_k:6.2f}%")
    print("=" * 80)

    # 9. 카메라 5대 시점(view1~view5)별 성능 편차 분석
    print("\n" + "=" * 90)
    print("  [카메라 시점별 분석] 5대 시점(view1~view5) 전수 성능 편차 검증")
    print("=" * 90)

    view_analysis = {}
    print(f"{'카메라 시점':<12} | {'시점 설명':<16} | {'샘플수':<8} | {'종합 F1':<10} | {'Exact Match':<12} | {'Hamming Loss'}")
    print("-" * 75)
    view_desc_map = {
        'view1': '후방 시점 (Rear)',
        'view2': '우측 45도 (Right 45°)',
        'view3': '정면 시점 (Frontal)',
        'view4': '좌측 45도 (Left 45°)',
        'view5': '우측 후방 (Right Rear)'
    }

    for v_name in sorted(np.unique(camera_views)):
        v_mask = (camera_views == v_name)
        v_n = np.sum(v_mask)
        v_f1 = f1_score(y_overall_true[v_mask], (defect_scores[v_mask] >= best_th).astype(int), zero_division=0) * 100.0
        v_exact = np.mean(exact_matches[v_mask]) * 100.0
        v_hloss = hamming_loss(Y_defects[v_mask], preds_multi[v_mask]) * 100.0

        view_analysis[v_name] = {
            'samples': int(v_n),
            'overall_f1': round(float(v_f1), 2),
            'exact_match': round(float(v_exact), 2),
            'hamming_loss': round(float(v_hloss), 2)
        }
        v_desc = view_desc_map.get(v_name, '기타')
        print(f"{v_name:<12} | {v_desc:<16} | {v_n:4d}건   | {v_f1:6.2f}%    | {v_exact:6.2f}%       | {v_hloss:6.2f}%")
    print("=" * 75)

    # 10. 종합 벤치마크 결과 JSON 저장
    full_results = {
        'total_samples': N_total,
        'clean_samples': int(np.sum(y_overall_true == 0)),
        'defect_samples': int(np.sum(y_overall_true == 1)),
        'single_model_metrics': single_metrics,
        'hard_or_comparison': {
            'f1': round(float(hard_or_f1), 2),
            'precision': round(float(hard_or_prec), 2),
            'recall': round(float(hard_or_rec), 2),
            'accuracy': round(float(hard_or_acc), 2),
            'specificity': round(float(spec_or), 2),
            'tn': int(tn_or), 'fp': int(fp_or), 'fn': int(fn_or), 'tp': int(tp_or)
        },
        'soft_fusion_metrics': soft_fusion_metrics,
        'best_threshold_clean': round(float(best_th), 2),
        'multi_label_metrics': {
            'exact_match_ratio': round(float(exact_match_ratio), 2),
            'hamming_loss': round(float(h_loss), 2),
            'macro_f1': round(float(macro_f1), 2),
            'micro_f1': round(float(micro_f1), 2)
        },
        'counts_analysis': counts_analysis,
        'view_analysis': view_analysis
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 통합 벤치마크 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()

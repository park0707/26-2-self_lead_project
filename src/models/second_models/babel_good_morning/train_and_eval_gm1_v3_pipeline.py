# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 굿모닝 (09_굿모닝)
Defect: 무릎 구부린 채 고정 위반 (gm1)
Script: train_and_eval_gm1_v3_pipeline.py
Purpose:
  1. gm1 v2 최적화 13D 피처셋을 기반으로 모델링 고도화
  2. Conformal Prediction 10% Bounded 3-State 완충망 도입 (Label Noise 극복)
  3. CatBoost 하이퍼파라미터 미세 조정 (Depth 5, L2 3, LR 0.03, Iter 300)
  4. v2 vs v3 (2-State 및 3-State) 성능 비교 평가 보고
"""

import os
import sys
import json
import pickle
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path: sys.path.insert(0, 'd:/')

from src.models.second_models.babel_stiff_deadlift.stiff_deadlift_knee_features_v6 import (
    extract_knee_invariance_features_v6,
    FEATURE_NAMES_19D
)

CACHE_PATH = os.path.join(CURRENT_DIR, "gm1_dataset_cache.npz")
V2_JSON_PATH = os.path.join(CURRENT_DIR, "gm1_v2_benchmark_results.json")
V3_JSON_PATH = os.path.join(CURRENT_DIR, "gm1_v3_benchmark_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "weights", "gm1_v3_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]

# GM1 v3 정예 피처 (13D) - v2와 동일
GM1_V3_FEATURE_INDICES = [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 18]
GM1_V3_FEATURE_NAMES = [FEATURE_NAMES_19D[i] for i in GM1_V3_FEATURE_INDICES]

def run_pipeline():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 굿모닝 gm1_v3 훈련 및 3-State 벤치마크 평가", flush=True)
    print("=" * 100, flush=True)

    data = np.load(CACHE_PATH, allow_pickle=True)
    train_labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    X_seqs = data['X']

    N_valid = len(train_labels)

    print("[*] GM1 특화 13D 생체역학 피처 추출 중...", flush=True)
    feats_19d_all = np.zeros((N_valid, 19), dtype=np.float32)
    for i in range(N_valid):
        feats_19d_all[i] = extract_knee_invariance_features_v6(X_seqs[i], p_sd3=0.0)

    meta_features = feats_19d_all[:, GM1_V3_FEATURE_INDICES]
    
    print(f"[*] 피처 추출 완료. 샘플 수={N_valid}, 정상={np.sum(train_labels==0)}, 결함={np.sum(train_labels==1)}")

    print("\n" + "-" * 80, flush=True)
    print("  [피험자 격리 5-Fold 교차검증 (StratifiedGroupKFold, 9 Subjects)]", flush=True)
    print("-" * 80, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_proba = np.zeros(N_valid, dtype=np.float32)
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(np.arange(N_valid), train_labels, subject_ids)):
        X_tr, y_tr = meta_features[tr_idx], train_labels[tr_idx]
        X_val, y_val = meta_features[val_idx], train_labels[val_idx]

        # 하이퍼파라미터 최적화 적용 (v3)
        cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        )
        cb.fit(X_tr, y_tr)
        fold_models.append(cb)

        val_preds = cb.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds) * 100.0
        print(f"  [Fold {fold+1}/5] Val AUC: {fold_auc:.2f}%", flush=True)

    # 5. 2-State 전수 평가
    preds_2s = (oof_proba >= 0.50).astype(int)
    auc_l13 = roc_auc_score(train_labels, oof_proba)
    f1_macro_l13 = f1_score(train_labels, preds_2s, average='macro', zero_division=0)
    f1_clean_l13 = f1_score(train_labels, preds_2s, pos_label=0, zero_division=0)
    f1_defect_l13 = f1_score(train_labels, preds_2s, pos_label=1, zero_division=0)
    tn_l13, fp_l13, fn_l13, tp_l13 = confusion_matrix(train_labels, preds_2s).ravel()
    spec_l13 = tn_l13 / (tn_l13 + fp_l13)
    acc_l13 = accuracy_score(train_labels, preds_2s)
    
    print("\n[★ 2-State 진단 성능 (v3 Base)]:")
    print(f"  - 전체 ROC-AUC:        {auc_l13*100:.2f}%")
    print(f"  - 결함 F1-score:       {f1_defect_l13*100:.2f}%")
    print(f"  - 정상 F1:             {f1_clean_l13*100:.2f}%")
    print(f"  - Macro F1:            {f1_macro_l13*100:.2f}%")

    # 6. Conformal Prediction 기반 10% Bounded 3-State 완충망 평가
    print("\n" + "-" * 80, flush=True)
    print("  [Conformal Prediction 기반 10% Bounded 3-State 캘리브레이션]", flush=True)
    print("-" * 80, flush=True)

    best_delta = 0.01
    best_abstention = 0.0
    for delta in np.linspace(0.005, 0.20, 500):
        t_low = 0.50 - delta
        t_high = 0.50 + delta
        abst = np.mean((oof_proba >= t_low) & (oof_proba < t_high)) * 100.0
        if abst <= 10.0:
            best_delta = delta
            best_abstention = abst

    tau_lower = float(0.50 - best_delta)
    tau_upper = float(0.50 + best_delta)

    states = np.zeros(N_valid, dtype=int)
    states[oof_proba >= tau_upper] = 1 # Defect
    states[oof_proba < tau_lower] = 0  # Clean
    states[(oof_proba >= tau_lower) & (oof_proba < tau_upper)] = 2 # Uncertain

    abstention_rate = float(np.mean(states == 2) * 100.0)
    coverage_rate = float(100.0 - abstention_rate)
    decided_mask = (states != 2)

    y_dec = train_labels[decided_mask]
    pred_dec = states[decided_mask]

    sel_f1_macro = float(f1_score(y_dec, pred_dec, average='macro', zero_division=0))
    sel_f1_clean = float(f1_score(y_dec, pred_dec, pos_label=0, zero_division=0))
    sel_f1_defect = float(f1_score(y_dec, pred_dec, pos_label=1, zero_division=0))
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_dec, pred_dec).ravel()
    sel_spec = float(tn_s / (tn_s + fp_s))
    sel_acc = float(accuracy_score(y_dec, pred_dec))

    print(f"[*] 3-State 밴드: tau_lower={tau_lower:.4f}, tau_upper={tau_upper:.4f}")
    print(f"[*] 평가 보류율: {abstention_rate:.2f}% (<= 10.0% PASS) | 진단 커버리지: {coverage_rate:.2f}%")
    print(f"\n[★ 3-State 선택적 진단 (v3 Selective) 성적]:")
    print(f"  - 3-State 결함 F1-score: {sel_f1_defect*100:.2f}%")
    print(f"  - 3-State 정상 F1:       {sel_f1_clean*100:.2f}%")
    print(f"  - 3-State Macro F1:      {sel_f1_macro*100:.2f}%")
    print(f"  - 3-State Specificity:   {sel_spec*100:.2f}%")

    # 7. v2 vs v3 비교
    if os.path.exists(V2_JSON_PATH):
        with open(V2_JSON_PATH, 'r', encoding='utf-8') as f:
            v2_data = json.load(f)
        v2_metrics = v2_data['metrics']
        
        print("\n" + "=" * 100, flush=True)
        print("  [★ v2 (2-State) vs v3 (2-State) vs v3 (3-State Selective) 성능 비교]", flush=True)
        print("=" * 100, flush=True)
        print(f"  {'평가지표':22s} | {'v2 (2-State)':15s} | {'v3 (2-State)':15s} | {'v3 (3-State)':15s}", flush=True)
        print("-" * 100, flush=True)
        
        print(f"  {'결함 F1-score':22s} | {v2_metrics['defect_f1']*100:14.2f}% | {f1_defect_l13*100:14.2f}% | {sel_f1_defect*100:14.2f}%", flush=True)
        print(f"  {'Macro F1':22s} | {v2_metrics['macro_f1']*100:14.2f}% | {f1_macro_l13*100:14.2f}% | {sel_f1_macro*100:14.2f}%", flush=True)
        print(f"  {'Clean Specificity':22s} | {v2_metrics['clean_specificity']*100:14.2f}% | {spec_l13*100:14.2f}% | {sel_spec*100:14.2f}%", flush=True)
        print(f"  {'전체 ROC-AUC':22s} | {v2_metrics['roc_auc']*100:14.2f}% | {auc_l13*100:14.2f}% | {'N/A':>14s}", flush=True)
    else:
        print("v2 결과 JSON이 없어 비교를 생략합니다.")

    # 8. 공식 5대 시점별 성적 (3-State 기준)
    print("\n" + "-" * 95, flush=True)
    print("  [공식 5대 시점별 결함 F1 (3-State Selective)]", flush=True)
    print("-" * 95, flush=True)
    view_results = {}
    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (view_ids == vn)
        v_dec = decided_mask[v_mask]
        y_v = train_labels[v_mask][v_dec]
        st_v = states[v_mask][v_dec]
        if len(y_v) > 0:
            v_f1_d = float(f1_score(y_v, st_v, pos_label=1, zero_division=0) * 100.0)
            v_f1_m = float(f1_score(y_v, st_v, average='macro', zero_division=0) * 100.0)
            view_results[vn] = {
                'alias': v_alias,
                'count': int(np.sum(v_mask)),
                'defect_f1': v_f1_d,
                'macro_f1': v_f1_m
            }
            print(f"  * {v_alias:25s}: 결함 F1={v_f1_d:5.2f}% | Macro F1={v_f1_m:5.2f}% (N={np.sum(v_mask)})", flush=True)

    # 10. JSON 저장
    benchmark_data = {
        'exercise': '09_굿모닝',
        'defect': '무릎 구부린 채 고정 위반 (gm1_v3)',
        'solutions_applied': [
            "Conformal Prediction 10% Bounded 3-State 완충망",
            "CatBoost Hyperparameter Tuning"
        ],
        'tier2_2state_metrics': {
            'macro_f1': float(f1_macro_l13),
            'clean_f1': float(f1_clean_l13),
            'defect_f1': float(f1_defect_l13),
            'clean_specificity': float(spec_l13),
            'roc_auc': float(auc_l13),
            'accuracy': float(acc_l13)
        },
        'tier2_3state_metrics': {
            'tau_lower': float(tau_lower),
            'tau_upper': float(tau_upper),
            'abstention_rate': float(abstention_rate),
            'coverage_rate': float(coverage_rate),
            'selective_macro_f1': float(sel_f1_macro),
            'selective_clean_f1': float(sel_f1_clean),
            'selective_defect_f1': float(sel_f1_defect),
            'selective_specificity': float(sel_spec),
            'selective_accuracy': float(sel_acc)
        },
        'view_results': view_results
    }

    with open(V3_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] gm1_v3 벤치마크 JSON 저장 완료: {V3_JSON_PATH}", flush=True)

    # 11. 가중치 번들 저장
    os.makedirs(os.path.dirname(BUNDLE_SAVE_PATH), exist_ok=True)
    bundle = {
        'version': 'gm1_v3',
        'fold_models': fold_models,
        'feature_indices': GM1_V3_FEATURE_INDICES,
        'feature_names': GM1_V3_FEATURE_NAMES,
        'thresholds': {'tau_lower': tau_lower, 'tau_upper': tau_upper}
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] gm1_v3 공식 가중치 번들 배포 완료: {BUNDLE_SAVE_PATH}", flush=True)

if __name__ == '__main__':
    run_pipeline()

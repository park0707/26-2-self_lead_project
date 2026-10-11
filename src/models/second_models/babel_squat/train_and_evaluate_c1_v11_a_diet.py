# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c1_v11_a_diet.py
Purpose: c1(척추의 중립 위반) v11_a_diet (27D 정예 피처셋) 모델 5-Fold 피험자 격리 학습 및 전수 실측 평가
"""

import os
import sys
import time
import json
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_a_diet import (
    extract_c1_v11_a_diet_features_from_seq,
    FEATURE_NAMES_27_C1_DIET,
    BabelSquatC1ModelV11ADiet
)

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_38D_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_38d_feats.npz"
FEATS_DIET_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_diet_27d_feats.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v11_a_diet_weights.pkl"
RESULTS_JSON_PATH = r"d:\src\models\second_models\babel_squat\c1_v11_a_diet_evaluation_results.json"


def evaluate_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2)
    }


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1 v11_a_diet (27D 정예 피처셋) 공식 모델 학습 & 전수 실측 평가", flush=True)
    print("  ★ v10 28D 중 노이즈 2개(dwell, harmony) 제거 + E0 핵심 1개(lumbopelvic_sd) 추가 (총 27D)", flush=True)
    print("  ★ 2,605개 마스터 데이터셋 5-Fold StratifiedGroupKFold 피험자 완전 격리", flush=True)
    print("  ★ CatBoost 60% + LightGBM 40% 앙상블 및 3-State 선택적 거부 실측", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c1 = master['Y_defects_golden'][:, 0].astype(int)
    N = len(seq_ids)

    # 38D 캐시로부터 27D 정예 피처 슬라이싱 구성
    # Diet D: v10 [0..27] 중 2, 21 제외 (26개) + E0 [37] (1개) = 총 27개
    indices_diet = [i for i in range(28) if i not in [2, 21]] + [37]
    
    if os.path.exists(FEATS_DIET_PATH):
        print(f"[*] 기존 27D 캐시 로드: {FEATS_DIET_PATH}", flush=True)
        X_27d = np.load(FEATS_DIET_PATH, allow_pickle=True)['X_c1_v11_a_diet']
    else:
        print(f"[*] 38D 캐시로부터 27D 정예 피처셋 생성 중...", flush=True)
        cached_38 = np.load(FEATS_38D_PATH, allow_pickle=True)['X_c1_v11_a']
        X_27d = cached_38[:, indices_diet]
        np.savez_compressed(
            FEATS_DIET_PATH,
            X_c1_v11_a_diet=X_27d,
            feature_names=FEATURE_NAMES_27_C1_DIET,
            sequence_ids=seq_ids
        )
        print(f"[*] 27D 캐시 저장 완료: {FEATS_DIET_PATH} shape={X_27d.shape}", flush=True)

    print(f"\n[*] [2/3] 5-Fold StratifiedGroupKFold 교차 검증 및 앙상블 학습 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(N, dtype=np.float32)
    cb_models = []
    lgb_models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_27d, y_c1, groups=subjects), 1):
        X_train, y_train = X_27d[train_idx], y_c1[train_idx]
        X_val, y_val = X_27d[val_idx], y_c1[val_idx]
        val_sub = np.unique(subjects[val_idx])

        cb = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            thread_count=-1,
            verbose=False
        )
        cb.fit(X_train, y_train)

        clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            colsample_bytree=0.85,
            subsample=0.85,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        p_cb_val = cb.predict_proba(X_val)[:, 1]
        p_lgb_val = clf.predict_proba(X_val)[:, 1]
        p_ens_val = 0.60 * p_cb_val + 0.40 * p_lgb_val

        oof_probs[val_idx] = p_ens_val
        cb_models.append(cb)
        lgb_models.append(clf)

        m_fold = evaluate_metrics(y_val, p_ens_val)
        fold_metrics.append(m_fold)
        print(f"    - Fold {fold} (검증 피험자 {val_sub}): F1={m_fold['f1']:.2f}%, AUC={m_fold['auc']:.2f}%, Rec={m_fold['rec']:.2f}%, Prec={m_fold['prec']:.2f}%", flush=True)

    # v11_a_diet 모델 객체 생성 및 가중치 저장
    model_diet = BabelSquatC1ModelV11ADiet()
    model_diet.cb_models = cb_models
    model_diet.lgb_models = lgb_models
    model_diet.is_fitted = True
    model_diet.save_weights(WEIGHTS_PATH)

    # 전체 OOF 성능 산출
    m_diet = evaluate_metrics(y_c1, oof_probs)

    v10_f1, v10_auc, v10_prec, v10_rec, v10_acc = 79.73, 88.33, 80.00, 79.46, 80.38
    v11_a_f1 = 79.56

    print("\n" + "=" * 115, flush=True)
    print("  [★ c1 모델 세대별 실측 성능 직접 대조 (N=2,605 전수, 5-Fold 피험자 완전 격리)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'지표 (Metric)':<18} | {'v10 (28D 베이스)':<18} | {'v11_a (E0 38D)':<18} | {'v11_a_diet (27D 정예)':<22} | {'v10 대비 증감'}", flush=True)
    print("-" * 115, flush=True)
    for name, v10_val, v11_val, diet_val in [
        ("F1-Score", v10_f1, v11_a_f1, m_diet['f1']),
        ("ROC-AUC", v10_auc, 87.83, m_diet['auc']),
        ("Precision", v10_prec, 81.35, m_diet['prec']),
        ("Recall", v10_rec, 77.85, m_diet['rec']),
        ("Accuracy", v10_acc, 80.73, m_diet['acc'])
    ]:
        delta = diet_val - v10_val
        delta_str = f"+{delta:.2f}%p" if delta >= 0 else f"{delta:.2f}%p"
        print(f"{name:<18} | {v10_val:>16.2f}% | {v11_val:>16.2f}% | {diet_val:>20.2f}% | {delta_str:>12}", flush=True)

    # 5대 시점별 세부 성능 감사
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (후방 180°)', 'view2 (우측 45°)', 'view3 (정면 0°)', 'view4 (좌측 45°)', 'view5 (우측 후방 45°)']
    v10_view_f1s = {'view1': 76.12, 'view2': 83.21, 'view3': 74.30, 'view4': 81.85, 'view5': 83.18}
    v11_a_view_f1s = {'view1': 75.20, 'view2': 80.99, 'view3': 81.74, 'view4': 81.18, 'view5': 78.69}

    print("\n" + "=" * 115, flush=True)
    print("  [5대 시점별 세부 성능 감사 (Multi-View Performance Audit)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (카메라)':<22} | {'v10 F1':<10} | {'v11_a F1':<10} | {'v11_a_diet F1':<15} | {'v10 대비 증감':<14} | {'diet AUC':<10} | {'diet Acc'}", flush=True)
    print("-" * 115, flush=True)

    view_results = {}
    for v, vn in zip(view_list, view_names):
        mask = (views == v)
        mv = evaluate_metrics(y_c1[mask], oof_probs[mask])
        v10_vf1 = v10_view_f1s.get(v, 0.0)
        v11_vf1 = v11_a_view_f1s.get(v, 0.0)
        d_f1 = mv['f1'] - v10_vf1
        d_f1_str = f"+{d_f1:.2f}%p" if d_f1 >= 0 else f"{d_f1:.2f}%p"
        print(f"{vn:<22} | {v10_vf1:>8.2f}% | {v11_vf1:>8.2f}% | {mv['f1']:>13.2f}% | {d_f1_str:>12} | {mv['auc']:>8.2f}% | {mv['acc']:>6.2f}%", flush=True)
        view_results[v] = {
            'view_name': vn,
            'count': int(np.sum(mask)),
            'v10_f1': v10_vf1,
            'v11_a_f1': v11_vf1,
            'diet_f1': mv['f1'],
            'delta_f1': round(d_f1, 2),
            'diet_auc': mv['auc'],
            'diet_acc': mv['acc']
        }

    # 3-State 선택적 거부 평가
    print("\n" + "=" * 115, flush=True)
    print("  [3-State 선택적 거부 (Selective Abstention) Risk-Coverage 비교]", flush=True)
    print("=" * 115, flush=True)

    threshold_pairs = [
        (0.50, 0.50, "Full Decision (강제 이진)"),
        (0.40, 0.60, "Balanced Abstention (표준 추천)"),
        (0.35, 0.65, "Standard Selective"),
        (0.30, 0.70, "High Confidence")
    ]
    print(f"{'임계치 설정':<32} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk (오류율)':<12}", flush=True)
    print("-" * 115, flush=True)

    selective_results = {}
    for t_norm, t_def, desc in threshold_pairs:
        results = model_diet.predict_selective_batch(
            X_27d,
            tau_normal=t_norm,
            tau_defect=t_def,
            disagreement_tol=0.35,
            fold_std_tol=0.25,
            min_rom_threshold=0.0004
        )
        statuses = np.array([r['status'] for r in results])
        det_mask = (statuses != "UNDETERMINED")
        n_det = int(np.sum(det_mask))
        cov = n_det / N * 100.0
        abs_rate = 100.0 - cov

        if n_det > 0:
            y_det_true = y_c1[det_mask]
            y_det_pred = (statuses[det_mask] == "DEFECT").astype(int)
            sel_f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
            sel_acc = accuracy_score(y_det_true, y_det_pred) * 100.0
            sel_risk = (1.0 - accuracy_score(y_det_true, y_det_pred)) * 100.0
        else:
            sel_f1, sel_acc, sel_risk = 0.0, 0.0, 100.0

        label_str = f"[{t_norm:.2f}, {t_def:.2f}] {desc}"
        print(f"{label_str:<32} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>11.2f}%", flush=True)
        selective_results[f"{t_norm}_{t_def}"] = {
            'desc': desc, 'coverage': round(cov, 1), 'abstention': round(abs_rate, 1),
            'selective_f1': round(sel_f1, 2), 'selective_acc': round(sel_acc, 2), 'risk': round(sel_risk, 2)
        }

    output_summary = {
        'model_name': 'c1_v11_a_diet',
        'experiment_code': 'E0_Refined',
        'feature_dim': 27,
        'description': '27D Refined Biomechanical Feature Set (v10 - dwell - harmony + lumbopelvic_sd)',
        'overall_metrics': m_diet,
        'baseline_v10_metrics': {'f1': v10_f1, 'auc': v10_auc, 'prec': v10_prec, 'rec': v10_rec, 'acc': v10_acc},
        'fold_metrics': fold_metrics,
        'view_metrics': view_results,
        'selective_metrics': selective_results
    }
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 최종 결과 JSON 저장 완료: {RESULTS_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c3_v6.py
Purpose: c3(무릎-발 방향 일치 / 무릎 외반) v6.0 22D 공식 모델 마스터 데이터셋 2,605건 전수 학습, 5-Fold 교차 검증 및 이전 버전 실측 비교
"""

import os
import sys
import time
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
from src.models.second_models.babel_squat.c3_model_v6 import (
    extract_c3_v6_features_from_seq,
    FEATURE_NAMES_22_C3,
    BabelSquatC3ModelV6
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c3_v6_22d_feats.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c3_model_v6_weights.pkl"
OOF_PATH = r"D:\Data\피처_데이터\oof_c3_v6_predictions.npz"
PREV_OOF_PATH = r"D:\Data\피처_데이터\oof_master_64f_predictions.npz"


def extract_or_load_all_features(seq_ids):
    """2,605개 시퀀스로부터 v6 22D 피처 추출 또는 캐시 로드"""
    if os.path.exists(FEATS_CACHE_PATH):
        print(f"[*] 기존 피처 캐시 파일 발견: {FEATS_CACHE_PATH}", flush=True)
        cached = np.load(FEATS_CACHE_PATH, allow_pickle=True)
        if len(cached['X_c3_v6']) == len(seq_ids):
            print(f"[*] 캐시 데이터 로드 완료: shape={cached['X_c3_v6'].shape}", flush=True)
            return cached['X_c3_v6']

    print(f"[*] [1/4] 2,605개 전 시퀀스에 대한 c3 v6 22D 생체역학 피처 추출 시작...", flush=True)
    t0 = time.time()
    feats_list = []
    N = len(seq_ids)

    for i, sid in enumerate(seq_ids, 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']

        feat_vec = extract_c3_v6_features_from_seq(kps_33)
        feats_list.append(feat_vec)

        if i % 500 == 0 or i == N:
            print(f"    - 피처 추출: {i}/{N} ({i/N*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    X_c3_v6 = np.array(feats_list, dtype=np.float32)
    print(f"[*] 피처 추출 완료: shape={X_c3_v6.shape} (소요: {time.time()-t0:.2f}초)", flush=True)

    os.makedirs(os.path.dirname(FEATS_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        FEATS_CACHE_PATH,
        X_c3_v6=X_c3_v6,
        feature_names=FEATURE_NAMES_22_C3,
        sequence_ids=seq_ids
    )
    print(f"[*] 피처 캐시 저장 완료: {FEATS_CACHE_PATH}", flush=True)
    return X_c3_v6


def evaluate_binary_metrics(y_true, y_prob, threshold=0.50):
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
    print("  [vidio-health-trainer] c3(무릎-발 방향 일치 / 무릎 외반) v6.0 공식 프로덕션 모델 학습 & 평가", flush=True)
    print("  ★ 골반 관상면(Coronal) 로컬 정규 직교 기저계(x_hat, y_hat, z_hat) 투영", flush=True)
    print("  ★ 관상면 절대 무릎-발목 너비 비율(KA Ratio) 6D + 좌우 독립 외반 6D + 3D 대퇴-경골각/발끝 8D", flush=True)
    print("  ★ 과거 레거시 텐서 슬라이싱(kd=X_b[:, 69] 등) 100% 영구 제거", flush=True)
    print("  ★ 마스터 데이터셋 2,605건 전수(5대 전 시점 x 9명 피험자) 5-Fold StratifiedGroupKFold 피험자 완전 격리", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c3 = master['Y_defects_golden'][:, 2].astype(int)
    y_norm = master['y_normal_golden'].astype(int)
    N = len(seq_ids)

    X = extract_or_load_all_features(seq_ids)

    print(f"\n[*] [2/4] 5-Fold StratifiedGroupKFold 교차 검증 및 앙상블 학습 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(N, dtype=np.float32)
    cb_models = []
    lgb_models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_c3, groups=subjects), 1):
        X_train, y_train = X[train_idx], y_c3[train_idx]
        X_val, y_val = X[val_idx], y_c3[val_idx]
        val_sub = np.unique(subjects[val_idx])

        # CatBoost
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

        # LightGBM
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

        m_fold = evaluate_binary_metrics(y_val, p_ens_val)
        fold_metrics.append(m_fold)
        print(f"    - Fold {fold} (검증 피험자 {val_sub}): F1={m_fold['f1']:.2f}%, AUC={m_fold['auc']:.2f}%, Rec={m_fold['rec']:.2f}%, Prec={m_fold['prec']:.2f}%", flush=True)

    # 전체 OOF 성능 산출
    m_v6 = evaluate_binary_metrics(y_c3, oof_probs)

    print("\n" + "=" * 115, flush=True)
    print(f"  [★ c3 v6.0 최종 5-Fold 실측 성능 요약 (전수 2,605개 시퀀스)]", flush=True)
    print(f"    - F1-Score : {m_v6['f1']:.2f}%")
    print(f"    - ROC-AUC  : {m_v6['auc']:.2f}%")
    print(f"    - Precision: {m_v6['prec']:.2f}%")
    print(f"    - Recall   : {m_v6['rec']:.2f}%")
    print(f"    - Accuracy : {m_v6['acc']:.2f}%")
    print("=" * 115, flush=True)

    # 5대 시점별 세부 성능 감사
    print(f"\n[*] [3/4] 5대 전 시점(view1~view5) 다차원 성능 감사:", flush=True)
    view_results = {}
    for v in sorted(np.unique(views)):
        v_mask = (views == v)
        mv = evaluate_binary_metrics(y_c3[v_mask], oof_probs[v_mask])
        view_results[v] = mv
        print(f"    - {v:5s}: F1={mv['f1']:.2f}%, AUC={mv['auc']:.2f}%, Prec={mv['prec']:.2f}%, Rec={mv['rec']:.2f}%, Acc={mv['acc']:.2f}%", flush=True)

    # 이전 버전과 성능 비교
    print("\n" + "=" * 115, flush=True)
    print("  [이전 버전 대비 실측 성능 비교표 (마스터 2,605건 동일 평가 기준)]", flush=True)
    print("=" * 115, flush=True)

    if os.path.exists(PREV_OOF_PATH):
        prev_oof = np.load(PREV_OOF_PATH, allow_pickle=True)
        prev_prob = prev_oof['oof_c3']
        m_prev = evaluate_binary_metrics(y_c3, prev_prob)

        print(f"{'평가 지표':<15} | {'이전 v4/5 (40D 레거시 슬라이싱)':<28} | {'신규 v6.0 (22D 표준 통합)':<25} | {'증감 (Δ)':<15}")
        print("-" * 95)
        for metric, name in [('f1', 'F1-Score'), ('auc', 'ROC-AUC'), ('prec', 'Precision'), ('rec', 'Recall'), ('acc', 'Accuracy')]:
            v_old = m_prev[metric]
            v_new = m_v6[metric]
            diff = v_new - v_old
            sign = "+" if diff > 0 else ""
            print(f"{name:<15} | {v_old:>24.2f}% | {v_new:>21.2f}% | {sign}{diff:>11.2f}%p")

        print("-" * 95)
        print("[5대 시점별 F1 비교]")
        for v in sorted(np.unique(views)):
            v_mask = (views == v)
            f1_old = f1_score(y_c3[v_mask], (prev_prob[v_mask] >= 0.50).astype(int), zero_division=0) * 100.0
            f1_new = view_results[v]['f1']
            diff = f1_new - f1_old
            sign = "+" if diff > 0 else ""
            print(f"  {v:<7} | 이전: {f1_old:>6.2f}% | v6.0: {f1_new:>6.2f}% | {sign}{diff:>6.2f}%p")

    # 4. 3-State 선택적 거부 평가
    print("\n" + "=" * 115, flush=True)
    print("  [4/4] 3-State 선택적 판정(Selective Classification) 실측 평가", flush=True)
    print("=" * 115, flush=True)

    model_temp = BabelSquatC3ModelV6()
    model_temp.cb_models = cb_models
    model_temp.lgb_models = lgb_models
    model_temp.is_fitted = True

    threshold_pairs = [
        (0.50, 0.50, "Full Decision (강제 이진)"),
        (0.45, 0.55, "Narrow Abstention (+/- 0.05)"),
        (0.40, 0.60, "Balanced Abstention (+/- 0.10, 추천)"),
        (0.35, 0.65, "Standard Selective (+/- 0.15)"),
        (0.30, 0.70, "High Confidence (+/- 0.20)")
    ]

    print(f"{'임계치 설정':<32} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk (오판율)':<12}", flush=True)
    print("-" * 115, flush=True)

    sel_results_dict = {}
    for t_norm, t_def, desc in threshold_pairs:
        results = model_temp.predict_selective_batch(
            X,
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
            y_det_true = y_c3[det_mask]
            y_det_pred = (statuses[det_mask] == "DEFECT").astype(int)
            sel_f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
            sel_acc = accuracy_score(y_det_true, y_det_pred) * 100.0
            sel_risk = (1.0 - sel_acc / 100.0) * 100.0
        else:
            sel_f1, sel_acc, sel_risk = 0.0, 0.0, 100.0

        label_str = f"[{t_norm:.2f}, {t_def:.2f}] {desc}"
        print(f"{label_str:<32} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>11.2f}%", flush=True)
        sel_results_dict[(t_norm, t_def)] = (cov, abs_rate, sel_f1, sel_acc, sel_risk)

    # 가중치 영구 저장
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    payload = {
        'cb_models': cb_models,
        'lgb_models': lgb_models,
        'decision_threshold': 0.50,
        'tau_normal': 0.40,
        'tau_defect': 0.60,
        'feature_names': FEATURE_NAMES_22_C3,
        'metrics_summary': m_v6,
        'view_metrics': view_results,
        'selective_summary': sel_results_dict
    }
    with open(WEIGHTS_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n[*] c3 v6.0 가중치 배포 완료: {WEIGHTS_PATH}", flush=True)

    np.savez_compressed(
        OOF_PATH,
        oof_c3_v6=oof_probs,
        y_true=y_c3,
        subjects=subjects,
        views=views,
        sequence_ids=seq_ids,
        metrics=m_v6
    )
    print(f"[*] c3 v6.0 OOF 예측값 저장 완료: {OOF_PATH}", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()

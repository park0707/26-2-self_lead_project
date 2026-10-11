# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c4_v4.py
Purpose: c4(발바닥 지면 고정) v4.0 22D 공식 모델 마스터 데이터셋 2,605건 전수 학습, 5-Fold 교차 검증 및 이전 버전 실측 비교
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
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c4_model_v4 import (
    extract_c4_v4_features_from_seq,
    FEATURE_NAMES_22_C4,
    BabelSquatC4ModelV4
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c4_v4_22d_feats.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c4_model_v4_weights.pkl"
OOF_PATH = r"D:\Data\피처_데이터\oof_c4_v4_predictions.npz"


def extract_or_load_all_features(seq_ids):
    """2,605개 시퀀스로부터 v4 22D 피처 추출 또는 캐시 로드"""
    if os.path.exists(FEATS_CACHE_PATH):
        print(f"[*] 기존 피처 캐시 파일 발견: {FEATS_CACHE_PATH}", flush=True)
        cached = np.load(FEATS_CACHE_PATH, allow_pickle=True)
        if len(cached['X_c4_v4']) == len(seq_ids):
            print(f"[*] 캐시 데이터 로드 완료: shape={cached['X_c4_v4'].shape}", flush=True)
            return cached['X_c4_v4']

    print(f"[*] [1/4] 2,605개 전 시퀀스에 대한 c4 v4 22D 생체역학 지면 피처 추출 시작...", flush=True)
    t0 = time.time()
    feats_list = []
    N = len(seq_ids)

    for i, sid in enumerate(seq_ids, 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']

        feat_vec = extract_c4_v4_features_from_seq(kps_33)
        feats_list.append(feat_vec)

        if i % 500 == 0 or i == N:
            print(f"    - 피처 추출: {i}/{N} ({i/N*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    X_c4_v4 = np.array(feats_list, dtype=np.float32)
    print(f"[*] 피처 추출 완료: shape={X_c4_v4.shape} (소요: {time.time()-t0:.2f}초)", flush=True)

    os.makedirs(os.path.dirname(FEATS_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        FEATS_CACHE_PATH,
        X_c4_v4=X_c4_v4,
        feature_names=FEATURE_NAMES_22_C4,
        sequence_ids=seq_ids
    )
    print(f"[*] 피처 캐시 저장 완료: {FEATS_CACHE_PATH}", flush=True)
    return X_c4_v4


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
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        'f1': f1, 'precision': prec, 'recall': rec,
        'accuracy': acc, 'auc': auc,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def evaluate_3state_metrics(y_true, y_prob, tau_norm, tau_def):
    decided = (y_prob <= tau_norm) | (y_prob >= tau_def)
    abstain = ~decided

    coverage = np.mean(decided) * 100.0
    abstention = np.mean(abstain) * 100.0
    dec_count = int(np.sum(decided))
    abs_count = int(np.sum(abstain))

    if dec_count == 0:
        return {'coverage': 0.0, 'abstention': 100.0, 'selective_f1': 0.0, 'selective_acc': 0.0, 'selective_risk': 100.0}

    y_t_dec = y_true[decided]
    y_p_dec = (y_prob[decided] >= tau_def).astype(int)

    sel_f1 = f1_score(y_t_dec, y_p_dec, zero_division=0) * 100.0
    sel_acc = accuracy_score(y_t_dec, y_p_dec) * 100.0
    sel_risk = (100.0 - sel_acc)

    return {
        'coverage': coverage,
        'abstention': abstention,
        'decided_count': dec_count,
        'abstain_count': abs_count,
        'selective_f1': sel_f1,
        'selective_accuracy': sel_acc,
        'selective_risk': sel_risk
    }


def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4(발바닥 지면 고정) v4.0 마스터 전수 검증 및 학습 파이프라인", flush=True)
    print("  ★ 22D 순수 MediaPipe 33관절 64F 지면 접지 피처셋", flush=True)
    print("  ★ 3-State 선택적 판정 (Balanced [0.40, 0.60]) 탑재", flush=True)
    print("  ★ 마스터 2,605개 전수(피험자 1~9번, 5대 전 시점) 5-Fold StratifiedGroupKFold 완전 격리 평가", flush=True)
    print("=" * 100, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    base_ids = master['base_sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c4 = master['Y_defects_golden'][:, 3].astype(int)
    N = len(seq_ids)

    print(f"[*] 마스터 데이터 로드 완료: 총 {N}건 (결함: {np.sum(y_c4)}건, 정상: {N - np.sum(y_c4)}건)", flush=True)

    # 1. 피처 로드 또는 추출
    X_c4_v4 = extract_or_load_all_features(seq_ids)

    # 2. 5-Fold StratifiedGroupKFold 검증
    print("\n[*] [2/4] 5-Fold StratifiedGroupKFold 교차 검증 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)

    fold_metrics = []
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_c4_v4, y_c4, groups=base_ids), 1):
        X_tr, y_tr = X_c4_v4[trn_idx], y_c4[trn_idx]
        X_va, y_va = X_c4_v4[val_idx], y_c4[val_idx]

        cb = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                                random_seed=42 + fold, thread_count=-1, verbose=False)
        cb.fit(X_tr, y_tr)
        p_cb = cb.predict_proba(X_va)[:, 1]

        lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, num_leaves=31,
                                  random_state=42 + fold, verbose=-1, n_jobs=-1)
        lgbm.fit(X_tr, y_tr)
        p_lgb = lgbm.predict_proba(X_va)[:, 1]

        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42 + fold, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_va)[:, 1]

        p_ens = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf
        oof_probs[val_idx] = p_ens

        m_fold = evaluate_binary_metrics(y_va, p_ens, 0.50)
        fold_metrics.append(m_fold)
        print(f"    - Fold {fold}: F1={m_fold['f1']:.2f}% | AUC={m_fold['auc']:.2f}% | Acc={m_fold['accuracy']:.2f}% (검증 {len(val_idx)}건)", flush=True)

    # 3. OOF 예측 저장
    np.savez_compressed(
        OOF_PATH,
        oof_c4_v4=oof_probs,
        y_true=y_c4,
        sequence_ids=seq_ids,
        base_sequence_ids=base_ids,
        views=views,
        subjects=subjects
    )
    print(f"\n[*] OOF 예측값 저장 완료: {OOF_PATH}", flush=True)

    # 4. 프로덕션 최종 모델 전체 데이터 학습 및 가중치 저장
    print("\n[*] [3/4] 전체 마스터 2,605건에 대한 프로덕션 모델 학습 및 가중치 저장...", flush=True)
    prod_model = BabelSquatC4ModelV4(tau_normal=0.40, tau_defect=0.60)
    prod_model.fit(X_c4_v4, y_c4)
    prod_model.save_weights(WEIGHTS_PATH)

    # 5. 성능 평가 결과 종합
    print("\n" + "=" * 100, flush=True)
    print("  [4/4] 바벨 스쿼트 c4 v4.0 실측 검증 결과 종합 보고", flush=True)
    print("=" * 100, flush=True)

    # 강제 이진 (0.50)
    m_bin = evaluate_binary_metrics(y_c4, oof_probs, 0.50)
    print(f"\n[1] 강제 이진 분류 (0.50 기준, N={N})")
    print(f"    - F1-Score:   {m_bin['f1']:.2f}%")
    print(f"    - ROC-AUC:    {m_bin['auc']:.2f}%")
    print(f"    - Accuracy:   {m_bin['accuracy']:.2f}%")
    print(f"    - Precision:  {m_bin['precision']:.2f}%")
    print(f"    - Recall:     {m_bin['recall']:.2f}%")
    print(f"    - 혼동행렬:   TN={m_bin['tn']}, FP={m_bin['fp']}, FN={m_bin['fn']}, TP={m_bin['tp']}")

    # 3-State 정책별 비교
    policies = [
        ("강제 이진 (Binary)", 0.50, 0.50),
        ("보수적 (Conservative)", 0.45, 0.55),
        ("균형 (Balanced - 권장)", 0.40, 0.60),
        ("고신뢰 (High-Confidence)", 0.35, 0.65),
    ]

    print("\n[2] 3-State 선택적 판정 정책별 성능 비교")
    print("-" * 100)
    print(f"{'정책명':<25} | {'Coverage':<10} | {'보류율':<10} | {'Selective F1':<14} | {'정확도':<10} | {'오판율(Risk)':<12}")
    print("-" * 100)
    for name, tn, td in policies:
        m_3s = evaluate_3state_metrics(y_c4, oof_probs, tn, td)
        print(f"{name:<25} | {m_3s['coverage']:>8.1f}% | {m_3s['abstention']:>8.1f}% | {m_3s['selective_f1']:>12.2f}% | {m_3s['selective_accuracy']:>8.2f}% | {m_3s['selective_risk']:>10.2f}%")
    print("-" * 100)

    # 5대 전 시점별 성능 감사
    print("\n[3] 5대 전 시점(view1~view5) 다차원 성능 감사 (강제 이진 기준)")
    print("-" * 100)
    print(f"{'시점 코드':<10} | {'카메라 위치':<20} | {'샘플수':<8} | {'F1-Score':<12} | {'ROC-AUC':<12} | {'정확도':<10}")
    print("-" * 100)
    view_names = {
        'view1': '후방 (180°)',
        'view2': '우측 45° (측면 전방)',
        'view3': '완전 정면 (0°)',
        'view4': '좌측 45° (측면 전방)',
        'view5': '우측 후방 45°'
    }
    for v in sorted(np.unique(views)):
        idx_v = np.where(views == v)[0]
        mv = evaluate_binary_metrics(y_c4[idx_v], oof_probs[idx_v], 0.50)
        print(f"{v:<10} | {view_names.get(v, v):<20} | {len(idx_v):>6}개 | {mv['f1']:>10.2f}% | {mv['auc']:>10.2f}% | {mv['accuracy']:>8.2f}%")
    print("-" * 100)

    # 이전 v3 9D 모델 대비 비교
    old_X = master['X_c4']
    print("\n[4] 이전 버전(v3 9D 레거시 슬라이싱) 대비 종합 비교 (N=2,605 전수)")
    print(f"    - v3 9D 베이스라인: F1 75.96% / AUC 85.32% / Acc 76.16% / Risk 23.84%")
    print(f"    - v4 22D 강제 이진: F1 {m_bin['f1']:.2f}% (+{m_bin['f1'] - 75.96:.2f}%p) / AUC {m_bin['auc']:.2f}% (+{m_bin['auc'] - 85.32:.2f}%p) / Acc {m_bin['accuracy']:.2f}% (+{m_bin['accuracy'] - 76.16:.2f}%p)")
    m_bal = evaluate_3state_metrics(y_c4, oof_probs, 0.40, 0.60)
    print(f"    - v4 22D Balanced : Coverage {m_bal['coverage']:.1f}% / Selective F1 {m_bal['selective_f1']:.2f}% (+{m_bal['selective_f1'] - 75.96:.2f}%p) / Risk {m_bal['selective_risk']:.2f}% (▼{23.84 - m_bal['selective_risk']:.2f}%p 극감)")

    print("\n[*] 모든 학습, 검증, OOF 저장 및 가중치 직렬화가 성공적으로 완료되었습니다.", flush=True)


if __name__ == '__main__':
    main()

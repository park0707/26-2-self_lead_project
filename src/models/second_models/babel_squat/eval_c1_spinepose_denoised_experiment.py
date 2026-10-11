# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: eval_c1_spinepose_denoised_experiment.py
Purpose: 데드리프트 d1 모델에서 도입된 SpinePose 관절 포인트 튐 방지 기법
         (1. One-Euro 적응형 지터 필터 + 2. 생체역학 뼈 길이 불변 제약)을
         바벨 스쿼트 c1 모델에 동일하게 전면 도입하여
         2,605개 마스터 데이터셋 5-Fold StratifiedGroupKFold 피험자 완전 격리 환경에서
         기존 원시 SpinePose (Model-B: 81.12%) 대비 성능 향상 여부를 정밀 실측 검증
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.features.spinepose_denoiser import denoise_spinepose_batch
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq
from src.models.second_models.babel_squat.c1_model_v11_a_diet import extract_c1_v11_a_diet_features_from_seq
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

# 경로 정의
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
RAW_VISION_SPINE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"
DENOISED_VISION_SPINE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_denoised_vision_64f.npz"
DENOISED_FEATS_PATH = r"D:\Data\피처_데이터\squat_c1_v11_denoised_spinepose_feats.npz"
ORIGINAL_COMPARISON_JSON = r"d:\src\models\second_models\babel_squat\c1_aihub_vs_spinepose_comparison_results.json"
OUT_COMPARISON_JSON = r"d:\src\models\second_models\babel_squat\c1_spinepose_denoised_comparison_results.json"

VIEW_NAMES = ['view1', 'view2', 'view3', 'view4', 'view5']
VIEW_ALIASES = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def compute_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0) * 100.0)
    prec = float(precision_score(y_true, preds, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, preds, zero_division=0) * 100.0)
    acc = float(accuracy_score(y_true, preds) * 100.0)
    try:
        auc = float(roc_auc_score(y_true, y_prob) * 100.0)
    except Exception:
        auc = 50.0
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2),
        'cm': cm
    }


def train_tcn_fast(X_tr, y_tr, X_val, in_channels=24, epochs=15, batch_size=64, lr=1.5e-3):
    """Dilated TCN 학습 (기존 compare_c1_aihub_vs_spinepose.py와 100% 동일)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquatC1DilatedTCN(in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32)

    ds_tr = TensorDataset(X_tr_t, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device)).squeeze(-1).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    return model, val_probs


def step1_denoise_spinepose():
    """데드리프트 d1 관절 튐 방지 기법 (One-Euro + Segment Constraint) 적용"""
    if os.path.exists(DENOISED_VISION_SPINE_PATH):
        print(f"[*] [Step 1] 정제된 SpinePose 캐시 이미 존재함: {DENOISED_VISION_SPINE_PATH}")
        return np.load(DENOISED_VISION_SPINE_PATH, allow_pickle=True)['spine_9_64f_vision']

    print(f"[*] [Step 1] 원시 SpinePose 로드 및 데드리프트 d1 관절 튐 방지 기법 일괄 적용...")
    t0 = time.time()
    raw_data = np.load(RAW_VISION_SPINE_PATH, allow_pickle=True)
    raw_spine = raw_data['spine_9_64f_vision']  # (2605, 64, 9, 3)
    sids = raw_data['sequence_ids']
    valid_mask = raw_data['valid_mask']

    print(f"    - 대상 시퀀스: {len(raw_spine)}개")
    print(f"    - 1) One-Euro 적응형 지터 필터 (min_cutoff=1.5, beta=0.005, fps=30.0)")
    print(f"    - 2) 생체역학 뼈 길이 불변 제약 (Segment Length Invariance, tol=0.08)")

    cleaned_spine = denoise_spinepose_batch(
        raw_spine,
        fps=30.0,
        min_cutoff=1.5,
        beta=0.005,
        segment_tol=0.08
    )

    np.savez_compressed(
        DENOISED_VISION_SPINE_PATH,
        spine_9_64f_vision=cleaned_spine,
        sequence_ids=sids,
        valid_mask=valid_mask
    )
    print(f"[OK] Step 1 완료: 정제 캐시 저장 완료 ({time.time()-t0:.1f}초)")
    return cleaned_spine


def step2_build_denoised_features(cleaned_spine):
    """정제된 SpinePose 기반 38D, 27D, 24D 피처셋 구축"""
    if os.path.exists(DENOISED_FEATS_PATH):
        print(f"[*] [Step 2] 정제된 피처셋 파일 이미 존재함: {DENOISED_FEATS_PATH}")
        f_data = np.load(DENOISED_FEATS_PATH, allow_pickle=True)
        return f_data['X_38d_vision'], f_data['X_27d_vision'], f_data['X_temporal_vision']

    print(f"[*] [Step 2] 정제된 SpinePose 기반 38D/27D/24D 피처셋 추출 시작...")
    t0 = time.time()
    master = np.load(MASTER_PATH, allow_pickle=True)
    master_sids = master['sequence_ids']
    N = len(master_sids)

    X_38d_vision = np.zeros((N, 38), dtype=np.float32)
    X_27d_vision = np.zeros((N, 27), dtype=np.float32)
    X_temporal_vision = np.zeros((N, 64, 24), dtype=np.float32)

    for i, sid in enumerate(master_sids):
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({(i+1)/N*100:.1f}%) | 소요: {time.time()-t0:.1f}초", flush=True)

        mp_path = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        mp_d = np.load(mp_path, allow_pickle=True)
        kps_33_64f = mp_d['kps_33_3d_64f']
        spine_9 = cleaned_spine[i]

        f38 = extract_c1_v11_a_features_from_seq(kps_33_64f, spine_9)
        f27 = extract_c1_v11_a_diet_features_from_seq(kps_33_64f, spine_9)
        t24 = extract_temporal_24d(kps_33_64f, spine_9)

        X_38d_vision[i] = f38
        X_27d_vision[i] = f27
        X_temporal_vision[i] = t24

    np.savez_compressed(
        DENOISED_FEATS_PATH,
        X_38d_vision=X_38d_vision,
        X_27d_vision=X_27d_vision,
        X_temporal_vision=X_temporal_vision,
        sequence_ids=master_sids
    )
    print(f"[OK] Step 2 완료: 피처셋 저장 완료 ({time.time()-t0:.1f}초)")
    return X_38d_vision, X_27d_vision, X_temporal_vision


def step3_train_and_evaluate(X_38d, X_27d, X_t24):
    """5-Fold StratifiedGroupKFold 피험자 완전 격리 실측 교차검증"""
    print(f"\n[*] [Step 3] 5-Fold StratifiedGroupKFold 피험자 완전 격리 모델 학습 및 감사...")
    master = np.load(MASTER_PATH, allow_pickle=True)
    y = master['Y_defects_golden'][:, 0].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)
    oof_p_38d = np.zeros(N, dtype=np.float32)
    oof_p_27d = np.zeros(N, dtype=np.float32)
    oof_p_tcn = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_38d, y, groups=subjects), 1):
        t_fold = time.time()

        # Branch 1: 38D GBDT (CB 60% + LGB 40%)
        cb_38 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_38.fit(X_38d[tr_idx], y[tr_idx])
        lgb_38 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_38.fit(X_38d[tr_idx], y[tr_idx])
        p_38 = 0.60 * cb_38.predict_proba(X_38d[val_idx])[:, 1] + 0.40 * lgb_38.predict_proba(X_38d[val_idx])[:, 1]
        oof_p_38d[val_idx] = p_38

        # Branch 2: 27D Diet GBDT
        cb_27 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_27.fit(X_27d[tr_idx], y[tr_idx])
        lgb_27 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_27.fit(X_27d[tr_idx], y[tr_idx])
        p_27 = 0.60 * cb_27.predict_proba(X_27d[val_idx])[:, 1] + 0.40 * lgb_27.predict_proba(X_27d[val_idx])[:, 1]
        oof_p_27d[val_idx] = p_27

        # Branch 3: 24D Dilated TCN
        tcn_model, p_tcn = train_tcn_fast(X_t24[tr_idx], y[tr_idx], X_t24[val_idx], in_channels=24, epochs=15, batch_size=64)
        oof_p_tcn[val_idx] = p_tcn

        # Tri-Ensemble (0.10:0.40:0.50)
        p_ens = 0.10 * p_38 + 0.40 * p_27 + 0.50 * p_tcn
        oof_probs[val_idx] = p_ens

        m_fold = compute_metrics(y[val_idx], p_ens)
        print(f"  >>> [Fold {fold}/5] ({time.time()-t_fold:.1f}초) | F1={m_fold['f1']:.2f}% | AUC={m_fold['auc']:.2f}% | Acc={m_fold['acc']:.2f}% (38D={compute_metrics(y[val_idx], p_38)['f1']:.2f}%, 27D={compute_metrics(y[val_idx], p_27)['f1']:.2f}%, TCN={compute_metrics(y[val_idx], p_tcn)['f1']:.2f}%)", flush=True)

    print(f"[*] 5-Fold 전체 교차검증 완수 ({time.time()-t0:.1f}초)")

    # 전체 성능 집계
    overall_metrics = compute_metrics(y, oof_probs)
    m_38 = compute_metrics(y, oof_p_38d)
    m_27 = compute_metrics(y, oof_p_27d)
    m_tcn = compute_metrics(y, oof_p_tcn)

    # 3-State 선택적 거부 (Selective Abstention)
    uncertain_mask = (oof_probs >= 0.40) & (oof_probs <= 0.60)
    decided_mask = ~uncertain_mask
    coverage = float(np.mean(decided_mask) * 100.0)
    selective_metrics = compute_metrics(y[decided_mask], oof_probs[decided_mask])

    # 5대 공식 시점별 성능 집계
    view_breakdown = {}
    for vn in VIEW_NAMES:
        mask_v = (views == vn)
        m_v = compute_metrics(y[mask_v], oof_probs[mask_v])
        view_breakdown[vn] = {
            'alias': VIEW_ALIASES.get(vn, vn),
            'count': int(np.sum(mask_v)),
            'metrics': m_v
        }

    # 피험자별(Subject 1~9) 성능 집계
    subject_breakdown = {}
    unique_subs = sorted(list(set(subjects)))
    for s in unique_subs:
        mask_s = (subjects == s)
        m_s = compute_metrics(y[mask_s], oof_probs[mask_s])
        subject_breakdown[f"피험자_{s}"] = {
            'subject': int(s),
            'count': int(np.sum(mask_s)),
            'metrics': m_s
        }

    # 기존 baseline 결과 로드
    with open(ORIGINAL_COMPARISON_JSON, 'r', encoding='utf-8') as f:
        orig_comp = json.load(f)

    orig_model_b = orig_comp['model_b_vision_spinepose']
    orig_model_a = orig_comp['model_a_oracle_aihub']

    delta_vs_raw_spinepose = {
        'delta_f1': round(overall_metrics['f1'] - orig_model_b['overall']['f1'], 2),
        'delta_auc': round(overall_metrics['auc'] - orig_model_b['overall']['auc'], 2),
        'delta_acc': round(overall_metrics['acc'] - orig_model_b['overall']['acc'], 2),
        'delta_prec': round(overall_metrics['prec'] - orig_model_b['overall']['prec'], 2),
        'delta_rec': round(overall_metrics['rec'] - orig_model_b['overall']['rec'], 2)
    }

    final_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': N,
        'model_c_denoised_spinepose': {
            'description': 'SpinePose + One-Euro Filter + Biomechanical Segment Constraint (데드리프트 d1 관절 튐 방지 동일 기법)',
            'overall': overall_metrics,
            'branches': {
                'branch_38d_gbdt': m_38,
                'branch_27d_diet_gbdt': m_27,
                'branch_24d_tcn': m_tcn
            },
            'selective_abstention': {
                'coverage': round(coverage, 2),
                'selective_f1': selective_metrics['f1'],
                'selective_acc': selective_metrics['acc']
            },
            'view_breakdown': view_breakdown,
            'subject_breakdown': subject_breakdown
        },
        'model_b_raw_spinepose': orig_model_b,
        'model_a_oracle_aihub': orig_model_a,
        'delta_c_vs_b': delta_vs_raw_spinepose
    }

    with open(OUT_COMPARISON_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 115)
    print("  [최종 대조 평가 결과 요약] 데드리프트 d1 관절 튐 방지 기법 도입 전 vs 후")
    print("=" * 115)
    print(f"  1) 전체 F1 점수: {orig_model_b['overall']['f1']:.2f}% (기존 원시) -> {overall_metrics['f1']:.2f}% (정제 적용) "
          f"[{delta_vs_raw_spinepose['delta_f1']:+.2f}%p]")
    print(f"  2) ROC-AUC     : {orig_model_b['overall']['auc']:.2f}% (기존 원시) -> {overall_metrics['auc']:.2f}% (정제 적용) "
          f"[{delta_vs_raw_spinepose['delta_auc']:+.2f}%p]")
    print(f"  3) 정확도(Acc) : {orig_model_b['overall']['acc']:.2f}% (기존 원시) -> {overall_metrics['acc']:.2f}% (정제 적용) "
          f"[{delta_vs_raw_spinepose['delta_acc']:+.2f}%p]")
    print(f"  4) 정밀도(Prec): {orig_model_b['overall']['prec']:.2f}% (기존 원시) -> {overall_metrics['prec']:.2f}% (정제 적용) "
          f"[{delta_vs_raw_spinepose['delta_prec']:+.2f}%p]")
    print(f"  5) 재현율(Rec) : {orig_model_b['overall']['rec']:.2f}% (기존 원시) -> {overall_metrics['rec']:.2f}% (정제 적용) "
          f"[{delta_vs_raw_spinepose['delta_rec']:+.2f}%p]")
    print(f"  6) 3-State 거부: F1 {orig_model_b['selective_abstention']['selective_f1']:.2f}% -> {selective_metrics['f1']:.2f}% "
          f"(커버리지: {coverage:.2f}%)")
    print("-" * 115)
    print("  [공식 5대 시점별 F1 변화 대조]")
    for vn in VIEW_NAMES:
        alias = VIEW_ALIASES.get(vn, vn)
        f1_b = orig_comp['view_breakdown'][vn]['model_b']['f1']
        f1_c = view_breakdown[vn]['metrics']['f1']
        delta_v = f1_c - f1_b
        print(f"    - {alias:25s}: {f1_b:6.2f}% -> {f1_c:6.2f}% ({delta_v:+6.2f}%p)")
    print("=" * 115)


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] c1 모델 SpinePose 관절 포인트 튐 방지 기법 도입 및 전수 성능 감사")
    print("  - 기법: 1. One-Euro 적응형 지터 필터 + 2. 생체역학 뼈 길이 불변 제약 (Segment Length Invariance)")
    print("  - 데이터셋: 2,605개 마스터 전수 시퀀스 (Subject 1~9 전원, 5대 전 시점)")
    print("=" * 115)

    cleaned_spine = step1_denoise_spinepose()
    X_38d, X_27d, X_t24 = step2_build_denoised_features(cleaned_spine)
    step3_train_and_evaluate(X_38d, X_27d, X_t24)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: eval_c1_two_proposed_denoisers.py
Purpose: 사용자가 제안 채택을 요청한 2가지 관절 튐 방지 기법:
         [제안 1] Zero-Phase Savitzky-Golay (국소 2차 다항 평활화) + 생체역학 뼈 길이 불변 제약
         [제안 2] 생체역학 뼈 길이 불변 제약 단독 (Segment Length Invariance Only - 위상 왜곡 0%)
         을 바벨 스쿼트 c1 모델에 전면 도입하여
         2,605개 마스터 전수 시퀀스 5-Fold StratifiedGroupKFold 피험자 완전 격리 환경에서
         원시 SpinePose(81.12%) 및 기존 One-Euro 모델(79.38%) 대비 성능 향상 여부를 실측 검증
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from scipy.signal import savgol_filter
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.features.spinepose_denoiser import apply_biomechanical_segment_constraint
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq
from src.models.second_models.babel_squat.c1_model_v11_a_diet import extract_c1_v11_a_diet_features_from_seq
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

# 경로 정의
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
RAW_VISION_SPINE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"

SEG_ONLY_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_seg_only_64f.npz"
SAVGOL_SEG_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_savgol_seg_64f.npz"

FEATS_SEG_ONLY_PATH = r"D:\Data\피처_데이터\squat_c1_v11_seg_only_feats.npz"
FEATS_SAVGOL_SEG_PATH = r"D:\Data\피처_데이터\squat_c1_v11_savgol_seg_feats.npz"

OUT_RESULTS_JSON = r"d:\src\models\second_models\babel_squat\c1_two_proposed_denoisers_results.json"

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


def prepare_denoised_spine_caches():
    raw_d = np.load(RAW_VISION_SPINE_PATH, allow_pickle=True)
    raw_spine = raw_d['spine_9_64f_vision']  # (2605, 64, 9, 3)
    sids = raw_d['sequence_ids']
    valid_mask = raw_d['valid_mask']
    N = len(raw_spine)

    # 1. 제안 2: Segment Constraint Only
    if os.path.exists(SEG_ONLY_CACHE_PATH):
        print(f"[*] [제안 2] Segment Constraint Only 캐시 이미 존재함")
        spine_seg_only = np.load(SEG_ONLY_CACHE_PATH, allow_pickle=True)['spine_9_64f_vision']
    else:
        print(f"[*] [제안 2] Segment Constraint Only 생성 중 (2,605개)...")
        t0 = time.time()
        spine_seg_only = np.zeros_like(raw_spine)
        for i in range(N):
            spine_seg_only[i] = apply_biomechanical_segment_constraint(raw_spine[i], tol=0.08)
        np.savez_compressed(SEG_ONLY_CACHE_PATH, spine_9_64f_vision=spine_seg_only, sequence_ids=sids, valid_mask=valid_mask)
        print(f"[OK] 제안 2 캐시 저장 완료 ({time.time()-t0:.1f}초)")

    # 2. 제안 1: Zero-Phase Savitzky-Golay + Segment Constraint
    if os.path.exists(SAVGOL_SEG_CACHE_PATH):
        print(f"[*] [제안 1] Savitzky-Golay + Segment Constraint 캐시 이미 존재함")
        spine_savgol_seg = np.load(SAVGOL_SEG_CACHE_PATH, allow_pickle=True)['spine_9_64f_vision']
    else:
        print(f"[*] [제안 1] Savitzky-Golay (win=7, poly=2) + Segment Constraint 생성 중 (2,605개)...")
        t0 = time.time()
        spine_savgol_seg = np.zeros_like(raw_spine)
        for i in range(N):
            sg_filtered = savgol_filter(raw_spine[i], window_length=7, polyorder=2, axis=0)
            spine_savgol_seg[i] = apply_biomechanical_segment_constraint(sg_filtered, tol=0.08)
        np.savez_compressed(SAVGOL_SEG_CACHE_PATH, spine_9_64f_vision=spine_savgol_seg, sequence_ids=sids, valid_mask=valid_mask)
        print(f"[OK] 제안 1 캐시 저장 완료 ({time.time()-t0:.1f}초)")

    return spine_seg_only, spine_savgol_seg


def extract_or_load_features(spine_data, cache_path, desc):
    if os.path.exists(cache_path):
        print(f"[*] [{desc}] 피처셋 파일 이미 존재함: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X_38d'], d['X_27d'], d['X_t24']

    print(f"[*] [{desc}] 피처셋 추출 시작 (2,605개)...")
    t0 = time.time()
    master = np.load(MASTER_PATH, allow_pickle=True)
    master_sids = master['sequence_ids']
    N = len(master_sids)

    X_38d = np.zeros((N, 38), dtype=np.float32)
    X_27d = np.zeros((N, 27), dtype=np.float32)
    X_t24 = np.zeros((N, 64, 24), dtype=np.float32)

    for i, sid in enumerate(master_sids):
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"    - {desc} 피처 추출: {i+1}/{N} ({(i+1)/N*100:.1f}%) | 소요: {time.time()-t0:.1f}초", flush=True)

        mp_path = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        mp_d = np.load(mp_path, allow_pickle=True)
        kps_33_64f = mp_d['kps_33_3d_64f']
        spine_9 = spine_data[i]

        X_38d[i] = extract_c1_v11_a_features_from_seq(kps_33_64f, spine_9)
        X_27d[i] = extract_c1_v11_a_diet_features_from_seq(kps_33_64f, spine_9)
        X_t24[i] = extract_temporal_24d(kps_33_64f, spine_9)

    np.savez_compressed(
        cache_path,
        X_38d=X_38d,
        X_27d=X_27d,
        X_t24=X_t24,
        sequence_ids=master_sids
    )
    print(f"[OK] [{desc}] 피처셋 저장 완료: {cache_path} ({time.time()-t0:.1f}초)")
    return X_38d, X_27d, X_t24


def run_5fold_evaluation(X_38d, X_27d, X_t24, desc):
    print(f"\n[*] [{desc}] 5-Fold StratifiedGroupKFold 교차검증 시작...")
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

        # Branch 1: 38D GBDT
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
        print(f"  [{desc}] Fold {fold}/5 ({time.time()-t_fold:.1f}초) | F1={m_fold['f1']:.2f}% | AUC={m_fold['auc']:.2f}% | Acc={m_fold['acc']:.2f}%", flush=True)

    print(f"[*] [{desc}] 5-Fold 완료 ({time.time()-t0:.1f}초)")

    overall = compute_metrics(y, oof_probs)
    m_38 = compute_metrics(y, oof_p_38d)
    m_27 = compute_metrics(y, oof_p_27d)
    m_tcn = compute_metrics(y, oof_p_tcn)

    # 3-State 거부
    uncertain = (oof_probs >= 0.40) & (oof_probs <= 0.60)
    decided = ~uncertain
    cov = float(np.mean(decided) * 100.0)
    m_sel = compute_metrics(y[decided], oof_probs[decided])

    # 시점별
    v_metrics = {}
    for vn in VIEW_NAMES:
        mask_v = (views == vn)
        v_metrics[vn] = {
            'alias': VIEW_ALIASES.get(vn, vn),
            'count': int(np.sum(mask_v)),
            'metrics': compute_metrics(y[mask_v], oof_probs[mask_v])
        }

    return {
        'description': desc,
        'overall': overall,
        'branches': {'38d': m_38, '27d': m_27, 'tcn': m_tcn},
        'selective_abstention': {'coverage': round(cov, 2), 'f1': m_sel['f1'], 'acc': m_sel['acc']},
        'view_breakdown': v_metrics
    }


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] c1 모델 제안 2대 관절 튐 방지 기법 실측 평가 토너먼트")
    print("  - [제안 1] Zero-Phase Savitzky-Golay (국소 2차 곡률 보존) + 생체역학 뼈 길이 불변 제약")
    print("  - [제안 2] 생체역학 뼈 길이 불변 제약 단독 (Segment Length Invariance Only - 위상 왜곡 0%)")
    print("  - 비교 대상: 기존 원시 SpinePose (81.12%) vs One-Euro 적용 모델 (79.38%)")
    print("=" * 115)

    spine_seg_only, spine_savgol_seg = prepare_denoised_spine_caches()

    # 제안 2 피처 및 평가
    X_38_s2, X_27_s2, X_t24_s2 = extract_or_load_features(spine_seg_only, FEATS_SEG_ONLY_PATH, "제안 2: Segment Constraint Only")
    res_prop2 = run_5fold_evaluation(X_38_s2, X_27_s2, X_t24_s2, "제안 2: Segment Constraint Only")

    # 제안 1 피처 및 평가
    X_38_s1, X_27_s1, X_t24_s1 = extract_or_load_features(spine_savgol_seg, FEATS_SAVGOL_SEG_PATH, "제안 1: Savitzky-Golay + Segment Constraint")
    res_prop1 = run_5fold_evaluation(X_38_s1, X_27_s1, X_t24_s1, "제안 1: Savitzky-Golay + Segment Constraint")

    # 기존 기준값 로드
    with open(r"d:\src\models\second_models\babel_squat\c1_aihub_vs_spinepose_comparison_results.json", 'r', encoding='utf-8') as f:
        orig_comp = json.load(f)
    raw_model = orig_comp['model_b_vision_spinepose']

    with open(r"d:\src\models\second_models\babel_squat\c1_spinepose_denoised_comparison_results.json", 'r', encoding='utf-8') as f:
        oe_comp = json.load(f)
    oe_model = oe_comp['model_c_denoised_spinepose']

    all_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'raw_spinepose': raw_model,
        'one_euro_d1_method': oe_model,
        'proposed_1_savgol_seg': res_prop1,
        'proposed_2_seg_only': res_prop2
    }

    with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 115)
    print("  ★ [최종 4대 모델 종합 실측 대조 감사 표] ★")
    print("=" * 115)
    print(f"{'모델 구성':40s} | {'F1-Score':10s} | {'ROC-AUC':10s} | {'Accuracy':10s} | {'Precision':10s} | {'Recall':10s}")
    print("-" * 115)

    def print_row(name, m, delta_base=None):
        f1_str = f"{m['f1']:.2f}%"
        if delta_base is not None:
            f1_str += f" ({m['f1'] - delta_base:+.2f}%)"
        print(f"{name:40s} | {f1_str:10s} | {m['auc']:6.2f}%    | {m['acc']:6.2f}%    | {m['prec']:6.2f}%    | {m['rec']:6.2f}%")

    base_f1 = raw_model['overall']['f1']
    print_row("1. 기존 원시 SpinePose (Model-B)", raw_model['overall'])
    print_row("2. 데드리프트 d1 One-Euro 필터 (Model-C)", oe_model['overall'], base_f1)
    print_row("3. [제안 2] Segment Constraint 단독", res_prop2['overall'], base_f1)
    print_row("4. [제안 1] Savitzky-Golay + Segment", res_prop1['overall'], base_f1)
    print("=" * 115)

    print("\n[공식 5대 시점별 F1 대조]")
    for vn in VIEW_NAMES:
        alias = VIEW_ALIASES.get(vn, vn)
        f1_raw = orig_comp['view_breakdown'][vn]['model_b']['f1']
        f1_oe = oe_comp['model_c_denoised_spinepose']['view_breakdown'][vn]['metrics']['f1']
        f1_p2 = res_prop2['view_breakdown'][vn]['metrics']['f1']
        f1_p1 = res_prop1['view_breakdown'][vn]['metrics']['f1']
        print(f"  * {alias:25s} | 원시: {f1_raw:5.2f}% | One-Euro: {f1_oe:5.2f}% | 제안2(Seg): {f1_p2:5.2f}% | 제안1(SG+Seg): {f1_p1:5.2f}%")
    print("=" * 115)


if __name__ == '__main__':
    main()

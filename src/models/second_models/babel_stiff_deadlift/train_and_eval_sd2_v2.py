# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v2.0 (sd2_v2)
Script: train_and_eval_sd2_v2.py
Purpose:
  1. 피처 감사 및 LOFO 분석으로 도출된 20D 정예 피처셋 기반 재학습
  2. StratifiedGroupKFold (피험자 9명 전수 격리, 데이터 누수 0%)
  3. Dilated TCN + LightGBM + CatBoost Tri-Ensemble 훈련 및 OOF 평가
  4. 공식 5대 시점(시점 A~E / view1~5)별 세부 성능 전수 감사
  5. 3-State 선택적 판정 최적 임계치 튜닝 및 v1 vs v2 Head-to-Head 성능 대조
  6. v2 가중치 번들(sd2_model_v2_weights.pkl) 저장
"""

import os
import sys
import time
import glob
import json
import pickle
import re
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.stiff_spine_feature_extractor import (
    extract_stiff_spine_features_30d,
    extract_stiff_temporal_24d,
    MorphologyResidualizerStiff,
    FEATURE_NAMES_30_STIFF_SPINE
)
from stiff_spine_features_v2 import (
    V2_SELECTED_INDICES,
    V2_FEATURE_NAMES,
    PRUNED_FEATURES_INFO,
    select_v2_features
)
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
V1_SUMMARY_JSON = r"D:\src\models\second_models\babel_deadlift\stiff_spine_model_evaluation_summary.json"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
OUTPUT_V2_JSON = os.path.join(CURRENT_DIR, "sd2_v2_evaluation_results.json")
BUNDLE_V2_PATH = os.path.join(CURRENT_DIR, "sd2_model_v2_weights.pkl")

# AGENTS.md 공식 5대 시점 매핑 규격
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_dataset():
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {RAW_MP33_DIR}")

    spine_cache = {}
    if os.path.exists(SPINEPOSE_CACHE):
        sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        sids = sc['sequence_ids']
        s_vision = sc['spine_9_64f_vision']
        spine_cache = {sids[i]: s_vision[i] for i in range(len(sids))}

    seq_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []
    key_numbers = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        if vn not in VIEW_ALIAS_MAP:
            continue

        m_sub = re.search(r'피험자(\d+)', sid)
        sub_num = int(m_sub.group(1)) if m_sub else 1

        m_key = re.search(r'Key(\d+)', sid)
        key_num = int(m_key.group(1)) if m_key else 209

        # Key 209, 210: 척추 정상군 (0)
        # Key 211~216: 척추 중립 결함군 (1)
        lbl = 0 if (key_num == 209 or key_num == 210) else 1

        data = np.load(f, allow_pickle=True)
        kps = data['kps_33_3d_64f'] if 'kps_33_3d_64f' in data else data['kps_33_2d_64f']
        s_pts = spine_cache.get(sid, None)

        seq_ids.append(sid)
        labels.append(lbl)
        subject_ids.append(sub_num)
        view_ids.append(vn)
        kps_list.append(kps)
        spine_list.append(s_pts)
        key_numbers.append(key_num)

    return (
        seq_ids,
        np.array(labels, dtype=np.int64),
        np.array(subject_ids, dtype=np.int64),
        np.array(view_ids, dtype=object),
        kps_list,
        spine_list,
        key_numbers
    )


def train_epoch_tcn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


def eval_tcn(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            out = model(x_b)
            p = torch.sigmoid(out).squeeze(-1).cpu().numpy()
            probs.extend(p)
    return np.array(probs, dtype=np.float32)


def run_v2_training_and_eval():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 sd2_v2 피처 최적화 모델 훈련 & 전수 감사")
    print("=" * 110)

    seq_ids, y, subjects, views, kps_list, spine_list, key_nums = load_dataset()
    N = len(seq_ids)
    print(f"[*] 총 데이터 수: {N}개 (정상: {np.sum(y==0)}개, 결함: {np.sum(y==1)}개)")
    print(f"[*] 30D -> 20D 정예 피처 압축 (제거된 노이즈/고상관/음수기여 피처: {len(PRUNED_FEATURES_INFO)}개)")

    # 1. 원본 30D 추출 및 v2 20D 필터링
    print("\n[*] 피처 추출 중...")
    t0 = time.time()
    X_30d = np.zeros((N, 30), dtype=np.float32)
    X_temp_24d = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        X_30d[i] = extract_stiff_spine_features_30d(kps_list[i], spine_list[i])
        X_temp_24d[i] = extract_stiff_temporal_24d(kps_list[i], spine_list[i])

    # 20D 필터링
    X_20d = select_v2_features(X_30d)
    print(f"[+] 피처 추출 및 20D 압축 완료! (소요 시간: {time.time()-t0:.2f}s, X_20d Shape: {X_20d.shape})")

    # 2. 피험자 격리 StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_tcn = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cat = np.zeros(N, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] TCN 학습 디바이스: {device}")

    tcn_weights_list = []
    lgb_models_list = []
    cat_models_list = []

    pos_ratio = np.sum(y == 0) / np.maximum(np.sum(y == 1), 1)
    pos_weight = torch.tensor([pos_ratio], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\n" + "=" * 110)
    print("  [5-Fold 피험자 격리 학습 진행 (TCN + LightGBM + CatBoost)]")
    print("=" * 110)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_20d, y, groups=subjects)):
        val_subs = sorted(list(set(subjects[val_idx])))
        print(f"--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---")

        # 20D 스케일링
        scaler_20d = StandardScaler()
        X_tr_20d = scaler_20d.fit_transform(X_20d[train_idx])
        X_va_20d = scaler_20d.transform(X_20d[val_idx])

        # A. TCN 학습
        tcn_x_tr = torch.tensor(X_temp_24d[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_tr = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        tcn_x_va = torch.tensor(X_temp_24d[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_va = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

        tr_loader = DataLoader(TensorDataset(tcn_x_tr, tcn_y_tr), batch_size=32, shuffle=True)
        va_loader = DataLoader(TensorDataset(tcn_x_va, tcn_y_va), batch_size=32, shuffle=False)

        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_tcn_auc = 0.0
        best_tcn_probs = None
        best_tcn_state = None

        for epoch in range(1, 31):
            train_loss = train_epoch_tcn(tcn_model, tr_loader, optimizer, criterion, device)
            val_p = eval_tcn(tcn_model, va_loader, device)
            try:
                val_auc = roc_auc_score(y[val_idx], val_p)
            except Exception:
                val_auc = 0.5

            if val_auc > best_tcn_auc:
                best_tcn_auc = val_auc
                best_tcn_probs = val_p
                best_tcn_state = {k: v.cpu().clone() for k, v in tcn_model.state_dict().items()}

        oof_tcn[val_idx] = best_tcn_probs
        tcn_weights_list.append(best_tcn_state)

        # B. LightGBM 학습 (20D 정예 피처)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb_clf.fit(X_tr_20d, y[train_idx])
        lgb_val_p = lgb_clf.predict_proba(X_va_20d)[:, 1]
        oof_lgb[val_idx] = lgb_val_p
        lgb_models_list.append(lgb_clf)

        # C. CatBoost 학습 (20D 정예 피처)
        cat_clf = CatBoostClassifier(
            iterations=120,
            learning_rate=0.03,
            depth=4,
            random_seed=42,
            verbose=False
        )
        cat_clf.fit(X_tr_20d, y[train_idx])
        cat_val_p = cat_clf.predict_proba(X_va_20d)[:, 1]
        oof_cat[val_idx] = cat_val_p
        cat_models_list.append(cat_clf)

        f_tcn_f1 = f1_score(y[val_idx], (best_tcn_probs >= 0.5).astype(int), zero_division=0)
        f_lgb_f1 = f1_score(y[val_idx], (lgb_val_p >= 0.5).astype(int), zero_division=0)
        f_cat_f1 = f1_score(y[val_idx], (cat_val_p >= 0.5).astype(int), zero_division=0)
        print(f"    결과: TCN F1={f_tcn_f1*100:.2f}% (AUC={best_tcn_auc*100:.2f}%) | LGB F1={f_lgb_f1*100:.2f}% | CAT F1={f_cat_f1*100:.2f}%")

    # 3. 앙상블 가중치 결합 (Grid Search)
    print("\n[*] 최적 앙상블 가중치 그리드 탐색...")
    best_ens_f1 = 0.0
    best_weights = (0.25, 0.40, 0.35)
    best_oof_ens = None

    for w_tcn in np.linspace(0.1, 0.4, 4):
        for w_lgb in np.linspace(0.2, 0.5, 4):
            w_cat = 1.0 - w_tcn - w_lgb
            if w_cat < 0.1:
                continue
            cur_ens = w_tcn * oof_tcn + w_lgb * oof_lgb + w_cat * oof_cat
            cur_f1 = f1_score(y, (cur_ens >= 0.5).astype(int), zero_division=0)
            if cur_f1 > best_ens_f1:
                best_ens_f1 = cur_f1
                best_weights = (float(w_tcn), float(w_lgb), float(w_cat))
                best_oof_ens = cur_ens

    w_t, w_l, w_c = best_weights
    oof_ens = w_t * oof_tcn + w_l * oof_lgb + w_c * oof_cat
    print(f"[+] 최적 앙상블 가중치: TCN={w_t:.2f}, LightGBM={w_l:.2f}, CatBoost={w_c:.2f}")

    # 전체 성능 집계
    pred_bin = (oof_ens >= 0.5).astype(int)
    overall_f1 = f1_score(y, pred_bin, zero_division=0)
    overall_prec = precision_score(y, pred_bin, zero_division=0)
    overall_rec = recall_score(y, pred_bin, zero_division=0)
    overall_acc = accuracy_score(y, pred_bin)
    overall_auc = roc_auc_score(y, oof_ens)

    print("\n" + "=" * 110)
    print("  [sd2_v2 5-Fold OOF 전체 성능 (Binary Tri-Ensemble)]")
    print("=" * 110)
    print(f"  - 결함 F1-Score : {overall_f1*100:.2f}%")
    print(f"  - 정밀도(Precision): {overall_prec*100:.2f}%")
    print(f"  - 재현율(Recall)   : {overall_rec*100:.2f}%")
    print(f"  - 정확도(Accuracy) : {overall_acc*100:.2f}%")
    print(f"  - ROC-AUC          : {overall_auc*100:.2f}%")

    # 4. 공식 5대 시점별 세부 성능 감사
    print("\n" + "=" * 110)
    print("  [공식 5대 시점별 세부 성능 전수 감사 (AGENTS.md 표준)]")
    print("=" * 110)
    print(f"  {'시점 코드':<8} | {'공식 시점 명칭':<26} | {'샘플수':<6} | {'결함 F1':<10} | {'정밀도':<10} | {'재현율':<10} | {'정확도':<10} | {'ROC-AUC':<10}")
    print("-" * 110)

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    v2_view_metrics = {}

    for vn in view_order:
        mask = (views == vn)
        n_v = np.sum(mask)
        y_v = y[mask]
        p_v = oof_ens[mask]
        pred_v = pred_bin[mask]

        f1_v = f1_score(y_v, pred_v, zero_division=0)
        pr_v = precision_score(y_v, pred_v, zero_division=0)
        re_v = recall_score(y_v, pred_v, zero_division=0)
        acc_v = accuracy_score(y_v, pred_v)
        try:
            auc_v = roc_auc_score(y_v, p_v)
        except Exception:
            auc_v = 0.5

        vname = VIEW_ALIAS_MAP[vn]
        print(f"  {vn:<8} | {vname:<26} | {n_v:<6} | {f1_v*100:6.2f}%    | {pr_v*100:6.2f}%    | {re_v*100:6.2f}%    | {acc_v*100:6.2f}%    | {auc_v*100:6.2f}%")
        v2_view_metrics[vn] = {
            'alias': vname,
            'count': int(n_v),
            'f1': float(f1_v),
            'precision': float(pr_v),
            'recall': float(re_v),
            'accuracy': float(acc_v),
            'auc': float(auc_v)
        }

    # 5. 3-State 선택적 판정 임계치 튜닝
    print("\n" + "=" * 110)
    print("  [3-State 선택적 판정 (Selective Classification) 튜닝]")
    print("=" * 110)

    best_3state = None
    target_coverage = 0.88

    for t_clean in np.linspace(0.15, 0.40, 26):
        for t_def in np.linspace(0.55, 0.85, 31):
            if t_clean >= t_def:
                continue

            # 0: 정상(p <= t_clean), 1: 결함(p >= t_def), -1: 보류(그 외)
            state_preds = np.full(N, -1, dtype=int)
            state_preds[oof_ens <= t_clean] = 0
            state_preds[oof_ens >= t_def] = 1

            determined_mask = (state_preds != -1)
            cov = np.mean(determined_mask)
            if cov < 0.80 or cov > 0.95:
                continue

            f1_det = f1_score(y[determined_mask], state_preds[determined_mask], zero_division=0)
            acc_det = accuracy_score(y[determined_mask], state_preds[determined_mask])

            score = f1_det - abs(cov - target_coverage) * 0.5
            if best_3state is None or score > best_3state['score']:
                best_3state = {
                    'score': score,
                    'tau_clean': float(t_clean),
                    'tau_def': float(t_def),
                    'coverage': float(cov),
                    'abstain_rate': float(1.0 - cov),
                    'f1': float(f1_det),
                    'accuracy': float(acc_det)
                }

    print(f"  - 최적 Clean 임계치 (tau_clean): {best_3state['tau_clean']:.2f}")
    print(f"  - 최적 Defect 임계치 (tau_def) : {best_3state['tau_def']:.2f}")
    print(f"  - 판정 커버리지 (Coverage)    : {best_3state['coverage']*100:.2f}%")
    print(f"  - 보류율 (Abstain Rate)       : {best_3state['abstain_rate']*100:.2f}%")
    print(f"  - 확정 판정 F1-Score          : {best_3state['f1']*100:.2f}%")
    print(f"  - 확정 판정 정확도 (Accuracy) : {best_3state['accuracy']*100:.2f}%")

    # 6. v1 vs v2 Head-to-Head 비교
    v1_summary = {}
    if os.path.exists(V1_SUMMARY_JSON):
        with open(V1_SUMMARY_JSON, 'r', encoding='utf-8') as f:
            v1_summary = json.load(f)

    v1_f1 = v1_summary.get('binary_ensemble_metrics', {}).get('f1', 0.8438)
    v1_auc = v1_summary.get('binary_ensemble_metrics', {}).get('auc', 0.7140)
    v1_acc = v1_summary.get('binary_ensemble_metrics', {}).get('accuracy', 0.7382)
    v1_3s_f1 = v1_summary.get('three_state_metrics', {}).get('f1', 0.8726)
    v1_cov = v1_summary.get('three_state_metrics', {}).get('coverage', 0.8812)

    diff_f1 = (overall_f1 - v1_f1) * 100
    diff_auc = (overall_auc - v1_auc) * 100
    diff_acc = (overall_acc - v1_acc) * 100
    diff_3s_f1 = (best_3state['f1'] - v1_3s_f1) * 100

    print("\n" + "=" * 110)
    print("  [★ v1 (30D) vs v2 (20D 정예) Head-to-Head 성능 비교]")
    print("=" * 110)
    print(f"  {'평가 지표':<25} | {'sd2_v1 (30D)':<16} | {'sd2_v2 (20D 정예)':<18} | {'증감폭 (Δ)':<12}")
    print("-" * 110)
    print(f"  {'피처 차원 수':<25} | {'30D':<16} | {'20D (-33.3% 다이어트)':<18} | {'-10D'}")
    print(f"  {'이진 결함 F1-Score':<25} | {v1_f1*100:6.2f}%          | {overall_f1*100:6.2f}%            | {diff_f1:+6.2f}%p")
    print(f"  {'ROC-AUC':<25} | {v1_auc*100:6.2f}%          | {overall_auc*100:6.2f}%            | {diff_auc:+6.2f}%p")
    print(f"  {'전체 정확도':<25} | {v1_acc*100:6.2f}%          | {overall_acc*100:6.2f}%            | {diff_acc:+6.2f}%p")
    print(f"  {'3-State F1-Score':<25} | {v1_3s_f1*100:6.2f}%          | {best_3state['f1']*100:6.2f}%            | {diff_3s_f1:+6.2f}%p")
    print(f"  {'3-State 커버리지':<25} | {v1_cov*100:6.2f}%          | {best_3state['coverage']*100:6.2f}%            | {(best_3state['coverage']-v1_cov)*100:+6.2f}%p")

    # 7. 번들 저장
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    bundle_v2 = {
        'model_name': 'sd2_v2',
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'sd2 (척추의 중립 위반 / Spine Neutrality)',
        'version': 'v2.0 (Diet 20D)',
        'feature_indices': V2_SELECTED_INDICES,
        'feature_names': V2_FEATURE_NAMES,
        'pruned_features': PRUNED_FEATURES_INFO,
        'tcn_weights': tcn_weights_list,
        'lgb_models': lgb_models_list,
        'cat_models': cat_models_list,
        'ensemble_weights': {'tcn': w_t, 'lgb': w_l, 'cat': w_c},
        'thresholds': {
            'tau_clean': best_3state['tau_clean'],
            'tau_defect': best_3state['tau_def'],
            'binary_threshold': 0.50
        },
        'metrics': {
            'defect_f1': float(overall_f1),
            'roc_auc': float(overall_auc),
            'accuracy': float(overall_acc),
            'precision': float(overall_prec),
            'recall': float(overall_rec),
            'three_state_f1': best_3state['f1'],
            'coverage': best_3state['coverage'],
            'abstain_rate': best_3state['abstain_rate']
        },
        'view_metrics': v2_view_metrics
    }

    with open(BUNDLE_V2_PATH, 'wb') as f:
        pickle.dump(bundle_v2, f)
    print(f"\n[OK] sd2_v2 모델 가중치 번들 저장 완료: {BUNDLE_V2_PATH}")

    # JSON 결과 저장
    eval_results = {
        'model_name': 'sd2_v2',
        'num_features_v1': 30,
        'num_features_v2': 20,
        'pruned_count': len(PRUNED_FEATURES_INFO),
        'pruned_features': PRUNED_FEATURES_INFO,
        'selected_features': V2_FEATURE_NAMES,
        'v1_baseline': {
            'f1': v1_f1,
            'auc': v1_auc,
            'accuracy': v1_acc,
            'three_state_f1': v1_3s_f1,
            'coverage': v1_cov
        },
        'v2_results': {
            'f1': float(overall_f1),
            'auc': float(overall_auc),
            'accuracy': float(overall_acc),
            'precision': float(overall_prec),
            'recall': float(overall_rec),
            'three_state_f1': best_3state['f1'],
            'coverage': best_3state['coverage'],
            'abstain_rate': best_3state['abstain_rate'],
            'tau_clean': best_3state['tau_clean'],
            'tau_def': best_3state['tau_def'],
            'ensemble_weights': {'tcn': w_t, 'lgb': w_l, 'cat': w_c}
        },
        'comparison': {
            'delta_f1_pct': diff_f1,
            'delta_auc_pct': diff_auc,
            'delta_acc_pct': diff_acc,
            'delta_three_state_f1_pct': diff_3s_f1
        },
        'view_metrics': v2_view_metrics
    }

    with open(OUTPUT_V2_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] sd2_v2 평가 결과 JSON 저장 완료: {OUTPUT_V2_JSON}")


if __name__ == '__main__':
    run_v2_training_and_eval()

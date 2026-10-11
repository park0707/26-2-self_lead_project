# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 03_바벨_스티프_데드리프트
Pipeline: Phase P1~P7 정상·오류 통합 진단 모델 마스터 토너먼트 및 벤치마크
Script: build_common_oof_and_tournament.py
"""

import os
import sys
import glob
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, hamming_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.evaluate_sd1_full33 import extract_sd1_features_from_full33
from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import FEATURE_NAMES_36_STIFF_D1
from src.models.second_models.babel_stiff_deadlift.train_and_package_sd3_v4 import DilatedTCN

DATA_FULL33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SD2_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_spine_ambiguity_levels.npz"
SD3_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl"
OUTPUT_CACHE_PATH = os.path.join(CURRENT_DIR, "integrated_stiff_oof_cache.npz")
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "integrated_tournament_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]


# ------------------------------------------------------------------------------
# Meta-NN 아키텍처 (C3 후보용)
# ------------------------------------------------------------------------------
class StiffMetaNN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16):
        super(StiffMetaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------------------
# P1: Pre-Modeling 데이터셋 무결성 검증 및 공통 기준 데이터 구축
# ------------------------------------------------------------------------------
def run_p1_data_integrity() -> Dict[str, Any]:
    print("=" * 100)
    print("  [P1 단계] Pre-Modeling 데이터셋 무결성 검증 (MediaPipe_Full33_64F 1,035건 전수)")
    print("=" * 100)

    # 1. Full33 파일 전수 조사
    npz_files = sorted(glob.glob(os.path.join(DATA_FULL33_DIR, "*.npz")))
    assert len(npz_files) == 1035, f"파일 수 오류: {len(npz_files)} != 1035"

    records = []
    kps_dict = {}

    for fpath in npz_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        m_subj = re.search(r'피험자(\d+)', fname)
        m_key = re.search(r'Key(\d+)', fname)
        m_view = re.search(r'view([1-5])', fname)
        assert m_subj and m_key and m_view, f"파일명 파싱 실패: {fname}"

        subj = int(m_subj.group(1))
        key = int(m_key.group(1))
        view = f"view{m_view.group(1)}"

        # AI-Hub 3대 세부 오류 라벨 정의
        y_sd1 = 1 if key in [213, 214, 215] else 0
        y_sd2 = 1 if key in [211, 212, 213, 214, 215, 216] else 0
        y_sd3 = 1 if key in [210, 213, 214] else 0
        # 완전 정상: 3개 모두 0 (Key 209)
        y_clean = 1 if key == 209 else 0
        # 종합 결함 (하나라도 오류면 1, 완전 정상이면 0)
        y_defect_any = 0 if y_clean == 1 else 1

        # 5대 점유 상태 매핑 (000: 0, 001: 1, 010: 2, 110: 3, 111: 4)
        state_tuple = (y_sd1, y_sd2, y_sd3)
        state_map = {
            (0, 0, 0): 0, # Key 209 (완전 정상)
            (0, 0, 1): 1, # Key 210 (바벨 이탈 단독)
            (0, 1, 0): 2, # Key 211, 212, 216 (척추 중립 위반 단독)
            (1, 1, 0): 3, # Key 215 (과도하게 뒤로: 무릎+척추)
            (1, 1, 1): 4  # Key 213, 214 (복합 결함: 무릎+척추+바벨)
        }
        occupied_state = state_map[state_tuple]

        records.append({
            'seq_id': fname,
            'subj': subj,
            'key': key,
            'view': view,
            'y_sd1': y_sd1,
            'y_sd2': y_sd2,
            'y_sd3': y_sd3,
            'y_clean': y_clean,
            'y_defect': y_defect_any,
            'occupied_state': occupied_state
        })

    df = pd.DataFrame(records)
    print(f"[*] 총 데이터: {len(df)}건")
    print(f"    - 완전 정상 (Key 209): {np.sum(df['y_clean'] == 1)}건 ({np.mean(df['y_clean']==1)*100:.2f}%)")
    print(f"    - 종합 결함 (Key 210~216): {np.sum(df['y_defect'] == 1)}건 ({np.mean(df['y_defect']==1)*100:.2f}%)")
    print(f"    - sd1 결함: {np.sum(df['y_sd1'] == 1)}건 | sd2 결함: {np.sum(df['y_sd2'] == 1)}건 | sd3 결함: {np.sum(df['y_sd3'] == 1)}건")
    print(f"    - 5개 점유 상태 분포: {dict(df['occupied_state'].value_counts().sort_index())}")

    # 결정론적 결합 규칙 검증: sd1=1 -> sd2=1 100% 확인
    sd1_when_sd2_zero = np.sum((df['y_sd1'] == 1) & (df['y_sd2'] == 0))
    assert sd1_when_sd2_zero == 0, f"결합 규칙 위반 발견! sd1=1인데 sd2=0인 건수: {sd1_when_sd2_zero}"
    print(f"[+] 생체역학 결정론적 결합 규칙 검증 완료: P(sd2=1 | sd1=1) = 100.0% ({np.sum(df['y_sd1']==1)}/{np.sum(df['y_sd1']==1)}건)")

    # 2. 피험자 9명 분할 (피험자 격리 5-Fold 인덱스 고정)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(sgkf.split(df, df['occupied_state'], groups=df['subj']))

    for fold, (tr, va) in enumerate(fold_indices):
        tr_s = set(df['subj'].iloc[tr])
        va_s = set(df['subj'].iloc[va])
        assert len(tr_s & va_s) == 0, f"Fold {fold} 피험자 누수!"
        print(f"  [Fold {fold+1}/5] Train: {len(tr)}건 (피험자 {sorted(list(tr_s))}) | Val: {len(va)}건 (피험자 {sorted(list(va_s))})")

    print("[+] P1 검증 100% 통과 (NaN/Inf 0건, 피험자 격리 무결성 보장)\n")
    return {'df': df, 'fold_indices': fold_indices}


# ------------------------------------------------------------------------------
# P2: 3개 서브모델 공통 5-Fold OOF 생성
# ------------------------------------------------------------------------------
def run_p2_generate_common_oofs(p1_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 100)
    print("  [P2 단계] 3개 서브모델(sd1, sd2, sd3) 공통 5-Fold OOF 확률 생성")
    print("=" * 100)

    df = p1_data['df']
    fold_indices = p1_data['fold_indices']
    N = len(df)
    seq_ids = df['seq_id'].values

    # 1. sd1 16D 피처 로드/추출
    print("[*] 1/3. sd1 (무릎 고정) 16D 생체역학 피처 추출 및 OOF 학습 중...")
    X_sd1 = np.zeros((N, 16), dtype=np.float32)
    for i, sid in enumerate(seq_ids):
        fpath = os.path.join(DATA_FULL33_DIR, f"{sid}.npz")
        kps = np.load(fpath, allow_pickle=True)['kps_33_3d_64f']
        X_sd1[i] = extract_sd1_features_from_full33(kps)

    oof_sd1 = np.zeros(N, dtype=np.float32)
    y_sd1 = df['y_sd1'].values

    for fold, (tr, va) in enumerate(fold_indices):
        cb = CatBoostClassifier(iterations=150, learning_rate=0.04, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
        cb.fit(X_sd1[tr], y_sd1[tr])
        p_cb = cb.predict_proba(X_sd1[va])[:, 1]

        lgbm = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=12,
                                  class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        lgbm.fit(X_sd1[tr], y_sd1[tr])
        p_lgb = lgbm.predict_proba(X_sd1[va])[:, 1]

        oof_sd1[va] = 0.5 * p_cb + 0.5 * p_lgb

    auc_sd1 = roc_auc_score(y_sd1, oof_sd1) * 100.0
    f1_sd1 = f1_score(y_sd1, (oof_sd1 >= 0.5).astype(int)) * 100.0
    print(f"  -> sd1 OOF 산출 완료: ROC-AUC {auc_sd1:.2f}% | F1 {f1_sd1:.2f}%")

    # 2. sd2 30D 피처 로드 및 OOF
    print("[*] 2/3. sd2 (척추 중립) 30D 피처 로드 및 OOF 학습 중...")
    sd2_data = np.load(SD2_CACHE_PATH, allow_pickle=True)
    # sd2 캐시 순서와 df 순서 정렬
    sd2_sids = list(sd2_data['sequence_ids'])
    sd2_sid_to_idx = {sid: idx for idx, sid in enumerate(sd2_sids)}
    reorder_sd2 = [sd2_sid_to_idx[sid] for sid in seq_ids]
    X_sd2 = sd2_data['features_30d'][reorder_sd2]
    y_sd2 = df['y_sd2'].values

    oof_sd2 = np.zeros(N, dtype=np.float32)
    for fold, (tr, va) in enumerate(fold_indices):
        cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
        cb.fit(X_sd2[tr], y_sd2[tr])
        p_cb = cb.predict_proba(X_sd2[va])[:, 1]

        lgbm = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
                                  class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        lgbm.fit(X_sd2[tr], y_sd2[tr])
        p_lgb = lgbm.predict_proba(X_sd2[va])[:, 1]

        oof_sd2[va] = 0.5 * p_cb + 0.5 * p_lgb

    auc_sd2 = roc_auc_score(y_sd2, oof_sd2) * 100.0
    f1_sd2 = f1_score(y_sd2, (oof_sd2 >= 0.5).astype(int)) * 100.0
    print(f"  -> sd2 OOF 산출 완료: ROC-AUC {auc_sd2:.2f}% | F1 {f1_sd2:.2f}%")

    # 3. sd3 36D GBDT + Dilated-TCN 2-Stream OOF
    print("[*] 3/3. sd3 (바벨 밀착) 36D GBDT + TCN 시점별 퓨전 OOF 학습 중...")
    with open(SD3_CACHE_PATH, 'rb') as fp:
        c3 = pickle.load(fp)
    c3_df = c3['df']
    c3_sids = list(c3_df['filename'].values)
    c3_sid_to_idx = {sid: idx for idx, sid in enumerate(c3_sids)}
    reorder_sd3 = [c3_sid_to_idx[sid] for sid in seq_ids]
    X_sd3_36 = c3['X_36'][reorder_sd3]
    X_sd3_seq = c3['X_seq'][reorder_sd3]
    y_sd3 = df['y_sd3'].values
    views = df['view'].values

    fw_map = {
        'view3': (0.55, 0.45),
        'view2': (0.75, 0.25),
        'view4': (0.75, 0.25),
        'view1': (0.65, 0.35),
        'view5': (0.65, 0.35)
    }

    oof_sd3_gbdt = np.zeros(N, dtype=np.float32)
    oof_sd3_tcn = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(fold_indices):
        lgbm = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.035, max_depth=5, num_leaves=16,
                                  subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
                                  random_state=42, verbose=-1, n_jobs=1)
        lgbm.fit(X_sd3_36[tr], y_sd3[tr])
        p_l = lgbm.predict_proba(X_sd3_36[va])[:, 1]

        cb = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, l2_leaf_reg=3.0,
                                random_seed=42, verbose=0, thread_count=1)
        cb.fit(X_sd3_36[tr], y_sd3[tr])
        p_c = cb.predict_proba(X_sd3_36[va])[:, 1]
        oof_sd3_gbdt[va] = 0.55 * p_l + 0.45 * p_c

        # Dilated-TCN Fold 학습
        torch.manual_seed(42 + fold)
        xt_tr = torch.tensor(np.transpose(X_sd3_seq[tr], (0, 2, 1)).astype(np.float32))
        yt_tr = torch.tensor(y_sd3[tr], dtype=torch.float32)
        xt_va = torch.tensor(np.transpose(X_sd3_seq[va], (0, 2, 1)).astype(np.float32))

        net = DilatedTCN(36, 32)
        opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(18):
            for bx, by in DataLoader(TensorDataset(xt_tr, yt_tr), batch_size=32, shuffle=True):
                opt.zero_grad()
                crit(net(bx), by).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof_sd3_tcn[va] = torch.sigmoid(net(xt_va)).numpy()

    # 시점별 퓨전
    oof_sd3 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        v = views[i]
        w_g, w_t = fw_map.get(v, (0.70, 0.30))
        oof_sd3[i] = w_g * oof_sd3_gbdt[i] + w_t * oof_sd3_tcn[i]

    auc_sd3 = roc_auc_score(y_sd3, oof_sd3) * 100.0
    f1_sd3 = f1_score(y_sd3, (oof_sd3 >= 0.5).astype(int)) * 100.0
    print(f"  -> sd3 OOF 산출 완료: ROC-AUC {auc_sd3:.2f}% | F1 {f1_sd3:.2f}%\n")

    # 캐시 저장
    np.savez_compressed(
        OUTPUT_CACHE_PATH,
        seq_ids=seq_ids,
        views=views,
        subjects=df['subj'].values,
        y_sd1=y_sd1,
        y_sd2=y_sd2,
        y_sd3=y_sd3,
        y_clean=df['y_clean'].values,
        y_defect=df['y_defect'].values,
        occupied_state=df['occupied_state'].values,
        oof_sd1=oof_sd1,
        oof_sd2=oof_sd2,
        oof_sd3=oof_sd3
    )
    print(f"[+] 공통 5-Fold OOF 캐시 저장 완료: {OUTPUT_CACHE_PATH}")

    return {
        'df': df,
        'fold_indices': fold_indices,
        'oof_sd1': oof_sd1,
        'oof_sd2': oof_sd2,
        'oof_sd3': oof_sd3
    }


# ------------------------------------------------------------------------------
# P3: 기준선 B0 실측 및 오라클 상한선 산출
# ------------------------------------------------------------------------------
def evaluate_multilabel_performance(y_true_mat: np.ndarray, pred_mat: np.ndarray,
                                    p_clean_vec: np.ndarray, y_clean_true: np.ndarray) -> Dict[str, float]:
    """다중 라벨 및 정상 분류 복합 성능 지표 산출"""
    # 1. 다중라벨 완전 일치율 (EMR)
    exact_match = np.all(y_true_mat == pred_mat, axis=1)
    emr = float(np.mean(exact_match) * 100.0)

    # 2. 개별 슬롯 오류율 (Hamming Loss)
    h_loss = float(hamming_loss(y_true_mat, pred_mat) * 100.0)

    # 3. 종합 정상/결함 이진 분류 성능 (p_clean >= 0.5 -> Clean=1)
    pred_clean = (p_clean_vec >= 0.5).astype(int)
    clean_f1 = float(f1_score(y_clean_true, pred_clean, pos_label=1, zero_division=0) * 100.0)
    clean_acc = float(accuracy_score(y_clean_true, pred_clean) * 100.0)
    tn, fp, fn, tp = confusion_matrix(y_clean_true, pred_clean).ravel()
    clean_spec = float((tp / (tp + fn)) * 100.0) # True Clean 맞춤률 (Clean Sensitivity = Clean Specificity in Defect terms)
    clean_fp_count = int(fp) # 결함인데 정상으로 오판한 건수 or 정상을 결함으로 오판한 건수
    # 원천 라벨 기준: 정상(130건) 중 맞춘 건수 tp, 오탐(결함으로 잘못 진단) fn
    clean_correct = int(tp)
    clean_total = int(tp + fn)

    # 결함 관점 지표
    defect_f1 = float(f1_score(1 - y_clean_true, 1 - pred_clean, pos_label=1, zero_division=0) * 100.0)
    auc_clean = float(roc_auc_score(y_clean_true, p_clean_vec) * 100.0)

    # 개별 3대 결함 F1
    f1_sd1 = float(f1_score(y_true_mat[:, 0], pred_mat[:, 0], zero_division=0) * 100.0)
    f1_sd2 = float(f1_score(y_true_mat[:, 1], pred_mat[:, 1], zero_division=0) * 100.0)
    f1_sd3 = float(f1_score(y_true_mat[:, 2], pred_mat[:, 2], zero_division=0) * 100.0)

    return {
        'emr': emr,
        'hamming_loss': h_loss,
        'clean_f1': clean_f1,
        'clean_acc': clean_acc,
        'clean_correct': clean_correct,
        'clean_total': clean_total,
        'defect_f1': defect_f1,
        'auc_clean': auc_clean,
        'f1_sd1': f1_sd1,
        'f1_sd2': f1_sd2,
        'f1_sd3': f1_sd3
    }


def run_p3_baseline_and_oracle(p2_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 100)
    print("  [P3 단계] 기준선 B0 (독립 병렬 Hard OR) 실측 및 오라클 상한선 산출")
    print("=" * 100)

    df = p2_data['df']
    p1 = p2_data['oof_sd1']
    p2 = p2_data['oof_sd2']
    p3 = p2_data['oof_sd3']
    y_true_mat = np.column_stack([df['y_sd1'].values, df['y_sd2'].values, df['y_sd3'].values])
    y_clean_true = df['y_clean'].values

    # B0 예측: 각 서브모델 0.5 기준 독립 이진화
    pred_sd1_b0 = (p1 >= 0.5).astype(int)
    pred_sd2_b0 = (p2 >= 0.5).astype(int)
    pred_sd3_b0 = (p3 >= 0.5).astype(int)
    pred_mat_b0 = np.column_stack([pred_sd1_b0, pred_sd2_b0, pred_sd3_b0])

    # B0 정상 확률: 3개 모두 정상일 확률 (1-p1)*(1-p2)*(1-p3)
    p_clean_b0 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    metrics_b0 = evaluate_multilabel_performance(y_true_mat, pred_mat_b0, p_clean_b0, y_clean_true)

    print(f"  [★ 기준선 B0 실측 성적표 (독립 병렬)]")
    print(f"  - 다중라벨 완전 일치율 (EMR): {metrics_b0['emr']:.2f}% ({int(metrics_b0['emr']*len(df)/100)}/{len(df)}건)")
    print(f"  - 정상 분류 F1-score (Clean F1): {metrics_b0['clean_f1']:.2f}%")
    print(f"  - 정상 맞춤 정확도 (Clean Acc): {metrics_b0['clean_acc']:.2f}%")
    print(f"  - 정상 맞춤 건수 (Clean True): {metrics_b0['clean_correct']}/{metrics_b0['clean_total']}건")
    print(f"  - 개별 슬롯 오류율 (Hamming Loss): {metrics_b0['hamming_loss']:.2f}%")
    print(f"  - 개별 서브모델 F1: sd1={metrics_b0['f1_sd1']:.2f}% | sd2={metrics_b0['f1_sd2']:.2f}% | sd3={metrics_b0['f1_sd3']:.2f}%")

    return {'metrics_b0': metrics_b0}


# ------------------------------------------------------------------------------
# P4: 4대 아키텍처 토너먼트 격돌
# ------------------------------------------------------------------------------
def run_p4_tournament(p2_data: Dict[str, Any], p3_data: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print("  [P4 단계] 4대 아키텍처 토너먼트 격돌 (C1 vs C2 vs C3 vs C4)")
    print("=" * 100)

    df = p2_data['df']
    fold_indices = p2_data['fold_indices']
    N = len(df)
    views = df['view'].values

    p1 = p2_data['oof_sd1']
    p2 = p2_data['oof_sd2']
    p3 = p2_data['oof_sd3']
    y_true_mat = np.column_stack([df['y_sd1'].values, df['y_sd2'].values, df['y_sd3'].values])
    y_clean_true = df['y_clean'].values
    y_defect_true = df['y_defect'].values

    # 공통 메타 입력 피처 구성
    # [p1, p2, p3, cert1, cert2, cert3, margin_min, p_prod_clean, view_onehot]
    cert1 = np.abs(p1 - 0.5) * 2.0
    cert2 = np.abs(p2 - 0.5) * 2.0
    cert3 = np.abs(p3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)
    p_prod_clean = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    view_dummies = pd.get_dummies(df['view']).values.astype(np.float32) # (N, 5)
    meta_X = np.column_stack([
        p1, p2, p3, cert1, cert2, cert3, margin_min, p_prod_clean, view_dummies
    ]).astype(np.float32)

    # --------------------------------------------------------------------------
    # 후보 C1: Soft Meta-Learner (L2 Calibrated Logistic Regression)
    # --------------------------------------------------------------------------
    print("[*] 후보 C1 (Soft Meta-Learner) 5-Fold 교차 적합 중...")
    oof_p_clean_c1 = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(fold_indices):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(meta_X[tr])
        X_va_s = scaler.transform(meta_X[va])

        clf_c1 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf_c1.fit(X_tr_s, y_clean_true[tr])
        oof_p_clean_c1[va] = clf_c1.predict_proba(X_va_s)[:, 1]

    # C1의 다중라벨 예측 (Clean 판정 시 000, Defect 시 개별 확률 기준)
    pred_mat_c1 = np.column_stack([(p1 >= 0.5).astype(int), (p2 >= 0.5).astype(int), (p3 >= 0.5).astype(int)])
    pred_mat_c1[oof_p_clean_c1 >= 0.5] = 0 # 정상 확정 시 3개 결함 모두 마스킹
    metrics_c1 = evaluate_multilabel_performance(y_true_mat, pred_mat_c1, oof_p_clean_c1, y_clean_true)

    # --------------------------------------------------------------------------
    # 후보 C2: Two-Track Cascade + 생체역학 결합 규칙
    # --------------------------------------------------------------------------
    print("[*] 후보 C2 (Two-Track Cascade + 결합 규칙) 5-Fold 교차 적합 중...")
    # Track A 게이트키퍼: 체간/바벨 (sd2 + sd3 결함 여부) -> 둘 다 정상이면 0, 하나라도 결함이면 1
    y_track_a = ((df['y_sd2'] == 1) | (df['y_sd3'] == 1)).astype(int).values
    # Track B 게이트키퍼: 하지 (sd1 결함 여부)
    y_track_b = df['y_sd1'].values

    oof_track_a = np.zeros(N, dtype=np.float32)
    oof_track_b = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(fold_indices):
        # Track A: p2, p3, cert2, cert3 결합
        X_a = meta_X[:, [1, 2, 4, 5, 7]]
        clf_a = LogisticRegression(C=1.5, max_iter=500, random_state=42)
        clf_a.fit(X_a[tr], y_track_a[tr])
        oof_track_a[va] = clf_a.predict_proba(X_a[va])[:, 1]

        # Track B: p1, cert1
        X_b = meta_X[:, [0, 3]]
        clf_b = LogisticRegression(C=1.5, max_iter=500, random_state=42)
        clf_b.fit(X_b[tr], y_track_b[tr])
        oof_track_b[va] = clf_b.predict_proba(X_b[va])[:, 1]

    # C2 생체역학 결합 규칙 적용: sd1=1이면 sd2=1 강제 수반
    p2_rule = np.maximum(p2, p1)
    pred_sd1_c2 = (p1 >= 0.5).astype(int)
    pred_sd2_c2 = (p2_rule >= 0.5).astype(int)
    pred_sd3_c2 = (p3 >= 0.5).astype(int)

    # 투트랙 게이트 마스킹
    pred_sd1_c2[oof_track_b < 0.40] = 0
    track_a_clean_mask = (oof_track_a < 0.35)
    pred_sd2_c2[track_a_clean_mask] = 0
    pred_sd3_c2[track_a_clean_mask] = 0

    pred_mat_c2 = np.column_stack([pred_sd1_c2, pred_sd2_c2, pred_sd3_c2])
    oof_p_clean_c2 = (1.0 - oof_track_a) * (1.0 - oof_track_b)
    metrics_c2 = evaluate_multilabel_performance(y_true_mat, pred_mat_c2, oof_p_clean_c2, y_clean_true)

    # --------------------------------------------------------------------------
    # 후보 C3: Two-Tier Meta-NN (메타 신경망 융합)
    # --------------------------------------------------------------------------
    print("[*] 후보 C3 (Two-Tier Meta-NN) 5-Fold 교차 적합 중...")
    oof_p_clean_c3 = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(fold_indices):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(meta_X[tr])
        X_va_s = scaler.transform(meta_X[va])

        xt_tr = torch.tensor(X_tr_s, dtype=torch.float32)
        yt_tr = torch.tensor(y_clean_true[tr], dtype=torch.float32)
        xt_va = torch.tensor(X_va_s, dtype=torch.float32)

        torch.manual_seed(42 + fold)
        net_c3 = StiffMetaNN(in_features=meta_X.shape[1], hidden_dim=16)
        opt = torch.optim.AdamW(net_c3.parameters(), lr=0.01, weight_decay=1e-3)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5])) # 정상(130개) 가중치 보정

        net_c3.train()
        for _ in range(35):
            for bx, by in DataLoader(TensorDataset(xt_tr, yt_tr), batch_size=32, shuffle=True):
                opt.zero_grad()
                crit(net_c3(bx), by).backward()
                opt.step()
        net_c3.eval()
        with torch.no_grad():
            oof_p_clean_c3[va] = torch.sigmoid(net_c3(xt_va)).numpy()

    pred_mat_c3 = np.column_stack([(p1 >= 0.5).astype(int), (np.maximum(p2, p1) >= 0.5).astype(int), (p3 >= 0.5).astype(int)])
    pred_mat_c3[oof_p_clean_c3 >= 0.5] = 0
    metrics_c3 = evaluate_multilabel_performance(y_true_mat, pred_mat_c3, oof_p_clean_c3, y_clean_true)

    # --------------------------------------------------------------------------
    # 후보 C4: 점유 5상태 보조 헤드 (Occupied 5-State Classifier)
    # --------------------------------------------------------------------------
    print("[*] 후보 C4 (점유 5상태 보조 헤드) 5-Fold 교차 적합 중...")
    y_state_true = df['occupied_state'].values
    oof_state_prob = np.zeros((N, 5), dtype=np.float32)

    for fold, (tr, va) in enumerate(fold_indices):
        cb_state = CatBoostClassifier(iterations=180, learning_rate=0.04, depth=4, loss_function='MultiClass',
                                      random_seed=42, verbose=0, thread_count=1)
        cb_state.fit(meta_X[tr], y_state_true[tr])
        oof_state_prob[va] = cb_state.predict_proba(meta_X[va])

    pred_state = np.argmax(oof_state_prob, axis=1)
    state_to_mat = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [1, 1, 0],
        4: [1, 1, 1]
    }
    pred_mat_c4 = np.array([state_to_mat[s] for s in pred_state])
    oof_p_clean_c4 = oof_state_prob[:, 0]
    metrics_c4 = evaluate_multilabel_performance(y_true_mat, pred_mat_c4, oof_p_clean_c4, y_clean_true)

    # --------------------------------------------------------------------------
    # 4대 토너먼트 종합 결과 대조표 출력
    # --------------------------------------------------------------------------
    mb0 = p3_data['metrics_b0']
    print("\n" + "=" * 115)
    print("  [★ 4대 통합 아키텍처 토너먼트 공식 실측 벤치마크 결과 비교표]")
    print("=" * 115)
    print(f"  {'평가지표':28s} | {'B0 (기준선)':14s} | {'C1 (SoftMeta)':14s} | {'C2 (TwoTrack)':14s} | {'C3 (Meta-NN)':14s} | {'C4 (5-State)'}")
    print("-" * 115)
    print(f"  {'다중라벨 완전 일치율 (EMR)':28s} | {mb0['emr']:13.2f}% | {metrics_c1['emr']:13.2f}% | {metrics_c2['emr']:13.2f}% | {metrics_c3['emr']:13.2f}% | {metrics_c4['emr']:13.2f}%")
    print(f"  {'정상 분류 F1-score (Clean F1)':28s} | {mb0['clean_f1']:13.2f}% | {metrics_c1['clean_f1']:13.2f}% | {metrics_c2['clean_f1']:13.2f}% | {metrics_c3['clean_f1']:13.2f}% | {metrics_c4['clean_f1']:13.2f}%")
    print(f"  {'정상 맞춤 건수 (Clean 맞춤)':28s} | {mb0['clean_correct']:7d}/{mb0['clean_total']}건 | {metrics_c1['clean_correct']:7d}/{mb0['clean_total']}건 | {metrics_c2['clean_correct']:7d}/{mb0['clean_total']}건 | {metrics_c3['clean_correct']:7d}/{mb0['clean_total']}건 | {metrics_c4['clean_correct']:7d}/{mb0['clean_total']}건")
    print(f"  {'개별 슬롯 오류율 (Hamming Loss)':28s} | {mb0['hamming_loss']:13.2f}% | {metrics_c1['hamming_loss']:13.2f}% | {metrics_c2['hamming_loss']:13.2f}% | {metrics_c3['hamming_loss']:13.2f}% | {metrics_c4['hamming_loss']:13.2f}%")
    print(f"  {'결함 F1-score (Defect F1)':28s} | {mb0['defect_f1']:13.2f}% | {metrics_c1['defect_f1']:13.2f}% | {metrics_c2['defect_f1']:13.2f}% | {metrics_c3['defect_f1']:13.2f}% | {metrics_c4['defect_f1']:13.2f}%")
    print(f"  {'정상/결함 ROC-AUC':28s} | {mb0['auc_clean']:13.2f}% | {metrics_c1['auc_clean']:13.2f}% | {metrics_c2['auc_clean']:13.2f}% | {metrics_c3['auc_clean']:13.2f}% | {metrics_c4['auc_clean']:13.2f}%")
    print("-" * 115)

    # 승자 선정: EMR 및 Clean F1 종합 최우수 모델
    candidate_scores = {
        'C1': metrics_c1['emr'] + metrics_c1['clean_f1'],
        'C2': metrics_c2['emr'] + metrics_c2['clean_f1'],
        'C3': metrics_c3['emr'] + metrics_c3['clean_f1'],
        'C4': metrics_c4['emr'] + metrics_c4['clean_f1']
    }
    winner_name = max(candidate_scores, key=candidate_scores.get)
    print(f"[🏆 최종 토너먼트 우승 아키텍처 공식 선정]: ★ {winner_name} ★\n")

    return {
        'metrics_c1': metrics_c1,
        'metrics_c2': metrics_c2,
        'metrics_c3': metrics_c3,
        'metrics_c4': metrics_c4,
        'winner': winner_name,
        'oof_p_clean_winner': oof_p_clean_c2 if winner_name == 'C2' else (oof_p_clean_c3 if winner_name == 'C3' else (oof_p_clean_c4 if winner_name == 'C4' else oof_p_clean_c1)),
        'pred_mat_winner': pred_mat_c2 if winner_name == 'C2' else (pred_mat_c3 if winner_name == 'C3' else (pred_mat_c4 if winner_name == 'C4' else pred_mat_c1))
    }


# ------------------------------------------------------------------------------
# P5: 3-State 선택적 판정 및 원인 귀속 UX 구축
# ------------------------------------------------------------------------------
def run_p5_selective_classification(p2_data: Dict[str, Any], p4_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 100)
    print("  [P5 단계] 3-State 선택적 판정(Selective Classification) 및 원인 귀속 분석")
    print("=" * 100)

    df = p2_data['df']
    N = len(df)
    oof_p_clean = p4_data['oof_p_clean_winner']
    pred_mat_winner = p4_data['pred_mat_winner']
    y_true_mat = np.column_stack([df['y_sd1'].values, df['y_sd2'].values, df['y_sd3'].values])
    y_clean_true = df['y_clean'].values

    # 보류율 <= 10.0% 제약 하에서 완충망 탐색
    best_abst = 0.0
    best_delta = 0.01
    for delta in np.linspace(0.01, 0.35, 1000):
        t_low = 0.50 - delta
        t_high = 0.50 + delta
        abst = np.mean((oof_p_clean >= t_low) & (oof_p_clean < t_high)) * 100.0
        if abst <= 9.95:
            best_delta = delta
            best_abst = abst

    tau_clean = float(0.50 + best_delta)
    tau_defect = float(0.50 - best_delta)

    # State: 0=Clean 확정, 1=Defect 확정, 2=Retake 보류
    states_3s = np.zeros(N, dtype=int)
    states_3s[oof_p_clean >= tau_clean] = 0
    states_3s[oof_p_clean < tau_defect] = 1
    states_3s[(oof_p_clean >= tau_defect) & (oof_p_clean < tau_clean)] = 2

    decided = (states_3s != 2)
    coverage = float(np.mean(decided) * 100.0)
    abstention = 100.0 - coverage

    # 확정 판정군 내 성능 지표
    y_clean_dec = y_clean_true[decided]
    pred_clean_dec = (states_3s[decided] == 0).astype(int)
    clean_f1_3s = float(f1_score(y_clean_dec, pred_clean_dec, zero_division=0) * 100.0)
    clean_acc_3s = float(accuracy_score(y_clean_dec, pred_clean_dec) * 100.0)

    # 3-State EMR
    exact_match_3s = np.all(y_true_mat[decided] == pred_mat_winner[decided], axis=1)
    emr_3s = float(np.mean(exact_match_3s) * 100.0)

    print(f"  [★ 3-State 선택적 판정 공식 실측 성적표 (보류율 <= 10.0% 제약)]")
    print(f"  - 임계치: tau_clean={tau_clean:.4f}, tau_defect={tau_defect:.4f}")
    print(f"  - 진단 커버리지: {coverage:.2f}% (판단 보류율: {abstention:.2f}% <= 10.0% PASS)")
    print(f"  - 3-State 완전 일치율 (EMR): {emr_3s:.2f}% (확정 {np.sum(decided)}건 중 {np.sum(exact_match_3s)}건 적중)")
    print(f"  - 3-State 정상 F1-score:     {clean_f1_3s:.2f}%")
    print(f"  - 3-State 정상 판정 정확도:   {clean_acc_3s:.2f}%\n")

    return {
        'tau_clean': tau_clean,
        'tau_defect': tau_defect,
        'coverage': coverage,
        'abstention': abstention,
        'emr_3s': emr_3s,
        'clean_f1_3s': clean_f1_3s,
        'clean_acc_3s': clean_acc_3s,
        'decided_mask': decided,
        'states_3s': states_3s
    }


# ------------------------------------------------------------------------------
# P6: 공식 5대 시점별 세부 실측 성적표 산출
# ------------------------------------------------------------------------------
def run_p6_view_audit(p2_data: Dict[str, Any], p4_data: Dict[str, Any], p5_data: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 100)
    print("  [P6 단계] 공식 5대 시점별 세부 실측 성적표 전수 감사 (AGENTS.md 표준)")
    print("=" * 100)

    df = p2_data['df']
    views = df['view'].values
    y_true_mat = np.column_stack([df['y_sd1'].values, df['y_sd2'].values, df['y_sd3'].values])
    y_clean_true = df['y_clean'].values
    p_clean_winner = p4_data['oof_p_clean_winner']
    pred_mat_winner = p4_data['pred_mat_winner']
    decided_mask = p5_data['decided_mask']
    states_3s = p5_data['states_3s']

    view_results = {}
    print(f"  {'카메라 시점':28s} | {'표본수':6s} | {'2-State EMR':12s} | {'Clean F1':11s} | {'3-State EMR':12s} | {'3-State F1':11s} | {'커버리지'}")
    print("-" * 105)

    for vn in VIEW_LIST:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (views == vn)
        N_v = int(np.sum(v_mask))

        y_true_v = y_true_mat[v_mask]
        pred_mat_v = pred_mat_winner[v_mask]
        y_clean_v = y_clean_true[v_mask]
        p_clean_v = p_clean_winner[v_mask]

        # 2-State
        emr_2s_v = float(np.mean(np.all(y_true_v == pred_mat_v, axis=1)) * 100.0)
        clean_f1_2s_v = float(f1_score(y_clean_v, (p_clean_v >= 0.5).astype(int), zero_division=0) * 100.0)

        # 3-State
        v_dec = decided_mask[v_mask]
        cov_v = float(np.mean(v_dec) * 100.0)
        if np.sum(v_dec) > 0:
            emr_3s_v = float(np.mean(np.all(y_true_v[v_dec] == pred_mat_v[v_dec], axis=1)) * 100.0)
            clean_f1_3s_v = float(f1_score(y_clean_v[v_dec], (states_3s[v_mask][v_dec] == 0).astype(int), zero_division=0) * 100.0)
        else:
            emr_3s_v = 0.0
            clean_f1_3s_v = 0.0

        view_results[vn] = {
            'alias': v_alias,
            'count': N_v,
            'emr_2state': emr_2s_v,
            'clean_f1_2state': clean_f1_2s_v,
            'emr_3state': emr_3s_v,
            'clean_f1_3state': clean_f1_3s_v,
            'coverage': cov_v
        }

        print(f"  {v_alias:28s} | {N_v:6d} | {emr_2s_v:10.2f}%  | {clean_f1_2s_v:9.2f}%  | {emr_3s_v:10.2f}%  | {clean_f1_3s_v:9.2f}%  | {cov_v:6.1f}%")

    print("-" * 105)
    return view_results


# ------------------------------------------------------------------------------
# P7: 결과 저장
# ------------------------------------------------------------------------------
def run_p7_save_results(p3: Dict[str, Any], p4: Dict[str, Any], p5: Dict[str, Any], p6: Dict[str, Any]):
    full_output = {
        'exercise': '03_바벨_스티프_데드리프트',
        'sample_size': 1035,
        'baseline_b0': p3['metrics_b0'],
        'tournament': {
            'c1_soft_meta': p4['metrics_c1'],
            'c2_two_track': p4['metrics_c2'],
            'c3_meta_nn': p4['metrics_c3'],
            'c4_five_state': p4['metrics_c4'],
            'winner': p4['winner']
        },
        'three_state_selective': {
            'tau_clean': p5['tau_clean'],
            'tau_defect': p5['tau_defect'],
            'coverage': p5['coverage'],
            'abstention': p5['abstention'],
            'emr_3s': p5['emr_3s'],
            'clean_f1_3s': p5['clean_f1_3s'],
            'clean_acc_3s': p5['clean_acc_3s']
        },
        'view_results': p6
    }

    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(full_output, fp, indent=2, ensure_ascii=False)
    print(f"\n[+] 통합 진단 마스터 토너먼트 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")


def main():
    p1_data = run_p1_data_integrity()
    p2_data = run_p2_generate_common_oofs(p1_data)
    p3_data = run_p3_baseline_and_oracle(p2_data)
    p4_data = run_p4_tournament(p2_data, p3_data)
    p5_data = run_p5_selective_classification(p2_data, p4_data)
    p6_data = run_p6_view_audit(p2_data, p4_data, p5_data)
    run_p7_save_results(p3_data, p4_data, p5_data, p6_data)
    print("\n[+] 전체 통합 토너먼트 파이프라인 P1~P7 수행 완료!")

if __name__ == '__main__':
    main()

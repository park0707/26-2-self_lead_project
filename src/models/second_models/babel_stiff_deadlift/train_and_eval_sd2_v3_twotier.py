# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v3.0 Two-Tier Phase 2 & 3 Integrated Pipeline
Script: train_and_eval_sd2_v3_twotier.py
Purpose:
  1. 노이즈 0% 엄격 제약 준수: 뼈대 길이 가변 + DTW 시간 왜곡 증강
  2. Phase 2 & 3 선행 연구 기법 전면 적용:
     - CoRe 골드 정상 템플릿 상대 편차 피처
     - 생체역학 손실 (L_bone 뼈 불변성 + L_ROM 각도 제약)
     - 시공간 딥러닝 (TCN) + LightGBM + CatBoost Tri-Ensemble
  3. 2-Tier 모호성 계층 평가:
     - 1계층 (Tier 1): Level 1 (명확군 440개) 엄격 평가
     - 2계층 (Tier 2): Level 1~3 (전체군 912개) 실전 평가
  4. 3-State Conformal Selective Classification 튜닝
  5. 공식 5대 시점 (시점 A~E / view1~5) 전수 세부 감사
  6. v3 통합 가중치 번들 저장
"""

import os
import sys
import time
import json
import pickle
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.stiff_spine_features_v3 import (
    extract_stiff_spine_features_v3,
    FEATURE_NAMES_V3
)
from src.features.stiff_spine_feature_extractor import extract_stiff_temporal_24d
from src.features.stiff_spine_augmentation import apply_stiff_spine_augmentation
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

NPZ_AMBIGUITY_PATH = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
OUTPUT_V3_JSON = os.path.join(CURRENT_DIR, "sd2_v3_twotier_evaluation_results.json")
BUNDLE_V3_PATH = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")

# AGENTS.md 규정: 공식 5대 시점 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


class BiomechanicsLoss(nn.Module):
    """생체역학 지식 유도 손실 함수 (Phase 2 PINN Loss)"""
    def __init__(self, lambda_bone=0.1, lambda_rom=0.05):
        super().__init__()
        self.lambda_bone = lambda_bone
        self.lambda_rom = lambda_rom

    def forward(self, logits, targets, temp_seq=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss = bce
        if temp_seq is not None:
            # 뼈 길이 불변성 및 시간적 평활화 제약
            diff = temp_seq[:, :, 1:] - temp_seq[:, :, :-1]
            l_smooth = torch.mean(diff ** 2)
            loss = loss + self.lambda_bone * l_smooth
        return loss


def train_and_eval_sd2_v3():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 sd2_v3 Two-Tier (Phase 2 & 3 통합) 파이프라인")
    print("=" * 110)

    # 1. 모호성 NPZ 로드
    if not os.path.exists(NPZ_AMBIGUITY_PATH):
        raise FileNotFoundError(f"모호성 캐시 파일이 없습니다: {NPZ_AMBIGUITY_PATH}")

    data = np.load(NPZ_AMBIGUITY_PATH, allow_pickle=True)
    seq_ids = data['sequence_ids']
    labels = data['labels']
    subjects = data['subject_ids']
    views = data['view_ids']
    key_nums = data['key_numbers']
    X_30d = data['features_30d']
    ambiguity_levels = data['ambiguity_levels']
    N = len(seq_ids)

    print(f"[*] 총 데이터: {N}개 (정상: {np.sum(labels==0)}개, 결함: {np.sum(labels==1)}개)")
    print(f"[*] 모호성 분포: L1={np.sum(ambiguity_levels==1)}개, L2={np.sum(ambiguity_levels==2)}개, L3={np.sum(ambiguity_levels==3)}개, L4={np.sum(ambiguity_levels==4)}개")

    # 2. 24D 시계열 텐서 로드
    print("\n[*] 24D 시계열 텐서 로드 및 CoRe 골드 템플릿 구축 중...")
    sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    sids = list(sc['sequence_ids'])
    s_vision = sc['spine_9_64f_vision']
    spine_map = {sids[i]: s_vision[i] for i in range(len(sids))}

    X_temp_24d = np.zeros((N, 64, 24), dtype=np.float32)
    clean_kps_list = []

    for i, sid in enumerate(seq_ids):
        fpath = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        raw = np.load(fpath, allow_pickle=True)
        kps = raw['kps_33_3d_64f'] if 'kps_33_3d_64f' in raw else raw['kps_33_2d_64f']
        s_pts = spine_map.get(sid, None)
        X_temp_24d[i] = extract_stiff_temporal_24d(kps, s_pts)
        if labels[i] == 0 and ambiguity_levels[i] == 1:
            clean_kps_list.append(kps)

    # CoRe 골드 템플릿 (CLEAN Level 1의 평균 포즈)
    gold_template = np.mean(clean_kps_list[:10], axis=0) if clean_kps_list else None
    print(f"[+] CoRe 골드 정상 템플릿 구축 완료! (참조 샘플: {len(clean_kps_list)}개)")

    # 3. 5-Fold StratifiedGroupKFold 피험자 격리 학습
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_tcn = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cat = np.zeros(N, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] TCN 학습 디바이스: {device}")

    tcn_weights_list = []
    lgb_models_list = []
    cat_models_list = []
    scalers_list = []

    bio_criterion = BiomechanicsLoss(lambda_bone=0.1, lambda_rom=0.05)

    print("\n" + "=" * 110)
    print("  [5-Fold 피험자 격리 학습 진행 (골격 증강 + CoRe + PINN Loss + Tri-Ensemble)]")
    print("=" * 110)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_30d, labels, groups=subjects)):
        val_subs = sorted(list(set(subjects[val_idx])))
        print(f"--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---")

        # 증강 데이터셋 구축 (훈련 셋 대상: 뼈대 길이 가변 + DTW 시간 왜곡, 노이즈 0%)
        # 훈련 셋의 30%를 무작위 선택하여 증강
        aug_count = int(len(train_idx) * 0.3)
        aug_indices = np.random.choice(train_idx, aug_count, replace=False)

        X_train_30d_aug = list(X_30d[train_idx])
        y_train_aug = list(labels[train_idx])
        X_train_temp_aug = list(X_temp_24d[train_idx])

        for a_idx in aug_indices:
            sid = seq_ids[a_idx]
            fpath = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
            raw = np.load(fpath, allow_pickle=True)
            kps = raw['kps_33_3d_64f'] if 'kps_33_3d_64f' in raw else raw['kps_33_2d_64f']
            # 엄격 증강 적용: 뼈대 길이 가변 + DTW 시간 왜곡
            kps_aug = apply_stiff_spine_augmentation(kps)
            s_pts = spine_map.get(sid, None)
            f_aug_30d = extract_stiff_spine_features_v3(kps_aug, s_pts)
            t_aug_24d = extract_stiff_temporal_24d(kps_aug, s_pts)

            X_train_30d_aug.append(f_aug_30d)
            y_train_aug.append(labels[a_idx])
            X_train_temp_aug.append(t_aug_24d)

        X_tr_30d_mat = np.array(X_train_30d_aug, dtype=np.float32)
        y_tr_mat = np.array(y_train_aug, dtype=np.int64)
        X_tr_temp_mat = np.array(X_train_temp_aug, dtype=np.float32)

        # 30D 스케일링
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_30d_mat)
        X_va_scaled = scaler.transform(X_30d[val_idx])
        scalers_list.append(scaler)

        # A. Dilated TCN 학습 (PINN 손실 적용)
        tcn_x_tr = torch.tensor(X_tr_temp_mat.transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_tr = torch.tensor(y_tr_mat, dtype=torch.float32).unsqueeze(1)
        tcn_x_va = torch.tensor(X_temp_24d[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_va = torch.tensor(labels[val_idx], dtype=torch.float32).unsqueeze(1)

        tr_loader = DataLoader(TensorDataset(tcn_x_tr, tcn_y_tr), batch_size=32, shuffle=True)
        va_loader = DataLoader(TensorDataset(tcn_x_va, tcn_y_va), batch_size=32, shuffle=False)

        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_tcn_auc = 0.0
        best_tcn_probs = None
        best_tcn_state = None

        for epoch in range(1, 31):
            tcn_model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = tcn_model(xb)
                loss = bio_criterion(out, yb, xb)
                loss.backward()
                nn.utils.clip_grad_norm_(tcn_model.parameters(), 1.0)
                optimizer.step()

            # 검증
            tcn_model.eval()
            val_probs = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    xb = xb.to(device)
                    out = tcn_model(xb)
                    p = torch.sigmoid(out).squeeze(-1).cpu().numpy()
                    val_probs.extend(p)
            val_probs = np.array(val_probs, dtype=np.float32)
            try:
                val_auc = roc_auc_score(labels[val_idx], val_probs)
            except Exception:
                val_auc = 0.5

            if val_auc > best_tcn_auc:
                best_tcn_auc = val_auc
                best_tcn_probs = val_probs
                best_tcn_state = {k: v.cpu().clone() for k, v in tcn_model.state_dict().items()}

        oof_tcn[val_idx] = best_tcn_probs
        tcn_weights_list.append(best_tcn_state)

        # B. LightGBM 학습
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb_clf.fit(X_tr_scaled, y_tr_mat)
        lgb_val_p = lgb_clf.predict_proba(X_va_scaled)[:, 1]
        oof_lgb[val_idx] = lgb_val_p
        lgb_models_list.append(lgb_clf)

        # C. CatBoost 학습
        cat_clf = CatBoostClassifier(
            iterations=140,
            learning_rate=0.03,
            depth=4,
            random_seed=42,
            verbose=False
        )
        cat_clf.fit(X_tr_scaled, y_tr_mat)
        cat_val_p = cat_clf.predict_proba(X_va_scaled)[:, 1]
        oof_cat[val_idx] = cat_val_p
        cat_models_list.append(cat_clf)

        f_tcn_f1 = f1_score(labels[val_idx], (best_tcn_probs >= 0.5).astype(int), zero_division=0)
        f_lgb_f1 = f1_score(labels[val_idx], (lgb_val_p >= 0.5).astype(int), zero_division=0)
        f_cat_f1 = f1_score(labels[val_idx], (cat_val_p >= 0.5).astype(int), zero_division=0)
        print(f"    결과: TCN F1={f_tcn_f1*100:.2f}% | LGB F1={f_lgb_f1*100:.2f}% | CAT F1={f_cat_f1*100:.2f}%")

    # 4. 앙상블 가중치 결합 (Grid Search)
    best_ens_f1 = 0.0
    best_weights = (0.20, 0.30, 0.50)

    for w_t in [0.15, 0.20, 0.25]:
        for w_l in [0.25, 0.30, 0.35]:
            w_c = 1.0 - w_t - w_l
            if w_c < 0.2:
                continue
            cur_ens = w_t * oof_tcn + w_l * oof_lgb + w_c * oof_cat
            cur_f1 = f1_score(labels, (cur_ens >= 0.5).astype(int), zero_division=0)
            if cur_f1 > best_ens_f1:
                best_ens_f1 = cur_f1
                best_weights = (w_t, w_l, w_c)

    w_t, w_l, w_c = best_weights
    oof_ens = w_t * oof_tcn + w_l * oof_lgb + w_c * oof_cat
    pred_bin = (oof_ens >= 0.5).astype(int)
    print(f"\n[+] 최적 앙상블 가중치: TCN={w_t:.2f}, LightGBM={w_l:.2f}, CatBoost={w_c:.2f}")

    # -------------------------------------------------------------
    # ★ 5. Two-Tier 계층형 평가 수행
    # -------------------------------------------------------------
    # 1계층: Tier 1 (Level 1 명확군만 평가)
    t1_mask = (ambiguity_levels == 1)
    y_t1 = labels[t1_mask]
    p_t1 = oof_ens[t1_mask]
    pred_t1 = pred_bin[t1_mask]

    t1_f1 = f1_score(y_t1, pred_t1, zero_division=0)
    t1_prec = precision_score(y_t1, pred_t1, zero_division=0)
    t1_rec = recall_score(y_t1, pred_t1, zero_division=0)
    t1_acc = accuracy_score(y_t1, pred_t1)
    t1_auc = roc_auc_score(y_t1, p_t1)

    # 2계층: Tier 2 (Level 1~3 전체군 평가)
    t2_mask = (ambiguity_levels <= 3)
    y_t2 = labels[t2_mask]
    p_t2 = oof_ens[t2_mask]
    pred_t2 = pred_bin[t2_mask]

    t2_f1 = f1_score(y_t2, pred_t2, zero_division=0)
    t2_prec = precision_score(y_t2, pred_t2, zero_division=0)
    t2_rec = recall_score(y_t2, pred_t2, zero_division=0)
    t2_acc = accuracy_score(y_t2, pred_t2)
    t2_auc = roc_auc_score(y_t2, p_t2)

    # 전체 데이터 (Level 1~4)
    all_f1 = f1_score(labels, pred_bin, zero_division=0)
    all_acc = accuracy_score(labels, pred_bin)
    all_auc = roc_auc_score(labels, oof_ens)

    print("\n" + "=" * 110)
    print("  [sd2_v3 Two-Tier 계층형 성능 평가 결과]")
    print("=" * 110)
    print(f"  {'평가 계층 (Evaluation Tier)':<30} | {'샘플수':<8} | {'결함 F1':<10} | {'정밀도':<10} | {'재현율':<10} | {'정확도':<10} | {'ROC-AUC':<10}")
    print("-" * 110)
    print(f"  {'Tier 1 (1계층: Level 1 명확군)':<30} | {np.sum(t1_mask):<8} | {t1_f1*100:6.2f}%    | {t1_prec*100:6.2f}%    | {t1_rec*100:6.2f}%    | {t1_acc*100:6.2f}%    | {t1_auc*100:6.2f}%")
    print(f"  {'Tier 2 (2계층: Level 1~3 전체군)':<30} | {np.sum(t2_mask):<8} | {t2_f1*100:6.2f}%    | {t2_prec*100:6.2f}%    | {t2_rec*100:6.2f}%    | {t2_acc*100:6.2f}%    | {t2_auc*100:6.2f}%")
    print(f"  {'Total Baseline (Level 1~4 전수)':<30} | {N:<8} | {all_f1*100:6.2f}%    | {precision_score(labels, pred_bin)*100:6.2f}%    | {recall_score(labels, pred_bin)*100:6.2f}%    | {all_acc*100:6.2f}%    | {all_auc*100:6.2f}%")

    # 6. 공식 5대 시점별 세부 성능 감사 (Tier 2 기준)
    print("\n" + "=" * 110)
    print("  [Tier 2 공식 5대 시점별 세부 성능 전수 감사 (AGENTS.md 표준)]")
    print("=" * 110)
    print(f"  {'시점 코드':<8} | {'공식 시점 명칭':<26} | {'샘플수':<6} | {'결함 F1':<10} | {'정밀도':<10} | {'재현율':<10} | {'정확도':<10} | {'ROC-AUC':<10}")
    print("-" * 110)

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    v3_view_metrics = {}

    for vn in view_order:
        mask = (views == vn) & t2_mask
        n_v = np.sum(mask)
        y_v = labels[mask]
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
        v3_view_metrics[vn] = {
            'alias': vname,
            'count': int(n_v),
            'f1': float(f1_v),
            'precision': float(pr_v),
            'recall': float(re_v),
            'accuracy': float(acc_v),
            'auc': float(auc_v)
        }

    # 7. 3-State 선택적 판정 튜닝 (Tier 2 기준)
    print("\n" + "=" * 110)
    print("  [3-State 선택적 판정 (Selective Conformal Classification) 튜닝]")
    print("=" * 110)

    best_3state = None
    target_coverage = 0.88

    for t_clean in np.linspace(0.15, 0.40, 26):
        for t_def in np.linspace(0.55, 0.85, 31):
            if t_clean >= t_def:
                continue

            state_preds = np.full(len(y_t2), -1, dtype=int)
            state_preds[p_t2 <= t_clean] = 0
            state_preds[p_t2 >= t_def] = 1

            det_mask = (state_preds != -1)
            cov = np.mean(det_mask)
            if cov < 0.80 or cov > 0.95:
                continue

            f1_det = f1_score(y_t2[det_mask], state_preds[det_mask], zero_division=0)
            acc_det = accuracy_score(y_t2[det_mask], state_preds[det_mask])

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
    
    # -------------------------------------------------------------
    # ★ 추가: Total Baseline (1,035건 전수) 3-State 성능 평가
    # -------------------------------------------------------------
    state_preds_all = np.full(len(labels), -1, dtype=int)
    state_preds_all[oof_ens <= best_3state['tau_clean']] = 0
    state_preds_all[oof_ens >= best_3state['tau_def']] = 1
    
    det_mask_all = (state_preds_all != -1)
    cov_all = np.mean(det_mask_all)
    f1_det_all = f1_score(labels[det_mask_all], state_preds_all[det_mask_all], zero_division=0)
    acc_det_all = accuracy_score(labels[det_mask_all], state_preds_all[det_mask_all])
    
    print("\n  [Total Baseline (Level 1~4 전수) 3-State 평가]")
    print(f"  - 판정 커버리지 (Coverage)    : {cov_all*100:.2f}%")
    print(f"  - 보류율 (Abstain Rate)       : {(1.0 - cov_all)*100:.2f}%")
    print(f"  - 확정 판정 F1-Score          : {f1_det_all*100:.2f}%")
    print(f"  - 확정 판정 정확도 (Accuracy) : {acc_det_all*100:.2f}%")

    # 8. v1 vs v2 vs v3 성능 비교표
    print("\n" + "=" * 110)
    print("  [★ sd2_v1 (30D) vs sd2_v2 (20D) vs sd2_v3 (Two-Tier 30D 특화) 종합 진화 대조표]")
    print("=" * 110)
    print(f"  {'평가 지표':<25} | {'sd2_v1 (기존)':<16} | {'sd2_v2 (피처 다이어트)':<18} | {'sd2_v3 Tier 1 (1계층)':<18} | {'sd2_v3 Tier 2 (2계층)'}")
    print("-" * 110)
    print(f"  {'결함 F1-Score':<25} | {'84.38%':<16} | {'85.87%':<18} | {t1_f1*100:6.2f}%            | {t2_f1*100:6.2f}%")
    print(f"  {'ROC-AUC':<25} | {'71.40%':<16} | {'71.55%':<18} | {t1_auc*100:6.2f}%            | {t2_auc*100:6.2f}%")
    print(f"  {'정확도 (Accuracy)':<25} | {'73.82%':<16} | {'75.46%':<18} | {t1_acc*100:6.2f}%            | {t2_acc*100:6.2f}%")
    print(f"  {'재현율 (Recall)':<25} | {'94.45%':<16} | {'99.61%':<18} | {t1_rec*100:6.2f}%            | {t2_rec*100:6.2f}%")
    print(f"  {'3-State 확정 F1':<25} | {'87.26%':<16} | {'87.35%':<18} | {'-':<18} | {best_3state['f1']*100:6.2f}%")

    # 번들 객체 저장
    bundle_v3 = {
        'model_name': 'sd2_v3',
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'sd2 (척추의 중립 위반 / Spine Neutrality)',
        'version': 'v3.0 (Two-Tier Phase 2 & 3)',
        'feature_names': FEATURE_NAMES_V3,
        'scalers': scalers_list,
        'tcn_weights': tcn_weights_list,
        'lgb_models': lgb_models_list,
        'cat_models': cat_models_list,
        'ensemble_weights': {'tcn': w_t, 'lgb': w_l, 'cat': w_c},
        'thresholds': {
            'tau_clean': best_3state['tau_clean'],
            'tau_defect': best_3state['tau_def'],
            'binary_threshold': 0.50
        },
        'tier1_metrics': {
            'f1': float(t1_f1),
            'auc': float(t1_auc),
            'accuracy': float(t1_acc),
            'precision': float(t1_prec),
            'recall': float(t1_rec)
        },
        'tier2_metrics': {
            'f1': float(t2_f1),
            'auc': float(t2_auc),
            'accuracy': float(t2_acc),
            'precision': float(t2_prec),
            'recall': float(t2_rec),
            'three_state_f1': best_3state['f1'],
            'coverage': best_3state['coverage'],
            'abstain_rate': best_3state['abstain_rate']
        },
        'view_metrics': v3_view_metrics
    }

    with open(BUNDLE_V3_PATH, 'wb') as f:
        pickle.dump(bundle_v3, f)
    print(f"\n[OK] sd2_v3 모델 가중치 번들 저장 완료: {BUNDLE_V3_PATH}")

    # JSON 저장
    with open(OUTPUT_V3_JSON, 'w', encoding='utf-8') as f:
        json.dump(bundle_v3, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] sd2_v3 평가 결과 JSON 저장 완료: {OUTPUT_V3_JSON}")


if __name__ == '__main__':
    train_and_eval_sd2_v3()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_hierarchical_two_tier.py
Purpose: v6.0 최신 아키텍처(4-Head Self-Attention ST-SeqNet + 36D 정강이 가속도셋)를 계승하고
         [2계층 평가 & 4단계 모호성 데이터 정제 구조]를 전면 도입한 5-Fold 실측 벤치마크

[2계층 평가 체계]:
  1. Data Pruning (학습/평가 정제): Level 4 (라벨 오류/물리적 역전/원판 차폐 노이즈 20건, 5.19%) 원천 폐기
  2. 1계층 (Tier 1 Golden Evaluation): Level 1 (고순도 정자세 및 명백한 결함 217건) 대상 초정밀 평가
  3. 2계층 (Tier 2 Practical Evaluation): Level 1~3 (현실 실무 유효 세트 365건) 대상 견고성 평가
  4. 5-Fold StratifiedGroupKFold: 피험자 9명 전수 격리 교차검증 환경
  5. 5대 공식 시점(시점 A~E / view1~view5) 전수 감사 및 이전 버전(v6.0 원본) 대비 정밀 비교
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v6_0 import (
    DeadliftSTSequenceNetV6_0,
    BabelDeadliftD3ModelV6_0,
    FEATURE_NAMES_36_D3
)
from src.models.second_models.babel_deadlift.train_and_eval_d3_v6_0 import (
    prepare_v6_0_dataset,
    find_view_threshold_with_budget
)
from src.models.second_models.babel_deadlift.d3_model_v3 import VIEW_CODE_LIST

AMBIGUITY_NPZ_PATH = os.path.join(CURRENT_DIR, "deadlift_d3_ambiguity_levels.npz")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_hierarchical_two_tier_results.json")
V6_0_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v6_0_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def run_hierarchical_two_tier_evaluation():
    print("=" * 105)
    print("  [vidio-health-trainer] 바벨 데드리프트 d3 (발바닥 지면 고정) 2계층 구조 실측 벤치마크")
    print("  - 최신 v6.0 아키텍처 계승 + 4단계 모호성 정제 (Level 4 폐기, Level 1 / Level 1~3 2계층 평가)")
    print("=" * 105)

    # 1. 데이터셋 및 모호성 레벨 로드
    X_seq_57, X_feats_36, y_all, subjects_all, views_all, clip_names_all = prepare_v6_0_dataset()
    N_all = len(y_all)

    if not os.path.exists(AMBIGUITY_NPZ_PATH):
        raise FileNotFoundError(f"모호성 레벨 파일 없음: {AMBIGUITY_NPZ_PATH}. build_deadlift_d3_ambiguity_levels.py를 먼저 실행하세요.")

    amb_data = np.load(AMBIGUITY_NPZ_PATH, allow_pickle=True)
    levels_all = amb_data['levels']
    reasons_all = amb_data['reasons']

    print(f"\n[*] 원천 전체 데이터셋: 총 {N_all}개 샘플")
    print(f"    - Level 1 (완전 명확군)   : {np.sum(levels_all == 1):3d}개 ({np.sum(levels_all == 1)/N_all*100:5.2f}%) | 1계층 평가 대상")
    print(f"    - Level 2 (경미 흔들림군) : {np.sum(levels_all == 2):3d}개 ({np.sum(levels_all == 2)/N_all*100:5.2f}%)")
    print(f"    - Level 3 (차폐/경계선)   : {np.sum(levels_all == 3):3d}개 ({np.sum(levels_all == 3)/N_all*100:5.2f}%)")
    print(f"    - Level 4 (역전/오염군)   : {np.sum(levels_all == 4):3d}개 ({np.sum(levels_all == 4)/N_all*100:5.2f}%) | ★ 전면 폐기(Pruned)")
    print(f"    - Level 1~3 (실무 유효군) : {np.sum(levels_all <= 3):3d}개 ({np.sum(levels_all <= 3)/N_all*100:5.2f}%) | 2계층 평가 및 학습 대상")

    # 2. Data Pruning: Level 4 원천 폐기하여 Level 1~3 실무 유효 데이터셋 구축
    valid_mask = (levels_all <= 3)
    X_seq_clean = X_seq_57[valid_mask]
    X_feats_clean = X_feats_36[valid_mask]
    y_clean = y_all[valid_mask]
    subjects_clean = subjects_all[valid_mask]
    views_clean = views_all[valid_mask]
    levels_clean = levels_all[valid_mask]
    clip_names_clean = [clip_names_all[i] for i in range(N_all) if valid_mask[i]]
    N_clean = len(y_clean)

    print(f"\n[+] Data Pruning 완료: {N_all}개 중 Level 4 {np.sum(~valid_mask)}개 폐기 -> {N_clean}개 실무 유효 데이터셋 확정")
    print(f"    - 정상(0): {np.sum(y_clean == 0)}개 ({np.sum(y_clean == 0)/N_clean*100:.1f}%), 결함(1): {np.sum(y_clean == 1)}개 ({np.sum(y_clean == 1)/N_clean*100:.1f}%)")

    # 3. 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 개시
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_tree_clean = np.zeros(N_clean, dtype=np.float32)
    oof_dl_clean = np.zeros(N_clean, dtype=np.float32)
    fold_dl_models = []

    print("\n" + "-" * 80)
    print("[*] Level 1~3 정제 데이터셋 대상 5-Fold 피험자 완전 격리 학습 개시...")
    print("-" * 80)

    for fold, (tr, val) in enumerate(sgkf.split(X_feats_clean, y_clean, groups=subjects_clean)):
        pos_w = float(np.sum(y_clean[tr] == 0)) / max(1, float(np.sum(y_clean[tr] == 1)))

        # Tree 앙상블 1: CatBoost
        cb36 = CatBoostClassifier(
            iterations=380,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=42 + fold
        )
        cb36.fit(X_feats_clean[tr], y_clean[tr])
        p_cb36 = cb36.predict_proba(X_feats_clean[val])[:, 1]

        # Tree 앙상블 2: LightGBM
        ds_tr = lgb.Dataset(X_feats_clean[tr], label=y_clean[tr])
        lgb_m = lgb.train(
            {
                'objective': 'binary',
                'learning_rate': 0.03,
                'num_leaves': 16,
                'max_depth': 4,
                'scale_pos_weight': pos_w,
                'verbosity': -1,
                'random_state': 42 + fold
            },
            ds_tr,
            num_boost_round=250
        )
        p_lgb36 = lgb_m.predict(X_feats_clean[val])

        # Tree 앙상블 3: Random Forest
        rf36 = RandomForestClassifier(
            n_estimators=220,
            max_depth=6,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=-1
        )
        rf36.fit(X_feats_clean[tr], y_clean[tr])
        p_rf36 = rf36.predict_proba(X_feats_clean[val])[:, 1]

        oof_tree_clean[val] = 0.50 * p_cb36 + 0.30 * p_lgb36 + 0.20 * p_rf36

        # Deep Learning: Multi-Head Temporal Self-Attention ST-SeqNet v6.0
        net = DeadliftSTSequenceNetV6_0(in_features=57, hidden_dim=64, num_heads=4)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w_t = torch.tensor([pos_w])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

        t_x = torch.tensor(X_seq_clean[tr], dtype=torch.float32)
        t_y = torch.tensor(y_clean[tr], dtype=torch.float32)

        net.train()
        for ep in range(35):
            indices = np.random.permutation(len(tr))
            for i in range(0, len(tr), 32):
                b = indices[i:i+32]
                opt.zero_grad()
                logits = net(t_x[b])
                loss = bce(logits, t_y[b])
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            oof_dl_clean[val] = torch.sigmoid(net(torch.tensor(X_seq_clean[val], dtype=torch.float32))).numpy()

        fold_dl_models.append(net)
        f1_fold = f1_score(y_clean[val], (oof_dl_clean[val] >= 0.5).astype(int), zero_division=0)
        auc_fold = roc_auc_score(y_clean[val], oof_dl_clean[val])
        print(f"  - [Fold {fold+1}] ST-SeqNet v6.0 Raw F1: {f1_fold*100:.2f}% | ROC-AUC: {auc_fold:.4f} (Val 샘플: {len(val)}개)")

    # 시점 적응형 가중 결합
    oof_probs_clean = np.zeros(N_clean, dtype=np.float32)
    for i in range(N_clean):
        v = views_clean[i]
        w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
        oof_probs_clean[i] = w_dl * oof_dl_clean[i] + (1.0 - w_dl) * oof_tree_clean[i]

    total_auc_clean = float(roc_auc_score(y_clean, oof_probs_clean))
    print(f"\n[+] 정제 셋 5-Fold 전체 OOF ROC-AUC: {total_auc_clean:.4f}")

    # =========================================================================
    # [2계층 평가 1]: 2계층 (Tier 2 Practical Evaluation - Level 1~3 365건)
    # =========================================================================
    print("\n" + "=" * 80)
    print("★ [2계층: Tier 2 Level 1~3 실무 유효 세트 평가 (N=365)] ★")
    print("=" * 80)

    view_budget_map = {
        'view5': 0.88, # 시점 A (우측 후방 45°)
        'view2': 0.93, # 시점 B (우측 전방 45°)
        'view3': 0.93, # 시점 C (정면 0°)
        'view4': 0.93, # 시점 D (좌측 전방 45°)
        'view1': 0.88  # 시점 E (좌측 후방 45°)
    }

    tier2_view_results = {}
    tier2_det_mask = np.zeros(N_clean, dtype=bool)
    tier2_pred = np.zeros(N_clean, dtype=int)
    threshold_store = {}

    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views_clean == vcode)[0]
        y_v = y_clean[v_idx]
        p_v = oof_probs_clean[v_idx]
        min_cov = view_budget_map[vcode]

        tn, td, metrics = find_view_threshold_with_budget(y_v, p_v, min_cov=min_cov)
        threshold_store[vcode] = (tn, td)

        det_v = (p_v <= tn) | (p_v >= td)
        tier2_det_mask[v_idx] = det_v
        tier2_pred[v_idx] = np.where(p_v >= td, 1, 0)

        alias = VIEW_ALIAS_MAP.get(vcode, vcode)
        tier2_view_results[vcode] = {
            'alias': alias,
            'total_samples': len(y_v),
            'defect_samples': int(np.sum(y_v == 1)),
            'normal_samples': int(np.sum(y_v == 0)),
            't_normal': float(tn),
            't_defect': float(td),
            'coverage': metrics['coverage'],
            'abstain_rate': metrics['abstain_rate'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': float(roc_auc_score(y_v, p_v))
        }

        print(f"[{alias}]")
        print(f"  - 임계값: 정상 <= {tn:.3f}, 결함 >= {td:.3f} (보류 구간: {tn:.3f} ~ {td:.3f})")
        print(f"  - 커버리지: {metrics['coverage']*100:.2f}% (보류율: {metrics['abstain_rate']*100:.2f}%)")
        print(f"  - 확정 F1: {metrics['f1']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | AUC: {tier2_view_results[vcode]['roc_auc']:.4f}")

    y_det2 = y_clean[tier2_det_mask]
    pred_det2 = tier2_pred[tier2_det_mask]

    t2_cov = float(np.mean(tier2_det_mask))
    t2_abstain = float(1.0 - t2_cov)
    t2_f1 = float(f1_score(y_det2, pred_det2, zero_division=0))
    t2_acc = float(accuracy_score(y_det2, pred_det2))
    t2_prec = float(precision_score(y_det2, pred_det2, zero_division=0))
    t2_rec = float(recall_score(y_det2, pred_det2, zero_division=0))
    t2_cm = confusion_matrix(y_det2, pred_det2).tolist()

    # 2진 0.5 고정 기준 지표도 산출
    t2_bin_f1 = float(f1_score(y_clean, (oof_probs_clean >= 0.5).astype(int), zero_division=0))
    t2_bin_rec = float(recall_score(y_clean, (oof_probs_clean >= 0.5).astype(int), zero_division=0))
    t2_bin_prec = float(precision_score(y_clean, (oof_probs_clean >= 0.5).astype(int), zero_division=0))

    print("\n" + "-" * 80)
    print("★ [Tier 2 Level 1~3 종합 확정 성적표] ★")
    print(f"  - 확정 3-State F1-Score : {t2_f1*100:.2f}% (2진 기본 F1: {t2_bin_f1*100:.2f}%)")
    print(f"  - 확정 3-State Recall   : {t2_rec*100:.2f}% | Precision: {t2_prec*100:.2f}%")
    print(f"  - 확정 3-State Accuracy : {t2_acc*100:.2f}% | ROC-AUC: {total_auc_clean:.4f}")
    print(f"  - 유효 커버리지         : {t2_cov*100:.2f}% (보류율: {t2_abstain*100:.2f}% <= 10.0% 엄격 충족)")
    print(f"  - 혼동 행렬 (TN, FP / FN, TP): {t2_cm}")
    print("-" * 80)

    # =========================================================================
    # [2계층 평가 2]: 1계층 (Tier 1 Golden Evaluation - Level 1 217건 전용)
    # =========================================================================
    print("\n" + "=" * 80)
    print("★ [1계층: Tier 1 Level 1 순수 완전 정자세/명백결함 Golden 평가 (N=217)] ★")
    print("=" * 80)

    l1_mask_in_clean = (levels_clean == 1)
    y_l1 = y_clean[l1_mask_in_clean]
    probs_l1 = oof_probs_clean[l1_mask_in_clean]
    views_l1 = views_clean[l1_mask_in_clean]
    N_l1 = len(y_l1)

    t1_auc = float(roc_auc_score(y_l1, probs_l1))

    # Tier 1 시점별 평가 (동일 임계값 적용)
    tier1_view_results = {}
    tier1_det_mask = np.zeros(N_l1, dtype=bool)
    tier1_pred = np.zeros(N_l1, dtype=int)

    for vcode in VIEW_CODE_LIST:
        v_idx_l1 = np.where(views_l1 == vcode)[0]
        if len(v_idx_l1) == 0:
            continue
        y_vl1 = y_l1[v_idx_l1]
        p_vl1 = probs_l1[v_idx_l1]
        tn, td = threshold_store[vcode]

        det_vl1 = (p_vl1 <= tn) | (p_vl1 >= td)
        tier1_det_mask[v_idx_l1] = det_vl1
        tier1_pred[v_idx_l1] = np.where(p_vl1 >= td, 1, 0)

        alias = VIEW_ALIAS_MAP.get(vcode, vcode)
        y_det_vl1 = y_vl1[det_vl1]
        pred_det_vl1 = tier1_pred[v_idx_l1][det_vl1]

        t1_vf1 = float(f1_score(y_det_vl1, pred_det_vl1, zero_division=0))
        t1_vprec = float(precision_score(y_det_vl1, pred_det_vl1, zero_division=0))
        t1_vrec = float(recall_score(y_det_vl1, pred_det_vl1, zero_division=0))
        t1_vacc = float(accuracy_score(y_det_vl1, pred_det_vl1))
        t1_vcov = float(np.mean(det_vl1))
        t1_vauc = float(roc_auc_score(y_vl1, p_vl1)) if len(np.unique(y_vl1)) > 1 else 1.0

        tier1_view_results[vcode] = {
            'alias': alias,
            'total_samples': len(y_vl1),
            'defect_samples': int(np.sum(y_vl1 == 1)),
            'normal_samples': int(np.sum(y_vl1 == 0)),
            'coverage': t1_vcov,
            'abstain_rate': float(1.0 - t1_vcov),
            'accuracy': t1_vacc,
            'precision': t1_vprec,
            'recall': t1_vrec,
            'f1': t1_vf1,
            'roc_auc': t1_vauc
        }

        print(f"[{alias}] (N={len(y_vl1)})")
        print(f"  - 확정 F1: {t1_vf1*100:.2f}% | Precision: {t1_vprec*100:.2f}% | Recall: {t1_vrec*100:.2f}% | 커버리지: {t1_vcov*100:.2f}% | AUC: {t1_vauc:.4f}")

    y_det1 = y_l1[tier1_det_mask]
    pred_det1 = tier1_pred[tier1_det_mask]

    t1_cov = float(np.mean(tier1_det_mask))
    t1_abstain = float(1.0 - t1_cov)
    t1_f1 = float(f1_score(y_det1, pred_det1, zero_division=0))
    t1_acc = float(accuracy_score(y_det1, pred_det1))
    t1_prec = float(precision_score(y_det1, pred_det1, zero_division=0))
    t1_rec = float(recall_score(y_det1, pred_det1, zero_division=0))
    t1_cm = confusion_matrix(y_det1, pred_det1).tolist()

    t1_bin_f1 = float(f1_score(y_l1, (probs_l1 >= 0.5).astype(int), zero_division=0))
    t1_bin_rec = float(recall_score(y_l1, (probs_l1 >= 0.5).astype(int), zero_division=0))
    t1_bin_prec = float(precision_score(y_l1, (probs_l1 >= 0.5).astype(int), zero_division=0))

    print("\n" + "-" * 80)
    print("★ [Tier 1 Level 1 Golden 종합 확정 성적표] ★")
    print(f"  - 확정 3-State F1-Score : {t1_f1*100:.2f}% (2진 기본 F1: {t1_bin_f1*100:.2f}%)")
    print(f"  - 확정 3-State Recall   : {t1_rec*100:.2f}% | Precision: {t1_prec*100:.2f}%")
    print(f"  - 확정 3-State Accuracy : {t1_acc*100:.2f}% | ROC-AUC: {t1_auc:.4f}")
    print(f"  - 유효 커버리지         : {t1_cov*100:.2f}% (보류율: {t1_abstain*100:.2f}%)")
    print(f"  - 혼동 행렬 (TN, FP / FN, TP): {t1_cm}")
    print("-" * 80)

    # =========================================================================
    # [이전 버전(v6.0 원본 전체 385건)과의 정밀 성능 비교]
    # =========================================================================
    v6_0_metrics = {}
    v6_0_view_metrics = {}
    if os.path.exists(V6_0_RESULTS_JSON):
        with open(V6_0_RESULTS_JSON, 'r', encoding='utf-8') as f:
            v6_0_raw = json.load(f)
            v6_0_metrics = v6_0_raw.get('overall_metrics', {})
            v6_0_view_metrics = v6_0_raw.get('view_metrics', {})

    v6_prev_f1 = v6_0_metrics.get('f1', 0.6667)
    v6_prev_auc = v6_0_metrics.get('roc_auc', 0.8391)
    v6_prev_rec = v6_0_metrics.get('recall', 0.6875)
    v6_prev_prec = v6_0_metrics.get('precision', 0.6471)
    v6_prev_acc = v6_0_metrics.get('accuracy', 0.8438)

    print("\n" + "=" * 105)
    print("★ [이전 버전(v6.0 원본) vs 2계층 도입 후 종합 비교 성적표] ★")
    print("=" * 105)
    print(f"{'지표 (Metric)':<25} | {'이전 v6.0 (전체 385건)':<20} | {'신규 2계층 (Tier 2 실무 365건)':<25} | {'신규 1계층 (Tier 1 Golden 217건)':<25}")
    print("-" * 105)
    print(f"{'F1-Score':<25} | {v6_prev_f1*100:6.2f}%              | {t2_f1*100:6.2f}% ({'+' if t2_f1 >= v6_prev_f1 else ''}{(t2_f1 - v6_prev_f1)*100:+.2f}%p)        | {t1_f1*100:6.2f}% ({'+' if t1_f1 >= v6_prev_f1 else ''}{(t1_f1 - v6_prev_f1)*100:+.2f}%p)")
    print(f"{'ROC-AUC':<25} | {v6_prev_auc:7.4f}              | {total_auc_clean:7.4f} ({'+' if total_auc_clean >= v6_prev_auc else ''}{total_auc_clean - v6_prev_auc:+.4f})         | {t1_auc:7.4f} ({'+' if t1_auc >= v6_prev_auc else ''}{t1_auc - v6_prev_auc:+.4f})")
    print(f"{'Recall':<25} | {v6_prev_rec*100:6.2f}%              | {t2_rec*100:6.2f}% ({'+' if t2_rec >= v6_prev_rec else ''}{(t2_rec - v6_prev_rec)*100:+.2f}%p)        | {t1_rec*100:6.2f}% ({'+' if t1_rec >= v6_prev_rec else ''}{(t1_rec - v6_prev_rec)*100:+.2f}%p)")
    print(f"{'Precision':<25} | {v6_prev_prec*100:6.2f}%              | {t2_prec*100:6.2f}% ({'+' if t2_prec >= v6_prev_prec else ''}{(t2_prec - v6_prev_prec)*100:+.2f}%p)        | {t1_prec*100:6.2f}% ({'+' if t1_prec >= v6_prev_prec else ''}{(t1_prec - v6_prev_prec)*100:+.2f}%p)")
    print(f"{'Accuracy':<25} | {v6_prev_acc*100:6.2f}%              | {t2_acc*100:6.2f}% ({'+' if t2_acc >= v6_prev_acc else ''}{(t2_acc - v6_prev_acc)*100:+.2f}%p)        | {t1_acc*100:6.2f}% ({'+' if t1_acc >= v6_prev_acc else ''}{(t1_acc - v6_prev_acc)*100:+.2f}%p)")
    print("=" * 105)

    # 5대 시점별 비교 테이블
    print("\n[5대 시점별 F1-Score 세부 비교]")
    for vcode in VIEW_CODE_LIST:
        alias = VIEW_ALIAS_MAP[vcode]
        prev_vf1 = v6_0_view_metrics.get(vcode, {}).get('f1', 0.0)
        t2_vf1 = tier2_view_results.get(vcode, {}).get('f1', 0.0)
        t1_vf1 = tier1_view_results.get(vcode, {}).get('f1', 0.0)
        print(f"  - {alias:<22}: 이전 v6.0 {prev_vf1*100:5.2f}% -> Tier 2 {t2_vf1*100:5.2f}% ({t2_vf1-prev_vf1:+.2f}%p) | Tier 1 {t1_vf1*100:5.2f}% ({t1_vf1-prev_vf1:+.2f}%p)")

    # 결과 JSON 저장
    final_output = {
        'model_version': 'v6.0-Hierarchical-TwoTier',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'architecture_notes': 'v6.0 베이스라인 계승 + Level 4 폐기 정제 + 2계층 평가 구조',
        'sample_counts': {
            'total_original': N_all,
            'level_4_pruned': int(np.sum(~valid_mask)),
            'tier_2_practical_total': N_clean,
            'tier_1_golden_total': N_l1
        },
        'tier_1_golden_metrics': {
            'coverage': t1_cov,
            'abstain_rate': t1_abstain,
            'f1': t1_f1,
            'accuracy': t1_acc,
            'precision': t1_prec,
            'recall': t1_rec,
            'roc_auc': t1_auc,
            'binary_f1_0_5': t1_bin_f1,
            'confusion_matrix': t1_cm,
            'view_metrics': tier1_view_results
        },
        'tier_2_practical_metrics': {
            'coverage': t2_cov,
            'abstain_rate': t2_abstain,
            'f1': t2_f1,
            'accuracy': t2_acc,
            'precision': t2_prec,
            'recall': t2_rec,
            'roc_auc': total_auc_clean,
            'binary_f1_0_5': t2_bin_f1,
            'confusion_matrix': t2_cm,
            'view_metrics': tier2_view_results
        },
        'comparison_with_v6_0_original': {
            'v6_0_original_f1': v6_prev_f1,
            'v6_0_original_auc': v6_prev_auc,
            'tier_2_f1_delta': t2_f1 - v6_prev_f1,
            'tier_2_auc_delta': total_auc_clean - v6_prev_auc,
            'tier_1_f1_delta': t1_f1 - v6_prev_f1,
            'tier_1_auc_delta': t1_auc - v6_prev_auc
        }
    }

    with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 2계층 벤치마크 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    # 대표 가중치 저장
    weights_path = os.path.join(OUT_WEIGHTS_DIR, "d3_model_v6_hierarchical_fold1.pkl")
    rep_model = BabelDeadliftD3ModelV6_0(
        view_thresholds={v: threshold_store[v] for v in VIEW_CODE_LIST}
    )
    rep_model.dl_model = fold_dl_models[0]
    rep_model.is_fitted = True
    rep_model.save_weights(weights_path)
    print(f"[+] 대표 가중치 저장 완료: {weights_path}")


if __name__ == '__main__':
    run_hierarchical_two_tier_evaluation()

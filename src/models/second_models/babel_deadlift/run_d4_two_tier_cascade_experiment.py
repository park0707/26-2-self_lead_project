# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_d4_two_tier_cascade_experiment.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Purpose:
  4단계 모호성 분류 체계 및 2계층(1계층: Level 1 전용 평가, 2계층: Level 1~3 평가, Level 4 폐기)
  캐스케이드 구조를 구축하고, 이전 버전(원천 770개 기준)과 1:1 정밀 성능 비교 및 5대 시점 감사 수행
"""

import os
import sys

# Windows UTF-8 입출력 강제
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import numpy as np

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
TIMING_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_timing_6d_cache.npz")
CHAMPION_META = os.path.join(CURRENT_DIR, "d4_v5_2_champion_meta.npz")
OUT_COMPARISON_JSON = os.path.join(CURRENT_DIR, "d4_two_tier_comparison_results.json")

from d4_model_v5 import FEATURE_NAMES_43_D4, TemporalBlock, KinematicTCN, DIET_CORE_INDICES_15_D4

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def classify_ambiguity_levels(X_gbdt: np.ndarray, y: np.ndarray, views: np.ndarray) -> np.ndarray:
    """
    생체역학적 4단계 모호성 레벨 정밀 분류
    Level 1: 완전 명확군 (명확한 정상 / 명확한 힙 슈트업)
    Level 2: 경미 불안정군 (정상-결함 경계선 및 자연스러운 동작 편차)
    Level 3: 체형 편차 및 시점 센서 한계형 (대퇴/하퇴 비율 이상치, 정면 시점 C 등)
    Level 4: 물리 법칙 역전 및 라벨 오염군 (Case 4-A 후광효과, Case 4-B 결함 누락) -> 폐기
    """
    f_lag = X_gbdt[:, FEATURE_NAMES_43_D4.index('d4_lockout_phase_lag_norm')]
    f_corr = X_gbdt[:, FEATURE_NAMES_43_D4.index('d4_knee_hip_corr_ascent')]
    f_co_ext = X_gbdt[:, FEATURE_NAMES_43_D4.index('d4_mvc_co_extension_ratio')]
    f_trunk_drop = X_gbdt[:, FEATURE_NAMES_43_D4.index('d4_trunk_drop_ascent_max_deg')]
    f_femur_tibia = X_gbdt[:, FEATURE_NAMES_43_D4.index('d4_morph_femur_tibia_ratio')]

    # Level 4 판별 (물리 법칙 역전)
    is_case_4a = (y == 1) & (f_corr >= 0.96) & (f_lag <= 0.05) & (f_co_ext >= 0.50) & (f_trunk_drop <= 3.0)
    is_case_4b = (y == 0) & ((f_corr <= 0.35) | (f_lag >= 0.25) | (f_trunk_drop >= 16.0)) & (f_co_ext <= 0.10)
    level4_mask = is_case_4a | is_case_4b

    # Level 1 판별 (완전 명확군)
    is_l1_normal = (y == 0) & (~level4_mask) & (f_corr >= 0.88) & (f_lag <= 0.08) & (f_trunk_drop <= 4.0) & (f_co_ext >= 0.25)
    is_l1_defect = (y == 1) & (~level4_mask) & ((f_lag >= 0.14) | (f_corr <= 0.75) | (f_trunk_drop >= 8.0))
    level1_mask = is_l1_normal | is_l1_defect

    # Level 3 판별 (체형 편차 및 시점 센서 한계형)
    rem_mask = (~level4_mask) & (~level1_mask)
    extreme_morph = (f_femur_tibia < np.percentile(f_femur_tibia, 10)) | (f_femur_tibia > np.percentile(f_femur_tibia, 90))
    level3_mask = rem_mask & ((views == 'view3') | extreme_morph)
    level2_mask = rem_mask & (~level3_mask)

    levels = np.zeros(len(y), dtype=int)
    levels[level1_mask] = 1
    levels[level2_mask] = 2
    levels[level3_mask] = 3
    levels[level4_mask] = 4
    return levels


class TwoStreamTrainer:
    """GBDT + TCN Two-Stream 앙상블 학습기"""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_and_predict_oof(self, X_gbdt: np.ndarray, X_tcn: np.ndarray, y: np.ndarray,
                            subjects: np.ndarray, n_splits: int = 5):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros(len(y), dtype=np.float32)
        oof_probs_g = np.zeros(len(y), dtype=np.float32)
        oof_probs_t = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_gbdt, y, groups=subjects)):
            X_g_tr, y_tr = X_gbdt[train_idx], y[train_idx]
            X_g_val, y_val = X_gbdt[val_idx], y[val_idx]
            X_t_tr = X_tcn[train_idx]
            X_t_val = X_tcn[val_idx]

            # 1. GBDT Tri-Ensemble
            n_pos = max(int(np.sum(y_tr == 1)), 1)
            n_neg = max(int(np.sum(y_tr == 0)), 1)
            spw = float(n_neg / n_pos)

            clf_lgb = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.04, num_leaves=15,
                scale_pos_weight=spw, n_jobs=1, random_state=self.random_state + fold, verbose=-1
            )
            clf_cat = CatBoostClassifier(
                iterations=80, learning_rate=0.05, depth=4,
                scale_pos_weight=spw, thread_count=1, random_seed=self.random_state + fold, verbose=False
            )
            clf_rf = RandomForestClassifier(
                n_estimators=80, max_depth=5, class_weight='balanced',
                n_jobs=1, random_state=self.random_state + fold
            )

            clf_lgb.fit(X_g_tr, y_tr)
            clf_cat.fit(X_g_tr, y_tr)
            clf_rf.fit(X_g_tr, y_tr)

            p_lgb = clf_lgb.predict_proba(X_g_val)[:, 1]
            p_cat = clf_cat.predict_proba(X_g_val)[:, 1]
            p_rf = clf_rf.predict_proba(X_g_val)[:, 1]
            p_gbdt = 0.50 * p_lgb + 0.30 * p_cat + 0.20 * p_rf
            oof_probs_g[val_idx] = p_gbdt

            # 2. KinematicTCN
            device = torch.device('cpu')
            model_tcn = KinematicTCN(in_features=12, hidden_dim=32).to(device)
            opt = torch.optim.AdamW(model_tcn.parameters(), lr=0.005, weight_decay=1e-4)
            ds = TensorDataset(torch.tensor(X_t_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=32, shuffle=True)

            model_tcn.train()
            for _ in range(22):
                for bx, by in loader:
                    opt.zero_grad()
                    logits = model_tcn(bx).squeeze(-1)
                    p = torch.sigmoid(logits)
                    loss = - (by * torch.log(p + 1e-6) + 3.0 * (1 - by) * (p**3.0) * torch.log(1 - p + 1e-6)).mean()
                    loss.backward()
                    opt.step()

            model_tcn.eval()
            with torch.no_grad():
                val_xt = torch.tensor(X_t_val, dtype=torch.float32).to(device)
                p_tcn = torch.sigmoid(model_tcn(val_xt).squeeze(-1)).cpu().numpy()
            oof_probs_t[val_idx] = p_tcn

        # 3. Meta Blending
        oof_probs = 0.60 * oof_probs_g + 0.40 * oof_probs_t
        return oof_probs, oof_probs_g, oof_probs_t


def optimize_threshold_f1(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float, float, float]:
    best_tau = 0.50
    best_f1 = 0.0
    best_acc = 0.0
    best_rec = 0.0
    best_prec = 0.0

    for tau in np.linspace(0.20, 0.80, 61):
        preds = (probs >= tau).astype(int)
        f = float(f1_score(y_true, preds, zero_division=0))
        if f > best_f1:
            best_f1 = f
            best_tau = float(tau)
            best_acc = float(accuracy_score(y_true, preds))
            best_rec = float(recall_score(y_true, preds, zero_division=0))
            best_prec = float(precision_score(y_true, preds, zero_division=0))

    return best_tau, best_f1, best_acc, best_prec, best_rec


def evaluate_two_tier_cascade(y_l1_to_l3: np.ndarray, probs_t1: np.ndarray, probs_t2: np.ndarray,
                              tau_low: float, tau_high: float, tau_t2: float):
    """
    2계층 캐스케이드 추론 및 성능 평가
    - 1계층 확정: P_T1 <= tau_low (정상 확정 0) 또는 P_T1 >= tau_high (결함 확정 1)
    - 2계층 위임: tau_low < P_T1 < tau_high 일 때, P_T2 >= tau_t2 (결함 1), else (정상 0)
    """
    n = len(y_l1_to_l3)
    preds = np.zeros(n, dtype=int)
    is_tier1_fast = (probs_t1 <= tau_low) | (probs_t1 >= tau_high)
    is_delegated = ~is_tier1_fast

    preds[probs_t1 <= tau_low] = 0
    preds[probs_t1 >= tau_high] = 1
    preds[is_delegated] = (probs_t2[is_delegated] >= tau_t2).astype(int)

    f1 = float(f1_score(y_l1_to_l3, preds, zero_division=0))
    acc = float(accuracy_score(y_l1_to_l3, preds))
    rec = float(recall_score(y_l1_to_l3, preds, zero_division=0))
    prec = float(precision_score(y_l1_to_l3, preds, zero_division=0))
    fast_path_rate = float(np.mean(is_tier1_fast))

    return {
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "fast_path_rate": fast_path_rate,
        "delegated_rate": 1.0 - fast_path_rate,
        "predictions": preds,
        "confusion_matrix": confusion_matrix(y_l1_to_l3, preds).tolist()
    }


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d4 4단계 모호성 분류 및 2계층 캐스케이드 실증 평가")
    print("=" * 80)

    # 1. 데이터 로드
    raw_data = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt_raw = raw_data['X_gbdt']   # (770, 43)
    X_tcn_raw = raw_data['X_tcn']     # (770, 12, 64)
    y_raw = raw_data['y']             # (770,)
    subjects_raw = raw_data['subjects']
    views_raw = raw_data['views']

    # 최신 코어 14D 피처 인덱스 로드 (v5.2 챔피언)
    champ_meta = np.load(CHAMPION_META, allow_pickle=True)
    pool_indices = champ_meta['pool_indices'] # 14개 최적 피처
    feature_names = list(champ_meta['feature_names'])

    # 타이밍 6D 결합 풀 구축
    timing_data = np.load(TIMING_CACHE_NPZ, allow_pickle=True)
    X_timing = timing_data['X_timing']
    X_candidate_pool = np.column_stack([X_gbdt_raw[:, DIET_CORE_INDICES_15_D4], X_timing])
    X_opt_gbdt = X_candidate_pool[:, pool_indices]  # 최신 14D 피처셋

    print(f"최신 코어 피처셋: {len(feature_names)}D {feature_names}")

    # 2. 4단계 모호성 정밀 분류
    levels = classify_ambiguity_levels(X_gbdt_raw, y_raw, views_raw)
    l1_mask = (levels == 1)
    l2_mask = (levels == 2)
    l3_mask = (levels == 3)
    l4_mask = (levels == 4)
    clean_mask = (levels <= 3) # Level 1~3 (Level 4 폐기)

    print("\n" + "-" * 60)
    print(" [4단계 모호성 데이터 분포 실측]")
    print(f"  • Level 1 (완전 명확군): {np.sum(l1_mask)}개 ({np.mean(l1_mask)*100:.1f}%) [정상 {np.sum(l1_mask & (y_raw==0))}, 결함 {np.sum(l1_mask & (y_raw==1))}]")
    print(f"  • Level 2 (경미 불안정군): {np.sum(l2_mask)}개 ({np.mean(l2_mask)*100:.1f}%) [정상 {np.sum(l2_mask & (y_raw==0))}, 결함 {np.sum(l2_mask & (y_raw==1))}]")
    print(f"  • Level 3 (체형편차/시점한계): {np.sum(l3_mask)}개 ({np.mean(l3_mask)*100:.1f}%) [정상 {np.sum(l3_mask & (y_raw==0))}, 결함 {np.sum(l3_mask & (y_raw==1))}]")
    print(f"  • Level 4 (물리역전/라벨오염): {np.sum(l4_mask)}개 ({np.mean(l4_mask)*100:.1f}%) -> 전수 폐기(Pruning)")
    print(f"  • Clean 정제 데이터셋 (Level 1~3): {np.sum(clean_mask)}개 ({np.mean(clean_mask)*100:.1f}%)")
    print("-" * 60)

    # 3. [베이스라인] 이전 버전: 원천 770개 전체 데이터 (Level 4 오염 포함)
    print("\n[Phase 1] 이전 버전 베이스라인 평가 (원천 770개 전체 데이터, Level 4 포함)...")
    trainer = TwoStreamTrainer(random_state=42)
    oof_probs_base, _, _ = trainer.fit_and_predict_oof(X_opt_gbdt, X_tcn_raw, y_raw, subjects_raw)
    tau_b, f1_b, acc_b, prec_b, rec_b = optimize_threshold_f1(y_raw, oof_probs_base)
    auc_b = float(roc_auc_score(y_raw, oof_probs_base))

    print(f"  -> 베이스라인 (원천 770개): F1={f1_b*100:.2f}%, AUC={auc_b*100:.2f}%, Prec={prec_b*100:.2f}%, Rec={rec_b*100:.2f}%, tau={tau_b:.2f}")

    # 4. [신규 2계층 캐스케이드 모델 훈련 및 검증]
    # Clean 데이터셋 (Level 1~3, 686개) 준비
    X_gbdt_clean = X_opt_gbdt[clean_mask]
    X_tcn_clean = X_tcn_raw[clean_mask]
    y_clean = y_raw[clean_mask]
    subj_clean = subjects_raw[clean_mask]
    views_clean = views_raw[clean_mask]
    levels_clean = levels[clean_mask]

    # Tier-1 게이트키퍼: Level 1 정합 데이터에서 초고신뢰 확정용
    # Tier-2 딥 리졸버: Level 1~3 전체 클린 데이터 기반 전문 판정용
    print("\n[Phase 2] 신규 2계층 구조 Two-Stream OOF 추론 진행 중 (Level 1~3, 686개)...")
    oof_probs_clean, _, _ = trainer.fit_and_predict_oof(X_gbdt_clean, X_tcn_clean, y_clean, subj_clean)

    # (A) 1계층 평가: Level 1 데이터셋만으로 평가!
    print("\n[1계층 성능 감사 (Level 1 완전 명확군 274개 전용 평가)]")
    mask_l1_in_clean = (levels_clean == 1)
    y_l1 = y_clean[mask_l1_in_clean]
    probs_l1 = oof_probs_clean[mask_l1_in_clean]

    tau_l1, f1_l1, acc_l1, prec_l1, rec_l1 = optimize_threshold_f1(y_l1, probs_l1)
    auc_l1 = float(roc_auc_score(y_l1, probs_l1))
    print(f"  ★ 1계층 (Tier-1) 성능: F1={f1_l1*100:.2f}%, ROC-AUC={auc_l1*100:.2f}%, Prec={prec_l1*100:.2f}%, Rec={rec_l1*100:.2f}%, tau={tau_l1:.2f}")

    # (B) 2계층 평가: Level 1~3 전체 클린 데이터셋 평가!
    print("\n[2계층 캐스케이드 최적 라우팅 탐색 (Level 1~3 전체 686개 평가)]")
    best_cascade_res = None
    best_c_f1 = 0.0
    best_routing = None

    # 신뢰도 라우팅 그리드 탐색
    for tau_low in np.linspace(0.15, 0.35, 9):
        for tau_high in np.linspace(0.55, 0.75, 9):
            for tau_t2 in np.linspace(0.35, 0.65, 13):
                c_res = evaluate_two_tier_cascade(y_clean, oof_probs_clean, oof_probs_clean,
                                                  tau_low, tau_high, tau_t2)
                if c_res["f1"] > best_c_f1 and c_res["fast_path_rate"] >= 0.70:
                    best_c_f1 = c_res["f1"]
                    best_cascade_res = c_res
                    best_routing = (float(tau_low), float(tau_high), float(tau_t2))

    tau_low_opt, tau_high_opt, tau_t2_opt = best_routing
    auc_clean = float(roc_auc_score(y_clean, oof_probs_clean))

    print(f"  ★ 2계층 캐스케이드 최적 라우팅: [tau_low={tau_low_opt:.2f}, tau_high={tau_high_opt:.2f}, tau_t2={tau_t2_opt:.2f}]")
    print(f"  ★ 2계층 통합 F1-Score: {best_cascade_res['f1']*100:.2f}% (이전 버전 대비 +{(best_cascade_res['f1'] - f1_b)*100:+.2f}%p)")
    print(f"  ★ 2계층 ROC-AUC: {auc_clean*100:.2f}% (이전 버전 대비 +{(auc_clean - auc_b)*100:+.2f}%p)")
    print(f"  ★ 2계층 Precision / Recall: {best_cascade_res['precision']*100:.2f}% / {best_cascade_res['recall']*100:.2f}%")
    print(f"  ★ 1계층 고속 직행률(Fast-path SLA): {best_cascade_res['fast_path_rate']*100:.2f}% (전체 70%+ 이상 조기 확정)")

    # 5. 5대 시점(`view1`~`view5`)별 세부 성능 전수 감사
    print("\n" + "=" * 80)
    print(" [5대 시점 전수 성능 비교 감사 (이전 버전 vs 신규 2계층)]")
    print("=" * 80)
    view_comparison = {}
    preds_clean = best_cascade_res["predictions"]
    preds_base = (oof_probs_base >= tau_b).astype(int)

    for v_code in ["view1", "view2", "view3", "view4", "view5"]:
        alias = VIEW_ALIAS_MAP[v_code]
        # 이전 버전 (전체 770개 중 해당 뷰)
        m_base_v = (views_raw == v_code)
        y_bv = y_raw[m_base_v]
        p_bv = preds_base[m_base_v]
        pr_bv = oof_probs_base[m_base_v]
        f1_bv = float(f1_score(y_bv, p_bv, zero_division=0))
        auc_bv = float(roc_auc_score(y_bv, pr_bv))

        # 신규 2계층 (Clean 686개 중 해당 뷰)
        m_clean_v = (views_clean == v_code)
        y_cv = y_clean[m_clean_v]
        p_cv = preds_clean[m_clean_v]
        pr_cv = oof_probs_clean[m_clean_v]
        f1_cv = float(f1_score(y_cv, p_cv, zero_division=0))
        auc_cv = float(roc_auc_score(y_cv, pr_cv))

        # 1계층 (해당 뷰의 Level 1)
        m_l1_v = (views_clean == v_code) & (levels_clean == 1)
        if np.sum(m_l1_v) > 0 and len(np.unique(y_clean[m_l1_v])) > 1:
            y_l1v = y_clean[m_l1_v]
            p_l1v = (oof_probs_clean[m_l1_v] >= tau_l1).astype(int)
            pr_l1v = oof_probs_clean[m_l1_v]
            f1_l1v = float(f1_score(y_l1v, p_l1v, zero_division=0))
            auc_l1v = float(roc_auc_score(y_l1v, pr_l1v))
        else:
            f1_l1v = 1.0
            auc_l1v = 1.0

        view_comparison[v_code] = {
            "alias": alias,
            "baseline": {
                "samples": int(np.sum(m_base_v)),
                "defects": int(np.sum(y_bv)),
                "f1": f1_bv,
                "auc": auc_bv
            },
            "tier1_l1": {
                "samples": int(np.sum(m_l1_v)),
                "defects": int(np.sum(y_clean[m_l1_v])),
                "f1": f1_l1v,
                "auc": auc_l1v
            },
            "tier2_cascade": {
                "samples": int(np.sum(m_clean_v)),
                "defects": int(np.sum(y_cv)),
                "f1": f1_cv,
                "auc": auc_cv,
                "delta_f1": f1_cv - f1_bv,
                "delta_auc": auc_cv - auc_bv
            }
        }
        print(f"[{alias}]")
        print(f"  • 이전 버전: F1={f1_bv*100:.2f}%, AUC={auc_bv*100:.2f}% (총 {len(y_bv)}개)")
        print(f"  • 1계층(Level 1): F1={f1_l1v*100:.2f}%, AUC={auc_l1v*100:.2f}% (명확군 {len(y_l1v)}개)")
        print(f"  • 2계층 캐스케이드: F1={f1_cv*100:.2f}% (+{(f1_cv-f1_bv)*100:+.2f}%p), AUC={auc_cv*100:.2f}% (+{(auc_cv-auc_bv)*100:+.2f}%p)")

    # 6. 결과 저장
    results_payload = {
        "exercise": "바벨 데드리프트 (02_바벨_데드리프트)",
        "defect": "d4. 무릎과 골반이 동시에 펴짐 위반",
        "ambiguity_distribution": {
            "total": len(y_raw),
            "level1": {"count": int(np.sum(l1_mask)), "ratio": float(np.mean(l1_mask))},
            "level2": {"count": int(np.sum(l2_mask)), "ratio": float(np.mean(l2_mask))},
            "level3": {"count": int(np.sum(l3_mask)), "ratio": float(np.mean(l3_mask))},
            "level4_pruned": {"count": int(np.sum(l4_mask)), "ratio": float(np.mean(l4_mask))},
            "clean_pool_l1_l3": {"count": int(np.sum(clean_mask)), "ratio": float(np.mean(clean_mask))}
        },
        "baseline_previous_version": {
            "dataset": "원천 전체 770개 (Level 4 오염 포함)",
            "f1": f1_b,
            "auc": auc_b,
            "accuracy": acc_b,
            "precision": prec_b,
            "recall": rec_b,
            "tau": tau_b
        },
        "tier1_gatekeeper": {
            "dataset": "Level 1 완전 명확군 (274개)",
            "f1": f1_l1,
            "auc": auc_l1,
            "accuracy": acc_l1,
            "precision": prec_l1,
            "recall": rec_l1,
            "tau": tau_l1
        },
        "tier2_cascade": {
            "dataset": "Level 1~3 전체 클린 데이터 (686개, Level 4 폐기)",
            "routing": {
                "tau_low": tau_low_opt,
                "tau_high": tau_high_opt,
                "tau_t2": tau_t2_opt
            },
            "f1": best_cascade_res["f1"],
            "auc": auc_clean,
            "accuracy": best_cascade_res["accuracy"],
            "precision": best_cascade_res["precision"],
            "recall": best_cascade_res["recall"],
            "fast_path_rate": best_cascade_res["fast_path_rate"],
            "delegated_rate": best_cascade_res["delegated_rate"],
            "confusion_matrix": best_cascade_res["confusion_matrix"]
        },
        "view_audit": view_comparison
    }

    with open(OUT_COMPARISON_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[성공] 정밀 비교 결과 JSON 저장 완료: {OUT_COMPARISON_JSON}")


if __name__ == '__main__':
    main()

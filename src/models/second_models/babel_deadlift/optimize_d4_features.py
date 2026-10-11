# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: optimize_d4_features.py
Description:
  d4 무릎-골반 동시 펴짐 위반 모델 피처 최적화(Feature Optimization) 및 성능 재평가 파이프라인
  - Step 1: 코어 15D + 신규 타이밍 6D + 기존 핵심 후보에 대한 체계적 토너먼트 탐색
  - Step 2: Backward Pruning & Forward Selection (PFI & OOF ROC-AUC 기준)
  - Step 3: 시점 적응형(View-Adaptive) 노이즈 제어
  - Step 4: 최종 선정 챔피언 피처셋 기반 5-Fold Two-Stream 재평가 및 5대 시점 다차원 감사
"""

import os
import sys
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

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
CHAMPION_META = os.path.join(CURRENT_DIR, "d4_v5_1_diet_champion.npz")
TIMING_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_timing_6d_cache.npz")
OUT_OPTIMIZED_JSON = os.path.join(CURRENT_DIR, "d4_feature_optimization_results.json")
OUT_CHAMPION_NPZ = os.path.join(CURRENT_DIR, "d4_v5_2_champion_meta.npz")

from d4_model_v4 import KinematicTCN, AsymmetricFocalLoss, BabelDeadliftD4ModelV4GBDT, FEATURE_NAMES_43_D4
from d4_velocity_timing_extractor import FEATURE_NAMES_TIMING_6D

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def train_tcn_quick(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, epochs: int = 25) -> np.ndarray:
    device = torch.device('cpu')
    model = KinematicTCN(in_features=12, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5)

    dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(val_x).squeeze(-1)).cpu().numpy()
    return probs


def evaluate_3state(y_true: np.ndarray, p_final: np.ndarray, tn: float, td: float):
    decided = (p_final <= tn) | (p_final >= td)
    cov = float(np.mean(decided))
    if cov < 0.5:
        return {"coverage": cov, "f1": 0.0, "acc": 0.0, "rec": 0.0, "prec": 0.0}
    y_dec_t = y_true[decided]
    y_dec_p = (p_final[decided] >= td).astype(int)
    f1 = float(f1_score(y_dec_t, y_dec_p, zero_division=0))
    acc = float(accuracy_score(y_dec_t, y_dec_p))
    rec = float(recall_score(y_dec_t, y_dec_p, zero_division=0))
    prec = float(precision_score(y_dec_t, y_dec_p, zero_division=0))
    return {
        "coverage": cov,
        "abstain_rate": 1.0 - cov,
        "f1": f1,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "n_decided": int(np.sum(decided)),
        "n_total": len(y_true)
    }


def eval_subset_gbdt(X_sub: np.ndarray, y: np.ndarray, subjects: np.ndarray):
    """빠른 피처셋 선별용 GBDT 5-Fold OOF AUC 및 최적 F1 계산"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    for train_idx, val_idx in sgkf.split(X_sub, y, groups=subjects):
        model = lgb.LGBMClassifier(n_estimators=80, max_depth=4, num_leaves=15, learning_rate=0.05,
                                   class_weight='balanced', random_state=42, verbose=-1)
        model.fit(X_sub[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X_sub[val_idx])[:, 1]
    auc = float(roc_auc_score(y, oof))
    best_f1 = max([float(f1_score(y, (oof >= th).astype(int), zero_division=0)) for th in np.linspace(0.35, 0.65, 31)])
    return auc, best_f1, oof


def run_full_two_stream(X_tab: np.ndarray, X_tcn: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray):
    """최종 확정 피처셋에 대한 Full Two-Stream Meta-Learner 5-Fold 교차검증"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_gbdt = np.zeros(len(y), dtype=np.float32)
    oof_tcn = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups=subjects), 1):
        lgb_m = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
                                   class_weight='balanced', random_state=42+fold, verbose=-1)
        cat_m = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                   auto_class_weights='Balanced', random_seed=42+fold, verbose=0)
        rf_m = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced',
                                      random_state=42+fold, n_jobs=-1)
        
        lgb_m.fit(X_tab[train_idx], y[train_idx])
        cat_m.fit(X_tab[train_idx], y[train_idx])
        rf_m.fit(X_tab[train_idx], y[train_idx])

        p_lgb = lgb_m.predict_proba(X_tab[val_idx])[:, 1]
        p_cat = cat_m.predict_proba(X_tab[val_idx])[:, 1]
        p_rf = rf_m.predict_proba(X_tab[val_idx])[:, 1]
        oof_gbdt[val_idx] = 0.45 * p_lgb + 0.40 * p_cat + 0.15 * p_rf

        torch.manual_seed(42 + fold)
        oof_tcn[val_idx] = train_tcn_quick(X_tcn[train_idx], y[train_idx], X_tcn[val_idx], epochs=25)

    stack_X = np.column_stack([oof_gbdt, oof_tcn])
    meta_lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
    meta_lr.fit(stack_X, y)
    oof_final = meta_lr.predict_proba(stack_X)[:, 1]

    auc_final = float(roc_auc_score(y, oof_final))
    
    # 2-State 최적 임계치
    best_2s_f1 = 0.0
    best_tau_2s = 0.5
    for tau in np.linspace(0.30, 0.75, 46):
        pred_tau = (oof_final >= tau).astype(int)
        f1_tau = float(f1_score(y, pred_tau, zero_division=0))
        if f1_tau > best_2s_f1:
            best_2s_f1 = f1_tau
            best_tau_2s = float(tau)

    p_opt = (oof_final >= best_tau_2s).astype(int)
    res_2s = {
        "best_tau": best_tau_2s,
        "f1": float(f1_score(y, p_opt)),
        "accuracy": float(accuracy_score(y, p_opt)),
        "precision": float(precision_score(y, p_opt, zero_division=0)),
        "recall": float(recall_score(y, p_opt, zero_division=0)),
        "auc": auc_final,
        "confusion_matrix": confusion_matrix(y, p_opt).tolist()
    }

    # 3-State 최적화 탐색 (Coverage >= 90%, >= 85%)
    best_3s_90, best_3s_85 = None, None
    max_f1_90, max_f1_85 = 0.0, 0.0

    for td in np.linspace(0.40, 0.78, 39):
        for tn in np.linspace(0.20, td - 0.02, 35):
            eval_res = evaluate_3state(y, oof_final, tn, td)
            cov = eval_res["coverage"]
            f1_val = eval_res["f1"]
            if cov >= 0.900 and f1_val > max_f1_90:
                max_f1_90 = f1_val
                eval_res["tau_normal"] = float(tn)
                eval_res["tau_defect"] = float(td)
                best_3s_90 = eval_res
            if cov >= 0.850 and f1_val > max_f1_85:
                max_f1_85 = f1_val
                eval_res["tau_normal"] = float(tn)
                eval_res["tau_defect"] = float(td)
                best_3s_85 = eval_res

    # 5대 시점별 세부 감사
    view_results = {}
    for vn in sorted(np.unique(views)):
        v_mask = (views == vn)
        y_v = y[v_mask]
        p_v = oof_final[v_mask]
        pred_v_2s = (p_v >= best_tau_2s).astype(int)
        
        td_sel = best_3s_90["tau_defect"] if best_3s_90 else 0.65
        tn_sel = best_3s_90["tau_normal"] if best_3s_90 else 0.40
        eval_v_3s = evaluate_3state(y_v, p_v, tn_sel, td_sel)

        view_results[vn] = {
            "alias": VIEW_ALIAS_MAP.get(vn, vn),
            "samples": int(len(y_v)),
            "defects": int(np.sum(y_v == 1)),
            "2state_f1": float(f1_score(y_v, pred_v_2s, zero_division=0)),
            "2state_acc": float(accuracy_score(y_v, pred_v_2s)),
            "2state_rec": float(recall_score(y_v, pred_v_2s, zero_division=0)),
            "2state_prec": float(precision_score(y_v, pred_v_2s, zero_division=0)),
            "auc": float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.0,
            "3state_f1": eval_v_3s["f1"],
            "3state_cov": eval_v_3s["coverage"],
            "3state_rec": eval_v_3s["recall"]
        }

    return {
        "auc": auc_final,
        "2state": res_2s,
        "3state_cov90": best_3s_90,
        "3state_cov85": best_3s_85,
        "views": view_results,
        "oof_final": oof_final
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d4 피처 최적화 토너먼트 및 다차원 성능 재평가")
    print("=" * 80)

    # 1. 데이터 로드
    cache_v4 = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt_43 = cache_v4["X_gbdt"]
    X_tcn = cache_v4["X_tcn"]
    y = cache_v4["y"]
    subjects = cache_v4["subjects"]
    views = cache_v4["views"]

    meta_v5_1 = np.load(CHAMPION_META, allow_pickle=True)
    indices_15 = list(meta_v5_1["indices"])  # 15D core
    names_15 = [FEATURE_NAMES_43_D4[i] for i in indices_15]

    timing_cache = np.load(TIMING_CACHE_NPZ, allow_pickle=True)
    X_timing_6 = timing_cache["X_timing"]

    # 2. 통합 후보 풀 (Total Candidate Pool: 24D)
    #  [0..14] : 기존 15D 코어 피처
    #  [15..20]: 신규 6D 타이밍 피처 (impulse, xcorr, phase_plane 등)
    #  [21]    : 기존 43D [2] d4_vel_ratio_first_half
    #  [22]    : 기존 43D [9] d4_hip_shoot_up_speed_norm
    #  [23]    : 기존 43D [34] d4_mvc_coupling_angle_mean_deg
    X_extra = np.column_stack([
        X_gbdt_43[:, 2],
        X_gbdt_43[:, 9],
        X_gbdt_43[:, 34]
    ])
    X_pool = np.hstack([X_gbdt_43[:, indices_15], X_timing_6, X_extra])
    
    pool_names = names_15 + FEATURE_NAMES_TIMING_6D + [
        'd4_vel_ratio_first_half',
        'd4_hip_shoot_up_speed_norm',
        'd4_mvc_coupling_angle_mean_deg'
    ]
    print(f"[*] 총 후보 피처 풀: {len(pool_names)}개 피처")

    # 3. 토너먼트 탐색
    print("\n" + "=" * 80)
    print("  [Step 1] 피처 선택 토너먼트 (Combinatorial Ablation Search)")
    print("=" * 80)

    # 기본 베이스라인: 0..14 (15D)
    baseline_auc, baseline_f1, _ = eval_subset_gbdt(X_pool[:, :15], y, subjects)
    print(f"  - Baseline (v5.1 15D Core)  : AUC={baseline_auc:.4f}, GBDT F1={baseline_f1*100:.2f}%")

    # 개별 후보 9개(15..23)의 1:1 추가 효과 테스트
    individual_add_results = []
    for cand_idx in range(15, len(pool_names)):
        cand_name = pool_names[cand_idx]
        sub_idxs = list(range(15)) + [cand_idx]
        auc, f1, _ = eval_subset_gbdt(X_pool[:, sub_idxs], y, subjects)
        diff_auc = auc - baseline_auc
        diff_f1 = f1 - baseline_f1
        individual_add_results.append((cand_idx, cand_name, auc, f1, diff_auc, diff_f1))
        print(f"  + 추가 [{cand_name:<30}]: AUC={auc:.4f} ({diff_auc:+.4f}), F1={f1*100:.2f}% ({diff_f1*100:+.2f}%p)")

    # 15D 내부에서 불필요한 하위 피처(Pruning 후보) 검토
    print("\n  [Step 2] 15D 코어 피처 Backward Pruning (제거 시 성능 변화)")
    prune_results = []
    for p_idx in range(15):
        p_name = pool_names[p_idx]
        sub_idxs = [i for i in range(15) if i != p_idx]
        auc, f1, _ = eval_subset_gbdt(X_pool[:, sub_idxs], y, subjects)
        diff_auc = auc - baseline_auc
        diff_f1 = f1 - baseline_f1
        prune_results.append((p_idx, p_name, auc, f1, diff_auc, diff_f1))
        sign_str = "제거 유리 (노이즈)" if diff_auc > 0 else "제거 불리 (필수)"
        print(f"  - 제외 [{p_name:<30}]: AUC={auc:.4f} ({diff_auc:+.4f}) -> {sign_str}")

    # 유망한 피처 조합 구성 및 토너먼트
    # 유효 추가 후보:
    #   cand 19: d4_first_pull_impulse_ratio (AUC +0.0076)
    #   cand 21: d4_vel_ratio_first_half
    #   cand 22: d4_hip_shoot_up_speed_norm
    # 유효 제거 후보 (제외 시 AUC 유지 또는 상승한 것)
    remove_candidates = [p[0] for p in prune_results if p[4] >= -0.002]
    print(f"\n  -> 가지치기(Pruning) 후보 인덱스: {[pool_names[i] for i in remove_candidates]}")

    candidate_combos = {}

    # Combo 1: Baseline 15D + ImpulseRatio (16D)
    candidate_combos["16D_ImpulseOnly"] = list(range(15)) + [19]

    # Combo 2: Baseline 15D + ImpulseRatio + VelRatioFirstHalf (17D)
    candidate_combos["17D_Impulse_VelRatio"] = list(range(15)) + [19, 21]

    # Combo 3: Baseline 15D + ImpulseRatio + HipShootUp (17D)
    candidate_combos["17D_Impulse_HipShoot"] = list(range(15)) + [19, 22]

    # Combo 4: Pruned 13D (하위 2개 제거) + ImpulseRatio (14D)
    # 최하위 기여 피처 2개 제거
    sorted_prune = sorted(prune_results, key=lambda x: x[4], reverse=True) # diff_auc 높은 순 (제거해도 좋은 순)
    worst_2 = [sorted_prune[0][0], sorted_prune[1][0]]
    pruned_13_base = [i for i in range(15) if i not in worst_2]
    candidate_combos["14D_Pruned_Impulse"] = pruned_13_base + [19]

    # Combo 5: Pruned 13D + ImpulseRatio + VelRatioFirstHalf (15D)
    candidate_combos["15D_Pruned_Impulse_VelRatio"] = pruned_13_base + [19, 21]

    # Combo 6: Pruned 13D + ImpulseRatio + XcorrLag (15D)
    candidate_combos["15D_Pruned_Impulse_Xcorr"] = pruned_13_base + [19, 16]

    # Combo 7: Pruned 12D (하위 3개 제거) + ImpulseRatio + VelRatio + HipShoot (15D)
    worst_3 = [sorted_prune[0][0], sorted_prune[1][0], sorted_prune[2][0]]
    pruned_12_base = [i for i in range(15) if i not in worst_3]
    candidate_combos["15D_Surgical_Replaced"] = pruned_12_base + [19, 21, 22]

    print("\n" + "=" * 80)
    print("  [Step 3] 최종 후보 조합 토너먼트 (GBDT OOF 평가)")
    print("=" * 80)
    combo_evals = []
    for c_name, c_idxs in candidate_combos.items():
        auc, f1, _ = eval_subset_gbdt(X_pool[:, c_idxs], y, subjects)
        combo_evals.append((c_name, c_idxs, auc, f1))
        print(f"  * {c_name:<30} ({len(c_idxs)}D): AUC={auc:.4f}, F1={f1*100:.2f}%")

    # 최고 성능 챔피언 조합 선택
    combo_evals.sort(key=lambda x: (x[2], x[3]), reverse=True)
    champ_name, champ_idxs, champ_auc, champ_f1 = combo_evals[0]

    print("\n" + "=" * 80)
    print(f"  [★ 토너먼트 챔피언 선정: {champ_name} ({len(champ_idxs)}D)]")
    print(f"  - GBDT 단독 AUC: {champ_auc:.4f} (베이스라인 대비 {champ_auc-baseline_auc:+.4f})")
    print(f"  - 선정된 피처 목록:")
    for idx_in_pool in champ_idxs:
        print(f"      - {pool_names[idx_in_pool]}")
    print("=" * 80)

    # 4. 시점 적응형(View-Adaptive) 보정 테스트
    # 정면(시점 C = view3)일 때 충격량 및 타이밍 피처의 가중치를 0.5로 완화하여 지터 방어
    print("\n[*] 시점 C(정면) 지터 방어용 시점 적응형 피처 스케일링 적용...")
    X_champ_raw = X_pool[:, champ_idxs].copy()
    X_champ_adaptive = X_champ_raw.copy()
    
    # 타이밍 피처(15번 이후 인덱스)는 view3(정면)에서 중앙값으로 감쇠
    for c_i, pool_i in enumerate(champ_idxs):
        if pool_i >= 15: # 신규 타이밍/동역학 피처
            med_val = np.median(X_champ_adaptive[:, c_i])
            v3_mask = (views == 'view3')
            # 정면 시점에서 극단치 완화
            X_champ_adaptive[v3_mask, c_i] = 0.5 * X_champ_adaptive[v3_mask, c_i] + 0.5 * med_val

    # 5. Full Two-Stream 메타러너 재평가 (Baseline vs Champion-Raw vs Champion-Adaptive)
    print("\n" + "=" * 80)
    print("  [Step 4] Two-Stream Meta-Learner (GBDT + 1D-TCN) 최종 성능 재평가")
    print("=" * 80)

    print("\n1) Baseline (v5.1 15D 코어) Two-Stream 재평가 중...")
    res_base = run_full_two_stream(X_pool[:, :15], X_tcn, y, subjects, views)

    print("\n2) Champion (최적 피처셋) Two-Stream 재평가 중...")
    res_champ_raw = run_full_two_stream(X_champ_raw, X_tcn, y, subjects, views)

    print("\n3) Champion + View-Adaptive(시점 적응형) Two-Stream 재평가 중...")
    res_champ_adapt = run_full_two_stream(X_champ_adaptive, X_tcn, y, subjects, views)

    # 6. 결과 종합 및 저장
    final_output = {
        "baseline_v5_1": {
            "n_features": 15,
            "feature_names": names_15,
            "auc": res_base["auc"],
            "2state": res_base["2state"],
            "3state_cov90": res_base["3state_cov90"],
            "3state_cov85": res_base["3state_cov85"],
            "views": res_base["views"]
        },
        "champion_optimized_raw": {
            "name": champ_name,
            "n_features": len(champ_idxs),
            "feature_names": [pool_names[i] for i in champ_idxs],
            "pool_indices": [int(i) for i in champ_idxs],
            "auc": res_champ_raw["auc"],
            "2state": res_champ_raw["2state"],
            "3state_cov90": res_champ_raw["3state_cov90"],
            "3state_cov85": res_champ_raw["3state_cov85"],
            "views": res_champ_raw["views"]
        },
        "champion_optimized_adaptive": {
            "name": f"{champ_name}_ViewAdaptive",
            "n_features": len(champ_idxs),
            "feature_names": [pool_names[i] for i in champ_idxs],
            "pool_indices": [int(i) for i in champ_idxs],
            "auc": res_champ_adapt["auc"],
            "2state": res_champ_adapt["2state"],
            "3state_cov90": res_champ_adapt["3state_cov90"],
            "3state_cov85": res_champ_adapt["3state_cov85"],
            "views": res_champ_adapt["views"]
        }
    }

    with open(OUT_OPTIMIZED_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 최종 재평가 결과 JSON 저장 완료: {OUT_OPTIMIZED_JSON}")

    # 최고 챔피언 메타 저장 (v5.2 공식 확정용)
    best_champ_dict = res_champ_adapt if res_champ_adapt["auc"] >= res_champ_raw["auc"] else res_champ_raw
    best_champ_key = "champion_optimized_adaptive" if res_champ_adapt["auc"] >= res_champ_raw["auc"] else "champion_optimized_raw"
    np.savez_compressed(
        OUT_CHAMPION_NPZ,
        feature_names=np.array([pool_names[i] for i in champ_idxs]),
        pool_indices=np.array(champ_idxs),
        auc=best_champ_dict["auc"],
        best_tau_2s=best_champ_dict["2state"]["best_tau"],
        tn_3s=best_champ_dict["3state_cov90"]["tau_normal"] if best_champ_dict["3state_cov90"] else 0.4,
        td_3s=best_champ_dict["3state_cov90"]["tau_defect"] if best_champ_dict["3state_cov90"] else 0.7
    )
    print(f"[+] v5.2 챔피언 메타 저장 완료: {OUT_CHAMPION_NPZ}")

    # 최종 보고서 콘솔 출력
    print("\n" + "=" * 80)
    print("  [★ d4 피처 최적화 전/후 다차원 성능 재평가 종합 보고]")
    print("=" * 80)
    
    models_to_report = [
        ("기존 베이스라인 (v5.1 15D 코어)", res_base),
        (f"최적화 챔피언 ({champ_name}, {len(champ_idxs)}D)", res_champ_raw),
        (f"최적화 챔피언 + 시점적응형 ({len(champ_idxs)}D)", res_champ_adapt)
    ]

    for m_label, m_res in models_to_report:
        auc = m_res["auc"]
        f1_2s = m_res["2state"]["f1"]
        rec_2s = m_res["2state"]["recall"]
        prec_2s = m_res["2state"]["precision"]
        acc_2s = m_res["2state"]["accuracy"]
        c90_f1 = m_res["3state_cov90"]["f1"] if m_res["3state_cov90"] else 0.0
        c90_cov = m_res["3state_cov90"]["coverage"] if m_res["3state_cov90"] else 0.0
        c85_f1 = m_res["3state_cov85"]["f1"] if m_res["3state_cov85"] else 0.0
        c85_cov = m_res["3state_cov85"]["coverage"] if m_res["3state_cov85"] else 0.0

        print(f"\n[Model] {m_label}")
        print(f"  - Two-Stream ROC-AUC : {auc:.4f}")
        print(f"  - 2-State (최적 임계치) : F1={f1_2s*100:.2f}%, Recall={rec_2s*100:.2f}%, Prec={prec_2s*100:.2f}%, Acc={acc_2s*100:.2f}%")
        print(f"  - 3-State (Cov >= 90%): F1={c90_f1*100:.2f}%, Coverage={c90_cov*100:.2f}% (보류율: {(1-c90_cov)*100:.2f}%)")
        print(f"  - 3-State (Cov >= 85%): F1={c85_f1*100:.2f}%, Coverage={c85_cov*100:.2f}% (보류율: {(1-c85_cov)*100:.2f}%)")
        print("  - 5대 시점별 2-State F1 / AUC:")
        for vn, vd in m_res["views"].items():
            alias = vd["alias"]
            f1_v = vd["2state_f1"]
            auc_v = vd["auc"]
            rec_v = vd["2state_rec"]
            print(f"      {alias:<24}: F1={f1_v*100:.2f}%, Recall={rec_v*100:.2f}%, AUC={auc_v:.4f}")


if __name__ == "__main__":
    main()

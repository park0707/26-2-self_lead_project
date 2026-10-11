# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_d4_timing_experiment.py
Description:
  d4 무릎과 골반 동시 펴짐 위반 모델에 생체역학 각속도 타이밍 6D 피처를 투입하고,
  기존 베이스라인(15D v5.1 및 43D v4) 대비 정량적 성능 향상을 5-Fold StratifiedGroupKFold로 전수 검증
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
CHAMPION_META = os.path.join(CURRENT_DIR, "d4_v5_1_diet_champion.npz")
TIMING_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_timing_6d_cache.npz")
EXP_RESULTS_JSON = os.path.join(CURRENT_DIR, "d4_timing_experiment_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

from d4_model_v4 import KinematicTCN, AsymmetricFocalLoss, BabelDeadliftD4ModelV4GBDT, FEATURE_NAMES_43_D4
from d4_velocity_timing_extractor import extract_d4_velocity_timing_features_from_seq, FEATURE_NAMES_TIMING_6D


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


def run_pipeline(X_tab: np.ndarray, X_tcn: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, exp_name: str):
    print("=" * 80)
    print(f" [*] 실험 실행: {exp_name} (피처 차원: {X_tab.shape[1]}D)")
    print("=" * 80)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_gbdt = np.zeros(len(y), dtype=np.float32)
    oof_tcn = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups=subjects), 1):
        # GBDT Tri-Ensemble
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

        # TCN
        torch.manual_seed(42 + fold)
        oof_tcn[val_idx] = train_tcn_quick(X_tcn[train_idx], y[train_idx], X_tcn[val_idx], epochs=25)

    # Meta Fusion
    stack_X = np.column_stack([oof_gbdt, oof_tcn])
    meta_lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
    meta_lr.fit(stack_X, y)
    oof_final = meta_lr.predict_proba(stack_X)[:, 1]

    auc_final = float(roc_auc_score(y, oof_final))
    
    # 2-State 최적 임계치 탐색
    best_2s_f1 = 0.0
    best_tau_2s = 0.5
    for tau in np.linspace(0.30, 0.75, 46):
        pred_tau = (oof_final >= tau).astype(int)
        f1_tau = float(f1_score(y, pred_tau, zero_division=0))
        if f1_tau > best_2s_f1:
            best_2s_f1 = f1_tau
            best_tau_2s = float(tau)

    # 2-State (tau=0.50, tau=0.70, 최적 tau)
    p_opt = (oof_final >= best_tau_2s).astype(int)
    res_2s = {
        "best_tau": best_tau_2s,
        "f1": float(f1_score(y, p_opt)),
        "accuracy": float(accuracy_score(y, p_opt)),
        "precision": float(precision_score(y, p_opt)),
        "recall": float(recall_score(y, p_opt)),
        "auc": auc_final
    }

    # 3-State 최적화 탐색 (Coverage >= 90%, >= 85%)
    best_3s_90 = None
    best_3s_85 = None
    max_f1_90 = 0.0
    max_f1_85 = 0.0

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

    # 5대 시점별 세부 성능 감사
    view_results = {}
    for vn in sorted(np.unique(views)):
        v_mask = (views == vn)
        y_v = y[v_mask]
        p_v = oof_final[v_mask]
        pred_v_2s = (p_v >= best_tau_2s).astype(int)
        
        # 3-State
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

    print(f"  -> ROC-AUC: {auc_final:.4f}")
    print(f"  -> 2-State 최적 (tau={best_tau_2s:.2f}): F1={res_2s['f1']*100:.2f}%, Recall={res_2s['recall']*100:.2f}%, Prec={res_2s['precision']*100:.2f}%")
    if best_3s_90:
        print(f"  -> 3-State (Cov>=90%): F1={best_3s_90['f1']*100:.2f}%, Coverage={best_3s_90['coverage']*100:.2f}%, Recall={best_3s_90['recall']*100:.2f}%")
    if best_3s_85:
        print(f"  -> 3-State (Cov>=85%): F1={best_3s_85['f1']*100:.2f}%, Coverage={best_3s_85['coverage']*100:.2f}%, Recall={best_3s_85['recall']*100:.2f}%")

    return {
        "exp_name": exp_name,
        "n_features": X_tab.shape[1],
        "auc": auc_final,
        "2state": res_2s,
        "3state_cov90": best_3s_90,
        "3state_cov85": best_3s_85,
        "views": view_results,
        "oof_final": oof_final.tolist()
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d4 각속도 타이밍 6D 피처 추가 비교 평가 실험")
    print("=" * 80)

    # 1. 원천 데이터 및 캐시 로드
    cache_v4 = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt_43 = cache_v4["X_gbdt"]
    X_tcn = cache_v4["X_tcn"]
    y = cache_v4["y"]
    subjects = cache_v4["subjects"]
    views = cache_v4["views"]

    meta = np.load(CHAMPION_META, allow_pickle=True)
    diet_15_indices = meta["indices"]
    X_gbdt_15 = X_gbdt_43[:, diet_15_indices]

    # 2. 6D 타이밍 피처 추출 (또는 캐시 로드)
    if os.path.exists(TIMING_CACHE_NPZ):
        print(f"[*] 캐시된 6D 타이밍 피처 로드: {TIMING_CACHE_NPZ}")
        X_timing_6d = np.load(TIMING_CACHE_NPZ)["X_timing"]
    else:
        print("[*] 770개 시퀀스로부터 6D 타이밍 피처 실시간 추출 중...")
        import glob
        raw_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
        feats_list = []
        for f in raw_files:
            kps = np.load(f)["kps_33_3d_64f"]
            feats_list.append(extract_d4_velocity_timing_features_from_seq(kps))
        X_timing_6d = np.array(feats_list, dtype=np.float32)
        np.savez_compressed(TIMING_CACHE_NPZ, X_timing=X_timing_6d)
        print(f"[+] 6D 타이밍 피처 캐싱 완료: {X_timing_6d.shape}")

    # 3. 피처 결합
    # Baseline: 15D
    # Model A: 15D + 6D = 21D (v5.2-Timing-Core)
    # Model B: 43D + 6D = 49D (v5.2-Timing-Full)
    X_tab_21 = np.hstack([X_gbdt_15, X_timing_6d])
    X_tab_49 = np.hstack([X_gbdt_43, X_timing_6d])

    results = {}

    # 실험 1: Baseline v5.1 (15D 코어)
    results["baseline_15d"] = run_pipeline(X_gbdt_15, X_tcn, y, subjects, views, "Baseline_v5.1_15D")

    # 실험 2: v5.2-Timing (21D 코어 + 타이밍)
    results["timing_21d"] = run_pipeline(X_tab_21, X_tcn, y, subjects, views, "v5.2_Timing_21D_Core")

    # 실험 3: v5.2-Timing-Full (49D 전체 + 타이밍)
    results["timing_49d"] = run_pipeline(X_tab_49, X_tcn, y, subjects, views, "v5.2_Timing_49D_Full")

    # 결과 저장
    save_obj = {k: {sk: sv for sk, sv in v.items() if sk != "oof_final"} for k, v in results.items()}
    with open(EXP_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 전체 실험 결과 저장 완료: {EXP_RESULTS_JSON}")

    # 요약 출력
    print("\n" + "=" * 80)
    print("  [★ d4 성능 비교 요약: 베이스라인 vs 신규 각속도 타이밍 모델]")
    print("=" * 80)
    for exp_k, exp_v in results.items():
        name = exp_v["exp_name"]
        dim = exp_v["n_features"]
        auc = exp_v["auc"]
        f1_2s = exp_v["2state"]["f1"]
        rec_2s = exp_v["2state"]["recall"]
        f1_3s_90 = exp_v["3state_cov90"]["f1"] if exp_v["3state_cov90"] else 0.0
        cov_90 = exp_v["3state_cov90"]["coverage"] if exp_v["3state_cov90"] else 0.0
        f1_3s_85 = exp_v["3state_cov85"]["f1"] if exp_v["3state_cov85"] else 0.0
        cov_85 = exp_v["3state_cov85"]["coverage"] if exp_v["3state_cov85"] else 0.0
        
        print(f"[{name} ({dim}D)]")
        print(f"  - ROC-AUC: {auc:.4f}")
        print(f"  - 2-State (최적): F1={f1_2s*100:.2f}%, Recall={rec_2s*100:.2f}%")
        print(f"  - 3-State (Cov>=90%): F1={f1_3s_90*100:.2f}%, Coverage={cov_90*100:.2f}%")
        print(f"  - 3-State (Cov>=85%): F1={f1_3s_85*100:.2f}%, Coverage={cov_85*100:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    main()

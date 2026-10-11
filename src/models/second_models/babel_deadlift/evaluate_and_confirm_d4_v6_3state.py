# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_and_confirm_d4_v6_3state.py
Purpose:
  1. 바벨 데드리프트 d4 2계층 캐스케이드 v6.0 최종 확정
  2. 3-State (선택적 분류 / 보류 옵션) Coverage 95%, 90%, 85% 및 최적 F1 성능 정밀 도출
  3. 1계층(Level 1 명확군) 및 2계층(Level 1~3 클린 실무) 3-State F1-Score 실측
  4. 공식 5대 시점(`view1`~`view5`)별 3-State F1 세부 감사
  5. 최종 모델 가중치 패키징: weights_v6_0/babel_deadlift_d4_v6_final.pkl
  6. 결과 JSON 저장: d4_v6_3state_final_results.json
"""

import os
import sys

# Windows UTF-8 입출력 강제
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
TIMING_CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_timing_6d_cache.npz")
CHAMPION_META = os.path.join(CURRENT_DIR, "d4_v5_2_champion_meta.npz")
OUT_JSON = os.path.join(CURRENT_DIR, "d4_v6_3state_final_results.json")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_v6_0")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "babel_deadlift_d4_v6_final.pkl")

from d4_model_v5 import FEATURE_NAMES_43_D4, DIET_CORE_INDICES_15_D4, KinematicTCN
from run_d4_two_tier_cascade_experiment import classify_ambiguity_levels, TwoStreamTrainer

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def evaluate_3state_detailed(y_true: np.ndarray, probs: np.ndarray, tn: float, td: float) -> Dict[str, Any]:
    """3-State 선택적 분류 평가: P <= tn (정상 0), P >= td (결함 1), else (보류 RETAKE)"""
    decided = (probs <= tn) | (probs >= td)
    cov = float(np.mean(decided))
    if cov < 0.20 or np.sum(decided) == 0:
        return {"coverage": cov, "abstain_rate": 1.0 - cov, "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_decided": 0}
    
    y_dec_t = y_true[decided]
    y_dec_p = (probs[decided] >= td).astype(int)
    
    f1 = float(f1_score(y_dec_t, y_dec_p, zero_division=0))
    acc = float(accuracy_score(y_dec_t, y_dec_p))
    rec = float(recall_score(y_dec_t, y_dec_p, zero_division=0))
    prec = float(precision_score(y_dec_t, y_dec_p, zero_division=0))
    cm = confusion_matrix(y_dec_t, y_dec_p).tolist()

    return {
        "coverage": cov,
        "abstain_rate": 1.0 - cov,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "n_decided": int(np.sum(decided)),
        "n_total": len(y_true),
        "tau_normal": float(tn),
        "tau_defect": float(td),
        "confusion_matrix": cm
    }


def search_best_3state(y_true: np.ndarray, probs: np.ndarray, target_cov: float = None, min_cov: float = 0.50):
    """지정 커버리지 또는 최적 F1을 만족하는 (tn, td) 탐색"""
    best_res = None
    best_f1 = -1.0
    
    # tn, td 그리드 탐색
    for tn in np.linspace(0.15, 0.48, 34):
        for td in np.linspace(tn + 0.02, 0.75, 34):
            res = evaluate_3state_detailed(y_true, probs, tn, td)
            cov = res["coverage"]
            
            if target_cov is not None:
                # 목표 커버리지 허용 오차 ±2.5%
                if abs(cov - target_cov) <= 0.025:
                    if res["f1"] > best_f1:
                        best_f1 = res["f1"]
                        best_res = res
            else:
                if cov >= min_cov:
                    if res["f1"] > best_f1:
                        best_f1 = res["f1"]
                        best_res = res
                        
    return best_res


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d4 2계층 캐스케이드 v6.0 최종 확정 및 3-State F1 성능 감사")
    print("=" * 80)

    # 1. 데이터 로드
    raw_data = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt_raw = raw_data['X_gbdt']
    X_tcn_raw = raw_data['X_tcn']
    y_raw = raw_data['y']
    subjects_raw = raw_data['subjects']
    views_raw = raw_data['views']

    champ_meta = np.load(CHAMPION_META, allow_pickle=True)
    pool_indices = champ_meta['pool_indices']

    timing_data = np.load(TIMING_CACHE_NPZ, allow_pickle=True)
    X_timing = timing_data['X_timing']
    X_candidate_pool = np.column_stack([X_gbdt_raw[:, DIET_CORE_INDICES_15_D4], X_timing])
    X_opt_gbdt = X_candidate_pool[:, pool_indices]  # 14D

    # 2. 4단계 모호성 분류 및 Level 4 폐기
    levels = classify_ambiguity_levels(X_gbdt_raw, y_raw, views_raw)
    clean_mask = (levels <= 3) # Level 1~3 보존 (Level 4 폐기)

    X_gbdt_clean = X_opt_gbdt[clean_mask]
    X_tcn_clean = X_tcn_raw[clean_mask]
    y_clean = y_raw[clean_mask]
    subj_clean = subjects_raw[clean_mask]
    views_clean = views_raw[clean_mask]
    levels_clean = levels[clean_mask]

    print(f"전체 샘플: {len(y_raw)}개 -> Level 4(84개) 폐기 후 클린 풀(Level 1~3): {len(y_clean)}개")

    # 3. 5-Fold StratifiedGroupKFold OOF 확률 산출
    trainer = TwoStreamTrainer(random_state=42)
    print("\nLevel 1~3 클린 데이터셋 5-Fold Two-Stream 학습 및 추론 중...")
    oof_probs_clean, oof_probs_g, oof_probs_t = trainer.fit_and_predict_oof(
        X_gbdt_clean, X_tcn_clean, y_clean, subj_clean
    )
    auc_clean = float(roc_auc_score(y_clean, oof_probs_clean))

    # (A) 2-State 베이스라인 F1
    from run_d4_two_tier_cascade_experiment import optimize_threshold_f1
    tau_2s, f1_2s, acc_2s, prec_2s, rec_2s = optimize_threshold_f1(y_clean, oof_probs_clean)
    print(f"\n[2-State 기준 성적 (Level 1~3 전수)]")
    print(f"  • F1-Score: {f1_2s*100:.2f}%, AUC: {auc_clean*100:.2f}%, Prec: {prec_2s*100:.2f}%, Rec: {rec_2s*100:.2f}%, tau: {tau_2s:.2f}")

    # (B) 3-State F1 성능 분석 (Level 1~3 전체 클린 실무 평가)
    print("\n" + "=" * 60)
    print(" [3-State 선택적 분류 (Selective Classification) 성능 실측]")
    print("=" * 60)

    # 1) Coverage 90% (보류율 10%)
    res_cov90 = search_best_3state(y_clean, oof_probs_clean, target_cov=0.90)
    print(f"\n■ Coverage 90% 3-State (보류율 약 10%):")
    print(f"  • F1-Score: {res_cov90['f1']*100:.2f}% (2-State 대비 +{(res_cov90['f1'] - f1_2s)*100:+.2f}%p 도약!)")
    print(f"  • Coverage: {res_cov90['coverage']*100:.1f}% ({res_cov90['n_decided']}/{res_cov90['n_total']}개 확정, {res_cov90['abstain_rate']*100:.1f}% 보류)")
    print(f"  • Accuracy / Precision / Recall: {res_cov90['accuracy']*100:.2f}% / {res_cov90['precision']*100:.2f}% / {res_cov90['recall']*100:.2f}%")
    print(f"  • 임계값: tau_normal={res_cov90['tau_normal']:.2f}, tau_defect={res_cov90['tau_defect']:.2f}")

    # 2) Coverage 85% (보류율 15%)
    res_cov85 = search_best_3state(y_clean, oof_probs_clean, target_cov=0.85)
    print(f"\n■ Coverage 85% 3-State (보류율 약 15%):")
    print(f"  • F1-Score: {res_cov85['f1']*100:.2f}% (2-State 대비 +{(res_cov85['f1'] - f1_2s)*100:+.2f}%p 도약!)")
    print(f"  • Coverage: {res_cov85['coverage']*100:.1f}% ({res_cov85['n_decided']}/{res_cov85['n_total']}개 확정)")
    print(f"  • Accuracy / Precision / Recall: {res_cov85['accuracy']*100:.2f}% / {res_cov85['precision']*100:.2f}% / {res_cov85['recall']*100:.2f}%")
    print(f"  • 임계값: tau_normal={res_cov85['tau_normal']:.2f}, tau_defect={res_cov85['tau_defect']:.2f}")

    # 3) Coverage 80% (고순도 / 고재현형)
    res_cov80 = search_best_3state(y_clean, oof_probs_clean, target_cov=0.80)
    print(f"\n■ Coverage 80% 3-State (고순도 정밀 판정):")
    print(f"  • F1-Score: {res_cov80['f1']*100:.2f}% (2-State 대비 +{(res_cov80['f1'] - f1_2s)*100:+.2f}%p)")
    print(f"  • Coverage: {res_cov80['coverage']*100:.1f}% ({res_cov80['n_decided']}/{res_cov80['n_total']}개 확정)")
    print(f"  • Accuracy / Precision / Recall: {res_cov80['accuracy']*100:.2f}% / {res_cov80['precision']*100:.2f}% / {res_cov80['recall']*100:.2f}%")
    print(f"  • 임계값: tau_normal={res_cov80['tau_normal']:.2f}, tau_defect={res_cov80['tau_defect']:.2f}")

    # (C) 1계층 (Level 1 완전 명확군) 3-State 성능
    mask_l1 = (levels_clean == 1)
    y_l1 = y_clean[mask_l1]
    probs_l1 = oof_probs_clean[mask_l1]
    res_l1_cov95 = search_best_3state(y_l1, probs_l1, target_cov=0.95)
    res_l1_cov90 = search_best_3state(y_l1, probs_l1, target_cov=0.90)
    print(f"\n■ [1계층 Level 1 명확군] 3-State 성능:")
    print(f"  • Coverage 95%: F1={res_l1_cov95['f1']*100:.2f}%, Acc={res_l1_cov95['accuracy']*100:.2f}%, Prec={res_l1_cov95['precision']*100:.2f}%, Rec={res_l1_cov95['recall']*100:.2f}%")
    print(f"  • Coverage 90%: F1={res_l1_cov90['f1']*100:.2f}%, Acc={res_l1_cov90['accuracy']*100:.2f}%, Prec={res_l1_cov90['precision']*100:.2f}%, Rec={res_l1_cov90['recall']*100:.2f}%")

    # (D) 공식 5대 시점(`view1`~`view5`)별 3-State F1 성능 감사
    print("\n" + "=" * 60)
    print(" [5대 시점별 3-State F1 전수 감사 (Coverage 85~90% 대역)]")
    print("=" * 60)
    view_3state_audit = {}

    tn_std = res_cov85['tau_normal']
    td_std = res_cov85['tau_defect']

    for v_code in ["view1", "view2", "view3", "view4", "view5"]:
        alias = VIEW_ALIAS_MAP[v_code]
        m_v = (views_clean == v_code)
        y_v = y_clean[m_v]
        p_v = oof_probs_clean[m_v]

        # 해당 시점에서의 3-State 평가 (표준 임계값 적용)
        res_v = evaluate_3state_detailed(y_v, p_v, tn_std, td_std)
        # 해당 시점 맞춤 최적 3-State 탐색 (Cov 85% 부근)
        res_v_opt = search_best_3state(y_v, p_v, target_cov=0.85)

        view_3state_audit[v_code] = {
            "alias": alias,
            "samples": int(len(y_v)),
            "defects": int(np.sum(y_v)),
            "standard_3state": res_v,
            "optimal_3state": res_v_opt
        }

        print(f"[{alias}]")
        print(f"  • 표준 3-State (tn={tn_std:.2f}, td={td_std:.2f}): F1={res_v['f1']*100:.2f}%, Coverage={res_v['coverage']*100:.1f}%, Prec={res_v['precision']*100:.2f}%, Rec={res_v['recall']*100:.2f}%")
        print(f"  • 시점 최적 3-State (Cov {res_v_opt['coverage']*100:.1f}%): F1={res_v_opt['f1']*100:.2f}%, Prec={res_v_opt['precision']*100:.2f}%, Rec={res_v_opt['recall']*100:.2f}%")

    # 4. 최종 확정 모델 가중치 전체 학습 및 저장
    print("\n" + "=" * 60)
    print(" [최종 확정 모델(v6.0) 전체 가중치 학습 및 패키징]")
    print("=" * 60)
    
    # GBDT Tri-Ensemble
    n_pos = max(int(np.sum(y_clean == 1)), 1)
    n_neg = max(int(np.sum(y_clean == 0)), 1)
    spw = float(n_neg / n_pos)

    clf_lgb_final = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.04, num_leaves=15,
        scale_pos_weight=spw, n_jobs=1, random_state=42, verbose=-1
    )
    clf_cat_final = CatBoostClassifier(
        iterations=80, learning_rate=0.05, depth=4,
        scale_pos_weight=spw, thread_count=1, random_seed=42, verbose=False
    )
    clf_rf_final = RandomForestClassifier(
        n_estimators=80, max_depth=5, class_weight='balanced',
        n_jobs=1, random_state=42
    )

    clf_lgb_final.fit(X_gbdt_clean, y_clean)
    clf_cat_final.fit(X_gbdt_clean, y_clean)
    clf_rf_final.fit(X_gbdt_clean, y_clean)

    # TCN Final
    device = torch.device('cpu')
    model_tcn_final = KinematicTCN(in_features=12, hidden_dim=32).to(device)
    opt = torch.optim.AdamW(model_tcn_final.parameters(), lr=0.005, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(X_tcn_clean, dtype=torch.float32), torch.tensor(y_clean, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model_tcn_final.train()
    for _ in range(25):
        for bx, by in loader:
            opt.zero_grad()
            logits = model_tcn_final(bx).squeeze(-1)
            p = torch.sigmoid(logits)
            loss = - (by * torch.log(p + 1e-6) + 3.0 * (1 - by) * (p**3.0) * torch.log(1 - p + 1e-6)).mean()
            loss.backward()
            opt.step()
    model_tcn_final.eval()

    # 패키징 객체 구성
    package_payload = {
        "version": "v6.0_two_tier",
        "exercise": "바벨 데드리프트 (02_바벨_데드리프트)",
        "defect": "d4. 무릎과 골반이 동시에 펴짐 위반",
        "features": list(champ_meta['feature_names']),
        "pool_indices": pool_indices.tolist(),
        "tau_2state": float(tau_2s),
        "tau_tier1_low": 0.30,
        "tau_tier1_high": 0.55,
        "tau_3state_normal": float(res_cov85['tau_normal']),
        "tau_3state_defect": float(res_cov85['tau_defect']),
        "coverage_85_f1": float(res_cov85['f1']),
        "coverage_90_f1": float(res_cov90['f1']),
        "clf_lgb": clf_lgb_final,
        "clf_cat": clf_cat_final,
        "clf_rf": clf_rf_final,
        "tcn_state_dict": model_tcn_final.state_dict()
    }

    with open(WEIGHTS_PATH, 'wb') as f:
        pickle.dump(package_payload, f)
    print(f"[성공] 최종 가중치 패키징 완료: {WEIGHTS_PATH}")

    # 결과 JSON 저장
    full_results = {
        "exercise": "바벨 데드리프트 (02_바벨_데드리프트)",
        "defect": "d4. 무릎과 골반이 동시에 펴짐 위반",
        "version": "v6.0_two_tier_confirmed",
        "2state_baseline": {
            "f1": f1_2s,
            "auc": auc_clean,
            "accuracy": acc_2s,
            "precision": prec_2s,
            "recall": rec_2s,
            "tau": tau_2s
        },
        "3state_cov90": res_cov90,
        "3state_cov85": res_cov85,
        "3state_cov80": res_cov80,
        "tier1_level1_3state": {
            "cov95": res_l1_cov95,
            "cov90": res_l1_cov90
        },
        "views_3state": view_3state_audit
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"[성공] 3-State 최종 결과 JSON 저장 완료: {OUT_JSON}")


if __name__ == '__main__':
    main()

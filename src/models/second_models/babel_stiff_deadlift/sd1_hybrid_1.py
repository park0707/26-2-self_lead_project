# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Model: sd1_hybrid_1.py
Purpose:
  1. MediaPipe 3D (24D) + RTMPose 2D (41D) Feature-level Hybrid Fusion
  2. View-Conditioning (user_selected_view_id) 적용
  3. No AUM (원천 1,035건 데이터셋 상에서 순수 성능 실측)
  4. 피험자 9명 완전 격리 StratifiedGroupKFold 5-Fold 교차검증
  5. 단일 모델(MP vs RTMP) 대비 하이브리드 결합 시 성능 변화 전수 분석
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(CURRENT_DIR, "stiff_sd1_hybrid_features_cache.npz")
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "sd1_hybrid_1_benchmark_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_TO_INT = {
    "view1": 1,
    "view2": 2,
    "view3": 3,
    "view4": 4,
    "view5": 5
}

# RTMPose 26D 최적화 피처 인덱스 (이전 단계에서 도출된 정예 26D)
RTMP_OPT_26_NAMES = [
    'knee_angle_bottom_deg', 'knee_hip_rom_ratio', 'hip_knee_velocity_corr', 'mcrp', 'vcrp', 
    'knee_angular_accel_max', 'knee_accel_sign_changes', 'excess_flexion_integral', 'kadr_max', 
    'kadr_rom', 'shank_svii', 'norm_knee_disp_y', 'norm_knee_disp_x', 'pli', 'hyperextension_ratio', 
    'bilateral_knee_asym', 'barbell_knee_clearance_nadir', 'kawr_nadir', 'kawr_rom', 
    'ankle_dorsiflexion_bottom', 'ankle_dorsiflexion_rom', 'barbell_to_pelvis_drop_ratio', 
    'torso_pitch_bottom', 'view_code', 'v_oh_2', 'v_oh_4'
]

def eval_3state_grid(y_true: np.ndarray, y_prob: np.ndarray, min_coverage: float = 0.85):
    """3-State Threshold Grid Search (Retake <= 15%, Coverage >= 85%)"""
    best_res = None
    best_f1 = -1.0
    best_tn, best_td = 0.5, 0.5
    
    t_cands = np.linspace(0.20, 0.55, 36)
    for tn in t_cands:
        for td in np.linspace(tn + 0.02, 0.75, 34):
            mask_decided = (y_prob < tn) | (y_prob >= td)
            cov = float(np.mean(mask_decided))
            if cov < min_coverage:
                continue
            
            y_sub = y_true[mask_decided]
            p_sub = (y_prob[mask_decided] >= td).astype(int)
            
            if len(np.unique(y_sub)) < 2 and len(np.unique(p_sub)) < 2:
                continue
                
            f1_def = f1_score(y_sub, p_sub, pos_label=1, zero_division=0)
            f1_clean = f1_score(y_sub, p_sub, pos_label=0, zero_division=0)
            acc = accuracy_score(y_sub, p_sub)
            prec = precision_score(y_sub, p_sub, pos_label=1, zero_division=0)
            rec = recall_score(y_sub, p_sub, pos_label=1, zero_division=0)
            
            if f1_def > best_f1:
                best_f1 = f1_def
                best_tn, best_td = tn, td
                best_res = {
                    "coverage": cov,
                    "retake_rate": 1.0 - cov,
                    "selective_defect_f1": float(f1_def),
                    "selective_clean_f1": float(f1_clean),
                    "selective_accuracy": float(acc),
                    "selective_precision": float(prec),
                    "selective_recall": float(rec),
                    "tau_clean": float(tn),
                    "tau_defect": float(td)
                }
                
    if best_res is None:
        best_res = {
            "coverage": 1.0,
            "retake_rate": 0.0,
            "selective_defect_f1": float(f1_score(y_true, (y_prob >= 0.5).astype(int), pos_label=1, zero_division=0)),
            "selective_clean_f1": float(f1_score(y_true, (y_prob >= 0.5).astype(int), pos_label=0, zero_division=0)),
            "selective_accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
            "selective_precision": float(precision_score(y_true, (y_prob >= 0.5).astype(int), pos_label=1, zero_division=0)),
            "selective_recall": float(recall_score(y_true, (y_prob >= 0.5).astype(int), pos_label=1, zero_division=0)),
            "tau_clean": 0.5,
            "tau_defect": 0.5
        }
    return best_res

def eval_view_benchmarks(y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray, tn: float, td: float):
    view_res = {}
    for v_code, alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if not np.any(v_mask):
            continue
        y_v = y_true[v_mask]
        p_v = y_prob[v_mask]
        
        dec_mask = (p_v < tn) | (p_v >= td)
        cov = float(np.mean(dec_mask))
        y_dec = y_v[dec_mask]
        p_dec = (p_v[dec_mask] >= td).astype(int)
        
        f1_def = f1_score(y_dec, p_dec, pos_label=1, zero_division=0) if len(y_dec) > 0 else 0.0
        acc = accuracy_score(y_dec, p_dec) if len(y_dec) > 0 else 0.0
        prec = precision_score(y_dec, p_dec, pos_label=1, zero_division=0) if len(y_dec) > 0 else 0.0
        rec = recall_score(y_dec, p_dec, pos_label=1, zero_division=0) if len(y_dec) > 0 else 0.0
        auc = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 0.5
        
        view_res[v_code] = {
            "alias": alias,
            "total_samples": int(np.sum(v_mask)),
            "coverage": cov,
            "retake_rate": 1.0 - cov,
            "selective_defect_f1": float(f1_def * 100.0),
            "selective_accuracy": float(acc * 100.0),
            "selective_precision": float(prec * 100.0),
            "selective_recall": float(rec * 100.0),
            "roc_auc": float(auc * 100.0)
        }
    return view_res

def train_and_eval_config(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, config_name: str) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"[*] 훈련 및 평가 시작: {config_name} (Shape: {X.shape})")
    print(f"================================================================================")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y), dtype=np.float32)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    pos_weight = float(n_neg) / float(n_pos)
    
    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.03, max_depth=5, num_leaves=18,
            class_weight='balanced', random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        clf_cb = CatBoostClassifier(
            iterations=250, learning_rate=0.035, depth=5,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=-1
        )
        
        clf_lgb.fit(X_trn, y_trn)
        clf_cb.fit(X_trn, y_trn)
        
        p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        p_cb = clf_cb.predict_proba(X_val)[:, 1]
        p_val = 0.50 * p_lgb + 0.50 * p_cb
        oof_prob[val_idx] = p_val
        
    auc_total = float(roc_auc_score(y, oof_prob) * 100.0)
    bin_f1 = float(f1_score(y, (oof_prob >= 0.5).astype(int), pos_label=1, zero_division=0) * 100.0)
    print(f"[+] 5-Fold OOF 생성 완료 ({time.time() - t0:.1f}s) | Binary F1: {bin_f1:.2f}%, AUC: {auc_total:.2f}%")
    
    # 3-State Threshold Optimization
    best_3s = eval_3state_grid(y, oof_prob, min_coverage=0.85)
    tn, td = best_3s["tau_clean"], best_3s["tau_defect"]
    
    f1_3s = best_3s["selective_defect_f1"] * 100.0
    cov = best_3s["coverage"] * 100.0
    prec = best_3s["selective_precision"] * 100.0
    rec = best_3s["selective_recall"] * 100.0
    acc = best_3s["selective_accuracy"] * 100.0
    
    print(f"[+] 3-State 최적화 결과 (Coverage >= 85%):")
    print(f"    tau_clean: {tn:.4f}, tau_defect: {td:.4f}")
    print(f"    Selective Defect F1: {f1_3s:.2f}% | Coverage: {cov:.2f}% | Prec: {prec:.2f}% | Rec: {rec:.2f}%")
    
    # View Breakdown
    view_bench = eval_view_benchmarks(y, oof_prob, views, tn, td)
    print(f"    시점별 Defect F1:")
    for vc in ['view5', 'view2', 'view3', 'view4', 'view1']:
        print(f"      - {view_bench[vc]['alias']}: F1 = {view_bench[vc]['selective_defect_f1']:.2f}% (Cov: {view_bench[vc]['coverage']*100:.1f}%, AUC: {view_bench[vc]['roc_auc']:.1f}%)")
        
    return {
        "config_name": config_name,
        "features_dim": int(X.shape[1]),
        "binary_f1": bin_f1,
        "roc_auc": auc_total,
        "selective_defect_f1": f1_3s,
        "coverage": cov,
        "retake_rate": 100.0 - cov,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "tau_clean": tn,
        "tau_defect": td,
        "view_benchmarks": view_bench
    }

def main():
    while not os.path.exists(CACHE_NPZ):
        print("[*] 캐시 생성 완료 대기 중...", flush=True)
        time.sleep(3)
        
    print(f"[*] 캐시 로드: {CACHE_NPZ}", flush=True)
    d = np.load(CACHE_NPZ, allow_pickle=True)
    X_mp_24 = d["X_mp_24"]       # (1035, 24)
    X_rtmp_41 = d["X_rtmp_41"]   # (1035, 41)
    view_ids = d["view_ids"][:, None] # (1035, 1)
    view_oh = d["view_oh"]       # (1035, 5)
    y = d["y"]
    subjects = d["subjects"]
    views = d["views"]
    
    rtmp_names = list(d["rtmp_names_41"])
    # 26D 최적화 인덱스 매핑 (41D 중 23개 생체역학 피처 + 3개 시점 피처)
    rtmp_bio_23_names = [f for f in RTMP_OPT_26_NAMES if f not in ['view_code', 'v_oh_2', 'v_oh_4']]
    rtmp_opt_idx = [rtmp_names.index(f) for f in rtmp_bio_23_names]
    X_rtmp_23_opt = X_rtmp_41[:, rtmp_opt_idx] # (1035, 23)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. 단일 MediaPipe Baseline (24D)
    # -------------------------------------------------------------------------
    results["mediapipe_24d"] = train_and_eval_config(
        X_mp_24, y, subjects, views, "1. MediaPipe_3D_Only (24D)"
    )
    
    # -------------------------------------------------------------------------
    # 2. 단일 RTMPose Baseline (41D)
    # -------------------------------------------------------------------------
    results["rtmpose_41d"] = train_and_eval_config(
        X_rtmp_41, y, subjects, views, "2. RTMPose_2D_Only (41D)"
    )
    
    # -------------------------------------------------------------------------
    # 3. 단일 RTMPose Optimized Baseline (23D + ViewID = 24D)
    # -------------------------------------------------------------------------
    X_rtmp_opt_view = np.hstack([X_rtmp_23_opt, view_ids])
    results["rtmpose_opt_view_24d"] = train_and_eval_config(
        X_rtmp_opt_view, y, subjects, views, "3. RTMPose_Optimized + ViewID (24D)"
    )
    
    # -------------------------------------------------------------------------
    # 4. 하이브리드 결합 (MP 24D + RTMP 41D = 65D) - 시점 정보 미제공
    # -------------------------------------------------------------------------
    X_hyb_65d = np.hstack([X_mp_24, X_rtmp_41])
    results["hybrid_no_view_65d"] = train_and_eval_config(
        X_hyb_65d, y, subjects, views, "4. Hybrid (MP 24D + RTMP 41D) - No View Info (65D)"
    )
    
    # -------------------------------------------------------------------------
    # 5. [핵심 제안] sd1_hybrid_1: MP 24D + RTMP 41D + user_selected_view_id (66D)
    # -------------------------------------------------------------------------
    X_hyb_view_66d = np.hstack([X_mp_24, X_rtmp_41, view_ids])
    results["sd1_hybrid_1"] = train_and_eval_config(
        X_hyb_view_66d, y, subjects, views, "5. [TARGET] sd1_hybrid_1 (MP 24D + RTMP 41D + View-ID) (66D)"
    )
    
    # -------------------------------------------------------------------------
    # 6. 하이브리드 + One-Hot View Conditioning (MP 24D + RTMP 41D + View OH = 70D)
    # -------------------------------------------------------------------------
    X_hyb_view_oh_70d = np.hstack([X_mp_24, X_rtmp_41, view_oh])
    results["hybrid_view_oh_70d"] = train_and_eval_config(
        X_hyb_view_oh_70d, y, subjects, views, "6. Hybrid + View One-Hot (70D)"
    )
    
    # -------------------------------------------------------------------------
    # 7. 하이브리드 정예 최적화 (MP 24D + RTMP Opt 23D + View-ID = 48D)
    # -------------------------------------------------------------------------
    X_hyb_opt_view_48d = np.hstack([X_mp_24, X_rtmp_23_opt, view_ids])
    results["hybrid_opt_view_48d"] = train_and_eval_config(
        X_hyb_opt_view_48d, y, subjects, views, "7. Hybrid Optimized (MP 24D + RTMP Opt 23D + View-ID) (48D)"
    )
    
    # 저장
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 모든 토너먼트 평가 완료! 결과 저장: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()

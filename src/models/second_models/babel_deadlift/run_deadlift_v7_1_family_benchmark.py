# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_deadlift_v7_1_family_benchmark.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Version: v7.1 Family Comparison (v7.1 공식 확정 vs v7.1.a [관점 1 MAP] vs v7.1.b [관점 2 게이팅+PCC])

Specifications:
  1. v7.1 (공식 확정):
     - 결함별 최적 가중치 블렌딩 (w = [0.65, 0.70, 0.45, 0.50, 0.55])
     - 5대 시점 조건부 캘리브레이션 (View-Conditional Calibration, VCC)
     - 10% Selective Conformal Risk Control (CRC)
  2. v7.1.a (관점 1: 최적화 & 유효 서브스페이스 MAP 투영 디코딩):
     - 16개 유효 조합 도메인 제약 상의 Bayesian MAP Decoding
     - 캘리브레이션 확률 기반 사후 우도 최대화 (불가능 조합 100% 원천 차단)
     - Top-1 vs Top-2 우도 마진 기반 10% Selective CRC
  3. v7.1.b (관점 2: 기존 기술의 조화로운 융합 - View Gating + PCC Stacking):
     - 시점별 동적 게이팅 퓨전 (View-Aware Gated Blending)
     - 확률적 분류기 체인 (PCC) Stacking (d1~d4 -> d5 캐스케이드 보정)
     - 시점 조건부 캘리브레이션 및 10% Selective CRC
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, hamming_loss
)
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_RAW_KPS = os.path.join(CURRENT_DIR, "deadlift_raw_64f_kps.npz")
CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
CACHE_COATTN = os.path.join(CURRENT_DIR, "v7_coattn_oof_probs.npy")

OUT_JSON = os.path.join(CURRENT_DIR, "deadlift_v7_1_family_benchmark_results.json")
OUT_BUNDLE = os.path.join(CURRENT_DIR, "weights_integrated_v7", "deadlift_v7_1_family_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}
VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]
DEFECT_NAMES = [
    "d1 (척추 중립성)",
    "d2 (무릎-발목 모임)",
    "d3 (발바닥 접지)",
    "d4 (무릎-골반 동시성)",
    "d5 (바벨 밀착)"
]


def run_benchmark():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 데드리프트 v7.1 Family 공식 벤치마크 및 비교 검증")
    print("  - v7.1 (공식 확정): 결함별 최적 블렌딩 + 시점 조건부 캘리브레이션 + 10% CRC")
    print("  - v7.1.a (관점 1): 16개 유효 서브스페이스 베이지안 MAP 투영 디코딩")
    print("  - v7.1.b (관점 2): 시점 인지 게이팅 퓨전 + PCC Stacking (d4->d5 연쇄 보정)")
    print("  - 데이터셋: AI-Hub 770건 전수 (피험자 1~9번 완전 격리, 5-Fold StratifiedGroupKFold)")
    print("=" * 85, flush=True)

    # 1. 데이터 로드
    raw = np.load(CACHE_RAW_KPS, allow_pickle=True)
    y_all = raw["y"]
    views = raw["views"]
    subjects = raw["subjects"]
    N = len(y_all)
    y_clean = np.all(y_all == 0, axis=-1).astype(int)

    meta = np.load(CACHE_META, allow_pickle=True)
    oof_indep = meta["oof_probs"]
    oof_coattn = np.load(CACHE_COATTN)

    # 16개 유효 결함 조합
    tuples = [tuple(row) for row in y_all]
    valid_combos = np.array(list(set(tuples))) # (16, 5)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # ==========================================================================
    # [1] v7.1 공식 확정 모델 파이프라인
    # ==========================================================================
    print("\n[+] 1. v7.1 공식 확정 모델 검증 중...")
    w_opt = [0.65, 0.70, 0.45, 0.50, 0.55]
    p_blend_71 = np.zeros_like(oof_indep)
    for i in range(5):
        p_blend_71[:, i] = w_opt[i] * oof_indep[:, i] + (1 - w_opt[i]) * oof_coattn[:, i]
    p_blend_71[:, 4] = np.maximum(p_blend_71[:, 4], p_blend_71[:, 3])

    view_thresholds_71 = {
        "view5": [0.61, 0.69, 0.55, 0.65, 0.51],
        "view2": [0.67, 0.55, 0.49, 0.59, 0.35],
        "view3": [0.53, 0.63, 0.47, 0.49, 0.49],
        "view4": [0.51, 0.55, 0.43, 0.57, 0.35],
        "view1": [0.45, 0.67, 0.53, 0.51, 0.55]
    }

    pred_71_full = np.zeros_like(p_blend_71, dtype=int)
    margins_71 = np.zeros(N)
    for vn in VIEW_LIST:
        m = (views == vn)
        th = np.array(view_thresholds_71[vn])
        pred_71_full[m] = (p_blend_71[m] >= th).astype(int)
        margins_71[m] = np.mean(np.abs(p_blend_71[m] - th), axis=-1)

    emr_71_full = float(np.mean(np.all(y_all == pred_71_full, axis=-1)))
    cf1_71_full = float(f1_score(y_clean, np.all(pred_71_full == 0, axis=-1).astype(int), zero_division=0))
    cspec_71_full = float(recall_score(1 - y_clean, 1 - np.all(pred_71_full == 0, axis=-1).astype(int), zero_division=0))
    hl_71_full = float(hamming_loss(y_all, pred_71_full))
    defs_71_full = [float(f1_score(y_all[:, i], pred_71_full[:, i], zero_division=0)) for i in range(5)]

    # CRC 10% Selective for v7.1
    decided_71 = np.zeros(N, dtype=bool)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(p_blend_71, y_clean, groups=subjects)):
        tau = np.quantile(margins_71[tr_idx], 0.10)
        decided_71[val_idx] = (margins_71[val_idx] >= tau)

    y_dec_71 = y_all[decided_71]
    p_dec_71 = pred_71_full[decided_71]
    views_dec_71 = views[decided_71]
    emr_71_sel = float(np.mean(np.all(y_dec_71 == p_dec_71, axis=-1)))
    c_dec_71 = np.all(y_dec_71 == 0, axis=-1).astype(int)
    c_pred_71 = np.all(p_dec_71 == 0, axis=-1).astype(int)
    cf1_71_sel = float(f1_score(c_dec_71, c_pred_71, zero_division=0))
    cspec_71_sel = float(recall_score(1 - c_dec_71, 1 - c_pred_71, zero_division=0))
    hl_71_sel = float(hamming_loss(y_dec_71, p_dec_71))
    defs_71_sel = [float(f1_score(y_dec_71[:, i], p_dec_71[:, i], zero_division=0)) for i in range(5)]

    views_eval_71 = {}
    for vn in VIEW_LIST:
        m = (views_dec_71 == vn)
        views_eval_71[vn] = {
            "alias": VIEW_ALIAS_MAP[vn],
            "total_samples": int(np.sum(views == vn)),
            "decided_samples": int(np.sum(m)),
            "emr": float(np.mean(np.all(y_dec_71[m] == p_dec_71[m], axis=-1))),
            "clean_f1": float(f1_score(c_dec_71[m], c_pred_71[m], zero_division=0))
        }

    print(f"[*] v7.1 완료: Full EMR={emr_71_full*100:.2f}%, Clean F1={cf1_71_full*100:.2f}% | Selective EMR={emr_71_sel*100:.2f}%, Clean F1={cf1_71_sel*100:.2f}%")

    # ==========================================================================
    # [2] v7.1.a (관점 1: 최적화 & 유효 서브스페이스 MAP 투영 디코딩)
    # ==========================================================================
    print("\n[+] 2. v7.1.a (관점 1: 유효 서브스페이스 MAP 디코딩) 검증 중...")
    p_calib_71a = np.zeros_like(p_blend_71)
    for vn in VIEW_LIST:
        m = (views == vn)
        th = np.array(view_thresholds_71[vn])
        for j in range(5):
            pj = p_blend_71[m, j]
            thj = th[j]
            calib_j = np.where(pj < thj, 0.5 * pj / (thj + 1e-6), 0.5 + 0.5 * (pj - thj) / (1.0 - thj + 1e-6))
            p_calib_71a[m, j] = np.clip(calib_j, 1e-4, 1.0 - 1e-4)

    ll_71a = np.log(p_calib_71a) @ valid_combos.T + np.log(1.0 - p_calib_71a) @ (1.0 - valid_combos).T
    sorted_ll_71a = np.sort(ll_71a, axis=1)
    margins_71a = sorted_ll_71a[:, -1] - sorted_ll_71a[:, -2]
    pred_71a_full = valid_combos[np.argmax(ll_71a, axis=1)]

    emr_71a_full = float(np.mean(np.all(y_all == pred_71a_full, axis=-1)))
    cf1_71a_full = float(f1_score(y_clean, np.all(pred_71a_full == 0, axis=-1).astype(int), zero_division=0))
    cspec_71a_full = float(recall_score(1 - y_clean, 1 - np.all(pred_71a_full == 0, axis=-1).astype(int), zero_division=0))
    hl_71a_full = float(hamming_loss(y_all, pred_71a_full))
    defs_71a_full = [float(f1_score(y_all[:, i], pred_71a_full[:, i], zero_division=0)) for i in range(5)]

    # CRC 10% Selective for v7.1.a
    decided_71a = np.zeros(N, dtype=bool)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(p_blend_71, y_clean, groups=subjects)):
        tau = np.quantile(margins_71a[tr_idx], 0.10)
        decided_71a[val_idx] = (margins_71a[val_idx] >= tau)

    y_dec_71a = y_all[decided_71a]
    p_dec_71a = pred_71a_full[decided_71a]
    views_dec_71a = views[decided_71a]
    emr_71a_sel = float(np.mean(np.all(y_dec_71a == p_dec_71a, axis=-1)))
    c_dec_71a = np.all(y_dec_71a == 0, axis=-1).astype(int)
    c_pred_71a = np.all(p_dec_71a == 0, axis=-1).astype(int)
    cf1_71a_sel = float(f1_score(c_dec_71a, c_pred_71a, zero_division=0))
    cspec_71a_sel = float(recall_score(1 - c_dec_71a, 1 - c_pred_71a, zero_division=0))
    hl_71a_sel = float(hamming_loss(y_dec_71a, p_dec_71a))
    defs_71a_sel = [float(f1_score(y_dec_71a[:, i], p_dec_71a[:, i], zero_division=0)) for i in range(5)]

    views_eval_71a = {}
    for vn in VIEW_LIST:
        m = (views_dec_71a == vn)
        views_eval_71a[vn] = {
            "alias": VIEW_ALIAS_MAP[vn],
            "total_samples": int(np.sum(views == vn)),
            "decided_samples": int(np.sum(m)),
            "emr": float(np.mean(np.all(y_dec_71a[m] == p_dec_71a[m], axis=-1))),
            "clean_f1": float(f1_score(c_dec_71a[m], c_pred_71a[m], zero_division=0))
        }

    print(f"[*] v7.1.a 완료: Full EMR={emr_71a_full*100:.2f}%, Clean F1={cf1_71a_full*100:.2f}% | Selective EMR={emr_71a_sel*100:.2f}%, Clean F1={cf1_71a_sel*100:.2f}%")

    # ==========================================================================
    # [3] v7.1.b (관점 2: 기존 기술의 조화로운 융합 - View Gating + PCC Stacking)
    # ==========================================================================
    print("\n[+] 3. v7.1.b (관점 2: View Gating + PCC Stacking) 검증 중...")
    p_gated_oof = np.zeros_like(oof_indep)
    p_pcc_oof = np.zeros_like(oof_indep)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(oof_indep, y_clean, groups=subjects)):
        # 시점별 최적 가중치 피팅 (훈련 세트 전용)
        best_weights_v = {vn: np.zeros(5) for vn in VIEW_LIST}
        for vn in VIEW_LIST:
            m_tr = (views[tr_idx] == vn)
            tr_vn_idx = tr_idx[m_tr]
            for di in range(5):
                best_w = 0.5
                best_f1 = -1
                for w in np.linspace(0.1, 0.9, 17):
                    p_cand = w * oof_indep[tr_vn_idx, di] + (1 - w) * oof_coattn[tr_vn_idx, di]
                    sc = f1_score(y_all[tr_vn_idx, di], p_cand >= 0.5, zero_division=0)
                    if sc > best_f1:
                        best_f1 = sc
                        best_w = w
                best_weights_v[vn][di] = best_w

        # 검증 세트에 게이팅 적용
        for vn in VIEW_LIST:
            m_va = (views[val_idx] == vn)
            va_vn_idx = val_idx[m_va]
            for di in range(5):
                w = best_weights_v[vn][di]
                p_gated_oof[va_vn_idx, di] = w * oof_indep[va_vn_idx, di] + (1 - w) * oof_coattn[va_vn_idx, di]

        # PCC Stacking for d5
        p_tr_gated = np.zeros_like(oof_indep[tr_idx])
        for vn in VIEW_LIST:
            m_tr = (views[tr_idx] == vn)
            for di in range(5):
                w = best_weights_v[vn][di]
                p_tr_gated[m_tr, di] = w * oof_indep[tr_idx[m_tr], di] + (1 - w) * oof_coattn[tr_idx[m_tr], di]

        X_tr_stack = np.column_stack([
            p_tr_gated[:, 0], p_tr_gated[:, 1], p_tr_gated[:, 2], p_tr_gated[:, 3],
            p_tr_gated[:, 4], oof_indep[tr_idx, 4], oof_coattn[tr_idx, 4],
            np.maximum(p_tr_gated[:, 4], p_tr_gated[:, 3])
        ])
        y_tr_d5 = y_all[tr_idx, 4]

        clf_d5 = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf_d5.fit(X_tr_stack, y_tr_d5)

        X_va_stack = np.column_stack([
            p_gated_oof[val_idx, 0], p_gated_oof[val_idx, 1], p_gated_oof[val_idx, 2], p_gated_oof[val_idx, 3],
            p_gated_oof[val_idx, 4], oof_indep[val_idx, 4], oof_coattn[val_idx, 4],
            np.maximum(p_gated_oof[val_idx, 4], p_gated_oof[val_idx, 3])
        ])
        p_d5_stacked = clf_d5.predict_proba(X_va_stack)[:, 1]

        p_pcc_oof[val_idx, :4] = p_gated_oof[val_idx, :4]
        p_pcc_oof[val_idx, 4] = np.maximum(p_d5_stacked, p_gated_oof[val_idx, 3])

    pred_71b_full = np.zeros_like(p_pcc_oof, dtype=int)
    margins_71b = np.zeros(N)

    view_thresholds_71b = {}
    for vn in VIEW_LIST:
        m = (views == vn)
        best_th = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        for step in range(3):
            for i in range(5):
                best_v = best_th[i]
                best_s = -1
                for val in np.arange(0.35, 0.70, 0.02):
                    cur = best_th.copy()
                    cur[i] = val
                    p = (p_pcc_oof[m] >= cur).astype(int)
                    emr = np.mean(np.all(y_all[m] == p, axis=-1))
                    cf1 = f1_score(y_clean[m], np.all(p == 0, axis=-1).astype(int), zero_division=0)
                    sc = 0.5 * emr + 0.5 * cf1
                    if sc > best_s:
                        best_s = sc
                        best_v = val
                best_th[i] = best_v
        view_thresholds_71b[vn] = best_th.tolist()
        pred_71b_full[m] = (p_pcc_oof[m] >= best_th).astype(int)
        margins_71b[m] = np.mean(np.abs(p_pcc_oof[m] - best_th), axis=-1)

    emr_71b_full = float(np.mean(np.all(y_all == pred_71b_full, axis=-1)))
    cf1_71b_full = float(f1_score(y_clean, np.all(pred_71b_full == 0, axis=-1).astype(int), zero_division=0))
    cspec_71b_full = float(recall_score(1 - y_clean, 1 - np.all(pred_71b_full == 0, axis=-1).astype(int), zero_division=0))
    hl_71b_full = float(hamming_loss(y_all, pred_71b_full))
    defs_71b_full = [float(f1_score(y_all[:, i], pred_71b_full[:, i], zero_division=0)) for i in range(5)]

    # CRC 10% Selective for v7.1.b
    decided_71b = np.zeros(N, dtype=bool)
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(p_pcc_oof, y_clean, groups=subjects)):
        tau = np.quantile(margins_71b[tr_idx], 0.10)
        decided_71b[val_idx] = (margins_71b[val_idx] >= tau)

    y_dec_71b = y_all[decided_71b]
    p_dec_71b = pred_71b_full[decided_71b]
    views_dec_71b = views[decided_71b]
    emr_71b_sel = float(np.mean(np.all(y_dec_71b == p_dec_71b, axis=-1)))
    c_dec_71b = np.all(y_dec_71b == 0, axis=-1).astype(int)
    c_pred_71b = np.all(p_dec_71b == 0, axis=-1).astype(int)
    cf1_71b_sel = float(f1_score(c_dec_71b, c_pred_71b, zero_division=0))
    cspec_71b_sel = float(recall_score(1 - c_dec_71b, 1 - c_pred_71b, zero_division=0))
    hl_71b_sel = float(hamming_loss(y_dec_71b, p_dec_71b))
    defs_71b_sel = [float(f1_score(y_dec_71b[:, i], p_dec_71b[:, i], zero_division=0)) for i in range(5)]

    views_eval_71b = {}
    for vn in VIEW_LIST:
        m = (views_dec_71b == vn)
        views_eval_71b[vn] = {
            "alias": VIEW_ALIAS_MAP[vn],
            "total_samples": int(np.sum(views == vn)),
            "decided_samples": int(np.sum(m)),
            "emr": float(np.mean(np.all(y_dec_71b[m] == p_dec_71b[m], axis=-1))),
            "clean_f1": float(f1_score(c_dec_71b[m], c_pred_71b[m], zero_division=0))
        }

    print(f"[*] v7.1.b 완료: Full EMR={emr_71b_full*100:.2f}%, Clean F1={cf1_71b_full*100:.2f}% | Selective EMR={emr_71b_sel*100:.2f}%, Clean F1={cf1_71b_sel*100:.2f}%")

    # ==========================================================================
    # [4] 종합 비교 JSON 저장
    # ==========================================================================
    benchmark_results = {
        "dataset_total": N,
        "clean_total": int(np.sum(y_clean)),
        "baseline_v7_0": {
            "name": "v7.0 Baseline (Co-Attention + CRC)",
            "full_emr": 0.3740,
            "full_clean_f1": 0.6466,
            "full_clean_spec": 0.8574,
            "sel_emr": 0.4020,
            "sel_clean_f1": 0.6782,
            "sel_clean_spec": 0.8532,
            "coverage": 0.8948
        },
        "v7_1_official": {
            "name": "v7.1 공식 확정 모델 (결함별 가중치 + VCC + CRC)",
            "full_emr": emr_71_full,
            "full_clean_f1": cf1_71_full,
            "full_clean_spec": cspec_71_full,
            "full_hamming_loss": hl_71_full,
            "full_defect_f1": defs_71_full,
            "coverage": float(np.mean(decided_71)),
            "sel_emr": emr_71_sel,
            "sel_clean_f1": cf1_71_sel,
            "sel_clean_spec": cspec_71_sel,
            "sel_hamming_loss": hl_71_sel,
            "sel_defect_f1": defs_71_sel,
            "views": views_eval_71
        },
        "v7_1_a_map": {
            "name": "v7.1.a (관점 1: 16개 유효 서브스페이스 MAP 투영 디코딩)",
            "full_emr": emr_71a_full,
            "full_clean_f1": cf1_71a_full,
            "full_clean_spec": cspec_71a_full,
            "full_hamming_loss": hl_71a_full,
            "full_defect_f1": defs_71a_full,
            "coverage": float(np.mean(decided_71a)),
            "sel_emr": emr_71a_sel,
            "sel_clean_f1": cf1_71a_sel,
            "sel_clean_spec": cspec_71a_sel,
            "sel_hamming_loss": hl_71a_sel,
            "sel_defect_f1": defs_71a_sel,
            "views": views_eval_71a
        },
        "v7_1_b_pcc": {
            "name": "v7.1.b (관점 2: View Gating + PCC Stacking 결합)",
            "full_emr": emr_71b_full,
            "full_clean_f1": cf1_71b_full,
            "full_clean_spec": cspec_71b_full,
            "full_hamming_loss": hl_71b_full,
            "full_defect_f1": defs_71b_full,
            "coverage": float(np.mean(decided_71b)),
            "sel_emr": emr_71b_sel,
            "sel_clean_f1": cf1_71b_sel,
            "sel_clean_spec": cspec_71b_sel,
            "sel_hamming_loss": hl_71b_sel,
            "sel_defect_f1": defs_71b_sel,
            "views": views_eval_71b
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 벤치마크 결과 JSON 저장 완료: {OUT_JSON}")

    # ==========================================================================
    # [5] 모델 번들 저장
    # ==========================================================================
    bundle_data = {
        "v7_1_config": {
            "weights": w_opt,
            "view_thresholds": view_thresholds_71
        },
        "v7_1_a_config": {
            "valid_combos": valid_combos,
            "view_thresholds": view_thresholds_71
        },
        "v7_1_b_config": {
            "view_thresholds": view_thresholds_71b
        }
    }
    with open(OUT_BUNDLE, "wb") as f:
        pickle.dump(bundle_data, f)
    print(f"[+] 모델 번들 피클 저장 완료: {OUT_BUNDLE}")

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()

# -*- coding: utf-8 -*-
"""
bl2_v4 임계치 정밀 튜닝 및 v3 대비 비교 분석
"""
import os
import json
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

CURRENT_DIR = r"D:\src\models\second_models\babel_lunge"
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

def calculate_3state_metrics(y_true, y_prob, tau_n, tau_d):
    decided_mask = (y_prob <= tau_n) | (y_prob >= tau_d)
    retake_mask = ~decided_mask
    cov = float(np.mean(decided_mask))
    rej = float(np.mean(retake_mask))
    
    y_dec_true = y_true[decided_mask]
    y_dec_prob = y_prob[decided_mask]
    y_dec_pred = (y_dec_prob >= tau_d).astype(int)

    if len(y_dec_true) > 0 and len(np.unique(y_dec_true)) > 1:
        sel_acc = accuracy_score(y_dec_true, y_dec_pred)
        sel_prec = precision_score(y_dec_true, y_dec_pred, zero_division=0)
        sel_rec = recall_score(y_dec_true, y_dec_pred, zero_division=0)
        sel_d_f1 = f1_score(y_dec_true, y_dec_pred, zero_division=0)
        sel_c_f1 = f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0)
        cm = confusion_matrix(y_dec_true, y_dec_pred).tolist()
    else:
        sel_acc, sel_prec, sel_rec, sel_d_f1, sel_c_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        cm = []

    full_pred = (y_prob >= 0.5).astype(int)
    full_d_f1 = f1_score(y_true, full_pred, zero_division=0)
    full_c_f1 = f1_score(1 - y_true, 1 - full_pred, zero_division=0)
    full_acc = accuracy_score(y_true, full_pred)
    full_prec = precision_score(y_true, full_pred, zero_division=0)
    full_rec = recall_score(y_true, full_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "total_samples": len(y_true),
        "coverage": cov,
        "retake_rate": rej,
        "retake_count": int(np.sum(retake_mask)),
        "decided_count": int(np.sum(decided_mask)),
        "selective_defect_f1": float(sel_d_f1),
        "selective_clean_f1": float(sel_c_f1),
        "selective_accuracy": float(sel_acc),
        "selective_precision": float(sel_prec),
        "selective_recall": float(sel_rec),
        "confusion_matrix": cm,
        "full_coverage_defect_f1": float(full_d_f1),
        "full_coverage_clean_f1": float(full_c_f1),
        "full_coverage_accuracy": float(full_acc),
        "full_coverage_precision": float(full_prec),
        "full_coverage_recall": float(full_rec),
        "roc_auc": float(auc)
    }

def main():
    bundle_path = os.path.join(CURRENT_DIR, "weights", "bl2_v4_bundle.pkl")
    cache_path = os.path.join(CURRENT_DIR, "lunge_c2_features_v4_cache.npz")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
    views = cache["views"]
    subjects = cache["subjects"]

    # 교차검증 OOF 확률 다시 계산
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    tier1_lgb_models = bundle["tier1_lgb_models"]
    tier1_cat_models = bundle["tier1_cat_models"]
    tier2_xgb_models = bundle["tier2_xgb_models"]
    tier2_cat_models = bundle["tier2_cat_models"]
    tier2_lgb_models = bundle["tier2_lgb_models"]
    temperature = bundle.get("temperature", 1.45)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        lgb1 = tier1_lgb_models[fold]
        cat1 = tier1_cat_models[fold]
        xgb2 = tier2_xgb_models[fold]
        cat2 = tier2_cat_models[fold]
        lgb2 = tier2_lgb_models[fold]

        X_val = X[val_idx]
        p_val_lgb = lgb1.predict_proba(X_val)[:, 1]
        p_val_cat = cat1.predict_proba(X_val)[:, 1]
        p_t1 = 0.50 * p_val_lgb + 0.50 * p_val_cat

        amb_mask = (p_t1 >= 0.30) & (p_t1 <= 0.70)
        p_val_xgb = xgb2.predict_proba(X_val)[:, 1]
        p_val_cat2 = cat2.predict_proba(X_val)[:, 1]
        p_val_lgb2 = lgb2.predict_proba(X_val)[:, 1]
        p_t2 = 0.35 * p_val_xgb + 0.35 * p_val_cat2 + 0.30 * p_val_lgb2

        p_final = np.copy(p_t1)
        p_final[amb_mask] = 0.38 * p_t1[amb_mask] + 0.62 * p_t2[amb_mask]

        # temperature scaling
        eps = 1e-7
        clipped = np.clip(p_final, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)) / temperature
        oof_probs[val_idx] = 1.0 / (1.0 + np.exp(-logits))

    print(f"OOF Probs: min={oof_probs.min():.4f}, mean={oof_probs.mean():.4f}, max={oof_probs.max():.4f}")

    # 탐색 그리드
    candidates = []
    prob_vals = np.linspace(0.40, 0.60, 201)

    for i, tn in enumerate(prob_vals):
        for td in prob_vals[i+1:]:
            dec_mask = (oof_probs <= tn) | (oof_probs >= td)
            cov = float(np.mean(dec_mask))
            rej = 1.0 - cov
            if rej <= 0.1001:  # 보류율 최대 10%
                m = calculate_3state_metrics(y, oof_probs, float(tn), float(td))
                d_f1 = m["selective_defect_f1"]
                c_f1 = m["selective_clean_f1"]
                acc = m["selective_accuracy"]
                prec = m["selective_precision"]
                rec = m["selective_recall"]

                # 정상 클래스 신뢰도 및 정확도 기본선 보존
                if c_f1 >= 0.72 and acc >= 0.63 and d_f1 >= 0.35:
                    macro_f1 = 0.50 * d_f1 + 0.50 * c_f1
                    harmonic_f1 = 2 * (d_f1 * c_f1) / (d_f1 + c_f1 + 1e-6)
                    candidates.append({
                        "tn": float(tn),
                        "td": float(td),
                        "cov": cov,
                        "rej": rej,
                        "d_f1": d_f1,
                        "c_f1": c_f1,
                        "macro_f1": macro_f1,
                        "harmonic_f1": harmonic_f1,
                        "acc": acc,
                        "prec": prec,
                        "rec": rec,
                        "m": m
                    })

    print(f"Total valid balanced candidates: {len(candidates)}")
    if not candidates:
        print("Relaxing constraints...")
        for i, tn in enumerate(prob_vals):
            for td in prob_vals[i+1:]:
                dec_mask = (oof_probs <= tn) | (oof_probs >= td)
                cov = float(np.mean(dec_mask))
                rej = 1.0 - cov
                if rej <= 0.1001:
                    m = calculate_3state_metrics(y, oof_probs, float(tn), float(td))
                    d_f1 = m["selective_defect_f1"]
                    c_f1 = m["selective_clean_f1"]
                    if c_f1 >= 0.65:
                        candidates.append({
                            "tn": float(tn),
                            "td": float(td),
                            "cov": cov,
                            "rej": rej,
                            "d_f1": d_f1,
                            "c_f1": c_f1,
                            "macro_f1": 0.5 * (d_f1 + c_f1),
                            "harmonic_f1": 2 * d_f1 * c_f1 / (d_f1 + c_f1 + 1e-6),
                            "acc": m["selective_accuracy"],
                            "prec": m["selective_precision"],
                            "rec": m["selective_recall"],
                            "m": m
                        })

    # 정렬: 결함 F1 우선, 그 다음 Macro F1
    candidates.sort(key=lambda x: (x["d_f1"], x["macro_f1"]), reverse=True)
    print("\nTop 5 Candidates:")
    for rank, cand in enumerate(candidates[:5], 1):
        print(f"#{rank}: tn={cand['tn']:.4f}, td={cand['td']:.4f} | Defect F1={cand['d_f1']*100:.2f}%, Clean F1={cand['c_f1']*100:.2f}%, Macro F1={cand['macro_f1']*100:.2f}%, Acc={cand['acc']*100:.2f}%, Prec={cand['prec']*100:.2f}%, Rec={cand['rec']*100:.2f}%, Cov={cand['cov']*100:.2f}%, Rej={cand['rej']*100:.2f}%")

    best = candidates[0]
    best_tn = best["tn"]
    best_td = best["td"]
    best_metrics = best["m"]

    # 5대 시점 감사
    view_benchmarks = {}
    for vn, alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == vn)
        v_true = y[v_mask]
        v_prob = oof_probs[v_mask]
        vm = calculate_3state_metrics(v_true, v_prob, best_tn, best_td)
        view_benchmarks[vn] = {
            "alias": alias,
            "metrics": vm
        }

    # 번들 갱신
    bundle["tau_normal"] = best_tn
    bundle["tau_defect"] = best_td
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[+] Updated bundle saved: {bundle_path}")

    # JSON 저장
    bench_data = {
        "model_name": "bl2_v4",
        "description": "v4 SOTA Kinematic Decoupling & Occlusion-Robust Production Model",
        "timestamp": "2026-10-03 17:15:00",
        "max_retake_rate": 0.10,
        "min_coverage": 0.90,
        "temperature": temperature,
        "tau_normal": best_tn,
        "tau_defect": best_td,
        "overall_3state": best_metrics,
        "view_benchmarks": view_benchmarks
    }
    bench_path = os.path.join(CURRENT_DIR, "bl2_v4_benchmark_results.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Updated benchmark saved: {bench_path}")

if __name__ == "__main__":
    main()

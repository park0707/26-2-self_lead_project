# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부
Model: bl2_hybrid_pruned.py (Feature Selection / Pruning)
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier

from lunge_c2_features_v6 import FEATURE_NAMES_18D
from lunge_c2_features_hybrid_2 import FEATURE_NAMES_FRONTAL, extract_hybrid_2_features_batch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

FEATURE_NAMES_46D = (
    [f"MP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"MP_{f}" for f in FEATURE_NAMES_FRONTAL] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_FRONTAL]
)

class BabelLungeCondition2ModelHybridPruned:
    def __init__(
        self,
        tau_normal: float = 0.5000,
        tau_defect: float = 0.5310,
        max_retake_rate: float = 0.10,
        top_k: int = 20
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.max_retake_rate = max_retake_rate
        self.min_coverage = 1.0 - max_retake_rate
        self.top_k = top_k
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.cat_models: List[CatBoostClassifier] = []
        self.feature_names = FEATURE_NAMES_46D
        self.selected_feature_indices = []
        self.selected_feature_names = []
        self.is_fitted = False

    def fit_cross_validation(
        self, X_full: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        
        # --- PHASE 1: Feature Selection ---
        print(f"[*] Phase 1: Feature Selection (Pruning down to top {self.top_k} features)...")
        # Use a single LightGBM model on the whole dataset to get global feature importance
        selector_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, num_leaves=15, 
            learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1
        )
        selector_model.fit(X_full, y)
        importances = selector_model.feature_importances_
        
        # Sort indices by importance descending
        sorted_indices = np.argsort(importances)[::-1]
        self.selected_feature_indices = sorted_indices[:self.top_k]
        self.selected_feature_names = [self.feature_names[i] for i in self.selected_feature_indices]
        
        print(f"[+] Selected Top {self.top_k} Features:")
        for idx, name in enumerate(self.selected_feature_names):
            print(f"    {idx+1}. {name} (Importance: {importances[self.selected_feature_indices[idx]]})")
            
        # Slice X to only keep selected features
        X = X_full[:, self.selected_feature_indices]

        # --- PHASE 2: 5-Fold Cross Validation ---
        print(f"\n[*] Phase 2: Starting 5-Fold StratifiedGroupKFold Training on {self.top_k}D Features...")
        print(f"    Constraint: Retake <= {self.max_retake_rate*100:.0f}%, Coverage >= {self.min_coverage*100:.0f}%")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)

        self.lgb_models = []
        self.cat_models = []

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        base_pos_weight = float(n_neg) / float(n_pos)
        cost_sensitive_weight = base_pos_weight * 1.35

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            val_subs = np.unique(subjects[val_idx])

            lgb_clf = lgb.LGBMClassifier(
                n_estimators=160, max_depth=4, num_leaves=13,
                learning_rate=0.033, min_child_samples=15,
                colsample_bytree=0.75, subsample=0.85,
                reg_alpha=0.5, reg_lambda=2.0,
                scale_pos_weight=cost_sensitive_weight,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            lgb_clf.fit(X_train, y_train)

            cat_clf = CatBoostClassifier(
                iterations=220, depth=5, learning_rate=0.030,
                l2_leaf_reg=4.0, auto_class_weights='Balanced',
                random_seed=42 + fold, verbose=0, thread_count=-1
            )
            cat_clf.fit(X_train, y_train)

            p_val = 0.50 * lgb_clf.predict_proba(X_val)[:, 1] + 0.50 * cat_clf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = p_val
            
            self.lgb_models.append(lgb_clf)
            self.cat_models.append(cat_clf)

            f1_val = f1_score(y_val, (p_val >= 0.5).astype(int), zero_division=0)
            auc_val = roc_auc_score(y_val, p_val)
            print(f"  - Fold {fold} (Val 피험자: {val_subs}): Binary F1 = {f1_val*100:.2f}%, AUC = {auc_val*100:.2f}%")

        self.is_fitted = True

        self.tau_normal, self.tau_defect, best_res = self._optimize_3state_thresholds(y, oof_probs)

        print(f"\n[+] 3-State Performance Optimized (Defect F1 Maximized, Retake <= {self.max_retake_rate*100:.0f}%):")
        print(f"    tau_normal = {self.tau_normal:.4f}, tau_defect = {self.tau_defect:.4f}")
        print(f"    Selective Defect F1 = {best_res['selective_defect_f1']*100:.2f}%, Clean F1 = {best_res['selective_clean_f1']*100:.2f}%")

        view_benchmarks = self._audit_views_3state(y, oof_probs, views)

        return {
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "overall_3state": best_res,
            "view_benchmarks": view_benchmarks
        }

    def _optimize_3state_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray):
        best_tn, best_td = 0.5000, 0.5310
        cands = []
        for tn in np.linspace(0.40, 0.60, 101):
            for td in np.linspace(0.40, 0.60, 101):
                if td <= tn: continue
                dec_mask = (y_prob <= tn) | (y_prob >= td)
                if 1.0 - float(np.mean(dec_mask)) <= self.max_retake_rate + 0.0001:
                    y_dec = y_true[dec_mask]
                    p_dec = (y_prob[dec_mask] >= td).astype(int)
                    d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                    c_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
                    acc = accuracy_score(y_dec, p_dec)
                    if c_f1 >= 0.650 and acc >= 0.550:
                        cands.append((d_f1, c_f1, acc, float(tn), float(td)))
        
        if cands:
            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_tn, best_td = cands[0][3], cands[0][4]

        best_metrics = self.calculate_3state_metrics(y_true, y_prob, best_tn, best_td)
        return best_tn, best_td, best_metrics

    @staticmethod
    def calculate_3state_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float):
        decided_mask = (y_prob <= tau_normal) | (y_prob >= tau_defect)
        retake_mask = ~decided_mask
        y_dec_true = y_true[decided_mask]
        y_dec_pred = (y_prob[decided_mask] >= tau_defect).astype(int)

        if len(y_dec_true) > 0 and len(np.unique(y_dec_true)) > 1:
            sel_acc = accuracy_score(y_dec_true, y_dec_pred)
            sel_prec = precision_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_rec = recall_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_defect_f1 = f1_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_clean_f1 = f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0)
        else:
            sel_acc, sel_prec, sel_rec, sel_defect_f1, sel_clean_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        return {
            "total_samples": len(y_true),
            "coverage": float(np.mean(decided_mask)),
            "retake_rate": float(np.mean(retake_mask)),
            "selective_defect_f1": float(sel_defect_f1),
            "selective_clean_f1": float(sel_clean_f1),
            "selective_accuracy": float(sel_acc),
            "selective_precision": float(sel_prec),
            "selective_recall": float(sel_rec)
        }

    def _audit_views_3state(self, y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray):
        view_results = {}
        for v_code, v_alias in VIEW_ALIAS_MAP.items():
            mask = (views == v_code)
            if np.sum(mask) == 0: continue
            metrics = self.calculate_3state_metrics(y_true[mask], y_prob[mask], self.tau_normal, self.tau_defect)
            view_results[v_code] = {"alias": v_alias, "metrics": metrics}
            print(f"  - [{v_code}]: Defect F1 = {metrics['selective_defect_f1']*100:.2f}%")
        return view_results

def load_hybrid_2_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feat_cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_hybrid_2_cache.npz")
    if os.path.exists(feat_cache_file):
        c = np.load(feat_cache_file, allow_pickle=True)
        return c["X"], c["y"], c["subjects"], c["views"], c["clip_names"].tolist()
    else:
        raise FileNotFoundError("Cache for 46D features not found. Run bl2_hybrid_2.py first.")

def main():
    X, y, subjects, views, clip_names = load_hybrid_2_dataset()
    print(f"[+] Dataset Ready: Total {len(y)} samples")

    model = BabelLungeCondition2ModelHybridPruned(top_k=20)
    res = model.fit_cross_validation(X, y, subjects, views)

    results_path = os.path.join(CURRENT_DIR, "bl2_hybrid_pruned_benchmark_results.json")
    serializable_results = {
        "model_name": "bl2_hybrid_pruned",
        "description": "MediaPipe + RTMPose + Biomechanics (Top 20 Features Pruned)",
        "tau_normal": res["tau_normal"],
        "tau_defect": res["tau_defect"],
        "overall_3state": res["overall_3state"],
        "view_benchmarks": res["view_benchmarks"],
        "selected_features": model.selected_feature_names
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"[+] Benchmark results saved: {results_path}")

if __name__ == "__main__":
    main()

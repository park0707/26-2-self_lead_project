# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Model: bl2_v5.py (v5 Pure Alignment & Noise-Free Production Model)

주요 혁신 및 최적화 포인트:
1. [상체 보상 노이즈 완전 제거]:
   - 바벨 중량 밸런싱으로 인해 정상인도 크게 발생하는 상체 횡방향 기울기/보상 흔들림 3종
     (trunk_canting_rom, trunk_counterbalance_score, decoupled_trunk_lateral_lean_bottom) 전면 배제.
2. [16D 정예 하지 기계축 피처 파이프라인 (lunge_c2_features_v5)]:
   - 발끝 내회전 4D + 슬관절 동적 외반 4D + 하지 연쇄 2D + 골반 수평축 3D + 체간 진행축 사잇각 2D + DKCD 기계축 편위 1D.
3. [고신뢰도 앙상블 아키텍처]:
   - LightGBM (L1/L2 규제 및 비용 민감도 가중치) + CatBoost (L2 정규화 및 클래스 밸런싱) 5-Fold 앙상블.
4. [3-State 안전 게이트 제약 준수]:
   - 사용자 명시 제약: 판정 보류율 최대 10% (Coverage >= 90.0%) 엄격 준수.
   - 5-Fold StratifiedGroupKFold 피험자 전원(9명) 격리 및 공식 5대 시점(시점 A~E) 전수 감사.
"""

import os
import sys
import glob
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier

from lunge_c2_features_v5 import (
    FEATURE_NAMES_16D,
    extract_lunge_c2_features_v5,
    extract_lunge_c2_features_v5_batch
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 5대 공식 시점 명칭 매핑
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class BabelLungeCondition2ModelV5:
    """
    바벨 런지 Condition 2 v5 상체 노이즈 제거 프로덕션 모델
    - 16D 순수 하지 정렬 정예 피처셋
    - LightGBM + CatBoost 정규화 앙상블
    - 3-State 안전 게이트 (판정 보류율 최대 10%, 커버리지 >= 90%)
    """
    def __init__(
        self,
        tau_normal: float = 0.5100,
        tau_defect: float = 0.5450,
        max_retake_rate: float = 0.10
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.max_retake_rate = max_retake_rate
        self.min_coverage = 1.0 - max_retake_rate
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.cat_models: List[CatBoostClassifier] = []
        self.feature_names = FEATURE_NAMES_16D
        self.is_fitted = False

    def extract_features(self, kps_33_3d: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v5(kps_33_3d)

    def extract_features_batch(self, X_seqs: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v5_batch(X_seqs)

    def fit_cross_validation(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        print(f"[*] Starting 5-Fold StratifiedGroupKFold Training for bl2_v5 (16D Noise-Free, Retake <= {self.max_retake_rate*100:.0f}%, Cov >= {self.min_coverage*100:.0f}%)...")
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
            val_subs = list(np.unique(subjects[val_idx]))

            # LightGBM (16D 정규화)
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=140,
                max_depth=4,
                num_leaves=14,
                learning_rate=0.038,
                min_child_samples=18,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.15,
                reg_lambda=1.2,
                scale_pos_weight=cost_sensitive_weight,
                random_state=42 + fold,
                n_jobs=-1,
                verbose=-1
            )
            lgb_clf.fit(X_train, y_train)

            # CatBoost (16D 정규화)
            cat_clf = CatBoostClassifier(
                iterations=160,
                depth=4,
                learning_rate=0.038,
                l2_leaf_reg=3.5,
                auto_class_weights='Balanced',
                random_seed=42 + fold,
                verbose=0,
                thread_count=-1
            )
            cat_clf.fit(X_train, y_train)

            p_val_lgb = lgb_clf.predict_proba(X_val)[:, 1]
            p_val_cat = cat_clf.predict_proba(X_val)[:, 1]
            p_val = 0.50 * p_val_lgb + 0.50 * p_val_cat

            oof_probs[val_idx] = p_val
            self.lgb_models.append(lgb_clf)
            self.cat_models.append(cat_clf)

            f1_val = f1_score(y_val, (p_val >= 0.5).astype(int), zero_division=0)
            auc_val = roc_auc_score(y_val, p_val)
            print(f"  - Fold {fold} (Val 피험자: {val_subs}): Binary F1 = {f1_val*100:.2f}%, AUC = {auc_val*100:.2f}%")

        self.is_fitted = True

        # OOF 확률 기반 3-State 임계값 최적화 (보류율 <= 10.0%, 커버리지 >= 90.0% 엄격 준수)
        self.tau_normal, self.tau_defect, best_res = self._optimize_3state_thresholds(y, oof_probs)

        print(f"\n[+] 3-State Performance Optimized (Retake <= {self.max_retake_rate*100:.0f}%, Coverage >= {self.min_coverage*100:.0f}%):")
        print(f"    tau_normal = {self.tau_normal:.4f}, tau_defect = {self.tau_defect:.4f}")
        print(f"    Coverage = {best_res['coverage']*100:.2f}%, Retake = {best_res['retake_rate']*100:.2f}%")
        print(f"    Selective Defect F1 = {best_res['selective_defect_f1']*100:.2f}%, Clean F1 = {best_res['selective_clean_f1']*100:.2f}%")
        print(f"    Macro F1 = {((best_res['selective_defect_f1'] + best_res['selective_clean_f1'])/2)*100:.2f}%")
        print(f"    Selective Accuracy = {best_res['selective_accuracy']*100:.2f}%, Precision = {best_res['selective_precision']*100:.2f}%, Recall = {best_res['selective_recall']*100:.2f}%")
        print(f"    Full ROC-AUC = {best_res['roc_auc']:.4f}, Full Defect F1 = {best_res['full_coverage_defect_f1']*100:.2f}%")

        view_benchmarks = self._audit_views_3state(y, oof_probs, views)

        overall_summary = {
            "oof_probs": oof_probs,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "max_retake_rate": self.max_retake_rate,
            "min_coverage": self.min_coverage,
            "overall_3state": best_res,
            "view_benchmarks": view_benchmarks
        }
        return overall_summary

    def _optimize_3state_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        best_score = -1.0
        best_metrics = {}
        best_tn = 0.5050
        best_td = 0.5450

        prob_vals_n = np.linspace(0.45, 0.54, 91)
        prob_vals_d = np.linspace(0.50, 0.60, 101)

        cands = []
        for tn in prob_vals_n:
            for td in prob_vals_d:
                if td <= tn:
                    continue
                dec_mask = (y_prob <= tn) | (y_prob >= td)
                cov = float(np.mean(dec_mask))
                rej = 1.0 - cov
                if rej <= self.max_retake_rate + 0.0001:  # 보류율 최대 10.0%
                    y_dec = y_true[dec_mask]
                    p_dec = (y_prob[dec_mask] >= td).astype(int)
                    d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                    c_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
                    acc = accuracy_score(y_dec, p_dec)
                    if c_f1 >= 0.73 and acc >= 0.63:
                        # 결함 F1 우선 극대화 및 정상 F1 균형
                        score = 0.60 * d_f1 + 0.40 * c_f1
                        cands.append((score, d_f1, c_f1, acc, float(tn), float(td)))

        if cands:
            cands.sort(key=lambda x: (x[1], x[0]), reverse=True)
            best_tn = cands[0][4]
            best_td = cands[0][5]

        best_metrics = self.calculate_3state_metrics(y_true, y_prob, best_tn, best_td)
        return best_tn, best_td, best_metrics

    @staticmethod
    def calculate_3state_metrics(
        y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float
    ) -> Dict[str, Any]:
        N = len(y_true)
        full_pred = (y_prob >= 0.5).astype(int)
        full_acc = accuracy_score(y_true, full_pred)
        full_prec = precision_score(y_true, full_pred, zero_division=0)
        full_rec = recall_score(y_true, full_pred, zero_division=0)
        full_defect_f1 = f1_score(y_true, full_pred, zero_division=0)
        full_clean_f1 = f1_score(1 - y_true, 1 - full_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)

        decided_mask = (y_prob <= tau_normal) | (y_prob >= tau_defect)
        retake_mask = ~decided_mask

        coverage = float(np.mean(decided_mask))
        retake_rate = float(np.mean(retake_mask))

        y_dec_true = y_true[decided_mask]
        y_dec_prob = y_prob[decided_mask]
        y_dec_pred = (y_dec_prob >= tau_defect).astype(int)

        if len(y_dec_true) > 0 and len(np.unique(y_dec_true)) > 1:
            sel_acc = accuracy_score(y_dec_true, y_dec_pred)
            sel_prec = precision_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_rec = recall_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_defect_f1 = f1_score(y_dec_true, y_dec_pred, zero_division=0)
            sel_clean_f1 = f1_score(1 - y_dec_true, 1 - y_dec_pred, zero_division=0)
            cm = confusion_matrix(y_dec_true, y_dec_pred).tolist()
        else:
            sel_acc, sel_prec, sel_rec, sel_defect_f1, sel_clean_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
            cm = []

        return {
            "total_samples": int(N),
            "coverage": float(coverage),
            "retake_rate": float(retake_rate),
            "retake_count": int(np.sum(retake_mask)),
            "decided_count": int(np.sum(decided_mask)),
            "selective_defect_f1": float(sel_defect_f1),
            "selective_clean_f1": float(sel_clean_f1),
            "selective_accuracy": float(sel_acc),
            "selective_precision": float(sel_prec),
            "selective_recall": float(sel_rec),
            "confusion_matrix": cm,
            "full_coverage_defect_f1": float(full_defect_f1),
            "full_coverage_clean_f1": float(full_clean_f1),
            "full_coverage_accuracy": float(full_acc),
            "full_coverage_precision": float(full_prec),
            "full_coverage_recall": float(full_rec),
            "roc_auc": float(auc)
        }

    def _audit_views_3state(
        self, y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for vn, v_alias in VIEW_ALIAS_MAP.items():
            mask = (views == vn)
            if np.sum(mask) == 0:
                continue
            v_true = y_true[mask]
            v_prob = y_prob[mask]
            m = self.calculate_3state_metrics(v_true, v_prob, self.tau_normal, self.tau_defect)
            results[vn] = {
                "alias": v_alias,
                "metrics": m
            }
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet!")

        n_models = len(self.lgb_models)
        p_ens = np.zeros(len(X), dtype=np.float32)

        for i in range(n_models):
            p_lgb = self.lgb_models[i].predict_proba(X)[:, 1]
            p_cat = self.cat_models[i].predict_proba(X)[:, 1]
            p_ens += 0.50 * p_lgb + 0.50 * p_cat
        p_ens /= n_models

        return p_ens

    def predict_3state(self, kps_seq: np.ndarray) -> Dict[str, Any]:
        feats = self.extract_features(kps_seq).reshape(1, -1)
        p = float(self.predict_proba(feats)[0])

        if p <= self.tau_normal:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "완벽합니다! 몸통, 무릎, 발의 방향이 올바르게 11자 정렬되어 있습니다."
        elif p >= self.tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "선행 발끝이 안쪽으로 돌아가며 무릎-체간 정렬이 무너졌습니다. 발끝을 11자 전방으로 정렬하고 무릎이 안쪽으로 쏠리지 않게 주의하세요."
        else:
            state = 2
            state_str = "RETAKE (보류)"
            feedback = "하지 관절의 정렬 궤적이 명확하지 않습니다. 대각선 45도 방향에서 전신이 다 보이도록 1회 다시 수행해 주세요."

        return {
            "state": state,
            "state_str": state_str,
            "defect_probability": p,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "feedback": feedback
        }

    def save_bundle(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "model_name": "bl2_v5",
            "exercise": "바벨 런지 (04_바벨_런지)",
            "defect": "Condition 2 (몸통 발 무릎 방향 일치 여부)",
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "max_retake_rate": self.max_retake_rate,
            "min_coverage": self.min_coverage,
            "feature_names": self.feature_names,
            "lgb_models": self.lgb_models,
            "cat_models": self.cat_models,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[+] Model bundle saved: {filepath}")

    @classmethod
    def load_bundle(cls, filepath: str) -> "BabelLungeCondition2ModelV5":
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        obj = cls(
            tau_normal=bundle["tau_normal"],
            tau_defect=bundle["tau_defect"],
            max_retake_rate=bundle.get("max_retake_rate", 0.10)
        )
        obj.lgb_models = bundle["lgb_models"]
        obj.cat_models = bundle["cat_models"]
        obj.feature_names = bundle["feature_names"]
        obj.is_fitted = True
        return obj


def load_lunge_dataset_v5() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """16D 노이즈 제거 피처 데이터셋 로드"""
    feat_cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_v4_cache.npz")
    if os.path.exists(feat_cache_file):
        print(f"[*] Loading pre-extracted features from cache: {feat_cache_file}")
        c = np.load(feat_cache_file, allow_pickle=True)
        from lunge_c2_features_v5 import FEATURE_INDICES_16D
        X = c["X"][:, FEATURE_INDICES_16D]  # 16D 정예 슬라이싱
        y = c["y"]
        subjects = c["subjects"]
        views = c["views"]
        clip_names = c["clip_names"].tolist()
        print(f"[+] Loaded 16D Noise-Free Features: X={X.shape}, y={y.shape} (Defects={np.sum(y==1)}, Normals={np.sum(y==0)})")
        return X, y, subjects, views, clip_names

    raise FileNotFoundError(f"Feature cache not found: {feat_cache_file}")


def main():
    t_start = time.time()
    bundle_path = os.path.join(CURRENT_DIR, "weights", "bl2_v5_bundle.pkl")
    results_path = os.path.join(CURRENT_DIR, "bl2_v5_benchmark_results.json")

    X, y, subjects, views, clip_names = load_lunge_dataset_v5()
    print(f"[+] Dataset Ready: Total {len(y)} samples (Defect: {np.sum(y==1)}, Normal: {np.sum(y==0)})")
    print(f"    Subjects: {np.unique(subjects)}, Views: {np.unique(views)}")

    model = BabelLungeCondition2ModelV5(max_retake_rate=0.10)
    bench_results = model.fit_cross_validation(X, y, subjects, views, n_splits=5)

    model.save_bundle(bundle_path)

    # JSON 저장 직렬화
    serializable_results = {
        "model_name": "bl2_v5",
        "description": "v5 Pure Lower-Body Alignment & Trunk Noise-Free Production Model",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_retake_rate": float(model.max_retake_rate),
        "min_coverage": float(model.min_coverage),
        "tau_normal": float(bench_results["tau_normal"]),
        "tau_defect": float(bench_results["tau_defect"]),
        "overall_3state": bench_results["overall_3state"],
        "view_benchmarks": bench_results["view_benchmarks"],
        "total_elapsed_sec": round(time.time() - t_start, 2)
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"[+] Benchmark results saved: {results_path}")


if __name__ == "__main__":
    main()

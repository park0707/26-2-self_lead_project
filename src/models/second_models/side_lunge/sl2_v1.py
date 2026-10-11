# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Model: sl2_v1.py (Side Lunge Condition 2 v1 Production Model - Inherited from bl2 v6)

설계 및 계승 사양:
1. [bl2 v6 베이스라인 100% 온전 계승]:
   - 상체 노이즈(바벨 밸런싱/체간 기울기 등) 완전 배제.
   - 생체역학 기립 베이스라인 정규화 및 수평 정사영 18차원 정예 피처 셋.
   - 17. knee_displacement_foot_heading_angle: 최하사점 무릎 수평 변위축 vs 발끝 장축 사잇각.
   - 18. delta_q_angle_standing_to_bottom: 기립 대비 최하사점 Q-angle 동적 변화량.
2. [사이드 런지 특화 생체역학 정밀 적응]:
   - 좌/우 다리 중 깊게 굽힌 다리(Knee Flexion Min)를 작업 다리(Working Leg)로 정밀 탐지.
   - 작업 다리 무릎 최대 굴곡 프레임을 최하사점(Bottom Position)으로 자동 동기화.
3. [결함 판정 F1(Defect F1) 극대화 3-State 최적화]:
   - 사용자 명시 제약: 판정 보류율 최대 10% (Coverage >= 90.0%) 엄격 준수.
   - LightGBM + CatBoost 앙상블.
   - 피험자 1~9 전수 5-Fold StratifiedGroupKFold 완전 격리 검증.
   - 공식 5대 시점(시점 A~E: view1~view5) 다차원 성능 감사.
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_features import (
    FEATURE_NAMES_18D,
    extract_side_lunge_c2_features_18d,
    extract_side_lunge_c2_features_batch
)

# 5대 공식 시점 명칭 매핑 (AGENTS.md 표준)
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class SideLungeCondition2ModelV1:
    """
    사이드 런지 Condition 2 v1 모델 (bl2 v6 정통 계승)
    - 18D (16D 하지 정렬 + 1D 수평 무릎변위-발끝 사잇각 + 1D 기립대비 동적 Q각 변화량)
    - LightGBM + CatBoost 앙상블
    - 3-State 안전 선택적 판정 게이트 (State 0: CLEAN, State 1: DEFECT, State 2: RETAKE)
    - 판정 보류율 최대 10% (Coverage >= 90%) 엄격 준수
    """
    def __init__(
        self,
        tau_normal: float = 0.5000,
        tau_defect: float = 0.5310,
        max_retake_rate: float = 0.10
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.max_retake_rate = max_retake_rate
        self.min_coverage = 1.0 - max_retake_rate
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.cat_models: List[CatBoostClassifier] = []
        self.feature_names = FEATURE_NAMES_18D
        self.is_fitted = False

    def extract_features(self, kps_33_3d: np.ndarray) -> np.ndarray:
        return extract_side_lunge_c2_features_18d(kps_33_3d)

    def extract_features_batch(self, X_seqs: np.ndarray) -> np.ndarray:
        return extract_side_lunge_c2_features_batch(X_seqs)

    def fit_cross_validation(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        print(f"[*] Starting 5-Fold StratifiedGroupKFold Training for sl2_v1 (18D Baseline Delta, Defect F1 Maximized)...")
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

            # LightGBM (18D)
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=130,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.040,
                min_child_samples=18,
                colsample_bytree=0.75,
                subsample=0.85,
                reg_alpha=0.5,
                reg_lambda=2.0,
                scale_pos_weight=cost_sensitive_weight,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=-1
            )
            lgb_clf.fit(X_train, y_train)

            # CatBoost (18D)
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

        # OOF 확률 기반 Defect F1 직접 극대화 3-State 임계값 최적화
        self.tau_normal, self.tau_defect, best_res = self._optimize_3state_thresholds(y, oof_probs)

        print(f"\n[+] 3-State Performance Optimized (Defect F1 Maximized, Retake <= {self.max_retake_rate*100:.0f}%):")
        print(f"    tau_normal = {self.tau_normal:.4f}, tau_defect = {self.tau_defect:.4f}")
        print(f"    Coverage = {best_res['coverage']*100:.2f}%, Retake = {best_res['retake_rate']*100:.2f}%")
        print(f"    Selective Defect F1 = {best_res['selective_defect_f1']*100:.2f}%, Clean F1 = {best_res['selective_clean_f1']*100:.2f}%")
        print(f"    Macro F1 = {((best_res['selective_defect_f1'] + best_res['selective_clean_f1'])/2)*100:.2f}%")
        print(f"    Selective Accuracy = {best_res['selective_accuracy']*100:.2f}%, Precision = {best_res['selective_precision']*100:.2f}%, Recall = {best_res['selective_recall']*100:.2f}%")
        print(f"    Full ROC-AUC = {best_res['roc_auc']:.4f}, Full Defect F1 = {best_res['full_coverage_defect_f1']*100:.2f}%")

        view_benchmarks = self._audit_views_3state(y, oof_probs, views)

        overall_summary = {
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "overall_3state": best_res,
            "view_benchmarks": view_benchmarks
        }
        return overall_summary

    def _optimize_3state_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        best_tn = 0.5000
        best_td = 0.5310

        prob_vals_n = np.linspace(0.40, 0.55, 76)
        prob_vals_d = np.linspace(0.48, 0.65, 86)

        cands = []
        for tn in prob_vals_n:
            for td in prob_vals_d:
                if td <= tn:
                    continue
                dec_mask = (y_prob <= tn) | (y_prob >= td)
                cov = float(np.mean(dec_mask))
                rej = 1.0 - cov
                if rej <= self.max_retake_rate + 0.0001:  # 보류율 최대 10.0% 엄격 준수
                    y_dec = y_true[dec_mask]
                    p_dec = (y_prob[dec_mask] >= td).astype(int)
                    d_f1 = f1_score(y_dec, p_dec, zero_division=0)
                    c_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
                    acc = accuracy_score(y_dec, p_dec)
                    if c_f1 >= 0.700 and acc >= 0.600:
                        cands.append((d_f1, c_f1, acc, float(tn), float(td)))

        if cands:
            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_tn = cands[0][3]
            best_td = cands[0][4]

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
            "macro_f1": float((sel_defect_f1 + sel_clean_f1) / 2.0),
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
        print("\n" + "=" * 80)
        print("  [sl2_v1] 5대 공식 시점별 3-State 심층 다차원 성능 감사 (Audit)")
        print("=" * 80)
        view_results = {}
        for vn in ["view5", "view2", "view3", "view4", "view1"]:
            v_mask = (views == vn)
            if not np.any(v_mask):
                continue
            y_v_true = y_true[v_mask]
            y_v_prob = y_prob[v_mask]
            metrics = self.calculate_3state_metrics(y_v_true, y_v_prob, self.tau_normal, self.tau_defect)
            view_results[vn] = metrics

            alias = VIEW_ALIAS_MAP.get(vn, vn)
            print(f"[*] {alias} [{vn}]:")
            print(f"    - Samples: {metrics['total_samples']} (Decided: {metrics['decided_count']}, Retake: {metrics['retake_count']})")
            print(f"    - Coverage: {metrics['coverage']*100:.2f}% | Retake: {metrics['retake_rate']*100:.2f}%")
            print(f"    - Selective Defect F1: {metrics['selective_defect_f1']*100:.2f}% | Clean F1: {metrics['selective_clean_f1']*100:.2f}% | Macro F1: {metrics['macro_f1']*100:.2f}%")
            print(f"    - Selective Accuracy: {metrics['selective_accuracy']*100:.2f}% | Precision: {metrics['selective_precision']*100:.2f}% | Recall: {metrics['selective_recall']*100:.2f}%")
            print(f"    - ROC-AUC: {metrics['roc_auc']:.4f} | Full Defect F1: {metrics['full_coverage_defect_f1']*100:.2f}%")
            print("-" * 80)
        return view_results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        p_lgb = np.mean([clf.predict_proba(X)[:, 1] for clf in self.lgb_models], axis=0)
        p_cat = np.mean([clf.predict_proba(X)[:, 1] for clf in self.cat_models], axis=0)
        return 0.50 * p_lgb + 0.50 * p_cat

    def predict_3state(self, kps_33_3d: np.ndarray) -> Dict[str, Any]:
        feat = self.extract_features(kps_33_3d).reshape(1, -1)
        prob = float(self.predict_proba(feat)[0])

        if prob <= self.tau_normal:
            state = 0
            state_label = "CLEAN (정상)"
        elif prob >= self.tau_defect:
            state = 1
            state_label = "DEFECT (결함: 몸통-발-무릎 불일치)"
        else:
            state = 2
            state_label = "RETAKE (판정 보류/재촬영 권고)"

        return {
            "defect_probability": prob,
            "state": state,
            "state_label": state_label,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect
        }

    def save_bundle(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "lgb_models": self.lgb_models,
            "cat_models": self.cat_models,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "max_retake_rate": self.max_retake_rate,
            "feature_names": self.feature_names
        }
        with open(filepath, "wb") as fp:
            pickle.dump(bundle, fp)
        print(f"[+] Model bundle successfully saved to: {filepath}")

    def load_bundle(self, filepath: str):
        with open(filepath, "rb") as fp:
            bundle = pickle.load(fp)
        self.lgb_models = bundle["lgb_models"]
        self.cat_models = bundle["cat_models"]
        self.tau_normal = bundle["tau_normal"]
        self.tau_defect = bundle["tau_defect"]
        self.max_retake_rate = bundle["max_retake_rate"]
        self.feature_names = bundle["feature_names"]
        self.is_fitted = True
        print(f"[+] Model bundle loaded from: {filepath}")


def load_side_lunge_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    라벨과 설명이 완벽히 일치하는 클린 사이드 런지 전처리 데이터셋 로드
    Returns: X_seqs (N, 64, 33, 3), y (N,), subjects (N,), views (N,)
    """
    DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
    LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
    CLEAN_LIST_PATH = os.path.join(DATA_DIR, "side_lunge_clean_folders.json")

    if not os.path.exists(CLEAN_LIST_PATH):
        raise FileNotFoundError(f"Clean folder list not found at: {CLEAN_LIST_PATH}")

    with open(CLEAN_LIST_PATH, "r", encoding="utf-8") as fp:
        clean_folders = json.load(fp)

    X_list = []
    y_list = []
    sub_list = []
    view_list = []

    print(f"[*] Loading clean side lunge sequences ({len(clean_folders)} folders)...")

    for folder in clean_folders:
        json_p = os.path.join(LABEL_DIR, f"{folder}.json")
        with open(json_p, "r", encoding="utf-8") as fp:
            d = json.load(fp)

        # Condition 2: 몸통 발 무릎 방향 일치여부 (False -> defect=1, True -> clean=0)
        conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
        is_c2_defect = int(conds.get("몸통 발 무릎 방향 일치여부") is False)

        # 피험자 번호 추출 (1~9)
        sub_id = 1
        for s in range(1, 10):
            if f"피험자{s}" in folder:
                sub_id = s
                break

        # 5대 시점 로드
        for vl, vn in [("A", "view5"), ("B", "view2"), ("C", "view3"), ("D", "view4"), ("E", "view1")]:
            npz_p = os.path.join(DATA_DIR, f"{folder}_{vn}.npz")
            if not os.path.exists(npz_p):
                continue
            npz_data = np.load(npz_p)
            kps_3d = npz_data["kps_33_3d_64f"][:, :, :3]  # (64, 33, 3)

            X_list.append(kps_3d)
            y_list.append(is_c2_defect)
            sub_list.append(sub_id)
            view_list.append(vn)

    X_seqs = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(sub_list, dtype=np.int32)
    views = np.array(view_list, dtype=object)

    print(f"[+] Loaded {len(y)} sequences across {len(np.unique(subjects))} subjects.")
    print(f"    - Defects (c2=1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.2f}%)")
    print(f"    - Clean (c2=0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.2f}%)")
    return X_seqs, y, subjects, views


def main():
    CACHE_PATH = os.path.join(CURRENT_DIR, "side_lunge_c2_features_18d_cache.npz")
    RESULTS_PATH = os.path.join(CURRENT_DIR, "sl2_v1_benchmark_results.json")
    WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "sl2_v1_bundle.pkl")

    if os.path.exists(CACHE_PATH):
        print(f"[*] Found cached features at: {CACHE_PATH}")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        X_feats = cache["X_feats"]
        y = cache["y"]
        subjects = cache["subjects"]
        views = cache["views"]
    else:
        X_seqs, y, subjects, views = load_side_lunge_dataset()
        print(f"[*] Extracting 18D features from {len(X_seqs)} sequences...")
        t0 = time.time()
        X_feats = extract_side_lunge_c2_features_batch(X_seqs)
        t1 = time.time()
        print(f"[+] Feature extraction completed in {t1-t0:.2f}s (Shape: {X_feats.shape})")

        np.savez_compressed(
            CACHE_PATH,
            X_feats=X_feats,
            y=y,
            subjects=subjects,
            views=views
        )
        print(f"[+] Saved feature cache to: {CACHE_PATH}")

    # 모델 학습 및 교차검증
    model = SideLungeCondition2ModelV1(max_retake_rate=0.10)
    summary = model.fit_cross_validation(X_feats, y, subjects, views, n_splits=5)

    # 모델 가중치 번들 저장
    model.save_bundle(WEIGHTS_PATH)

    # 벤치마크 결과 저장
    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[+] Benchmark results saved to: {RESULTS_PATH}")


if __name__ == '__main__':
    main()

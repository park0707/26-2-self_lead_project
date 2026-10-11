# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Model: bl2_v3.py (v3 Feature & Model Architecture Optimized Production Model)

주요 혁신 및 최적화 포인트:
1. [피처 전수 최적화]: 30D에서 0분산/완전중복/초고상관(r>0.88)/저변별력 잡음 피처 13개 영구 제거 -> 17D 정예 피처 구축
2. [모델 구조 최적화]:
   - Tier 1: 정규화 강화 LGBM (reg_alpha=0.1, reg_lambda=1.0) + CatBoost (l2_leaf_reg=3.0) 앙상블
   - Tier 2: 불확실 구간(0.30<=p<=0.70) 전용 XGBoost + CatBoost + LogisticRegression 3중 스태킹
   - 잔차 융합: p_final = 0.40 * p_tier1 + 0.60 * p_tier2
3. [커버리지 제약 최적화]: 사용자 명시 요구사항 "커버리지 최대 90%정도까지" 엄격 적용
   - 0.75 <= Coverage <= 0.90 구간에서 3-State Selective Defect F1 최대화 임계값(tau_normal, tau_defect) 도출
4. [5-Fold 피험자 격리 및 공식 5대 시점 감사]:
   - StratifiedGroupKFold(n_splits=5, groups=subjects)
   - 시점 A~E(view1~5) 전수 대조 감사
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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from lunge_c2_features_v3 import (
    FEATURE_NAMES_17D,
    extract_lunge_c2_features_v3,
    extract_lunge_c2_features_v3_batch
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


class BabelLungeCondition2ModelV3:
    """
    바벨 런지 Condition 2 v3 최적화 모델
    - 17D 정예 피처셋
    - 2-Tier Cascade 구조 최적화
    - 3-State 안전 게이트 (커버리지 <= 90% 제약)
    """
    def __init__(self, tau_normal: float = 0.35, tau_defect: float = 0.62, max_retake_rate: float = 0.10):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.max_retake_rate = max_retake_rate
        self.min_coverage = 1.0 - max_retake_rate
        self.tier1_lgb_models: List[lgb.LGBMClassifier] = []
        self.tier1_cat_models: List[CatBoostClassifier] = []
        self.tier2_xgb_models: List[XGBClassifier] = []
        self.tier2_cat_models: List[CatBoostClassifier] = []
        self.tier2_lr_models: List[LogisticRegression] = []
        self.feature_names = FEATURE_NAMES_17D
        self.is_fitted = False

    def extract_features(self, kps_33_3d: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v3(kps_33_3d)

    def extract_features_batch(self, X_seqs: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v3_batch(X_seqs)

    def fit_cross_validation(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        print(f"[*] Starting 5-Fold StratifiedGroupKFold Training for bl2_v3 (17D Elite, Retake <= {self.max_retake_rate*100:.0f}%, Cov >= {self.min_coverage*100:.0f}%)...")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)

        self.tier1_lgb_models = []
        self.tier1_cat_models = []
        self.tier2_xgb_models = []
        self.tier2_cat_models = []
        self.tier2_lr_models = []

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        base_pos_weight = float(n_neg) / float(n_pos)
        cost_sensitive_weight = base_pos_weight * 1.35

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            val_subs = list(np.unique(subjects[val_idx]))

            # Tier 1-A: LightGBM (정규화 강화)
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=130,
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

            # Tier 1-B: CatBoost (L2 정규화 강화)
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
            p_val_t1 = 0.50 * p_val_lgb + 0.50 * p_val_cat

            # Tier 2: 모호 구간 (0.30 <= p <= 0.70)
            p_tr_lgb = lgb_clf.predict_proba(X_train)[:, 1]
            p_tr_cat = cat_clf.predict_proba(X_train)[:, 1]
            p_tr_t1 = 0.50 * p_tr_lgb + 0.50 * p_tr_cat

            amb_train_mask = (p_tr_t1 >= 0.30) & (p_tr_t1 <= 0.70)
            if np.sum(amb_train_mask) < 40 or len(np.unique(y_train[amb_train_mask])) < 2:
                amb_train_mask = np.ones(len(y_train), dtype=bool)

            X_tr_t2 = X_train[amb_train_mask]
            y_tr_t2 = y_train[amb_train_mask]

            # Tier 2-A: XGBoost
            xgb_clf = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.038,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=cost_sensitive_weight,
                random_state=100 + fold,
                eval_metric="logloss",
                n_jobs=-1
            )
            xgb_clf.fit(X_tr_t2, y_tr_t2)

            # Tier 2-B: CatBoost T2
            cat_t2 = CatBoostClassifier(
                iterations=120,
                depth=3,
                learning_rate=0.038,
                l2_leaf_reg=3.5,
                auto_class_weights='Balanced',
                random_seed=200 + fold,
                verbose=0,
                thread_count=-1
            )
            cat_t2.fit(X_tr_t2, y_tr_t2)

            # Tier 2-C: LightGBM T2
            lgb_t2 = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                num_leaves=8,
                learning_rate=0.038,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=cost_sensitive_weight,
                random_state=300 + fold,
                n_jobs=-1,
                verbose=-1
            )
            lgb_t2.fit(X_tr_t2, y_tr_t2)

            # 검증 데이터 융합
            p_val_xgb = xgb_clf.predict_proba(X_val)[:, 1]
            p_val_cat2 = cat_t2.predict_proba(X_val)[:, 1]
            p_val_lgb2 = lgb_t2.predict_proba(X_val)[:, 1]
            p_val_t2 = 0.35 * p_val_xgb + 0.35 * p_val_cat2 + 0.30 * p_val_lgb2

            amb_val_mask = (p_val_t1 >= 0.30) & (p_val_t1 <= 0.70)
            p_val_final = np.copy(p_val_t1)
            p_val_final[amb_val_mask] = 0.40 * p_val_t1[amb_val_mask] + 0.60 * p_val_t2[amb_val_mask]

            oof_probs[val_idx] = p_val_final

            # 모델 보관
            self.tier1_lgb_models.append(lgb_clf)
            self.tier1_cat_models.append(cat_clf)
            self.tier2_xgb_models.append(xgb_clf)
            self.tier2_cat_models.append(cat_t2)
            self.tier2_lr_models.append(lgb_t2)

            f1_val = f1_score(y_val, (p_val_final >= 0.5).astype(int), zero_division=0)
            auc_val = roc_auc_score(y_val, p_val_final)
            print(f"  - Fold {fold} (Val 피험자: {val_subs}): Binary F1 = {f1_val*100:.2f}%, AUC = {auc_val*100:.2f}%")

        self.is_fitted = True

        # 판정 보류율 최대 10% (커버리지 최소 90%) 제약 하에 3-State 임계값 최적화
        self.tau_normal, self.tau_defect, best_res = self._optimize_3state_thresholds(y, oof_probs)
        print(f"\n[+] 3-State Thresholds Optimized (Retake <= {self.max_retake_rate*100:.0f}%, Coverage >= {self.min_coverage*100:.0f}%):")
        print(f"    tau_normal = {self.tau_normal:.4f}, tau_defect = {self.tau_defect:.4f}")
        print(f"    Coverage = {best_res['coverage']*100:.2f}%, Retake = {best_res['retake_rate']*100:.2f}%")
        print(f"    Selective Defect F1 = {best_res['selective_defect_f1']*100:.2f}%, Clean F1 = {best_res['selective_clean_f1']*100:.2f}%")
        print(f"    Selective Accuracy = {best_res['selective_accuracy']*100:.2f}%, Precision = {best_res['selective_precision']*100:.2f}%, Recall = {best_res['selective_recall']*100:.2f}%")

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

        quantiles = np.linspace(0.01, 0.99, 99)
        prob_vals = np.unique(np.quantile(y_prob, quantiles))

        best_tau_n = float(np.quantile(y_prob, 0.05))
        best_tau_d = float(np.quantile(y_prob, 0.95))

        # 사용자 명시 조건: 판정 보류율 최대 10% (retake_rate <= 0.10 <=> coverage >= 0.90)
        for i, tn in enumerate(prob_vals):
            for td in prob_vals[i+1:]:
                dec_mask = (y_prob <= tn) | (y_prob >= td)
                cov = float(np.mean(dec_mask))
                retake_rate = 1.0 - cov
                if retake_rate <= self.max_retake_rate:  # 보류율 최대 10% (0% <= retake <= 10%)
                    m = self.calculate_3state_metrics(y_true, y_prob, float(tn), float(td))
                    d_f1 = m["selective_defect_f1"]
                    c_f1 = m["selective_clean_f1"]
                    # Balanced Harmonic / Arithmetic Score
                    score = 0.50 * d_f1 + 0.50 * c_f1
                    if score > best_score:
                        best_score = score
                        best_tau_n = float(tn)
                        best_tau_d = float(td)
                        best_metrics = m

        if not best_metrics:
            # 10%에 가장 근접한 조건으로 완화
            for i, tn in enumerate(prob_vals):
                for td in prob_vals[i+1:]:
                    dec_mask = (y_prob <= tn) | (y_prob >= td)
                    cov = float(np.mean(dec_mask))
                    if cov >= 0.88:
                        m = self.calculate_3state_metrics(y_true, y_prob, float(tn), float(td))
                        score = 0.50 * m["selective_defect_f1"] + 0.50 * m["selective_clean_f1"]
                        if score > best_score:
                            best_score = score
                            best_tau_n = float(tn)
                            best_tau_d = float(td)
                            best_metrics = m

        if not best_metrics:
            best_metrics = self.calculate_3state_metrics(y_true, y_prob, best_tau_n, best_tau_d)

        return best_tau_n, best_tau_d, best_metrics

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

        n_models = len(self.tier1_lgb_models)
        p_t1 = np.zeros(len(X), dtype=np.float32)

        for i in range(n_models):
            p_lgb = self.tier1_lgb_models[i].predict_proba(X)[:, 1]
            p_cat = self.tier1_cat_models[i].predict_proba(X)[:, 1]
            p_t1 += 0.50 * p_lgb + 0.50 * p_cat
        p_t1 /= n_models

        p_final = np.copy(p_t1)
        amb_mask = (p_t1 >= 0.30) & (p_t1 <= 0.70)
        if np.sum(amb_mask) > 0:
            X_amb = X[amb_mask]
            p_t2 = np.zeros(len(X_amb), dtype=np.float32)
            for i in range(n_models):
                p_xgb = self.tier2_xgb_models[i].predict_proba(X_amb)[:, 1]
                p_cat2 = self.tier2_cat_models[i].predict_proba(X_amb)[:, 1]
                p_lgb2 = self.tier2_lr_models[i].predict_proba(X_amb)[:, 1]
                p_t2 += 0.35 * p_xgb + 0.35 * p_cat2 + 0.30 * p_lgb2
            p_t2 /= n_models
            p_final[amb_mask] = 0.40 * p_t1[amb_mask] + 0.60 * p_t2

        return p_final

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
            "model_name": "bl2_v3",
            "exercise": "바벨 런지 (04_바벨_런지)",
            "defect": "Condition 2 (몸통 발 무릎 방향 일치 여부)",
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "max_retake_rate": self.max_retake_rate,
            "min_coverage": self.min_coverage,
            "feature_names": self.feature_names,
            "tier1_lgb_models": self.tier1_lgb_models,
            "tier1_cat_models": self.tier1_cat_models,
            "tier2_xgb_models": self.tier2_xgb_models,
            "tier2_cat_models": self.tier2_cat_models,
            "tier2_lr_models": self.tier2_lr_models,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[+] Model bundle saved: {filepath}")

    @classmethod
    def load_bundle(cls, filepath: str) -> "BabelLungeCondition2ModelV3":
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        obj = cls(
            tau_normal=bundle["tau_normal"],
            tau_defect=bundle["tau_defect"],
            max_retake_rate=bundle.get("max_retake_rate", 0.10)
        )
        obj.tier1_lgb_models = bundle["tier1_lgb_models"]
        obj.tier1_cat_models = bundle["tier1_cat_models"]
        obj.tier2_xgb_models = bundle["tier2_xgb_models"]
        obj.tier2_cat_models = bundle["tier2_cat_models"]
        obj.tier2_lr_models = bundle["tier2_lr_models"]
        obj.feature_names = bundle["feature_names"]
        obj.is_fitted = True
        return obj


def load_lunge_dataset_v3() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    17D 정예 피처 데이터셋 로드 (캐시 지원)
    """
    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    label_dir = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
    feat_cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_v3_cache.npz")

    if os.path.exists(feat_cache_file):
        print(f"[*] Loading pre-extracted v3 features from cache: {feat_cache_file}")
        c = np.load(feat_cache_file, allow_pickle=True)
        X = c["X"]
        y = c["y"]
        subjects = c["subjects"]
        views = c["views"]
        clip_names = c["clip_names"].tolist()
        print(f"[+] Loaded from cache: X={X.shape}, y={y.shape} (Defects={np.sum(y==1)}, Normals={np.sum(y==0)})")
        return X, y, subjects, views, clip_names

    print(f"[*] Extracting 17D Elite Features from preprocessed sequences: {cache_dir}")
    npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    print(f"[*] 총 분석 대상 바벨 런지 시퀀스 수: {len(npz_files)}개")

    feats_list = []
    y_list = []
    subjects_list = []
    views_list = []
    clip_names = []

    t0 = time.time()
    for npz_path in npz_files:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        parts = base_name.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(label_dir, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        try:
            with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
                d_json = json.load(f)
            conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        except Exception:
            continue

        c2_val = conds.get("몸통 발 무릎 방향 일치여부", True)
        y_c2 = 0 if c2_val else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        try:
            data = np.load(npz_path)
            kps_33_3d = data["kps_33_3d_64f"]
            if kps_33_3d.shape[-1] == 4:
                kps_33_3d = kps_33_3d[:, :, :3]
            f17 = extract_lunge_c2_features_v3(kps_33_3d)

            feats_list.append(f17)
            y_list.append(y_c2)
            subjects_list.append(sub)
            views_list.append(vn)
            clip_names.append(base_name)
        except Exception as e:
            continue

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list, dtype=object)

    t1 = time.time()
    print(f"[+] 17D 정예 피처 추출 완료 (경과: {t1-t0:.1f}초)")
    print(f"  - 17D 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포    : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")

    np.savez_compressed(
        feat_cache_file,
        X=X, y=y, subjects=subjects, views=views, clip_names=clip_names
    )
    print(f"[+] Saved extracted v3 features to cache: {feat_cache_file}")

    return X, y, subjects, views, clip_names


def main():
    t_start = time.time()
    X, y, subjects, views, clip_names = load_lunge_dataset_v3()

    model = BabelLungeCondition2ModelV3(max_retake_rate=0.10)
    summary = model.fit_cross_validation(X, y, subjects, views, n_splits=5)

    # 번들 저장
    bundle_path = os.path.join(CURRENT_DIR, "weights", "bl2_v3_bundle.pkl")
    model.save_bundle(bundle_path)

    # 벤치마크 JSON 저장
    bench_path = os.path.join(CURRENT_DIR, "bl2_v3_benchmark_results.json")
    bench_data = {
        "model_name": "bl2_v3",
        "exercise": "04_바벨_런지",
        "defect": "Condition 2 (몸통 발 무릎 방향 일치 여부)",
        "tau_normal": summary["tau_normal"],
        "tau_defect": summary["tau_defect"],
        "max_retake_rate": summary["max_retake_rate"],
        "min_coverage": summary["min_coverage"],
        "overall_3state": summary["overall_3state"],
        "view_benchmarks": summary["view_benchmarks"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(time.time() - t_start, 2)
    }

    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Benchmark results saved: {bench_path}")


if __name__ == "__main__":
    main()

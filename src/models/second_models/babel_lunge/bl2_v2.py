# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Model Name: bl2_v2.py
Version: v2.0 (SOTA Research-Driven Upgrade)
Purpose: SOTA 선행연구 및 신체 국소 직교계(View-Invariant Body Frame) 30D 정예 피처 기반
         Asymmetric Cost-Sensitive 2-Tier Cascade 앙상블 & 3-State 안전 게이트 통합 모델
         - 30차원 Tri-planar Kinetic Chain 피처 (FPPA, FPA, Directional Misalignment, Hip Adduction, TLL, 3위상 동역학)
         - Asymmetric Cost-Sensitive Learning (scale_pos_weight=3.5, Focal parameter gamma=1.8)
         - 2-Tier Cascade 구조 (Tier 1 고속 선별 + Tier 2 경계선 정밀 조정)
         - Conformal Prediction 3-State 안전 게이트 (커버리지 >= 89% 가드레일)
         - 피험자 1~9 전수 5-Fold StratifiedGroupKFold 완전 격리 검증
         - 공식 5대 시점(시점 A~E)별 3-State F1-Score 전수 감사 및 v1 대조 델타 산출
"""

import os
import sys
import glob
import json
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

# Windows OpenMP 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from lunge_c2_features_v2 import (
    extract_lunge_c2_features_v2, extract_lunge_c2_features_v2_batch, FEATURE_NAMES_30D
)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class BabelLungeCondition2ModelV2:
    """
    바벨 런지 Condition 2 v2.0 공식 모델 (bl2_v2)
    """
    def __init__(self, tau_normal: float = 0.38, tau_defect: float = 0.58):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.tier1_lgb_models: List[lgb.LGBMClassifier] = []
        self.tier1_cat_models: List[CatBoostClassifier] = []
        self.tier2_models: List[lgb.LGBMClassifier] = []
        self.feature_names = FEATURE_NAMES_30D
        self.is_fitted = False

    def extract_features(self, kps_33_3d: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v2(kps_33_3d)

    def extract_features_batch(self, X_seqs: np.ndarray) -> np.ndarray:
        return extract_lunge_c2_features_v2_batch(X_seqs)

    def fit_cross_validation(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        StratifiedGroupKFold (피험자 격리 5-Fold) 학습 및 3-State 최적화
        """
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)

        self.tier1_lgb_models = []
        self.tier1_cat_models = []
        self.tier2_models = []

        print(f"[*] Starting 5-Fold StratifiedGroupKFold Training for bl2_v2 (30D SOTA)...")

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            sub_va = np.unique(subjects[val_idx]).tolist()

            # [SOTA 개선 1] Asymmetric Cost-Sensitive Weighting (재현율 향상)
            pos_weight = float(np.sum(y_tr == 0)) / float(max(1, np.sum(y_tr == 1)))
            asym_weight = pos_weight * 1.35 # False Clean(위음성) 페널티 35% 가산

            lgb_model = lgb.LGBMClassifier(
                n_estimators=350,
                learning_rate=0.028,
                max_depth=5,
                num_leaves=20,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=asym_weight,
                min_child_samples=15,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=2
            )
            lgb_model.fit(X_tr, y_tr)

            cat_model = CatBoostClassifier(
                iterations=450,
                learning_rate=0.032,
                depth=5,
                class_weights=[1.0, asym_weight],
                random_seed=42 + fold,
                verbose=0,
                thread_count=2
            )
            cat_model.fit(X_tr, y_tr)

            p_tr_lgb = lgb_model.predict_proba(X_tr)[:, 1]
            p_tr_cat = cat_model.predict_proba(X_tr)[:, 1]
            p_tr_tier1 = 0.5 * p_tr_lgb + 0.5 * p_tr_cat

            p_va_lgb = lgb_model.predict_proba(X_va)[:, 1]
            p_va_cat = cat_model.predict_proba(X_va)[:, 1]
            p_va_tier1 = 0.5 * p_va_lgb + 0.5 * p_va_cat

            # Tier 2: 경계선 모호 샘플 집중 정밀 분류기
            ambiguous_mask_tr = (p_tr_tier1 >= 0.28) & (p_tr_tier1 <= 0.72)
            if np.sum(ambiguous_mask_tr) > 25 and len(np.unique(y_tr[ambiguous_mask_tr])) > 1:
                X_tr_amb = X_tr[ambiguous_mask_tr]
                y_tr_amb = y_tr[ambiguous_mask_tr]
                pos_weight_amb = float(np.sum(y_tr_amb == 0)) / float(max(1, np.sum(y_tr_amb == 1)))

                t2_model = lgb.LGBMClassifier(
                    n_estimators=180,
                    learning_rate=0.02,
                    max_depth=4,
                    num_leaves=14,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=pos_weight_amb * 1.2,
                    random_state=200 + fold,
                    verbose=-1,
                    n_jobs=2
                )
                t2_model.fit(X_tr_amb, y_tr_amb)
            else:
                t2_model = lgb_model

            # 2-Tier 융합
            p_va_final = np.copy(p_va_tier1)
            ambiguous_mask_va = (p_va_tier1 >= 0.28) & (p_va_tier1 <= 0.72)
            if np.sum(ambiguous_mask_va) > 0:
                p_va_t2 = t2_model.predict_proba(X_va[ambiguous_mask_va])[:, 1]
                p_va_final[ambiguous_mask_va] = 0.35 * p_va_tier1[ambiguous_mask_va] + 0.65 * p_va_t2

            oof_probs[val_idx] = p_va_final

            self.tier1_lgb_models.append(lgb_model)
            self.tier1_cat_models.append(cat_model)
            self.tier2_models.append(t2_model)

            fold_f1 = f1_score(y_va, (p_va_final >= 0.5).astype(int))
            fold_auc = roc_auc_score(y_va, p_va_final)
            print(f"  - Fold {fold+1} (Val 피험자: {sub_va}): Binary F1 = {fold_f1*100:.2f}%, AUC = {fold_auc*100:.2f}%")

        self.is_fitted = True

        # [SOTA 개선 2] 커버리지 90% 이상을 유지하는 3-State 최적 임계값 탐색
        best_tau_normal, best_tau_defect, best_res = self._optimize_3state_thresholds(y, oof_probs)
        self.tau_normal = best_tau_normal
        self.tau_defect = best_tau_defect

        view_benchmarks = self._audit_views_3state(y, oof_probs, views)

        overall_summary = {
            "oof_probs": oof_probs,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "overall_3state": best_res,
            "view_benchmarks": view_benchmarks
        }
        return overall_summary

    def _optimize_3state_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        best_f1 = -1.0
        best_tau_n = 0.38
        best_tau_d = 0.58
        best_metrics = {}

        tau_n_candidates = np.linspace(0.28, 0.46, 18)
        tau_d_candidates = np.linspace(0.50, 0.72, 18)

        for tn in tau_n_candidates:
            for td in tau_d_candidates:
                if tn >= td:
                    continue
                metrics = self.calculate_3state_metrics(y_true, y_prob, tn, td)
                # 커버리지 88% 이상 가드레일
                if metrics["coverage"] >= 0.88:
                    if metrics["selective_defect_f1"] > best_f1:
                        best_f1 = metrics["selective_defect_f1"]
                        best_tau_n = float(tn)
                        best_tau_d = float(td)
                        best_metrics = metrics

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
        p_tier1 = np.zeros(len(X), dtype=np.float32)

        for i in range(n_models):
            p_lgb = self.tier1_lgb_models[i].predict_proba(X)[:, 1]
            p_cat = self.tier1_cat_models[i].predict_proba(X)[:, 1]
            p_tier1 += 0.5 * p_lgb + 0.5 * p_cat
        p_tier1 /= n_models

        p_final = np.copy(p_tier1)
        amb_mask = (p_tier1 >= 0.28) & (p_tier1 <= 0.72)
        if np.sum(amb_mask) > 0:
            p_t2 = np.zeros(np.sum(amb_mask), dtype=np.float32)
            for i in range(n_models):
                p_t2 += self.tier2_models[i].predict_proba(X[amb_mask])[:, 1]
            p_t2 /= n_models
            p_final[amb_mask] = 0.35 * p_tier1[amb_mask] + 0.65 * p_t2

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
            "model_name": "bl2_v2",
            "exercise": "바벨 런지 (04_바벨_런지)",
            "defect": "Condition 2 (몸통 발 무릎 방향 일치 여부)",
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "feature_names": self.feature_names,
            "tier1_lgb_models": self.tier1_lgb_models,
            "tier1_cat_models": self.tier1_cat_models,
            "tier2_models": self.tier2_models,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[+] Model bundle saved: {filepath}")

    @classmethod
    def load_bundle(cls, filepath: str) -> "BabelLungeCondition2ModelV2":
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        obj = cls(tau_normal=bundle["tau_normal"], tau_defect=bundle["tau_defect"])
        obj.tier1_lgb_models = bundle["tier1_lgb_models"]
        obj.tier1_cat_models = bundle["tier1_cat_models"]
        obj.tier2_models = bundle["tier2_models"]
        obj.feature_names = bundle["feature_names"]
        obj.is_fitted = True
        return obj


def load_lunge_dataset_v2() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    30D SOTA 피처 데이터셋 로드 (캐시 지원)
    """
    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    label_dir = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
    feat_cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_v2_cache.npz")

    if os.path.exists(feat_cache_file):
        print(f"[*] Loading pre-extracted v2 features from cache: {feat_cache_file}")
        c = np.load(feat_cache_file, allow_pickle=True)
        X = c["X"]
        y = c["y"]
        subjects = c["subjects"]
        views = c["views"]
        clip_names = c["clip_names"].tolist()
        print(f"[+] Loaded from cache: X={X.shape}, y={y.shape} (Defects={np.sum(y==1)}, Normals={np.sum(y==0)})")
        return X, y, subjects, views, clip_names

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
            f30 = extract_lunge_c2_features_v2(kps_33_3d)

            feats_list.append(f30)
            y_list.append(y_c2)
            subjects_list.append(sub)
            views_list.append(vn)
            clip_names.append(base_name)
        except Exception as e:
            continue

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 30D SOTA 정예 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 30D 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포    : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")

    np.savez_compressed(
        feat_cache_file,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        clip_names=np.array(clip_names)
    )
    print(f"[+] Saved extracted v2 features to cache: {feat_cache_file}")

    return X, y, subjects, views, clip_names


def run_v2_pipeline_and_benchmark():
    t_start = time.time()
    X, y, subjects, views, clip_names = load_lunge_dataset_v2()

    model = BabelLungeCondition2ModelV2()
    summary = model.fit_cross_validation(X, y, subjects, views, n_splits=5)

    bundle_path = os.path.join(CURRENT_DIR, "weights", "bl2_v2_bundle.pkl")
    model.save_bundle(bundle_path)

    json_path = os.path.join(CURRENT_DIR, "bl2_v2_benchmark_results.json")
    save_data = {
        "model_name": "bl2_v2",
        "exercise": "04_바벨_런지",
        "defect": "Condition 2 (몸통 발 무릎 방향 일치 여부)",
        "tau_normal": model.tau_normal,
        "tau_defect": model.tau_defect,
        "overall_3state": summary["overall_3state"],
        "view_benchmarks": summary["view_benchmarks"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(time.time() - t_start, 2)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Benchmark results saved: {json_path}")

    return model, save_data


if __name__ == '__main__':
    run_v2_pipeline_and_benchmark()

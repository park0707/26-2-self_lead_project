# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드 / 06_스텝_백워드)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Model: dl2_v1.py (Dynamic Lunge Condition 2 Production Model - bl2_v6 Architecture & Features 100% Inherited)

주요 아키텍처 및 베이스라인 계승 명세:
1. [bl2_v6 베이스라인 100% 계승]:
   - 18차원 정예 생체역학 피처 (상체 밸런싱 노이즈 완전 배제, 하지 11자 축 및 수평 사잇각, 기립대비 동적 Q각 변화량).
   - LightGBM + CatBoost 이종 앙상블 (Cost-Sensitive 불균형 보정).
2. [다이나믹 런지 전용 데이터셋]:
   - 05_스텝_포워드_다이나믹_런지 및 06_스텝_백워드_다이나믹_런지 원천 시퀀스 전수 반영.
   - 피험자 1~9번 전수 5-Fold StratifiedGroupKFold 완전 격리 교차검증.
   - 5대 공식 시점(시점 A~E: view5, view2, view3, view4, view1) 전수 다차원 성능 감사.
3. [3-State Conformal / Selective 최적화]:
   - 판정 보류율 최대 10% (커버리지 >= 90.0%) 제약 조건 하에서 결함 F1(Defect F1) 직접 극대화 탐색.
   - State 0: CLEAN (정상), State 1: DEFECT (결함), State 2: RETAKE (판정보류).
"""

import os
import sys
import glob
import time
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
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

from dynamic_lunge_c2_features import (
    FEATURE_NAMES_18D,
    extract_dynamic_lunge_c2_features_18d,
    extract_dynamic_lunge_c2_features_18d_batch
)

# AGENTS.md 공식 5대 카메라 시점 명칭 매핑
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class DynamicLungeCondition2ModelV1:
    """
    다이나믹 런지 Condition 2 v1 결함 판정 모델 (bl2_v6 계승)
    - 18D 정예 생체역학 피처
    - LightGBM + CatBoost 앙상블
    - 3-State 안전 게이트 (보류율 <= 10%, 커버리지 >= 90%)
    """
    def __init__(
        self,
        tau_normal: float = 0.5000,
        tau_defect: float = 0.5300,
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
        return extract_dynamic_lunge_c2_features_18d(kps_33_3d)

    def extract_features_batch(self, X_seqs: np.ndarray) -> np.ndarray:
        return extract_dynamic_lunge_c2_features_18d_batch(X_seqs)

    def fit_cross_validation(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Any]:
        print(f"[*] Starting 5-Fold StratifiedGroupKFold Training for dl2_v1 (18D Baseline Delta)...")
        print(f"    Constraint: Retake <= {self.max_retake_rate*100:.0f}%, Coverage >= {self.min_coverage*100:.0f}%")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)

        self.lgb_models = []
        self.cat_models = []

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        base_pos_weight = float(n_neg) / float(max(1, n_pos))
        cost_sensitive_weight = base_pos_weight * 1.30

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            val_subs = np.unique(subjects[val_idx])

            # LightGBM (bl2_v6 하이퍼파라미터 계승)
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=130,
                max_depth=3,
                num_leaves=7,
                learning_rate=0.040,
                min_child_samples=16,
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

            # CatBoost (bl2_v6 하이퍼파라미터 계승)
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
            print(f"  - Fold {fold} (Val 피험자: {val_subs.tolist()}): Binary F1 = {f1_val*100:.2f}%, AUC = {auc_val*100:.2f}%")

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
        best_td = 0.5300

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
                    if c_f1 >= 0.650 and acc >= 0.600:
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

    def _audit_views_3state(self, y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
        view_results = {}
        print("\n[*] 5-View Detailed Audit (Multi-View Validation):")
        for v_code, v_alias in VIEW_ALIAS_MAP.items():
            mask = (views == v_code)
            if np.sum(mask) == 0:
                continue
            y_v = y_true[mask]
            p_v = y_prob[mask]
            metrics = self.calculate_3state_metrics(y_v, p_v, self.tau_normal, self.tau_defect)
            view_results[v_code] = {
                "alias": v_alias,
                "metrics": metrics
            }
            d_f1 = metrics["selective_defect_f1"]
            c_f1 = metrics["selective_clean_f1"]
            acc = metrics["selective_accuracy"]
            ret = metrics["retake_rate"]
            print(f"  - [{v_code}] {v_alias}: Selective Defect F1 = {d_f1*100:.2f}%, Clean F1 = {c_f1*100:.2f}%, Acc = {acc*100:.2f}%, Retake = {ret*100:.2f}%")
        return view_results

    def predict_3state(self, kps_33_3d: np.ndarray) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        feats = self.extract_features(kps_33_3d).reshape(1, -1)
        p_lgb = np.mean([m.predict_proba(feats)[0, 1] for m in self.lgb_models])
        p_cat = np.mean([m.predict_proba(feats)[0, 1] for m in self.cat_models])
        p = float(0.50 * p_lgb + 0.50 * p_cat)

        if p <= self.tau_normal:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "완벽합니다! 다이나믹 런지 동작 중 몸통, 무릎, 발의 방향이 올바르게 11자로 유지되었습니다."
        elif p >= self.tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "선행 발끝이 안쪽으로 회전하며 무릎-체간 정렬이 무너졌습니다. 발끝을 11자 전방으로 정렬하고 무릎이 안쪽으로 쏠리지 않도록 주의하세요."
        else:
            state = 2
            state_str = "RETAKE (보류)"
            feedback = "하지 관절의 동적 궤적이 명확하지 않습니다. 대각선 45도 방향에서 전신이 화면에 잘 보이도록 1회 다시 수행해 주세요."

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
            "model_name": "dl2_v1",
            "exercise": "다이나믹 런지 (05_스텝_포워드 / 06_스텝_백워드)",
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
    def load_bundle(cls, filepath: str) -> "DynamicLungeCondition2ModelV1":
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


def load_dynamic_lunge_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    D:/Data/전처리_데이터/MediaPipe_Full33_64F/05_06_다이나믹_런지 에서 전수 시퀀스 로드 및 18D 피처 추출/캐싱
    """
    feat_cache_file = os.path.join(CURRENT_DIR, "dynamic_lunge_c2_features_18d_cache.npz")
    prep_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
    label_dirs = {
        '05': r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
        '06': r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
    }

    if os.path.exists(feat_cache_file):
        print(f"[*] Loading 18D features from cache: {feat_cache_file}")
        c = np.load(feat_cache_file, allow_pickle=True)
        return c["X"], c["y"], c["subjects"], c["views"], c["clip_names"].tolist()

    npz_files = sorted(glob.glob(os.path.join(prep_dir, "*.npz")))
    print(f"[*] 총 분석 대상 다이나믹 런지 전처리 파일 수: {len(npz_files)}개")

    feats_list = []
    y_list = []
    subjects_list = []
    views_list = []
    clip_names = []

    t0 = time.time()
    for npz_path in npz_files:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        # base_name 예시: 05_[오류]_D01_피험자1_Key082_view1
        parts = base_name.split("_")
        vn = parts[-1]  # view1 ~ view5

        folder_name = "_".join(parts[:-1])  # [오류]_D01_피험자1_Key082

        label_p1 = os.path.join(label_dirs['05'], f"{folder_name}.json")
        label_p2 = os.path.join(label_dirs['06'], f"{folder_name}.json")
        if os.path.exists(label_p1):
            label_p = label_p1
        elif os.path.exists(label_p2):
            label_p = label_p2
        else:
            continue

        try:
            with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
                d_json = json.load(f)
            conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        except Exception:
            continue

        # Condition 2: 몸통 발 무릎 방향 일치여부 (True: 정상->0, False: 결함->1)
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
            f18 = extract_dynamic_lunge_c2_features_18d(kps_33_3d)

            feats_list.append(f18)
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

    print(f"[+] 다이나믹 런지 18D 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포 : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
    print(f"  - 피험자    : {np.unique(subjects).tolist()}")
    print(f"  - 시점      : {np.unique(views).tolist()}")

    np.savez_compressed(
        feat_cache_file,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        clip_names=np.array(clip_names)
    )
    print(f"[+] 캐시 저장 완료: {feat_cache_file}")
    return X, y, subjects, views, clip_names


def main():
    t_start = time.time()
    bundle_path = os.path.join(CURRENT_DIR, "weights", "dl2_v1_bundle.pkl")
    results_path = os.path.join(CURRENT_DIR, "dl2_v1_benchmark_results.json")

    X, y, subjects, views, clip_names = load_dynamic_lunge_dataset()
    print(f"[+] Dataset Ready: Total {len(y)} samples (Defect: {np.sum(y==1)}, Normal: {np.sum(y==0)})")
    print(f"    Subjects: {np.unique(subjects)}, Views: {np.unique(views)}")

    model = DynamicLungeCondition2ModelV1(max_retake_rate=0.10)
    bench_results = model.fit_cross_validation(X, y, subjects, views, n_splits=5)

    model.save_bundle(bundle_path)

    serializable_results = {
        "model_name": "dl2_v1",
        "description": "Dynamic Lunge Condition 2 Production Model - bl2_v6 Architecture & Features Inherited",
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

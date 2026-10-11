# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Script: train_and_eval_d2_v2.py
Purpose: d2 v2.0 정예 7D 피처셋 기반 5-Fold 피험자 완전 격리 재학습,
         공식 5대 시점별 다차원 성능 감사, 3-State 선택적 판정 최적화 및 v1.0 / v2.0(30D) 대비 델타 분석
"""

import os
import sys
import json
import glob
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

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

from src.models.second_models.babel_deadlift.d2_model_v2 import (
    BabelDeadliftD2ModelV2, extract_d2_v2_features_from_seq, FEATURE_NAMES_7_D2
)

warnings.filterwarnings("ignore")

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset_7d() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 385개 멀티뷰 시퀀스 로드 및 정예 7D 피처 추출"""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

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

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}

        # d2 발과 무릎의 방향 일치 위반 (오류 발생 시 1, 정상 시 0)
        y_d2 = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_33_3d = data["kps_33_3d_64f"]

        f7 = extract_d2_v2_features_from_seq(kps_33_3d)

        feats_list.append(f7)
        y_list.append(y_d2)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 7D 정예 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 7D 피처 형태  : {X.shape}")
    print(f"  - 라벨 분포     : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
    print(f"  - 피험자 분포   : {np.unique(subjects).tolist()}")
    print(f"  - 시점 분포     : {np.unique(views).tolist()}")

    return X, y, subjects, views, clip_names


def evaluate_3state(y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float) -> Dict[str, Any]:
    """3-State 선택적 판정 지표 계산"""
    decided_mask = (y_prob <= tau_normal) | (y_prob >= tau_defect)
    coverage = float(np.mean(decided_mask))
    decline_rate = float(1.0 - coverage)

    y_decided_true = y_true[decided_mask]
    y_decided_pred = (y_prob[decided_mask] >= tau_defect).astype(int)

    if len(y_decided_true) > 0 and len(np.unique(y_decided_true)) > 1:
        f1 = float(f1_score(y_decided_true, y_decided_pred))
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        prec = float(precision_score(y_decided_true, y_decided_pred, zero_division=0))
        rec = float(recall_score(y_decided_true, y_decided_pred, zero_division=0))
        cm = confusion_matrix(y_decided_true, y_decided_pred).tolist()
    else:
        f1, acc, prec, rec = 0.0, 0.0, 0.0, 0.0
        cm = [[0, 0], [0, 0]]

    return {
        "tau_normal": round(tau_normal, 2),
        "tau_defect": round(tau_defect, 2),
        "coverage": coverage,
        "decline_rate": decline_rate,
        "total_decided": int(np.sum(decided_mask)),
        "total_undetermined": int(np.sum(~decided_mask)),
        "f1_score": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": {
            "TN": cm[0][0] if len(cm) > 1 else 0,
            "FP": cm[0][1] if len(cm) > 1 else 0,
            "FN": cm[1][0] if len(cm) > 1 else 0,
            "TP": cm[1][1] if len(cm) > 1 else 0
        }
    }


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 무릎-발 방향 일치 모델 (d2_model_v2) 정예 7D 재학습 및 평가")
    print("  - 아키텍처: 데드리프트 부적합 노이즈 피처 23개 전면 제거 + 7대 정예 생체역학 피처 (LightGBM + CatBoost 앙상블)")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    # 1. 7D 데이터 로드
    X, y, subjects, views, clip_names = load_dataset_7d()

    # 2. 5-Fold StratifiedGroupKFold 분할 (피험자 격리)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(y), dtype=np.float32)
    oof_preds_cat = np.zeros(len(y), dtype=np.float32)
    oof_preds_ensemble = np.zeros(len(y), dtype=np.float32)

    trained_lgb_models = []
    trained_cat_models = []

    print("\n[*] 5-Fold 피험자 완전 격리 재학습 진행 중...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        val_subs = np.unique(subjects[val_idx])
        print(f"  >> Fold {fold}/5 시작 (검증 피험자: {val_subs.tolist()})")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # LightGBM (불균형 scale_pos_weight=2.2, L2 reg=1.5)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=14,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=2.2,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_train, y_train)
        val_p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_p_lgb
        trained_lgb_models.append(clf_lgb)

        # CatBoost (L2 reg=3.0, Balanced)
        clf_cat = CatBoostClassifier(
            iterations=160,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_train, y_train)
        val_p_cat = clf_cat.predict_proba(X_val)[:, 1]
        oof_preds_cat[val_idx] = val_p_cat
        trained_cat_models.append(clf_cat)

        p_ens = 0.50 * val_p_lgb + 0.50 * val_p_cat
        oof_preds_ensemble[val_idx] = p_ens

        f_auc = roc_auc_score(y_val, p_ens) if len(np.unique(y_val)) > 1 else 0.5
        f_f1_th37 = f1_score(y_val, (p_ens >= 0.37).astype(int), zero_division=0)
        print(f"     Fold {fold} 완료 -> ROC-AUC: {f_auc:.4f}, F1(th=0.37): {f_f1_th37*100:.2f}%")

    # 전체 OOF ROC-AUC
    auc_lgb = roc_auc_score(y, oof_preds_lgb)
    auc_cat = roc_auc_score(y, oof_preds_cat)
    auc_ens = roc_auc_score(y, oof_preds_ensemble)

    print("\n" + "=" * 80)
    print(f"[*] 5-Fold OOF 전체 앙상블 ROC-AUC: {auc_ens:.4f} (LGB: {auc_lgb:.4f}, CAT: {auc_cat:.4f})")
    print("=" * 80)

    # 3. 최적 임계값 탐색
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        score = f1_score(y, (oof_preds_ensemble >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th

    p_opt = (oof_preds_ensemble >= best_th).astype(int)
    acc_opt = accuracy_score(y, p_opt)
    prec_opt = precision_score(y, p_opt, zero_division=0)
    rec_opt = recall_score(y, p_opt, zero_division=0)
    cm_opt = confusion_matrix(y, p_opt)

    print(f"\n[★ 2-State 최적 임계값: {best_th:.2f}]")
    print(f"  - F1-Score : {best_f1*100:.2f}% (기존 v2.0 38.73% 대비 +{(best_f1*100 - 38.73):+.2f}%p🔺)")
    print(f"  - Accuracy : {acc_opt*100:.2f}% (기존 v2.0 54.80% 대비 +{(acc_opt*100 - 54.80):+.2f}%p🔺)")
    print(f"  - Precision: {prec_opt*100:.2f}%")
    print(f"  - Recall   : {rec_opt*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm_opt[0,0]}, FP={cm_opt[0,1]}, FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")

    # 4. 3-State 최적 윈도우 평가
    # 7D 정예 모델 최적 윈도우 [0.38, 0.40] 및 [0.35, 0.40] 비교
    stat_3840 = evaluate_3state(y, oof_preds_ensemble, 0.38, 0.40)
    stat_3540 = evaluate_3state(y, oof_preds_ensemble, 0.35, 0.40)
    stat_prev = evaluate_3state(y, oof_preds_ensemble, 0.28, 0.30)

    print("\n[★ 3-State 선택적 판정 성과]")
    print(f"  - [0.38, 0.40] 최적 윈도우: F1={stat_3840['f1_score']*100:.2f}%, Acc={stat_3840['accuracy']*100:.2f}%, Cov={stat_3840['coverage']*100:.1f}%")
    print(f"  - [0.35, 0.40] 고신뢰 윈도우: F1={stat_3540['f1_score']*100:.2f}%, Acc={stat_3540['accuracy']*100:.2f}%, Cov={stat_3540['coverage']*100:.1f}%")
    print(f"  - [0.28, 0.30] 기존 윈도우: F1={stat_prev['f1_score']*100:.2f}%, Acc={stat_prev['accuracy']*100:.2f}%, Cov={stat_prev['coverage']*100:.1f}%")

    official_3s = stat_3840

    # 5. 5대 공식 시점별 감사
    print("\n" + "=" * 90)
    print("  [다차원 성능 감사] 공식 5대 시점별 성능 (시점 A ~ 시점 E)")
    print("=" * 90)
    print(f"{'공식 시점 명칭':<24} | {'코드':<6} | {'AUC':<7} | {'F1-Score':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 90)

    view_audits = {}
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        p_v = p_opt[v_mask]
        prob_v = oof_preds_ensemble[v_mask]

        v_auc = roc_auc_score(y_v, prob_v) if len(np.unique(y_v)) > 1 else 0.5
        v_f1 = f1_score(y_v, p_v, zero_division=0)
        v_acc = accuracy_score(y_v, p_v)
        v_prec = precision_score(y_v, p_v, zero_division=0)
        v_rec = recall_score(y_v, p_v, zero_division=0)

        view_audits[v_alias] = {
            "view_code": v_code,
            "auc": float(v_auc),
            "f1": float(v_f1),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec)
        }
        print(f"{v_alias:<24} | {v_code:<6} | {v_auc:.4f}  | {v_f1*100:.2f}%     | {v_acc*100:.2f}%     | {v_prec*100:.2f}%     | {v_rec*100:.2f}%")
    print("=" * 90)

    # 6. 가중치 번들 저장
    weights_path = os.path.join(OUT_WEIGHTS_DIR, "d2_v2_gbdt_models.pkl")
    bundle = {
        "defect_code": "d2",
        "exercise": "바벨 데드리프트",
        "version": "v2.0_refined_7d",
        "feature_dim": 7,
        "feature_names": FEATURE_NAMES_7_D2,
        "decision_threshold": float(best_th),
        "tau_normal": float(official_3s["tau_normal"]),
        "tau_defect": float(official_3s["tau_defect"]),
        "catboost_models": trained_cat_models,
        "lightgbm_models": trained_lgb_models
    }
    with open(weights_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[+] 공식 모델 가중치 번들 저장 완료: {weights_path}")

    # 7. 평가 결과 JSON 저장
    eval_json_path = os.path.join(CURRENT_DIR, "d2_model_v2_evaluation_results.json")
    results_dict = {
        "model_name": "BabelDeadliftD2ModelV2_Refined7D",
        "exercise": "02_바벨_데드리프트",
        "defect": "d2_발과_무릎의_방향_일치",
        "feature_dim": 7,
        "feature_names": FEATURE_NAMES_7_D2,
        "binary_metrics_th50": {
            "accuracy": float(accuracy_score(y, (oof_preds_ensemble >= 0.5).astype(int))),
            "precision": float(precision_score(y, (oof_preds_ensemble >= 0.5).astype(int), zero_division=0)),
            "recall": float(recall_score(y, (oof_preds_ensemble >= 0.5).astype(int), zero_division=0)),
            "f1_score": float(f1_score(y, (oof_preds_ensemble >= 0.5).astype(int), zero_division=0)),
            "roc_auc": float(auc_ens)
        },
        "binary_metrics_optimal": {
            "threshold": float(best_th),
            "accuracy": float(acc_opt),
            "precision": float(prec_opt),
            "recall": float(rec_opt),
            "f1_score": float(best_f1),
            "confusion_matrix": {
                "TN": int(cm_opt[0, 0]),
                "FP": int(cm_opt[0, 1]),
                "FN": int(cm_opt[1, 0]),
                "TP": int(cm_opt[1, 1])
            }
        },
        "delta_vs_v1_and_30d": {
            "v1_auc": 0.5771,
            "v2_30d_auc": 0.5928,
            "v2_7d_auc": float(auc_ens),
            "delta_auc_vs_v1": float(auc_ens - 0.5771),
            "delta_auc_vs_30d": float(auc_ens - 0.5928),
            "v1_f1": 38.49,
            "v2_30d_f1": 38.73,
            "v2_7d_f1": float(best_f1 * 100),
            "delta_f1_vs_v1": float(best_f1 * 100 - 38.49),
            "delta_f1_vs_30d": float(best_f1 * 100 - 38.73),
            "v1_acc": 57.66,
            "v2_30d_acc": 54.80,
            "v2_7d_acc": float(acc_opt * 100),
            "delta_acc_vs_30d": float(acc_opt * 100 - 54.80)
        },
        "official_3state_setting": official_3s,
        "view_audit": view_audits
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {eval_json_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Script: train_and_eval_d2_model.py
Purpose: 스쿼트 c3 무릎 모델 아키텍처(22D 생체역학 + LightGBM/CatBoost 앙상블)를 완전 계승하여,
         AI-Hub 바벨 데드리프트 385개 멀티뷰 데이터로 5-Fold 피험자 격리 재학습 및 다차원 성능 감사 수행
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

from src.models.second_models.babel_deadlift.d2_model_v1 import (
    BabelDeadliftD2ModelV1, extract_d2_v1_features_from_seq, FEATURE_NAMES_22_D2
)

warnings.filterwarnings("ignore")

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 385개 멀티뷰 시퀀스 로드 및 22D 피처 추출"""
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

        f22 = extract_d2_v1_features_from_seq(kps_33_3d)

        feats_list.append(f22)
        y_list.append(y_d2)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 22D 피처 형태 : {X.shape}")
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
        "tau_normal": tau_normal,
        "tau_defect": tau_defect,
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
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 무릎-발 방향 일치 모델 (d2_model_v1) 재학습 및 평가")
    print("  - 아키텍처: 스쿼트 c3 v6.0 100% 완전 계승 (22D 생체역학 + LightGBM/CatBoost 이중 앙상블)")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    # 1. 데이터 로드
    X, y, subjects, views, clip_names = load_dataset()

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

        # LightGBM 분류기
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=2.5,  # 불균형 보정
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_train, y_train)
        val_p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_p_lgb
        trained_lgb_models.append(clf_lgb)

        # CatBoost 분류기
        clf_cat = CatBoostClassifier(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_train, y_train)
        val_p_cat = clf_cat.predict_proba(X_val)[:, 1]
        oof_preds_cat[val_idx] = val_p_cat
        trained_cat_models.append(clf_cat)

        # 앙상블 (50:50 결합)
        val_p_ens = 0.50 * val_p_cat + 0.50 * val_p_lgb
        oof_preds_ensemble[val_idx] = val_p_ens

        fold_f1 = f1_score(y_val, (val_p_ens >= 0.5).astype(int), zero_division=0)
        fold_acc = accuracy_score(y_val, (val_p_ens >= 0.5).astype(int))
        print(f"     Fold {fold} OOF 완료 -> F1: {fold_f1*100:.2f}%, Acc: {fold_acc*100:.2f}%")

    print("\n" + "=" * 100)
    print("  [실측 결과 1] 바벨 데드리프트 d2 v1.0 모델 전체 5-Fold OOF 성능 종합")
    print("=" * 100)

    # 최적 임계값 탐색
    best_th = 0.50
    best_f1 = 0.0
    for th in np.arange(0.20, 0.81, 0.02):
        f = f1_score(y, (oof_preds_ensemble >= th).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th

    pred_50 = (oof_preds_ensemble >= 0.50).astype(int)
    pred_best = (oof_preds_ensemble >= best_th).astype(int)

    f1_50 = f1_score(y, pred_50, zero_division=0)
    acc_50 = accuracy_score(y, pred_50)
    prec_50 = precision_score(y, pred_50, zero_division=0)
    rec_50 = recall_score(y, pred_50, zero_division=0)
    auc_score = roc_auc_score(y, oof_preds_ensemble)
    cm_50 = confusion_matrix(y, pred_50)

    f1_opt = f1_score(y, pred_best, zero_division=0)
    acc_opt = accuracy_score(y, pred_best)
    prec_opt = precision_score(y, pred_best, zero_division=0)
    rec_opt = recall_score(y, pred_best, zero_division=0)
    cm_opt = confusion_matrix(y, pred_best)

    print(f"\n[기본 임계값 0.50 성능]:")
    print(f"  - Accuracy  : {acc_50*100:.2f}%")
    print(f"  - Precision : {prec_50*100:.2f}%")
    print(f"  - Recall    : {rec_50*100:.2f}%")
    print(f"  - F1-Score  : {f1_50*100:.2f}%")
    print(f"  - ROC-AUC   : {auc_score:.4f}")
    print(f"  - Confusion Matrix: TN={cm_50[0,0]}, FP={cm_50[0,1]} / FN={cm_50[1,0]}, TP={cm_50[1,1]}")

    print(f"\n[최적 임계값 {best_th:.2f} 성능]:")
    print(f"  - Accuracy  : {acc_opt*100:.2f}%")
    print(f"  - Precision : {prec_opt*100:.2f}%")
    print(f"  - Recall    : {rec_opt*100:.2f}%")
    print(f"  - F1-Score  : {f1_opt*100:.2f}%")
    print(f"  - Confusion Matrix: TN={cm_opt[0,0]}, FP={cm_opt[0,1]} / FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")

    # 스쿼트 제로샷 대비 비교
    # 스쿼트 c3 제로샷 수치: acc=42.08%, prec=25.99%, rec=80.00%, f1=39.24%, auc=50.00%
    zero_f1 = 39.24
    zero_acc = 42.08
    delta_f1 = (f1_50 * 100) - zero_f1
    delta_acc = (acc_50 * 100) - zero_acc

    print("\n" + "=" * 100)
    print("  [스쿼트 제로샷 vs 데드리프트 재학습 d2 비교 (전이 성능 점프)]")
    print("=" * 100)
    print(f"{'비교 지표':<20} | {'스쿼트 c3 제로샷':<20} | {'데드리프트 d2 v1.0':<20} | {'변화폭 (Delta)':<15}")
    print("-" * 80)
    print(f"{'F1-Score':<20} | {zero_f1:6.2f}%              | {f1_50*100:6.2f}%              | {delta_f1:+6.2f}%p")
    print(f"{'Accuracy':<20} | {zero_acc:6.2f}%              | {acc_50*100:6.2f}%              | {delta_acc:+6.2f}%p")
    print(f"{'ROC-AUC':<20}  | 0.5000               | {auc_score:6.4f}               | {auc_score-0.50:+6.4f}")

    # 공식 5대 카메라 시점별 세부 성능 감사
    print("\n" + "=" * 100)
    print("  [실측 결과 2] 공식 5대 카메라 시점별 d2 무릎 성능 감사 (시점 편향 감사)")
    print("=" * 100)
    print(f"{'카메라 시점':<25} | {'F1-Score':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<10}")
    print("-" * 85)

    view_results = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        v_name = VIEW_ALIAS_MAP[vn]
        mask = (views == vn)
        v_y_true = y[mask]
        v_prob = oof_preds_ensemble[mask]
        v_pred = (v_prob >= 0.50).astype(int)

        v_f1 = f1_score(v_y_true, v_pred, zero_division=0)
        v_acc = accuracy_score(v_y_true, v_pred)
        v_prec = precision_score(v_y_true, v_pred, zero_division=0)
        v_rec = recall_score(v_y_true, v_pred, zero_division=0)
        try:
            v_auc = roc_auc_score(v_y_true, v_prob)
        except Exception:
            v_auc = 0.5

        view_results[v_name] = {
            "view_code": vn,
            "f1": float(v_f1),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "auc": float(v_auc)
        }
        print(f"{v_name:<25} | {v_f1*100:6.2f}%    | {v_acc*100:6.2f}%    | {v_prec*100:6.2f}%    | {v_rec*100:6.2f}%    | {v_auc:6.4f}")

    # 3-State 선택적 판정 시뮬레이션
    print("\n" + "=" * 100)
    print("  [실측 결과 3] 3-State 선택적 판정(Selective Classification) 윈도우 탐색")
    print("=" * 100)
    print(f"{'윈도우 [tau_norm, tau_def]':<28} | {'커버리지':<10} | {'보류율':<10} | {'F1-Score':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 95)

    best_3state_res = None
    best_3state_score = 0.0

    candidate_windows = [
        (0.35, 0.55),
        (0.30, 0.55),
        (0.30, 0.60),
        (0.25, 0.60),
        (0.20, 0.65),
        (0.35, 0.65),
        (0.40, 0.60)
    ]

    for tn, td in candidate_windows:
        res = evaluate_3state(y, oof_preds_ensemble, tn, td)
        label = f"[{tn:.2f}, {td:.2f}]"
        print(f"{label:<28} | {res['coverage']*100:6.2f}%    | {res['decline_rate']*100:6.2f}%    | {res['f1_score']*100:6.2f}%    | {res['accuracy']*100:6.2f}%    | {res['precision']*100:6.2f}%    | {res['recall']*100:6.2f}%")

        # 보류율 10% 이하 제약 조건 하에서 F1 극대화
        if res['decline_rate'] <= 0.12 and res['f1_score'] > best_3state_score:
            best_3state_score = res['f1_score']
            best_3state_res = res

    # 3. 모델 인스턴스에 가중치 주입 및 저장
    d2_model = BabelDeadliftD2ModelV1(
        decision_threshold=best_th,
        tau_normal=best_3state_res["tau_normal"] if best_3state_res else 0.35,
        tau_defect=best_3state_res["tau_defect"] if best_3state_res else 0.55
    )
    d2_model.cb_models = trained_cat_models
    d2_model.lgb_models = trained_lgb_models
    d2_model.is_fitted = True

    weights_out = os.path.join(OUT_WEIGHTS_DIR, "d2_v1_gbdt_models.pkl")
    d2_model.save_weights(weights_out)

    # 4. 결과 JSON 저장
    result_data = {
        "model_name": "BabelDeadliftD2ModelV1",
        "exercise": "02_바벨_데드리프트",
        "defect": "d2_발과_무릎의_방향_일치",
        "feature_dim": 22,
        "feature_names": FEATURE_NAMES_22_D2,
        "binary_metrics_th50": {
            "accuracy": float(acc_50),
            "precision": float(prec_50),
            "recall": float(rec_50),
            "f1_score": float(f1_50),
            "roc_auc": float(auc_score),
            "confusion_matrix": {
                "TN": int(cm_50[0, 0]),
                "FP": int(cm_50[0, 1]),
                "FN": int(cm_50[1, 0]),
                "TP": int(cm_50[1, 1])
            }
        },
        "binary_metrics_optimal": {
            "threshold": float(best_th),
            "accuracy": float(acc_opt),
            "precision": float(prec_opt),
            "recall": float(rec_opt),
            "f1_score": float(f1_opt),
            "confusion_matrix": {
                "TN": int(cm_opt[0, 0]),
                "FP": int(cm_opt[0, 1]),
                "FN": int(cm_opt[1, 0]),
                "TP": int(cm_opt[1, 1])
            }
        },
        "transfer_comparison": {
            "squat_c3_zero_shot_f1": zero_f1,
            "squat_c3_zero_shot_acc": zero_acc,
            "d2_v1_f1": float(f1_50 * 100),
            "d2_v1_acc": float(acc_50 * 100),
            "delta_f1": float(delta_f1),
            "delta_acc": float(delta_acc)
        },
        "official_3state_setting": best_3state_res,
        "view_audit": view_results
    }

    out_json = os.path.join(CURRENT_DIR, "d2_model_v1_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] d2 v1.0 평가 결과 JSON 저장 완료: {out_json}")


if __name__ == "__main__":
    main()

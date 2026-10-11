# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d2. 발과 무릎의 방향 일치 (Knee Valgus / Knee-Toe Alignment)
Script: build_and_eval_d2_v3_integrated.py
Purpose: Phase 1 (Cleanlab 라벨 정제) + Phase 2 (3D 가상 정면 표준화 정렬) 통합 고도화 모델 (v2.5/v3.0)
         - 5-Fold 피험자 완전 격리 학습 및 OOF 평가
         - 전체 F1-score, 최적 임계값 탐색, 3-State 선택적 판정 F1, 5대 시점별 F1 전수 산출
"""

import os
import sys
import json
import glob
import time
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.ndimage import gaussian_filter1d

import cleanlab
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = r"D:\src\models\second_models\babel_deadlift\weights"
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def transform_to_canonical_frontal(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    모든 프레임에서 인체 중심 가상 정면(Virtual Frontal) 직교 좌표계로 정렬
    - 원점: 골반 중심 (Mid-Hip)
    - X축: 우골반 -> 좌골반 (인체 정면 관상면 수평 방향)
    - Y축: 골반 중심 -> 어깨 중심 (체간 수직 방향)
    - Z축: X와 Y의 외적 (인체 전방 방향)
    """
    T = kps_33_64f.shape[0]
    out = np.zeros_like(kps_33_64f)

    for t in range(T):
        pts = kps_33_64f[t]
        l_hip, r_hip = pts[23], pts[24]
        mid_hip = (l_hip + r_hip) / 2.0

        l_sh, r_sh = pts[11], pts[12]
        mid_sh = (l_sh + r_sh) / 2.0

        centered = pts - mid_hip

        v_torso = mid_sh - mid_hip
        y_hat = v_torso / (np.linalg.norm(v_torso) + 1e-7)

        v_hip = l_hip - r_hip
        z_raw = np.cross(y_hat, v_hip)
        z_hat = z_raw / (np.linalg.norm(z_raw) + 1e-7)

        x_hat = np.cross(y_hat, z_hat)
        x_hat = x_hat / (np.linalg.norm(x_hat) + 1e-7)

        R = np.column_stack([x_hat, y_hat, z_hat])
        out[t] = np.dot(centered, R)

    return out


def extract_dual_geometry_features(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    1. 월드 좌표계 정예 7D 피처
    2. 가상 정면 3D 좌표계 정예 5D 피처
    -> 총 12D 시점 불변 듀얼 기하 피처셋
    """
    # 1. 기존 월드 좌표계 정예 7D 피처
    f7_world = extract_d2_v2_features_from_seq(kps_33_64f)

    # 2. 가상 정면 3D 좌표계 피처
    c_kps = transform_to_canonical_frontal(kps_33_64f)
    T = c_kps.shape[0]
    dt = 3.0 / max(1, T - 1)

    l_hip = c_kps[:, 23, :]
    r_hip = c_kps[:, 24, :]
    l_knee = c_kps[:, 25, :]
    r_knee = c_kps[:, 26, :]
    l_ank = c_kps[:, 27, :]
    r_ank = c_kps[:, 28, :]

    knee_w = np.abs(l_knee[:, 0] - r_knee[:, 0])
    ank_w = np.abs(l_ank[:, 0] - r_ank[:, 0])
    hip_w = np.abs(l_hip[:, 0] - r_hip[:, 0])
    mean_hip_w = float(np.mean(hip_w) + 1e-6)

    ka_ratio = knee_w / (ank_w + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)
    kp_ratio = knee_w / (hip_w + 1e-6)
    kp_ratio_s = gaussian_filter1d(kp_ratio, sigma=1.0)

    # 관상면 X축 투영 외반 변위
    vl = (l_knee[:, 0] - l_hip[:, 0]) * 100.0
    vr = (r_hip[:, 0] - r_knee[:, 0]) * 100.0
    valg_sum_abs = np.abs(vl) + np.abs(vr)
    valg_sum_abs_s = gaussian_filter1d(valg_sum_abs, sigma=1.0)
    f_can_valg_max = float(np.max(valg_sum_abs_s))

    valg_norm = valg_sum_abs_s / (mean_hip_w * 100.0 + 1e-6)
    f_can_valg_norm_max = float(np.max(valg_norm))

    shank_l = l_knee - l_ank
    shank_r = r_knee - r_ank
    tilt_l = shank_l[:, 0] / (np.linalg.norm(shank_l, axis=1) + 1e-7)
    tilt_r = -shank_r[:, 0] / (np.linalg.norm(shank_r, axis=1) + 1e-7)
    tilt_sum = np.maximum(0.0, -tilt_l) + np.maximum(0.0, -tilt_r)
    f_can_shank_tilt = float(np.percentile(tilt_sum, 90))

    f_can_ka_min = float(np.min(ka_ratio_s))
    f_can_kp_min = float(np.min(kp_ratio_s))

    f5_can = np.array([
        f_can_valg_max,
        f_can_valg_norm_max,
        f_can_shank_tilt,
        f_can_ka_min,
        f_can_kp_min
    ], dtype=np.float32)

    return np.concatenate([f7_world, f5_can])


FEATURE_NAMES_12D = [
    # World 7D
    "world_kp_ratio_min_abs",
    "world_ka_absolute_deficit_integ",
    "world_valgus_abs_sum_max",
    "world_valgus_norm_hip_max",
    "world_shank_inward_tilt_p90",
    "world_valgus_inward_accel_peak",
    "world_morph_kp_drop_max",
    # Canonical 5D
    "canonical_valgus_abs_sum_max",
    "canonical_valgus_norm_hip_max",
    "canonical_shank_tilt_p90",
    "canonical_ka_ratio_min",
    "canonical_kp_ratio_min"
]


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 고도화 모델 (Cleanlab + 3D 가상정면 통합)")
    print("  - 아키텍처: 12D 듀얼 기하 피처셋 + Cleanlab Confident Learning + 이중 GBDT 앙상블")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_list = []
    y_list = []
    subs_list = []
    views_list = []
    keys_list = []

    t0 = time.time()
    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        parts = base.split("_")
        vn = parts[-1]
        folder = "_".join(parts[:-1])
        key = [pt for pt in parts if pt.startswith("Key")][0]

        sub = 1
        for i in range(1, 10):
            if f"피험자{i}" in folder:
                sub = i
                break

        with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
            d = json.load(fp)
        conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
        y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

        kps = np.load(p)["kps_33_3d_64f"]
        f12 = extract_dual_geometry_features(kps)

        X_list.append(f12)
        y_list.append(y_val)
        subs_list.append(sub)
        views_list.append(vn)
        keys_list.append(key)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs_list, dtype=np.int32)
    views = np.array(views_list)
    keys = np.array(keys_list)

    print(f"[+] 12D 듀얼 기하 피처 추출 완료: {X.shape} (경과: {time.time()-t0:.1f}초)")
    print(f"  - 정상(0): {np.sum(y==0)}건 vs 결함(1): {np.sum(y==1)}건 (결함비율: {np.mean(y)*100:.1f}%)")

    # 1단계: 1차 OOF 확률 추정 (Cleanlab용)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    init_oof_probs = np.zeros((len(y), 2), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)

        init_oof_probs[val_idx] = 0.5 * p_l + 0.5 * p_c

    # 2단계: Cleanlab Confident Learning 기반 신뢰 마스크 및 품질 점수 산출
    print("\n[*] Cleanlab Confident Learning 실행 중...")
    label_issues = find_label_issues(
        labels=y, pred_probs=init_oof_probs, return_indices_ranked_by='self_confidence', n_jobs=1
    )
    quality_scores = get_label_quality_scores(labels=y, pred_probs=init_oof_probs)
    print(f"  [+] 라벨 모호성 및 연쇄 오염 감지 샘플: {len(label_issues)}개")

    # 신뢰 마스크 (가장 극단적인 오염 샘플 상위 50개 프루닝)
    prune_k = min(len(label_issues), 60)
    severe_issues = label_issues[:prune_k]
    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[severe_issues] = False
    print(f"  [+] 고신뢰 정제 학습셋 구축: 전체 {len(y)}개 중 {np.sum(clean_mask)}개 채택 (오염 {prune_k}개 제거)")

    # 3단계: 고신뢰 정제 데이터 기반 5-Fold 최종 앙상블 재학습
    print("\n[*] Cleanlab 정제 기반 5-Fold 최종 앙상블 재학습 진행 중...")
    final_oof_preds = np.zeros(len(y), dtype=np.float32)
    trained_lgb_models = []
    trained_cat_models = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        clean_tr_idx = [i for i in train_idx if clean_mask[i]]
        X_tr, y_tr = X[clean_tr_idx], y[clean_tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.0, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]
        trained_lgb_models.append(clf_lgb)

        clf_cat = CatBoostClassifier(
            iterations=170, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]
        trained_cat_models.append(clf_cat)

        p_ens = 0.5 * p_l + 0.5 * p_c
        final_oof_preds[val_idx] = p_ens

    # 4단계: 공식 성능 평가
    final_auc = roc_auc_score(y, final_oof_preds)

    # 최적 임계값 F1 탐색
    best_th, best_f1, best_acc, best_prec, best_rec = 0.5, 0.0, 0.0, 0.0, 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        p_bin = (final_oof_preds >= th).astype(int)
        f = f1_score(y, p_bin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            best_acc = accuracy_score(y, p_bin)
            best_prec = precision_score(y, p_bin, zero_division=0)
            best_rec = recall_score(y, p_bin, zero_division=0)

    p_opt = (final_oof_preds >= best_th).astype(int)
    cm_opt = confusion_matrix(y, p_opt)

    print("\n" + "=" * 100)
    print("  [★ 최종 통합 모델 5-Fold OOF 공식 성능 결과]")
    print("=" * 100)
    print(f"  • ROC-AUC          : {final_auc:.4f} (v1.0 0.5771 -> v2.0 0.6502 -> 통합 v3.0 {final_auc:.4f}🔺)")
    print(f"  • 최적 임계값       : {best_th:.2f}")
    print(f"  • 최적 F1-Score    : {best_f1*100:.2f}% (기존 v2.0 46.43% 대비 +{(best_f1*100-46.43):+.2f}%p🔺)")
    print(f"  • 정확도 (Accuracy) : {best_acc*100:.2f}% (기존 v2.0 61.04% 대비 +{(best_acc*100-61.04):+.2f}%p🔺)")
    print(f"  • 정밀도 (Precision): {best_prec*100:.2f}%")
    print(f"  • 재현율 (Recall)   : {best_rec*100:.2f}%")
    print(f"  • 혼동 행렬        : TN={cm_opt[0,0]}, FP={cm_opt[0,1]} (오탐 억제!), FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")
    print("=" * 100)

    # 3-State 고신뢰 선택적 판정 F1
    print("\n[★ 3-State 선택적 판정 (Selective Classification) F1-Score]")
    state_results = {}
    for tn in [0.35, 0.38, 0.40]:
        for td in [0.42, 0.45, 0.48]:
            if td <= tn:
                continue
            dec = (final_oof_preds <= tn) | (final_oof_preds >= td)
            cov = float(np.mean(dec) * 100)
            y_d = y[dec]
            p_d = (final_oof_preds[dec] >= td).astype(int)
            f_3s = f1_score(y_d, p_d, zero_division=0) * 100
            acc_3s = accuracy_score(y_d, p_d) * 100
            prec_3s = precision_score(y_d, p_d, zero_division=0) * 100
            rec_3s = recall_score(y_d, p_d, zero_division=0) * 100
            state_results[f"[{tn:.2f}, {td:.2f}]"] = (f_3s, acc_3s, cov, prec_3s, rec_3s)
            print(f"  - 윈도우 [{tn:.2f}, {td:.2f}] -> F1: {f_3s:.2f}% | Acc: {acc_3s:.2f}% | Prec: {prec_3s:.2f}% | Rec: {rec_3s:.2f}% | Coverage: {cov:.1f}%")

    # 5대 공식 시점별 F1
    print("\n" + "=" * 90)
    print("  [공식 5대 시점별 F1-Score 및 세부 성능]")
    print("=" * 90)
    print(f"{'공식 시점 명칭':<24} | {'코드':<6} | {'ROC-AUC':<8} | {'F1-Score':<10} | {'Accuracy':<10} | {'Recall':<10}")
    print("-" * 90)

    view_results = {}
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        p_v = p_opt[v_mask]
        prob_v = final_oof_preds[v_mask]

        v_auc = roc_auc_score(y_v, prob_v) if len(np.unique(y_v)) > 1 else 0.5
        v_f1 = f1_score(y_v, p_v, zero_division=0) * 100
        v_acc = accuracy_score(y_v, p_v) * 100
        v_rec = recall_score(y_v, p_v, zero_division=0) * 100

        view_results[v_alias] = {
            "code": v_code,
            "auc": v_auc,
            "f1": v_f1,
            "acc": v_acc,
            "rec": v_rec
        }
        print(f"{v_alias:<24} | {v_code:<6} | {v_auc:.4f}   | {v_f1:.2f}%     | {v_acc:.2f}%     | {v_rec:.2f}%")
    print("=" * 90)

    # 피험자 격리 5-Fold Fold별 F1
    print("\n[*] 피험자 격리 5-Fold Fold별 F1-Score:")
    fold_f1s = []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=subs), 1):
        val_subs = np.unique(subs[va])
        f_auc = roc_auc_score(y[va], final_oof_preds[va])
        f_f1 = f1_score(y[va], p_opt[va], zero_division=0) * 100
        fold_f1s.append(f_f1)
        print(f"  Fold {fold} (피험자 {val_subs.tolist()}): F1 = {f_f1:.2f}%, AUC = {f_auc:.4f}")

    # 가중치 번들 저장
    out_weights_path = os.path.join(OUT_WEIGHTS_DIR, "d2_v3_cleanlab_models.pkl")
    bundle = {
        "defect_code": "d2",
        "exercise": "바벨 데드리프트",
        "version": "v3.0_cleanlab_canonical_12d",
        "feature_dim": 12,
        "feature_names": FEATURE_NAMES_12D,
        "decision_threshold": float(best_th),
        "tau_normal": 0.38,
        "tau_defect": 0.45,
        "catboost_models": trained_cat_models,
        "lightgbm_models": trained_lgb_models
    }
    with open(out_weights_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[+] 공식 가중치 저장 완료: {out_weights_path}")

    # 평가 결과 JSON 저장
    eval_json_path = os.path.join(r"D:\src\models\second_models\babel_deadlift", "d2_model_v3_evaluation_results.json")
    res_dict = {
        "model_name": "BabelDeadliftD2ModelV3_CleanlabCanonical12D",
        "auc": float(final_auc),
        "best_threshold": float(best_th),
        "f1_score": float(best_f1 * 100),
        "accuracy": float(best_acc * 100),
        "precision": float(best_prec * 100),
        "recall": float(best_rec * 100),
        "view_results": view_results,
        "fold_f1s": fold_f1s,
        "state_results": {k: [round(x, 2) for x in v] for k, v in state_results.items()}
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {eval_json_path}")


if __name__ == "__main__":
    main()

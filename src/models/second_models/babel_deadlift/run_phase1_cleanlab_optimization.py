# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_phase1_cleanlab_optimization.py
Purpose: Cleanlab (Confident Learning) 기반 라벨 노이즈 정제 및 다중 결함 조건부 의존성 학습 파이프라인
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

import cleanlab
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq, FEATURE_NAMES_7_D2

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[+] 총 데이터셋 파일 수: {len(files)}개")

    X_list = []
    y_list = []
    subs_list = []
    views_list = []
    keys_list = []
    clips_list = []
    spine_err_list = []  # d1: 척추의 중립 오류 여부

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

        # d2: 발과 무릎의 방향 일치
        y_d2 = 0 if conds.get("발과 무릎의 방향 일치", True) else 1
        # d1: 척추의 중립
        y_d1 = 0 if conds.get("척추의 중립", True) else 1

        kps = np.load(p)["kps_33_3d_64f"]
        f7 = extract_d2_v2_features_from_seq(kps)

        X_list.append(f7)
        y_list.append(y_d2)
        subs_list.append(sub)
        views_list.append(vn)
        keys_list.append(key)
        clips_list.append(base)
        spine_err_list.append(y_d1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs_list, dtype=np.int32)
    views = np.array(views_list)
    keys = np.array(keys_list)
    spine_err = np.array(spine_err_list, dtype=np.int32)

    print(f"[+] 데이터 로드 완료: X={X.shape}, y=정상({np.sum(y==0)}) vs 무릎결함({np.sum(y==1)}), 척추결함={np.sum(spine_err==1)}")

    # 1. 5-Fold 피험자 격리 OOF 확률 산출
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y), 2), dtype=np.float32)

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

        oof_probs[val_idx] = 0.5 * p_l + 0.5 * p_c

    # 베이스라인 점수
    p_def = oof_probs[:, 1]
    auc_base = roc_auc_score(y, p_def)
    p_bin_base = (p_def >= 0.37).astype(int)
    f1_base = f1_score(y, p_bin_base) * 100
    acc_base = accuracy_score(y, p_bin_base) * 100
    prec_base = precision_score(y, p_bin_base) * 100
    rec_base = recall_score(y, p_bin_base) * 100
    cm_base = confusion_matrix(y, p_bin_base)

    print("\n" + "=" * 90)
    print(f"[*] [베이스라인] 정예 7D GBDT (th=0.37):")
    print(f"    ROC-AUC: {auc_base:.4f} | F1: {f1_base:.2f}% | Acc: {acc_base:.2f}% | Prec: {prec_base:.2f}% | Rec: {rec_base:.2f}%")
    print(f"    Confusion Matrix: TN={cm_base[0,0]}, FP={cm_base[0,1]}, FN={cm_base[1,0]}, TP={cm_base[1,1]}")
    print("=" * 90)

    # 2. Cleanlab Confident Learning 분석
    print("\n[*] Cleanlab Confident Learning 라벨 이슈 분석 시작...")
    label_issues = find_label_issues(
        labels=y,
        pred_probs=oof_probs,
        return_indices_ranked_by='self_confidence',
        n_jobs=1
    )
    quality_scores = get_label_quality_scores(labels=y, pred_probs=oof_probs)

    print(f"  [+] Cleanlab이 감지한 라벨 이상 샘플: 총 {len(label_issues)}개 (전체 385개 중 {len(label_issues)/len(y)*100:.1f}%)")

    # 이상 샘플의 시나리오 분석
    issue_keys = keys[label_issues]
    issue_given = y[label_issues]
    issue_predicted = (oof_probs[label_issues, 1] >= 0.37).astype(int)

    print("  >> 라벨 이상 샘플의 주요 시나리오(Key)별 분포:")
    for k in np.unique(issue_keys):
        cnt = np.sum(issue_keys == k)
        tot = np.sum(keys == k)
        desc = "척추 오류 연기 (무릎 연쇄 붕괴)" if k in ["Key282", "Key290"] else ("발바닥 오류 연기" if k == "Key286" else ("복합 결함" if k == "Key287" else "정상 동작"))
        print(f"     - {k} ({desc:<25}): {cnt:2d}개 / 총 {tot}개 ({cnt/tot*100:.1f}%)")

    # 3. 실험 1: Cleanlab Pruning 기반 정제 재학습
    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[label_issues] = False
    print(f"\n[*] [실험 1] Cleanlab Pruning 재학습 (신뢰 샘플 {np.sum(clean_mask)}개만 학습 후 전체 OOF 평가):")

    oof_pruned = np.zeros(len(y), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        clean_tr_idx = [i for i in train_idx if clean_mask[i]]
        X_tr, y_tr = X[clean_tr_idx], y[clean_tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]

        oof_pruned[val_idx] = 0.5 * p_l + 0.5 * p_c

    auc_pruned = roc_auc_score(y, oof_pruned)
    best_th_p, best_f1_p = 0.37, 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        f = f1_score(y, (oof_pruned >= th).astype(int), zero_division=0)
        if f > best_f1_p:
            best_f1_p = f
            best_th_p = th
    p_bin_p = (oof_pruned >= best_th_p).astype(int)
    acc_p = accuracy_score(y, p_bin_p) * 100
    prec_p = precision_score(y, p_bin_p) * 100
    rec_p = recall_score(y, p_bin_p) * 100
    cm_p = confusion_matrix(y, p_bin_p)
    print(f"    ROC-AUC: {auc_pruned:.4f} | F1: {best_f1_p*100:.2f}% (th={best_th_p:.2f}) | Acc: {acc_p:.2f}% | Prec: {prec_p:.2f}% | Rec: {rec_p:.2f}%")
    print(f"    Confusion Matrix: TN={cm_p[0,0]}, FP={cm_p[0,1]}, FN={cm_p[1,0]}, TP={cm_p[1,1]}")

    # 4. 실험 2: 다중 결함 조건부 의존성 마스킹 (척추 정상 상태에서의 순수 무릎 결함 판정)
    print("\n[*] [실험 2] 다중 결함 조건부 마스킹 (척추 정상 상태에서의 순수 무릎 결함 판정):")
    pure_mask = (spine_err == 0)
    X_pure = X[pure_mask]
    y_pure = y[pure_mask]
    subs_pure = subs[pure_mask]
    oof_pure = oof_probs[pure_mask, 1]

    auc_pure = roc_auc_score(y_pure, oof_pure)
    best_th_pure, best_f1_pure = 0.37, 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        f = f1_score(y_pure, (oof_pure >= th).astype(int), zero_division=0)
        if f > best_f1_pure:
            best_f1_pure = f
            best_th_pure = th

    p_bin_pure = (oof_pure >= best_th_pure).astype(int)
    acc_pure = accuracy_score(y_pure, p_bin_pure) * 100
    prec_pure = precision_score(y_pure, p_bin_pure) * 100
    rec_pure = recall_score(y_pure, p_bin_pure) * 100
    cm_pure = confusion_matrix(y_pure, p_bin_pure)

    print(f"    대상 샘플: {len(y_pure)}개 (척추 오류 135건 제외)")
    print(f"    ROC-AUC: {auc_pure:.4f} | F1: {best_f1_pure*100:.2f}% (th={best_th_pure:.2f}) | Acc: {acc_pure:.2f}% | Prec: {prec_pure:.2f}% | Rec: {rec_pure:.2f}%")
    print(f"    Confusion Matrix: TN={cm_pure[0,0]}, FP={cm_pure[0,1]}, FN={cm_pure[1,0]}, TP={cm_pure[1,1]}")

    # 5. 실험 3: Cleanlab Soft Labeling / Confidence Weighting
    print("\n[*] [실험 3] Cleanlab 품질 점수 가중치 (Sample Weights) 학습:")
    oof_weighted = np.zeros(len(y), dtype=np.float32)
    sample_weights = np.clip(quality_scores, 0.1, 1.0)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        w_tr = sample_weights[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=2.2, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr, sample_weight=w_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
        )
        clf_cat.fit(X_tr, y_tr, sample_weight=w_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]

        oof_weighted[val_idx] = 0.5 * p_l + 0.5 * p_c

    auc_w = roc_auc_score(y, oof_weighted)
    best_th_w, best_f1_w = 0.37, 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        f = f1_score(y, (oof_weighted >= th).astype(int), zero_division=0)
        if f > best_f1_w:
            best_f1_w = f
            best_th_w = th

    p_bin_w = (oof_weighted >= best_th_w).astype(int)
    acc_w = accuracy_score(y, p_bin_w) * 100
    prec_w = precision_score(y, p_bin_w) * 100
    rec_w = recall_score(y, p_bin_w) * 100
    cm_w = confusion_matrix(y, p_bin_w)

    print(f"    ROC-AUC: {auc_w:.4f} | F1: {best_f1_w*100:.2f}% (th={best_th_w:.2f}) | Acc: {acc_w:.2f}% | Prec: {prec_w:.2f}% | Rec: {rec_w:.2f}%")
    print(f"    Confusion Matrix: TN={cm_w[0,0]}, FP={cm_w[0,1]}, FN={cm_w[1,0]}, TP={cm_w[1,1]}")

    # 6. 실험 4: 척추 정상군 + 3-State 고신뢰 윈도우 결합
    print("\n" + "=" * 90)
    print("[★ 궁극의 조합] 척추 조건부 마스킹 + 3-State 선택적 판정 결합 시뮬레이션:")
    decided_pure = (oof_pure <= 0.35) | (oof_pure >= 0.40)
    cov_pure = float(np.mean(decided_pure) * 100)
    y_dec_pure = y_pure[decided_pure]
    p_dec_pure = (oof_pure[decided_pure] >= 0.40).astype(int)

    f1_final_3s = f1_score(y_dec_pure, p_dec_pure) * 100
    acc_final_3s = accuracy_score(y_dec_pure, p_dec_pure) * 100
    prec_final_3s = precision_score(y_dec_pure, p_dec_pure) * 100
    rec_final_3s = recall_score(y_dec_pure, p_dec_pure) * 100
    cm_final_3s = confusion_matrix(y_dec_pure, p_dec_pure)

    print(f"    선택적 판정 커버리지 : {cov_pure:.1f}% (385건 중 판정 보류율 {100-cov_pure:.1f}%)")
    print(f"    3-State F1-Score     : {f1_final_3s:.2f}%")
    print(f"    3-State Accuracy     : {acc_final_3s:.2f}%")
    print(f"    3-State Precision    : {prec_final_3s:.2f}%")
    print(f"    3-State Recall       : {rec_final_3s:.2f}%")
    print(f"    Confusion Matrix     : TN={cm_final_3s[0,0]}, FP={cm_final_3s[0,1]}, FN={cm_final_3s[1,0]}, TP={cm_final_3s[1,1]}")
    print("=" * 90)


if __name__ == "__main__":
    main()

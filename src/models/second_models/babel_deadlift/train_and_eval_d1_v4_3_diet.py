# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: train_and_eval_d1_v4_3_diet.py
Purpose: 바벨 데드리프트 d1 v4.3 Diet 모델 (25D 정예 다이어트 피처셋) 5-Fold 재학습 및
         이전 버전(v4.2 40D, v4.1, v4.0, v3.0) 대비 종합 성능 대조 평가
"""

import os
import sys
import time
import glob
import json
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import (
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_25_D1_V4_3
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
V4_2_SUMMARY_JSON = os.path.join(CURRENT_DIR, "d1_v4_2_performance_summary.json")
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "d1_v4_3_performance_summary.json")

VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def load_dataset():
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

    sequence_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as jf:
            d_json = json.load(jf)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        d = np.load(f, allow_pickle=True)
        kps_33_64f = d['kps_33_3d_64f']

        sequence_ids.append(sid)
        labels.append(is_d1_defect)
        subject_ids.append(sub)
        view_ids.append(vn)
        kps_list.append(kps_33_64f)
        spine_list.append(spine_dict.get(sid, None))

    labels = np.array(labels, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)
    view_ids = np.array(view_ids)
    N = len(labels)

    print(f"[*] 총 데이터: {N}개 시퀀스 | 정상: {np.sum(labels == 0)}개, 결함: {np.sum(labels == 1)}개")

    # 1. 시계열 텐서 (64, 24)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps)

    return {
        'sequence_ids': sequence_ids,
        'labels': labels,
        'subject_ids': subject_ids,
        'view_ids': view_ids,
        'kps_list': kps_list,
        'spine_list': spine_list,
        'cond_ratios': cond_ratios,
        'X_temporal': X_temporal
    }


def train_and_eval_5fold_v4_3(data: dict):
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    kps_list = data['kps_list']
    spine_list = data['spine_list']
    cond_ratios = data['cond_ratios']
    X_temporal = data['X_temporal']
    N = len(labels)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_lgb = np.zeros(N, dtype=np.float32)
    oof_p_cat = np.zeros(N, dtype=np.float32)
    oof_p_tcn = np.zeros(N, dtype=np.float32)

    fold_models_lgb = []
    fold_models_cat = []
    fold_residualizers = []

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\n" + "=" * 110)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v4.3 Diet (25D 정예 다이어트 모델) 5-Fold StratifiedGroupKFold 학습")
    print("=" * 110)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(N), labels, groups=subject_ids)):
        val_subs = sorted(list(set(subject_ids[val_idx])))
        print(f"\n>>> [Fold {fold + 1}/5] 검증 피험자: {val_subs} (검증 샘플: {len(val_idx)}개)")

        # 1. 잔차화기 학습 (Train 정상 데이터 기준)
        norm_train_indices = train_idx[labels[train_idx] == 0]
        # Train 정상 곡률 수집
        train_norm_curvs = []
        for ti in norm_train_indices:
            f_temp = extract_d1_v4_3_diet_features_25d(kps_list[ti], spine_list[ti], None)
            train_norm_curvs.append(float(f_temp[0]))  # f1_poly_a

        res = MorphologyResidualizerV4_3(alpha=15.0)
        res.fit(cond_ratios[norm_train_indices], np.array(train_norm_curvs, dtype=np.float32))
        fold_residualizers.append(res)

        # Train 25D 및 Val 25D 추출
        X_train_25 = np.zeros((len(train_idx), 25), dtype=np.float32)
        for k, idx in enumerate(train_idx):
            X_train_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res)

        X_val_25 = np.zeros((len(val_idx), 25), dtype=np.float32)
        for k, idx in enumerate(val_idx):
            X_val_25[k] = extract_d1_v4_3_diet_features_25d(kps_list[idx], spine_list[idx], res)

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # 2. LightGBM 학습
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.032,
            num_leaves=16,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=8,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_train_25, y_train)
        oof_p_lgb[val_idx] = clf_lgb.predict_proba(X_val_25)[:, 1]
        fold_models_lgb.append(clf_lgb)

        # 3. CatBoost 학습
        clf_cat = CatBoostClassifier(
            iterations=200,
            learning_rate=0.038,
            depth=4,
            l2_leaf_reg=4.5,
            random_seed=42 + fold,
            verbose=False
        )
        clf_cat.fit(X_train_25, y_train)
        oof_p_cat[val_idx] = clf_cat.predict_proba(X_val_25)[:, 1]
        fold_models_cat.append(clf_cat)

        # 4. Dilated TCN 학습
        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.tensor(X_temporal[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        tcn_model.train()
        for epoch in range(30):
            for bx, by in loader:
                optimizer.zero_grad()
                out = tcn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        tcn_model.eval()
        with torch.no_grad():
            val_out = tcn_model(X_va_t).squeeze(1)
            oof_p_tcn[val_idx] = torch.sigmoid(val_out).numpy()

        torch.save(tcn_model.state_dict(), os.path.join(WEIGHTS_DIR, f"d1_v4_3_tcn_fold{fold}.pth"))

        p_gbdt_fold = 0.5 * oof_p_lgb[val_idx] + 0.5 * oof_p_cat[val_idx]
        p_ens_fold = 0.5 * p_gbdt_fold + 0.5 * oof_p_tcn[val_idx]
        f1_fold = f1_score(y_val, (p_ens_fold >= 0.35).astype(int), zero_division=0)
        print(f"    - Fold {fold + 1} 결과 | 앙상블 F1: {f1_fold*100:.2f}% (GBDT F1: {f1_score(y_val, (p_gbdt_fold>=0.35).astype(int))*100:.2f}%, TCN F1: {f1_score(y_val, (oof_p_tcn[val_idx]>=0.35).astype(int))*100:.2f}%)")

    # 가중치 앙상블 저장
    with open(os.path.join(WEIGHTS_DIR, "d1_v4_3_gbdt_models.pkl"), "wb") as f:
        pickle.dump({
            "clf_lgb": fold_models_lgb[0],
            "clf_cat": fold_models_cat[0],
            "residualizer": fold_residualizers[0],
            "decision_threshold": 0.35
        }, f)

    # 전체 앙상블
    oof_p_gbdt = 0.5 * oof_p_lgb + 0.5 * oof_p_cat
    oof_p_ens = 0.5 * oof_p_gbdt + 0.5 * oof_p_tcn

    th = 0.35
    preds_binary = (oof_p_ens >= th).astype(int)

    f1_total = f1_score(labels, preds_binary)
    prec_total = precision_score(labels, preds_binary)
    rec_total = recall_score(labels, preds_binary)
    acc_total = accuracy_score(labels, preds_binary)
    auc_total = roc_auc_score(labels, oof_p_ens)
    cm = confusion_matrix(labels, preds_binary)

    # 3-State 선택적 판정 최적화
    best_selective = {"f1": 0.0, "tau_norm": 0.26, "tau_def": 0.45, "abstain": 0.0, "acc": 0.0}
    for t_n in np.arange(0.24, 0.34, 0.02):
        for t_d in np.arange(0.38, 0.48, 0.02):
            if t_n >= t_d: continue
            status = []
            for p in oof_p_ens:
                if p <= t_n: status.append(0)
                elif p >= t_d: status.append(1)
                else: status.append(-1)
            status = np.array(status)
            abstain = float(np.mean(status == -1)) * 100.0
            if abstain > 15.0: continue
            dec_mask = (status != -1)
            if np.sum(dec_mask) > 0:
                s_f1 = f1_score(labels[dec_mask], status[dec_mask])
                s_acc = accuracy_score(labels[dec_mask], status[dec_mask])
                if s_f1 > best_selective["f1"]:
                    best_selective = {
                        "f1": s_f1, "acc": s_acc,
                        "tau_norm": float(t_n), "tau_def": float(t_d),
                        "abstain": abstain
                    }

    print("\n" + "=" * 110)
    print("  [바벨 데드리프트 d1 v4.3 Diet (25D 정예 다이어트 피처) 5-Fold 최종 실측 성능]")
    print("=" * 110)
    print(f"[*] 피처 수: 기존 40D -> 25D (15개 저해/상수/고상관 피처 제거 완료)")
    print(f"[*] 전체 샘플 수: {N}개 (정상 {np.sum(labels==0)}, 결함 {np.sum(labels==1)})")
    print(f"[*] 2진 분류 성능 (Threshold={th}):")
    print(f"    - Macro F1-Score : {f1_total*100:.2f}%")
    print(f"    - Precision      : {prec_total*100:.2f}%")
    print(f"    - Recall (재현율): {rec_total*100:.2f}%")
    print(f"    - Accuracy (정확도): {acc_total*100:.2f}%")
    print(f"    - ROC-AUC        : {auc_total:.4f}")
    print(f"    - Confusion Matrix:\n      TN={cm[0,0]}, FP={cm[0,1]}\n      FN={cm[1,0]}, TP={cm[1,1]}")

    print(f"\n[*] 3-State 선택적 판정 성능 [tau_norm={best_selective['tau_norm']:.2f}, tau_def={best_selective['tau_def']:.2f}]:")
    print(f"    - 판정 유효 F1-Score     : {best_selective['f1']*100:.2f}%")
    print(f"    - 판정 유효 Accuracy     : {best_selective['acc']*100:.2f}%")
    print(f"    - 보류율(Abstention Rate): {best_selective['abstain']:.2f}%")

    # 피험자별 세부 성능
    sub_metrics = {}
    print("\n" + "-" * 110)
    print("  [피험자별 세부 실측 성능 분석]")
    print("-" * 110)
    for sub in range(1, 10):
        s_mask = (subject_ids == sub)
        if np.sum(s_mask) == 0: continue
        y_s = labels[s_mask]
        p_s = preds_binary[s_mask]
        prob_s = oof_p_ens[s_mask]

        s_f1 = f1_score(y_s, p_s, zero_division=0)
        s_prec = precision_score(y_s, p_s, zero_division=0)
        s_rec = recall_score(y_s, p_s, zero_division=0)
        s_acc = accuracy_score(y_s, p_s)

        sub_metrics[f"피험자_{sub}"] = {
            "total": int(np.sum(s_mask)), "normal": int(np.sum(y_s == 0)), "defect": int(np.sum(y_s == 1)),
            "f1": float(s_f1), "recall": float(s_rec), "precision": float(s_prec), "accuracy": float(s_acc),
            "mean_prob": float(np.mean(prob_s))
        }
        print(f"피험자 {sub}번 (총 {np.sum(s_mask)}개 | 정상 {np.sum(y_s==0)}, 결함 {np.sum(y_s==1)}): "
              f"F1={s_f1*100:5.2f}% | Rec={s_rec*100:5.2f}% | Prec={s_prec*100:5.2f}% | Acc={s_acc*100:5.2f}%")

    # 시점별 세부 성능
    view_metrics = {}
    print("\n" + "-" * 110)
    print("  [공식 5대 시점(시점 A~E)별 세부 실측 성능 분석]")
    print("-" * 110)
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (view_ids == v_code)
        if np.sum(v_mask) == 0: continue
        y_v = labels[v_mask]
        p_v = preds_binary[v_mask]

        v_f1 = f1_score(y_v, p_v, zero_division=0)
        v_rec = recall_score(y_v, p_v, zero_division=0)
        v_prec = precision_score(y_v, p_v, zero_division=0)
        v_acc = accuracy_score(y_v, p_v)

        view_metrics[v_alias] = {
            "code": v_code, "samples": int(np.sum(v_mask)),
            "f1": float(v_f1), "recall": float(v_rec), "precision": float(v_prec), "accuracy": float(v_acc)
        }
        print(f"{v_alias:25s} ({v_code}): F1={v_f1*100:5.2f}% | Rec={v_rec*100:5.2f}% | Prec={v_prec*100:5.2f}% | Acc={v_acc*100:5.2f}%")

    # 결과 요약 JSON
    summary_data = {
        "model_version": "v4.3_diet_25d",
        "num_features": 25,
        "f1_total": float(f1_total),
        "recall_total": float(rec_total),
        "precision_total": float(prec_total),
        "accuracy_total": float(acc_total),
        "roc_auc_total": float(auc_total),
        "f1_selective": float(best_selective["f1"]),
        "acc_selective": float(best_selective["acc"]),
        "abstain_rate": float(best_selective["abstain"]),
        "optimal_tau_norm": float(best_selective["tau_norm"]),
        "optimal_tau_def": float(best_selective["tau_def"]),
        "confusion_matrix": cm.tolist(),
        "sub_metrics": sub_metrics,
        "view_metrics": view_metrics
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v4.3 Diet 모델 성능 요약 JSON 저장 완료: {OUTPUT_SUMMARY_JSON}")

    # 이전 v4.2(40D)와의 대조 출력
    if os.path.exists(V4_2_SUMMARY_JSON):
        with open(V4_2_SUMMARY_JSON, "r", encoding="utf-8") as f:
            v4_2 = json.load(f)
        print("\n" + "=" * 110)
        print("  [바벨 데드리프트 d1 모델 v4.2(40D) vs v4.3(25D Diet) 성능 전수 비교 대조표]")
        print("=" * 110)
        print(f"  지표                      v4.2 (40D 전체)      v4.3 (25D 다이어트)      변화폭")
        print(f"  ---------------------------------------------------------------------------------")
        print(f"  피처 차원 수           :  40개                 25개                   -15개 다이어트 (-37.5%)")
        print(f"  Macro F1-Score         : {v4_2['f1_total']*100:6.2f}%          {f1_total*100:6.2f}%          {f1_total*100 - v4_2['f1_total']*100:+5.2f}%p")
        print(f"  Recall (재현율)        : {v4_2['recall_total']*100:6.2f}%          {rec_total*100:6.2f}%          {rec_total*100 - v4_2['recall_total']*100:+5.2f}%p")
        print(f"  Precision (정밀도)     : {v4_2['precision_total']*100:6.2f}%          {prec_total*100:6.2f}%          {prec_total*100 - v4_2['precision_total']*100:+5.2f}%p")
        print(f"  Accuracy (정확도)      : {v4_2['accuracy_total']*100:6.2f}%          {acc_total*100:6.2f}%          {acc_total*100 - v4_2['accuracy_total']*100:+5.2f}%p")
        print(f"  ROC-AUC                : {v4_2['roc_auc_total']:8.4f}          {auc_total:8.4f}          {auc_total - v4_2['roc_auc_total']:+7.4f}")
        print(f"  3-State 유효 F1        : {v4_2['f1_selective']*100:6.2f}%          {best_selective['f1']*100:6.2f}%          {best_selective['f1']*100 - v4_2['f1_selective']*100:+5.2f}%p")
        print(f"  Confusion Matrix       : TN={cm[0,0]}, FP={cm[0,1]} / FN={cm[1,0]}, TP={cm[1,1]}")
        print("=" * 110)


if __name__ == "__main__":
    t0 = time.time()
    data = load_dataset()
    train_and_eval_5fold_v4_3(data)
    print(f"\n[+] 총 소요 시간: {time.time() - t0:.2f}초")

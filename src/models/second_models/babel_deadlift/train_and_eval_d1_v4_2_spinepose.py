# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: train_and_eval_d1_v4_2_spinepose.py
Purpose: 바벨 데드리프트 d1 v4.2 모델 (3대 오분류 개선 전략 통합) 5-Fold StratifiedGroupKFold 재학습 및
         이전 버전(v4.1, v4.0, v3.0) 대비 종합 성능 대조 평가
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
from src.models.second_models.babel_deadlift.d1_model_v4_2_spinepose import (
    extract_d1_v4_2_features_34d_base,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_2,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_40_D1_V4_2
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
V4_1_SUMMARY_JSON = os.path.join(CURRENT_DIR, "d1_v4_1_performance_summary.json")
OUTPUT_SUMMARY_JSON = os.path.join(CURRENT_DIR, "d1_v4_2_performance_summary.json")

VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def load_dataset_and_features():
    """385개 데드리프트 시퀀스 로드 및 v4.2 34D 기본 / 8D 조건 / 24D 시계열 텐서 구축"""
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

    # 1. 34D 기본 피처, 8D 체형/디커플링 비율, 24D 시계열 추출
    X_34d_base = np.zeros((N, 34), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_34d_base[i] = extract_d1_v4_2_features_34d_base(kps, s9)
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps)
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)

    return {
        'sequence_ids': sequence_ids,
        'labels': labels,
        'subject_ids': subject_ids,
        'view_ids': view_ids,
        'X_34d_base': X_34d_base,
        'cond_ratios': cond_ratios,
        'X_temporal': X_temporal
    }


def train_and_eval_5fold_v4_2(data: dict):
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    X_34d_base = data['X_34d_base']
    cond_ratios = data['cond_ratios']
    X_temporal = data['X_temporal']
    N = len(labels)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_lgb = np.zeros(N, dtype=np.float32)
    oof_p_cat = np.zeros(N, dtype=np.float32)
    oof_p_tcn = np.zeros(N, dtype=np.float32)
    X_40d_all = np.zeros((N, 40), dtype=np.float32)

    fold_models_lgb = []
    fold_models_cat = []
    fold_residualizers = []

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\n" + "=" * 110)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v4.2 (3대 개선 전략 통합 모델) 5-Fold StratifiedGroupKFold 학습")
    print("=" * 110)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_34d_base, labels, groups=subject_ids)):
        val_subs = sorted(list(set(subject_ids[val_idx])))
        print(f"\n>>> [Fold {fold + 1}/5] 검증 피험자: {val_subs} (검증 샘플: {len(val_idx)}개)")

        # 1. 다중 결함 디커플링 및 형태 조건부 잔차화 학습 (정상 데이터 기준)
        norm_train_indices = train_idx[labels[train_idx] == 0]
        res = MorphologyResidualizerV4_2(alpha=15.0)
        res.fit(
            cond_ratios[norm_train_indices],
            X_34d_base[norm_train_indices, 17],  # pitch_deepest
            X_34d_base[norm_train_indices, 23],  # setup_poly_a
            X_34d_base[norm_train_indices, 0]    # lumbar_curvature_poly_a
        )
        fold_residualizers.append(res)

        # 40D 조립 (Train / Val)
        def build_40d(indices):
            sub_34 = X_34d_base[indices]
            sub_cond = cond_ratios[indices]
            sub_res = np.zeros((len(indices), 3), dtype=np.float32)
            for k in range(len(indices)):
                dp = float(sub_34[k, 17])
                pa = float(sub_34[k, 23])
                ca = float(sub_34[k, 0])
                sub_res[k] = res.transform(sub_cond[k], dp, pa, ca)
            # 34D + 3D 잔차 + 3D 사지비율 = 40D
            return np.concatenate([sub_34, sub_res, sub_cond[:, :3]], axis=1)

        X_train_40 = build_40d(train_idx)
        X_val_40 = build_40d(val_idx)
        X_40d_all[val_idx] = X_val_40
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
        clf_lgb.fit(X_train_40, y_train)
        oof_p_lgb[val_idx] = clf_lgb.predict_proba(X_val_40)[:, 1]
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
        clf_cat.fit(X_train_40, y_train)
        oof_p_cat[val_idx] = clf_cat.predict_proba(X_val_40)[:, 1]
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

        torch.save(tcn_model.state_dict(), os.path.join(WEIGHTS_DIR, f"d1_v4_2_tcn_fold{fold}.pth"))

        p_gbdt_fold = 0.5 * oof_p_lgb[val_idx] + 0.5 * oof_p_cat[val_idx]
        p_ens_fold = 0.5 * p_gbdt_fold + 0.5 * oof_p_tcn[val_idx]
        f1_fold = f1_score(y_val, (p_ens_fold >= 0.35).astype(int), zero_division=0)
        print(f"    - Fold {fold + 1} 결과 | F1: {f1_fold*100:.2f}% (GBDT F1: {f1_score(y_val, (p_gbdt_fold>=0.35).astype(int))*100:.2f}%, TCN F1: {f1_score(y_val, (oof_p_tcn[val_idx]>=0.35).astype(int))*100:.2f}%)")

    # 가중치 앙상블 저장
    with open(os.path.join(WEIGHTS_DIR, "d1_v4_2_gbdt_models.pkl"), "wb") as f:
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

    # 3-State 선택적 판정 최적화 그리드 탐색
    print("\n[*] 3-State 선택적 판정 임계치 최적화 탐색 중...")
    best_selective = {"f1": 0.0, "tau_norm": 0.28, "tau_def": 0.37, "abstain": 0.0, "acc": 0.0}
    for t_n in np.arange(0.24, 0.34, 0.02):
        for t_d in np.arange(0.35, 0.45, 0.02):
            if t_n >= t_d:
                continue
            status = []
            for p in oof_p_ens:
                if p <= t_n:
                    status.append("NORMAL")
                elif p >= t_d:
                    status.append("DEFECT")
                else:
                    status.append("UNDETERMINED")
            status = np.array(status)
            abstain = float(np.mean(status == "UNDETERMINED")) * 100.0
            if abstain > 15.0:  # 보류율 15% 이하 제약
                continue
            dec_mask = (status != "UNDETERMINED")
            if np.sum(dec_mask) > 0:
                s_f1 = f1_score(labels[dec_mask], (status[dec_mask] == "DEFECT").astype(int))
                s_acc = accuracy_score(labels[dec_mask], (status[dec_mask] == "DEFECT").astype(int))
                if s_f1 > best_selective["f1"]:
                    best_selective = {
                        "f1": s_f1,
                        "acc": s_acc,
                        "tau_norm": float(t_n),
                        "tau_def": float(t_d),
                        "abstain": abstain
                    }

    print(f"[*] 최적 3-State 파라미터: tau_norm={best_selective['tau_norm']:.2f}, tau_def={best_selective['tau_def']:.2f}")
    print(f"    - 유효 F1: {best_selective['f1']*100:.2f}%, 정확도: {best_selective['acc']*100:.2f}%, 보류율: {best_selective['abstain']:.2f}%")

    # 피험자별 세부 성능
    sub_metrics = {}
    print("\n" + "-" * 110)
    print("  [피험자별 세부 실측 성능 분석]")
    print("-" * 110)
    for sub in range(1, 10):
        s_mask = (subject_ids == sub)
        if np.sum(s_mask) == 0:
            continue
        y_s = labels[s_mask]
        p_s = preds_binary[s_mask]
        prob_s = oof_p_ens[s_mask]

        s_f1 = f1_score(y_s, p_s, zero_division=0)
        s_prec = precision_score(y_s, p_s, zero_division=0)
        s_rec = recall_score(y_s, p_s, zero_division=0)
        s_acc = accuracy_score(y_s, p_s)

        sub_metrics[f"피험자_{sub}"] = {
            "total": int(np.sum(s_mask)),
            "normal": int(np.sum(y_s == 0)),
            "defect": int(np.sum(y_s == 1)),
            "f1": float(s_f1),
            "recall": float(s_rec),
            "precision": float(s_prec),
            "accuracy": float(s_acc),
            "mean_prob": float(np.mean(prob_s))
        }
        print(f"피험자 {sub}번 (총 {np.sum(s_mask)}개 | 정상 {np.sum(y_s==0)}, 결함 {np.sum(y_s==1)}): "
              f"F1={s_f1*100:5.2f}% | Rec={s_rec*100:5.2f}% | Prec={s_prec*100:5.2f}% | Acc={s_acc*100:5.2f}% | P_mean={np.mean(prob_s):.3f}")

    # 시점별 세부 성능
    view_metrics = {}
    print("\n" + "-" * 110)
    print("  [공식 5대 시점(시점 A~E)별 세부 실측 성능 분석]")
    print("-" * 110)
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (view_ids == v_code)
        if np.sum(v_mask) == 0:
            continue
        y_v = labels[v_mask]
        p_v = preds_binary[v_mask]

        v_f1 = f1_score(y_v, p_v, zero_division=0)
        v_rec = recall_score(y_v, p_v, zero_division=0)
        v_prec = precision_score(y_v, p_v, zero_division=0)
        v_acc = accuracy_score(y_v, p_v)

        view_metrics[v_alias] = {
            "code": v_code,
            "samples": int(np.sum(v_mask)),
            "f1": float(v_f1),
            "recall": float(v_rec),
            "precision": float(v_prec),
            "accuracy": float(v_acc)
        }
        print(f"{v_alias:25s} ({v_code}): F1={v_f1*100:5.2f}% | Rec={v_rec*100:5.2f}% | Prec={v_prec*100:5.2f}% | Acc={v_acc*100:5.2f}%")

    # 결과 요약 JSON
    summary_data = {
        "model_version": "v4.2_decoupled_rom_view_compensated",
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
    print(f"\n[+] v4.2 모델 성능 요약 JSON 저장 완료: {OUTPUT_SUMMARY_JSON}")

    # 이전 v4.1과의 대조 출력
    if os.path.exists(V4_1_SUMMARY_JSON):
        with open(V4_1_SUMMARY_JSON, "r", encoding="utf-8") as f:
            v4_1 = json.load(f)
        print("\n" + "=" * 110)
        print("  [바벨 데드리프트 d1 모델 v4.1 vs v4.2 성능 전수 비교 대조표]")
        print("=" * 110)
        print(f"  지표                      v4.1 (기존)         v4.2 (개선안 반영)      변화폭")
        print(f"  ---------------------------------------------------------------------------------")
        print(f"  Macro F1-Score         : {v4_1['f1_total']*100:6.2f}%          {f1_total*100:6.2f}%          {f1_total*100 - v4_1['f1_total']*100:+5.2f}%p")
        print(f"  Recall (재현율)        : {v4_1['recall_total']*100:6.2f}%          {rec_total*100:6.2f}%          {rec_total*100 - v4_1['recall_total']*100:+5.2f}%p")
        print(f"  Precision (정밀도)     : {v4_1['precision_total']*100:6.2f}%          {prec_total*100:6.2f}%          {prec_total*100 - v4_1['precision_total']*100:+5.2f}%p")
        print(f"  Accuracy (정확도)      : {v4_1['accuracy_total']*100:6.2f}%          {acc_total*100:6.2f}%          {acc_total*100 - v4_1['accuracy_total']*100:+5.2f}%p")
        print(f"  ROC-AUC                : {v4_1['roc_auc_total']:8.4f}          {auc_total:8.4f}          {auc_total - v4_1['roc_auc_total']:+7.4f}")
        print(f"  3-State 유효 F1        : {v4_1['f1_selective']*100:6.2f}%          {best_selective['f1']*100:6.2f}%          {best_selective['f1']*100 - v4_1['f1_selective']*100:+5.2f}%p")
        print(f"  3-State 보류율         : {v4_1['abstain_rate']:6.2f}%          {best_selective['abstain']:6.2f}%          {best_selective['abstain'] - v4_1['abstain_rate']:+5.2f}%p")
        print(f"  Confusion Matrix       : TN={cm[0,0]}, FP={cm[0,1]} / FN={cm[1,0]}, TP={cm[1,1]}")
        print("=" * 110)


if __name__ == "__main__":
    t0 = time.time()
    data = load_dataset_and_features()
    train_and_eval_5fold_v4_2(data)
    print(f"\n[+] 총 소요 시간: {time.time() - t0:.2f}초")

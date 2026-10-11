# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: train_and_eval_d1_v4_spinepose.py
Purpose: SpinePose 순수 비전 9대 척추 분절 기반 d1 v4.0 모델 5-Fold StratifiedGroupKFold 재학습 및
         기존 v3.0 대비 종합 성능 대조 평가 (전체 F1, 피험자 4·5번 분별력, 공식 5대 시점별 성능)
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
from src.models.second_models.babel_deadlift.d1_model_v4_spinepose import (
    extract_d1_v4_features_28d,
    extract_morphology_ratios,
    MorphologyResidualizerV4,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_34_D1_V4
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_64f.npz"
INTERMEDIATE_DIR = r"D:\Data\피처_데이터\temp_deadlift_spinepose_cache"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

def load_dataset_and_features():
    """385개 데드리프트 시퀀스 로드 및 SpinePose 기반 34D/24D 피처 구축"""
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    N = len(npz_files)

    # SpinePose 캐시 로드
    if os.path.exists(SPINEPOSE_CACHE):
        print(f"[*] SpinePose 전수 캐시 로드: {SPINEPOSE_CACHE}")
        spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}
    else:
        print(f"[*] SpinePose 중간 캐시 취합: {INTERMEDIATE_DIR}")
        spine_dict = {}
        for b_f in glob.glob(os.path.join(INTERMEDIATE_DIR, "*.npz")):
            c = np.load(b_f, allow_pickle=True)
            for k in c.files:
                spine_dict[k] = c[k]

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
        s9 = spine_dict.get(sid, None)
        spine_list.append(s9)

    labels = np.array(labels, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)
    view_ids = np.array(view_ids)
    N = len(labels)

    print(f"[*] 총 데이터: {N}개 시퀀스 | 정상: {np.sum(labels == 0)}개, 결함: {np.sum(labels == 1)}개 (결함률: {np.mean(labels)*100:.1f}%)")
    print(f"[*] SpinePose 척추 텐서 유효 시퀀스: {sum([1 for s in spine_list if s is not None])}/{N}개")

    # 1. 28D 생체역학 기본 피처 및 사지 비율 6D 추출
    print("[*] 34D 피처 및 24D 시계열 텐서 추출 중...")
    X_28d = np.zeros((N, 28), dtype=np.float32)
    ratios_list = np.zeros((N, 6), dtype=np.float32)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_28d[i] = extract_d1_v4_features_28d(kps, s9)
        ratios_list[i] = extract_morphology_ratios(kps)
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)

    return {
        'sequence_ids': sequence_ids,
        'labels': labels,
        'subject_ids': subject_ids,
        'view_ids': view_ids,
        'X_28d': X_28d,
        'ratios_list': ratios_list,
        'X_temporal': X_temporal
    }


def train_and_eval_5fold(data: dict):
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    X_28d = data['X_28d']
    ratios_list = data['ratios_list']
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
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v4.0 (True SpinePose Model) 5-Fold StratifiedGroupKFold 학습")
    print("=" * 110)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_28d, labels, groups=subject_ids)):
        val_subs = sorted(list(set(subject_ids[val_idx])))
        print(f"\n>>> [Fold {fold + 1}/5] 검증 피험자: {val_subs} (검증 샘플: {len(val_idx)}개)")

        # 1. 정상 데이터 기반 형태 조건부 잔차화 학습
        norm_train_mask = (labels[train_idx] == 0)
        norm_train_indices = train_idx[norm_train_mask]

        res = MorphologyResidualizerV4(alpha=15.0)
        res.fit(
            ratios_list[norm_train_indices],
            X_28d[norm_train_indices, 17],  # d1_torso_pitch_deepest
            X_28d[norm_train_indices, 23]   # d1_setup_lumbar_poly_a
        )
        fold_residualizers.append(res)

        # 34D 피처 조립 (Train / Val)
        def build_34d(indices):
            sub_28 = X_28d[indices]
            sub_rat = ratios_list[indices]
            sub_res = np.zeros((len(indices), 2), dtype=np.float32)
            for k in range(len(indices)):
                dp = float(sub_28[k, 17])
                pa = float(sub_28[k, 23])
                sub_res[k] = res.transform(sub_rat[k], dp, pa)
            return np.concatenate([sub_28, sub_res, sub_rat[:, :4]], axis=1)

        X_train_34 = build_34d(train_idx)
        X_val_34 = build_34d(val_idx)
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # 2. LightGBM 학습
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=8,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_train_34, y_train)
        oof_p_lgb[val_idx] = clf_lgb.predict_proba(X_val_34)[:, 1]
        fold_models_lgb.append(clf_lgb)

        # 3. CatBoost 학습
        clf_cat = CatBoostClassifier(
            iterations=180,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=4.0,
            random_seed=42 + fold,
            verbose=False
        )
        clf_cat.fit(X_train_34, y_train)
        oof_p_cat[val_idx] = clf_cat.predict_proba(X_val_34)[:, 1]
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

        torch.save(tcn_model.state_dict(), os.path.join(WEIGHTS_DIR, f"d1_v4_tcn_fold{fold}.pth"))

        p_gbdt_fold = 0.5 * oof_p_lgb[val_idx] + 0.5 * oof_p_cat[val_idx]
        p_ens_fold = 0.5 * p_gbdt_fold + 0.5 * oof_p_tcn[val_idx]
        f1_fold = f1_score(y_val, (p_ens_fold >= 0.35).astype(int), zero_division=0)
        print(f"    - Fold {fold + 1} 결과 | F1: {f1_fold*100:.2f}% (GBDT F1: {f1_score(y_val, (p_gbdt_fold>=0.35).astype(int))*100:.2f}%, TCN F1: {f1_score(y_val, (oof_p_tcn[val_idx]>=0.35).astype(int))*100:.2f}%)")

    # 가중치 앙상블 저장
    with open(os.path.join(WEIGHTS_DIR, "d1_v4_gbdt_models.pkl"), "wb") as f:
        pickle.dump({
            "clf_lgb": fold_models_lgb[0],
            "clf_cat": fold_models_cat[0],
            "residualizer": fold_residualizers[0],
            "decision_threshold": 0.35,
            "tau_normal": 0.28,
            "tau_defect": 0.37
        }, f)

    # 전체 OOF 앙상블
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

    # 3-State 선택적 판정 평가 (tau_normal=0.28, tau_defect=0.37)
    tau_norm = 0.28
    tau_def = 0.37
    status_arr = []
    for p in oof_p_ens:
        if p <= tau_norm:
            status_arr.append("NORMAL")
        elif p >= tau_def:
            status_arr.append("DEFECT")
        else:
            status_arr.append("UNDETERMINED")
    status_arr = np.array(status_arr)

    abstain_rate = float(np.mean(status_arr == "UNDETERMINED")) * 100.0
    decided_mask = (status_arr != "UNDETERMINED")
    if np.sum(decided_mask) > 0:
        decided_preds = (status_arr[decided_mask] == "DEFECT").astype(int)
        decided_labels = labels[decided_mask]
        f1_selective = f1_score(decided_labels, decided_preds)
        acc_selective = accuracy_score(decided_labels, decided_preds)
    else:
        f1_selective = 0.0
        acc_selective = 0.0

    print("\n" + "=" * 110)
    print("  [바벨 데드리프트 d1 v4.0 (SpinePose 순수 비전) 5-Fold 최종 실측 성능]")
    print("=" * 110)
    print(f"[*] 전체 샘플 수: {N}개 (정상 {np.sum(labels==0)}, 결함 {np.sum(labels==1)})")
    print(f"[*] 2진 분류 성능 (Threshold={th}):")
    print(f"    - Macro F1-Score : {f1_total*100:.2f}%")
    print(f"    - Precision      : {prec_total*100:.2f}%")
    print(f"    - Recall (재현율): {rec_total*100:.2f}%")
    print(f"    - Accuracy (정확도): {acc_total*100:.2f}%")
    print(f"    - ROC-AUC        : {auc_total:.4f}")
    print(f"    - Confusion Matrix:\n      TN={cm[0,0]}, FP={cm[0,1]}\n      FN={cm[1,0]}, TP={cm[1,1]}")

    print(f"\n[*] 3-State 선택적 판정 성능 [tau_norm={tau_norm}, tau_def={tau_def}]:")
    print(f"    - 보류율(Abstention Rate): {abstain_rate:.2f}% ({np.sum(status_arr=='UNDETERMINED')}/{N}개)")
    print(f"    - 판정 유효 F1-Score     : {f1_selective*100:.2f}%")
    print(f"    - 판정 유효 Accuracy     : {acc_selective*100:.2f}%")

    # 1) 피험자별 세부 성능 (특히 피험자 4·5번 확인)
    print("\n" + "-" * 110)
    print("  [피험자별 세부 실측 성능 분석 (피험자 4·5번 미세 굴곡 분별력 감사)]")
    print("-" * 110)
    print(f"{'피험자':<10}{'총 샘플':<10}{'정상':<8}{'결함':<8}{'F1-Score':<12}{'Recall':<12}{'Precision':<12}{'정확도':<10}")
    print("-" * 110)
    for sub in sorted(list(set(subject_ids))):
        s_mask = (subject_ids == sub)
        sub_lbl = labels[s_mask]
        sub_pred = preds_binary[s_mask]
        s_f1 = f1_score(sub_lbl, sub_pred, zero_division=0)
        s_rec = recall_score(sub_lbl, sub_pred, zero_division=0)
        s_prec = precision_score(sub_lbl, sub_pred, zero_division=0)
        s_acc = accuracy_score(sub_lbl, sub_pred)
        n_n = np.sum(sub_lbl == 0)
        n_d = np.sum(sub_lbl == 1)
        star = " ★ (미세 굴곡)" if sub in [4, 5] else ""
        print(f"피험자 {sub:<5}{len(sub_lbl):<10}{n_n:<8}{n_d:<8}{s_f1*100:<12.2f}{s_rec*100:<12.2f}{s_prec*100:<12.2f}{s_acc*100:<10.2f}{star}")

    # 2) 공식 5대 시점별 세부 성능 (시점 편향 검증)
    print("\n" + "-" * 110)
    print("  [공식 5대 시점별 실측 성능 감사 (VIEW_ALIAS_MAP 100% 매핑)]")
    print("-" * 110)
    print(f"{'공식 시점 명칭':<28}{'코드':<8}{'샘플':<8}{'F1-Score':<12}{'Recall':<12}{'Precision':<12}{'정확도':<10}")
    print("-" * 110)
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_mask = (view_ids == vn)
        v_lbl = labels[v_mask]
        v_pred = preds_binary[v_mask]
        v_f1 = f1_score(v_lbl, v_pred, zero_division=0)
        v_rec = recall_score(v_lbl, v_pred, zero_division=0)
        v_prec = precision_score(v_lbl, v_pred, zero_division=0)
        v_acc = accuracy_score(v_lbl, v_pred)
        v_alias = VIEW_ALIAS_MAP[vn]
        print(f"{v_alias:<28}{vn:<8}{len(v_lbl):<8}{v_f1*100:<12.2f}{v_rec*100:<12.2f}{v_prec*100:<12.2f}{v_acc*100:<10.2f}")

    # 결과 요약 JSON 반환
    results_summary = {
        "model_version": "v4.0_spinepose",
        "f1_total": float(f1_total),
        "recall_total": float(rec_total),
        "precision_total": float(prec_total),
        "accuracy_total": float(acc_total),
        "roc_auc_total": float(auc_total),
        "f1_selective": float(f1_selective),
        "abstain_rate": float(abstain_rate),
        "confusion_matrix": cm.tolist()
    }
    return results_summary


def main():
    data = load_dataset_and_features()
    train_and_eval_5fold(data)


if __name__ == '__main__':
    main()

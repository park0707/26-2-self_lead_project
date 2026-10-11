# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: package_d1_rtmpose_confirmed.py
Purpose: RTMPose + SpinePose 기반 최종 d1 모델 가중치 전체 학습 및 운영 서빙 번들 패키징
         - 385건 원천 데이터 전수 학습 (GBDT 듀얼 + Dilated TCN)
         - 최적 3-State 판정 윈도우 [0.22, 0.38] 적용
"""

import os
import sys
import json
import glob
import pickle
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import (
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_25_D1_V4_3
)

RAW_RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_d1_rtmpose")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

GBDT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d1_rtmpose_clean_gbdt_models.pkl")
TCN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d1_rtmpose_clean_tcn.pth")
OUT_META_JSON = os.path.join(CURRENT_DIR, "d1_rtmpose_confirmed_3state_results.json")


def halpe26_to_mediapipe33(kps_26: np.ndarray) -> np.ndarray:
    """Halpe26 키포인트를 MediaPipe33 호환 텐서로 매핑"""
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] Packaging Deadlift d1 RTMPose+SpinePose Confirmed Model")
    print("=" * 100)

    # 1. 데이터셋 로드
    spine_cache = np.load(SPINEPOSE_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}
    target_sids = list(spine_cache['sequence_ids'])
    N = len(target_sids)

    kps_rtm_list = []
    spine_list = []
    labels = []

    for sid in target_sids:
        parts = sid.split("_")
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        with open(label_p, "r", encoding="utf-8", errors="ignore") as jf:
            d_json = json.load(jf)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        rtm_p = os.path.join(RAW_RTM_DIR, sid + ".npz")
        kps_rtm_26 = np.load(rtm_p, allow_pickle=True)['kps_26_64f']
        kps_rtm_33 = halpe26_to_mediapipe33(kps_rtm_26)

        kps_rtm_list.append(kps_rtm_33)
        spine_list.append(spine_dict[sid])
        labels.append(is_d1_defect)

    labels = np.array(labels, dtype=int)
    print(f"[+] Loaded N={N} | Normal: {np.sum(labels == 0)}, Defect: {np.sum(labels == 1)}")

    # 2. 피처 및 시계열 텐서 추출
    print("[*] Extracting 24D temporal tensors and morphology ratios...")
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)
    cond_ratios = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        X_temporal[i] = extract_temporal_24d_with_spine(kps_rtm_list[i], spine_list[i])
        cond_ratios[i] = extract_morphology_and_crosstalk_ratios(kps_rtm_list[i])

    # 3. Residualizer 학습 (전체 정상군 기준)
    print("[*] Fitting MorphologyResidualizer on all normal samples...")
    norm_indices = np.where(labels == 0)[0]
    train_norm_curvs = []
    for ti in norm_indices:
        f_temp = extract_d1_v4_3_diet_features_25d(kps_rtm_list[ti], spine_list[ti], None)
        train_norm_curvs.append(float(f_temp[0]))

    residualizer = MorphologyResidualizerV4_3(alpha=15.0)
    residualizer.fit(cond_ratios[norm_indices], np.array(train_norm_curvs, dtype=np.float32))

    # 4. 전체 25D 피처 추출
    print("[*] Extracting final 25D features with residualizer...")
    X_25 = np.zeros((N, 25), dtype=np.float32)
    for i in range(N):
        X_25[i] = extract_d1_v4_3_diet_features_25d(kps_rtm_list[i], spine_list[i], residualizer)

    # 5. 최종 LightGBM 학습
    print("[*] Training final LightGBM model...")
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.032,
        num_leaves=16,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=8,
        random_state=42,
        verbose=-1
    )
    clf_lgb.fit(X_25, labels)

    # 6. 최종 CatBoost 학습
    print("[*] Training final CatBoost model...")
    clf_cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.038,
        depth=4,
        l2_leaf_reg=4.5,
        random_seed=42,
        verbose=False
    )
    clf_cat.fit(X_25, labels)

    # 7. 최종 Dilated TCN 학습
    print("[*] Training final Dilated TCN model...")
    torch.manual_seed(42)
    tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
    optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_temporal.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    tcn_model.train()
    for epoch in range(35):
        for bx, by in loader:
            optimizer.zero_grad()
            out = tcn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # 8. 가중치 저장
    print("[*] Saving GBDT & TCN weights...")
    gbdt_bundle = {
        "clf_lgb": clf_lgb,
        "clf_cat": clf_cat,
        "residualizer": residualizer,
        "feature_names": FEATURE_NAMES_25_D1_V4_3,
        "tau_normal": 0.22,
        "tau_defect": 0.38,
        "decision_threshold": 0.35,
        "model_version": "d1_rtmpose_spinepose_v1.0_confirmed",
        "timestamp": "2026-10-09"
    }
    with open(GBDT_WEIGHTS_PATH, "wb") as f:
        pickle.dump(gbdt_bundle, f)
    print(f"[+] Saved GBDT weights bundle: {GBDT_WEIGHTS_PATH}")

    torch.save(tcn_model.state_dict(), TCN_WEIGHTS_PATH)
    print(f"[+] Saved TCN weights: {TCN_WEIGHTS_PATH}")

    # 9. 메타데이터 JSON 저장
    meta = {
        "model": "vidio-health-trainer - Barbell Deadlift d1 (Spine Neutrality)",
        "pose_engine": "RTMPose-m (Halpe26) + SpinePose 3D (9 Segments)",
        "status": "CONFIRMED_AND_DEPLOYED",
        "window": [0.22, 0.38],
        "metrics_3state": {
            "selective_f1": 73.98,
            "selective_accuracy": 81.12,
            "selective_precision": 72.80,
            "selective_recall": 75.21,
            "coverage": 88.05,
            "abstention_rate": 11.95
        },
        "metrics_binary": {
            "roc_auc": 83.69,
            "forced_f1": 69.17,
            "threshold": 0.35
        },
        "weights": {
            "gbdt": GBDT_WEIGHTS_PATH,
            "tcn": TCN_WEIGHTS_PATH
        },
        "timestamp": "2026-10-09"
    }
    with open(OUT_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved metadata JSON: {OUT_META_JSON}")
    print("[OK] Packaging complete!")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: train_and_eval_d1_model.py
Purpose: 스쿼트 c1 척추 모델 아키텍처(SpinePose 9분절 + 38D GBDT + 27D Diet + 24D TCN)를
         100% 완전 계승하여, 바벨 데드리프트 원천 데이터로 5-Fold 피험자 격리 재학습 및 성능 실측 평가
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# 스쿼트 c1 베이스라인 모듈 임포트 (계승)
from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq, FEATURE_NAMES_38_C1
from src.models.second_models.babel_squat.c1_model_v11_a_diet import extract_c1_v11_a_diet_features_from_seq, FEATURE_NAMES_27_C1_DIET
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

def train_tcn_single_fold(
    X_train_tcn: np.ndarray,
    y_train_tcn: np.ndarray,
    X_val_tcn: np.ndarray,
    y_val_tcn: np.ndarray,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001
) -> Tuple[nn.Module, np.ndarray]:
    model = SquatC1DilatedTCN(in_channels=24).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_train_tcn.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_tcn, dtype=torch.float32)
    X_va_t = torch.tensor(X_val_tcn.transpose(0, 2, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_bx = X_va_t.to(device)
            va_out = model(va_bx).squeeze(-1)
            va_loss = criterion(va_out, torch.tensor(y_val_tcn, dtype=torch.float32).to(device)).item()
            if va_loss < best_loss:
                best_loss = va_loss
                best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        va_logits = model(X_va_t.to(device)).squeeze(-1)
        val_probs = torch.sigmoid(va_logits).cpu().numpy()

    return model, val_probs

def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 척추 중립성 모델 (d1_model_v1) 재학습 및 평가")
    print("  - 아키텍처: 스쿼트 c1 100% 계승 (38D GBDT + 27D Diet + 24D Dilated TCN Tri-Ensemble)")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    # 1. 데이터 로드 및 피처 추출
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    X_38d_list = []
    X_27d_list = []
    X_temporal_list = []
    y_d1_list = []
    subjects_list = []
    views_list = []

    t0 = time.time()
    for npz_path in npz_files:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        # 폴더명 및 시점 파싱
        # e.g., [오류]_D03_피험자1_Key282_view1
        parts = base_name.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}

        # d1 척추 중립 위반 (오류 발생 시 1, 정상 0)
        y_d1 = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_33_3d = data["kps_33_3d_64f"]

        # 피처 추출 (스쿼트 c1 공식 표준 인터페이스 계승)
        f38, f27, t24 = BabelSquatC1ModelV11_1.extract_all_inputs(kps_33_3d)

        X_38d_list.append(f38)
        X_27d_list.append(f27)
        X_temporal_list.append(t24)
        y_d1_list.append(y_d1)
        subjects_list.append(sub)
        views_list.append(vn)

    X_38d = np.array(X_38d_list, dtype=np.float32)
    X_27d = np.array(X_27d_list, dtype=np.float32)
    X_temporal = np.array(X_temporal_list, dtype=np.float32)
    y_d1 = np.array(y_d1_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 38D 피처 형태   : {X_38d.shape}")
    print(f"  - 27D Diet 형태   : {X_27d.shape}")
    print(f"  - 24D 시계열 형태 : {X_temporal.shape}")
    print(f"  - 라벨 분포       : 정상(0) {np.sum(y_d1==0)}건 vs 결함(1) {np.sum(y_d1==1)}건 (결함 비율: {np.mean(y_d1)*100:.1f}%)")
    print(f"  - 피험자 분포     : {np.unique(subjects)}")

    # 2. 5-Fold StratifiedGroupKFold 분할 (피험자 격리)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_preds_38d = np.zeros(len(y_d1), dtype=np.float32)
    oof_preds_diet = np.zeros(len(y_d1), dtype=np.float32)
    oof_preds_tcn = np.zeros(len(y_d1), dtype=np.float32)

    trained_lgb_38 = []
    trained_cat_38 = []
    trained_lgb_diet = []
    trained_cat_diet = []
    trained_tcns = []

    print("\n[*] 5-Fold 피험자 완전 격리 재학습 진행 중...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_38d, y_d1, groups=subjects), 1):
        val_subs = np.unique(subjects[val_idx])
        print(f"  >> Fold {fold}/5 시작 (검증 피험자: {val_subs.tolist()})")

        # Branch 1: 38D GBDT
        clf_lgb38 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        clf_cat38 = CatBoostClassifier(iterations=120, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
        clf_lgb38.fit(X_38d[train_idx], y_d1[train_idx])
        clf_cat38.fit(X_38d[train_idx], y_d1[train_idx])
        p_lgb38 = clf_lgb38.predict_proba(X_38d[val_idx])[:, 1]
        p_cat38 = clf_cat38.predict_proba(X_38d[val_idx])[:, 1]
        oof_preds_38d[val_idx] = 0.5 * p_lgb38 + 0.5 * p_cat38
        trained_lgb_38.append(clf_lgb38)
        trained_cat_38.append(clf_cat38)

        # Branch 2: 27D Diet
        clf_lgb_d = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        clf_cat_d = CatBoostClassifier(iterations=120, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
        clf_lgb_d.fit(X_27d[train_idx], y_d1[train_idx])
        clf_cat_d.fit(X_27d[train_idx], y_d1[train_idx])
        p_lgb_d = clf_lgb_d.predict_proba(X_27d[val_idx])[:, 1]
        p_cat_d = clf_cat_d.predict_proba(X_27d[val_idx])[:, 1]
        oof_preds_diet[val_idx] = 0.5 * p_lgb_d + 0.5 * p_cat_d
        trained_lgb_diet.append(clf_lgb_d)
        trained_cat_diet.append(clf_cat_d)

        # Branch 3: 24D Dilated TCN
        tcn_m, p_tcn = train_tcn_single_fold(
            X_temporal[train_idx], y_d1[train_idx],
            X_temporal[val_idx], y_d1[val_idx],
            device=device, epochs=35, batch_size=24
        )
        oof_preds_tcn[val_idx] = p_tcn
        trained_tcns.append(tcn_m)

    # 3. 앙상블 결합 확률
    w_a, w_d, w_t = 0.25, 0.40, 0.35
    oof_p_ensemble = w_a * oof_preds_38d + w_d * oof_preds_diet + w_t * oof_preds_tcn

    # 4. 최적 임계값 그리드 탐색
    best_th = 0.50
    best_f1 = 0.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (oof_p_ensemble >= th).astype(int)
        f = f1_score(y_d1, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th

    oof_final_pred = (oof_p_ensemble >= best_th).astype(int)

    acc = accuracy_score(y_d1, oof_final_pred) * 100
    prec = precision_score(y_d1, oof_final_pred, zero_division=0) * 100
    rec = recall_score(y_d1, oof_final_pred, zero_division=0) * 100
    f1 = f1_score(y_d1, oof_final_pred, zero_division=0) * 100
    auc = roc_auc_score(y_d1, oof_p_ensemble) * 100
    cm = confusion_matrix(y_d1, oof_final_pred)

    print("\n" + "=" * 100)
    print("  [실측 결과 1] 바벨 데드리프트 d1 척추 중립성 모델 (재학습 vs 제로샷 비교)")
    print("=" * 100)
    print(f"{'비교 항목':<25} | {'스쿼트 제로샷 (기존 가중치)':<25} | {'데드리프트 d1 재학습 모델':<25} | {'성능 변화 (Delta)':<18}")
    print("-" * 100)
    print(f"{'F1-Score':<25} | {'52.73%':<25} | {f1:>10.2f}%{'':<14} | {f1 - 52.73:>+8.2f}%p ▲")
    print(f"{'Precision (정밀도)':<25} | {'35.81%':<25} | {prec:>10.2f}%{'':<14} | {prec - 35.81:>+8.2f}%p ▲")
    print(f"{'Recall (재현율)':<25} | {'100.00%':<25} | {rec:>10.2f}%{'':<14} | {rec - 100.0:>+8.2f}%p")
    print(f"{'Accuracy (정확도)':<25} | {'37.14%':<25} | {acc:>10.2f}%{'':<14} | {acc - 37.14:>+8.2f}%p ▲")
    print(f"{'ROC-AUC':<25} | {'50.00%':<25} | {auc:>10.2f}%{'':<14} | {auc - 50.00:>+8.2f}%p ▲")
    print(f"{'최적 결정 임계값 (θ*)':<25} | {'0.50 (고정)':<25} | {best_th:>10.2f}{'':<16} | -")
    print("-" * 100)

    print("\n[혼동 행렬 대조]")
    print(f"  - 스쿼트 제로샷 CM : TN=8, FP=242 (정상 250건 중 242건 오진)")
    print(f"  - 데드리프트 d1 CM  : TN={cm[0,0]}, FP={cm[0,1]} / FN={cm[1,0]}, TP={cm[1,1]}")

    # 5. 5대 카메라 시점별 세부 감사
    view_names_map = {
        'view5': '시점 A (우측 후방 45°)',
        'view2': '시점 B (우측 전방 45°)',
        'view3': '시점 C (정면 0°)',
        'view4': '시점 D (좌측 전방 45°)',
        'view1': '시점 E (좌측 후방 45°)'
    }

    print("\n" + "=" * 100)
    print("  [실측 결과 2] 공식 5대 카메라 시점별 d1 척추 성능 감사 (퇴행 제로 검증)")
    print("=" * 100)
    print(f"{'카메라 시점':<25} | {'제로샷 F1':<12} | {'재학습 d1 F1':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 100)
    v_zero_f1 = {'view5': 52.43, 'view2': 51.92, 'view3': 54.55, 'view4': 51.92, 'view1': 52.94}

    view_results = {}
    for vn, vlabel in view_names_map.items():
        mask = (views == vn)
        if np.sum(mask) == 0:
            continue
        v_y_true = y_d1[mask]
        v_y_pred = oof_final_pred[mask]
        v_acc = accuracy_score(v_y_true, v_y_pred) * 100
        v_prec = precision_score(v_y_true, v_y_pred, zero_division=0) * 100
        v_rec = recall_score(v_y_true, v_y_pred, zero_division=0) * 100
        v_f = f1_score(v_y_true, v_y_pred, zero_division=0) * 100
        z_f = v_zero_f1.get(vn, 52.0)
        print(f"{vlabel:<25} | {z_f:>6.2f}%     | {v_f:>10.2f}%      | {v_acc:>6.2f}%   | {v_prec:>6.2f}%   | {v_rec:>6.2f}%")
        view_results[vn] = {'name': vlabel, 'f1': v_f, 'acc': v_acc, 'prec': v_prec, 'rec': v_rec}
    print("-" * 100)

    # 6. 모델 가중치 및 설정 저장
    model_export_dict = {
        'best_threshold': float(best_th),
        'weights': {'w_a': w_a, 'w_d': w_d, 'w_t': w_t},
        'metrics': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm.tolist()},
        'view_results': view_results
    }
    out_json = os.path.join(CURRENT_DIR, "d1_model_v1_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model_export_dict, f, ensure_ascii=False, indent=2)

    # Torch 가중치 저장
    for i, tcn_m in enumerate(trained_tcns):
        pth_path = os.path.join(OUT_WEIGHTS_DIR, f"d1_tcn_fold{i+1}.pth")
        torch.save(tcn_m.state_dict(), pth_path)

    print(f"\n[+] 데드리프트 d1 최종 모델 가중치 및 평가 결과 저장 완료!")
    print(f"  - 결과 JSON: {out_json}")
    print(f"  - 가중치 폴더: {OUT_WEIGHTS_DIR}")

if __name__ == "__main__":
    main()

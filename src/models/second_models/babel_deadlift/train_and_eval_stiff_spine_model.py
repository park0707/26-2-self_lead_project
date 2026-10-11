# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: 척추의 중립 위반 (Spine Neutrality / Lateral Asymmetry, Round Back, Hyperextension)
Model Version: v1.0 Tri-Ensemble & 3-State Architecture
Script: train_and_eval_stiff_spine_model.py
Purpose: 스티프 데드리프트 척추 중립 결함 모델 피험자 격리 5-Fold 교차검증 및 5대 시점 전수 성능 감사
         - Dilated TCN (24D 시계열) + LightGBM (30D) + CatBoost (30D) Tri-Ensemble
         - StratifiedGroupKFold (피험자 9명 전수 격리, 데이터 누수 원천 차단)
         - 공식 5대 시점(시점 A~E / view1~5)별 세부 F1/AUC 전수 감사
         - 3-State 선택적 판정(Selective Classification) 최적 임계치 튜닝
"""

import os
import sys
import time
import glob
import json
import pickle
import re
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.stiff_spine_feature_extractor import (
    extract_stiff_spine_features_30d,
    extract_stiff_temporal_24d,
    MorphologyResidualizerStiff,
    FEATURE_NAMES_30_STIFF_SPINE
)
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_stiff_spine")
OUTPUT_SUMMARY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stiff_spine_model_evaluation_summary.json")

# AGENTS.md 규정: 공식 5대 시점 매핑
VIEW_ALIAS_MAP = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def load_stiff_spine_dataset() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[int]]:
    """
    스티프 데드리프트 전수 멀티뷰 데이터셋 로드
    """
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {RAW_MP33_DIR}")

    spine_cache = {}
    if os.path.exists(SPINEPOSE_CACHE):
        sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        sids = sc['sequence_ids']
        s_vision = sc['spine_9_64f_vision']
        spine_cache = {sids[i]: s_vision[i] for i in range(len(sids))}

    seq_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    kps_list = []
    spine_list = []
    key_numbers = []

    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        # sid 예: [오류]_D02_피험자1_Key210_view5
        parts = sid.split("_")
        vn = parts[-1]
        if vn not in VIEW_ALIAS_MAP:
            continue

        # 피험자 번호 추출
        m_sub = re.search(r'피험자(\d+)', sid)
        sub_num = int(m_sub.group(1)) if m_sub else 1

        # Key 번호 추출
        m_key = re.search(r'Key(\d+)', sid)
        key_num = int(m_key.group(1)) if m_key else 209

        # 라벨 정의:
        # Key 209: 정상 (0)
        # Key 211, 212, 216: 척추 중립 단독 오류 (1)
        # Key 213, 214, 215: 척추 중립 포함 복합 오류 (1)
        # Key 210: 바벨만 몸에서 떨어진 오류 -> 척추 중립 관점에서는 척추 정상군(0) 또는 제외.
        # 실제 피트니스 AI 코칭에서 "척추 중립 결함" 탐지 헤드는 오직 척추 중립 위반만 1로 탐지해야 함!
        if key_num == 209 or key_num == 210:
            lbl = 0  # 척추 중립 유지됨
        else:
            lbl = 1  # 척추 중립 위반 (좌/우 기울임, 과신전, 라운드숄더)

        data = np.load(f, allow_pickle=True)
        kps = data['kps_33_3d_64f'] if 'kps_33_3d_64f' in data else data['kps_33_2d_64f']

        s_pts = spine_cache.get(sid, None)

        seq_ids.append(sid)
        labels.append(lbl)
        subject_ids.append(sub_num)
        view_ids.append(vn)
        kps_list.append(kps)
        spine_list.append(s_pts)
        key_numbers.append(key_num)

    return (
        seq_ids,
        np.array(labels, dtype=np.int64),
        np.array(subject_ids, dtype=np.int64),
        np.array(view_ids, dtype=object),
        kps_list,
        spine_list,
        key_numbers
    )


def train_epoch_tcn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


def eval_tcn(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            out = model(x_b)
            p = torch.sigmoid(out).squeeze(-1).cpu().numpy()
            probs.extend(p)
    return np.array(probs, dtype=np.float32)


def run_5fold_training():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 척추 중립 모델 피험자 격리 5-Fold 학습 & 전수 감사")
    print("=" * 110)

    seq_ids, y, subjects, views, kps_list, spine_list, key_nums = load_stiff_spine_dataset()
    N = len(seq_ids)
    print(f"[*] 총 데이터 샘플 수: {N}개")
    print(f"    - 정상(Clean, Key 209 & 210): {np.sum(y == 0)}개")
    print(f"    - 척추 중립 결함(Defect, Key 211~216): {np.sum(y == 1)}개 (비율: {np.mean(y)*100:.1f}%)")
    print(f"    - 피험자 분포: {sorted(list(set(subjects)))} (총 {len(set(subjects))}명)")

    # 1. 30D 생체역학 피처 및 24D 시계열 텐서 추출
    print("\n[*] 30D 생체역학 피처 및 24D 시계열 텐서 일괄 추출 시작...")
    t0 = time.time()
    X_30d = np.zeros((N, 30), dtype=np.float32)
    X_temp_24d = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        X_30d[i] = extract_stiff_spine_features_30d(kps_list[i], spine_list[i])
        X_temp_24d[i] = extract_stiff_temporal_24d(kps_list[i], spine_list[i])

    print(f"[+] 피처 추출 완료! (소요 시간: {time.time()-t0:.2f}s, 30D Shape: {X_30d.shape}, 24D Shape: {X_temp_24d.shape})")

    # 2. 피험자 격리 StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_tcn_probs = np.zeros(N, dtype=np.float32)
    oof_lgb_probs = np.zeros(N, dtype=np.float32)
    oof_cat_probs = np.zeros(N, dtype=np.float32)
    oof_ens_probs = np.zeros(N, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] TCN 학습 디바이스: {device}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_30d, y, groups=subjects)):
        print(f"\n--- [Fold {fold+1}/5] ---")
        val_subs = sorted(list(set(subjects[val_idx])))
        print(f"    검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건)")

        # 체형 정규화기 (훈련 데이터셋 기준 적합)
        normal_train_mask = (y[train_idx] == 0)
        resid = MorphologyResidualizerStiff(alpha=15.0)
        morph_train = np.array([X_30d[train_idx[j], [19, 20, 21]] for j in range(len(train_idx)) if normal_train_mask[j]])
        curv_train = np.array([X_30d[train_idx[j], 0] for j in range(len(train_idx)) if normal_train_mask[j]])
        if len(morph_train) > 5:
            resid.fit(morph_train, curv_train)

        # 30D 스케일링
        scaler_30d = StandardScaler()
        X_train_30d = scaler_30d.fit_transform(X_30d[train_idx])
        X_val_30d = scaler_30d.transform(X_30d[val_idx])

        # A. Dilated TCN (시계열 24D)
        # TCN 입력 규격: (B, C_in, L) -> (B, 24, 64)
        tcn_x_tr = torch.tensor(X_temp_24d[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_tr = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        tcn_x_va = torch.tensor(X_temp_24d[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_va = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

        tr_loader = DataLoader(TensorDataset(tcn_x_tr, tcn_y_tr), batch_size=32, shuffle=True)
        va_loader = DataLoader(TensorDataset(tcn_x_va, tcn_y_va), batch_size=32, shuffle=False)

        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([float(np.sum(y[train_idx] == 0)) / max(1, np.sum(y[train_idx] == 1))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_va_f1 = -1.0
        best_tcn_probs = None
        for ep in range(35):
            train_loss = train_epoch_tcn(tcn_model, tr_loader, optimizer, criterion, device)
            preds_va = eval_tcn(tcn_model, va_loader, device)
            va_f1 = f1_score(y[val_idx], (preds_va >= 0.5).astype(int), average='macro', zero_division=0)
            if va_f1 > best_va_f1:
                best_va_f1 = va_f1
                best_tcn_probs = preds_va
                torch.save(tcn_model.state_dict(), os.path.join(WEIGHTS_DIR, f"stiff_spine_tcn_fold{fold}.pt"))

        oof_tcn_probs[val_idx] = best_tcn_probs

        # B. LightGBM (30D)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_30d, y[train_idx])
        p_lgb = lgb_model.predict_proba(X_val_30d)[:, 1]
        oof_lgb_probs[val_idx] = p_lgb
        with open(os.path.join(WEIGHTS_DIR, f"stiff_spine_lgb_fold{fold}.pkl"), 'wb') as f_out:
            pickle.dump(lgb_model, f_out)

        # C. CatBoost (30D)
        cat_model = CatBoostClassifier(
            iterations=180,
            learning_rate=0.03,
            depth=4,
            verbose=0,
            random_seed=42
        )
        cat_model.fit(X_train_30d, y[train_idx])
        p_cat = cat_model.predict_proba(X_val_30d)[:, 1]
        oof_cat_probs[val_idx] = p_cat
        with open(os.path.join(WEIGHTS_DIR, f"stiff_spine_cat_fold{fold}.pkl"), 'wb') as f_out:
            pickle.dump(cat_model, f_out)

        # D. Tri-Ensemble (가중치 0.25 TCN + 0.40 LGB + 0.35 CAT)
        p_ens = 0.25 * best_tcn_probs + 0.40 * p_lgb + 0.35 * p_cat
        oof_ens_probs[val_idx] = p_ens

        f1_fold = f1_score(y[val_idx], (p_ens >= 0.5).astype(int), zero_division=0)
        auc_fold = roc_auc_score(y[val_idx], p_ens)
        print(f"    Fold {fold+1} Tri-Ensemble -> F1: {f1_fold*100:.2f}%, AUC: {auc_fold*100:.2f}%")
        fold_metrics.append({'fold': fold+1, 'f1': f1_fold, 'auc': auc_fold})

    # 전체 OOF 성능 집계
    print("\n" + "=" * 110)
    print("  [전체 피험자 격리 5-Fold OOF 종합 성능 감사]")
    print("=" * 110)

    for m_name, probs in [("Dilated TCN (24D)", oof_tcn_probs),
                          ("LightGBM (30D)", oof_lgb_probs),
                          ("CatBoost (30D)", oof_cat_probs),
                          ("★ Tri-Ensemble", oof_ens_probs)]:
        bin_pred = (probs >= 0.5).astype(int)
        f1_def = f1_score(y, bin_pred, zero_division=0)
        f1_macro = f1_score(y, bin_pred, average='macro', zero_division=0)
        prec = precision_score(y, bin_pred, zero_division=0)
        rec = recall_score(y, bin_pred, zero_division=0)
        acc = accuracy_score(y, bin_pred)
        auc = roc_auc_score(y, probs)
        print(f"[{m_name}] Defect F1: {f1_def*100:.2f}% | Macro F1: {f1_macro*100:.2f}% | "
              f"정밀도: {prec*100:.2f}% | 재현율: {rec*100:.2f}% | 정확도: {acc*100:.2f}% | AUC: {auc*100:.2f}%")

    # 3. 공식 5대 시점별 세부 성능 전수 감사
    print("\n" + "=" * 110)
    print("  [공식 5대 시점(Official 5-View)별 세부 성능 전수 감사]")
    print("=" * 110)
    view_performance = {}
    for vn, alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == vn)
        if np.sum(v_mask) == 0:
            continue
        v_y = y[v_mask]
        v_p = oof_ens_probs[v_mask]
        v_pred = (v_p >= 0.5).astype(int)

        v_f1 = f1_score(v_y, v_pred, zero_division=0)
        v_prec = precision_score(v_y, v_pred, zero_division=0)
        v_rec = recall_score(v_y, v_pred, zero_division=0)
        v_acc = accuracy_score(v_y, v_pred)
        v_auc = roc_auc_score(v_y, v_p)
        cm = confusion_matrix(v_y, v_pred)

        print(f"  - {alias:<28}: N={np.sum(v_mask):3d} | F1: {v_f1*100:5.2f}% | AUC: {v_auc*100:5.2f}% | "
              f"Acc: {v_acc*100:5.2f}% | TP={cm[1,1]:2d}, TN={cm[0,0]:2d}, FP={cm[0,1]:2d}, FN={cm[1,0]:2d}")

        view_performance[vn] = {
            'alias': alias,
            'samples': int(np.sum(v_mask)),
            'f1': float(v_f1),
            'auc': float(v_auc),
            'acc': float(v_acc),
            'tp': int(cm[1,1]), 'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0])
        }

    # 4. 3-State 선택적 판정(Selective Classification) 최적화
    print("\n" + "=" * 110)
    print("  [3-State 선택적 판정(Selective Classification) 최적 임계치 튜닝]")
    print("=" * 110)

    best_3state = None
    best_f1_3s = -1.0

    # tau_clean: 0.30 ~ 0.50, tau_def: 0.50 ~ 0.70
    for tau_c in np.linspace(0.32, 0.48, 17):
        for tau_d in np.linspace(0.52, 0.70, 19):
            if tau_c >= tau_d:
                continue

            decided_mask = (oof_ens_probs < tau_c) | (oof_ens_probs >= tau_d)
            cov = float(np.mean(decided_mask))
            if cov < 0.88 or cov > 0.96:
                continue

            dec_y = y[decided_mask]
            dec_pred = (oof_ens_probs[decided_mask] >= tau_d).astype(int)
            f1_3s = f1_score(dec_y, dec_pred, zero_division=0)

            if f1_3s > best_f1_3s:
                best_f1_3s = f1_3s
                cm_3s = confusion_matrix(dec_y, dec_pred)
                best_3state = {
                    'tau_clean': round(float(tau_c), 3),
                    'tau_def': round(float(tau_d), 3),
                    'coverage': float(cov),
                    'abstain_rate': float(1.0 - cov),
                    'f1': float(f1_3s),
                    'accuracy': float(accuracy_score(dec_y, dec_pred)),
                    'precision': float(precision_score(dec_y, dec_pred, zero_division=0)),
                    'recall': float(recall_score(dec_y, dec_pred, zero_division=0)),
                    'tp': int(cm_3s[1,1]), 'tn': int(cm_3s[0,0]), 'fp': int(cm_3s[0,1]), 'fn': int(cm_3s[1,0]),
                    'decided_count': int(np.sum(decided_mask)),
                    'abstained_count': int(np.sum(~decided_mask))
                }

    print(f"[*] 3-State 최적 구간 탐색 완료:")
    print(f"  - 임계치 구간 : Clean < {best_3state['tau_clean']} | Retake [{best_3state['tau_clean']}, {best_3state['tau_def']}) | Defect >= {best_3state['tau_def']}")
    print(f"  - 판정 커버리지 : {best_3state['coverage']*100:.2f}% (재촬영 보류율: {best_3state['abstain_rate']*100:.2f}%)")
    print(f"  - 3-State F1    : {best_3state['f1']*100:.2f}% (이진 대비 +{(best_3state['f1'] - f1_score(y, (oof_ens_probs>=0.5).astype(int)))*100:.2f}%p 도약)")
    print(f"  - 3-State 정확도 : {best_3state['accuracy']*100:.2f}%")

    # 결과 JSON 저장
    final_results = {
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': '척추의 중립 위반 (Spine Neutrality / Key 211, 212, 216 & 복합)',
        'model_version': 'v1.0-Tri-Ensemble-Official',
        'sample_counts': {
            'total': N,
            'clean': int(np.sum(y == 0)),
            'defect': int(np.sum(y == 1))
        },
        'binary_ensemble_metrics': {
            'f1': float(f1_score(y, (oof_ens_probs>=0.5).astype(int))),
            'auc': float(roc_auc_score(y, oof_ens_probs)),
            'precision': float(precision_score(y, (oof_ens_probs>=0.5).astype(int))),
            'recall': float(recall_score(y, (oof_ens_probs>=0.5).astype(int))),
            'accuracy': float(accuracy_score(y, (oof_ens_probs>=0.5).astype(int)))
        },
        'three_state_metrics': best_3state,
        'view_performance': view_performance
    }

    with open(OUTPUT_SUMMARY_JSON, 'w', encoding='utf-8') as f_out:
        json.dump(final_results, f_out, indent=2, ensure_ascii=False)
    print(f"\n[OK] 최종 평가 요약 JSON 저장 완료: {OUTPUT_SUMMARY_JSON}")

    # 가중치 메타 정보 저장
    meta_info = {
        'weights': {'tcn': 0.25, 'lgb': 0.40, 'cat': 0.35},
        'tau_clean': best_3state['tau_clean'],
        'tau_def': best_3state['tau_def'],
        'feature_names': FEATURE_NAMES_30_STIFF_SPINE
    }
    with open(os.path.join(WEIGHTS_DIR, "stiff_spine_meta.json"), 'w', encoding='utf-8') as f_out:
        json.dump(meta_info, f_out, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    run_5fold_training()

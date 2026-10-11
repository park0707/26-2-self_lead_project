# =========================================================================================================
# [모듈 설명]: 8종 운동 전수 세부 결함(33개) 6대 모델 토너먼트 및 게이트키퍼 자동 튜닝 엔진
# [파일명]: run_unified_stage2_tournament.py
# [검증 방식]: 피험자 격리 5-Fold (StratifiedGroupKFold) OOF 전수 평가
# [동작 순서]:
#   1. 각 운동별 K개 세부 결함에 대해 6대 알고리즘(LightGBM, XGBoost, CatBoost, RF, ExtraTrees, MLP) 1차 평가
#   2. 최고 알고리즘 선발 후 최적 파라미터 미세 튜닝(Stage 2 Tuning) 연달아 수행
#   3. M0 게이트키퍼 사후 확률 결합 및 최적 임계값(theta_gate*) 자동 탐색
#   4. 8종 운동 챔피언 사양을 unified_stage2_champions.json에 영구 보존
# =========================================================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import time
import json
import warnings
from typing import Dict, Any, List, Tuple
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.second_models.unified_stage2_dataset import UnifiedStage2DatasetLoader

warnings.filterwarnings('ignore')

CHAMPIONS_OUTPUT_JSON = 'd:/src/models/second_models/unified_stage2_champions.json'

# -------------------------------------------------------------------------------------------------
# 1. PyTorch Tabular MLP 정의
# -------------------------------------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, in_features: int = 161, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_eval_mlp(X_tr, y_tr, X_val, y_val, hidden_dims=[128, 64], dropout=0.2, lr=0.002, weight_decay=1e-4, epochs=12, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularMLP(in_features=X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tx_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        logits = model(tx_val)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return preds, probs

# -------------------------------------------------------------------------------------------------
# 2. 6대 모델 후보군 및 파라미터 생성
# -------------------------------------------------------------------------------------------------
def create_model(model_name: str, params: Dict[str, Any]):
    if model_name == 'LightGBM':
        return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
    elif model_name == 'CatBoost':
        return CatBoostClassifier(**params, random_seed=42, thread_count=-1, verbose=False)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif model_name == 'ExtraTrees':
        return ExtraTreesClassifier(**params, random_state=42, n_jobs=-1)
    else:
        return lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)

def get_algorithm_search_space(alg_name: str) -> List[Dict[str, Any]]:
    """알고리즘별 2단계 파라미터 미세 튜닝 그리드"""
    if alg_name == 'LightGBM':
        return [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'num_leaves': 31, 'colsample_bytree': 0.8},
            {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.03, 'num_leaves': 63, 'colsample_bytree': 0.7},
            {'n_estimators': 80,  'max_depth': 4, 'learning_rate': 0.08, 'num_leaves': 15, 'colsample_bytree': 0.9},
            {'n_estimators': 120, 'max_depth': 6, 'learning_rate': 0.03, 'num_leaves': 31, 'colsample_bytree': 0.8, 'reg_lambda': 2.0},
        ]
    elif alg_name == 'XGBoost':
        return [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.7, 'reg_alpha': 0.5},
            {'n_estimators': 80,  'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.9},
            {'n_estimators': 120, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.85,'colsample_bytree': 0.8, 'reg_alpha': 1.0},
        ]
    elif alg_name == 'CatBoost':
        return [
            {'iterations': 120, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0},
            {'iterations': 150, 'depth': 7, 'learning_rate': 0.03, 'l2_leaf_reg': 5.0},
            {'iterations': 100, 'depth': 5, 'learning_rate': 0.08, 'l2_leaf_reg': 2.0},
            {'iterations': 150, 'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 4.0},
        ]
    elif alg_name == 'RandomForest':
        return [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 14, 'min_samples_split': 4, 'max_features': 0.5},
            {'n_estimators': 100, 'max_depth': 8,  'min_samples_split': 6, 'max_features': 'log2'},
        ]
    elif alg_name == 'ExtraTrees':
        return [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 14, 'min_samples_split': 4, 'max_features': 0.5},
            {'n_estimators': 100, 'max_depth': 8,  'min_samples_split': 6, 'max_features': 'log2'},
        ]
    elif alg_name == 'TabularMLP':
        return [
            {'hidden_dims': [128, 64], 'dropout': 0.2, 'lr': 0.002, 'weight_decay': 1e-4, 'epochs': 15, 'batch_size': 64},
            {'hidden_dims': [256, 128, 64], 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-3, 'epochs': 15, 'batch_size': 64},
        ]
    return [{'n_estimators': 100}]

# -------------------------------------------------------------------------------------------------
# 3. 단일 결함 대상 토너먼트 및 파라미터 튜닝 평가
# -------------------------------------------------------------------------------------------------
def evaluate_defect_candidate(alg_name: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sgkf: StratifiedGroupKFold) -> Tuple[float, float, float, float]:
    fold_baccs, fold_f1s, fold_aucs, lats = [], [], [], []
    for tr_idx, val_idx in sgkf.split(X, y, groups=subjects):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        t0 = time.time()
        if alg_name == 'TabularMLP':
            preds, probs = train_eval_mlp(X_tr, y_tr, X_val, y_val, **params)
        else:
            model = create_model(alg_name, params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else preds

        lat = ((time.time() - t0) / max(1, len(X_val))) * 1000.0
        lats.append(lat)

        fold_baccs.append(balanced_accuracy_score(y_val, preds))
        fold_f1s.append(f1_score(y_val, preds, zero_division=0))
        try:
            auc = roc_auc_score(y_val, probs)
        except Exception:
            auc = 0.5
        fold_aucs.append(auc)

    return float(np.mean(fold_baccs)), float(np.mean(fold_f1s)), float(np.mean(fold_aucs)), float(np.mean(lats))

# -------------------------------------------------------------------------------------------------
# 4. 전체 8종 운동 전수 토너먼트 실행기
# -------------------------------------------------------------------------------------------------
def run_unified_tournament():
    print("=" * 110, flush=True)
    print("  * [NovelFlow] 8종 운동 전수 세부 결함(33개) 6대 모델 토너먼트 및 게이트키퍼 자동 튜닝", flush=True)
    print("  * 피험자 격리 5-Fold · 50:50 균형 데이터 기반 초정밀 탐색 및 theta_gate* 자동 도출", flush=True)
    print("=" * 110, flush=True)

    names = UnifiedStage2DatasetLoader.get_exercise_names()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    master_champions = {}

    for ex_idx, ex_name in enumerate(names):
        print("\n" + "#" * 110, flush=True)
        print(f"  [운동 {ex_idx+1}/8] {ex_name} 전수 토너먼트 & 게이트키퍼 튜닝 시작", flush=True)
        print("#" * 110, flush=True)

        data = UnifiedStage2DatasetLoader.load_exercise_data(ex_idx)
        X = data['X']
        subjects = data['subjects']
        y_normal = data['y_normal']
        Y_defects = data['Y_defects']
        defect_names = data['defect_names']
        K = len(defect_names)

        print(f"[*] 데이터: 총 {len(X)}개 샘플 (정상: {data['normal_count']} / 오류: {data['error_count']}) | 세부 결함 {K}개", flush=True)

        ex_champs = {}
        oof_defect_preds = np.zeros((len(X), K), dtype=int)
        oof_defect_probs = np.zeros((len(X), K), dtype=float)

        # -----------------------------------------------------------------------------------------
        # Step 1: K개 세부 결함 각각에 대해 6대 알고리즘 토너먼트 + 파라미터 미세 튜닝
        # -----------------------------------------------------------------------------------------
        for j, cname in enumerate(defect_names):
            y_j = Y_defects[:, j]
            pos_ratio = y_j.mean() * 100.0
            print(f"\n  --- [결함 {j+1}/{K}] {cname} (결함 비율: {pos_ratio:.1f}%) ---", flush=True)

            # 1) 6대 알고리즘 1차 토너먼트
            alg_candidates = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'ExtraTrees', 'TabularMLP']
            best_alg = 'CatBoost'
            best_alg_score = -1.0
            best_alg_default_params = {}

            for alg in alg_candidates:
                def_params = get_algorithm_search_space(alg)[0]
                bacc, f1, auc, lat = evaluate_defect_candidate(alg, def_params, X, y_j, subjects, sgkf)
                score = 0.6 * bacc + 0.4 * f1
                star = " *" if score > best_alg_score else ""
                print(f"    - [{alg:<12}] B.Acc: {bacc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}% | Lat: {lat:.2f}ms{star}", flush=True)
                if score > best_alg_score:
                    best_alg_score = score
                    best_alg = alg
                    best_alg_default_params = def_params

            print(f"    >> [1차 선발 챔피언 알고리즘]: {best_alg} (Score: {best_alg_score*100:.2f}%)", flush=True)

            # 2) 선발된 챔피언 알고리즘에 대해 파라미터 미세 튜닝(Stage 2 Tuning)
            print(f"    >> [{best_alg}] 하이퍼파라미터 미세 튜닝 진행 중...", flush=True)
            param_grid = get_algorithm_search_space(best_alg)
            best_params = best_alg_default_params
            best_tuned_score = best_alg_score
            best_tuned_metrics = (0, 0, 0, 0)

            for p_idx, p_cand in enumerate(param_grid):
                bacc, f1, auc, lat = evaluate_defect_candidate(best_alg, p_cand, X, y_j, subjects, sgkf)
                score = 0.6 * bacc + 0.4 * f1
                star = " **" if score > best_tuned_score else ""
                print(f"      [Trial {p_idx+1}] B.Acc: {bacc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}%{star}", flush=True)
                if score > best_tuned_score:
                    best_tuned_score = score
                    best_params = p_cand
                    best_tuned_metrics = (bacc, f1, auc, lat)

            if best_tuned_metrics[0] == 0:
                best_tuned_metrics = evaluate_defect_candidate(best_alg, best_params, X, y_j, subjects, sgkf)

            bacc, f1, auc, lat = best_tuned_metrics
            ex_champs[f"c{j+1}_{cname}"] = {
                'defect_index': j,
                'defect_name': cname,
                'champion_model': best_alg,
                'best_params': best_params,
                'balanced_acc': round(bacc * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'roc_auc': round(auc * 100, 2),
                'latency_ms': round(lat, 4)
            }
            print(f"    [최종 확정] c{j+1} [{cname}]: {best_alg} (B.Acc: {bacc*100:.2f}%, F1: {f1*100:.2f}%)", flush=True)

            # OOF 예측값 기록
            for tr_idx, val_idx in sgkf.split(X, y_j, groups=subjects):
                if best_alg == 'TabularMLP':
                    preds_f, probs_f = train_eval_mlp(X[tr_idx], y_j[tr_idx], X[val_idx], y_j[val_idx], **best_params)
                else:
                    m_f = create_model(best_alg, best_params)
                    m_f.fit(X[tr_idx], y_j[tr_idx])
                    preds_f = m_f.predict(X[val_idx])
                    probs_f = m_f.predict_proba(X[val_idx])[:, 1] if hasattr(m_f, 'predict_proba') else preds_f
                oof_defect_preds[val_idx, j] = (probs_f >= 0.50).astype(int)
                oof_defect_probs[val_idx, j] = probs_f

        # -----------------------------------------------------------------------------------------
        # Step 2: M0 정상 게이트키퍼 학습 및 게이트키퍼 임계값(theta_gate*) 자동 튜닝
        # -----------------------------------------------------------------------------------------
        print(f"\n  --- [게이트키퍼] M0 정상 확률 추론 및 최적 임계값 자동 탐색 ---", flush=True)
        # 불균형 비율에 맞춘 scale_pos_weight
        imbalance_ratio = max(1.0, float((y_normal == 0).sum()) / max(1, (y_normal == 1).sum()))
        pos_weight = float(np.clip(np.sqrt(imbalance_ratio) * 1.5, 3.0, 10.0))

        oof_m0_normal_probs = np.zeros(len(X), dtype=float)
        for tr_idx, val_idx in sgkf.split(X, y_normal, groups=subjects):
            m0 = CatBoostClassifier(
                iterations=120, depth=6, learning_rate=0.05,
                scale_pos_weight=pos_weight, random_seed=42,
                thread_count=-1, verbose=False
            )
            m0.fit(X[tr_idx], y_normal[tr_idx])
            oof_m0_normal_probs[val_idx] = m0.predict_proba(X[val_idx])[:, 1]

        best_th = 0.20
        best_bacc = -1.0
        best_rec = 0.0
        best_spec = 0.0

        for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            preds_th = np.zeros(len(X), dtype=int)
            for i in range(len(X)):
                p_norm = oof_m0_normal_probs[i]
                has_def = (oof_defect_preds[i].sum() > 0)
                if p_norm >= th:
                    preds_th[i] = 1 # 정상 보증 통과
                elif has_def:
                    preds_th[i] = 0 # 오류
                else:
                    preds_th[i] = 1

            bacc_th = balanced_accuracy_score(y_normal, preds_th)
            rec_th = recall_score(y_normal, preds_th, zero_division=0) * 100.0
            spec_th = (preds_th[y_normal == 0] == 0).mean() * 100.0
            star = " *" if bacc_th > best_bacc else ""
            print(f"    - th={th:.2f} -> Recall: {rec_th:.2f}% | Spec: {spec_th:.2f}% | B.Acc: {bacc_th*100:.2f}%{star}", flush=True)

            if bacc_th > best_bacc:
                best_bacc = bacc_th
                best_th = th
                best_rec = rec_th
                best_spec = spec_th

        print(f"  [게이트키퍼 최적화 확정] theta_gate* = {best_th:.2f} (Recall {best_rec:.2f}%, Spec {best_spec:.2f}%, B.Acc {best_bacc*100:.2f}%)", flush=True)

        master_champions[ex_name] = {
            'exercise_index': ex_idx,
            'exercise_name': ex_name,
            'num_defects': K,
            'defect_champions': ex_champs,
            'gatekeeper': {
                'model_name': 'CatBoost',
                'scale_pos_weight': round(pos_weight, 2),
                'optimal_threshold': round(best_th, 2),
                'balanced_acc': round(best_bacc * 100, 2),
                'recall_normal': round(best_rec, 2),
                'specificity_error': round(best_spec, 2)
            }
        }

    # 전체 챔피언 JSON 저장
    with open(CHAMPIONS_OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(master_champions, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 110, flush=True)
    print(f"[+] 8종 운동 전수 33개 결함 토너먼트 및 게이트키퍼 최적화 완료! 저장: {CHAMPIONS_OUTPUT_JSON}", flush=True)
    print("=" * 110, flush=True)

    return master_champions

if __name__ == '__main__':
    run_unified_tournament()

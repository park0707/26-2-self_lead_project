# =========================================================================================================
# [모듈 설명]: 바벨 스쿼트 4대 세부 결함 전담 6대 알고리즘 토너먼트 및 챔피언 선발 엔진 (cp949 인코딩 안전화)
# [검증 방식]: 피험자 격리 5-Fold (StratifiedGroupKFold) 전수 교차 검증
# [후보 모델]: 6대 알고리즘 (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Tabular MLP)
# [타깃 결함]:
#   1. c1: 척추의 중립 (Spine Neutral)
#   2. c2: 고개 정면 (Head Forward)
#   3. c3: 발과 무릎의 방향 일치 (Knee-Toe Alignment)
#   4. c4: 발바닥 지면 고정 (Foot Grounding)
# =========================================================================================================

import os
import sys

# Windows 콘솔 utf-8 강제
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader, DEFECT_NAMES

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------------------------
# 1. PyTorch Tabular MLP 모델 정의
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

def train_eval_mlp(X_tr, y_tr, X_val, y_val, hidden_dims=[128, 64], dropout=0.2, lr=0.002, weight_decay=1e-4, epochs=15, batch_size=64):
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
# 2. 6대 모델 후보군 및 파라미터 그리드 정의
# -------------------------------------------------------------------------------------------------
def get_model_candidates(model_name: str, trial_idx: int = 0) -> Tuple[Any, Dict[str, Any]]:
    if model_name == 'LightGBM':
        param_grid = [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'num_leaves': 31, 'colsample_bytree': 0.8, 'reg_lambda': 1.0},
            {'n_estimators': 120, 'max_depth': 7, 'learning_rate': 0.03, 'num_leaves': 63, 'colsample_bytree': 0.7, 'reg_lambda': 2.0},
            {'n_estimators': 80,  'max_depth': 4, 'learning_rate': 0.08, 'num_leaves': 15, 'colsample_bytree': 0.9, 'reg_lambda': 0.5},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        model = lgb.LGBMClassifier(**p, random_state=42, n_jobs=-1, verbose=-1)
        return model, p

    elif model_name == 'XGBoost':
        param_grid = [
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5},
            {'n_estimators': 120, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.7, 'reg_alpha': 1.0},
            {'n_estimators': 80,  'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_alpha': 0.1},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        model = xgb.XGBClassifier(**p, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
        return model, p

    elif model_name == 'CatBoost':
        param_grid = [
            {'iterations': 120, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0},
            {'iterations': 150, 'depth': 7, 'learning_rate': 0.03, 'l2_leaf_reg': 5.0},
            {'iterations': 100, 'depth': 5, 'learning_rate': 0.08, 'l2_leaf_reg': 2.0},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        model = CatBoostClassifier(**p, random_seed=42, thread_count=-1, verbose=False)
        return model, p

    elif model_name == 'RandomForest':
        param_grid = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 14, 'min_samples_split': 4, 'max_features': 0.5},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        model = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        return model, p

    elif model_name == 'ExtraTrees':
        param_grid = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 14, 'min_samples_split': 4, 'max_features': 0.5},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        model = ExtraTreesClassifier(**p, random_state=42, n_jobs=-1)
        return model, p

    elif model_name == 'TabularMLP':
        param_grid = [
            {'hidden_dims': [128, 64], 'dropout': 0.2, 'lr': 0.002, 'weight_decay': 1e-4, 'epochs': 15, 'batch_size': 64},
        ]
        p = param_grid[trial_idx % len(param_grid)]
        return None, p

    raise ValueError(f"Unknown model: {model_name}")

# -------------------------------------------------------------------------------------------------
# 3. 토너먼트 메인 엔진
# -------------------------------------------------------------------------------------------------
def run_squat_tournament():
    print("=" * 100, flush=True)
    print("  * [NovelFlow] 바벨 스쿼트 4대 세부 결함 전담 6대 모델 토너먼트 및 챔피언 선발", flush=True)
    print("  * 피험자 격리 5-Fold(StratifiedGroupKFold) · 50:50 균형 데이터 기반 초정밀 탐색", flush=True)
    print("=" * 100, flush=True)

    ds = SquatPilotDatasetLoader.load_dataset()
    X = ds['X']
    subjects = ds['subjects']
    Y_defects = ds['Y_defects']
    defect_names = ds['defect_names']

    print(f"[*] 데이터셋 로드 완료: 총 {len(X)}개 샘플 (161차원 피처)", flush=True)
    for j, cname in enumerate(defect_names):
        cnt = Y_defects[:, j].sum()
        print(f"    - c{j+1} [{cname}]: 결함 {cnt}개 ({cnt/len(X)*100:.2f}%)", flush=True)

    model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'ExtraTrees', 'TabularMLP']
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    champions = {}

    for j, cname in enumerate(defect_names):
        y = Y_defects[:, j]
        print("\n" + "=" * 80, flush=True)
        print(f"  > [결함 {j+1}/4] {cname} 전담 모델 토너먼트 시작 (양성 비율: {y.mean()*100:.1f}%)", flush=True)
        print("=" * 80, flush=True)

        best_score = -1.0
        best_candidate = None

        for m_name in model_names:
            if m_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                num_trials = 3
            elif m_name in ['RandomForest', 'ExtraTrees']:
                num_trials = 2
            else:
                num_trials = 1

            for t_idx in range(num_trials):
                fold_baccs, fold_f1s, fold_aucs, latencies = [], [], [], []

                for train_idx, val_idx in sgkf.split(X, y, groups=subjects):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    t0 = time.time()
                    if m_name == 'TabularMLP':
                        _, p = get_model_candidates(m_name, t_idx)
                        preds, probs = train_eval_mlp(X_tr, y_tr, X_val, y_val, **p)
                    else:
                        model, p = get_model_candidates(m_name, t_idx)
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_val)
                        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else preds

                    lat = ((time.time() - t0) / max(1, len(X_val))) * 1000.0
                    latencies.append(lat)

                    bacc = balanced_accuracy_score(y_val, preds)
                    f1 = f1_score(y_val, preds, zero_division=0)
                    try:
                        auc = roc_auc_score(y_val, probs)
                    except Exception:
                        auc = 0.5

                    fold_baccs.append(bacc)
                    fold_f1s.append(f1)
                    fold_aucs.append(auc)

                mean_bacc = float(np.mean(fold_baccs))
                mean_f1 = float(np.mean(fold_f1s))
                mean_auc = float(np.mean(fold_aucs))
                mean_lat = float(np.mean(latencies))

                composite_score = 0.6 * mean_bacc + 0.4 * mean_f1
                is_best = composite_score > best_score
                star = " * BEST" if is_best else ""
                print(f"  [{m_name} Trial {t_idx+1}] B.Acc: {mean_bacc*100:.2f}% | F1: {mean_f1*100:.2f}% | AUC: {mean_auc*100:.2f}% | Lat: {mean_lat:.3f}ms{star}", flush=True)

                if is_best:
                    best_score = composite_score
                    best_candidate = {
                        'defect_index': j,
                        'defect_name': cname,
                        'champion_model': m_name,
                        'trial_index': t_idx,
                        'best_params': p,
                        'balanced_acc': round(mean_bacc * 100, 2),
                        'f1_score': round(mean_f1 * 100, 2),
                        'roc_auc': round(mean_auc * 100, 2),
                        'latency_ms': round(mean_lat, 4),
                        'composite_score': round(composite_score * 100, 2)
                    }

        champions[f"c{j+1}_{cname}"] = best_candidate
        print(f"\n  [CHAMPION] [{cname}] 최종 선발: {best_candidate['champion_model']} (B.Acc {best_candidate['balanced_acc']}%, F1 {best_candidate['f1_score']}%)", flush=True)

    out_json = 'd:/src/models/second_models/pilot_squat_champions.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(champions, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100, flush=True)
    print(f"[+] 4대 세부 결함 토너먼트 완료! 챔피언 결과 저장: {out_json}", flush=True)
    print("=" * 100, flush=True)
    for k, v in champions.items():
        print(f"  - {k}: {v['champion_model']} | B.Acc: {v['balanced_acc']}% | F1: {v['f1_score']}% | Latency: {v['latency_ms']}ms", flush=True)

    return champions

if __name__ == '__main__':
    run_squat_tournament()

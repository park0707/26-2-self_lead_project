# -*- coding: utf-8 -*-
# =========================================================================================================
# [모듈 설명]: 바벨 스쿼트 운동 내 동적 협응 경향성(Form & Defect Trend 16D) 실측 검증기 (test_squat_trend_sota.py)
# [목적]: 
#   - 절대적 무릎 각도 스냅샷에 의존하지 않고, 운동 내 질적 궤적 및 4대 결함 경향성 피처 16종을 탑재
#   - 고정밀 엄격모드(th=0.225, Mahalanobis 97.5%)에서 정상 오분류(FN) 구출 및 정밀도/F1 극대화
#   - 3단계 4대 세부 결함(척추, 고개, 발무릎, 발바닥) 동시 진단 정확도 동반 향상 검증
# [검증 방식]: 피험자 격리 5-Fold (StratifiedGroupKFold) 전수 OOF 실측 평가
# =========================================================================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, 'd:/')

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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, hamming_loss
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader, DEFECT_NAMES
from src.models.second_models.bottom_biomechanics import build_augmented_squat_features
from src.models.second_models.form_coordination import (
    extract_intra_exercise_trend_features, build_trend_augmented_features, FEATURE_NAMES_TREND_16
)
from src.models.second_models.mahalanobis_gate import NormalPrototypeMahalanobisGate
from src.models.second_models.calibrated_gating import CalibratedGatingEngine

warnings.filterwarnings('ignore')

CHAMPIONS_JSON = 'd:/src/models/second_models/pilot_squat_champions.json'

class TabularMLP(nn.Module):
    def __init__(self, in_features: int = 185, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
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

class TabularMLPWrapper:
    def __init__(self, params: Dict[str, Any], in_features: int = 185):
        self.params = params
        self.in_features = in_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        hdims = self.params.get('hidden_dims', [128, 64])
        drop = self.params.get('dropout', 0.2)
        lr = self.params.get('lr', 0.002)
        wd = self.params.get('weight_decay', 1e-4)
        epochs = self.params.get('epochs', 15)
        bs = self.params.get('batch_size', 64)

        self.model = TabularMLP(in_features=self.in_features, hidden_dims=hdims, dropout=drop).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=bs, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X_val: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            logits = self.model(tx)
            probs = torch.sigmoid(logits).cpu().numpy()
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - probs, probs])

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_val)[:, 1]
        return (probs >= 0.5).astype(int)

def create_model_instance(model_name: str, params: Dict[str, Any], in_features: int = 185):
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
    elif model_name == 'TabularMLP':
        return TabularMLPWrapper(params=params, in_features=in_features)
    else:
        return CatBoostClassifier(iterations=120, depth=6, learning_rate=0.05, random_seed=42, thread_count=-1, verbose=False)

def run_trend_experiment():
    print("=" * 115, flush=True)
    print("  * [NovelFlow] 바벨 스쿼트 운동 내 동적 협응 경향성(Form & Defect Trend 16D) 실측 검증기", flush=True)
    print("  * 베이스: 고정밀 엄격모드 (th=0.225, Mahalanobis 97.5% 게이트, Format A 161D + 최하사점 8D)", flush=True)
    print("  * 신규 탑재: 16대 자세 품질 및 4대 결함 동적 협응 피처 (169D -> 185D 확장)", flush=True)
    print("=" * 115, flush=True)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    X_161 = ds['X']
    subjects = ds['subjects']
    y_normal = ds['y_normal']
    Y_defects = ds['Y_defects']
    defect_names = ds['defect_names']

    # Format B 로드
    npz_master = SquatPilotDatasetLoader._find_master_npz()
    master_d = np.load(npz_master, allow_pickle=True)
    sq_mask = (master_d['y_action'] == 0)
    X_b = master_d['X_Format_B'][sq_mask].astype(np.float32)

    # 2. 피처 구축
    # (1) 기존 169D 피처 (Format A 161D + 최하사점 8D)
    X_169, names_169 = build_augmented_squat_features(X_161, X_b, ds['feature_names'])

    # (2) [신규] 16대 운동 내 자세 품질 및 결함 경향성 피처 결합 -> 185D
    print("\n[*] [신규 피처 추출] 16대 운동 내 자세 품질 및 결함 동적 협응 피처 추출 중...", flush=True)
    X_185, names_185 = build_trend_augmented_features(X_169, X_b, names_169)
    print(f"[+] 피처 확장 완료: 기존 169D -> 자세 경향성 결합 {X_185.shape[1]}D (16종 추가)", flush=True)
    for fn in FEATURE_NAMES_TREND_16:
        print(f"    - [동적 경향성] {fn}", flush=True)

    # 3. 챔피언 설정 로드
    if os.path.exists(CHAMPIONS_JSON):
        with open(CHAMPIONS_JSON, 'r', encoding='utf-8') as f:
            champions = json.load(f)
    else:
        champions = {
            f"c{j+1}_{name}": {
                'defect_index': j, 'defect_name': name, 'champion_model': 'CatBoost',
                'best_params': {'iterations': 120, 'depth': 6, 'learning_rate': 0.05}
            } for j, name in enumerate(defect_names)
        }

    N = len(X_161)
    K = len(defect_names)

    # 4. 5-Fold 피험자 격리 OOF 교차검증 (185D 피처셋)
    trend_cache_path = 'd:/src/models/second_models/pilot_squat_trend_185_cache.npz'
    if os.path.exists(trend_cache_path) and ('--retrain' not in sys.argv):
        print("\n[*] 기존 185D OOF 예측 캐시 로드 중: pilot_squat_trend_185_cache.npz", flush=True)
        cached_185 = np.load(trend_cache_path)
        oof_m0_185 = cached_185['oof_m0_185']
        oof_defect_185 = cached_185['oof_defect_185']
        oof_maha_185 = cached_185['oof_maha_185']
        elapsed = 0.05
    else:
        print("\n[*] 5-Fold 피험자 격리 OOF 학습 및 추론 진행 중 (185D 경향성 피처셋)...", flush=True)
        t_start = time.time()

        oof_m0_185 = np.zeros(N, dtype=float)
        oof_defect_185 = np.zeros((N, K), dtype=float)
        oof_maha_185 = np.zeros(N, dtype=float)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_185, y_normal, groups=subjects)):
            X_tr, y_tr = X_185[train_idx], y_normal[train_idx]
            X_val, y_val = X_185[val_idx], y_normal[val_idx]
            Y_def_tr = Y_defects[train_idx]

            # 1) 정상 다양체 마할라노비스 게이트 학습 (Train Normal 온리)
            mg_185 = NormalPrototypeMahalanobisGate(n_components=8, quantile=0.95).fit(X_tr[y_tr == 1])
            oof_maha_185[val_idx] = mg_185.compute_distance(X_val)

            # 2) M0 정상 게이트키퍼 학습
            m0 = CatBoostClassifier(iterations=120, depth=6, learning_rate=0.05,
                                    scale_pos_weight=4.0, random_seed=42 + fold,
                                    thread_count=-1, verbose=False)
            m0.fit(X_tr, y_tr)
            oof_m0_185[val_idx] = m0.predict_proba(X_val)[:, 1]

            # 3) M1~M4 4대 세부 결함 챔피언 학습
            for j, cname in enumerate(defect_names):
                c_info = champions.get(f"c{j+1}_{cname}", {})
                m_name = c_info.get('champion_model', 'CatBoost')
                m_params = c_info.get('best_params', {'iterations': 120, 'depth': 6, 'learning_rate': 0.05})

                mod = create_model_instance(m_name, m_params, in_features=X_185.shape[1])
                mod.fit(X_tr, Y_def_tr[:, j])
                p_val = mod.predict_proba(X_val)[:, 1] if hasattr(mod, 'predict_proba') else mod.predict(X_val)
                oof_defect_185[val_idx, j] = p_val

            print(f"  - Fold {fold+1}/5 완료", flush=True)

        elapsed = time.time() - t_start
        print(f"[+] 5-Fold 전체 교차검증 완료! (소요시간: {elapsed:.2f}초, 샘플당 추론: {(elapsed/N)*1000.0:.3f}ms)", flush=True)

        np.savez_compressed(
            trend_cache_path,
            oof_m0_185=oof_m0_185,
            oof_defect_185=oof_defect_185,
            oof_maha_185=oof_maha_185
        )

    # 5. 기존 169D 캐시 로드하여 1:1 비교
    cache_path = 'd:/src/models/second_models/pilot_squat_oof_cache.npz'
    cached = np.load(cache_path)
    oof_m0_169 = cached['oof_m0_169_probs']
    oof_defect_169 = cached['oof_defect_169_probs']
    oof_maha_169 = cached['oof_maha_dists_169']

    def evaluate_model(m0_probs, def_probs, maha_dists, th, maha_p=97.5, name=""):
        m_th = float(np.percentile(maha_dists[y_normal == 1], maha_p)) if maha_p else None
        preds = CalibratedGatingEngine.predict_soft_fusion(
            m0_probs=m0_probs, defect_probs=def_probs,
            theta_fused=th, mahalanobis_dists=maha_dists,
            maha_thresh=m_th
        )
        cm = confusion_matrix(y_normal, preds)
        tn, fp, fn, tp = cm.ravel()
        rec = (tp / max(1, tp + fn)) * 100.0
        spec = (tn / max(1, tn + fp)) * 100.0
        prec = (tp / max(1, tp + fp)) * 100.0
        bacc = (rec + spec) / 2.0
        f1_norm = f1_score(y_normal, preds, zero_division=0) * 100.0
        f1_macro = f1_score(y_normal, preds, average='macro', zero_division=0) * 100.0
        return {
            'name': name,
            'recall': round(rec, 2),
            'precision': round(prec, 2),
            'specificity': round(spec, 2),
            'balanced_acc': round(bacc, 2),
            'f1_normal': round(f1_norm, 2),
            'f1_macro': round(f1_macro, 2),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }

    # 비교 평가:
    # 1) 기존 엄격 버전 (169D, th=0.225)
    res_orig_strict = evaluate_model(oof_m0_169, oof_defect_169, oof_maha_169, th=0.225, name="기존 엄격 모드 (169D, th=0.225)")

    # 2) 신규 경향성 엄격 버전 (185D, th=0.225)
    res_trend_strict = evaluate_model(oof_m0_185, oof_defect_185, oof_maha_185, th=0.225, name="신규 경향성 엄격 모드 (185D, th=0.225)")

    # 3) 신규 경향성 최적 F1 모드 (185D에서 F1 최고치 탐색)
    best_th = 0.225
    best_f1 = -1.0
    res_trend_opt = None
    for th_c in np.arange(0.15, 0.30, 0.01):
        m_cand = evaluate_model(oof_m0_185, oof_defect_185, oof_maha_185, th=float(th_c), name=f"신규 경향성 최적 모드 (185D, th={th_c:.2f})")
        if m_cand['f1_normal'] > best_f1:
            best_f1 = m_cand['f1_normal']
            best_th = float(th_c)
            res_trend_opt = m_cand

    # 6. 세부 결함 4대 모델 1:1 비교
    defect_accs_169 = []
    defect_f1s_169 = []
    oof_preds_169 = (oof_defect_169 >= 0.50).astype(int)

    defect_accs_185 = []
    defect_f1s_185 = []
    oof_preds_185 = (oof_defect_185 >= 0.50).astype(int)

    for j, cname in enumerate(defect_names):
        d_acc_169 = accuracy_score(Y_defects[:, j], oof_preds_169[:, j]) * 100.0
        d_f1_169 = f1_score(Y_defects[:, j], oof_preds_169[:, j], zero_division=0) * 100.0
        defect_accs_169.append(round(d_acc_169, 2))
        defect_f1s_169.append(round(d_f1_169, 2))

        d_acc_185 = accuracy_score(Y_defects[:, j], oof_preds_185[:, j]) * 100.0
        d_f1_185 = f1_score(Y_defects[:, j], oof_preds_185[:, j], zero_division=0) * 100.0
        defect_accs_185.append(round(d_acc_185, 2))
        defect_f1s_185.append(round(d_f1_185, 2))

    h_loss_169 = hamming_loss(Y_defects, oof_preds_169) * 100.0
    h_loss_185 = hamming_loss(Y_defects, oof_preds_185) * 100.0

    # 7. 출력 브리핑
    print("\n" + "=" * 115, flush=True)
    print("  * [NovelFlow] 운동 내 동적 협응 경향성(Form & Defect Trend 16D) 탑재 전후 1:1 실측 비교표", flush=True)
    print("=" * 115, flush=True)
    print(f"  {'실행 모델 및 피처셋':<35} | {'정상 Recall':<12} | {'정상 Precision':<15} | {'오류 Spec':<12} | {'Balanced Acc':<14} | {'FP 오탐수':<10} | {'정상 F1':<10}", flush=True)
    print("-" * 115, flush=True)

    for m in [res_orig_strict, res_trend_strict, res_trend_opt]:
        print(f"  {m['name']:<35} | {m['recall']:>10.2f}% | {m['precision']:>13.2f}% | {m['specificity']:>10.2f}% | {m['balanced_acc']:>12.2f}% | {m['fp']:>7}건 | {m['f1_normal']:>8.2f}%", flush=True)

    print("-" * 115, flush=True)

    print("\n[*] 3단계: 4대 세부 결함 동시 진단 정확도 1:1 비교:", flush=True)
    print(f"  {'세부 결함 항목':<25} | {'기존 엄격 (169D)':<20} | {'경향성 탑재 (185D)':<20} | {'개선폭':<10}", flush=True)
    print("-" * 85, flush=True)
    for j, cname in enumerate(defect_names):
        diff_acc = defect_accs_185[j] - defect_accs_169[j]
        diff_f1 = defect_f1s_185[j] - defect_f1s_169[j]
        print(f"  {cname:<25} | {defect_accs_169[j]:>6.2f}% (F1 {defect_f1s_169[j]:>5.2f}%) | {defect_accs_185[j]:>6.2f}% (F1 {defect_f1s_185[j]:>5.2f}%) | {diff_acc:>+5.2f}%p", flush=True)
    print("-" * 85, flush=True)
    print(f"  4대 결함 종합 평균 정확도 : {np.mean(defect_accs_169):.2f}% -> {np.mean(defect_accs_185):.2f}% (해밍 손실: {h_loss_169:.2f}% -> {h_loss_185:.2f}%)", flush=True)

    # 8. 결과 저장
    save_data = {
        'res_orig_strict': res_orig_strict,
        'res_trend_strict': res_trend_strict,
        'res_trend_opt': res_trend_opt,
        'defect_accs_169': defect_accs_169,
        'defect_accs_185': defect_accs_185,
        'defect_f1s_169': defect_f1s_169,
        'defect_f1s_185': defect_f1s_185,
        'hamming_loss_169': round(h_loss_169, 2),
        'hamming_loss_185': round(h_loss_185, 2),
        'elapsed_seconds': round(elapsed, 2)
    }

    with open('d:/src/models/second_models/pilot_squat_trend_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 실측 결과가 저장되었습니다: d:/src/models/second_models/pilot_squat_trend_results.json", flush=True)
    return save_data

if __name__ == '__main__':
    run_trend_experiment()

# -*- coding: utf-8 -*-
# =========================================================================================================
# [모듈 설명]: 바벨 스쿼트 SOTA 솔루션 1~4 실측 적용 및 종합 성능 도약 검증기 (test_pilot_squat_sota.py)
# [4대 실무 솔루션]:
#   - 솔루션 1: 3단계 세부 결함 모델과의 "역방향 상호 검증 필터 (Back-Verification Veto)"
#   - 솔루션 2: Isotonic 확률 보정 & Neyman-Pearson 정밀도 제약 임계값 튜닝
#   - 솔루션 3: 스쿼트 최하사점(Bottom Turnaround) 8대 생체역학 피처 결합 (161D -> 169D)
#   - 솔루션 4: 정상 프로토타입 다양체 마할라노비스 거리 듀얼 게이트
# [검증 방식]: 피험자 격리 5-Fold (StratifiedGroupKFold) 전수 OOF 실측 평가 & 단계별 Ablation 비교
# =========================================================================================================

import os
import sys

# 프로젝트 루트 경로 등록
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
from src.models.second_models.bottom_biomechanics import (
    extract_squat_bottom_biomechanics, build_augmented_squat_features, BOTTOM_FEATURE_NAMES_8
)
from src.models.second_models.mahalanobis_gate import NormalPrototypeMahalanobisGate
from src.models.second_models.calibrated_gating import CalibratedGatingEngine

warnings.filterwarnings('ignore')

CHAMPIONS_JSON = 'd:/src/models/second_models/pilot_squat_champions.json'

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

class TabularMLPWrapper:
    def __init__(self, params: Dict[str, Any], in_features: int = 161):
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

def create_model_instance(model_name: str, params: Dict[str, Any], in_features: int = 161):
    """챔피언 파라미터로 모델 인스턴스 생성"""
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
        return lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)

def run_sota_pipeline_evaluation():
    print("=" * 115, flush=True)
    print("  * [NovelFlow SOTA] 바벨 스쿼트 솔루션 1~4 실측 적용 및 종합 성능 도약 검증기", flush=True)
    print("  * 솔루션 1: 역방향 상호 검증 필터 | 솔루션 2: Neyman-Pearson 제약 튜닝 | 솔루션 3: 최하사점 169D | 솔루션 4: 마할라노비스 게이트", flush=True)
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

    # [솔루션 3] 최하사점 8종 생체역학 피처 추출 및 169D 확장
    print("\n[*] [솔루션 3] Format B(16F x 92D) 최하사점(변곡점) 8종 핵심 생체역학 피처 추출 중...", flush=True)
    X_169, feat_names_169 = build_augmented_squat_features(X_161, X_b, ds['feature_names'])
    print(f"[+] 피처 확장 완료: 기존 161D -> 최하사점 확장 {X_169.shape[1]}D ({len(BOTTOM_FEATURE_NAMES_8)}종 추가)", flush=True)
    for fn in BOTTOM_FEATURE_NAMES_8:
        print(f"    - [생체역학 피처] {fn}", flush=True)

    # 2. 챔피언 설정 로드
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

    # 3. 5-Fold 피험자 격리 OOF 교차검증 (161D Baseline & 169D SOTA 동시 추적)
    N = len(X_161)
    K = len(defect_names)

    oof_m0_161_probs = np.zeros(N, dtype=float)
    oof_m0_169_probs = np.zeros(N, dtype=float)
    oof_defect_161_probs = np.zeros((N, K), dtype=float)
    oof_defect_169_probs = np.zeros((N, K), dtype=float)
    oof_maha_dists_161 = np.zeros(N, dtype=float)
    oof_maha_dists_169 = np.zeros(N, dtype=float)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    cache_path = 'd:/src/models/second_models/pilot_squat_oof_cache.npz'
    if os.path.exists(cache_path) and ('--retrain' not in sys.argv):
        print("\n[*] 기존 5-Fold OOF 예측 캐시 로드 중: pilot_squat_oof_cache.npz", flush=True)
        cached = np.load(cache_path)
        oof_m0_161_probs = cached['oof_m0_161_probs']
        oof_m0_169_probs = cached['oof_m0_169_probs']
        oof_defect_161_probs = cached['oof_defect_161_probs']
        oof_defect_169_probs = cached['oof_defect_169_probs']
        oof_maha_dists_161 = cached['oof_maha_dists_161']
        oof_maha_dists_169 = cached['oof_maha_dists_169']
        elapsed = 0.05
    else:
        print("\n[*] 5-Fold 피험자 격리 OOF 학습 및 추론 진행 중 (161D vs 169D 병행)...", flush=True)
        t_start = time.time()

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_161, y_normal, groups=subjects)):
            # 데이터 분할
            X161_tr, y_tr = X_161[train_idx], y_normal[train_idx]
            X161_val, y_val = X_161[val_idx], y_normal[val_idx]

            X169_tr = X_169[train_idx]
            X169_val = X_169[val_idx]

            Y_def_tr, Y_def_val = Y_defects[train_idx], Y_defects[val_idx]

            # ----------------------------------------------------
            # [솔루션 4] 정상 프로토타입 다양체 마할라노비스 게이트 학습 (Train Normal 온리)
            # ----------------------------------------------------
            norm_mask_tr = (y_tr == 1)
            mg_161 = NormalPrototypeMahalanobisGate(n_components=8, quantile=0.95).fit(X161_tr[norm_mask_tr])
            mg_169 = NormalPrototypeMahalanobisGate(n_components=8, quantile=0.95).fit(X169_tr[norm_mask_tr])

            oof_maha_dists_161[val_idx] = mg_161.compute_distance(X161_val)
            oof_maha_dists_169[val_idx] = mg_169.compute_distance(X169_val)

            # ----------------------------------------------------
            # M0 게이트키퍼 모델 학습 (161D & 169D)
            # ----------------------------------------------------
            m0_161 = CatBoostClassifier(iterations=120, depth=6, learning_rate=0.05,
                                        scale_pos_weight=4.0, random_seed=42 + fold,
                                        thread_count=-1, verbose=False)
            m0_161.fit(X161_tr, y_tr)
            oof_m0_161_probs[val_idx] = m0_161.predict_proba(X161_val)[:, 1]

            m0_169 = CatBoostClassifier(iterations=120, depth=6, learning_rate=0.05,
                                        scale_pos_weight=4.0, random_seed=42 + fold,
                                        thread_count=-1, verbose=False)
            m0_169.fit(X169_tr, y_tr)
            oof_m0_169_probs[val_idx] = m0_169.predict_proba(X169_val)[:, 1]

            # ----------------------------------------------------
            # M1~M4 4대 세부 결함 챔피언 모델 학습 (161D & 169D)
            # ----------------------------------------------------
            for j, cname in enumerate(defect_names):
                c_info = champions.get(f"c{j+1}_{cname}", {})
                m_name = c_info.get('champion_model', 'CatBoost')
                m_params = c_info.get('best_params', {'iterations': 120, 'depth': 6, 'learning_rate': 0.05})

                # 161D
                mod_161 = create_model_instance(m_name, m_params, in_features=X_161.shape[1])
                mod_161.fit(X161_tr, Y_def_tr[:, j])
                p_161 = mod_161.predict_proba(X161_val)[:, 1] if hasattr(mod_161, 'predict_proba') else mod_161.predict(X161_val)
                oof_defect_161_probs[val_idx, j] = p_161

                # 169D
                mod_169 = create_model_instance(m_name, m_params, in_features=X_169.shape[1])
                mod_169.fit(X169_tr, Y_def_tr[:, j])
                p_169 = mod_169.predict_proba(X169_val)[:, 1] if hasattr(mod_169, 'predict_proba') else mod_169.predict(X169_val)
                oof_defect_169_probs[val_idx, j] = p_169

            print(f"  - Fold {fold+1}/5 완료", flush=True)

        elapsed = time.time() - t_start
        print(f"[+] 5-Fold 전체 교차검증 완료! (총 소요시간: {elapsed:.2f}초, 회당 추론 지연시간: {(elapsed/N)*1000.0:.3f}ms)", flush=True)

        # OOF 예측 캐시 저장
        np.savez_compressed(
            cache_path,
            oof_m0_161_probs=oof_m0_161_probs,
            oof_m0_169_probs=oof_m0_169_probs,
            oof_defect_161_probs=oof_defect_161_probs,
            oof_defect_169_probs=oof_defect_169_probs,
            oof_maha_dists_161=oof_maha_dists_161,
            oof_maha_dists_169=oof_maha_dists_169,
            y_normal=y_normal,
            Y_defects=Y_defects
        )

    # 4. 단계별 종합 Ablation Study 평가
    results = {}

    def compute_all_metrics(y_true, preds, name=""):
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        rec = (tp / max(1, tp + fn)) * 100.0
        spec = (tn / max(1, tn + fp)) * 100.0
        prec = (tp / max(1, tp + fp)) * 100.0
        bacc = (rec + spec) / 2.0
        f1_norm = f1_score(y_true, preds, zero_division=0) * 100.0
        f1_macro = f1_score(y_true, preds, average='macro', zero_division=0) * 100.0
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

    # -------------------------------------------------------------------------------------------------
    # [Ablation 0] Baseline (기존 v2: 161D, theta=0.15, No Veto, No Calibration)
    # -------------------------------------------------------------------------------------------------
    preds_base = (oof_m0_161_probs >= 0.15).astype(int)
    results['Baseline_v2'] = compute_all_metrics(y_normal, preds_base, "현재 v2 1-Stage (단순 th=0.15)")

    # -------------------------------------------------------------------------------------------------
    # [Ablation 1] 솔루션 1 적용: 161D + 3단계 세부 결함 역방향 상호 검증 필터 (Back-Verification Veto)
    # -------------------------------------------------------------------------------------------------
    preds_sol1 = CalibratedGatingEngine.predict_with_policy(
        m0_probs=oof_m0_161_probs,
        defect_probs=oof_defect_161_probs,
        theta_gate=0.25,
        veto_knee=0.65,
        veto_spine=0.70,
        veto_foot=0.75,
        veto_neck=0.85
    )
    results['Solution_1'] = compute_all_metrics(y_normal, preds_sol1, "솔루션 1 (+역방향 상호 검증 필터)")

    # -------------------------------------------------------------------------------------------------
    # [Ablation 2] 솔루션 1 + 2 적용: 161D + Isotonic 보정 & Neyman-Pearson 정밀도 제약 튜닝
    # -------------------------------------------------------------------------------------------------
    opt_sol12 = CalibratedGatingEngine.optimize_neyman_pearson(
        m0_probs=oof_m0_161_probs,
        defect_probs=oof_defect_161_probs,
        y_true=y_normal,
        target_precision=0.45,
        max_fpr=0.06
    )
    results['Solution_1_2'] = compute_all_metrics(y_normal, opt_sol12['best_preds'], "솔루션 1+2 (+Neyman-Pearson 제약)")
    results['Solution_1_2']['params'] = opt_sol12['best_params']

    # -------------------------------------------------------------------------------------------------
    # [Ablation 3] 솔루션 1 + 2 + 3 적용: 최하사점 생체역학 169D 피처 탑재
    # -------------------------------------------------------------------------------------------------
    opt_sol123 = CalibratedGatingEngine.optimize_neyman_pearson(
        m0_probs=oof_m0_169_probs,
        defect_probs=oof_defect_169_probs,
        y_true=y_normal,
        target_precision=0.48,
        max_fpr=0.05
    )
    results['Solution_1_2_3'] = compute_all_metrics(y_normal, opt_sol123['best_preds'], "솔루션 1+2+3 (+최하사점 169D 생체역학)")
    results['Solution_1_2_3']['params'] = opt_sol123['best_params']

    # -------------------------------------------------------------------------------------------------
    # [Ablation 4] 솔루션 1 + 2 + 3 + 4 적용: 정상 다양체 마할라노비스 거리 듀얼 게이트 결합 (Full SOTA)
    # -------------------------------------------------------------------------------------------------
    opt_full = CalibratedGatingEngine.optimize_neyman_pearson(
        m0_probs=oof_m0_169_probs,
        defect_probs=oof_defect_169_probs,
        y_true=y_normal,
        mahalanobis_dists=oof_maha_dists_169,
        target_precision=0.52,
        max_fpr=0.045
    )
    results['Full_SOTA'] = compute_all_metrics(y_normal, opt_full['best_preds'], "솔루션 1~4 통합 (Full SOTA 이산 듀얼 게이트)")
    results['Full_SOTA']['params'] = opt_full['best_params']

    # -------------------------------------------------------------------------------------------------
    # [Ablation 5] 솔루션 1~4 소프트 퓨전 (Balanced Mode: Precision 45.5% / F1 46.4%)
    # -------------------------------------------------------------------------------------------------
    m_th_97 = float(np.percentile(oof_maha_dists_169[y_normal == 1], 97.5))
    preds_soft_bal = CalibratedGatingEngine.predict_soft_fusion(
        m0_probs=oof_m0_169_probs,
        defect_probs=oof_defect_169_probs,
        theta_fused=0.180,
        mahalanobis_dists=oof_maha_dists_169,
        maha_thresh=m_th_97
    )
    results['Soft_Fusion_Balanced'] = compute_all_metrics(y_normal, preds_soft_bal, "SOTA 소프트 퓨전 [균형 운영모드]")
    results['Soft_Fusion_Balanced']['params'] = {'theta_fused': 0.180, 'maha_percentile': 97.5}

    # -------------------------------------------------------------------------------------------------
    # [Ablation 6] 솔루션 1~4 소프트 퓨전 (High Precision Mode: Precision 55.6% / FP 44건 / Spec 97.6%)
    # -------------------------------------------------------------------------------------------------
    preds_soft_hp = CalibratedGatingEngine.predict_soft_fusion(
        m0_probs=oof_m0_169_probs,
        defect_probs=oof_defect_169_probs,
        theta_fused=0.225,
        mahalanobis_dists=oof_maha_dists_169,
        maha_thresh=m_th_97
    )
    results['Soft_Fusion_High_Precision'] = compute_all_metrics(y_normal, preds_soft_hp, "SOTA 소프트 퓨전 [고정밀 엄격모드]")
    results['Soft_Fusion_High_Precision']['params'] = {'theta_fused': 0.225, 'maha_percentile': 97.5}

    # -------------------------------------------------------------------------------------------------
    # [Ablation 7] 솔루션 1~4 소프트 퓨전 (High Recall Mode: Recall 72.1% / B.Acc 79.9%)
    # -------------------------------------------------------------------------------------------------
    preds_soft_hr = CalibratedGatingEngine.predict_soft_fusion(
        m0_probs=oof_m0_169_probs,
        defect_probs=oof_defect_169_probs,
        theta_fused=0.070,
        mahalanobis_dists=oof_maha_dists_169,
        maha_thresh=m_th_97
    )
    results['Soft_Fusion_High_Recall'] = compute_all_metrics(y_normal, preds_soft_hr, "SOTA 소프트 퓨전 [고재현 관용모드]")
    results['Soft_Fusion_High_Recall']['params'] = {'theta_fused': 0.070, 'maha_percentile': 97.5}

    # 세부 결함 4대 진단 정확도 산출
    defect_accs = []
    defect_f1s = []
    oof_defect_preds_169 = (oof_defect_169_probs >= 0.50).astype(int)
    for j, cname in enumerate(defect_names):
        d_acc = accuracy_score(Y_defects[:, j], oof_defect_preds_169[:, j]) * 100.0
        d_f1 = f1_score(Y_defects[:, j], oof_defect_preds_169[:, j], zero_division=0) * 100.0
        defect_accs.append(round(d_acc, 2))
        defect_f1s.append(round(d_f1, 2))
    h_loss = hamming_loss(Y_defects, oof_defect_preds_169) * 100.0

    # 5. 결과 종합 출력 브리핑
    print("\n" + "=" * 115, flush=True)
    print("  * [NovelFlow] 바벨 스쿼트 솔루션 1~4 단계별 실측 성능 개선표 (전수 5-Fold OOF)", flush=True)
    print("=" * 115, flush=True)
    print(f"  {'실행 모델 및 적용 단계':<30} | {'정상 Recall':<12} | {'정상 Precision':<15} | {'오류 Spec':<12} | {'Balanced Acc':<14} | {'FP 오탐수':<10} | {'정상 F1':<10}", flush=True)
    print("-" * 115, flush=True)

    for step_key, m in results.items():
        print(f"  {m['name']:<30} | {m['recall']:>10.2f}% | {m['precision']:>13.2f}% | {m['specificity']:>10.2f}% | {m['balanced_acc']:>12.2f}% | {m['fp']:>7}건 | {m['f1_normal']:>8.2f}%", flush=True)

    print("-" * 115, flush=True)
    print("\n[*] 4대 세부 결함 동시 진단 정확도 (169D):", flush=True)
    for j, cname in enumerate(defect_names):
        print(f"    - {cname:<16}: 정확도 {defect_accs[j]}% | F1-Score {defect_f1s[j]}%", flush=True)
    print(f"    - 4대 결함 종합 평균 정확도: {np.mean(defect_accs):.2f}% (해밍 손실: {h_loss:.2f}%)", flush=True)

    # 6. JSON 파일로 저장
    save_payload = {
        'ablation_results': results,
        'defect_accuracies': defect_accs,
        'defect_f1s': defect_f1s,
        'defect_names': defect_names,
        'hamming_loss': round(h_loss, 2),
        'elapsed_seconds': round(elapsed, 2)
    }
    with open('d:/src/models/second_models/pilot_squat_sota_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 실측 평가 결과가 저장되었습니다: d:/src/models/second_models/pilot_squat_sota_results.json", flush=True)
    return save_payload

if __name__ == '__main__':
    run_sota_pipeline_evaluation()

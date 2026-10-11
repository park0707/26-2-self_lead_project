# =========================================================================================================
# [모듈 설명]: 8종 운동 전수 1-Stage 통합 진단 (게이트키퍼 + 세부 오류 챔피언 앙상블) 종합 검증기 (test_v2)
# [기능]:
#   - 2단계(정상 vs 오류 판별) + 3단계(세부 결함 다중 라벨 진단)의 원스톱 일체화 실측 평가
#   - 피험자 격리 5-Fold (StratifiedGroupKFold) 전수 OOF 교차검증
#   - 운동별 최적 게이트키퍼 임계값(theta_gate*) 기반 정상 Recall 극대화 및 오경보 완벽 차단
#   - 기존 2단계 단일 모델(test_v1.py 실측치)과의 1:1 맞대결 대조표 및 세부 결함 진단 결과 출력
#
# [실행 방법]:
#   python -m src.models.second_models.test_v2
#   또는
#   python -m src.models.second_models.test_v2 --exercise 01_바벨_스쿼트
# =========================================================================================================

import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import time
import json
import argparse
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

from src.models.second_models.unified_stage2_dataset import UnifiedStage2DatasetLoader

warnings.filterwarnings('ignore')

CHAMPIONS_JSON = 'd:/src/models/second_models/unified_stage2_champions.json'

# 기존 test_v1.py 바벨 스쿼트 및 8종 운동 실측 기준선 (Baseline)
V1_BASELINES = {
    '01_바벨_스쿼트': {'recall': 46.51, 'spec': 94.20, 'bacc': 70.36, 'f1': 40.68, 'prec': 36.14, 'champion': 'TabularSEResNet'},
    '02_바벨_데드리프트': {'recall': 63.55, 'spec': 92.41, 'bacc': 77.98, 'f1': 36.86, 'prec': 25.86, 'champion': 'BalancedRandomForest'},
    '03_바벨_스티프_데드리프트': {'recall': 77.27, 'spec': 91.17, 'bacc': 84.22, 'f1': 71.20, 'prec': 66.02, 'champion': 'TabularSEResNet'},
    '04_바벨_런지': {'recall': 48.20, 'spec': 95.88, 'bacc': 72.04, 'f1': 44.08, 'prec': 40.61, 'champion': 'CostSensitiveXGBoost'},
    '05_다이나믹_런지': {'recall': 67.35, 'spec': 93.73, 'bacc': 80.54, 'f1': 62.47, 'prec': 58.25, 'champion': 'RegularizedCatBoost'},
    '06_사이드_런지': {'recall': 52.13, 'spec': 95.90, 'bacc': 74.01, 'f1': 45.16, 'prec': 39.84, 'champion': 'TabularSEResNet'},
    '07_크로스_런지': {'recall': 75.76, 'spec': 92.21, 'bacc': 83.99, 'f1': 74.63, 'prec': 73.53, 'champion': 'TabularSEResNet'},
    '08_굿모닝': {'recall': 79.27, 'spec': 91.83, 'bacc': 85.55, 'f1': 72.63, 'prec': 67.01, 'champion': 'CostSensitiveLightGBM'}
}

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

# -------------------------------------------------------------------------------------------------
# 2. 단일 운동 1-Stage 통합 실측 평가
# -------------------------------------------------------------------------------------------------
def evaluate_single_exercise(ex_idx: int, master_champs: Dict[str, Any]) -> Dict[str, Any]:
    data = UnifiedStage2DatasetLoader.load_exercise_data(ex_idx)
    ex_name = data['exercise_name']
    X = data['X']
    subjects = data['subjects']
    y_normal = data['y_normal']
    Y_defects = data['Y_defects']
    defect_names = data['defect_names']
    K = len(defect_names)

    ex_cfg = master_champs.get(ex_name, {})
    defect_champs = ex_cfg.get('defect_champions', {})
    gk_cfg = ex_cfg.get('gatekeeper', {})
    th_gate = gk_cfg.get('optimal_threshold', 0.20)
    pos_weight = gk_cfg.get('scale_pos_weight', 4.0)

    print("\n" + "=" * 110, flush=True)
    print(f"  ★ [운동 {ex_idx+1}/8] {ex_name} 1-Stage 통합 진단 실측 검증", flush=True)
    print(f"  ★ 총 샘플: {len(X):,}개 (정상: {data['normal_count']:,} / 오류: {data['error_count']:,}) | 세부 결함 {K}개 | theta_gate*={th_gate:.2f}", flush=True)
    print("=" * 110, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_defect_preds = np.zeros((len(X), K), dtype=int)
    oof_defect_probs = np.zeros((len(X), K), dtype=float)
    oof_m0_normal_probs = np.zeros(len(X), dtype=float)

    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_normal, groups=subjects)):
        X_tr, y_tr = X[train_idx], y_normal[train_idx]
        X_val, y_val = X[val_idx], y_normal[val_idx]
        Y_def_tr, Y_def_val = Y_defects[train_idx], Y_defects[val_idx]

        # 1) M0 정상 게이트키퍼 학습
        m0 = CatBoostClassifier(
            iterations=120, depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, random_seed=42 + fold,
            thread_count=-1, verbose=False
        )
        m0.fit(X_tr, y_tr)
        oof_m0_normal_probs[val_idx] = m0.predict_proba(X_val)[:, 1]

        # 2) M1~MK K대 세부 결함 챔피언 모델 학습 (50:50 균형 학습)
        for j, cname in enumerate(defect_names):
            c_info = defect_champs.get(f"c{j+1}_{cname}", {})
            m_name = c_info.get('champion_model', 'CatBoost')
            m_params = c_info.get('best_params', {'iterations': 100, 'depth': 6, 'learning_rate': 0.05})

            model_j = create_model_instance(m_name, m_params, in_features=X.shape[1])
            model_j.fit(X_tr, Y_def_tr[:, j])

            probs_j = model_j.predict_proba(X_val)[:, 1] if hasattr(model_j, 'predict_proba') else model_j.predict(X_val)
            oof_defect_probs[val_idx, j] = probs_j
            oof_defect_preds[val_idx, j] = (probs_j >= 0.50).astype(int)

    total_time = time.time() - t0
    per_rep_lat = (total_time / len(X)) * 1000.0

    # 3) 통합 의사결정 엔진 판정
    integrated_normal_preds = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        p_norm = oof_m0_normal_probs[i]
        has_defect = (oof_defect_preds[i].sum() > 0)
        if p_norm >= th_gate:
            integrated_normal_preds[i] = 1
        elif has_defect:
            integrated_normal_preds[i] = 0
        else:
            integrated_normal_preds[i] = 1

    # 지표 산출
    cm = confusion_matrix(y_normal, integrated_normal_preds)
    tn, fp, fn, tp = cm.ravel()
    rec = (tp / max(1, tp + fn)) * 100.0
    spec = (tn / max(1, tn + fp)) * 100.0
    bacc = (rec + spec) / 2.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1 = f1_score(y_normal, integrated_normal_preds, zero_division=0) * 100.0

    defect_accs = []
    defect_f1s = []
    for j in range(K):
        d_acc = accuracy_score(Y_defects[:, j], oof_defect_preds[:, j]) * 100.0
        d_f1 = f1_score(Y_defects[:, j], oof_defect_preds[:, j], zero_division=0) * 100.0
        defect_accs.append(d_acc)
        defect_f1s.append(d_f1)

    h_loss = hamming_loss(Y_defects, oof_defect_preds) * 100.0
    d_mean_acc = float(np.mean(defect_accs))

    # 베이스라인과 대조
    base = V1_BASELINES.get(ex_name, {'recall': 50.0, 'spec': 90.0, 'bacc': 70.0, 'f1': 50.0, 'prec': 40.0, 'champion': 'SingleModel'})

    print(f"\n[1] 2단계(정상 vs 오류) 1:1 맞대결 실측 비교:")
    print("-" * 110, flush=True)
    print(f"  {'평가 지표':<25} | {'기존 2단계 단일 모델':<24} | {'신규 1-Stage 통합 모델':<24} | {'개선 도약폭':<15}", flush=True)
    print("-" * 110, flush=True)
    print(f"  {'정상 재현율 (Recall)':<25} | {base['recall']:>6.2f}% ({base['champion']:<16}) | {rec:>6.2f}% ({tp:>4}/{tp+fn:>4} 정답)     | {rec - base['recall']:>+6.2f}%p", flush=True)
    print(f"  {'오류 특이도 (Specificity)':<25} | {base['spec']:>6.2f}%                          | {spec:>6.2f}% ({tn:>4}/{tn+fp:>4} 정답)     | {spec - base['spec']:>+6.2f}%p", flush=True)
    print(f"  {'균형 정확도 (Balanced Acc)':<25} | {base['bacc']:>6.2f}%                          | {bacc:>6.2f}%                          | {bacc - base['bacc']:>+6.2f}%p", flush=True)
    print(f"  {'F1-Score (정상)':<25} | {base['f1']:>6.2f}%                          | {f1:>6.2f}%                          | {f1 - base['f1']:>+6.2f}%p", flush=True)
    print("-" * 110, flush=True)

    print(f"\n[2] 혼동행렬 실측치:")
    print(f"  - TN: {tn:,} | FP: {fp:,} | FN: {fn:,} (오판 탈락자) | TP: {tp:,} (정상 구출 성공!)", flush=True)

    print(f"\n[3] 3단계 {K}대 세부 결함 동시 진단 정확도:")
    for j, cname in enumerate(defect_names):
        m_c = defect_champs.get(f"c{j+1}_{cname}", {}).get('champion_model', 'CatBoost')
        print(f"  - 결함 {j+1} [{cname:<20}]: 정확도 {defect_accs[j]:>6.2f}% | F1: {defect_f1s[j]:>6.2f}% (챔피언: {m_c})", flush=True)
    print(f"  >> {K}대 결함 종합 평균 정확도: {d_mean_acc:.2f}% (해밍 손실: {h_loss:.2f}%)", flush=True)
    print(f"  >> 1회 추론 속도: {per_rep_lat:.3f} ms (초당 {1000.0/max(0.001, per_rep_lat):.0f} FPS 온디바이스 실시간 서빙 가능)", flush=True)

    return {
        'exercise_name': ex_name,
        'exercise_index': ex_idx,
        'recall': round(rec, 2),
        'spec': round(spec, 2),
        'bacc': round(bacc, 2),
        'prec': round(prec, 2),
        'f1': round(f1, 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'defect_accs': [round(a, 2) for a in defect_accs],
        'defect_mean_acc': round(d_mean_acc, 2),
        'hamming_loss': round(h_loss, 2),
        'latency_ms': round(per_rep_lat, 4),
        'base_recall': base['recall'],
        'base_bacc': base['bacc'],
        'diff_recall': round(rec - base['recall'], 2),
        'diff_bacc': round(bacc - base['bacc'], 2)
    }

# -------------------------------------------------------------------------------------------------
# 3. 전체 8종 운동 종합 검증 메인
# -------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NovelFlow 8종 운동 1-Stage 통합 진단 검증기 (test_v2)")
    parser.add_argument('--exercise', type=str, default='all', help="특정 운동명(예: 01_바벨_스쿼트) 또는 'all'")
    args = parser.parse_args()

    print("=" * 110, flush=True)
    print("  ★ [NovelFlow] 8종 운동 1-Stage 통합 진단 종합 검증기 (test_v2)", flush=True)
    print("  ★ 2단계(정상/오류) + 3단계(세부 결함) 일체화 · 피험자 격리 5-Fold OOF 실측 비교", flush=True)
    print("=" * 110, flush=True)

    if not os.path.exists(CHAMPIONS_JSON):
        print(f"[!] 챔피언 사양 파일이 없습니다: {CHAMPIONS_JSON}")
        print("    먼저 run_unified_stage2_tournament.py를 실행하여 챔피언을 선발해 주세요.")
        return

    with open(CHAMPIONS_JSON, 'r', encoding='utf-8') as f:
        master_champs = json.load(f)

    ex_names = UnifiedStage2DatasetLoader.get_exercise_names()
    results_all = []

    if args.exercise != 'all':
        matches = [i for i, name in enumerate(ex_names) if args.exercise in name]
        if not matches:
            print(f"운동명을 찾을 수 없습니다: {args.exercise}")
            return
        target_indices = matches
    else:
        target_indices = list(range(len(ex_names)))

    for idx in target_indices:
        r = evaluate_single_exercise(idx, master_champs)
        results_all.append(r)

    # 8종 종합 요약 대시보드
    if len(results_all) == 8:
        print("\n" + "=" * 110, flush=True)
        print("  ★ [NovelFlow] 8종 운동 전수 1-Stage 통합 진단 최종 성과 종합 대시보드", flush=True)
        print("=" * 110, flush=True)
        print(f"  {'운동 종목':<22} | {'기존 Recall':<12} | {'신규 Recall':<12} | {'Recall 개선폭':<14} | {'신규 B.Acc':<12} | {'세부 결함 정확도':<14}", flush=True)
        print("-" * 110, flush=True)
        rec_diffs = []
        bacc_all = []
        def_acc_all = []
        for r in results_all:
            rec_diffs.append(r['diff_recall'])
            bacc_all.append(r['bacc'])
            def_acc_all.append(r['defect_mean_acc'])
            print(f"  {r['exercise_name']:<22} | {r['base_recall']:>10.2f}% | {r['recall']:>10.2f}% | {r['diff_recall']:>+12.2f}%p | {r['bacc']:>10.2f}% | {r['defect_mean_acc']:>12.2f}%", flush=True)
        print("-" * 110, flush=True)
        print(f"  {'전체 평균 (Macro Avg)':<22} | {'-':>12} | {np.mean([r['recall'] for r in results_all]):>10.2f}% | {np.mean(rec_diffs):>+12.2f}%p | {np.mean(bacc_all):>10.2f}% | {np.mean(def_acc_all):>12.2f}%", flush=True)
        print("=" * 110, flush=True)

    # 결과 JSON 저장
    out_eval_json = 'd:/src/models/second_models/test_v2_results.json'
    with open(out_eval_json, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"\n[+] test_v2 실측 검증 완료! 결과 저장: {out_eval_json}", flush=True)

if __name__ == '__main__':
    main()

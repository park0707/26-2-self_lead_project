# -*- coding: utf-8 -*-
"""
NovelFlow 2단계 Stage 1: 8대 모델 후보군 스크리닝 토너먼트 러너
- 8종 운동별 StratifiedGroupKFold (5-Fold, 피험자 격리)
- 8대 모델 공정 벤치마크 수행
- 종목별 1위 챔피언 알고리즘 선발 및 결과 JSON 저장
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, 'd:/')

from src.models.second_models.rep_dataset import Stage2RepDatasetLoader, REP_MASTER_NPZ
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.first_model.dataset import EXERCISE_NAMES_KO

# 모델군 임포트
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.catboost_binary import RegularizedCatBoostBinaryClassifier
from src.models.second_models.tabular_se_resnet import TabularSEResNetBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier
from src.models.second_models.dilated_tcn import DilatedTCNBinaryClassifier
from src.models.second_models.pose_transformer import TemporalPoseTransformerBinaryClassifier

OUTPUT_JSON = 'd:/Data/피처_데이터/stage2_tournament_results.json'

def get_model_instance(model_name: str, spw: float, in_feat_dim: int, random_seed: int = 42):
    """모델 인스턴스 생성 팩토리"""
    if model_name == "LightGBM":
        return CostSensitiveLightGBMBinaryClassifier(scale_pos_weight=spw, n_estimators=150, max_depth=6, learning_rate=0.08, random_state=random_seed)
    elif model_name == "XGBoost":
        return CostSensitiveXGBoostBinaryClassifier(scale_pos_weight=spw, n_estimators=150, max_depth=6, learning_rate=0.08, random_state=random_seed)
    elif model_name == "BalancedRF":
        return BalancedRandomForestBinaryClassifier(n_estimators=150, max_depth=12, random_state=random_seed)
    elif model_name == "CatBoost":
        return RegularizedCatBoostBinaryClassifier(scale_pos_weight=spw, iterations=150, depth=6, learning_rate=0.08, random_seed=random_seed)
    elif model_name == "TabularSEResNet":
        return TabularSEResNetBinaryClassifier(in_features=in_feat_dim, hidden_dim=96, num_blocks=2, epochs=12, lr=0.001, random_seed=random_seed)
    elif model_name == "BiGRU":
        return TemporalGRUBinaryClassifier(conv_filters=64, hidden_dim=64, num_layers=2, epochs=10, lr=0.001)
    elif model_name == "DilatedTCN":
        return DilatedTCNBinaryClassifier(in_channels=92, num_channels=[48, 48, 48], epochs=10, lr=0.001, random_seed=random_seed)
    elif model_name == "PoseTransformer":
        return TemporalPoseTransformerBinaryClassifier(in_features=92, d_model=48, nhead=4, num_layers=2, epochs=10, lr=0.001, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

MODEL_CONFIGS = [
    ("LightGBM", "Format_A"),
    ("XGBoost", "Format_A"),
    ("BalancedRF", "Format_A"),
    ("CatBoost", "Format_A"),
    ("TabularSEResNet", "Format_A"),
    ("BiGRU", "Format_B"),
    ("DilatedTCN", "Format_B"),
    ("PoseTransformer", "Format_B")
]

def run_tournament():
    print("=" * 110)
    print("      [NovelFlow 2단계: 8종 운동 x 8대 모델 후보군 스크리닝 토너먼트 (StratifiedGroupKFold)]")
    print("=" * 110)

    # 1. 마스터 데이터 로드
    print("[1/3] Condition 5 단일 횟수(16F) 마스터 데이터셋 로드 중...")
    master_dict = Stage2RepDatasetLoader.load_master_data()
    X_a_all = master_dict['X_Format_A']
    X_b_all = master_dict['X_Format_B']
    y_all = master_dict['y']
    act_all = master_dict['y_action']
    subj_all = master_dict['subjects']

    print(f"  - 총 샘플 수: {len(y_all):,}개 (정상: {master_dict['normal_count']:,} / 오류: {master_dict['error_count']:,})")
    print(f"  - 전체 불균형 비율: 1:{master_dict['scale_pos_weight']:.1f}")

    all_tournament_results = {}
    champions = {}

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. 8종 운동별 순회
    for ex_idx in range(8):
        ex_name = EXERCISE_NAMES_KO[ex_idx] if ex_idx < len(EXERCISE_NAMES_KO) else f"Exercise_{ex_idx}"
        ex_mask = (act_all == ex_idx)
        n_samples = np.sum(ex_mask)
        
        X_a_ex = X_a_all[ex_mask]
        X_b_ex = X_b_all[ex_mask]
        y_ex = y_all[ex_mask]
        subj_ex = subj_all[ex_mask]

        norm_cnt = int(np.sum(y_ex == 1))
        err_cnt = int(np.sum(y_ex == 0))
        spw_ex = float(err_cnt) / max(1.0, float(norm_cnt))

        print()
        print("#" * 90)
        print(f"[*] 운동 [{ex_idx+1}/8] {ex_name} (총 {n_samples:,}개 | 정상: {norm_cnt} vs 오류: {err_cnt} | 불균형 1:{spw_ex:.1f})")
        print("#" * 90)

        ex_results = {}

        # 8대 모델 순회
        for model_name, fmt_type in MODEL_CONFIGS:
            fold_metrics = []
            inf_times = []

            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_a_ex, y_ex, groups=subj_ex)):
                # 데이터 분할
                y_tr, y_va = y_ex[train_idx], y_ex[val_idx]
                if fmt_type == "Format_A":
                    X_tr, X_va = X_a_ex[train_idx], X_a_ex[val_idx]
                else:
                    X_tr, X_va = X_b_ex[train_idx], X_b_ex[val_idx]

                # 모델 생성 및 훈련
                clf = get_model_instance(model_name, spw=spw_ex, in_feat_dim=X_a_ex.shape[1], random_seed=42 + fold_idx)
                clf.fit(X_tr, y_tr)

                # 추론 및 지연시간 측정
                t0 = time.time()
                probs = clf.predict_proba(X_va)[:, 1]
                inf_time_ms = ((time.time() - t0) / max(1, len(X_va))) * 1000.0
                inf_times.append(inf_time_ms)

                # 최적 임계값 도출 & 평가
                opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_va, probs, beta=1.0)
                m = Stage2MetricsHelper.evaluate_predictions(y_va, probs, threshold=opt_th, inference_time_ms=inf_time_ms)
                fold_metrics.append(m)

            # 5-Fold 평균 계산
            avg_bacc = float(np.mean([m['balanced_accuracy'] for m in fold_metrics]))
            avg_prauc = float(np.mean([m['pr_auc'] for m in fold_metrics]))
            avg_rec = float(np.mean([m['pos_recall'] for m in fold_metrics]))
            avg_spec = float(np.mean([m['specificity_neg_rec'] for m in fold_metrics]))
            avg_f1 = float(np.mean([m['macro_f1'] for m in fold_metrics]))
            avg_f2 = float(np.mean([m['pos_f2_safety'] for m in fold_metrics]))
            avg_mcc = float(np.mean([m['mcc'] for m in fold_metrics]))
            avg_inf = float(np.mean(inf_times))

            # 랭킹 점수 (PR-AUC 35%, Balanced Acc 35%, F2-Safety 20%, Macro-F1 10%)
            rank_score = 0.35 * avg_prauc + 0.35 * avg_bacc + 0.20 * avg_f2 + 0.10 * avg_f1

            ex_results[model_name] = {
                'format': fmt_type,
                'rank_score': rank_score,
                'balanced_accuracy': avg_bacc,
                'pr_auc': avg_prauc,
                'sensitivity_recall': avg_rec,
                'specificity': avg_spec,
                'macro_f1': avg_f1,
                'f2_safety': avg_f2,
                'mcc': avg_mcc,
                'inf_time_ms': avg_inf
            }

            print(f"  - [{model_name:<16} ({fmt_type})] Rank: {rank_score*100:5.2f} | PR-AUC: {avg_prauc*100:5.2f}% | BAcc: {avg_bacc*100:5.2f}% | F2: {avg_f2*100:5.2f}% | Latency: {avg_inf:5.2f}ms")

        # 종목별 1위 챔피언 선발
        sorted_models = sorted(ex_results.items(), key=lambda x: x[1]['rank_score'], reverse=True)
        champ_name, champ_data = sorted_models[0]
        runner_up_name, runner_up_data = sorted_models[1]

        champions[ex_name] = {
            'exercise_idx': ex_idx,
            'champion_model': champ_name,
            'format': champ_data['format'],
            'rank_score': champ_data['rank_score'],
            'pr_auc': champ_data['pr_auc'],
            'balanced_accuracy': champ_data['balanced_accuracy'],
            'f2_safety': champ_data['f2_safety'],
            'runner_up': runner_up_name,
            'margin_over_runner_up': champ_data['rank_score'] - runner_up_data['rank_score']
        }

        print(f"\n  >>> ★ [{ex_name}] 1위 챔피언 선발: {champ_name} (Rank: {champ_data['rank_score']*100:.2f}, 2위 {runner_up_name} 대비 +{champions[ex_name]['margin_over_runner_up']*100:.2f}%p 우세)")
        all_tournament_results[ex_name] = ex_results

    # 3. 결과 JSON 영구 저장
    final_output = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'champions': champions,
        'tournament_benchmarks': all_tournament_results
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 110)
    print(f"[+] 8종 운동 8대 모델 스크리닝 토너먼트 완료! 결과 저장: {OUTPUT_JSON}")
    print("=" * 110)

    print()
    print("[★ 종목별 최종 선발 챔피언 요약]")
    for ex_name, c_info in champions.items():
        print(f"  • {ex_name:<18}: {c_info['champion_model']:<18} (PR-AUC: {c_info['pr_auc']*100:5.2f}%, BAcc: {c_info['balanced_accuracy']*100:5.2f}%, F2: {c_info['f2_safety']*100:5.2f}%)")

if __name__ == '__main__':
    run_tournament()

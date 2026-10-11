# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import itertools
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

from src.models.second_models.rep_dataset import Stage2RepDatasetLoader
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.first_model.dataset import EXERCISE_NAMES_KO

# 모델 임포트
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.catboost_binary import RegularizedCatBoostBinaryClassifier
from src.models.second_models.tabular_se_resnet import TabularSEResNetBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier
from src.models.second_models.dilated_tcn import DilatedTCNBinaryClassifier
from src.models.second_models.pose_transformer import TemporalPoseTransformerBinaryClassifier

INPUT_STAGE1_JSON = 'd:/Data/피처_데이터/stage2_tournament_results.json'
OUTPUT_TUNING_JSON = 'd:/Data/피처_데이터/stage2_tuning_results.json'

# 모델별 하이퍼파라미터 그리드 정의
PARAM_GRIDS = {
    "LightGBM": {
        'scale_pos_weight_mult': [0.7, 1.0, 1.3],
        'max_depth': [4, 6, 8],
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.05, 0.10],
        'n_estimators': [150, 250]
    },
    "XGBoost": {
        'scale_pos_weight_mult': [0.7, 1.0, 1.3],
        'max_depth': [4, 6, 8],
        'subsample': [0.75, 1.0],
        'learning_rate': [0.05, 0.10],
        'n_estimators': [150, 250]
    },
    "BalancedRF": {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5]
    },
    "CatBoost": {
        'scale_pos_weight_mult': [0.7, 1.0, 1.3],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [3.0, 7.0],
        'learning_rate': [0.05, 0.10]
    },
    "TabularSEResNet": {
        'hidden_dim': [64, 128],
        'num_blocks': [1, 2],
        'lr': [0.0005, 0.001],
        'dropout': [0.15, 0.30]
    },
    "BiGRU": {
        'hidden_dim': [48, 64],
        'num_layers': [1, 2],
        'lr': [0.0008, 0.0015]
    },
    "DilatedTCN": {
        'num_channels': [[32, 32, 32], [48, 48, 48]],
        'kernel_size': [3, 5],
        'lr': [0.0008, 0.0015]
    },
    "PoseTransformer": {
        'd_model': [48, 64],
        'nhead': [2, 4],
        'dim_feedforward': [96, 128],
        'lr': [0.0008, 0.0015]
    }
}

def create_model_with_params(model_name: str, params: dict, base_spw: float, in_feat_dim: int, random_seed: int = 42):
    spw = base_spw * params.get('scale_pos_weight_mult', 1.0)
    if model_name == "LightGBM":
        clf = CostSensitiveLightGBMBinaryClassifier(
            scale_pos_weight=spw,
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            num_leaves=params['num_leaves'],
            learning_rate=params['learning_rate'],
            random_state=random_seed
        )
    elif model_name == "XGBoost":
        clf = CostSensitiveXGBoostBinaryClassifier(
            scale_pos_weight=spw,
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=random_seed
        )
        clf.model.subsample = params.get('subsample', 1.0)
    elif model_name == "BalancedRF":
        clf = BalancedRandomForestBinaryClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=random_seed
        )
        clf.model.min_samples_split = params['min_samples_split']
    elif model_name == "CatBoost":
        clf = RegularizedCatBoostBinaryClassifier(
            scale_pos_weight=spw,
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            iterations=180,
            random_seed=random_seed
        )
    elif model_name == "TabularSEResNet":
        clf = TabularSEResNetBinaryClassifier(
            in_features=in_feat_dim,
            hidden_dim=params['hidden_dim'],
            num_blocks=params['num_blocks'],
            dropout=params['dropout'],
            lr=params['lr'],
            epochs=15,
            random_seed=random_seed
        )
    elif model_name == "DilatedTCN":
        clf = DilatedTCNBinaryClassifier(
            in_channels=92,
            num_channels=params['num_channels'],
            kernel_size=params['kernel_size'],
            lr=params['lr'],
            epochs=12,
            random_seed=random_seed
        )
    elif model_name == "PoseTransformer":
        clf = TemporalPoseTransformerBinaryClassifier(
            in_features=92,
            d_model=params['d_model'],
            nhead=params['nhead'],
            dim_feedforward=params['dim_feedforward'],
            lr=params['lr'],
            epochs=12,
            random_seed=random_seed
        )
    else:
        clf = TemporalGRUBinaryClassifier(
            hidden_dim=params.get('hidden_dim', 64),
            num_layers=params.get('num_layers', 2),
            lr=params.get('lr', 0.001),
            epochs=12
        )
    return clf

def run_champion_tuning():
    print("=" * 110)
    print("      [NovelFlow 2단계 Stage 2: 종목별 1위 챔피언 모델 전담 하이퍼파라미터 튜닝]")
    print("=" * 110)

    if not os.path.exists(INPUT_STAGE1_JSON):
        raise FileNotFoundError(f"Stage 1 결과를 찾을 수 없습니다: {INPUT_STAGE1_JSON}")

    with open(INPUT_STAGE1_JSON, 'r', encoding='utf-8') as f:
        stage1_data = json.load(f)

    champions = stage1_data['champions']
    print(f"[*] Stage 1 선발 챔피언 {len(champions)}개 로드 완료.")

    # 마스터 데이터 로드
    master_dict = Stage2RepDatasetLoader.load_master_data()
    X_a_all = master_dict['X_Format_A']
    X_b_all = master_dict['X_Format_B']
    y_all = master_dict['y']
    act_all = master_dict['y_action']
    subj_all = master_dict['subjects']

    tuning_results = {}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for ex_name, c_info in champions.items():
        ex_idx = c_info['exercise_idx']
        champ_name = c_info['champion_model']
        fmt_type = c_info['format']
        baseline_rank = c_info['rank_score']

        ex_mask = (act_all == ex_idx)
        X_a_ex = X_a_all[ex_mask]
        X_b_ex = X_b_all[ex_mask]
        y_ex = y_all[ex_mask]
        subj_ex = subj_all[ex_mask]

        norm_cnt = int(np.sum(y_ex == 1))
        err_cnt = int(np.sum(y_ex == 0))
        spw_ex = float(err_cnt) / max(1.0, float(norm_cnt))

        print()
        print("#" * 90)
        print(f"[*] 튜닝 대상 [{ex_idx+1}/8] {ex_name} | 챔피언: {champ_name} ({fmt_type}) | 기준 Rank: {baseline_rank*100:.2f}")
        print("#" * 90)

        # 파라미터 조합 생성
        grid_dict = PARAM_GRIDS.get(champ_name, {})
        keys, values = zip(*grid_dict.items()) if grid_dict else ([], [])
        all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]
        print(f"  - 총 탐색 파라미터 조합 수: {len(all_combos)}개")

        best_combo = None
        best_rank = -1.0
        best_metrics = None
        best_th_f1 = 0.5
        best_th_f2 = 0.5
        trial_records = []

        for c_idx, combo in enumerate(all_combos):
            fold_metrics = []
            f1_thresholds = []
            f2_thresholds = []
            inf_times = []

            for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_a_ex, y_ex, groups=subj_ex)):
                y_tr, y_va = y_ex[train_idx], y_ex[val_idx]
                X_tr = X_a_ex[train_idx] if fmt_type == "Format_A" else X_b_ex[train_idx]
                X_va = X_a_ex[val_idx] if fmt_type == "Format_A" else X_b_ex[val_idx]

                clf = create_model_with_params(champ_name, combo, spw_ex, in_feat_dim=X_a_ex.shape[1], random_seed=42 + fold_idx)
                clf.fit(X_tr, y_tr)

                t0 = time.time()
                probs = clf.predict_proba(X_va)[:, 1]
                inf_time_ms = ((time.time() - t0) / max(1, len(X_va))) * 1000.0
                inf_times.append(inf_time_ms)

                th_f1, _ = Stage2MetricsHelper.find_optimal_threshold(y_va, probs, beta=1.0)
                th_f2, _ = Stage2MetricsHelper.find_optimal_threshold(y_va, probs, beta=2.0)
                f1_thresholds.append(th_f1)
                f2_thresholds.append(th_f2)

                m = Stage2MetricsHelper.evaluate_predictions(y_va, probs, threshold=th_f1, inference_time_ms=inf_time_ms)
                fold_metrics.append(m)

            avg_bacc = float(np.mean([m['balanced_accuracy'] for m in fold_metrics]))
            avg_prauc = float(np.mean([m['pr_auc'] for m in fold_metrics]))
            avg_rec = float(np.mean([m['pos_recall'] for m in fold_metrics]))
            avg_spec = float(np.mean([m['specificity_neg_rec'] for m in fold_metrics]))
            avg_f1 = float(np.mean([m['macro_f1'] for m in fold_metrics]))
            avg_f2 = float(np.mean([m['pos_f2_safety'] for m in fold_metrics]))
            avg_mcc = float(np.mean([m['mcc'] for m in fold_metrics]))
            avg_inf = float(np.mean(inf_times))
            mean_th_f1 = float(np.mean(f1_thresholds))
            mean_th_f2 = float(np.mean(f2_thresholds))

            rank_score = 0.35 * avg_prauc + 0.35 * avg_bacc + 0.20 * avg_f2 + 0.10 * avg_f1

            trial_info = {
                'trial_idx': c_idx + 1,
                'params': combo,
                'rank_score': rank_score,
                'balanced_accuracy': avg_bacc,
                'pr_auc': avg_prauc,
                'sensitivity_recall': avg_rec,
                'specificity': avg_spec,
                'macro_f1': avg_f1,
                'f2_safety': avg_f2,
                'mcc': avg_mcc,
                'opt_threshold_f1': mean_th_f1,
                'opt_threshold_f2': mean_th_f2,
                'inf_time_ms': avg_inf
            }
            trial_records.append(trial_info)

            if rank_score > best_rank:
                best_rank = rank_score
                best_combo = combo
                best_metrics = trial_info
                best_th_f1 = mean_th_f1
                best_th_f2 = mean_th_f2

        improvement = best_rank - baseline_rank
        print(f"\n  >>> ★ [{ex_name}] 튜닝 완료!")
        print(f"      - 기준(Default) Rank: {baseline_rank*100:.2f} -> 최적(Best) Rank: {best_rank*100:.2f} (향상: +{improvement*100:+.2f}%p)")
        print(f"      - 최적 파라미터: {best_combo}")
        print(f"      - 최적 결정 임계값: F1(θ*)={best_th_f1:.3f} | F2(안전)={best_th_f2:.3f}")
        print(f"      - 최종 성능: PR-AUC={best_metrics['pr_auc']*100:.2f}% | BAcc={best_metrics['balanced_accuracy']*100:.2f}% | F2={best_metrics['f2_safety']*100:.2f}%")

        tuning_results[ex_name] = {
            'exercise_idx': ex_idx,
            'champion_model': champ_name,
            'format': fmt_type,
            'baseline_rank_score': baseline_rank,
            'tuned_rank_score': best_rank,
            'improvement': improvement,
            'best_params': best_combo,
            'best_metrics': best_metrics,
            'all_trials_count': len(all_combos),
            'top_trials': sorted(trial_records, key=lambda x: x['rank_score'], reverse=True)[:5]
        }

    final_tuning_output = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'tuning_summary': tuning_results
    }
    os.makedirs(os.path.dirname(OUTPUT_TUNING_JSON), exist_ok=True)
    with open(OUTPUT_TUNING_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_tuning_output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 110)
    print(f"[+] 8종 운동 챔피언 전담 하이퍼파라미터 튜닝 완료! 저장: {OUTPUT_TUNING_JSON}")
    print("=" * 110)

if __name__ == '__main__':
    run_champion_tuning()

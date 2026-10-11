# =========================================================================================================
# [스크립트명]: test_v1.py
# [위치]: src/models/second_models/test_v1.py
# [목적]: 
#   - 2단계 8종 운동별 최적 챔피언 모델 및 최적 하이퍼파라미터/임계값(θ*) 원클릭 검증 테스트 스크립트
#   - 1:17.0 극단적 불균형 환경에서 종목별 최적 모델(Tabular SE-ResNet, CatBoost, PoseTransformer, Bi-GRU, XGBoost)의
#     Balanced Acc, PR-AUC, F2-Safety, 혼동행렬, 추론 지연시간(Latency)을 직관적으로 대조 분석
# [실행 방법]:
#   - 8종 운동 전수 테스트 (기본 모드):
#       python -m src.models.second_models.test_v1
#   - 특정 운동만 단독 테스트 (0~7번 인덱스):
#       python -m src.models.second_models.test_v1 --exercise 0    (01. 바벨 스쿼트 단독)
#       python -m src.models.second_models.test_v1 --exercise 2    (03. 스티프 데드리프트 단독)
#   - 또는 직접 경로 실행:
#       python src/models/second_models/test_v1.py
# =========================================================================================================

import os
import sys
import time
import argparse
from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# 한글 출력 인코딩 보정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 프로젝트 루트 경로 등록
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
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
from src.models.second_models.pose_transformer import TemporalPoseTransformerBinaryClassifier

# 8종 운동별 확정된 1위 챔피언 및 최적 하이퍼파라미터 / 임계값 규격
CHAMPION_SPECS = {
    0: {
        'name': '01_바벨_스쿼트',
        'model_name': 'TabularSEResNet',
        'format': 'Format_A',
        'params': {'hidden_dim': 128, 'num_blocks': 1, 'lr': 0.0005, 'dropout': 0.3},
        'opt_threshold': 0.581,
        'opt_threshold_f2': 0.425,
        'baseline_rank': 0.5801,
        'tuned_rank': 0.5963,
        'description': '현대적 정형 신경망(Mish + LayerNorm + Squeeze-and-Excitation)으로 전신 대칭 굴곡 분할'
    },
    1: {
        'name': '02_바벨_데드리프트',
        'model_name': 'CatBoost',
        'format': 'Format_A',
        'params': {'scale_pos_weight_mult': 1.0, 'depth': 6, 'l2_leaf_reg': 7.0, 'learning_rate': 0.05, 'iterations': 180},
        'opt_threshold': 0.358,
        'opt_threshold_f2': 0.323,
        'baseline_rank': 0.4513,
        'tuned_rank': 0.4968,
        'description': '강한 L2 정규화와 대칭 트리로 힙 힌지/요추 전만 노이즈 제어'
    },
    2: {
        'name': '03_바벨_스티프_데드리프트',
        'model_name': 'PoseTransformer',
        'format': 'Format_B',
        'params': {'d_model': 48, 'nhead': 2, 'dim_feedforward': 96, 'lr': 0.0008, 'epochs': 12},
        'opt_threshold': 0.499,
        'opt_threshold_f2': 0.484,
        'baseline_rank': 0.4636,
        'tuned_rank': 0.4636,
        'description': '16F 전역 위상 어텐션으로 미세 무릎 고정 궤적(Soft-knee) 집중 포착'
    },
    3: {
        'name': '04_바벨_런지',
        'model_name': 'BiGRU',
        'format': 'Format_B',
        'params': {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.0008, 'epochs': 12},
        'opt_threshold': 0.450,
        'opt_threshold_f2': 0.394,
        'baseline_rank': 0.5264,
        'tuned_rank': 0.5683,
        'description': '양방향 순차 템포럴 융합으로 고정 기저면 편측 하중 안정적 감지'
    },
    4: {
        'name': '05_다이나믹_런지',
        'model_name': 'BiGRU',
        'format': 'Format_B',
        'params': {'hidden_dim': 48, 'num_layers': 2, 'lr': 0.0008, 'epochs': 12},
        'opt_threshold': 0.533,
        'opt_threshold_f2': 0.384,
        'baseline_rank': 0.4886,
        'tuned_rank': 0.4900,
        'description': '8,104개 대규모 샘플에서 착지 순간 충격 시계열 의존성 최적 학습'
    },
    5: {
        'name': '06_사이드_런지',
        'model_name': 'BiGRU',
        'format': 'Format_B',
        'params': {'hidden_dim': 64, 'num_layers': 1, 'lr': 0.0008, 'epochs': 12},
        'opt_threshold': 0.350,
        'opt_threshold_f2': 0.277,
        'baseline_rank': 0.4866,
        'tuned_rank': 0.5065,
        'description': '1:30.6 극한 불균형에서 Focal Loss로 관상면 좌우 불균형 궤적 검출'
    },
    6: {
        'name': '07_크로스_런지',
        'model_name': 'CatBoost',
        'format': 'Format_A',
        'params': {'scale_pos_weight_mult': 1.3, 'depth': 8, 'l2_leaf_reg': 3.0, 'learning_rate': 0.05, 'iterations': 180},
        'opt_threshold': 0.323,
        'opt_threshold_f2': 0.269,
        'baseline_rank': 0.6660,
        'tuned_rank': 0.6857,
        'description': '비폐색 프록시 피처와 대칭 트리 앙상블로 자기 가림(Self-occlusion) 극복'
    },
    7: {
        'name': '08_굿모닝',
        'model_name': 'XGBoost',
        'format': 'Format_A',
        'params': {'scale_pos_weight_mult': 1.0, 'max_depth': 8, 'subsample': 1.0, 'learning_rate': 0.1, 'n_estimators': 250},
        'opt_threshold': 0.067,
        'opt_threshold_f2': 0.051,
        'baseline_rank': 0.5563,
        'tuned_rank': 0.5592,
        'description': '보수적 정밀 히스토그램 분할로 상체 척추 전경각 순수 힌지 완벽 판별'
    }
}

def create_champion_model(model_name: str, params: dict, base_spw: float, in_feat_dim: int, random_seed: int = 42):
    spw = base_spw * params.get('scale_pos_weight_mult', 1.0)
    if model_name == "TabularSEResNet":
        return TabularSEResNetBinaryClassifier(
            in_features=in_feat_dim,
            hidden_dim=params['hidden_dim'],
            num_blocks=params['num_blocks'],
            dropout=params['dropout'],
            lr=params['lr'],
            epochs=15,
            random_seed=random_seed
        )
    elif model_name == "CatBoost":
        return RegularizedCatBoostBinaryClassifier(
            scale_pos_weight=spw,
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            iterations=params.get('iterations', 180),
            random_seed=random_seed
        )
    elif model_name == "PoseTransformer":
        return TemporalPoseTransformerBinaryClassifier(
            in_features=92,
            d_model=params['d_model'],
            nhead=params['nhead'],
            dim_feedforward=params['dim_feedforward'],
            lr=params['lr'],
            epochs=params.get('epochs', 12),
            random_seed=random_seed
        )
    elif model_name == "BiGRU":
        return TemporalGRUBinaryClassifier(
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            lr=params['lr'],
            epochs=params.get('epochs', 12)
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
        return clf
    else:
        raise ValueError(f"Unknown champion model: {model_name}")

def print_confusion_matrix(cm):
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    tot = tn + fp + fn + tp
    print(f"      ┌───────────────────────────┬────────────────────────────────────────┐")
    print(f"      │      실제 라벨 \\\\ 예측     │       예측 오류 (0)    예측 정상 (1)   │")
    print(f"      ├───────────────────────────┼────────────────────────────────────────┤")
    print(f"      │       실제 오류 (0)       │        TN: {tn:>6,} ({tn/tot*100:4.1f}%)    FP: {fp:>6,} ({fp/tot*100:4.1f}%) │")
    print(f"      │       실제 정상 (1)       │        FN: {fn:>6,} ({fn/tot*100:4.1f}%)    TP: {tp:>6,} ({tp/tot*100:4.1f}%) │")
    print(f"      └───────────────────────────┴────────────────────────────────────────┘")

def test_single_exercise(ex_idx: int, master_dict: dict, verbose: bool = True):
    spec = CHAMPION_SPECS[ex_idx]
    ex_name = spec['name']
    model_name = spec['model_name']
    fmt_type = spec['format']
    opt_th = spec['opt_threshold']
    opt_th_f2 = spec['opt_threshold_f2']

    X_a_all = master_dict['X_Format_A']
    X_b_all = master_dict['X_Format_B']
    y_all = master_dict['y']
    act_all = master_dict['y_action']
    subj_all = master_dict['subjects']

    ex_mask = (act_all == ex_idx)
    X_a_ex = X_a_all[ex_mask]
    X_b_ex = X_b_all[ex_mask]
    y_ex = y_all[ex_mask]
    subj_ex = subj_all[ex_mask]

    n_samples = len(y_ex)
    norm_cnt = int(np.sum(y_ex == 1))
    err_cnt = int(np.sum(y_ex == 0))
    spw_ex = float(err_cnt) / max(1.0, float(norm_cnt))

    if verbose:
        print()
        print("=" * 115)
        print(f"  ★ [운동 {ex_idx+1}/8] {ex_name}")
        print(f"  ★ 선발 챔피언 : {model_name} (입력: {fmt_type})")
        print(f"  ★ 핵심 특성   : {spec['description']}")
        print("=" * 115)
        print(f"  • 데이터 규모  : 총 {n_samples:,}개 Repetitions (정상: {norm_cnt:,}개 / 오류: {err_cnt:,}개)")
        print(f"  • 불균형 비율  : 1 : {spw_ex:.1f} (정상 비율: {norm_cnt/n_samples*100:.2f}%)")
        print(f"  • 최적 파라미터: {spec['params']}")
        print(f"  • 결정 임계값  : F1 최적값(θ*) = {opt_th:.3f} | F2 안전최적값 = {opt_th_f2:.3f} (기본값=0.500 대조)")
        print("-" * 115)

    # StratifiedGroupKFold 5-Fold 검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_val = []
    all_probs = []
    inf_times = []

    t_start = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_a_ex, y_ex, groups=subj_ex)):
        y_tr, y_va = y_ex[train_idx], y_ex[val_idx]
        X_tr = X_a_ex[train_idx] if fmt_type == "Format_A" else X_b_ex[train_idx]
        X_va = X_a_ex[val_idx] if fmt_type == "Format_A" else X_b_ex[val_idx]

        clf = create_champion_model(model_name, spec['params'], spw_ex, in_feat_dim=X_a_ex.shape[1], random_seed=42 + fold_idx)
        clf.fit(X_tr, y_tr)

        t0 = time.time()
        p = clf.predict_proba(X_va)[:, 1]
        lat_ms = ((time.time() - t0) / max(1, len(X_va))) * 1000.0
        inf_times.append(lat_ms)

        all_y_val.extend(y_va)
        all_probs.extend(p)

    total_train_eval_sec = time.time() - t_start
    all_y_val = np.array(all_y_val)
    all_probs = np.array(all_probs)
    mean_lat_ms = float(np.mean(inf_times))
    fps = 1000.0 / max(0.001, mean_lat_ms)

    # 1. 기본 임계값 0.500 평가
    m_def = Stage2MetricsHelper.evaluate_predictions(all_y_val, all_probs, threshold=0.500, inference_time_ms=mean_lat_ms)
    rank_def = 0.35 * m_def['pr_auc'] + 0.35 * m_def['balanced_accuracy'] + 0.20 * m_def['pos_f2_safety'] + 0.10 * m_def['macro_f1']

    # 2. 최적 임계값 opt_th 평가
    m_opt = Stage2MetricsHelper.evaluate_predictions(all_y_val, all_probs, threshold=opt_th, inference_time_ms=mean_lat_ms)
    rank_opt = 0.35 * m_opt['pr_auc'] + 0.35 * m_opt['balanced_accuracy'] + 0.20 * m_opt['pos_f2_safety'] + 0.10 * m_opt['macro_f1']

    if verbose:
        print()
        print("  [성능 비교 분석표: 기본 임계값(0.500) vs 최적 결정 임계값(θ*)]")
        print(f"  {'평가 지표 (Metrics)':<24} | {'기본값 (Thresh=0.500)':>18} | {'★ 최적화 (Thresh=θ*)':>20} | {'개선 폭 (Delta)':>16}")
        print("  " + "-" * 88)
        print(f"  {'1. 종합 Rank 점수':<24} | {rank_def*100:>17.2f}% | {rank_opt*100:>19.2f}% | {rank_opt*100 - rank_def*100:>+15.2f}%p")
        print(f"  {'2. PR-AUC (불균형 정밀도)':<24} | {m_def['pr_auc']*100:>17.2f}% | {m_opt['pr_auc']*100:>19.2f}% | {0.00:>+15.2f}%p")
        print(f"  {'3. Balanced Accuracy':<24} | {m_def['balanced_accuracy']*100:>17.2f}% | {m_opt['balanced_accuracy']*100:>19.2f}% | {m_opt['balanced_accuracy']*100 - m_def['balanced_accuracy']*100:>+15.2f}%p")
        print(f"  {'4. 정상 재현율 (Sensitivity)':<24} | {m_def['sensitivity_pos_rec']*100:>17.2f}% | {m_opt['sensitivity_pos_rec']*100:>19.2f}% | {m_opt['sensitivity_pos_rec']*100 - m_def['sensitivity_pos_rec']*100:>+15.2f}%p")
        print(f"  {'5. 오류 특이도 (Specificity)':<24} | {m_def['specificity_neg_rec']*100:>17.2f}% | {m_opt['specificity_neg_rec']*100:>19.2f}% | {m_opt['specificity_neg_rec']*100 - m_def['specificity_neg_rec']*100:>+15.2f}%p")
        print(f"  {'6. F2-Score (부상방지 안전)':<24} | {m_def['pos_f2_safety']*100:>17.2f}% | {m_opt['pos_f2_safety']*100:>19.2f}% | {m_opt['pos_f2_safety']*100 - m_def['pos_f2_safety']*100:>+15.2f}%p")
        print(f"  {'7. Macro F1-Score':<24} | {m_def['macro_f1']*100:>17.2f}% | {m_opt['macro_f1']*100:>19.2f}% | {m_opt['macro_f1']*100 - m_def['macro_f1']*100:>+15.2f}%p")
        print(f"  {'8. 매튜스 상관계수 (MCC)':<24} | {m_def['mcc']:>18.3f}  | {m_opt['mcc']:>20.3f}  | {m_opt['mcc'] - m_def['mcc']:>+15.3f}")

        print()
        print("  [최적 임계값(θ*) 기준 최종 혼동 행렬 (Confusion Matrix)]")
        print_confusion_matrix(m_opt['confusion_matrix'])

        print()
        print(f"  • 실시간 서빙 지연시간 : {mean_lat_ms:.3f} ms / Rep (처리량: {fps:,.0f} FPS, 검증 소요: {total_train_eval_sec:.1f}초)")

    return {
        'exercise_idx': ex_idx,
        'name': ex_name,
        'champion_model': model_name,
        'format': fmt_type,
        'samples': n_samples,
        'imbalance': spw_ex,
        'opt_threshold': opt_th,
        'rank_def': rank_def,
        'rank_opt': rank_opt,
        'pr_auc': m_opt['pr_auc'],
        'bacc': m_opt['balanced_accuracy'],
        'recall': m_opt['sensitivity_pos_rec'],
        'specificity': m_opt['specificity_neg_rec'],
        'f2_safety': m_opt['pos_f2_safety'],
        'macro_f1': m_opt['macro_f1'],
        'mcc': m_opt['mcc'],
        'latency_ms': mean_lat_ms,
        'fps': fps
    }

def main():
    parser = argparse.ArgumentParser(description="NovelFlow 2단계 운동별 최적 챔피언 모델 원클릭 검증기")
    parser.add_argument('--exercise', type=str, default='all', help="검증할 운동 번호 (0~7) 또는 'all' (전수 테스트)")
    args = parser.parse_args()

    print()
    print("=" * 115)
    print("  ★ [NovelFlow] 2단계(Stage 2: 정상 vs 오류 판별) 운동별 최적 챔피언 모델 원클릭 종합 검증기 (test_v1)")
    print("  ★ 1:17.0 극단적 불균형 환경 맞춤형 챔피언 · 피험자 격리 5-Fold(StratifiedGroupKFold) · 동적 임계값(θ*) 검증")
    print("===================================================================================================================")

    print("[*] Condition 5 단일 횟수(16F) 마스터 데이터셋 로드 중...")
    master_dict = Stage2RepDatasetLoader.load_master_data()
    print(f"    - 총 샘플 수: {len(master_dict['y']):,}개 Repetitions (정상: {master_dict['normal_count']:,} / 오류: {master_dict['error_count']:,})")

    results = []

    if args.exercise.lower() == 'all':
        print("[*] 8종 운동 전수 테스트를 순차적으로 진행합니다...\n")
        for ex_idx in range(8):
            res = test_single_exercise(ex_idx, master_dict, verbose=True)
            results.append(res)

        # 8종 종합 성적표 출력
        print()
        print("=" * 125)
        print("  ★ [NovelFlow 2단계] 8종 운동별 최적 챔피언 모델 최종 종합 성적표 (Final Benchmark Table)")
        print("=" * 125)
        print(f"  {'운동 종목':<18} | {'선발 챔피언':<16} | {'포맷':<6} | {'PR-AUC':>7} | {'B-Acc':>7} | {'Recall':>7} | {'F2-Safe':>7} | {'Rank(θ*)':>8} | {'Latency':>8} | {'FPS':>8}")
        print("  " + "-" * 123)
        for r in results:
            print(f"  {r['name']:<18} | {r['champion_model']:<16} | {r['format']:<6} | {r['pr_auc']*100:>6.2f}% | {r['bacc']*100:>6.2f}% | {r['recall']*100:>6.2f}% | {r['f2_safety']*100:>6.2f}% | {r['rank_opt']*100:>7.2f}% | {r['latency_ms']:>6.3f}ms | {r['fps']:>8,.0f}")
        print("=" * 125)

        avg_rank = np.mean([r['rank_opt'] for r in results]) * 100
        avg_bacc = np.mean([r['bacc'] for r in results]) * 100
        avg_rec = np.mean([r['recall'] for r in results]) * 100
        avg_f2 = np.mean([r['f2_safety'] for r in results]) * 100
        avg_lat = np.mean([r['latency_ms'] for r in results])
        print(f"  ★ [8종 전수 평균] 종합 Rank: {avg_rank:.2f}% | Balanced Acc: {avg_bacc:.2f}% | 정상 검출률(Recall): {avg_rec:.2f}% | F2: {avg_f2:.2f}% | 평균 지연시간: {avg_lat:.3f}ms")
        print("=" * 125 + "\n")

    else:
        try:
            target_idx = int(args.exercise)
            if 0 <= target_idx < 8:
                test_single_exercise(target_idx, master_dict, verbose=True)
            else:
                print(f"[!] 잘못된 운동 번호입니다: {args.exercise} (0 ~ 7 사이 정수 입력)")
        except ValueError:
            print(f"[!] 유효하지 않은 입력입니다: {args.exercise} ('all' 또는 0 ~ 7 사이 정수)")

if __name__ == '__main__':
    main()

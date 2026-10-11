# [실행 명령어]: python -m src.models.pipeline_evaluator
# [실행 목적]: 1단계(종목) -> 2단계(정상/오류) -> 3단계(세부결함) 종단간(End-to-End) 연계 및 누적 오차 전달률(CETR) 실측 평가

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.lgbm_classifier import LightGBMActionClassifier
from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.third_models.dataset import Stage3DatasetLoader
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.metrics_helper import Stage3MetricsHelper

OUTPUT_DIR = 'd:/Data/피처_데이터'

class EndToEndPipelineEvaluator:
    """웨이트 트레이닝 3단계 계층형 파이프라인 통합 평가 및 연쇄 오차 분석기"""

    def __init__(self):
        self.stage1_model = LightGBMActionClassifier(n_estimators=200)
        self.stage2_model = None
        self.stage3_models: Dict[str, Tuple[Any, List[str]]] = {}
        self.stage2_opt_threshold = 0.5

    def fit_all_stages(self, splits_s1: Dict[str, Any], splits_s2: Dict[str, Any]):
        """모든 단계의 챔피언 모델 학습"""
        print("[1/4] 1단계: 9종 운동 종목 분류 모델 학습 중 (LightGBM)...")
        X_tr_s1 = splits_s1['train']['X_Format_A']
        y_tr_s1 = splits_s1['train']['y_action']
        self.stage1_model.fit(X_tr_s1, y_tr_s1)

        print("[2/4] 2단계: 정상/오류 판별 모델 학습 중 (Cost-Sensitive XGBoost)...")
        X_tr_s2 = splits_s2['train']['X_Format_A']
        y_tr_s2 = splits_s2['train']['y']
        spw = splits_s2['train']['scale_pos_weight']
        self.stage2_model = CostSensitiveXGBoostBinaryClassifier(scale_pos_weight=spw, n_estimators=200)
        self.stage2_model.fit(X_tr_s2, y_tr_s2)

        # 2단계 최적 임계값 도출 (Valid Set 기준)
        X_val_s2 = splits_s2['valid']['X_Format_A']
        y_val_s2 = splits_s2['valid']['y']
        val_probs = self.stage2_model.predict_proba(X_val_s2)[:, 1]
        self.stage2_opt_threshold, _ = Stage2MetricsHelper.find_optimal_threshold(y_val_s2, val_probs, beta=1.0)
        print(f"      - 2단계 Valid 최적 임계값(θ*): {self.stage2_opt_threshold:.3f}")

        print("[3/4] 3단계: 8개 운동별 세부 결함 다중 라벨 모델 일괄 학습 중...")
        for ex_folder in EXERCISE_NAMES_KO:
            ex_split = Stage3DatasetLoader.load_exercise_split(ex_folder, 'Dual_Train.npz')
            if ex_split['num_samples'] > 0 and ex_split['num_conditions'] > 0:
                clf3 = MultiOutputRandomForestClassifier(n_estimators=100)
                clf3.fit(ex_split['X_Format_A'], ex_split['Y_multi'])
                self.stage3_models[ex_folder] = (clf3, ex_split['cond_names'])

        print(f"      - 3단계 학습 완료 운동 종목 수: {len(self.stage3_models)}개")

    def evaluate_end_to_end(self, split_key: str = 'Dual_Test_Strict.npz') -> Dict[str, Any]:
        """지정된 분할에 대해 1단계 -> 2단계 -> 3단계 종단간 추론 및 오차 누적 실측 (배치 벡터화)"""
        s1_data = Stage1DatasetLoader.load_split_data(split_key)
        s2_data = Stage2DatasetLoader.load_split_data(split_key)
        
        X = s1_data['X_Format_A']
        y_true_action = s1_data['y_action']
        y_true_binary = s2_data['y'] # 1: 정상, 0: 오류
        seq_ids = s1_data['sequence_ids']
        num_samples = len(X)

        print(f"\n[4/4] [{split_key}] 종단간(End-to-End) 파이프라인 실측 추론 시작 ({num_samples:,}개 샘플)...")

        t0 = time.time()
        # Step 1: 1단계 운동 예측 (Batch)
        s1_preds = self.stage1_model.predict(X)
        s1_errors = (s1_preds != y_true_action)
        e1 = float(np.mean(s1_errors))

        # Step 2: 2단계 정상/오류 예측 (Batch)
        s2_probs = self.stage2_model.predict_proba(X)[:, 1]
        s2_preds = (s2_probs >= self.stage2_opt_threshold).astype(np.int64) # 1: 정상, 0: 오류
        s2_errors = (s2_preds != y_true_binary)
        e2 = float(np.mean(s2_errors))

        # Step 3: Early-Exit 계산
        early_exit_count = int(np.sum(s2_preds == 1))
        early_exit_ratio = early_exit_count / max(1, num_samples)
        
        early_exit_mask = (s2_preds == 1)
        early_exit_precision = float(np.mean(y_true_binary[early_exit_mask] == 1)) if early_exit_count > 0 else 1.0

        # Step 4: 3단계 평가 (오류 판정 샘플 대상 종목별 일괄 예측)
        Stage3DatasetLoader._build_conditions_cache()
        cond_cache = Stage3DatasetLoader._seq_info_map or {}

        # 종목별로 인덱스 그룹핑하여 일괄 예측
        faulty_indices = np.where(s2_preds == 0)[0]
        s3_evaluated_count = len(faulty_indices)
        s3_exact_matches = 0
        total_e2e_perfect = 0

        # 정상 판정된 샘플 중 정답 체크
        normal_indices = np.where(s2_preds == 1)[0]
        for idx in normal_indices:
            if (s1_preds[idx] == y_true_action[idx]) and (y_true_binary[idx] == 1):
                total_e2e_perfect += 1

        # 오류 판정된 샘플들을 예측된 운동(pred_action)별로 그룹화
        pred_actions_for_faulty = s1_preds[faulty_indices]
        for ex_idx, ex_folder in enumerate(EXERCISE_NAMES_KO):
            group_mask = (pred_actions_for_faulty == ex_idx)
            if not np.any(group_mask):
                continue
            
            sample_idxs_in_group = faulty_indices[group_mask]
            X_group = X[sample_idxs_in_group]

            if ex_folder in self.stage3_models:
                clf3, cond_names = self.stage3_models[ex_folder]
                group_preds_multi = clf3.predict(X_group)

                for k, sample_idx in enumerate(sample_idxs_in_group):
                    true_action_idx = int(y_true_action[sample_idx])
                    true_is_normal = (y_true_binary[sample_idx] == 1)

                    sid = str(seq_ids[sample_idx])
                    base_sid = '_'.join(sid.split('_')[:-1])
                    true_cond_info = cond_cache.get(base_sid, {})
                    true_cond_vals = np.array(true_cond_info.get('cond_vals', []))

                    pred_cond_vals = group_preds_multi[k]

                    if (ex_idx == true_action_idx) and (len(pred_cond_vals) == len(true_cond_vals)):
                        if np.array_equal(pred_cond_vals, true_cond_vals):
                            s3_exact_matches += 1
                            if not true_is_normal:
                                total_e2e_perfect += 1

        total_time_sec = time.time() - t0
        fps = num_samples / max(1e-4, total_time_sec)
        
        # 3단계 오차율
        s3_error_rate = 1.0 - (s3_exact_matches / max(1, s3_evaluated_count)) if s3_evaluated_count > 0 else 0.0
        
        # 이론적 CETR: e1 + (1-e1)*e2 + (1-e1)*(1-e2)*s3
        theoretical_cetr = e1 + (1.0 - e1) * e2 + (1.0 - e1) * (1.0 - e2) * s3_error_rate
        # 실측 End-to-End 정확도
        e2e_exact_accuracy = total_e2e_perfect / max(1, num_samples)
        e2e_error_rate = 1.0 - e2e_exact_accuracy

        # 3단계 스킵 연산 절감률 추정
        flops_saved_ratio = early_exit_ratio * 50.0

        return {
            'split_name': split_key,
            'num_samples': num_samples,
            'stage1_error_rate': float(e1),
            'stage2_error_rate': float(e2),
            'stage3_error_rate': float(s3_error_rate),
            'theoretical_cetr': float(theoretical_cetr),
            'empirical_e2e_error': float(e2e_error_rate),
            'e2e_exact_accuracy': float(e2e_exact_accuracy),
            'early_exit_ratio': float(early_exit_ratio),
            'early_exit_precision': float(early_exit_precision),
            'flops_saved_percent': float(flops_saved_ratio),
            'total_time_sec': float(total_time_sec),
            'pipeline_fps': float(fps)
        }

def run_pipeline_benchmark():
    print("=" * 115)
    print("      [종단간(End-to-End) 3단계 계층형 파이프라인 통합 평가 & 캐스케이드 연쇄 오차 실측]")
    print("=" * 115)

    splits_s1 = Stage1DatasetLoader.load_all_dual_splits()
    splits_s2 = Stage2DatasetLoader.load_all_dual_splits()

    pipeline = EndToEndPipelineEvaluator()
    pipeline.fit_all_stages(splits_s1, splits_s2)

    eval_splits = [
        'Dual_Valid.npz',
        'Dual_Test_Strict.npz',
        'Dual_Test_Unseen_Subj.npz',
        'Dual_Test_Unseen_View.npz'
    ]

    all_results = {}
    print("\n" + "=" * 125)
    print(f"{'분할 데이터셋':<28} | {'1단계오차':^8} | {'2단계오차':^8} | {'3단계오차':^8} | {'CETR(누적오차)':^13} | {'E2E 정확도':^11} | {'조기차단정밀도':^12} | {'FPS':^7}")
    print("-" * 125)

    for s_file in eval_splits:
        res = pipeline.evaluate_end_to_end(s_file)
        all_results[s_file] = res
        
        s_name = s_file.replace('.npz', '')
        e1 = res['stage1_error_rate'] * 100
        e2 = res['stage2_error_rate'] * 100
        e3 = res['stage3_error_rate'] * 100
        cetr = res['theoretical_cetr'] * 100
        e2e_acc = res['e2e_exact_accuracy'] * 100
        ee_prec = res['early_exit_precision'] * 100
        fps = res['pipeline_fps']
        
        print(f"{s_name:<28} | {e1:>7.2f}% | {e2:>7.2f}% | {e3:>7.2f}% | {cetr:>12.2f}% | {e2e_acc:>10.2f}% | {ee_prec:>11.2f}% | {fps:>6.1f}")

    print("=" * 125)

    out_json_path = os.path.join(OUTPUT_DIR, 'pipeline_e2e_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 종단간 파이프라인 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_pipeline_benchmark()

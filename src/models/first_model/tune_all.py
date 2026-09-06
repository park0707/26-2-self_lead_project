# [실행 명령어]: python -m src.models.first_model.tune_all

import os
import sys
import time
import json
from typing import Dict, List, Any
import numpy as np

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper, TUNING_RESULTS_DIR

from src.models.first_model.rf_classifier import run_rf_grid_search
from src.models.first_model.extratrees_classifier import run_extratrees_grid_search
from src.models.first_model.xgb_classifier import run_xgb_grid_search
from src.models.first_model.lgbm_classifier import run_lgbm_grid_search
from src.models.first_model.catboost_classifier import run_catboost_grid_search
from src.models.first_model.mlp_classifier import run_mlp_grid_search
from src.models.first_model.gru_classifier import run_gru_grid_search
from src.models.first_model.tcn_classifier import run_tcn_grid_search

def run_all_stage1_tuning():
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 95)
    print("      [1단계 9종 운동 대분류: 전체 모델 하이퍼파라미터 그리드 탐색 및 챔피언 선발전]")
    print("=" * 95)

    print("[1/3] 2차원 복합 분할 데이터셋 (Dual Splits) 로드 중...")
    data = Stage1DatasetLoader.load_all_dual_splits()
    print(f"  - Train Set   : {data['train']['X_Format_A'].shape[0]:,}개 (피험자 1~6 x 시점 A, B, E)")
    print(f"  - Valid Set   : {data['valid']['X_Format_A'].shape[0]:,}개 (피험자 7~8 x 시점 D)")
    print(f"  - Strict Test : {data['test_strict']['X_Format_A'].shape[0]:,}개 (피험자 9 x 측면 시점 C - 복합 홀드아웃)")
    print(f"  - Subj Test   : {data['test_unseen_subj']['X_Format_A'].shape[0]:,}개 (피험자 9 x 시점 A, B, E - 인물 일반화)")
    print(f"  - View Test   : {data['test_unseen_view']['X_Format_A'].shape[0]:,}개 (피험자 1~6 x 측면 시점 C - 시점 일반화)")

    model_runners = [
        ("LightGBM", run_lgbm_grid_search, 'Format_A', ['num_leaves', 'max_depth', 'learning_rate']),
        ("XGBoost", run_xgb_grid_search, 'Format_A', ['max_depth', 'learning_rate', 'n_estimators']),
        ("CatBoost", run_catboost_grid_search, 'Format_A', ['depth', 'l2_leaf_reg', 'learning_rate']),
        ("Random Forest", run_rf_grid_search, 'Format_A', ['n_estimators', 'max_depth', 'min_samples_split']),
        ("ExtraTrees", run_extratrees_grid_search, 'Format_A', ['n_estimators', 'max_depth', 'min_samples_split']),
        ("MLP (Dense NN)", run_mlp_grid_search, 'Format_A', ['hidden_layer_sizes', 'learning_rate_init', 'alpha']),
        ("1D-CNN + Bi-GRU", run_gru_grid_search, 'Format_B', ['conv_filters', 'hidden_dim', 'num_layers', 'lr']),
        ("Temporal TCN", run_tcn_grid_search, 'Format_B', ['kernel_size', 'base_channels', 'dropout', 'lr'])
    ]

    print("\n[2/3] 8대 모델별 그리드 탐색 및 최적 파라미터 도출 중...")
    print("-" * 95)

    leaderboard = []

    for m_name, runner_fn, input_fmt, p_keys in model_runners:
        t0 = time.time()
        print(f"\n[*] >>> [{m_name}] 하이퍼파라미터 탐색 실행 중...")
        best_trial, other_5, best_clf = runner_fn(data)
        elapsed = time.time() - t0

        feat_key = 'X_Format_A' if input_fmt == 'Format_A' else 'X_Format_B'
        
        # 4대 분할 성능 측정
        res_v = best_clf.evaluate(data['valid'][feat_key], data['valid']['y_action'], split_name='valid')
        res_st = best_clf.evaluate(data['test_strict'][feat_key], data['test_strict']['y_action'], split_name='test_strict')
        res_sb = best_clf.evaluate(data['test_unseen_subj'][feat_key], data['test_unseen_subj']['y_action'], split_name='test_unseen_subj')
        res_vw = best_clf.evaluate(data['test_unseen_view'][feat_key], data['test_unseen_view']['y_action'], split_name='test_unseen_view')
        res_tr = best_clf.evaluate(data['train'][feat_key], data['train']['y_action'], split_name='train')

        adv_metrics = Stage1MetricsHelper.calculate_advanced_metrics(
            res_train=res_tr,
            res_valid=res_v,
            res_strict=res_st,
            res_subj=res_sb,
            res_view=res_vw
        )

        # 모델별 단독 결과 출력
        Stage1MetricsHelper.print_multi_split_summary(
            split_results={
                'Valid Set (피험자 7~8, 시점 D)': res_v,
                '★ Strict Test (피험자 9, 측면 C)': res_st,
                'Subj Test (피험자 9, 시점 A,B,E)': res_sb,
                'View Test (피험자 1~6, 측면 C)': res_vw
            },
            adv_metrics=adv_metrics
        )

        Stage1MetricsHelper.print_top_vs_random_comparison_table(
            best_trial=best_trial,
            other_trials=other_5,
            param_keys=p_keys
        )

        Stage1MetricsHelper.save_best_params_json(
            model_name=m_name,
            best_params=best_trial['params'],
            summary_metrics={
                'valid_macro_f1': res_v['macro_f1'],
                'strict_test_macro_f1': res_st['macro_f1'],
                'subj_test_macro_f1': res_sb['macro_f1'],
                'fps': res_st['fps'],
                'advanced_metrics': adv_metrics
            }
        )

        leaderboard.append({
            'model_name': m_name,
            'input_format': input_fmt,
            'best_params': best_trial['params'],
            'valid_macro_f1': float(res_v['macro_f1']),
            'strict_test_macro_f1': float(res_st['macro_f1']),
            'subj_test_macro_f1': float(res_sb['macro_f1']),
            'pdr_percent': float(adv_metrics['PDR_percent']),
            'vis_percent': float(adv_metrics['VIS_percent']),
            'fps': float(res_st['fps']),
            'elapsed_sec': float(elapsed)
        })

    # [3/3] 챔피언 선발 종합 순위표 출력
    print("\n" + "=" * 105)
    print("                 [1단계 9종 운동 대분류: 최종 하이퍼파라미터 최적화 챔피언 랭킹표]")
    print("=" * 105)
    header = f"{'순위':<4} | {'모델명':<18} | {'입력':^8} | {'Valid F1':^10} | {'Strict F1':^10} | {'Subj F1':^10} | {'PDR (시점저하)':^12} | {'FPS':^10}"
    print(header)
    print("-" * 105)

    # Strict Test F1 및 Subj Test F1 종합 고려하여 정렬
    leaderboard.sort(key=lambda x: (x['strict_test_macro_f1'] * 0.6 + x['subj_test_macro_f1'] * 0.4), reverse=True)

    for rank, entry in enumerate(leaderboard, 1):
        medal = "🥇 1위" if rank == 1 else ("🥈 2위" if rank == 2 else ("🥉 3위" if rank == 3 else f"  {rank}위"))
        m_name = entry['model_name']
        fmt = entry['input_format']
        v_f1 = entry['valid_macro_f1'] * 100.0
        st_f1 = entry['strict_test_macro_f1'] * 100.0
        sb_f1 = entry['subj_test_macro_f1'] * 100.0
        pdr = entry['pdr_percent']
        fps = entry['fps']

        print(f"{medal:<4} | {m_name:<18} | {fmt:^8} | {v_f1:>8.2f}% | {st_f1:>8.2f}% | {sb_f1:>8.2f}% | {pdr:>10.2f}% | {fps:>10.1f}")

    print("=" * 105)
    print(f"★ 1단계 종합 대상 챔피언: [{leaderboard[0]['model_name']}] (Strict F1: {leaderboard[0]['strict_test_macro_f1']*100:.2f}%, Subj F1: {leaderboard[0]['subj_test_macro_f1']*100:.2f}%)")
    print("=" * 105 + "\n")

    # 종합 리더보드 JSON 저장
    board_path = os.path.join(TUNING_RESULTS_DIR, "stage1_tuning_leaderboard.json")
    with open(board_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    print(f"[+] 1단계 종합 최적화 랭킹 JSON 저장 완료: {board_path}")

if __name__ == '__main__':
    run_all_stage1_tuning()

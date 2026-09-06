# [실행 명령어]: python -m src.models.first_model.evaluate_all

import os
import sys
import time
import json
from typing import Dict, List, Any
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.baseline_classifier import BaselineActionClassifier
from src.models.first_model.rf_classifier import RandomForestActionClassifier
from src.models.first_model.xgb_classifier import XGBoostActionClassifier
from src.models.first_model.lgbm_classifier import LightGBMActionClassifier
from src.models.first_model.mlp_classifier import MLPActionClassifier
from src.models.first_model.gru_classifier import TemporalGRUActionClassifier

OUTPUT_DIR = 'd:/Data/피처_데이터'

def run_all_stage1_benchmarks():
    print("=" * 95)
    print("           [1단계: 9종 운동 종목 분류 모델 다각도 종합 벤치마크 평가]")
    print("=" * 95)

    # 1. 데이터 로드
    print("[1/3] 2차원 복합 분할 (Subject x View Dual Split) 데이터셋 로드 중...")
    splits = Stage1DatasetLoader.load_all_dual_splits()
    
    X_train_a = splits['train']['X_Format_A']
    X_train_b = splits['train']['X_Format_B']
    y_train = splits['train']['y_action']
    print(f"  - Train Set : {X_train_a.shape[0]:>5,} 샘플 (피험자 1~6 x 시점 A, B, E)")
    print(f"  - Valid Set : {splits['valid']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 7~8 x 시점 D)")
    print(f"  - Strict Test: {splits['test_strict']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 9 x 시점 C - 복합 홀드아웃)")
    print(f"  - Subj Test : {splits['test_unseen_subj']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 9 x 시점 A, B, E)")
    print(f"  - View Test : {splits['test_unseen_view']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 1~6 x 시점 C)")

    # 2. 모델 후보군 정의 (입력 포맷 매핑)
    models = [
        ('Baseline (Decision Tree)', BaselineActionClassifier(model_type='decision_tree'), 'Format_A'),
        ('Baseline (Logistic Reg)',  BaselineActionClassifier(model_type='logistic_regression'), 'Format_A'),
        ('Random Forest (200 Trees)', RandomForestActionClassifier(n_estimators=200), 'Format_A'),
        ('XGBoost (Hist Gradient)',   XGBoostActionClassifier(n_estimators=200), 'Format_A'),
        ('LightGBM (Fast GBDT)',      LightGBMActionClassifier(n_estimators=200), 'Format_A'),
        ('MLP (Dense Neural Net)',    MLPActionClassifier(hidden_layer_sizes=(128, 64)), 'Format_A'),
        ('1D-CNN + Bi-GRU (Temporal)', TemporalGRUActionClassifier(epochs=15, batch_size=64), 'Format_B')
    ]

    # 3. 모델 학습 및 평가
    results = {}
    test_splits = ['valid', 'test_strict', 'test_unseen_subj', 'test_unseen_view']

    print("\n[2/3] 모델 학습 및 분할별 성능 측정 시작...")
    for model_name, clf, feat_fmt in models:
        print(f"\n[*] Training {model_name} (Input: {feat_fmt})...")
        t0 = time.time()
        
        X_tr = X_train_a if feat_fmt == 'Format_A' else X_train_b
        
        # Validation set for early stopping if supported
        if feat_fmt == 'Format_A' and hasattr(clf, 'fit') and 'eval_set' in clf.fit.__code__.co_varnames:
            eval_set = [(splits['valid']['X_Format_A'], splits['valid']['y_action'])]
            clf.fit(X_tr, y_train, eval_set=eval_set)
        else:
            clf.fit(X_tr, y_train)
            
        train_time = time.time() - t0
        print(f"    - Training time: {train_time:.2f}s")

        model_results = {'input_format': feat_fmt, 'train_time_sec': train_time, 'splits': {}}
        for s_key in test_splits:
            X_ev = splits[s_key]['X_Format_A'] if feat_fmt == 'Format_A' else splits[s_key]['X_Format_B']
            eval_res = clf.evaluate(
                X_ev,
                splits[s_key]['y_action'],
                split_name=s_key
            )
            model_results['splits'][s_key] = {
                'accuracy': round(float(eval_res['accuracy']), 4),
                'macro_f1': round(float(eval_res['macro_f1']), 4),
                'weighted_f1': round(float(eval_res['weighted_f1']), 4),
                'macro_precision': round(float(eval_res['macro_precision']), 4),
                'macro_recall': round(float(eval_res['macro_recall']), 4),
                'fps': round(float(eval_res['fps']), 1)
            }
            print(f"    [{s_key:18s}] Acc: {eval_res['accuracy']*100:6.2f}% | Macro-F1: {eval_res['macro_f1']*100:6.2f}% | Speed: {eval_res['fps']:6.1f} FPS")

        results[model_name] = model_results

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 105)
    print("                    [1단계 9종 운동 분류 모델 최종 벤치마크 종합 비교표]")
    print("=" * 105)
    header = f"{'모델명':<27} | {'입력':^8} | {'Valid F1':^10} | {'Strict Test F1':^14} | {'Subj Test F1':^12} | {'View Test F1':^12} | {'FPS':^8}"
    print(header)
    print("-" * 105)

    for m_name, m_res in results.items():
        fmt = m_res['input_format']
        v_f1 = m_res['splits']['valid']['macro_f1'] * 100
        st_f1 = m_res['splits']['test_strict']['macro_f1'] * 100
        sub_f1 = m_res['splits']['test_unseen_subj']['macro_f1'] * 100
        vw_f1 = m_res['splits']['test_unseen_view']['macro_f1'] * 100
        fps = m_res['splits']['test_strict']['fps']
        print(f"{m_name:<27} | {fmt:^8} | {v_f1:>9.2f}% | {st_f1:>13.2f}% | {sub_f1:>11.2f}% | {vw_f1:>11.2f}% | {fps:>7.1f}")
    print("=" * 105)

    # 5. 과적합 및 데이터 누수 진단 (라벨 셔플링 대조 실험 - LightGBM 기준)
    print("\n[★ 연구 계획서 검증] 라벨 셔플링(Label Shuffling) 널 가설 대조 실험 (데이터 누수 진단)")
    print("-" * 95)
    from src.models.first_model.metrics_helper import Stage1MetricsHelper
    null_test_res = Stage1MetricsHelper.evaluate_label_shuffling_test(
        LightGBMActionClassifier,
        X_train_a,
        y_train,
        splits['test_strict']['X_Format_A'],
        splits['test_strict']['y_action']
    )
    print(f"  - 무작위 추측 기댓값 (1/9) : {null_test_res['expected_random_accuracy']*100:5.2f}%")
    print(f"  - 라벨 셔플 학습 후 정확도 : {null_test_res['shuffled_accuracy']*100:5.2f}% (Macro-F1: {null_test_res['shuffled_macro_f1']*100:5.2f}%)")
    leak_msg = "경고: 데이터 누수 의심!" if null_test_res['leakage_detected'] else "통과: 우연 수준으로 성능 붕괴 확인 (데이터 누수 없음 정상)"
    print(f"  - 검증 판정 결과           : [ {leak_msg} ]")
    print("-" * 95)

    # 6. JSON 저장
    out_json_path = os.path.join(OUTPUT_DIR, 'first_model_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_all_stage1_benchmarks()

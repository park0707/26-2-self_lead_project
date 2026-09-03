# [실행 명령어]: python -m src.models.third_models.evaluate_all

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

from src.models.third_models.dataset import Stage3DatasetLoader
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.xgb_multilabel import MultiOutputXGBoostClassifier
from src.models.third_models.lgbm_multilabel import MultiOutputLightGBMClassifier
from src.models.third_models.gru_multilabel import TemporalGRUMultiLabelClassifier

OUTPUT_DIR = 'd:/Data/피처_데이터'

def run_all_stage3_benchmarks(target_exercise: str = '01_바벨_스쿼트'):
    print("=" * 105)
    print(f"       [3단계: {target_exercise} 세부 결함(Conditions) 다중 라벨 진단 벤치마크 평가]")
    print("=" * 105)

    # 1. 데이터 로드
    print(f"[1/3] {target_exercise} 다중 라벨 결함 데이터셋 로드 중...")
    splits = Stage3DatasetLoader.load_all_splits_for_exercise(target_exercise)
    
    cond_names = splits['train']['cond_names']
    num_conds = splits['train']['num_conditions']
    print(f"  - 감점 조건 수 : {num_conds}개 ({cond_names})")
    print(f"  - Train Set   : {splits['train']['num_samples']:>4} 샘플 (피험자 1~6 x 시점 A, B, E)")
    print(f"  - Valid Set   : {splits['valid']['num_samples']:>4} 샘플 (피험자 7~8 x 시점 D)")
    print(f"  - Strict Test : {splits['test_strict']['num_samples']:>4} 샘플 (피험자 9 x 시점 C - 복합 홀드아웃)")
    print(f"  - Subj Test   : {splits['test_unseen_subj']['num_samples']:>4} 샘플 (피험자 9 x 시점 A, B, E)")
    print(f"  - View Test   : {splits['test_unseen_view']['num_samples']:>4} 샘플 (피험자 1~6 x 시점 C)")

    X_train_a = splits['train']['X_Format_A']
    X_train_b = splits['train']['X_Format_B']
    Y_train = splits['train']['Y_multi']

    # 2. 모델 후보군 정의
    models = [
        ('Multi-Output Random Forest', MultiOutputRandomForestClassifier(n_estimators=150), 'Format_A'),
        ('Multi-Output XGBoost', MultiOutputXGBoostClassifier(n_estimators=150), 'Format_A'),
        ('Multi-Output LightGBM', MultiOutputLightGBMClassifier(n_estimators=150), 'Format_A'),
        ('1D-CNN + Bi-GRU (Multi-Head)', TemporalGRUMultiLabelClassifier(num_labels=num_conds, epochs=15), 'Format_B')
    ]

    # 3. 모델 학습 및 평가
    results = {}
    test_splits = ['valid', 'test_strict', 'test_unseen_subj', 'test_unseen_view']

    print(f"\n[2/3] 3단계 모델 학습 및 다각도 평가 시작...")
    for model_name, clf, feat_fmt in models:
        print(f"\n[*] Training {model_name} (Input: {feat_fmt})...")
        t0 = time.time()
        
        X_tr = X_train_a if feat_fmt == 'Format_A' else X_train_b
        clf.fit(X_tr, Y_train)
        train_time = time.time() - t0
        print(f"    - Training time: {train_time:.2f}s")

        model_results = {'input_format': feat_fmt, 'train_time_sec': train_time, 'splits': {}}
        for s_key in test_splits:
            X_ev = splits[s_key]['X_Format_A'] if feat_fmt == 'Format_A' else splits[s_key]['X_Format_B']
            Y_ev = splits[s_key]['Y_multi']
            
            eval_res = clf.evaluate(X_ev, Y_ev, split_name=s_key)
            model_results['splits'][s_key] = {
                'subset_accuracy': round(float(eval_res['subset_accuracy']), 4),
                'hamming_loss': round(float(eval_res['hamming_loss']), 4),
                'micro_f1': round(float(eval_res['micro_f1']), 4),
                'macro_f1': round(float(eval_res['macro_f1']), 4),
                'fps': round(float(eval_res['fps']), 1)
            }
            print(f"    [{s_key:18s}] Subset-Acc: {eval_res['subset_accuracy']*100:5.2f}% | Hamming-Loss: {eval_res['hamming_loss']:.4f} | Micro-F1: {eval_res['micro_f1']*100:5.2f}% | FPS: {eval_res['fps']:5.1f}")

        results[model_name] = model_results

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 115)
    print(f"              [3단계: {target_exercise} 세부 결함 진단 모델 최종 벤치마크 비교표]")
    print("=" * 115)
    header = f"{'모델명':<28} | {'입력':^8} | {'Valid Micro-F1':^14} | {'Strict Test F1':^14} | {'Subj Test F1':^12} | {'Hamming Loss':^12} | {'FPS':^8}"
    print(header)
    print("-" * 115)

    for m_name, m_res in results.items():
        fmt = m_res['input_format']
        v_f1 = m_res['splits']['valid']['micro_f1'] * 100
        st_f1 = m_res['splits']['test_strict']['micro_f1'] * 100
        sub_f1 = m_res['splits']['test_unseen_subj']['micro_f1'] * 100
        st_hloss = m_res['splits']['test_strict']['hamming_loss']
        fps = m_res['splits']['test_strict']['fps']
        print(f"{m_name:<28} | {fmt:^8} | {v_f1:>13.2f}% | {st_f1:>13.2f}% | {sub_f1:>11.2f}% | {st_hloss:>12.4f} | {fps:>7.1f}")
    print("=" * 115)

    # 5. JSON 저장
    out_json_path = os.path.join(OUTPUT_DIR, 'third_model_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'exercise': target_exercise,
            'conditions': cond_names,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 3단계 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_all_stage3_benchmarks('01_바벨_스쿼트')

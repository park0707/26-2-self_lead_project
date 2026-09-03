# [실행 명령어]: python -m src.models.second_models.evaluate_all

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

from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier

OUTPUT_DIR = 'd:/Data/피처_데이터'

def run_all_stage2_benchmarks():
    print("=" * 105)
    print("           [2단계: 동작 정상/오류 판별 전문가 모델 다각도 종합 벤치마크 평가]")
    print("=" * 105)

    # 1. 데이터 로드
    print("[1/3] 2차원 복합 분할 정상(1) vs 오류(0) 데이터셋 로드 중...")
    splits = Stage2DatasetLoader.load_all_dual_splits()
    
    X_train_a = splits['train']['X_Format_A']
    X_train_b = splits['train']['X_Format_B']
    y_train = splits['train']['y']
    spw = splits['train']['scale_pos_weight']
    
    print(f"  - Train Set : {X_train_a.shape[0]:>5,} 샘플 (정상: {splits['train']['normal_count']:,} / 오류: {splits['train']['error_count']:,}, 불균형비율: 1:{spw:.1f})")
    print(f"  - Valid Set : {splits['valid']['X_Format_A'].shape[0]:>5,} 샘플 (정상: {splits['valid']['normal_count']} / 오류: {splits['valid']['error_count']})")
    print(f"  - Strict Test: {splits['test_strict']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 9 x 시점 C - 복합 홀드아웃)")
    print(f"  - Subj Test : {splits['test_unseen_subj']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 9 x 시점 A, B, E)")
    print(f"  - View Test : {splits['test_unseen_view']['X_Format_A'].shape[0]:>5,} 샘플 (피험자 1~6 x 시점 C)")

    # 2. 모델 후보군 정의
    models = [
        ('Balanced Random Forest', BalancedRandomForestBinaryClassifier(n_estimators=200), 'Format_A'),
        ('Cost-Sensitive XGBoost', CostSensitiveXGBoostBinaryClassifier(scale_pos_weight=spw, n_estimators=200), 'Format_A'),
        ('Cost-Sensitive LightGBM', CostSensitiveLightGBMBinaryClassifier(scale_pos_weight=spw, n_estimators=200), 'Format_A'),
        ('1D-CNN + Bi-GRU (Focal)', TemporalGRUBinaryClassifier(epochs=15, batch_size=64), 'Format_B')
    ]

    # 3. 모델 학습 및 평가
    results = {}
    test_splits = ['valid', 'test_strict', 'test_unseen_subj', 'test_unseen_view']

    print("\n[2/3] 2단계 모델 학습 및 분할별 평가 시작...")
    for model_name, clf, feat_fmt in models:
        print(f"\n[*] Training {model_name} (Input: {feat_fmt})...")
        t0 = time.time()
        
        X_tr = X_train_a if feat_fmt == 'Format_A' else X_train_b
        
        if feat_fmt == 'Format_A' and hasattr(clf, 'fit') and 'eval_set' in clf.fit.__code__.co_varnames:
            eval_set = [(splits['valid']['X_Format_A'], splits['valid']['y'])]
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
                splits[s_key]['y'],
                split_name=s_key
            )
            model_results['splits'][s_key] = {
                'accuracy': round(float(eval_res['accuracy']), 4),
                'macro_f1': round(float(eval_res['macro_f1']), 4),
                'pos_f1': round(float(eval_res['pos_f1']), 4),
                'pos_precision': round(float(eval_res['pos_precision']), 4),
                'pos_recall': round(float(eval_res['pos_recall']), 4),
                'pr_auc': round(float(eval_res['pr_auc']), 4),
                'roc_auc': round(float(eval_res['roc_auc']), 4),
                'fps': round(float(eval_res['fps']), 1)
            }
            print(f"    [{s_key:18s}] Acc: {eval_res['accuracy']*100:5.2f}% | Macro-F1: {eval_res['macro_f1']*100:5.2f}% | Pos-F1: {eval_res['pos_f1']*100:5.2f}% | PR-AUC: {eval_res['pr_auc']:.3f} | FPS: {eval_res['fps']:5.1f}")

        results[model_name] = model_results

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 115)
    print("                    [2단계 정상/오류 판별 모델 최종 벤치마크 종합 비교표]")
    print("=" * 115)
    header = f"{'모델명':<27} | {'입력':^8} | {'Valid Macro-F1':^14} | {'Strict Test F1':^14} | {'Subj Test F1':^12} | {'Strict PR-AUC':^13} | {'FPS':^8}"
    print(header)
    print("-" * 115)

    for m_name, m_res in results.items():
        fmt = m_res['input_format']
        v_f1 = m_res['splits']['valid']['macro_f1'] * 100
        st_f1 = m_res['splits']['test_strict']['macro_f1'] * 100
        sub_f1 = m_res['splits']['test_unseen_subj']['macro_f1'] * 100
        st_prauc = m_res['splits']['test_strict']['pr_auc']
        fps = m_res['splits']['test_strict']['fps']
        print(f"{m_name:<27} | {fmt:^8} | {v_f1:>13.2f}% | {st_f1:>13.2f}% | {sub_f1:>11.2f}% | {st_prauc:>12.3f} | {fps:>7.1f}")
    print("=" * 115)

    # 5. JSON 저장
    out_json_path = os.path.join(OUTPUT_DIR, 'second_model_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 2단계 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_all_stage2_benchmarks()

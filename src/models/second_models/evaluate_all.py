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
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier

OUTPUT_DIR = 'd:/Data/피처_데이터'

def run_all_stage2_benchmarks():
    print("=" * 115)
    print("      [2단계: 동작 정상/오류 판별 전문가 모델 초정밀 다차원 벤치마크 평가 (1:17.0 불균형 대응)]")
    print("=" * 115)

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

    print("\n[2/3] 2단계 모델 학습 및 최적 임계값 도출 & 다각도 평가 시작...")
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

        # Step A: Validation set에서 최적 임계값 도출 (F1 최대화 & F2 안전 최대화)
        X_val = splits['valid']['X_Format_A'] if feat_fmt == 'Format_A' else splits['valid']['X_Format_B']
        y_val = splits['valid']['y']
        
        val_probs = clf.predict_proba(X_val)
        val_pos_probs = val_probs[:, 1] if val_probs.ndim > 1 else val_probs
        
        opt_thresh_f1, best_val_f1 = Stage2MetricsHelper.find_optimal_threshold(y_val, val_pos_probs, beta=1.0)
        opt_thresh_f2, best_val_f2 = Stage2MetricsHelper.find_optimal_threshold(y_val, val_pos_probs, beta=2.0)
        print(f"    [Valid 최적 임계값] Default=0.500 | F1 최적값(θ*)={opt_thresh_f1:.3f} (Val F1: {best_val_f1*100:.1f}%) | F2 안전최적값={opt_thresh_f2:.3f}")

        model_results = {
            'input_format': feat_fmt,
            'train_time_sec': train_time,
            'optimal_threshold_f1': opt_thresh_f1,
            'optimal_threshold_f2': opt_thresh_f2,
            'splits': {}
        }

        for s_key in test_splits:
            X_ev = splits[s_key]['X_Format_A'] if feat_fmt == 'Format_A' else splits[s_key]['X_Format_B']
            y_ev = splits[s_key]['y']
            
            t_eval_0 = time.time()
            probs_ev = clf.predict_proba(X_ev)
            pos_probs_ev = probs_ev[:, 1] if probs_ev.ndim > 1 else probs_ev
            inf_time_ms = ((time.time() - t_eval_0) / max(1, len(X_ev))) * 1000.0

            # 1) 기본 0.5 임계값 평가
            res_def = Stage2MetricsHelper.evaluate_predictions(y_ev, pos_probs_ev, threshold=0.5, inference_time_ms=inf_time_ms)
            # 2) Valid 최적 임계값(θ*) 적용 평가
            res_opt = Stage2MetricsHelper.evaluate_predictions(y_ev, pos_probs_ev, threshold=opt_thresh_f1, inference_time_ms=inf_time_ms)

            model_results['splits'][s_key] = {
                'default_thresh_0_5': res_def,
                'optimal_thresh': res_opt
            }
            
            print(f"    [{s_key:18s}] [기본 θ=0.5] Macro-F1: {res_def['macro_f1']*100:5.2f}% | Pos-F1: {res_def['pos_f1']*100:5.2f}% | PR-AUC: {res_def['pr_auc']:.3f} | MCC: {res_def['mcc']:.3f} | G-Mean: {res_def['g_mean']:.3f}")
            print(f"    {' '*20} [최적 θ*={opt_thresh_f1:.3f}] Macro-F1: {res_opt['macro_f1']*100:5.2f}% | Pos-F1: {res_opt['pos_f1']*100:5.2f}% | BAcc: {res_opt['balanced_accuracy']*100:5.2f}% | Safety-F2: {res_opt['pos_f2_safety']*100:5.2f}%")

        results[model_name] = model_results

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 125)
    print("                    [2단계 정상/오류 판별 모델 최종 벤치마크 고도화 종합 비교표]")
    print("=" * 125)
    header = f"{'모델명':<27} | {'입력':^8} | {'Strict PR-AUC':^13} | {'Strict MCC':^10} | {'Strict G-Mean':^13} | {'Strict F1(θ*)':^13} | {'Safety F2(θ*)':^13} | {'FPS':^8}"
    print(header)
    print("-" * 125)

    for m_name, m_res in results.items():
        fmt = m_res['input_format']
        st_res_opt = m_res['splits']['test_strict']['optimal_thresh']
        st_prauc = st_res_opt['pr_auc']
        st_mcc = st_res_opt['mcc']
        st_gmean = st_res_opt['g_mean']
        st_f1 = st_res_opt['macro_f1'] * 100
        st_f2 = st_res_opt['pos_f2_safety'] * 100
        fps = st_res_opt['fps']
        print(f"{m_name:<27} | {fmt:^8} | {st_prauc:>12.3f} | {st_mcc:>9.3f} | {st_gmean:>12.3f} | {st_f1:>12.2f}% | {st_f2:>12.2f}% | {fps:>7.1f}")
    print("=" * 125)

    # 5. JSON 저장
    out_json_path = os.path.join(OUTPUT_DIR, 'second_model_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 2단계 고도화 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_all_stage2_benchmarks()

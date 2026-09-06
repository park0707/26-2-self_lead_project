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
from src.models.third_models.metrics_helper import Stage3MetricsHelper
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.xgb_multilabel import MultiOutputXGBoostClassifier
from src.models.third_models.lgbm_multilabel import MultiOutputLightGBMClassifier
from src.models.third_models.gru_multilabel import TemporalGRUMultiLabelClassifier

OUTPUT_DIR = 'd:/Data/피처_데이터'

def run_all_stage3_benchmarks(target_exercise: str = '01_바벨_스쿼트'):
    print("=" * 115)
    print(f"       [3단계: {target_exercise} 세부 결함(Conditions) 다중 라벨 진단 초정밀 벤치마크 평가]")
    print("=" * 115)

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

    print(f"\n[2/3] 3단계 모델 학습 및 다각도 다중 라벨 정밀 평가 시작...")
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
            
            t_eval_0 = time.time()
            preds_ev = clf.predict(X_ev)
            inf_time_ms = ((time.time() - t_eval_0) / max(1, len(X_ev))) * 1000.0
            
            # Predict proba if available
            proba_ev = None
            if hasattr(clf, 'predict_proba'):
                try:
                    raw_p = clf.predict_proba(X_ev)
                    if isinstance(raw_p, list): # MultiOutputClassifier produces list of (N, 2)
                        proba_ev = np.column_stack([p[:, 1] if p.ndim > 1 else p for p in raw_p])
                    elif isinstance(raw_p, np.ndarray):
                        proba_ev = raw_p
                except Exception:
                    proba_ev = None

            eval_res = Stage3MetricsHelper.evaluate_predictions(
                Y_ev,
                preds_ev,
                Y_proba=proba_ev,
                cond_names=cond_names,
                inference_time_ms=inf_time_ms
            )
            model_results['splits'][s_key] = eval_res
            
            print(f"    [{s_key:18s}] ExactMatch: {eval_res['subset_accuracy']*100:5.2f}% | Hamming-Loss: {eval_res['hamming_loss']:.4f} | Jaccard-IoU: {eval_res['jaccard_samples_iou']*100:5.2f}% | Micro-F1: {eval_res['micro_f1']*100:5.2f}% | FPS: {eval_res['fps']:5.1f}")

        results[model_name] = model_results

    # 4. 결과 요약 테이블 출력
    print("\n" + "=" * 125)
    print(f"              [3단계: {target_exercise} 세부 결함 진단 모델 최종 벤치마크 고도화 종합 비교표]")
    print("=" * 125)
    header = f"{'모델명':<28} | {'입력':^8} | {'Strict Exact':^13} | {'Strict H-Loss':^14} | {'Strict Jaccard':^14} | {'Strict Micro-F1':^15} | {'FPS':^8}"
    print(header)
    print("-" * 125)

    for m_name, m_res in results.items():
        fmt = m_res['input_format']
        st_res = m_res['splits']['test_strict']
        st_exact = st_res['subset_accuracy'] * 100
        st_hloss = st_res['hamming_loss']
        st_iou = st_res['jaccard_samples_iou'] * 100
        st_f1 = st_res['micro_f1'] * 100
        fps = st_res['fps']
        print(f"{m_name:<28} | {fmt:^8} | {st_exact:>12.2f}% | {st_hloss:>13.4f} | {st_iou:>13.2f}% | {st_f1:>14.2f}% | {fps:>7.1f}")
    print("=" * 125)

    # 5. 베스트 모델의 결함별 세부 성적표 출력
    best_model_name = list(results.keys())[0]
    best_st = results[best_model_name]['splits']['test_strict']
    print(f"\n[★ 베스트 모델 ({best_model_name}) 결함 조건별 세부 성적표 - Strict Test]")
    print("-" * 105)
    print(f"{'결함 항목명':<45} | {'발생수':^6} | {'비율':^7} | {'정밀도':^9} | {'재현율':^9} | {'F1-Score':^9}")
    print("-" * 105)
    for c_name, c_m in best_st['per_condition_metrics'].items():
        cnt = c_m['support_defects']
        ratio = c_m['defect_ratio'] * 100
        prec = c_m['precision'] * 100
        rec = c_m['recall'] * 100
        f1 = c_m['f1'] * 100
        print(f"{c_name:<45} | {cnt:>6d} | {ratio:>6.1f}% | {prec:>8.2f}% | {rec:>8.2f}% | {f1:>8.2f}%")
    print("-" * 105)

    # 6. JSON 저장
    out_json_path = os.path.join(OUTPUT_DIR, 'third_model_benchmark_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'exercise': target_exercise,
            'conditions': cond_names,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 3단계 고도화 벤치마크 결과 JSON 저장 완료: {out_json_path}")

if __name__ == '__main__':
    run_all_stage3_benchmarks('01_바벨_스쿼트')

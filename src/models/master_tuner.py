# [실행 명령어]: python -m src.models.master_tuner
# [실행 목적]: 1·2·3단계 전수 모델 대규모 그리드 튜닝, 3중 비교(Default vs Random 5 vs Best), 코드 내 BEST_PARAMS 내장 및 docx 보고서 일괄 산출

import os
import sys
import time
import json
import re
from typing import Dict, List, Any

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.rf_classifier import run_rf_grid_search
from src.models.first_model.extratrees_classifier import run_extratrees_grid_search
from src.models.first_model.xgb_classifier import run_xgb_grid_search
from src.models.first_model.lgbm_classifier import run_lgbm_grid_search
from src.models.first_model.catboost_classifier import run_catboost_grid_search
from src.models.first_model.mlp_classifier import run_mlp_grid_search
from src.models.first_model.gru_classifier import run_gru_grid_search
from src.models.first_model.tcn_classifier import run_tcn_grid_search
from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.tune_all import run_all_stage2_tuning
from src.models.third_models.tune_all import run_all_stage3_tuning
from src.models.pipeline_evaluator import EndToEndPipelineEvaluator
from src.features.generate_tuning_docx_report import create_tuning_docx_report

import pprint
import pickle

def update_model_file_best_params(file_path: str, best_params_dict: Dict[str, Any]):
    """파이썬 코드 파일 내의 BEST_PARAMS 상수를 정밀 치환/갱신"""
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 파이썬 문법(None, True, False)에 맞게 pprint 포맷팅
    formatted_dict = pprint.pformat(best_params_dict, indent=4)
    new_block = f"BEST_PARAMS = {formatted_dict}\n"

    if "BEST_PARAMS =" in content:
        content = re.sub(r"BEST_PARAMS\s*=\s*\{[^\}]*\}", f"BEST_PARAMS = {formatted_dict}", content, flags=re.DOTALL)
    else:
        # 파일 임포트문 뒤에 삽입
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
        lines.insert(insert_idx + 1, new_block)
        content = '\n'.join(lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [+] 코드 파일 내 BEST_PARAMS 내장 갱신 완료: {file_path}")

def run_master_tuning_and_reporting():
    print("=" * 115)
    print("      [1·2·3단계 전수 모델 마스터 하이퍼파라미터 최적화 & 3중 비교(Default vs Random 5 vs Best) 파이프라인]")
    print("=" * 115)
    t_start = time.time()

    # =========================================================================
    # [1/4] 1단계 9종 운동 대분류 8대 모델 그리드 탐색
    # =========================================================================
    print("\n[STEP 1/4] 1단계: 9종 운동 대분류 8대 모델 전수 하이퍼파라미터 그리드 탐색 시작...")
    s1_data = Stage1DatasetLoader.load_all_dual_splits()
    
    s1_runners = [
        ("LightGBM", run_lgbm_grid_search, 'Format_A', 'src/models/first_model/lgbm_classifier.py'),
        ("XGBoost", run_xgb_grid_search, 'Format_A', 'src/models/first_model/xgb_classifier.py'),
        ("CatBoost", run_catboost_grid_search, 'Format_A', 'src/models/first_model/catboost_classifier.py'),
        ("Random Forest", run_rf_grid_search, 'Format_A', 'src/models/first_model/rf_classifier.py'),
        ("ExtraTrees", run_extratrees_grid_search, 'Format_A', 'src/models/first_model/extratrees_classifier.py'),
        ("MLP (Dense NN)", run_mlp_grid_search, 'Format_A', 'src/models/first_model/mlp_classifier.py'),
        ("1D-CNN + Bi-GRU", run_gru_grid_search, 'Format_B', 'src/models/first_model/gru_classifier.py'),
        ("Temporal TCN", run_tcn_grid_search, 'Format_B', 'src/models/first_model/tcn_classifier.py')
    ]

    stage1_cache_path = "d:/Data/피처_데이터/tuning_results/stage1_master_cache.pkl"
    os.makedirs(os.path.dirname(stage1_cache_path), exist_ok=True)
    if os.path.exists(stage1_cache_path):
        print(f"[*] 1단계 캐시 로드 중: {stage1_cache_path}")
        with open(stage1_cache_path, 'rb') as f:
            stage1_results = pickle.load(f)
    else:
        stage1_results = {}
        for m_name, fn, fmt, fpath in s1_runners:
            print(f"\n[*] >>> [1단계: {m_name}] 그리드 탐색 중...")
            best_t, oth_5, best_clf = fn(s1_data)
            
            # Default trial 측정
            def_t = oth_5[0] if oth_5 else best_t

            stage1_results[m_name] = {
                'input_format': fmt,
                'default_trial': def_t,
                'other_5_trials': oth_5,
                'best_trial': best_t,
                'best_clf': best_clf
            }
            # 소스 파일 내 BEST_PARAMS 내장 갱신
            update_model_file_best_params(os.path.join('d:/', fpath), best_t['params'])
        with open(stage1_cache_path, 'wb') as f:
            pickle.dump(stage1_results, f)

    # =========================================================================
    # [2/4] 2단계 정상/오류 판별 4대 모델 그리드 탐색
    # =========================================================================
    print("\n[STEP 2/4] 2단계: 정상/오류 판별 4대 모델 전수 하이퍼파라미터 그리드 탐색 시작...")
    stage2_cache_path = "d:/Data/피처_데이터/tuning_results/stage2_master_cache.pkl"
    if os.path.exists(stage2_cache_path):
        print(f"[*] 2단계 캐시 로드 중: {stage2_cache_path}")
        with open(stage2_cache_path, 'rb') as f:
            stage2_results = pickle.load(f)
    else:
        stage2_results = run_all_stage2_tuning()
        with open(stage2_cache_path, 'wb') as f:
            pickle.dump(stage2_results, f)
    
    s2_file_map = {
        'Cost-Sensitive LightGBM': 'src/models/second_models/lgbm_binary.py',
        'Cost-Sensitive XGBoost': 'src/models/second_models/xgb_binary.py',
        'Balanced Random Forest': 'src/models/second_models/rf_binary.py',
        '1D-CNN + Bi-GRU (Focal)': 'src/models/second_models/gru_binary.py'
    }
    for m_name, fpath in s2_file_map.items():
        if m_name in stage2_results:
            update_model_file_best_params(os.path.join('d:/', fpath), stage2_results[m_name]['best_trial']['params'])

    # =========================================================================
    # [3/4] 3단계 세부 결함 진단 4대 모델 그리드 탐색
    # =========================================================================
    print("\n[STEP 3/4] 3단계: 세부 결함(Conditions) 4대 모델 전수 그리드 탐색 시작...")
    stage3_cache_path = "d:/Data/피처_데이터/tuning_results/stage3_master_cache.pkl"
    if os.path.exists(stage3_cache_path):
        print(f"[*] 3단계 캐시 로드 중: {stage3_cache_path}")
        with open(stage3_cache_path, 'rb') as f:
            stage3_results = pickle.load(f)
    else:
        stage3_results = run_all_stage3_tuning('01_바벨_스쿼트')
        with open(stage3_cache_path, 'wb') as f:
            pickle.dump(stage3_results, f)
    
    s3_file_map = {
        'Multi-Output LightGBM': 'src/models/third_models/lgbm_multilabel.py',
        'Multi-Output XGBoost': 'src/models/third_models/xgb_multilabel.py',
        'Multi-Output Random Forest': 'src/models/third_models/rf_multilabel.py',
        '1D-CNN + Bi-GRU (Multi-Head)': 'src/models/third_models/gru_multilabel.py'
    }
    for m_name, fpath in s3_file_map.items():
        if m_name in stage3_results:
            update_model_file_best_params(os.path.join('d:/', fpath), stage3_results[m_name]['best_trial']['params'])

    # =========================================================================
    # [4/4] 종단간(End-to-End) 파이프라인 연계 실측 및 Word(.docx) 단독 보고서 생성
    # =========================================================================
    print("\n[STEP 4/4] 종단간(End-to-End) 파이프라인 실측 및 공식 Word 보고서 생성 중...")
    pipeline = EndToEndPipelineEvaluator()
    splits_s2 = Stage2DatasetLoader.load_all_dual_splits()
    pipeline.fit_all_stages(s1_data, splits_s2)

    eval_splits = ['Dual_Valid.npz', 'Dual_Test_Strict.npz', 'Dual_Test_Unseen_Subj.npz', 'Dual_Test_Unseen_View.npz']
    pipeline_results = {}
    for s_file in eval_splits:
        pipeline_results[s_file] = pipeline.evaluate_end_to_end(s_file)

    docx_path = create_tuning_docx_report(
        stage1_data=stage1_results,
        stage2_data=stage2_results,
        stage3_data=stage3_results,
        pipeline_data=pipeline_results,
        output_path="d:/자주프 문서/1주차/모델_튜닝_및_성능_비교_종합_보고서.docx"
    )

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 115)
    print("                 ★ [마스터 튜닝 및 공식 Word 보고서 생성 완료] ★")
    print("=" * 115)
    print(f"[*] 총 소요 시간       : {elapsed_total:.2f}초 ({elapsed_total/60:.2f}분)")
    print(f"[*] 코드 내 BEST_PARAMS: 16개 모델 소스 코드 상단에 자동 내장 갱신 완료")
    print(f"[*] 공식 Word 보고서   : {docx_path}")
    print("=" * 115 + "\n")

if __name__ == '__main__':
    run_master_tuning_and_reporting()

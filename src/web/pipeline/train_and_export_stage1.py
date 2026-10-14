# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Script: src/web/pipeline/train_and_export_stage1.py
Purpose: AI-Hub 8종 운동 전수 데이터(Master Pool 100% Full-Fit)를 학습하여
         웹 서비스 배포용 1단계 운동 종목 분류 단일 프로덕션 번들(.pkl) 생성
Target Output:
  - D:\src\web\weights\stage1_v6_production_bundle.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch

# 한글 콘솔 인코딩 보정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.models.first_model.train_v6_hierarchical_model import V6HierarchicalPipelineTrainer
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier
from src.models.first_model.dataset import EXERCISE_NAMES_KO
from src.models.first_model.metrics_helper import EXERCISE_NAMES_SHORT
from src.models.first_model.train_hierarchical_kfold import (
    CLASS_TO_MACRO_GROUP,
    MACRO_GROUP_NAMES
)

OUTPUT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "src", "web", "weights")
OUTPUT_BUNDLE_PATH = os.path.join(OUTPUT_WEIGHTS_DIR, "stage1_v6_production_bundle.pkl")


def train_and_export_production_bundle():
    print("=" * 105)
    print("  ★ [vidio-health-trainer] 1단계 운동 분류 100% Full-Fit 프로덕션 번들 학습 및 익스포트")
    print("=" * 105)

    os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)
    t0_global = time.time()

    # 1. 마스터 풀 전수 데이터 로드 및 59D Model F Direct KCF 피처 추출
    print("\n[*] 1/4. 마스터 풀 전수 데이터셋 로드 및 59D Direct KCF 피처 추출 중...")
    trainer = V6HierarchicalPipelineTrainer()
    master = trainer.master_data

    X_kcf = master['X_Format_A']          # (N, 59)
    X_raw_b = master['X_Format_B']        # (N, 64, 45)
    X_lunge_b = master['X_Format_B_lunge']
    y_act = master['y_action']            # 8개 클래스 (0~7)
    y_mac = master['y_macro']             # 6개 대그룹 (0~5)
    subjs = master['subjects']
    sids = [str(s) for s in master['sequence_ids']]
    num_macro = master['num_classes_macro']

    N = len(y_act)
    print(f"    ✔ 전체 데이터 크기: {N:,}개 시퀀스 (AI-Hub 피험자 1~9 전수 & 5대 시점 전수)")
    print(f"    ✔ 6대 상위 그룹 분포: {dict(zip(*np.unique(y_mac, return_counts=True)))}")

    # 클래스 가중치 계산 (불균형 완화)
    counts_mac = np.bincount(y_mac, minlength=num_macro)
    raw_w = N / (float(num_macro) * np.maximum(counts_mac, 1).astype(np.float32))
    smooth_w = np.sqrt(raw_w)
    sample_w = smooth_w[y_mac]

    # 2. Stage 1: 6-Group 마스터 3중 앙상블 학습
    print("\n[*] 2/4. Stage 1 (6대 그룹 3중 앙상블: Bi-GRU + LightGBM + XGBoost) 100% 학습 중...")
    
    # (A) Bi-GRU 학습
    print("    -> [Bi-GRU] 12 Epochs 학습 중...", flush=True)
    t_gru = time.time()
    gru_macro = AdvancedGRUClassifier(
        input_dim=X_raw_b.shape[2],
        num_classes=num_macro,
        epochs=12,
        lr=0.003,
        batch_size=512
    )
    gru_macro.fit(X_raw_b, y_mac, class_weights=smooth_w)
    print(f"       ✔ Bi-GRU 학습 완료 ({time.time()-t_gru:.1f}초)")

    # (B) LightGBM 학습
    print("    -> [LightGBM] 100 Trees 학습 중...", flush=True)
    t_lgb = time.time()
    lgb_macro = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, n_jobs=4, verbose=-1
    )
    lgb_macro.fit(X_kcf, y_mac, sample_weight=sample_w)
    print(f"       ✔ LightGBM 학습 완료 ({time.time()-t_lgb:.1f}초)")

    # (C) XGBoost 학습
    print("    -> [XGBoost] 100 Trees 학습 중...", flush=True)
    t_xgb = time.time()
    xgb_macro = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        eval_metric='mlogloss', n_jobs=4
    )
    xgb_macro.fit(X_kcf, y_mac, sample_weight=sample_w)
    print(f"       ✔ XGBoost 학습 완료 ({time.time()-t_xgb:.1f}초)")

    # 3. Stage 2-A & 2-B 세부 Expert 모델 100% 학습
    print("\n[*] 3/4. Stage 2-A (데드리프트 캐스케이드) & Stage 2-B (무빙 런지 Expert) 학습 중...")

    # Stage 2-A: 데드리프트 2중 캐스케이드 (일반 vs 스티프)
    dead_mask = (y_act == 1) | (y_act == 2)
    X_dead_raw = X_raw_b[dead_mask]
    y_dead_bin = (y_act[dead_mask] == 2).astype(int)
    dead_subjs = subjs[dead_mask]
    dead_sids = [sids[i] for i in range(len(sids)) if dead_mask[i]]

    sub_dead_clf = DualCascadeDeadliftClassifier()
    sub_dead_clf.fit(X_dead_raw, y_dead_bin, subjects=dead_subjs, sequence_ids=dead_sids)
    print(f"    ✔ Stage 2-A 데드리프트 캐스케이드 학습 완료 (샘플 수: {len(y_dead_bin):,})")

    # Stage 2-B: 무빙 런지 2진 Expert (다이나믹 vs 크로스)
    lunge_mask = (y_act == 4) | (y_act == 6)
    X_lunge_sub = LungeSubClassifier.slice_features(X_lunge_b[lunge_mask])
    y_lunge_bin = (y_act[lunge_mask] == 6).astype(int)

    sub_lunge_clf = LungeSubClassifier()
    lunge_tune_res = sub_lunge_clf.tune_and_fit(X_lunge_sub, y_lunge_bin)
    print(f"    ✔ Stage 2-B 무빙 런지 Expert 학습 완료 (샘플 수: {len(y_lunge_bin):,}, Best τ*={lunge_tune_res['best_threshold']:.2f})")

    # 4. 프로덕션 번들 패키징 및 직렬화 저장
    print("\n[*] 4/4. 웹 서비스 배포용 통합 번들 직렬화 저장 중...")
    production_bundle = {
        'version': 'v6.0-production-fullfit',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Stage 1 8-Exercise Classification Production Bundle (AI-Hub 100% Full-Fit)',
        'gru_model_state_dict': gru_macro.model.state_dict(),
        'gru_config': {
            'input_dim': X_raw_b.shape[2],
            'hidden_dim': 64,
            'num_layers': 2,
            'num_classes': num_macro
        },
        'lgb_macro': lgb_macro,
        'xgb_macro': xgb_macro,
        'macro_ensemble_weights': np.array([0.50, 0.30, 0.20], dtype=np.float32),
        'sub_dead_clf': sub_dead_clf,
        'sub_lunge_clf': sub_lunge_clf,
        'lunge_best_threshold': float(lunge_tune_res['best_threshold']),
        'dead_decision_threshold': float(sub_dead_clf.stage2_decision_threshold),
        'exercise_names_ko': EXERCISE_NAMES_KO,
        'exercise_names_short': EXERCISE_NAMES_SHORT,
        'class_to_macro_group': CLASS_TO_MACRO_GROUP,
        'macro_group_names': MACRO_GROUP_NAMES,
        'total_training_samples': N
    }

    with open(OUTPUT_BUNDLE_PATH, 'wb') as f:
        pickle.dump(production_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(OUTPUT_BUNDLE_PATH) / (1024 * 1024)
    total_elapsed = time.time() - t0_global

    print("=" * 105)
    print(f"  ★ [성공] 1단계 배포용 프로덕션 번들 생성 완료!")
    print(f"  - 저장 경로 : {OUTPUT_BUNDLE_PATH}")
    print(f"  - 파일 크기 : {file_size_mb:.2f} MB")
    print(f"  - 총 소요시간: {total_elapsed:.1f}초")
    print("=" * 105)

    return OUTPUT_BUNDLE_PATH


if __name__ == '__main__':
    train_and_export_production_bundle()

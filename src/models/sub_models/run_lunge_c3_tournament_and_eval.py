# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: run_lunge_c3_tournament_and_eval.py
Purpose: 바벨 런지 c3(뒤다리 무릎 각도 90도) 5대 모델 토너먼트 실측 및
         전체 1,085개 표본 대상 3-State F1-Score 정밀 평가
         - 피험자 1~9 전수 5-Fold StratifiedGroupKFold 교차검증
         - 공식 5대 시점(시점 A~E) 전수 성능 감사
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report
)

# 5대 모델 임포트
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features import extract_lunge_c3_14d_features, FEATURE_NAMES_14D

DATA_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c3_dataset_14d.npz"
RESULTS_JSON_PATH = r"D:\Data\피처_데이터\tuning_results\lunge_c3_tournament_results.json"

os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)

def load_or_build_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """전수 데이터셋 로드 또는 14D 피처 추출 및 캐시 저장"""
    if os.path.exists(OUT_CACHE_PATH):
        print(f"[*] Loading cached dataset from {OUT_CACHE_PATH}...")
        data = np.load(OUT_CACHE_PATH, allow_pickle=True)
        return data['X'], data['y'], data['groups'], data['views'], list(data['feature_names'])
        
    print("[*] Building 14D feature dataset from raw npz files...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"[*] Found {len(npz_files)} sequences.")
    
    X_list = []
    y_list = [] # 1: 결함 (conditions[2] == False), 0: 정상 (conditions[2] == True)
    group_list = [] # 피험자 번호 (1~9)
    view_list = [] # 시점명 (시점A~E)
    
    for idx, p in enumerate(npz_files):
        try:
            d = np.load(p, allow_pickle=True)
            kps = d['keypoints_2d_norm'] # (64, 15, 2)
            conds = d['conditions']
            meta_raw = d['metadata']
            meta_str = str(meta_raw) if isinstance(meta_raw, np.ndarray) else meta_raw
            meta = json.loads(meta_str)
            
            # 결함 타깃: 뒤다리 무릎 각도 90도 (conditions[2]) 위반 여부 (False -> 1, True -> 0)
            is_defect = 1 if not conds[2] else 0
            
            # 피험자 ID 파싱 (1~9)
            sub_str = meta.get('subject_id', '피험자1')
            sub_id = int(''.join(filter(str.isdigit, sub_str)) or 1)
            
            # 시점 파싱 (시점A ~ 시점E)
            view_name = meta.get('view', '시점A')
            
            # 14D 피처 추출
            feat_14d = extract_lunge_c3_14d_features(kps)
            
            X_list.append(feat_14d)
            y_list.append(is_defect)
            group_list.append(sub_id)
            view_list.append(view_name)
        except Exception as e:
            print(f"[!] Error processing {p}: {e}")
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    views = np.array(view_list, dtype=object)
    
    np.savez_compressed(
        OUT_CACHE_PATH,
        X=X, y=y, groups=groups, views=views,
        feature_names=np.array(FEATURE_NAMES_14D)
    )
    print(f"[+] Dataset saved to {OUT_CACHE_PATH} (Shape: {X.shape}, Defects: {np.sum(y)} / {len(y)})")
    return X, y, groups, views, FEATURE_NAMES_14D

def evaluate_3state(y_true: np.ndarray, y_prob: np.ndarray, tau_low: float = 0.35, tau_high: float = 0.65) -> Dict[str, Any]:
    """
    3-State 선택적 판정 메트릭 산출
    State 0: NORMAL (p < tau_low) -> pred = 0
    State 1: DEFECT (p > tau_high) -> pred = 1
    State 2: UNDETERMINED (tau_low <= p <= tau_high) -> 판정 보류
    """
    total = len(y_true)
    is_undetermined = (y_prob >= tau_low) & (y_prob <= tau_high)
    is_determined = ~is_undetermined
    
    n_det = int(np.sum(is_determined))
    coverage = float(n_det) / total
    
    if n_det == 0:
        return {'coverage': 0.0, 'f1_3state': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        
    y_true_det = y_true[is_determined]
    y_pred_det = (y_prob[is_determined] > 0.50).astype(int)
    
    f1 = f1_score(y_true_det, y_pred_det, zero_division=0)
    prec = precision_score(y_true_det, y_pred_det, zero_division=0)
    rec = recall_score(y_true_det, y_pred_det, zero_division=0)
    acc = accuracy_score(y_true_det, y_pred_det)
    
    # 정상 정밀도(Specificity / Clean Precision)
    cm = confusion_matrix(y_true_det, y_pred_det, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-7)
    
    return {
        'coverage': float(coverage),
        'n_determined': n_det,
        'n_undetermined': int(total - n_det),
        'f1_3state': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'accuracy': float(acc),
        'specificity': float(spec),
        'cm': [int(tn), int(fp), int(fn), int(tp)]
    }

def run_tournament(X: np.ndarray, y: np.ndarray, groups: np.ndarray, views: np.ndarray):
    """5대 알고리즘 토너먼트 및 3-State F1-Score 전수 감사"""
    print("\n" + "="*80)
    print(" [vidio-health-trainer] 바벨 런지 c3 5대 알고리즘 토너먼트 및 3-State 전수 실측")
    print("="*80)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups=groups))
    
    # 5대 모델 정의
    # 클래스 불균형: 결함 28.2% vs 정상 71.8% -> scale_pos_weight 약 2.55
    models_dict = {
        'LightGBM': lambda: lgb.LGBMClassifier(
            objective='binary',
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            scale_pos_weight=2.55,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            n_jobs=4
        ),
        'XGBoost': lambda: xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            scale_pos_weight=2.55,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='logloss',
            n_jobs=4
        ),
        'CatBoost': lambda: CatBoostClassifier(
            iterations=150,
            learning_rate=0.04,
            depth=4,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
            thread_count=4
        ),
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=4
        ),
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=4
        )
    }
    
    tournament_summary = {}
    oof_predictions = {}
    
    for name, model_fn in models_dict.items():
        print(f"\n[*] Evaluating Candidate Model: {name}...")
        oof_probs = np.zeros(len(y), dtype=np.float32)
        infer_latencies = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            clf = model_fn()
            clf.fit(X_tr, y_tr)
            
            t0 = time.perf_counter()
            p_va = clf.predict_proba(X_va)[:, 1]
            t1 = time.perf_counter()
            infer_latencies.append((t1 - t0) / len(val_idx) * 1000.0) # ms per sample
            
            oof_probs[val_idx] = p_va
            
        oof_predictions[name] = oof_probs
        
        # 1. 표준 2진 성능 (기본 tau = 0.50)
        y_pred_05 = (oof_probs >= 0.50).astype(int)
        f1_05 = f1_score(y, y_pred_05, zero_division=0)
        auc = roc_auc_score(y, oof_probs)
        acc = accuracy_score(y, y_pred_05)
        prec = precision_score(y, y_pred_05, zero_division=0)
        rec = recall_score(y, y_pred_05, zero_division=0)
        avg_lat = np.mean(infer_latencies)
        
        # 2. 최적 결정 임계값(tau*) 탐색
        best_tau = 0.50
        best_f1 = f1_05
        for tau in np.arange(0.20, 0.70, 0.01):
            f1_t = f1_score(y, (oof_probs >= tau).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_tau = float(tau)
                
        # 3. 전체 3-State 선택적 분류 성능 산출
        # tau_low = best_tau - 0.12, tau_high = best_tau + 0.12 (마진 0.12)
        res_3state = evaluate_3state(y, oof_probs, tau_low=best_tau - 0.10, tau_high=best_tau + 0.10)
        
        tournament_summary[name] = {
            'binary_f1_05': float(f1_05),
            'optimal_tau': float(best_tau),
            'binary_f1_optimal': float(best_f1),
            'roc_auc': float(auc),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'latency_ms': float(avg_lat),
            '3state_coverage': res_3state['coverage'],
            '3state_f1': res_3state['f1_3state'],
            '3state_precision': res_3state['precision'],
            '3state_recall': res_3state['recall'],
            '3state_accuracy': res_3state['accuracy'],
            '3state_specificity': res_3state['specificity'],
            'n_determined': res_3state['n_determined'],
            'n_undetermined': res_3state['n_undetermined']
        }
        
        print(f" -> {name} | Standard F1: {f1_05*100:.2f}% | Opt Tau F1({best_tau:.2f}): {best_f1*100:.2f}% | 3-State F1: {res_3state['f1_3state']*100:.2f}% (Cov: {res_3state['coverage']*100:.1f}%) | AUC: {auc*100:.2f}% | Latency: {avg_lat:.3f}ms")

    # 챔피언 선발 (3-State F1 및 전체 F1 종합)
    champ_name = max(tournament_summary.keys(), key=lambda k: tournament_summary[k]['3state_f1'])
    print(f"\n[★] Tournament Champion Selected: {champ_name} (3-State F1: {tournament_summary[champ_name]['3state_f1']*100:.2f}%)")
    
    # 챔피언 모델에 대한 공식 5대 시점 전수 감사
    champ_oof = oof_predictions[champ_name]
    champ_opt_tau = tournament_summary[champ_name]['optimal_tau']
    tau_low = champ_opt_tau - 0.10
    tau_high = champ_opt_tau + 0.10
    
    # AGENTS.md 규격 공식 5대 시점
    view_alias = {
        '시점A': ('view5', '우측 후방 (45도) [Right Back 45°]'),
        '시점B': ('view2', '우측 전방 (45도) [Right Front 45°]'),
        '시점C': ('view3', '정면 (0도) [Front View 0°]'),
        '시점D': ('view4', '좌측 전방 (45도) [Left Front 45°]'),
        '시점E': ('view1', '좌측 후방 (45도) [Left Back 45°]')
    }
    
    view_audit_results = {}
    print("\n" + "="*80)
    print(f" [공식 5대 시점 전수 감사 (AGENTS.md 규격)] - 챔피언 모델: {champ_name}")
    print("="*80)
    
    for v_kr in ['시점A', '시점B', '시점C', '시점D', '시점E']:
        v_mask = (views == v_kr)
        n_v = int(np.sum(v_mask))
        if n_v == 0:
            continue
        y_v = y[v_mask]
        p_v = champ_oof[v_mask]
        
        # 3-State 평가
        res_v = evaluate_3state(y_v, p_v, tau_low=tau_low, tau_high=tau_high)
        f1_bin = f1_score(y_v, (p_v >= champ_opt_tau).astype(int), zero_division=0)
        auc_v = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 1.0
        
        v_code, v_desc = view_alias.get(v_kr, ('view?', v_kr))
        view_audit_results[v_kr] = {
            'view_code': v_code,
            'view_desc': v_desc,
            'total_samples': n_v,
            'defects': int(np.sum(y_v)),
            'binary_f1': float(f1_bin),
            'roc_auc': float(auc_v),
            '3state_f1': res_v['f1_3state'],
            '3state_coverage': res_v['coverage'],
            '3state_accuracy': res_v['accuracy'],
            '3state_precision': res_v['precision'],
            '3state_recall': res_v['recall'],
            'n_determined': res_v['n_determined'],
            'n_undetermined': res_v['n_undetermined']
        }
        
        print(f"[{v_kr} / {v_code}] {v_desc} (N={n_v}): 3-State F1={res_v['f1_3state']*100:.2f}% (Cov: {res_v['coverage']*100:.1f}%) | Bin F1={f1_bin*100:.2f}% | AUC={auc_v*100:.2f}% | Acc={res_v['accuracy']*100:.2f}%")
        
    final_output = {
        'exercise': '04_바벨_런지',
        'target_defect': 'c3_뒤다리_무릎_각도_90도',
        'total_samples': len(y),
        'normal_count': int(np.sum(y == 0)),
        'defect_count': int(np.sum(y == 1)),
        'champion_model': champ_name,
        'tournament_summary': tournament_summary,
        'view_audit_results': view_audit_results,
        'optimal_tau': champ_opt_tau,
        'tau_low': tau_low,
        'tau_high': tau_high
    }
    
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Tournament results saved to {RESULTS_JSON_PATH}")
    return final_output, champ_name

if __name__ == "__main__":
    X, y, groups, views, feat_names = load_or_build_dataset()
    run_tournament(X, y, groups, views)

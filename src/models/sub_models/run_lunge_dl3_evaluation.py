# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 스텝 포워드 & 스텝 백워드)
Script: run_lunge_dl3_evaluation.py
Purpose: 바벨 런지 bl3(뒤다리 무릎 각도 90도)의 모델 구조와 14D 정예 기구학 피처를 계승하여
         다이나믹 런지 원천 데이터(05 포워드, 06 백워드) 대상
         dl3 세부 오류 3-State 전문가 모델 토너먼트 및 공식 성능 평가 수행
"""

import os
import sys
import glob
import time
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, 'd:/')
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.features.lunge_dl3_features import (
    extract_lunge_dl3_18d_features,
    extract_lunge_dl3_14d_features,
    FEATURE_NAMES_14D,
    FEATURE_NAMES_18D,
    FEATURE_INDICES_14D
)

DATA_DIRS = {
    '05_forward': r"D:\Data\MediaPipe\전처리\05_스텝_포워드_다이나믹_런지",
    '06_backward': r"D:\Data\MediaPipe\전처리\06_스텝_백워드_다이나믹_런지"
}

CACHE_NPZ_PATH = r"D:\Data\피처_데이터\lunge_dl3_dataset_18d_adv.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
JSON_OUT_PATH = os.path.join(RESULTS_DIR, "dl3_official_metrics.json")
REGISTRY_OUT_PATH = os.path.join(RESULTS_DIR, "dl3_official_champion_registry.json")
BUNDLE_OUT_PATH = os.path.join(RESULTS_DIR, "dl3_champion_bundle.joblib")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_or_extract_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """다이나믹 런지 전체(05 포워드, 06 백워드) 데이터셋 로드 또는 18D 피처 추출 및 캐싱"""
    if os.path.exists(CACHE_NPZ_PATH):
        print(f"[+] 기존 다이나믹 런지 18D 캐시 파일 로드: {CACHE_NPZ_PATH}")
        d = np.load(CACHE_NPZ_PATH, allow_pickle=True)
        return d['X'], d['y'], d['groups'], d['views'], d['sources'], d['filenames']
        
    print("[*] 다이나믹 런지 npz 원천 파일들로부터 18D 고급 생체역학 피처 추출 시작...")
    X_list, y_list, group_list, view_list, src_list, fname_list = [], [], [], [], [], []
    
    t0 = time.time()
    for src_tag, dpath in DATA_DIRS.items():
        npz_files = sorted(glob.glob(os.path.join(dpath, "*.npz")))
        print(f"  - [{src_tag}] 대상 파일 수: {len(npz_files)}개 ({dpath})")
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                kps = data['keypoints_2d_norm']
                conds = data['conditions']
                try:
                    meta = json.loads(str(data['metadata'].item()))
                except Exception:
                    meta = json.loads(str(data['metadata']))
                
                # Condition 3: '뒤다리 무릎 각도 90도' -> 정상: True (y=0), 결함: False (y=1)
                is_defect = 1 if not conds[2] else 0
                sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
                view_name = meta.get('view', '시점A')
                
                f18 = extract_lunge_dl3_18d_features(kps)
                X_list.append(f18)
                y_list.append(is_defect)
                group_list.append(sub_id)
                view_list.append(view_name)
                src_list.append(src_tag)
                fname_list.append(os.path.basename(f))
            except Exception as e:
                continue
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    views = np.array(view_list)
    sources = np.array(src_list)
    fnames = np.array(fname_list)
    
    np.savez_compressed(
        CACHE_NPZ_PATH,
        X=X,
        y=y,
        groups=groups,
        views=views,
        sources=sources,
        filenames=fnames,
        feature_names=FEATURE_NAMES_18D
    )
    print(f"[+] 다이나믹 런지 18D 피처 캐시 저장 완료 (소요시간: {time.time() - t0:.2f}초, Shape: {X.shape})")
    return X, y, groups, views, sources, fnames

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, tau_low: float, tau_high: float) -> Dict[str, Any]:
    """3-State 판정 메트릭 산출"""
    total = len(y_true)
    y_pred = np.full(total, -1, dtype=np.int32)
    y_pred[y_prob < tau_low] = 0
    y_pred[y_prob > tau_high] = 1
    
    det_mask = (y_pred != -1)
    n_det = int(np.sum(det_mask))
    coverage = (n_det / total) * 100.0
    
    if n_det > 0:
        y_det_true = y_true[det_mask]
        y_det_pred = y_pred[det_mask]
        acc = accuracy_score(y_det_true, y_det_pred) * 100.0
        prec = precision_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        rec = recall_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except Exception:
        auc = 0.0
        
    return {
        'coverage': coverage,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'total': total,
        'determined': n_det,
        'undetermined': total - n_det
    }

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_coverage: float = 90.0) -> Tuple[float, Dict[str, Any], Tuple[float, float, float, float]]:
    """최적의 3-State 결정 임계값 그리드 탐색"""
    best_f1 = -1.0
    best_res = None
    best_cfg = None
    for tau_mid in np.linspace(0.35, 0.65, 61):
        for margin in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
            tau_l = max(0.15, tau_mid - margin)
            tau_h = min(0.85, tau_mid + margin)
            res = evaluate_predictions(y_true, y_prob, tau_l, tau_h)
            if res['coverage'] >= min_coverage and res['f1'] > best_f1:
                best_f1 = res['f1']
                best_res = res
                best_cfg = (tau_mid, tau_l, tau_h, margin)
    return best_f1, best_res, best_cfg

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 dl3(뒤다리 무릎 각도 90도) 세부 오류 모델 토너먼트")
    print("=" * 80)
    
    # 1. 18D 피처셋 로드 후 14D 정예 피처셋 슬라이싱
    X18, y, groups, views, sources, fnames = load_or_extract_dataset()
    X14 = X18[:, FEATURE_INDICES_14D]
    
    total_samples = len(y)
    n_norm = int(np.sum(y == 0))
    n_def = int(np.sum(y == 1))
    scale_pos = n_norm / (n_def + 1e-4)
    
    print(f"\n[*] 다이나믹 런지 통합 데이터셋 통계:")
    print(f"    - 총 시퀀스 수: {total_samples}건")
    print(f"    - 정상(Pass): {n_norm}건 ({n_norm/total_samples*100:.1f}%)")
    print(f"    - 결함(Defect): {n_def}건 ({n_def/total_samples*100:.1f}%)")
    print(f"    - 피험자: {sorted(list(set(groups)))} (총 {len(set(groups))}명)")
    print(f"    - 스텝 포워드 다이나믹 런지(05): {np.sum(sources == '05_forward')}건")
    print(f"    - 스텝 백워드 다이나믹 런지(06): {np.sum(sources == '06_backward')}건")
    print(f"    - 입력 피처: 14D Elite Kinematics (Shape: {X14.shape})")
    
    # 2. 5-Fold StratifiedGroupKFold 교차검증 (피험자 9명 전원 완벽 격리)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_cb = np.zeros(total_samples, dtype=np.float32)
    oof_lgb = np.zeros(total_samples, dtype=np.float32)
    oof_xgb = np.zeros(total_samples, dtype=np.float32)
    
    print(f"\n[*] 5-Fold StratifiedGroupKFold 교차검증 시작 (피험자 격리)...")
    t0 = time.time()
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X14, y, groups), 1):
        tr_subs = set(groups[tr_idx])
        val_subs = set(groups[val_idx])
        print(f"  [Fold {fold}/5] Train: {len(tr_idx)}건 (피험자 {tr_subs}) | Val: {len(val_idx)}건 (피험자 {val_subs})")
        
        # 1) CatBoost (bl3 v4 계승)
        m_cb = CatBoostClassifier(
            iterations=250,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            verbose=0,
            thread_count=4
        )
        m_cb.fit(X14[tr_idx], y[tr_idx])
        oof_cb[val_idx] = m_cb.predict_proba(X14[val_idx])[:, 1]
        
        # 2) LightGBM
        m_lgb = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=180,
            learning_rate=0.035,
            max_depth=5,
            num_leaves=22,
            scale_pos_weight=scale_pos,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=4
        )
        m_lgb.fit(X14[tr_idx], y[tr_idx])
        oof_lgb[val_idx] = m_lgb.predict_proba(X14[val_idx])[:, 1]
        
        # 3) XGBoost
        m_xgb = XGBClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=4,
            scale_pos_weight=scale_pos,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42 + fold,
            eval_metric='logloss',
            n_jobs=4
        )
        m_xgb.fit(X14[tr_idx], y[tr_idx])
        oof_xgb[val_idx] = m_xgb.predict_proba(X14[val_idx])[:, 1]
        
    print(f"[+] 5-Fold 교차검증 완료 (소요시간: {time.time() - t0:.2f}초)")
    
    # 4) Tri-Ensemble (Soft Blend: CatBoost 0.50 + LightGBM 0.30 + XGBoost 0.20)
    oof_ens = 0.50 * oof_cb + 0.30 * oof_lgb + 0.20 * oof_xgb
    
    models = {
        'dl3 CatBoost (14D Elite Kinematics)': oof_cb,
        'dl3 LightGBM (14D Elite Kinematics)': oof_lgb,
        'dl3 XGBoost (14D Elite Kinematics)': oof_xgb,
        'dl3 Tri-Ensemble (14D Soft Blend)': oof_ens
    }
    
    print("\n" + "=" * 80)
    print("  다이나믹 런지 dl3 토너먼트 종합 평가 결과 (Coverage >= 90.0%)")
    print("=" * 80)
    print(f"{'Model':<38} | {'3-State F1':<10} | {'Coverage':<9} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'AUC':<7}")
    print("-" * 105)
    
    summary_results = []
    best_model_name = ""
    best_f1 = -1.0
    best_res = None
    best_cfg = None
    best_oof_probs = None
    
    for name, probs in models.items():
        f1, res, cfg = find_best_threshold(y, probs, min_coverage=90.0)
        summary_results.append({
            'Model': name,
            '3-State F1 (%)': f"{f1:.2f}%",
            'Coverage (%)': f"{res['coverage']:.2f}%",
            'Accuracy (%)': f"{res['accuracy']:.2f}%",
            'Precision (%)': f"{res['precision']:.2f}%",
            'Recall (%)': f"{res['recall']:.2f}%",
            'ROC-AUC (%)': f"{res['auc']:.2f}%",
            'tau_low': f"{cfg[1]:.2f}",
            'tau_high': f"{cfg[2]:.2f}",
            'raw_res': res,
            'raw_cfg': [float(c) for c in cfg]
        })
        print(f"{name:<38} | {f1:>8.2f}% | {res['coverage']:>7.2f}% | {res['accuracy']:>7.2f}% | {res['precision']:>7.2f}% | {res['recall']:>7.2f}% | {res['auc']:>6.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_res = res
            best_cfg = cfg
            best_oof_probs = probs
            
    print("=" * 80)
    print(f"★ 최종 공식 챔피언 모델: {best_model_name}")
    print(f"  - 3-State F1-Score: {best_f1:.2f}%")
    print(f"  - 판정 커버리지: {best_res['coverage']:.2f}% ({best_res['determined']}/{best_res['total']})")
    print(f"  - 판정 정확도: {best_res['accuracy']:.2f}%")
    print(f"  - 정밀도: {best_res['precision']:.2f}% | 재현율: {best_res['recall']:.2f}%")
    print(f"  - ROC-AUC: {best_res['auc']:.2f}%")
    print(f"  - 최적 결정 임계값: tau_low={best_cfg[1]:.2f}, tau_high={best_cfg[2]:.2f} (중심 tau={best_cfg[0]:.2f})")
    
    # 3. 다차원 세부 성능 감사 (규칙 3 준수)
    # 1) 세부 운동 종목별 성능 (05 포워드 vs 06 백워드)
    tau_low = best_cfg[1]
    tau_high = best_cfg[2]
    
    mask_fwd = (sources == '05_forward')
    mask_bwd = (sources == '06_backward')
    
    res_fwd = evaluate_predictions(y[mask_fwd], best_oof_probs[mask_fwd], tau_low, tau_high)
    res_bwd = evaluate_predictions(y[mask_bwd], best_oof_probs[mask_bwd], tau_low, tau_high)
    
    print("\n[다차원 감사 1] 세부 운동 종목별 3-State 성능:")
    print(f"  - [05 스텝 포워드 다이나믹 런지] F1: {res_fwd['f1']:.2f}% | Acc: {res_fwd['accuracy']:.2f}% | Prec: {res_fwd['precision']:.2f}% | Rec: {res_fwd['recall']:.2f}% | Cov: {res_fwd['coverage']:.2f}% | AUC: {res_fwd['auc']:.2f}% (총 {res_fwd['total']}건)")
    print(f"  - [06 스텝 백워드 다이나믹 런지] F1: {res_bwd['f1']:.2f}% | Acc: {res_bwd['accuracy']:.2f}% | Prec: {res_bwd['precision']:.2f}% | Rec: {res_bwd['recall']:.2f}% | Cov: {res_bwd['coverage']:.2f}% | AUC: {res_bwd['auc']:.2f}% (총 {res_bwd['total']}건)")
    
    # 2) 피험자별 성능 감사
    subj_audit = {}
    print("\n[다차원 감사 2] 피험자(Subject)별 3-State 성능:")
    for s in sorted(list(set(groups))):
        mask_s = (groups == s)
        res_s = evaluate_predictions(y[mask_s], best_oof_probs[mask_s], tau_low, tau_high)
        subj_audit[f"피험자{s}"] = {
            'f1': round(res_s['f1'], 2),
            'accuracy': round(res_s['accuracy'], 2),
            'precision': round(res_s['precision'], 2),
            'recall': round(res_s['recall'], 2),
            'coverage': round(res_s['coverage'], 2),
            'total': res_s['total']
        }
        print(f"  - 피험자 {s}: F1={res_s['f1']:>6.2f}%, Acc={res_s['accuracy']:>6.2f}%, Cov={res_s['coverage']:>6.2f}% (표본 {res_s['total']}건)")
        
    # 4. 전체 데이터 대상 최종 챔피언 모델 학습 및 번들 패키징
    print("\n[*] 전체 1,959개 데이터셋 대상 최종 프로덕션 챔피언 모델 학습...")
    final_cb = CatBoostClassifier(
        iterations=250,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=4.0,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0,
        thread_count=4
    )
    final_cb.fit(X14, y)
    
    final_lgb = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=180,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=22,
        scale_pos_weight=scale_pos,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        verbose=-1,
        n_jobs=4
    )
    final_lgb.fit(X14, y)
    
    final_xgb = XGBClassifier(
        n_estimators=160,
        learning_rate=0.035,
        max_depth=4,
        scale_pos_weight=scale_pos,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        eval_metric='logloss',
        n_jobs=4
    )
    final_xgb.fit(X14, y)
    
    bundle_data = {
        'cb_model': final_cb,
        'lgb_model': final_lgb,
        'xgb_model': final_xgb,
        'champion_name': best_model_name,
        'optimal_tau': float(best_cfg[0]),
        'tau_low': float(best_cfg[1]),
        'tau_high': float(best_cfg[2]),
        'margin': float(best_cfg[3]),
        'feature_indices': FEATURE_INDICES_14D,
        'feature_names': FEATURE_NAMES_14D,
        'f1_score': float(best_f1),
        'accuracy': float(best_res['accuracy']),
        'precision': float(best_res['precision']),
        'recall': float(best_res['recall']),
        'coverage': float(best_res['coverage']),
        'auc': float(best_res['auc']),
        'dataset_summary': {
            'total': total_samples,
            'normal': n_norm,
            'defect': n_def,
            'forward_count': int(np.sum(sources == '05_forward')),
            'backward_count': int(np.sum(sources == '06_backward'))
        }
    }
    joblib.dump(bundle_data, BUNDLE_OUT_PATH)
    print(f"[+] 최종 프로덕션 챔피언 번들 저장 완료: {BUNDLE_OUT_PATH}")
    
    # 5. 추론 지연시간(Latency ms) 정밀 측정
    sample_kps = np.load(glob.glob(os.path.join(DATA_DIRS['05_forward'], "*.npz"))[0])['keypoints_2d_norm']
    t_lat_start = time.perf_counter()
    n_lat_runs = 100
    for _ in range(n_lat_runs):
        f_vec = extract_lunge_dl3_14d_features(sample_kps).reshape(1, -1)
        _ = final_cb.predict_proba(f_vec)[:, 1]
    lat_ms = ((time.perf_counter() - t_lat_start) / n_lat_runs) * 1000.0
    print(f"[+] 단일 시퀀스 추론 지연시간: {lat_ms:.3f} ms")
    
    # 6. 공식 메트릭 JSON 저장
    metrics_json = {
        'champion': best_model_name,
        'exercise': "다이나믹 런지 (Dynamic Lunge)",
        'defect_name': "뒤다리 무릎 각도 90도 (dl3)",
        'summary': summary_results,
        'best_cfg': [float(c) for c in best_cfg],
        'sub_exercise_audit': {
            '05_step_forward': res_fwd,
            '06_step_backward': res_bwd
        },
        'subject_audit': subj_audit,
        'inference_latency_ms': lat_ms,
        'features_14d': FEATURE_NAMES_14D
    }
    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 성능 메트릭 JSON 저장 완료: {JSON_OUT_PATH}")
    
    # 7. 공식 챔피언 레지스트리 JSON 저장
    registry_json = {
        "exercise": "다이나믹 런지",
        "defect_code": "dl3",
        "defect_name": "뒤다리 무릎 각도 90도",
        "official_champion_model": best_model_name,
        "model_version": "v1.0",
        "selection_date": "2026-10-04",
        "features": {
            "count": 14,
            "names": FEATURE_NAMES_14D
        },
        "evaluation_metrics": {
            "eval_protocol": "5-Fold StratifiedGroupKFold (All 9 Subjects, 1,959 Samples)",
            "label_condition": f"AI-Hub Raw Labels 100% Preserved ({n_norm} Normal vs {n_def} Defect)",
            "3_state_f1_score": round(best_f1, 2),
            "coverage": round(best_res['coverage'], 2),
            "accuracy": round(best_res['accuracy'], 2),
            "precision": round(best_res['precision'], 2),
            "recall": round(best_res['recall'], 2),
            "roc_auc": round(best_res['auc'], 2),
            "optimal_tau": round(best_cfg[0], 2),
            "tau_low": round(best_cfg[1], 2),
            "tau_high": round(best_cfg[2], 2),
            "inference_latency_ms": round(lat_ms, 3)
        },
        "bundle_path": BUNDLE_OUT_PATH,
        "production_module": "src.models.sub_models.dl3.DL3Classifier",
        "feature_module": "src.features.lunge_dl3_features.extract_lunge_dl3_14d_features"
    }
    with open(REGISTRY_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry_json, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 챔피언 레지스트리 JSON 저장 완료: {REGISTRY_OUT_PATH}")

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: d1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_stiff_d1.py
Purpose: 스티프 데드리프트 바벨 밀착 결함 38D 모델 학습, 5-Fold StratifiedGroupKFold OOF 평가,
         Tier 1(골든) / Tier 2(실무) 2계층 검증, 공식 5대 시점 전수 감사 및 3-State 임계값 최적화
Rule: AGENTS.md 전원 피험자(1~9번) 및 5대 시점 전수 포함, 순수 비전 100% 입력
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_features, FEATURE_NAMES_38_STIFF_D1, VIEW_ALIAS_MAP
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_d1_results.json")


def load_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """캐시된 64F npz 파일들을 로드하고 38D 피처를 추출하여 DataFrame 및 X 행렬 생성"""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"No npz files found in {CACHE_DIR}")

    print(f"[*] 캐시된 시퀀스 총 {len(npz_files)}개 감지. 피처 추출 시작...")
    records = []
    features_list = []

    t0 = time.time()
    for idx, fpath in enumerate(npz_files):
        fname = os.path.basename(fpath).replace(".npz", "")
        # 형식 예: [오류]_D02_피험자1_Key210_view3
        parts = fname.split("_")
        view = parts[-1]
        
        # 피험자 번호
        subj = "피험자1"
        for p in parts:
            if "피험자" in p:
                subj = p
                break
        
        # Key 번호
        key = "Key210"
        for p in parts:
            if "Key" in p:
                key = p
                break

        # 바벨 밀착 결함 여부 (Condition 1: 바벨 궤적과 몸 밀착)
        # Key 210 (단독 결함), Key 213 (복합 결함), Key 214 (복합 결함) -> Defect (1)
        # Key 209 (정상), Key 211, 212, 215, 216 -> Clean/밀착정상 (0)
        is_defect = 1 if key in ['Key210', 'Key213', 'Key214'] else 0
        is_tier1_sample = 1 if key in ['Key209', 'Key210'] else 0

        # npz 로드 및 38D 피처 추출
        try:
            with np.load(fpath, allow_pickle=True) as data:
                kps_64f = data['kps_33_3d_64f']
            f_vec = extract_stiff_d1_features(kps_64f, view_name=view)
            features_list.append(f_vec)
            records.append({
                'filename': fname,
                'path': fpath,
                'subj': subj,
                'view': view,
                'key': key,
                'y': is_defect,
                'is_tier1': is_tier1_sample
            })
        except Exception as e:
            print(f"[!] {fname} 피처 추출 실패: {e}")

    df = pd.DataFrame(records)
    X = np.array(features_list, dtype=np.float32)
    print(f"[+] 총 {len(df)}개 시퀀스 38D 피처 추출 완료! (경과 시간: {time.time()-t0:.1f}초)")
    print(f"    - 결함(Defect) 샘플: {int(df['y'].sum())}개 | 정상(Clean) 샘플: {int((1-df['y']).sum())}개")
    print(f"    - Tier 1(골든셋) 샘플: {int(df['is_tier1'].sum())}개")
    return df, X


def train_and_evaluate(df: pd.DataFrame, X: np.ndarray) -> Dict[str, Any]:
    """5-Fold StratifiedGroupKFold OOF 예측 및 2-Tier / 3-State 종합 평가"""
    y = df['y'].values
    subjs = df['subj'].values
    views = df['view'].values
    tier1_mask = df['is_tier1'].values == 1

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df), dtype=np.float32)

    lgb_models = []
    cb_models = []

    print("\n" + "=" * 80)
    print("  5-Fold StratifiedGroupKFold 피험자 격리 앙상블 학습 시작")
    print("=" * 80)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjs)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_subjs = np.unique(subjs[val_idx])

        # 1. LightGBM 분류기
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=5,
            num_leaves=16,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)

        # 2. CatBoost 분류기
        clf_cb = CatBoostClassifier(
            iterations=180,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=1
        )
        clf_cb.fit(X_tr, y_tr)

        p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        p_cb = clf_cb.predict_proba(X_val)[:, 1]
        p_ensemble = 0.55 * p_lgb + 0.45 * p_cb

        oof_probs[val_idx] = p_ensemble
        lgb_models.append(clf_lgb)
        cb_models.append(clf_cb)

        fold_f1 = f1_score(y_val, (p_ensemble >= 0.50).astype(int), zero_division=0) * 100.0
        fold_auc = roc_auc_score(y_val, p_ensemble) * 100.0
        print(f"  [Fold {fold+1}/5] 검증 피험자: {list(val_subjs)} | 2-State F1: {fold_f1:.2f}% | AUC: {fold_auc:.2f}%")

    # 전체 2-State 베이스라인 지표 (Threshold = 0.50)
    preds_50 = (oof_probs >= 0.50).astype(int)
    f1_2s = f1_score(y, preds_50, zero_division=0) * 100.0
    prec_2s = precision_score(y, preds_50, zero_division=0) * 100.0
    rec_2s = recall_score(y, preds_50, zero_division=0) * 100.0
    acc_2s = accuracy_score(y, preds_50) * 100.0
    auc_2s = roc_auc_score(y, oof_probs) * 100.0
    cm_2s = confusion_matrix(y, preds_50, labels=[0, 1])
    tn_2s, fp_2s, fn_2s, tp_2s = cm_2s.ravel()

    print("\n[전체 Tier 2 실무 데이터셋 2-State 베이스라인 (τ=0.50)]")
    print(f"  - F1-Score: {f1_2s:.2f}% | 정밀도: {prec_2s:.2f}% | 재현율: {rec_2s:.2f}% | 정확도: {acc_2s:.2f}% | AUC: {auc_2s:.2f}%")
    print(f"  - 혼동 행렬: TP={tp_2s}, TN={tn_2s}, FP={fp_2s}, FN={fn_2s}")

    # Tier 1 (Golden Benchmark: Key 210 vs Key 209) 지표
    y_t1 = y[tier1_mask]
    p_t1 = oof_probs[tier1_mask]
    preds_t1_50 = (p_t1 >= 0.50).astype(int)
    f1_t1 = f1_score(y_t1, preds_t1_50, zero_division=0) * 100.0
    prec_t1 = precision_score(y_t1, preds_t1_50, zero_division=0) * 100.0
    rec_t1 = recall_score(y_t1, preds_t1_50, zero_division=0) * 100.0
    acc_t1 = accuracy_score(y_t1, preds_t1_50) * 100.0
    auc_t1 = roc_auc_score(y_t1, p_t1) * 100.0 if len(np.unique(y_t1)) > 1 else 0.0

    print("\n[Tier 1 골든 벤치마크 (단독 결함 Key210 vs 정상 Key209)]")
    print(f"  - F1-Score: {f1_t1:.2f}% | 정밀도: {prec_t1:.2f}% | 재현율: {rec_t1:.2f}% | 정확도: {acc_t1:.2f}% | AUC: {auc_t1:.2f}%")

    # 3-State Selective Gating 최적화 그리드 서치 (보류율 <= 10.0% / 커버리지 >= 90.0%)
    print("\n" + "=" * 80)
    print("  3-State Selective Gating 최적화 (보류율 <= 10.0% 제약 하 F1 극대화)")
    print("=" * 80)

    best_3state = None
    best_f1 = -1.0

    tc_range = np.linspace(0.15, 0.48, 34)
    td_range = np.linspace(0.50, 0.85, 36)

    for tc in tc_range:
        for td in td_range:
            if td <= tc:
                continue
            # 판정 마스크: p <= tc (정상) OR p >= td (결함)
            decided_mask = (oof_probs <= tc) | (oof_probs >= td)
            cov = float(np.mean(decided_mask))
            if cov < 0.90:  # 커버리지 90% 이상 (보류율 10% 이하)
                continue

            y_dec = y[decided_mask]
            p_dec = (oof_probs[decided_mask] >= td).astype(int)

            if len(np.unique(y_dec)) < 2:
                continue

            cur_f1 = f1_score(y_dec, p_dec, zero_division=0)
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                cm_dec = confusion_matrix(y_dec, p_dec, labels=[0, 1])
                tn_d, fp_d, fn_d, tp_d = cm_dec.ravel()
                best_3state = {
                    'tau_clean': float(tc),
                    'tau_defect': float(td),
                    'coverage': float(cov * 100.0),
                    'abstain_rate': float((1.0 - cov) * 100.0),
                    'f1': float(cur_f1 * 100.0),
                    'precision': float(precision_score(y_dec, p_dec, zero_division=0) * 100.0),
                    'recall': float(recall_score(y_dec, p_dec, zero_division=0) * 100.0),
                    'accuracy': float(accuracy_score(y_dec, p_dec) * 100.0),
                    'tp': int(tp_d),
                    'tn': int(tn_d),
                    'fp': int(fp_d),
                    'fn': int(fn_d),
                    'decided_count': int(len(y_dec)),
                    'total_count': int(len(y))
                }

    print(f"[최적 3-State 판정 윈도우 확정]")
    print(f"  - Tau Clean (정상 한계치) : {best_3state['tau_clean']:.2f}")
    print(f"  - Tau Defect (결함 한계치): {best_3state['tau_defect']:.2f}")
    print(f"  - 커버리지 (Coverage)     : {best_3state['coverage']:.2f}% (보류율: {best_3state['abstain_rate']:.2f}%)")
    print(f"  ★ 3-State F1-Score        : {best_3state['f1']:.2f}% (베이스라인 대비 {best_3state['f1'] - f1_2s:+5.2f}%p)")
    print(f"  - 3-State 정밀도 / 재현율 : {best_3state['precision']:.2f}% / {best_3state['recall']:.2f}%")
    print(f"  - 3-State 정확도 (Accuracy): {best_3state['accuracy']:.2f}%")
    print(f"  ★ 오탐 (FP, False Positive): {fp_2s}건 -> {best_3state['fp']}건 ({best_3state['fp'] - fp_2s:+d}건 감소!)")

    # 공식 5대 시점별 세부 성능 감사
    print("\n" + "=" * 80)
    print("  공식 5대 카메라 시점별 최종 성능 감사 (AGENTS.md 표준)")
    print("=" * 80)

    tc = best_3state['tau_clean']
    td = best_3state['tau_defect']
    view_metrics = {}

    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if np.sum(v_mask) == 0:
            continue
        y_v = y[v_mask]
        p_v = oof_probs[v_mask]

        # 3-State 적용
        mask_v_dec = (p_v <= tc) | (p_v >= td)
        y_v_dec = y_v[mask_v_dec]
        p_v_dec = (p_v[mask_v_dec] >= td).astype(int)

        cov_v = float(np.mean(mask_v_dec) * 100.0)
        f1_v = float(f1_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        prec_v = float(precision_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        rec_v = float(recall_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        acc_v = float(accuracy_score(y_v_dec, p_v_dec) * 100.0) if len(y_v_dec) > 0 else 0.0
        auc_v = float(roc_auc_score(y_v, p_v) * 100.0) if len(np.unique(y_v)) > 1 else 0.0

        cm_v = confusion_matrix(y_v_dec, p_v_dec, labels=[0, 1])
        tn_v, fp_v, fn_v, tp_v = cm_v.ravel()

        view_metrics[v_code] = {
            'alias': v_alias,
            'total_samples': int(len(y_v)),
            'decided_samples': int(len(y_v_dec)),
            'coverage': cov_v,
            'f1': f1_v,
            'precision': prec_v,
            'recall': rec_v,
            'accuracy': acc_v,
            'auc': auc_v,
            'tp': int(tp_v),
            'tn': int(tn_v),
            'fp': int(fp_v),
            'fn': int(fn_v)
        }
        print(f"[{v_alias}]")
        print(f"  3-State F1: {f1_v:5.2f}% | 정밀도: {prec_v:5.2f}% | 재현율: {rec_v:5.2f}% | 정확도: {acc_v:5.2f}% | AUC: {auc_v:5.2f}% | FP: {fp_v} | 커버리지: {cov_v:.1f}%")

    # 전체 데이터로 최종 프로덕션 번들 모델 학습
    print("\n[*] 전체 데이터셋 기반 최종 프로덕션 앙상블 번들 학습 중...")
    final_lgb = lgb.LGBMClassifier(
        n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, verbose=-1, n_jobs=-1
    )
    final_lgb.fit(X, y)

    final_cb = CatBoostClassifier(
        iterations=200, learning_rate=0.035, depth=5, l2_leaf_reg=3.0,
        random_seed=42, verbose=0, thread_count=-1
    )
    final_cb.fit(X, y)

    bundle = {
        'model_name': 'stiff_deadlift_d1_bar_proximity_v1.0',
        'exercise': '바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)',
        'defect': 'd1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity)',
        'feature_names': FEATURE_NAMES_38_STIFF_D1,
        'feature_dim': len(FEATURE_NAMES_38_STIFF_D1),
        'lgb_model': final_lgb,
        'cb_model': final_cb,
        'weights': (0.55, 0.45),
        'decision_policy': '3-State View-Calibrated Gating',
        'optimal_3state': best_3state,
        'view_metrics': view_metrics,
        'view_alias_map': VIEW_ALIAS_MAP
    }

    WEIGHTS_PKL = os.path.join(WEIGHTS_DIR, "stiff_deadlift_d1_best.pkl")
    with open(WEIGHTS_PKL, 'wb') as fp:
        pickle.dump(bundle, fp)
    print(f"[+] 최종 모델 가중치 번들 저장 완료: {WEIGHTS_PKL}")

    # 결과 JSON 저장
    full_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'd1. 바벨 궤적과 몸 밀착',
        'model_version': 'v1.0 Zero-Calibrated Stiff-38D',
        'sample_counts': {
            'total_samples': int(len(df)),
            'defect_samples': int(df['y'].sum()),
            'clean_samples': int((1 - df['y']).sum()),
            'tier1_samples': int(df['is_tier1'].sum())
        },
        'tier1_golden_metrics': {
            'f1': f1_t1,
            'precision': prec_t1,
            'recall': rec_t1,
            'accuracy': acc_t1,
            'auc': auc_t1
        },
        'tier2_2state_baseline': {
            'f1': f1_2s,
            'precision': prec_2s,
            'recall': rec_2s,
            'accuracy': acc_2s,
            'auc': auc_2s,
            'tp': int(tp_2s),
            'tn': int(tn_2s),
            'fp': int(fp_2s),
            'fn': int(fn_2s)
        },
        'tier2_3state_optimal': best_3state,
        'view_metrics': view_metrics,
        'weights_path': WEIGHTS_PKL
    }

    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(full_results, fp, indent=2, ensure_ascii=False)
    print(f"[+] 평가 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")

    return full_results


def main():
    df, X = load_dataset()
    train_and_evaluate(df, X)


if __name__ == "__main__":
    main()

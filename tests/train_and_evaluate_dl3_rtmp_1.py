# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_포워드 & 06_백워드)
Defect: Condition 3 - 뒤다리 무릎 각도 90도 (dl3)
Module: train_and_evaluate_dl3_rtmp_1.py
Purpose: 바벨 런지 bl3_v4의 14D 정예 기구학 엔진을 계승하고
         다이나믹 런지 RTMPose 2D Halpe26에 맞추어 최적화한
         dl3_rtmp_1 모델의 학습, 앙상블, 3-State 캘리브레이션 및
         AI-Hub 9명 피험자 전수(9,795건) 5-Fold 완전 격리 교차검증
"""

import os
import sys
import json
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, log_loss
)
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")
sys.path.append(r"D:\src\models\second_models\dynamic_lunge")

from lunge_dl3_rtmp_features import (
    FEATURE_NAMES_27D,
    FEATURE_NAMES_22D,
    FEATURE_NAMES_14D
)

CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_dl3_rtmp_27d_cache.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
WEIGHTS_DIR = r"D:\src\models\second_models\dynamic_lunge\weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

def find_best_3state_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_retake_rate: float = 0.15
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """
    3-State 최적 임계값 (tau_clean, tau_defect) 탐색
    - 보류율(Retake Rate) <= max_retake_rate 제약 하에서 F1 최대화
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    
    best_f1 = -1.0
    best_tau = (0.5, 0.5)
    best_metrics = {}
    
    p_min = float(np.percentile(y_prob, 1))
    p_max = float(np.percentile(y_prob, 99))
    taus = np.linspace(max(0.05, p_min), min(0.95, p_max), 50)
    
    for i in range(len(taus)):
        t_c = taus[i]
        for j in range(i, len(taus)):
            t_d = taus[j]
            pred = np.full(N, 2, dtype=int)
            pred[y_prob < t_c] = 0
            pred[y_prob >= t_d] = 1
            
            cov_mask = (pred != 2)
            retake = 1.0 - np.mean(cov_mask)
            if retake > max_retake_rate:
                continue
                
            if np.sum(cov_mask) > 0:
                y_c = y_true[cov_mask]
                p_c = pred[cov_mask]
                f1 = f1_score(y_c, p_c, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = (float(t_c), float(t_d))
                    best_metrics = {
                        'f1': float(f1),
                        'acc': float(accuracy_score(y_c, p_c)),
                        'prec': float(precision_score(y_c, p_c, zero_division=0)),
                        'rec': float(recall_score(y_c, p_c, zero_division=0)),
                        'cov': float(np.mean(cov_mask)),
                        'retake': float(retake)
                    }
                    
    if best_f1 < 0:
        pred_bin = (y_prob >= 0.5).astype(int)
        best_tau = (0.5, 0.5)
        best_metrics = {
            'f1': float(f1_score(y_true, pred_bin, zero_division=0)),
            'acc': float(accuracy_score(y_true, pred_bin)),
            'prec': float(precision_score(y_true, pred_bin, zero_division=0)),
            'rec': float(recall_score(y_true, pred_bin, zero_division=0)),
            'cov': 1.0,
            'retake': 0.0
        }
    return best_tau, best_metrics

def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 다이나믹 런지 dl3_rtmp_1 모델 5-Fold 학습 및 최적화")
    print("=" * 85)
    
    t_start = time.time()
    
    # 1. 캐시 데이터 로드
    print("[*] 1. 캐시 데이터 로드 중...")
    d = np.load(CACHE_PATH, allow_pickle=True)
    X_27 = d['X']
    y = d['y']
    subjects = d['subjects']
    views = d['views']
    exercise_types = d['exercise_types']
    fnames = d['filenames']
    
    N, D = X_27.shape
    print(f"[+] 총 샘플 수: {N:,} 건, 피처 수: {D}D")
    print(f"[+] 라벨 분포: 정상(0) {np.sum(y == 0):,} 건, 결함(1) {np.sum(y == 1):,} 건 (결함율 {np.mean(y)*100:.2f}%)")
    
    # 피처 서브셋 정의
    X_14 = X_27[:, :14] # bl3 14D 베이스라인
    X_22 = X_27[:, :22] # 14D + 8D RTMPose/다이나믹 기구학
    
    feature_configs = {
        '14D_baseline': X_14,
        '22D_kinematics': X_22,
        '27D_all_views': X_27
    }
    
    # 2. 5-Fold StratifiedGroupKFold 분할 (피험자 1~9번 완전 격리)
    print("\n[*] 2. 5-Fold StratifiedGroupKFold 분할 (피험자 9명 완전 격리, Zero Leakage)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X_27, y, groups=subjects))
    
    for f_idx, (tr, va) in enumerate(folds):
        tr_subs = sorted(list(set(subjects[tr])))
        va_subs = sorted(list(set(subjects[va])))
        print(f"  - Fold {f_idx+1}: Train Subs={tr_subs} ({len(tr):,}건), Val Subs={va_subs} ({len(va):,}건)")
        assert len(set(tr_subs).intersection(set(va_subs))) == 0, f"Fold {f_idx+1} Data Leakage detected!"
        
    # 3. 모델 토너먼트 및 피처 서브셋 비교 실험
    print("\n" + "=" * 85)
    print("  [*] 3. 모델 토너먼트: 14D vs 22D vs 27D & CatBoost vs LightGBM vs XGBoost vs Ensemble")
    print("=" * 85)
    
    tournament_results = {}
    oof_predictions = {}
    
    # CatBoost 하이퍼파라미터 (bl3_v4 계승 및 최적화)
    cb_params = {
        'iterations': 400,
        'learning_rate': 0.04,
        'depth': 6,
        'l2_leaf_reg': 4.0,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 4
    }
    
    # LightGBM 하이퍼파라미터
    lgb_params = {
        'n_estimators': 350,
        'learning_rate': 0.04,
        'max_depth': 6,
        'num_leaves': 31,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    }
    
    # XGBoost 하이퍼파라미터
    scale_pos = float(np.sum(y == 0)) / float(np.sum(y == 1))
    xgb_params = {
        'n_estimators': 350,
        'learning_rate': 0.04,
        'max_depth': 5,
        'scale_pos_weight': scale_pos,
        'random_state': 42,
        'n_jobs': 4,
        'eval_metric': 'logloss'
    }
    
    for cfg_name, X_feat in feature_configs.items():
        print(f"\n>>> 피처 구성: [{cfg_name}] (입력 차원: {X_feat.shape[1]}D) <<<")
        oof_cb = np.zeros(N, dtype=np.float32)
        oof_lgb = np.zeros(N, dtype=np.float32)
        oof_xgb = np.zeros(N, dtype=np.float32)
        
        for f_idx, (tr, va) in enumerate(folds):
            X_tr, y_tr = X_feat[tr], y[tr]
            X_va, y_va = X_feat[va], y[va]
            
            # 1. CatBoost
            model_cb = CatBoostClassifier(**cb_params)
            model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
            oof_cb[va] = model_cb.predict_proba(X_va)[:, 1]
            
            # 2. LightGBM
            model_lgb = lgb.LGBMClassifier(**lgb_params)
            model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            oof_lgb[va] = model_lgb.predict_proba(X_va)[:, 1]
            
            # 3. XGBoost
            model_xgb = xgb.XGBClassifier(**xgb_params)
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[va] = model_xgb.predict_proba(X_va)[:, 1]
            
        # 앙상블: 40% CB + 35% LGB + 25% XGB
        oof_ens = 0.40 * oof_cb + 0.35 * oof_lgb + 0.25 * oof_xgb
        
        oof_predictions[f"{cfg_name}_CB"] = oof_cb
        oof_predictions[f"{cfg_name}_LGB"] = oof_lgb
        oof_predictions[f"{cfg_name}_XGB"] = oof_xgb
        oof_predictions[f"{cfg_name}_Ensemble"] = oof_ens
        
        # 모델별 성능 요약
        for m_name, oof_p in [
            (f"{cfg_name}_CatBoost", oof_cb),
            (f"{cfg_name}_LightGBM", oof_lgb),
            (f"{cfg_name}_XGBoost", oof_xgb),
            (f"{cfg_name}_Ensemble", oof_ens)
        ]:
            auc = roc_auc_score(y, oof_p)
            f1_2s = f1_score(y, (oof_p >= 0.5).astype(int), zero_division=0)
            tau, m_3s = find_best_3state_thresholds(y, oof_p, max_retake_rate=0.15)
            
            tournament_results[m_name] = {
                'auc': float(auc),
                'f1_2state': float(f1_2s),
                'tau': tau,
                'f1_3state': float(m_3s['f1']),
                'acc_3state': float(m_3s['acc']),
                'prec_3state': float(m_3s['prec']),
                'rec_3state': float(m_3s['rec']),
                'cov_3state': float(m_3s['cov']),
                'retake_3state': float(m_3s['retake'])
            }
            print(f"  [{m_name}] AUC: {auc:.4f} | 2-State F1: {f1_2s*100:.2f}% | 3-State F1: {m_3s['f1']*100:.2f}% (Tau={tau[0]:.2f}~{tau[1]:.2f}, Retake={m_3s['retake']*100:.2f}%)")
            
    # 4. 최우수 챔피언 모델 선정
    print("\n" + "=" * 85)
    print("  [*] 4. 최우수 챔피언 모델 선정 및 심층 성능 감사")
    print("=" * 85)
    
    # F1_3state 기준으로 최고 모델 선정
    best_model_name = max(tournament_results.keys(), key=lambda k: tournament_results[k]['f1_3state'])
    best_info = tournament_results[best_model_name]
    print(f"\n[★] 공식 챔피언 모델 선정: {best_model_name}")
    print(f"    - 3-State F1: {best_info['f1_3state']*100:.2f}%")
    print(f"    - ROC-AUC: {best_info['auc']:.4f}")
    print(f"    - 3-State Accuracy: {best_info['acc_3state']*100:.2f}%")
    print(f"    - 최적 임계값: [{best_info['tau'][0]:.3f}, {best_info['tau'][1]:.3f}]")
    print(f"    - 불확실성 보류율: {best_info['retake_3state']*100:.2f}% (Coverage: {best_info['cov_3state']*100:.2f}%)")
    
    # 챔피언 모델의 OOF 확률
    best_oof_key = best_model_name.replace("CatBoost", "CB").replace("LightGBM", "LGB").replace("XGBoost", "XGB")
    champ_oof_prob = oof_predictions[best_oof_key]
    tau_clean, tau_defect = best_info['tau']
    
    champ_pred_3s = np.full(N, 2, dtype=int)
    champ_pred_3s[champ_oof_prob < tau_clean] = 0
    champ_pred_3s[champ_oof_prob >= tau_defect] = 1
    
    # 5. AGENTS.md 공식 5대 카메라 시점 전수 감사
    print("\n[*] 5. 공식 5대 시점(A~E)별 세부 성능 감사:")
    view_performance = {}
    for v_code, v_name in VIEW_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        p_v = champ_oof_prob[v_mask]
        pred_v = champ_pred_3s[v_mask]
        
        auc_v = roc_auc_score(y_v, p_v)
        f1_2s_v = f1_score(y_v, (p_v >= 0.5).astype(int), zero_division=0)
        
        cov_v = (pred_v != 2)
        f1_3s_v = f1_score(y_v[cov_v], pred_v[cov_v], zero_division=0) if np.sum(cov_v) > 0 else 0.0
        acc_3s_v = accuracy_score(y_v[cov_v], pred_v[cov_v]) if np.sum(cov_v) > 0 else 0.0
        retake_v = 1.0 - np.mean(cov_v)
        
        view_performance[v_code] = {
            'view_name': v_name,
            'count': int(np.sum(v_mask)),
            'defect_count': int(np.sum(y_v == 1)),
            'auc': float(auc_v),
            'f1_2state': float(f1_2s_v),
            'f1_3state': float(f1_3s_v),
            'acc_3state': float(acc_3s_v),
            'retake': float(retake_v)
        }
        print(f"  - {v_name:25s}: 3-State F1={f1_3s_v*100:.2f}% | 2-State F1={f1_2s_v*100:.2f}% | AUC={auc_v:.4f} | Acc={acc_3s_v*100:.2f}% | Retake={retake_v*100:.2f}% (N={len(y_v)})")
        
    # 6. 운동 유형별 (포워드 vs 백워드) 세부 감사
    print("\n[*] 6. 운동 유형별 (05_스텝_포워드 vs 06_스텝_백워드) 세부 감사:")
    exercise_performance = {}
    for ex_code in ['05_forward', '06_backward']:
        ex_mask = (exercise_types == ex_code)
        y_ex = y[ex_mask]
        p_ex = champ_oof_prob[ex_mask]
        pred_ex = champ_pred_3s[ex_mask]
        
        auc_ex = roc_auc_score(y_ex, p_ex)
        f1_2s_ex = f1_score(y_ex, (p_ex >= 0.5).astype(int), zero_division=0)
        cov_ex = (pred_ex != 2)
        f1_3s_ex = f1_score(y_ex[cov_ex], pred_ex[cov_ex], zero_division=0) if np.sum(cov_ex) > 0 else 0.0
        acc_3s_ex = accuracy_score(y_ex[cov_ex], pred_ex[cov_ex]) if np.sum(cov_ex) > 0 else 0.0
        retake_ex = 1.0 - np.mean(cov_ex)
        
        ex_label = "05_스텝_포워드" if ex_code == '05_forward' else "06_스텝_백워드"
        exercise_performance[ex_code] = {
            'label': ex_label,
            'count': int(np.sum(ex_mask)),
            'auc': float(auc_ex),
            'f1_2state': float(f1_2s_ex),
            'f1_3state': float(f1_3s_ex),
            'acc_3state': float(acc_3s_ex),
            'retake': float(retake_ex)
        }
        print(f"  - {ex_label:18s}: 3-State F1={f1_3s_ex*100:.2f}% | 2-State F1={f1_2s_ex*100:.2f}% | AUC={auc_ex:.4f} | Retake={retake_ex*100:.2f}% (N={len(y_ex)})")
        
    # 7. 피험자 9명별 세부 성능 감사
    print("\n[*] 7. 피험자 1~9번별 세부 성능 감사:")
    subject_performance = {}
    for s in range(1, 10):
        s_mask = (subjects == s)
        y_s = y[s_mask]
        p_s = champ_oof_prob[s_mask]
        pred_s = champ_pred_3s[s_mask]
        
        auc_s = roc_auc_score(y_s, p_s) if len(np.unique(y_s)) > 1 else 0.0
        f1_2s_s = f1_score(y_s, (p_s >= 0.5).astype(int), zero_division=0)
        cov_s = (pred_s != 2)
        f1_3s_s = f1_score(y_s[cov_s], pred_s[cov_s], zero_division=0) if np.sum(cov_s) > 0 else 0.0
        acc_3s_s = accuracy_score(y_s[cov_s], pred_s[cov_s]) if np.sum(cov_s) > 0 else 0.0
        
        subject_performance[f"피험자{s}"] = {
            'count': int(len(y_s)),
            'defect_count': int(np.sum(y_s == 1)),
            'auc': float(auc_s),
            'f1_2state': float(f1_2s_s),
            'f1_3state': float(f1_3s_s),
            'acc_3state': float(acc_3s_s),
            'retake': float(1.0 - np.mean(cov_s))
        }
        print(f"  - 피험자 {s}: 3-State F1={f1_3s_s*100:.2f}% | 2-State F1={f1_2s_s*100:.2f}% | AUC={auc_s:.4f} | Acc={acc_3s_s*100:.2f}% (N={len(y_s)})")
        
    # 8. 최종 프로덕션 모델 학습 및 가중치 번들 저장
    print("\n" + "=" * 85)
    print("  [*] 8. 최종 프로덕션 dl3_rtmp_1 모델 전수 학습 및 가중치 번들 패키징")
    print("=" * 85)
    
    # 챔피언 모델 유형에 맞추어 모델 학습
    # 27D 피처셋을 사용하는 앙상블 또는 CatBoost
    target_feature_set = X_27 if '27D' in best_model_name else (X_22 if '22D' in best_model_name else X_14)
    target_feat_names = FEATURE_NAMES_27D if '27D' in best_model_name else (FEATURE_NAMES_22D if '22D' in best_model_name else FEATURE_NAMES_14D)
    
    print(f"[*] 최종 모델 학습 피처셋: {target_feature_set.shape[1]}D")
    
    # 전수 데이터에 대한 CatBoost 학습
    final_cb = CatBoostClassifier(**cb_params)
    final_cb.fit(target_feature_set, y)
    
    # 전수 데이터에 대한 LightGBM 학습
    final_lgb = lgb.LGBMClassifier(**lgb_params)
    final_lgb.fit(target_feature_set, y)
    
    # 전수 데이터에 대한 XGBoost 학습
    final_xgb = xgb.XGBClassifier(**xgb_params)
    final_xgb.fit(target_feature_set, y)
    
    # CatBoost 전용 모델 파일 저장
    cbm_save_path = r"D:\src\models\second_models\dynamic_lunge\dl3_rtmp_1.cbm"
    final_cb.save_model(cbm_save_path)
    print(f"[+] dl3_rtmp_1.cbm 단독 모델 저장 완료: {cbm_save_path}")
    
    # 번들 객체 구성
    bundle_data = {
        'model_name': 'dl3_rtmp_1',
        'exercise_name': '다이나믹 런지 (Dynamic Lunge)',
        'defect_name': 'Condition 3: 뒤다리 무릎 각도 90도 (Rear Knee 90°)',
        'defect_code': 'dl3',
        'version': 'dl3_rtmp_1',
        'input_type': 'RTMPose Halpe26 2D Keypoints (64, 26, 3)',
        'feature_dim': len(target_feat_names),
        'feature_names': target_feat_names,
        'cb_model': final_cb,
        'lgb_model': final_lgb,
        'xgb_model': final_xgb,
        'ensemble_weights': [0.40, 0.35, 0.25], # CB, LGB, XGB
        'optimal_tau': float((tau_clean + tau_defect) / 2.0),
        'tau_clean': float(tau_clean),
        'tau_defect': float(tau_defect),
        'tournament_results': tournament_results,
        'view_performance': view_performance,
        'exercise_performance': exercise_performance,
        'subject_performance': subject_performance,
        'overall_metrics': best_info,
        'total_samples': int(N),
        'creation_date': '2026-10-11'
    }
    
    bundle_save_path = os.path.join(WEIGHTS_DIR, "dl3_rtmp_1_bundle.pkl")
    with open(bundle_save_path, 'wb') as fp:
        pickle.dump(bundle_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] dl3_rtmp_1 프로덕션 번들 저장 완료: {bundle_save_path}")
    
    # 결과 JSON 저장
    json_summary_path = os.path.join(RESULTS_DIR, "dl3_rtmp_1_optimization_results.json")
    with open(json_summary_path, 'w', encoding='utf-8') as fp:
        # 직렬화 가능하도록 정리
        clean_json_data = {
            'model_name': 'dl3_rtmp_1',
            'champion_config': best_model_name,
            'tournament_results': tournament_results,
            'view_performance': view_performance,
            'exercise_performance': exercise_performance,
            'subject_performance': subject_performance,
            'overall_metrics': best_info,
            'elapsed_sec': float(time.time() - t_start)
        }
        json.dump(clean_json_data, fp, indent=2, ensure_ascii=False)
    print(f"[+] 결과 요약 JSON 저장 완료: {json_summary_path}")
    print(f"[+] 전체 파이프라인 소요 시간: {time.time() - t_start:.2f} 초")

if __name__ == '__main__':
    main()

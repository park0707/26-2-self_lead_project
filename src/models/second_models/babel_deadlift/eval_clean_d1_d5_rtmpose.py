# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: src/models/second_models/babel_deadlift/eval_clean_d1_d5_rtmpose.py
Purpose: 전처리 통일 완료된 클린 RTMPose 770개 마스터 데이터셋 기반
         d1 ~ d5 세부 모델 5-Fold 피험자 완전 격리 원천 3-State F1 및 보류율 전수 산출
"""

import os
import sys
import time
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = r'D:/'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.models.second_models.babel_deadlift.run_deadlift_mediapipe_vs_rtmpose_benchmark import (
    halpe26_to_mediapipe33, evaluate_metrics,
    DEFECT_INFO, CACHE_META, RTM_DIR
)
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import extract_d1_v4_3_diet_features_25d
from src.models.second_models.babel_deadlift.d2_model_v5_final import extract_d2_v5_final_features
from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq
from src.models.second_models.babel_deadlift.d4_model_v5 import extract_d4_v5_features_from_seq, DIET_CORE_INDICES_15_D4
from src.models.second_models.babel_deadlift.d5_model_v5_1 import extract_d5_v4_gated_features_from_seq, KEPT_FEATURE_INDICES_D5

def main():
    print("=" * 90, flush=True)
    print("  [vidio-health-trainer] 클린 RTMPose d1~d5 원천 3-State F1 및 보류율 전수 산출", flush=True)
    print("=" * 90, flush=True)

    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    oof_mp_all = meta['oof_probs']
    N = len(clip_names)

    print(f"[*] 770개 마스터 시퀀스에서 RTMPose 피처 추출 중...", flush=True)
    t0 = time.time()
    feats_d1, feats_d2, feats_d3, feats_d4, feats_d5 = [], [], [], [], []

    for cname in clip_names:
        rtm_p = os.path.join(RTM_DIR, cname + '.npz')
        data = np.load(rtm_p)
        kps_26 = data['kps_26_64f']
        kps_33 = halpe26_to_mediapipe33(kps_26)
        
        feats_d1.append(extract_d1_v4_3_diet_features_25d(kps_33))
        r2 = extract_d2_v5_final_features(kps_33)
        feats_d2.append(r2[0] if isinstance(r2, tuple) else r2)
        feats_d3.append(extract_d3_v5_3_opt_features_from_seq(kps_33))
        feats_d4.append(extract_d4_v5_features_from_seq(kps_33)[DIET_CORE_INDICES_15_D4])
        feats_d5.append(extract_d5_v4_gated_features_from_seq(kps_33)[KEPT_FEATURE_INDICES_D5])

    feats_dict = {
        'd1': np.nan_to_num(np.array(feats_d1, dtype=np.float32), nan=0.0),
        'd2': np.nan_to_num(np.array(feats_d2, dtype=np.float32), nan=0.0),
        'd3': np.nan_to_num(np.array(feats_d3, dtype=np.float32), nan=0.0),
        'd4': np.nan_to_num(np.array(feats_d4, dtype=np.float32), nan=0.0),
        'd5': np.nan_to_num(np.array(feats_d5, dtype=np.float32), nan=0.0),
    }
    print(f"[+] 피처 추출 완료 ({time.time()-t0:.2f}초)\n", flush=True)

    results = {}
    oof_probs_rtm = np.zeros((N, 5), dtype=np.float32)

    # 5대 시점별 성능도 함께 감사
    unique_views = ['view1', 'view2', 'view3', 'view4', 'view5']

    for d_code, d_desc, d_idx, feat_dim in DEFECT_INFO:
        t_sub = time.time()
        X = feats_dict[d_code]
        y = y_golden[:, d_idx]
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(sgkf.split(X, y, groups=subjects))
        oof_prob = np.zeros(N, dtype=np.float32)
        
        for fold_idx, (trn_idx, val_idx) in enumerate(folds):
            X_trn, y_trn = X[trn_idx], y[trn_idx]
            X_val = X[val_idx]
            
            cb = CatBoostClassifier(iterations=140, depth=5, learning_rate=0.05, auto_class_weights='Balanced', random_seed=42 + fold_idx, thread_count=2, verbose=False)
            cb.fit(X_trn, y_trn)
            p_cb = cb.predict_proba(X_val)[:, 1]
            
            clf = lgb.LGBMClassifier(n_estimators=110, max_depth=5, learning_rate=0.05, num_leaves=24, class_weight='balanced', random_state=42 + fold_idx, verbose=-1, n_jobs=2)
            clf.fit(X_trn, y_trn)
            p_lgb = clf.predict_proba(X_val)[:, 1]
            
            if d_code == 'd4':
                rf = RandomForestClassifier(n_estimators=60, max_depth=6, class_weight='balanced', random_state=42 + fold_idx, n_jobs=2)
                rf.fit(X_trn, y_trn)
                p_rf = rf.predict_proba(X_val)[:, 1]
                oof_prob[val_idx] = 0.45 * p_lgb + 0.40 * p_cb + 0.15 * p_rf
            else:
                oof_prob[val_idx] = 0.50 * p_cb + 0.50 * p_lgb
                
        oof_probs_rtm[:, d_idx] = oof_prob
        m_rtm = evaluate_metrics(y, oof_prob, threshold=0.50, tau_norm=0.40, tau_def=0.60)
        m_mp = evaluate_metrics(y, oof_mp_all[:, d_idx], threshold=0.50, tau_norm=0.40, tau_def=0.60)

        # 시점별 3-State F1 및 보류율 산출
        view_perf_rtm = {}
        view_perf_mp = {}
        for vn in unique_views:
            v_mask = (views == vn)
            vm_rtm = evaluate_metrics(y[v_mask], oof_prob[v_mask], threshold=0.50, tau_norm=0.40, tau_def=0.60)
            vm_mp = evaluate_metrics(y[v_mask], oof_mp_all[v_mask, d_idx], threshold=0.50, tau_norm=0.40, tau_def=0.60)
            view_perf_rtm[vn] = vm_rtm
            view_perf_mp[vn] = vm_mp

        results[d_code] = {
            'name': d_desc,
            'feat_dim': feat_dim,
            'rtmpose': m_rtm,
            'mediapipe': m_mp,
            'views_rtm': view_perf_rtm,
            'views_mp': view_perf_mp
        }

        print(f"[{d_code.upper()}: {d_desc[:12]}] (소요: {time.time()-t_sub:.1f}s)", flush=True)
        print(f"  - RTMPose   : 3-State F1={m_rtm['sel_f1']:5.2f}% | 보류율={m_rtm['retake']:5.2f}% | 강제F1={m_rtm['forced_f1']:5.2f}% | AUC={m_rtm['roc_auc']:5.2f}% | 정확도={m_rtm['sel_acc']:5.2f}%", flush=True)
        print(f"  - MediaPipe : 3-State F1={m_mp['sel_f1']:5.2f}% | 보류율={m_mp['retake']:5.2f}% | 강제F1={m_mp['forced_f1']:5.2f}% | AUC={m_mp['roc_auc']:5.2f}% | 정확도={m_mp['sel_acc']:5.2f}%", flush=True)
        diff_f1 = m_rtm['sel_f1'] - m_mp['sel_f1']
        diff_retake = m_rtm['retake'] - m_mp['retake']
        print(f"  - 3-State F1 향상: {'+' if diff_f1>=0 else ''}{diff_f1:.2f}%p | 보류율 변화: {'+' if diff_retake>=0 else ''}{diff_retake:.2f}%p\n", flush=True)

    # 4. 통합 clean benchmark json 저장
    out_json = r'D:\src\models\second_models\babel_deadlift\deadlift_rtmpose_clean_benchmark_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. OOF 확률도 캐시 저장
    np.savez_compressed(
        r'D:\src\models\second_models\babel_deadlift\deadlift_rtmpose_clean_oof_probs.npz',
        oof_probs_rtm=oof_probs_rtm,
        clip_names=clip_names,
        subjects=subjects,
        views=views,
        y=y_golden
    )
    print(f"[+] 최종 클린 벤치마크 결과 저장 완료: {out_json}", flush=True)

if __name__ == '__main__':
    main()

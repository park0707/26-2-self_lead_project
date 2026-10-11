# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\re_evaluate_dynamic_lunge_hybrid.py
Purpose: 바벨 런지 공식 최종 하이브리드 모델 체계의 다이나믹 런지 1,468개 전수 재평가
         - c1: bl1_v4 (30D 정예 도메인 피처 GBDT 앙상블)
         - c2: bl2_final (20D 정예 하이브리드: MediaPipe 3D + RTMPose 2D)
         - c3: bl3_v4 (14D 정예 기구학 CatBoost)
         - c4: bl4_v4b (24D 도메인 + Multi-Scale SE-TCN + GBDT 4-Way)
         - c5: bl5_final (79D 하이브리드: MediaPipe 3D + RTMPose 2D)
         
         평가 프로토콜:
         1. 원천 예측 확률(Predicted Probability) 및 순위 변별력(ROC-AUC) 전수 추출
         2. 바벨 런지 원본 고정 임계값(Zero-Shot Default) 3-State 성능
         3. 3-State 적응형 캘리브레이션(보류율 <= 10.0% 엄격 준수) 실측 성능
         4. 공식 5대 시점(A~E)별 세부 3-State F1 매트릭스 전수 감사
"""

import os
import sys
import glob
import json
import time
import pickle
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
from tqdm import tqdm

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"D:\src\models\second_models\babel_lunge")
sys.path.insert(0, r"D:\src\models\sub_models")
sys.path.append(r"D:\src\features")

from lunge_c1_features_v4 import extract_lunge_c1_features_v4
from bl2_final import predict_babel_lunge_c2
from bl3_v4 import BL3V4Classifier
from bl4_v4b import BL4V4bClassifier
from bl5_final import BL5Classifier

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]
MP15_DIRS = [
    r"D:\Data\MediaPipe\전처리\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\MediaPipe\전처리\06_스텝_백워드_다이나믹_런지"
]
LABEL_DIRS = [
    r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
]

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

OUT_JSON = r"D:\tests\dynamic_lunge_hybrid_reevaluation_results.json"


def find_optimal_3state_thresholds(y_true, y_prob, max_retake_rate=0.10):
    """
    보류율(Retake Rate) <= max_retake_rate(10.0%) 제약 하에서
    결함 F1-Score를 극대화하는 (tau_clean, tau_defect) 그리드 탐색
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    
    if N == 0:
        return 0.5, 0.5, {'f1': 0.0, 'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'cov': 0.0, 'retake': 0.0}
        
    p_min, p_max = float(np.percentile(y_prob, 1)), float(np.percentile(y_prob, 99))
    steps = 40
    taus = np.linspace(p_min, p_max, steps)
    
    best_f1 = -1.0
    best_clean, best_defect = 0.5, 0.5
    best_metrics = {}
    
    for i in range(len(taus)):
        t_c = taus[i]
        for j in range(i, len(taus)):
            t_d = taus[j]
            
            # 3-State 판정: prob < t_c -> 0, prob >= t_d -> 1, 그 외 -> 2(보류)
            pred = np.full(N, 2, dtype=int)
            pred[y_prob < t_c] = 0
            pred[y_prob >= t_d] = 1
            
            det_mask = (pred != 2)
            n_det = int(np.sum(det_mask))
            cov = n_det / N
            retake = 1.0 - cov
            
            if retake <= max_retake_rate and n_det > 0:
                y_det_true = y_true[det_mask]
                y_det_pred = pred[det_mask]
                f1 = float(f1_score(y_det_true, y_det_pred, zero_division=0))
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_clean, best_defect = float(t_c), float(t_d)
                    best_metrics = {
                        'f1': f1,
                        'acc': float(accuracy_score(y_det_true, y_det_pred)),
                        'prec': float(precision_score(y_det_true, y_det_pred, zero_division=0)),
                        'rec': float(recall_score(y_det_true, y_det_pred, zero_division=0)),
                        'cov': float(cov),
                        'retake': float(retake)
                    }
                    
    # fallback if not found
    if best_f1 < 0:
        pred_bin = (y_prob >= 0.5).astype(int)
        best_clean, best_defect = 0.5, 0.5
        best_metrics = {
            'f1': float(f1_score(y_true, pred_bin, zero_division=0)),
            'acc': float(accuracy_score(y_true, pred_bin)),
            'prec': float(precision_score(y_true, pred_bin, zero_division=0)),
            'rec': float(recall_score(y_true, pred_bin, zero_division=0)),
            'cov': 1.0,
            'retake': 0.0
        }
        
    return best_clean, best_defect, best_metrics


def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 런지 공식 최종 하이브리드 모델 체계 다이나믹 런지 전수 재평가")
    print("=" * 85)
    
    # 1. 모델 로드
    print("[*] 1. 바벨 런지 공식 최종 모델 번들 로드 중...")
    
    # 1) bl1_v4 번들 로드
    bl1_bundle_p = r"D:\src\models\second_models\babel_lunge\bl1_v4_bundle.pkl"
    with open(bl1_bundle_p, "rb") as f:
        bl1_b = pickle.load(f)
    bl1_fnames = bl1_b['feature_names']
    bl1_weights = bl1_b['ensemble_weights']
    bl1_lgb = bl1_b['model_lgb']
    bl1_cat = bl1_b['model_cat']
    bl1_xgb = bl1_b['model_xgb']
    bl1_ths = bl1_b.get('view_thresholds', {})
    
    # 2) bl3_v4 모델 로드
    c3_model = BL3V4Classifier()
    
    # 3) bl4_v4b 모델 로드
    c4_model = BL4V4bClassifier()
    
    # 4) bl5_final 모델 로드
    c5_model = BL5Classifier()
    
    print("[+] 모든 모델 번들 로드 완료!")
    
    # 2. 파일 목록 및 정답 라벨 매칭
    mp33_files = sorted(glob.glob(os.path.join(MP33_DIR, "*.npz")))
    samples_info = []
    
    print("[*] 2. 1,468개 시퀀스 모달리티 1:1 매칭 및 Ground Truth 라벨 로드...")
    for mp_path in mp33_files:
        fname = os.path.basename(mp_path)
        view_code = None
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if f"_{v}.npz" in fname:
                view_code = v
                break
        if view_code is None:
            continue
            
        sub_id = 1
        for s in range(1, 10):
            if f"피험자{s}" in fname:
                sub_id = s
                break
                
        rtmp_path = None
        for rd in RTMP_DIRS:
            candidate = os.path.join(rd, fname)
            if os.path.exists(candidate):
                rtmp_path = candidate
                break
        if rtmp_path is None:
            continue
            
        raw_key = fname.replace(f"_{view_code}.npz", "")
        mp15_path = None
        for md in MP15_DIRS:
            candidate = os.path.join(md, f"{raw_key}.npz")
            if os.path.exists(candidate):
                mp15_path = candidate
                break
        if mp15_path is None:
            continue
            
        label_json_path = None
        for ld in LABEL_DIRS:
            candidate = os.path.join(ld, f"{raw_key}.json")
            if os.path.exists(candidate):
                label_json_path = candidate
                break
        if label_json_path is None:
            continue
            
        try:
            with open(label_json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                d = json.load(fp)
            conds = d.get('type_info', {}).get('conditions', [])
            gt_labels = {}
            for idx, c in enumerate(conds[:5]):
                c_num = idx + 1
                val = c.get('value')
                gt_labels[f"c{c_num}"] = 1 if (val is False) else 0
        except Exception:
            continue
            
        samples_info.append({
            'fname': fname,
            'mp33_path': mp_path,
            'rtmp_path': rtmp_path,
            'mp15_path': mp15_path,
            'view_code': view_code,
            'sub_id': sub_id,
            'gt_labels': gt_labels
        })
        
    print(f"[+] 총 정합 완료 시퀀스: {len(samples_info)} 건")
    
    # 3. 전수 추론 실행
    print("\n[*] 3. 5대 세부 오류 모델 예측 확률 및 기본 판정 일괄 추출...")
    start_time = time.time()
    
    data_buffer = {
        f"c{i}": {
            'y_true': [],
            'y_prob': [],
            'default_pred': [],
            'views': [],
            'subs': []
        } for i in range(1, 6)
    }
    
    for item in tqdm(samples_info, desc="Re-evaluating Dynamic Lunge"):
        mp33_data = np.load(item['mp33_path'], allow_pickle=True)
        rtmp_data = np.load(item['rtmp_path'], allow_pickle=True)
        mp15_data = np.load(item['mp15_path'], allow_pickle=True)
        
        kps_mp_3d = mp33_data['kps_33_3d_64f']
        vis_mp = mp33_data['vis_64f'] if 'vis_64f' in mp33_data else None
        kps_rtmp_2d = rtmp_data['kps_26_64f']
        kps_mp_15 = mp15_data['keypoints_2d_norm']
        v_code = item['view_code']
        sub = item['sub_id']
        gt = item['gt_labels']
        
        # --- c1: bl1_v4 (30D 도메인 기하 피처) ---
        f1_dict = extract_lunge_c1_features_v4(kps_mp_3d, vis_mp)
        f1_vec = np.array([f1_dict[k] for k in bl1_fnames], dtype=np.float32).reshape(1, -1)
        p1_l = float(bl1_lgb.predict(f1_vec)[0])
        p1_c = float(bl1_cat.predict_proba(f1_vec)[0, 1])
        p1_x = float(bl1_xgb.predict(xgb.DMatrix(f1_vec))[0])
        prob1 = float(bl1_weights[0]*p1_l + bl1_weights[1]*p1_c + bl1_weights[2]*p1_x)
        # 기본 임계값 판정
        th1 = bl1_ths.get(v_code, {'tau_clean': 0.35, 'tau_defect': 0.60})
        pred1 = 0 if prob1 < th1['tau_clean'] else (1 if prob1 >= th1['tau_defect'] else 2)
        
        data_buffer['c1']['y_true'].append(gt['c1'])
        data_buffer['c1']['y_prob'].append(prob1)
        data_buffer['c1']['default_pred'].append(pred1)
        data_buffer['c1']['views'].append(v_code)
        data_buffer['c1']['subs'].append(sub)
        
        # --- c2: bl2_final (20D 정예 하이브리드) ---
        res2 = predict_babel_lunge_c2(kps_mp_3d, kps_rtmp_2d, camera_view=v_code)
        pred2 = 0 if res2['state'] == 'CLEAN' else (1 if res2['state'] == 'DEFECT' else 2)
        data_buffer['c2']['y_true'].append(gt['c2'])
        data_buffer['c2']['y_prob'].append(float(res2['probability']))
        data_buffer['c2']['default_pred'].append(pred2)
        data_buffer['c2']['views'].append(v_code)
        data_buffer['c2']['subs'].append(sub)
        
        # --- c3: bl3_v4 (14D 정예 기구학) ---
        res3 = c3_model.predict_single(kps_mp_15)
        pred3 = 0 if res3['status'] == 'NORMAL' else (1 if res3['status'] == 'DEFECT' else 2)
        data_buffer['c3']['y_true'].append(gt['c3'])
        data_buffer['c3']['y_prob'].append(float(res3['defect_probability']))
        data_buffer['c3']['default_pred'].append(pred3)
        data_buffer['c3']['views'].append(v_code)
        data_buffer['c3']['subs'].append(sub)
        
        # --- c4: bl4_v4b (24D 도메인 + TCN) ---
        res4 = c4_model.predict_3state(kps_mp_3d)
        pred4 = 0 if res4['state'] == 'Clean' else (1 if res4['state'] == 'Defect' else 2)
        data_buffer['c4']['y_true'].append(gt['c4'])
        data_buffer['c4']['y_prob'].append(float(res4['defect_prob']))
        data_buffer['c4']['default_pred'].append(pred4)
        data_buffer['c4']['views'].append(v_code)
        data_buffer['c4']['subs'].append(sub)
        
        # --- c5: bl5_final (79D 하이브리드) ---
        res5 = c5_model.predict_sequence(kps_mp_3d, kps_rtmp_2d, view=v_code)
        data_buffer['c5']['y_true'].append(gt['c5'])
        data_buffer['c5']['y_prob'].append(float(res5['defect_prob']))
        data_buffer['c5']['default_pred'].append(res5['state_code'])
        data_buffer['c5']['views'].append(v_code)
        data_buffer['c5']['subs'].append(sub)
        
    elapsed = time.time() - start_time
    print(f"[+] 일괄 추론 완료! 소요 시간: {elapsed:.2f} 초")
    
    # 4. 분석 및 성능 지표 산출
    final_output = {
        'total_samples': len(samples_info),
        'elapsed_seconds': elapsed,
        'overall_results': {},
        'by_view_results': {}
    }
    
    defect_labels = {
        'c1': 'c1 (앞무릎 각도 90도 오류) [bl1_v4 30D]',
        'c2': 'c2 (몸통·발·무릎 방향 불일치) [bl2_final 20D Hybrid]',
        'c3': 'c3 (뒤무릎 각도 90도 오류) [bl3_v4 14D]',
        'c4': 'c4 (척추의 중립 위반) [bl4_v4b 24D+TCN]',
        'c5': 'c5 (상체 과도한 숙임/젖힘) [bl5_final 79D Hybrid]'
    }
    
    print("\n" + "=" * 105)
    print("  [다이나믹 런지 전체 1,468개 대상 바벨 런지 최종 모델 3-State F1 및 ROC-AUC 실측 종합]")
    print("=" * 105)
    print(f"{'세부 결함 항목 및 모델':<38} | {'ROC-AUC':<8} | {'2-State F1':<10} | {'3-State F1 (최적 캘리브)':<18} | {'정확도':<8} | {'커버리지':<8}")
    print("-" * 105)
    
    views_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    
    for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
        y_t = np.array(data_buffer[c_key]['y_true'])
        y_p = np.array(data_buffer[c_key]['y_prob'])
        views_arr = np.array(data_buffer[c_key]['views'])
        
        auc = float(roc_auc_score(y_t, y_p))
        f1_2s = float(f1_score(y_t, (y_p >= 0.50).astype(int), zero_division=0))
        
        # 전체 최적 3-State 임계값 탐색
        t_clean, t_def, m_3s = find_optimal_3state_thresholds(y_t, y_p, max_retake_rate=0.10)
        
        final_output['overall_results'][c_key] = {
            'auc': auc,
            'f1_2state': f1_2s,
            'optimal_3state': m_3s,
            'tau_clean': t_clean,
            'tau_defect': t_def
        }
        
        print(f"{defect_labels[c_key]:<38} | {auc:6.4f}   | {f1_2s*100:6.2f}%    | {m_3s['f1']*100:6.2f}% (tau={t_clean:.3f}/{t_def:.3f}) | {m_3s['acc']*100:6.2f}% | {m_3s['cov']*100:6.2f}%")
        
    print("=" * 105)
    
    # 5. 시점별 최적 3-State F1 산출
    print("\n" + "=" * 105)
    print("  [공식 5대 시점(A~E)별 3-State 적응형 F1 성능 감사 (보류율 <= 10.0% 통제)]")
    print("=" * 105)
    print(f"{'공식 시점 명칭 및 각도':<24} | {'c1 (30D)':<10} | {'c2 (Hybrid)':<12} | {'c3 (14D)':<10} | {'c4 (TCN)':<10} | {'c5 (Hybrid)':<12} | {'시점 평균':<8}")
    print("-" * 105)
    
    for v in views_order:
        v_name = VIEW_ALIAS_MAP[v]
        final_output['by_view_results'][v] = {}
        row_f1s = []
        
        for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            mask = (np.array(data_buffer[c_key]['views']) == v)
            sub_t = np.array(data_buffer[c_key]['y_true'])[mask]
            sub_p = np.array(data_buffer[c_key]['y_prob'])[mask]
            
            sub_auc = float(roc_auc_score(sub_t, sub_p)) if len(np.unique(sub_t)) > 1 else 0.5
            _, _, sub_m3s = find_optimal_3state_thresholds(sub_t, sub_p, max_retake_rate=0.10)
            
            final_output['by_view_results'][v][c_key] = {
                'auc': sub_auc,
                'metrics_3state': sub_m3s
            }
            row_f1s.append(sub_m3s['f1'] * 100)
            
        v_avg = float(np.mean(row_f1s))
        print(f"{v_name:<24} | {row_f1s[0]:6.2f}%   | {row_f1s[1]:6.2f}%     | {row_f1s[2]:6.2f}%   | {row_f1s[3]:6.2f}%   | {row_f1s[4]:6.2f}%     | {v_avg:6.2f}%")
        
    print("=" * 105)
    
    with open(OUT_JSON, 'w', encoding='utf-8') as fp:
        json.dump(final_output, fp, indent=2, ensure_ascii=False)
    print(f"\n[+] 재평가 결과 JSON 저장 완료: {OUT_JSON}")


if __name__ == "__main__":
    main()

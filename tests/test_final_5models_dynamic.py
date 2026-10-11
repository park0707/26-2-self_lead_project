# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\tests\test_final_5models_dynamic.py
Purpose: 바벨 런지 공식 확정 5대 모델 체계 다이나믹 런지 1,468개 전수 평가
         - c1: bl1_rtmp_1.cbm (RTMPose 2D 30D CatBoost)
         - c2: bl2_final_bundle.pkl (Hybrid 20D: MediaPipe 3D + RTMPose 2D)
         - c3: bl3_v4_champion_bundle.joblib (MediaPipe 2D 14D CatBoost)
         - c4: bl4_rtmp_1.cbm (RTMPose 2D 24D CatBoost)
         - c5: bl5_final_bundle.pkl (Hybrid 79D: MediaPipe 3D + RTMPose 2D)
"""

import os
import sys
import glob
import json
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"D:\src\models\second_models\babel_lunge")
sys.path.insert(0, r"D:\src\models\sub_models")
sys.path.append(r"D:\src\features")

from lunge_c1_features_v4 import extract_lunge_c1_feature_vector_v4
from bl2_final import predict_babel_lunge_c2
from bl3_v4 import BL3V4Classifier
from lunge_c4_features_v3 import extract_lunge_c4_features_v3
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

def map_rtmp26_to_mp33_with_fake_z(kps26):
    """
    RTMPose Halpe26 2D -> MediaPipe 33 format with fake Z depth
    (extract_rtmp_fair.py 동일 규격)
    """
    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    kps33[:, 0] = kps26[:, 0]
    kps33[:, 2] = kps26[:, 1]
    kps33[:, 5] = kps26[:, 2]
    kps33[:, 7] = kps26[:, 3]
    kps33[:, 8] = kps26[:, 4]
    kps33[:, 11] = kps26[:, 5]
    kps33[:, 12] = kps26[:, 6]
    kps33[:, 13] = kps26[:, 7]
    kps33[:, 14] = kps26[:, 8]
    kps33[:, 15] = kps26[:, 9]
    kps33[:, 16] = kps26[:, 10]
    kps33[:, 23] = kps26[:, 11]
    kps33[:, 24] = kps26[:, 12]
    kps33[:, 25] = kps26[:, 13]
    kps33[:, 26] = kps26[:, 14]
    kps33[:, 27] = kps26[:, 15]
    kps33[:, 28] = kps26[:, 16]
    kps33[:, 29] = kps26[:, 24] # LEFT_HEEL
    kps33[:, 30] = kps26[:, 25] # RIGHT_HEEL
    kps33[:, 31] = kps26[:, 20] # LEFT_FOOT_INDEX
    kps33[:, 32] = kps26[:, 21] # RIGHT_FOOT_INDEX
    
    # Fake Z depth injection
    left_mp_indices = [1,2,3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4,5,6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33

def find_optimal_3state_thresholds(y_true, y_prob, max_retake_rate=0.15):
    """
    3-State 최적 임계값 탐색 (보류율 max_retake_rate 이하 조건)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    if N == 0:
        return 0.5, 0.5, {'f1': 0.0, 'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'cov': 0.0, 'retake': 0.0}
    
    p_min, p_max = float(np.percentile(y_prob, 1)), float(np.percentile(y_prob, 99))
    taus = np.linspace(p_min, p_max, 40)
    best_f1 = -1.0
    best_clean, best_defect = 0.5, 0.5
    best_metrics = {}
    
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
                    best_clean, best_defect = t_c, t_d
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
    print("  [vidio-health-trainer] 바벨 런지 공식 확정 5대 모델 체계 다이나믹 런지 전수 평가")
    print("=" * 85)
    
    # 1. 모델 로드
    print("[*] 1. 모델 로드 중...")
    
    # c1: bl1_rtmp_1.cbm
    model_c1 = CatBoostClassifier()
    model_c1.load_model(r"D:\src\models\second_models\babel_lunge\bl1_rtmp_1.cbm")
    
    # c2: bl2_final
    # predict_babel_lunge_c2 사용
    
    # c3: bl3_v4
    model_c3 = BL3V4Classifier()
    
    # c4: bl4_rtmp_1.cbm
    model_c4 = CatBoostClassifier()
    model_c4.load_model(r"D:\src\models\second_models\babel_lunge\bl4_rtmp_1.cbm")
    
    # c5: bl5_final
    model_c5 = BL5Classifier()
    
    print("[+] 모든 모델 로드 완료!")
    
    # 2. 데이터 매칭
    mp33_files = sorted(glob.glob(os.path.join(MP33_DIR, "*.npz")))
    samples_info = []
    
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
            
        exercise_type = "05_스텝_포워드" if "05_" in rtmp_path or "포워드" in rtmp_path else "06_스텝_백워드"
            
        samples_info.append({
            'fname': fname,
            'exercise_type': exercise_type,
            'mp33_path': mp_path,
            'rtmp_path': rtmp_path,
            'mp15_path': mp15_path,
            'view_code': view_code,
            'sub_id': sub_id,
            'gt_labels': gt_labels
        })
        
    print(f"[+] 총 매칭 시퀀스 수: {len(samples_info)} 건")
    
    # 3. 전수 추론 실행
    data_buffer = {
        f"c{i}": {
            'y_true': [],
            'y_prob': [],
            'default_pred': [],
            'views': [],
            'subs': [],
            'exercise_types': []
        } for i in range(1, 6)
    }
    
    # 바벨 런지 원천 임계값
    # c1: [0.15, 0.75]
    # c2: [0.40, 0.50]
    # c3: [0.35, 0.50]
    # c4: [0.35, 0.45]
    # c5: [0.35, 0.60]
    
    start_t = time.time()
    for item in tqdm(samples_info, desc="Evaluating Dynamic Lunge"):
        mp33_data = np.load(item['mp33_path'], allow_pickle=True)
        rtmp_data = np.load(item['rtmp_path'], allow_pickle=True)
        mp15_data = np.load(item['mp15_path'], allow_pickle=True)
        
        kps_mp_3d = mp33_data['kps_33_3d_64f']
        kps_rtmp_2d = rtmp_data['kps_26_64f']
        kps_mp_15 = mp15_data['keypoints_2d_norm']
        v_code = item['view_code']
        sub = item['sub_id']
        gt = item['gt_labels']
        ex_type = item['exercise_type']
        
        # RTMP 26 -> MP 33 + Fake Z
        kps_rtmp_mp33 = map_rtmp26_to_mp33_with_fake_z(kps_rtmp_2d)
        
        # 1. c1: bl1_rtmp_1 (30D RTMP 2D)
        fc1 = extract_lunge_c1_feature_vector_v4(kps_rtmp_mp33)
        prob1 = float(model_c1.predict_proba(fc1.reshape(1, -1))[0, 1])
        pred1 = 0 if prob1 < 0.15 else (1 if prob1 >= 0.75 else 2)
        data_buffer['c1']['y_true'].append(gt['c1'])
        data_buffer['c1']['y_prob'].append(prob1)
        data_buffer['c1']['default_pred'].append(pred1)
        data_buffer['c1']['views'].append(v_code)
        data_buffer['c1']['subs'].append(sub)
        data_buffer['c1']['exercise_types'].append(ex_type)
        
        # 2. c2: bl2_final (20D Hybrid: MP 3D + RTMP 2D)
        res2 = predict_babel_lunge_c2(kps_mp_3d, kps_rtmp_2d, camera_view=v_code)
        prob2 = float(res2['probability'])
        pred2 = 0 if res2['state'] == 'CLEAN' else (1 if res2['state'] == 'DEFECT' else 2)
        data_buffer['c2']['y_true'].append(gt['c2'])
        data_buffer['c2']['y_prob'].append(prob2)
        data_buffer['c2']['default_pred'].append(pred2)
        data_buffer['c2']['views'].append(v_code)
        data_buffer['c2']['subs'].append(sub)
        data_buffer['c2']['exercise_types'].append(ex_type)
        
        # 3. c3: bl3_v4 (14D MP 2D)
        res3 = model_c3.predict_single(kps_mp_15)
        prob3 = float(res3['defect_probability'])
        pred3 = 0 if res3['status'] == 'NORMAL' else (1 if res3['status'] == 'DEFECT' else 2)
        data_buffer['c3']['y_true'].append(gt['c3'])
        data_buffer['c3']['y_prob'].append(prob3)
        data_buffer['c3']['default_pred'].append(pred3)
        data_buffer['c3']['views'].append(v_code)
        data_buffer['c3']['subs'].append(sub)
        data_buffer['c3']['exercise_types'].append(ex_type)
        
        # 4. c4: bl4_rtmp_1 (24D RTMP 2D)
        fc4 = extract_lunge_c4_features_v3(kps_rtmp_mp33)
        prob4 = float(model_c4.predict_proba(fc4.reshape(1, -1))[0, 1])
        pred4 = 0 if prob4 < 0.35 else (1 if prob4 >= 0.45 else 2)
        data_buffer['c4']['y_true'].append(gt['c4'])
        data_buffer['c4']['y_prob'].append(prob4)
        data_buffer['c4']['default_pred'].append(pred4)
        data_buffer['c4']['views'].append(v_code)
        data_buffer['c4']['subs'].append(sub)
        data_buffer['c4']['exercise_types'].append(ex_type)
        
        # 5. c5: bl5_final (79D Hybrid: MP 3D + RTMP 2D)
        res5 = model_c5.predict_sequence(kps_mp_3d, kps_rtmp_2d, view=v_code)
        prob5 = float(res5['defect_prob'])
        pred5 = res5['state_code']
        data_buffer['c5']['y_true'].append(gt['c5'])
        data_buffer['c5']['y_prob'].append(prob5)
        data_buffer['c5']['default_pred'].append(pred5)
        data_buffer['c5']['views'].append(v_code)
        data_buffer['c5']['subs'].append(sub)
        data_buffer['c5']['exercise_types'].append(ex_type)
        
    print(f"\n[+] 전수 추론 완료! 소요 시간: {time.time() - start_t:.2f} 초")
    
    # 4. 결과 종합 분석
    results_summary = {}
    
    model_names = {
        'c1': 'bl1_rtmp_1 (30D RTMP 2D)',
        'c2': 'bl2_final (20D Hybrid)',
        'c3': 'bl3_v4 (14D MediaPipe 2D)',
        'c4': 'bl4_rtmp_1 (24D RTMP 2D)',
        'c5': 'bl5_final (79D Hybrid)',
    }
    
    print("\n" + "=" * 105)
    print("  [다이나믹 런지 1,468개 전수 바벨 런지 공식 5대 모델 3-State F1 및 ROC-AUC 실측 종합]")
    print("=" * 105)
    print(f"{'세부 결함 항목 및 모델':<35} | {'ROC-AUC':<8} | {'2-State F1':<10} | {'3-State F1 (최적 캘리브)':<26} | {'정확도':<8} | {'커버리지':<8}")
    print("-" * 105)
    
    for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
        y_true = np.array(data_buffer[c_key]['y_true'])
        y_prob = np.array(data_buffer[c_key]['y_prob'])
        views = np.array(data_buffer[c_key]['views'])
        ex_types = np.array(data_buffer[c_key]['exercise_types'])
        
        # 2-state at 0.5
        p_bin = (y_prob >= 0.5).astype(int)
        f1_2s = f1_score(y_true, p_bin, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5
            
        # 3-State 최적 캘리브레이션 (보류율 <= 15% 허용)
        tc, td, m3s = find_optimal_3state_thresholds(y_true, y_prob, max_retake_rate=0.15)
        
        results_summary[c_key] = {
            'model_name': model_names[c_key],
            'roc_auc': float(auc),
            'f1_2state': float(f1_2s),
            'f1_3state': float(m3s['f1']),
            'tau_clean': float(tc),
            'tau_defect': float(td),
            'accuracy': float(m3s['acc']),
            'precision': float(m3s['prec']),
            'recall': float(m3s['rec']),
            'coverage': float(m3s['cov']),
            'retake_rate': float(m3s['retake']),
            'per_view': {},
            'per_exercise': {}
        }
        
        print(f"{model_names[c_key]:<35} | {auc:.4f}   |  {f1_2s*100:6.2f}%   | {m3s['f1']*100:6.2f}% (tau={tc:.3f}/{td:.3f}) | {m3s['acc']*100:6.2f}% | {m3s['cov']*100:6.2f}%")
        
        # 시점별 분석
        for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
            mask = (views == v)
            if np.sum(mask) > 0:
                y_v = y_true[mask]
                p_v = y_prob[mask]
                tc_v, td_v, m_v = find_optimal_3state_thresholds(y_v, p_v, max_retake_rate=0.15)
                try:
                    auc_v = roc_auc_score(y_v, p_v)
                except Exception:
                    auc_v = 0.5
                results_summary[c_key]['per_view'][v] = {
                    'view_name': VIEW_ALIAS_MAP[v],
                    'count': int(np.sum(mask)),
                    'f1_3state': float(m_v['f1']),
                    'roc_auc': float(auc_v),
                    'tau_clean': float(tc_v),
                    'tau_defect': float(td_v),
                    'retake_rate': float(m_v['retake'])
                }
                
        # 운동 유형별 분석
        for et in ['05_스텝_포워드', '06_스텝_백워드']:
            mask = (ex_types == et)
            if np.sum(mask) > 0:
                y_e = y_true[mask]
                p_e = y_prob[mask]
                tc_e, td_e, m_e = find_optimal_3state_thresholds(y_e, p_e, max_retake_rate=0.15)
                try:
                    auc_e = roc_auc_score(y_e, p_e)
                except Exception:
                    auc_e = 0.5
                results_summary[c_key]['per_exercise'][et] = {
                    'count': int(np.sum(mask)),
                    'f1_3state': float(m_e['f1']),
                    'roc_auc': float(auc_e),
                    'retake_rate': float(m_e['retake'])
                }

    print("=" * 105)
    
    # 5대 시점 매트릭스 출력
    print("\n" + "=" * 105)
    print("  [공식 5대 시점(A~E)별 3-State 적응형 F1 성능 감사 (보류율 <= 15%)]")
    print("=" * 105)
    print(f"{'공식 시점 명칭 및 각도':<28} | {'c1 (RTMP)':<11} | {'c2 (Hybrid)':<11} | {'c3 (MP)':<11} | {'c4 (RTMP)':<11} | {'c5 (Hybrid)':<11} | {'시점 평균':<9}")
    print("-" * 105)
    
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = VIEW_ALIAS_MAP[v]
        f1_list = []
        c_strs = []
        for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            f1_v = results_summary[c_key]['per_view'][v]['f1_3state'] * 100
            f1_list.append(f1_v)
            c_strs.append(f"{f1_v:6.2f}%")
        avg_f1 = np.mean(f1_list)
        print(f"{v_name:<28} | {c_strs[0]:<11} | {c_strs[1]:<11} | {c_strs[2]:<11} | {c_strs[3]:<11} | {c_strs[4]:<11} | {avg_f1:6.2f}%")
    print("=" * 105)
    
    # JSON 저장
    out_json = r"D:\tests\dynamic_lunge_5models_official_results.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results_summary, fp, indent=2, ensure_ascii=False)
    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {out_json}")

if __name__ == '__main__':
    main()

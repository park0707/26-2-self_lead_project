# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\eval_babel_lunge_on_dynamic_lunge.py
Purpose: 바벨 런지 공식 5대 세부 오류 최종 모델(bl1~bl5)에 다이나믹 런지 1,468개 전수 데이터 적용 평가
         - c1: bl1_final (MediaPipe 33관절 3D, 28D 도메인 기하 GBDT 앙상블)
         - c2: bl2_final (MediaPipe 33 3D + RTMPose 26 2D, 20D 정예 하이브리드)
         - c3: bl3_v4 (MediaPipe 15 2D, 14D 정예 기구학 CatBoost)
         - c4: bl4_v4b (MediaPipe 33 3D, 24D + Multi-Scale SE-TCN + GBDT 4-Way)
         - c5: bl5_final (MediaPipe 33 3D + RTMPose 26 2D, 79D 하이브리드 3-Way 앙상블)
         - 공식 5대 시점(A~E) 및 피험자(1~9) 전수 다차원 성능 감사
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from tqdm import tqdm

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, r"D:\src\models\second_models\babel_lunge")
sys.path.insert(0, r"D:\src\models\sub_models")

from bl1_final import BL1Classifier
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

OUT_JSON = r"D:\tests\dynamic_lunge_evaluation_results.json"


def calc_3state_metrics(y_true, y_pred, y_prob):
    """
    y_pred: 0(정상), 1(결함), 2(보류)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    total = len(y_true)
    det_mask = (y_pred != 2) & (y_pred != -1)
    n_det = int(np.sum(det_mask))
    coverage = float(n_det / total) if total > 0 else 0.0
    retake_rate = 1.0 - coverage
    
    if n_det > 0:
        y_d_true = y_true[det_mask]
        y_d_pred = y_pred[det_mask]
        acc = float(accuracy_score(y_d_true, y_d_pred))
        prec = float(precision_score(y_d_true, y_d_pred, zero_division=0))
        rec = float(recall_score(y_d_true, y_d_pred, zero_division=0))
        f1 = float(f1_score(y_d_true, y_d_pred, zero_division=0))
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
        
    # 2-state metrics (threshold 0.50)
    y_2s_pred = (y_prob >= 0.50).astype(int)
    f1_2s = float(f1_score(y_true, y_2s_pred, zero_division=0))
    acc_2s = float(accuracy_score(y_true, y_2s_pred))
    
    return {
        'total': total,
        'n_det': n_det,
        'coverage': coverage,
        'retake_rate': retake_rate,
        'f1_3state': f1,
        'acc_3state': acc,
        'prec_3state': prec,
        'rec_3state': rec,
        'auc': auc,
        'f1_2state': f1_2s,
        'acc_2state': acc_2s
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 데이터셋 대상 바벨 런지 최종 모델 전이 평가")
    print("=" * 80)
    
    # 1. 파일 목록 수집
    mp33_files = sorted(glob.glob(os.path.join(MP33_DIR, "*.npz")))
    print(f"[*] MediaPipe 33관절 64F 파일 수: {len(mp33_files)}")
    
    # 2. 모델 인스턴스 초기화
    print("[*] 바벨 런지 공식 5대 세부 오류 최종 모델 로드 중...")
    c1_model = BL1Classifier()
    c3_model = BL3V4Classifier()
    c4_model = BL4V4bClassifier()
    c5_model = BL5Classifier()
    print("[+] 모든 모델 로드 완료!")
    
    # 데이터 수집 버퍼
    samples_info = []
    
    print("[*] 1,468개 시퀀스 메타데이터 및 정답 라벨 매칭 중...")
    for mp_path in mp33_files:
        fname = os.path.basename(mp_path) # e.g. [정상]_D01_피험자1_Key082_view1.npz
        
        # view 추출
        view_code = None
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if f"_{v}.npz" in fname:
                view_code = v
                break
        if view_code is None:
            continue
            
        # 피험자 번호 추출
        sub_id = 1
        for s in range(1, 10):
            if f"피험자{s}" in fname:
                sub_id = s
                break
                
        # rtmp 파일 경로
        rtmp_path = None
        for rd in RTMP_DIRS:
            candidate = os.path.join(rd, fname)
            if os.path.exists(candidate):
                rtmp_path = candidate
                break
        if rtmp_path is None:
            continue
            
        # mp15 파일 경로
        raw_key = fname.replace(f"_{view_code}.npz", "")
        mp15_path = None
        for md in MP15_DIRS:
            candidate = os.path.join(md, f"{raw_key}.npz")
            if os.path.exists(candidate):
                mp15_path = candidate
                break
        if mp15_path is None:
            continue
            
        # label json 경로
        label_json_path = None
        for ld in LABEL_DIRS:
            candidate = os.path.join(ld, f"{raw_key}.json")
            if os.path.exists(candidate):
                label_json_path = candidate
                break
        if label_json_path is None:
            continue
            
        # 정답 라벨 파싱
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
        
    print(f"[+] 최종 평가 대상 정합 시퀀스: {len(samples_info)} 건")
    
    # 3. 전수 추론 실행
    print("\n[*] 5대 세부 오류 모델 전수 추론 시작...")
    start_time = time.time()
    
    results = {
        f"c{i}": {
            'y_true': [],
            'y_pred': [],
            'y_prob': [],
            'views': [],
            'subs': []
        } for i in range(1, 6)
    }
    
    for item in tqdm(samples_info, desc="Evaluating Dynamic Lunge"):
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
        
        # --- c1 (MediaPipe 3D) ---
        res1 = c1_model.predict_sequence(kps_mp_3d, vis_mp)
        results['c1']['y_true'].append(gt['c1'])
        results['c1']['y_pred'].append(res1['state']) # 0: clean, 1: defect, 2: retain
        results['c1']['y_prob'].append(float(res1['defect_probability']))
        results['c1']['views'].append(v_code)
        results['c1']['subs'].append(sub)
        
        # --- c2 (Hybrid: MP 3D + RTMP 2D) ---
        res2 = predict_babel_lunge_c2(kps_mp_3d, kps_rtmp_2d, camera_view=v_code)
        # res2: state 'CLEAN', 'DEFECT', 'RETAKE'
        pred_c2 = 0 if res2['state'] == 'CLEAN' else (1 if res2['state'] == 'DEFECT' else 2)
        results['c2']['y_true'].append(gt['c2'])
        results['c2']['y_pred'].append(pred_c2)
        results['c2']['y_prob'].append(float(res2['probability']))
        results['c2']['views'].append(v_code)
        results['c2']['subs'].append(sub)
        
        # --- c3 (MediaPipe 15 2D) ---
        res3 = c3_model.predict_single(kps_mp_15)
        # res3: status 'NORMAL', 'DEFECT', 'UNDETERMINED'
        pred_c3 = 0 if res3['status'] == 'NORMAL' else (1 if res3['status'] == 'DEFECT' else 2)
        results['c3']['y_true'].append(gt['c3'])
        results['c3']['y_pred'].append(pred_c3)
        results['c3']['y_prob'].append(float(res3['defect_probability']))
        results['c3']['views'].append(v_code)
        results['c3']['subs'].append(sub)
        
        # --- c4 (MediaPipe 3D) ---
        res4 = c4_model.predict_3state(kps_mp_3d)
        # res4: state 'Clean', 'Defect', 'Retake'
        pred_c4 = 0 if res4['state'] == 'Clean' else (1 if res4['state'] == 'Defect' else 2)
        results['c4']['y_true'].append(gt['c4'])
        results['c4']['y_pred'].append(pred_c4)
        results['c4']['y_prob'].append(float(res4['defect_prob']))
        results['c4']['views'].append(v_code)
        results['c4']['subs'].append(sub)
        
        # --- c5 (Hybrid: MP 3D + RTMP 2D) ---
        res5 = c5_model.predict_sequence(kps_mp_3d, kps_rtmp_2d, view=v_code)
        # res5: state_code 0: normal, 1: defect, 2: retain
        results['c5']['y_true'].append(gt['c5'])
        results['c5']['y_pred'].append(res5['state_code'])
        results['c5']['y_prob'].append(float(res5['defect_prob']))
        results['c5']['views'].append(v_code)
        results['c5']['subs'].append(sub)
        
    elapsed = time.time() - start_time
    print(f"\n[+] 전수 추론 완료! 소요 시간: {elapsed:.2f} 초 ({elapsed / len(samples_info) * 1000:.2f} ms/시퀀스)")
    
    # 4. 성능 지표 집계 및 분석
    final_report = {
        'total_samples': len(samples_info),
        'elapsed_seconds': elapsed,
        'overall': {},
        'by_view': {},
        'by_subject': {}
    }
    
    defect_names = {
        'c1': 'c1 (앞무릎 90도 미달)',
        'c2': 'c2 (몸통·발·무릎 방향 불일치)',
        'c3': 'c3 (뒤무릎 90도 미달)',
        'c4': 'c4 (척추의 중립 위반)',
        'c5': 'c5 (상체 과도한 숙임/젖힘)'
    }
    
    print("\n" + "=" * 90)
    print("  [다이나믹 런지 전체 1,468개 시퀀스 대상 바벨 런지 최종 모델 종합 실측 성능]")
    print("=" * 90)
    print(f"{'세부 결함 항목':<28} | {'3-State F1':<10} | {'ROC-AUC':<10} | {'정확도':<10} | {'정밀도':<10} | {'재현율':<10} | {'커버리지':<10}")
    print("-" * 90)
    
    for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
        data = results[c_key]
        m = calc_3state_metrics(data['y_true'], data['y_pred'], data['y_prob'])
        final_report['overall'][c_key] = m
        print(f"{defect_names[c_key]:<28} | {m['f1_3state']*100:6.2f}%    | {m['auc']:6.4f}     | {m['acc_3state']*100:6.2f}%    | {m['prec_3state']*100:6.2f}%    | {m['rec_3state']*100:6.2f}%    | {m['coverage']*100:6.2f}%")
        
    # 종합 평균 계산
    avg_f1 = np.mean([final_report['overall'][k]['f1_3state'] for k in final_report['overall']])
    avg_auc = np.mean([final_report['overall'][k]['auc'] for k in final_report['overall']])
    avg_cov = np.mean([final_report['overall'][k]['coverage'] for k in final_report['overall']])
    print("-" * 90)
    print(f"{'전체 5대 결함 종합 평균':<28} | {avg_f1*100:6.2f}%    | {avg_auc:6.4f}     | {'-':<10} | {'-':<10} | {'-':<10} | {avg_cov*100:6.2f}%")
    print("=" * 90)
    
    # 5. 시점별 성능 집계
    views_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    for v in views_order:
        v_name = VIEW_ALIAS_MAP[v]
        final_report['by_view'][v] = {}
        for c_key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            v_mask = [v_item == v for v_item in results[c_key]['views']]
            sub_true = [t for t, m in zip(results[c_key]['y_true'], v_mask) if m]
            sub_pred = [p for p, m in zip(results[c_key]['y_pred'], v_mask) if m]
            sub_prob = [pb for pb, m in zip(results[c_key]['y_prob'], v_mask) if m]
            final_report['by_view'][v][c_key] = calc_3state_metrics(sub_true, sub_pred, sub_prob)
            
    print("\n" + "=" * 90)
    print("  [공식 5대 시점(A~E)별 3-State F1 성능 감사]")
    print("=" * 90)
    print(f"{'시점 명칭 및 각도':<24} | {'c1':<8} | {'c2':<8} | {'c3':<8} | {'c4':<8} | {'c5':<8} | {'시점 평균':<8}")
    print("-" * 90)
    for v in views_order:
        v_name = VIEW_ALIAS_MAP[v]
        v_m = final_report['by_view'][v]
        f1s = [v_m[k]['f1_3state'] * 100 for k in ['c1', 'c2', 'c3', 'c4', 'c5']]
        v_avg = np.mean(f1s)
        print(f"{v_name:<24} | {f1s[0]:6.2f}% | {f1s[1]:6.2f}% | {f1s[2]:6.2f}% | {f1s[3]:6.2f}% | {f1s[4]:6.2f}% | {v_avg:6.2f}%")
    print("=" * 90)
    
    # 6. JSON 파일로 저장
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 상세 실측 결과가 저장되었습니다: {OUT_JSON}")


if __name__ == "__main__":
    main()

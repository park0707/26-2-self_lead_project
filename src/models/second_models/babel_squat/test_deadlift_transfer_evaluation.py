# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_deadlift_transfer_evaluation.py
Purpose: 바벨 스쿼트 3대 결함 모델(척추 c1, 무릎 c3, 발바닥 c4)의 
         바벨 데드리프트(02_바벨_데드리프트) 제로샷 전이(Transfer/Zero-Shot) 실측 평가
         - 피험자 1번~9번 전원 전수 포함
         - 공식 5대 시점(A~E, view1~view5) 전수 평가
         - 스쿼트 원래 성능 대비 성능 유지/저하율 정밀 감사
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.adapters.mediapipe_adapter import MediaPipeAdapter
from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1

RAW_IMG_DIR = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"

def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
    T, J, C = kps_arr.shape
    if T == 64:
        return kps_arr.astype(np.float32)
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    flat_orig = kps_arr.reshape(T, -1)
    f = interp1d(t_orig, flat_orig, kind=kind, axis=0, fill_value='extrapolate')
    flat_64 = f(t_64)
    return flat_64.reshape(64, J, C).astype(np.float32)

def extract_single_folder(folder_name: str) -> Optional[List[Dict[str, Any]]]:
    seq_dir = os.path.join(RAW_IMG_DIR, folder_name)
    if not os.path.isdir(seq_dir):
        return None

    # 라벨 로드
    label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
    if not os.path.exists(label_p):
        return None

    with open(label_p, 'r', encoding='utf-8', errors='ignore') as f:
        meta_d = json.load(f)
    conds = meta_d.get('type_info', {}).get('conditions', [])
    cond_dict = {c['condition']: c['value'] for c in conds}

    # 정답 라벨 (오류 발생 시 1, 정상 시 0)
    y_spine = 0 if cond_dict.get('척추의 중립', True) else 1
    y_knee = 0 if cond_dict.get('발과 무릎의 방향 일치', True) else 1
    y_barbell = 0 if cond_dict.get('바벨 궤적과 몸 밀착', True) else 1
    y_sync = 0 if cond_dict.get('무릎과 골반이 동시에 펴짐', True) else 1
    y_foot = 0 if cond_dict.get('발바닥 지면 고정', True) else 1

    # 피험자 번호
    sub_num = 1
    for d in range(1, 10):
        if f"피험자{d}" in folder_name:
            sub_num = d
            break

    view_letters = ['A', 'B', 'C', 'D', 'E']
    view_names = ['view5', 'view2', 'view3', 'view4', 'view1']  # 시점 A=view5, B=view2, C=view3, D=view4, E=view1

    all_files = sorted(os.listdir(seq_dir))
    detector = MediaPipeAdapter.get_detector()
    results = []

    for vl, vn in zip(view_letters, view_names):
        cache_p = os.path.join(OUT_CACHE_DIR, f"{folder_name}_{vn}.npz")
        if os.path.exists(cache_p):
            cd = np.load(cache_p)
            kps_33_3d_64f = cd['kps_33_3d_64f']
        else:
            v_files = [f for f in all_files if (f'시점{vl}' in f or f'_{vl}_' in f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(v_files) < 4:
                continue

            raw_33_3d = []
            for vf in v_files:
                img_p = os.path.join(seq_dir, vf)
                res = MediaPipeAdapter.extract_frame_landmarks_full33(img_p, detector=detector)
                raw_33_3d.append(res['kps_33_3d'])

            arr_33_3d = np.array(raw_33_3d, dtype=np.float32)
            kps_33_3d_64f = interpolate_seq_to_64f(arr_33_3d)

            os.makedirs(OUT_CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_p, kps_33_3d_64f=kps_33_3d_64f)

        results.append({
            'folder': folder_name,
            'view': vn,
            'view_letter': f"시점 {vl}",
            'subject': sub_num,
            'kps_33_3d_64f': kps_33_3d_64f,
            'y_spine': y_spine,
            'y_knee': y_knee,
            'y_barbell': y_barbell,
            'y_sync': y_sync,
            'y_foot': y_foot
        })

    return results

def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 스쿼트 3대 결함 모델 -> 바벨 데드리프트 제로샷 전이 실측 평가")
    print("  - 대상 모델: c1 척추(v11.1), c3 무릎(v6.0), c4 발바닥(v5.1)")
    print("  - 평가 대상: AI-Hub 바벨 데드리프트 (피험자 1~9번 전원, 5대 전 시점)")
    print("=" * 100)

    # 1. 피험자 1~9번별 균형 잡힌 테스트 세트 샘플링
    img_folders = sorted(os.listdir(RAW_IMG_DIR))
    sub_clean_map = {d: [] for d in range(1, 10)}
    sub_spine_map = {d: [] for d in range(1, 10)}
    sub_knee_map = {d: [] for d in range(1, 10)}
    sub_foot_map = {d: [] for d in range(1, 10)}

    for fn in img_folders:
        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in fn:
                sub = d
                break
        label_p = os.path.join(LABEL_JSON_DIR, f"{fn}.json")
        if not os.path.exists(label_p):
            continue
        with open(label_p, 'r', encoding='utf-8', errors='ignore') as f:
            d_json = json.load(f)
        conds = {c['condition']: c['value'] for c in d_json.get('type_info', {}).get('conditions', [])}
        if all(conds.values()):
            sub_clean_map[sub].append(fn)
        if not conds.get('척추의 중립', True):
            sub_spine_map[sub].append(fn)
        if not conds.get('발과 무릎의 방향 일치', True):
            sub_knee_map[sub].append(fn)
        if not conds.get('발바닥 지면 고정', True):
            sub_foot_map[sub].append(fn)

    selected_folders = set()
    for d in range(1, 10):
        # 각 피험자당 정상 전수
        selected_folders.update(sub_clean_map[d])
        # 각 피험자당 각 오류 대표 1~2개
        selected_folders.update(sub_spine_map[d][:2])
        selected_folders.update(sub_knee_map[d][:2])
        selected_folders.update(sub_foot_map[d][:2])

    selected_list = sorted(list(selected_folders))
    print(f"[*] 선정된 테스트 시퀀스 폴더 수: {len(selected_list)}개 (예상 시점별 샘플 약 {len(selected_list)*5}개)")

    # 2. 병렬 관절 추출 및 64F 스플라인 생성
    t0 = time.time()
    all_samples = []
    print("[*] MediaPipe 33관절 64프레임 스플라인 추출 시작 (ProcessPoolExecutor)...")
    workers = min(6, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_single_folder, fn): fn for fn in selected_list}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_samples.extend(res)
            except Exception as e:
                print(f"[!] Error in {futures[future]}: {e}")

    print(f"[+] 총 {len(all_samples)}개 멀티뷰 데드리프트 시퀀스 준비 완료! (경과 시간: {time.time()-t0:.1f}초)")

    # 3. 모델 로드
    print("[*] 스쿼트 3대 결함 모델 로드 중...")
    c1_model = BabelSquatC1ModelV11_1()
    c3_model = BabelSquatC3ModelV6()
    c4_model = BabelSquatC4ModelV5_1()
    print("[+] c1(척추), c3(무릎), c4(발바닥) 모델 로드 완료!")

    # 4. 추론 수행
    y_true_spine, y_pred_spine, p_pred_spine = [], [], []
    y_true_knee, y_pred_knee, p_pred_knee = [], [], []
    y_true_foot, y_pred_foot, p_pred_foot = [], [], []

    view_stats = {
        'view5': {'name': '시점 A (우측 후방)', 'c1_true': [], 'c1_pred': [], 'c3_true': [], 'c3_pred': [], 'c4_true': [], 'c4_pred': []},
        'view2': {'name': '시점 B (우측 전방)', 'c1_true': [], 'c1_pred': [], 'c3_true': [], 'c3_pred': [], 'c4_true': [], 'c4_pred': []},
        'view3': {'name': '시점 C (정면)',     'c1_true': [], 'c1_pred': [], 'c3_true': [], 'c3_pred': [], 'c4_true': [], 'c4_pred': []},
        'view4': {'name': '시점 D (좌측 전방)', 'c1_true': [], 'c1_pred': [], 'c3_true': [], 'c3_pred': [], 'c4_true': [], 'c4_pred': []},
        'view1': {'name': '시점 E (좌측 후방)', 'c1_true': [], 'c1_pred': [], 'c3_true': [], 'c3_pred': [], 'c4_true': [], 'c4_pred': []}
    }

    c1_abstain_count = 0
    c3_abstain_count = 0
    c4_abstain_count = 0

    print("[*] 데드리프트 시퀀스 대상 스쿼트 모델 추론 실행...")
    for s in all_samples:
        kps = s['kps_33_3d_64f']
        vn = s['view']

        # c1 척추 추론
        try:
            r1 = c1_model.predict_selective(kps)
            prob1 = r1.get('defect_probability')
            if prob1 is None or r1['status'] == 'UNDETERMINED':
                c1_abstain_count += 1
                pred1 = 1 if (prob1 is not None and prob1 >= 0.5) else 0
            else:
                pred1 = 1 if r1['status'] == 'DEFECT' else 0
                prob1 = r1['defect_probability']
        except Exception:
            prob1 = 0.5
            pred1 = 0

        # c3 무릎 추론
        try:
            r3 = c3_model.predict_selective(kps)
            prob3 = r3.get('probability', 0.5)
            if r3.get('status') == 'UNDETERMINED':
                c3_abstain_count += 1
                pred3 = 1 if prob3 >= 0.5 else 0
            else:
                pred3 = 1 if r3.get('status') == 'DEFECT' else 0
        except Exception:
            prob3 = 0.5
            pred3 = 0

        # c4 발바닥 추론
        try:
            r4 = c4_model.predict_selective(kps)
            prob4 = r4.get('probability', 0.5)
            if r4.get('status') == 'UNDETERMINED':
                c4_abstain_count += 1
                pred4 = 1 if prob4 >= 0.5 else 0
            else:
                pred4 = 1 if r4.get('status') == 'DEFECT' else 0
        except Exception:
            prob4 = 0.5
            pred4 = 0

        # 정답 및 예측 수집
        y_true_spine.append(s['y_spine'])
        y_pred_spine.append(pred1)
        p_pred_spine.append(prob1)

        y_true_knee.append(s['y_knee'])
        y_pred_knee.append(pred3)
        p_pred_knee.append(prob3)

        y_true_foot.append(s['y_foot'])
        y_pred_foot.append(pred4)
        p_pred_foot.append(prob4)

        if vn in view_stats:
            view_stats[vn]['c1_true'].append(s['y_spine'])
            view_stats[vn]['c1_pred'].append(pred1)
            view_stats[vn]['c3_true'].append(s['y_knee'])
            view_stats[vn]['c3_pred'].append(pred3)
            view_stats[vn]['c4_true'].append(s['y_foot'])
            view_stats[vn]['c4_pred'].append(pred4)

    # 5. 성능 메트릭 산출
    def calc_metrics(y_true, y_pred, p_pred):
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        try:
            auc = roc_auc_score(y_true, p_pred) * 100
        except Exception:
            auc = 50.0
        cm = confusion_matrix(y_true, y_pred)
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm.tolist()}

    m_spine = calc_metrics(y_true_spine, y_pred_spine, p_pred_spine)
    m_knee = calc_metrics(y_true_knee, y_pred_knee, p_pred_knee)
    m_foot = calc_metrics(y_true_foot, y_pred_foot, p_pred_foot)

    print("\n" + "=" * 100)
    print("  [실측 결과 1] 바벨 스쿼트 3대 결함 모델의 데드리프트 제로샷(Zero-Shot) 분류 성능")
    print("=" * 100)
    print(f"{'평가 모델 (결함 라벨)':<28} | {'스쿼트 원래 F1':<12} | {'데드리프트 F1':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<8}")
    print("-" * 100)
    print(f"{'1. c1 척추의 중립 (Spine)':<28} | {'97.13%':<12} | {m_spine['f1']:>6.2f}%      | {m_spine['acc']:>6.2f}%   | {m_spine['prec']:>6.2f}%   | {m_spine['rec']:>6.2f}%   | {m_spine['auc']:>6.2f}%")
    print(f"{'2. c3 발과 무릎 일치 (Knee)':<28} | {'91.82%':<12} | {m_knee['f1']:>6.2f}%      | {m_knee['acc']:>6.2f}%   | {m_knee['prec']:>6.2f}%   | {m_knee['rec']:>6.2f}%   | {m_knee['auc']:>6.2f}%")
    print(f"{'3. c4 발바닥 지면고정 (Foot)':<28} | {'90.54%':<12} | {m_foot['f1']:>6.2f}%      | {m_foot['acc']:>6.2f}%   | {m_foot['prec']:>6.2f}%   | {m_foot['rec']:>6.2f}%   | {m_foot['auc']:>6.2f}%")
    print("-" * 100)

    print("\n" + "=" * 100)
    print("  [실측 결과 2] 5대 카메라 시점별 데드리프트 전이 성능 (F1-Score)")
    print("=" * 100)
    print(f"{'카메라 시점':<22} | {'척추 중립 (c1)':<18} | {'무릎 외반 (c3)':<18} | {'발바닥 고정 (c4)':<18}")
    print("-" * 100)
    for vn, vs in view_stats.items():
        f1_c1 = f1_score(vs['c1_true'], vs['c1_pred'], zero_division=0) * 100
        f1_c3 = f1_score(vs['c3_true'], vs['c3_pred'], zero_division=0) * 100
        f1_c4 = f1_score(vs['c4_true'], vs['c4_pred'], zero_division=0) * 100
        print(f"{vs['name']:<22} | {f1_c1:>10.2f}%        | {f1_c3:>10.2f}%        | {f1_c4:>10.2f}%")
    print("-" * 100)

    # JSON 결과 저장
    out_results = {
        'total_samples': len(all_samples),
        'metrics': {
            'c1_spine': m_spine,
            'c3_knee': m_knee,
            'c4_foot': m_foot
        },
        'view_breakdown': {
            vn: {
                'name': vs['name'],
                'c1_f1': f1_score(vs['c1_true'], vs['c1_pred'], zero_division=0) * 100,
                'c3_f1': f1_score(vs['c3_true'], vs['c3_pred'], zero_division=0) * 100,
                'c4_f1': f1_score(vs['c4_true'], vs['c4_pred'], zero_division=0) * 100
            } for vn, vs in view_stats.items()
        }
    }
    out_json = os.path.join(CURRENT_DIR, "deadlift_transfer_test_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 상세 실측 결과 JSON 저장: {out_json}")

if __name__ == '__main__':
    main()

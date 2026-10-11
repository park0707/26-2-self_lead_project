# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: 무릎 구부린 채 고정 위반 (sd1_v2)
Script: analyze_sd1_misclassifications.py
Purpose: sd1_v2 모델의 OOF 오분류 샘플(FP, FN, Uncertain)을 전수 추출하고,
         AI-Hub 시나리오 Key, 생체역학 각도, 카메라 시점, 피험자 신체 편차 관점에서
         오분류 심층 원인을 통계적·물리적으로 전수 분석 및 규명
"""

import os
import sys
import glob
import json
import pickle
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from stiff_deadlift_knee_features_v2 import (
    extract_knee_invariance_features_6d, extract_knee_temporal_8ch_v2, FEATURE_NAMES_6D
)
from sd1_v2_classifier import KneeTCNClassifierV2

CACHE_PATH = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "sd1_v2_bundle.pkl")
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D"
OUT_JSON_PATH = os.path.join(CURRENT_DIR, "sd1_misclassification_analysis.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

def analyze():
    print("=" * 100)
    print("  [vidio-health-trainer] 스티프 데드리프트 sd1_v2 오분류 심층 원인 전수 감사")
    print("=" * 100)

    # 1. 캐시 및 데이터 로드
    data = np.load(CACHE_PATH, allow_pickle=True)
    labels = data['labels']
    levels = data['levels']
    l4_mask = data['l4_mask']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    sequence_ids = data['sequence_ids']

    with open(BUNDLE_PATH, 'rb') as f:
        bundle = pickle.load(f)

    fold_models_lgb = bundle['fold_models_lgb']
    tcn_state_dicts = bundle['tcn_state_dicts']
    tau_lower = bundle['thresholds']['tau_lower']
    tau_upper = bundle['thresholds']['tau_upper']
    w_lgb = bundle['stream_weights']['lgb']
    w_tcn = bundle['stream_weights']['tcn']

    tcn_models = []
    for s_dict in tcn_state_dicts:
        net = KneeTCNClassifierV2(in_channels=8, num_channels=[16, 24, 32], kernel_size=3, dropout=0.2)
        net.load_state_dict(s_dict)
        net.eval()
        tcn_models.append(net)

    from src.models.first_model.dataset import Stage1DatasetLoader
    splits = Stage1DatasetLoader.load_all_dual_splits()
    seen_sids = {}
    for sname, sdata in splits.items():
        stiff_mask = (sdata['y_action'] == 2)
        s_sids = sdata['sequence_ids'][stiff_mask]
        s_X = sdata['X_Format_B'][stiff_mask]
        for sid, x in zip(s_sids, s_X):
            seen_sids[str(sid)] = x

    valid_mask = ~l4_mask
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)

    # 2. Key 및 Description 매핑 딕셔너리 구축
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.json"))
    key_info_map = {}
    file_to_key = {}
    for lf in label_files:
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                d = json.load(f).get('type_info', {})
            k = str(d.get('key', 'unknown'))
            desc = d.get('description', '')
            conds = {c['condition']: c['value'] for c in d.get('conditions', [])}
            base = os.path.splitext(os.path.basename(lf))[0]
            file_to_key[base] = k
            if k not in key_info_map:
                key_info_map[k] = {
                    'description': desc,
                    'conditions': conds
                }
        except Exception:
            pass

    # 3. 5-Fold OOF 예측 수행 (학습 당시와 동일한 피험자 격리 분할)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    valid_subs = subject_ids[valid_indices]
    valid_labels = labels[valid_indices]

    oof_probs = np.zeros(N_valid, dtype=np.float32)
    feats_6d_valid = np.zeros((N_valid, 6), dtype=np.float32)

    for i, idx in enumerate(valid_indices):
        sid = str(sequence_ids[idx])
        x_seq = seen_sids[sid]
        feats_6d_valid[i] = extract_knee_invariance_features_6d(x_seq)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(valid_indices, valid_labels, valid_subs)):
        m_lgb = fold_models_lgb[fold]
        m_tcn = tcn_models[fold]

        # LightGBM 추론
        f_val = feats_6d_valid[val_idx]
        p_lgb = m_lgb.predict_proba(f_val)[:, 1]

        # TCN 추론
        tcn_inputs = []
        for v_i in val_idx:
            idx = valid_indices[v_i]
            sid = str(sequence_ids[idx])
            tcn_inputs.append(extract_knee_temporal_8ch_v2(seen_sids[sid]))
        bx = torch.tensor(np.array(tcn_inputs), dtype=torch.float32)
        with torch.no_grad():
            p_tcn = F.softmax(m_tcn(bx), dim=-1)[:, 1].numpy()

        # 앙상블 OOF 확률
        oof_probs[val_idx] = w_lgb * p_lgb + w_tcn * p_tcn

    # 4. 개별 샘플 진단 및 메타데이터 결합
    samples = []
    for i, idx in enumerate(valid_indices):
        sid = str(sequence_ids[idx])
        y_true = int(labels[idx])
        p = float(oof_probs[i])
        lvl = int(levels[idx])
        sub = int(subject_ids[idx])
        view = str(view_ids[idx])

        # Key 매핑
        parts = sid.split('_')
        folder_name = '_'.join(parts[:-1])
        key_num = file_to_key.get(folder_name, 'unknown')
        if key_num == 'unknown':
            # 매칭 검색
            for base_name, k_val in file_to_key.items():
                if len(parts) >= 4 and parts[1] in base_name and parts[2] in base_name and parts[3] in base_name:
                    key_num = k_val
                    break

        key_desc = key_info_map.get(key_num, {}).get('description', '설명 없음')
        key_conds = key_info_map.get(key_num, {}).get('conditions', {})

        # 2-State 예측
        pred_2s = 1 if p >= 0.50 else 0

        # 3-State 예측
        if p < tau_lower:
            pred_3s = 0 # Clean
        elif p >= tau_upper:
            pred_3s = 1 # Defect
        else:
            pred_3s = 2 # Uncertain

        # 오분류 범주
        # 0: TN, 1: TP, 2: FP (정상인데 결함 판정), 3: FN (결함인데 정상 판정)
        if y_true == 0 and pred_2s == 0:
            category_2s = "TN"
        elif y_true == 1 and pred_2s == 1:
            category_2s = "TP"
        elif y_true == 0 and pred_2s == 1:
            category_2s = "FP"
        else:
            category_2s = "FN"

        feat_dict = {name: float(feats_6d_valid[i, j]) for j, name in enumerate(FEATURE_NAMES_6D)}

        samples.append({
            'seq_id': sid,
            'index': int(idx),
            'subject_id': sub,
            'view': view,
            'view_alias': VIEW_ALIAS_MAP.get(view, view),
            'level': lvl,
            'key': key_num,
            'key_description': key_desc,
            'key_conditions': key_conds,
            'y_true': y_true,
            'prob': p,
            'pred_2s': pred_2s,
            'pred_3s': pred_3s,
            'category_2s': category_2s,
            'features': feat_dict
        })

    # 5. 오분류 전수 집계
    n_tp = sum(1 for s in samples if s['category_2s'] == "TP")
    n_tn = sum(1 for s in samples if s['category_2s'] == "TN")
    n_fp = sum(1 for s in samples if s['category_2s'] == "FP")
    n_fn = sum(1 for s in samples if s['category_2s'] == "FN")
    n_uncertain = sum(1 for s in samples if s['pred_3s'] == 2)

    fp_samples = [s for s in samples if s['category_2s'] == "FP"]
    fn_samples = [s for s in samples if s['category_2s'] == "FN"]
    uncertain_samples = [s for s in samples if s['pred_3s'] == 2]

    print(f"\n[*] 전체 유효 샘플: {N_valid}개")
    print(f"  - TN (정상 적중): {n_tn}개 ({n_tn/N_valid*100:.2f}%)")
    print(f"  - TP (결함 적중): {n_tp}개 ({n_tp/N_valid*100:.2f}%)")
    print(f"  - FP (정상->결함 과잉탐지 오분류): {n_fp}개 ({n_fp/N_valid*100:.2f}%)")
    print(f"  - FN (결함->정상 미탐지 오분류):   {n_fn}개 ({n_fn/N_valid*100:.2f}%)")
    print(f"  - 3-State 판단보류 (Uncertain):   {n_uncertain}개 ({n_uncertain/N_valid*100:.2f}%)")

    # ==============================================================================
    # 심층 분석 1: 시나리오 Key별 분석 (특히 FN의 Key 분포)
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  [심층 원인 1: AI-Hub 시나리오 Key별 오분류 통계 (Halo Effect 추적)]")
    print("=" * 90)

    key_stats = {}
    for s in samples:
        k = s['key']
        if k not in key_stats:
            key_stats[k] = {
                'desc': s['key_description'],
                'conds': s['key_conditions'],
                'total': 0, 'TN': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'Uncertain': 0,
                'probs': []
            }
        key_stats[k]['total'] += 1
        key_stats[k][s['category_2s']] += 1
        if s['pred_3s'] == 2:
            key_stats[k]['Uncertain'] += 1
        key_stats[k]['probs'].append(s['prob'])

    for k in sorted(key_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        ks = key_stats[k]
        mean_p = float(np.mean(ks['probs']))
        print(f"  * Key {k:4s} ({ks['desc'][:30]:30s}) | Total: {ks['total']:3d}개 | "
              f"TN:{ks['TN']:3d} | TP:{ks['TP']:3d} | FP:{ks['FP']:3d} | FN:{ks['FN']:3d} | "
              f"보류:{ks['Uncertain']:2d} | 평균확률 P={mean_p:.3f}")

    # ==============================================================================
    # 심층 분석 2: 생체역학 6D 피처값 대조 (TN vs FP vs TP vs FN)
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  [심층 원인 2: 범주별(TN / FP / TP / FN) 생체역학 피처 평균 및 표준편차 대조]")
    print("=" * 90)

    cat_feature_stats = {}
    for cat in ["TN", "FP", "TP", "FN"]:
        cat_s = [s for s in samples if s['category_2s'] == cat]
        cat_feature_stats[cat] = {}
        for feat_name in FEATURE_NAMES_6D:
            vals = [s['features'][feat_name] for s in cat_s]
            cat_feature_stats[cat][feat_name] = {
                'mean': float(np.mean(vals)) if vals else 0.0,
                'std': float(np.std(vals)) if vals else 0.0
            }

    print(f"  {'피처 식별자':25s} | {'TN (진짜 정상)':16s} | {'FP (과잉 탐지)':16s} | {'TP (진짜 결함)':16s} | {'FN (미탐지)':16s}")
    print("-" * 95)
    for feat_name in FEATURE_NAMES_6D:
        tn_m = cat_feature_stats["TN"][feat_name]['mean']
        fp_m = cat_feature_stats["FP"][feat_name]['mean']
        tp_m = cat_feature_stats["TP"][feat_name]['mean']
        fn_m = cat_feature_stats["FN"][feat_name]['mean']
        print(f"  {feat_name:25s} | {tn_m:14.2f}   | {fp_m:14.2f}   | {tp_m:14.2f}   | {fn_m:14.2f}")

    # ==============================================================================
    # 심층 분석 3: 카메라 5대 시점별 오분류 통계 (시점 D F1 하락 원인)
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  [심층 원인 3: 5대 공식 카메라 시점(시점 A~E)별 오분류 분포]")
    print("=" * 90)

    view_stats = {}
    for s in samples:
        v = s['view']
        if v not in view_stats:
            view_stats[v] = {
                'alias': s['view_alias'],
                'total': 0, 'TN': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'Uncertain': 0,
                'probs': []
            }
        view_stats[v]['total'] += 1
        view_stats[v][s['category_2s']] += 1
        if s['pred_3s'] == 2:
            view_stats[v]['Uncertain'] += 1
        view_stats[v]['probs'].append(s['prob'])

    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        vs = view_stats.get(vn, {})
        if not vs:
            continue
        acc = (vs['TN'] + vs['TP']) / vs['total'] * 100.0
        print(f"  * {vs['alias']:24s} ({vn}) | Total: {vs['total']:3d}개 | "
              f"TN:{vs['TN']:3d} | TP:{vs['TP']:3d} | FP:{vs['FP']:3d} | FN:{vs['FN']:3d} | "
              f"정확도: {acc:5.2f}% | 보류율: {vs['Uncertain']/vs['total']*100:5.2f}%")

    # ==============================================================================
    # 심층 분석 4: 피험자별(1~9번) 오분류 통계 (신체 편차 및 유연성 영향)
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  [심층 원인 4: 피험자 1~9번별 오분류 및 생체역학 편차]")
    print("=" * 90)

    sub_stats = {}
    for s in samples:
        sub = s['subject_id']
        if sub not in sub_stats:
            sub_stats[sub] = {
                'total': 0, 'TN': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'Uncertain': 0,
                'knee_angles': [], 'probs': []
            }
        sub_stats[sub]['total'] += 1
        sub_stats[sub][s['category_2s']] += 1
        if s['pred_3s'] == 2:
            sub_stats[sub]['Uncertain'] += 1
        sub_stats[sub]['knee_angles'].append(s['features']['bottom_knee_angle'])
        sub_stats[sub]['probs'].append(s['prob'])

    for sub in sorted(sub_stats.keys()):
        ss = sub_stats[sub]
        acc = (ss['TN'] + ss['TP']) / ss['total'] * 100.0
        mean_ang = float(np.mean(ss['knee_angles']))
        std_ang = float(np.std(ss['knee_angles']))
        print(f"  * 피험자 {sub:2d} | Total: {ss['total']:3d}개 | TN:{ss['TN']:3d} | TP:{ss['TP']:3d} | "
              f"FP:{ss['FP']:3d} | FN:{ss['FN']:3d} | 정확도:{acc:5.2f}% | "
              f"최저점 무릎각도: {mean_ang:5.1f}° (±{std_ang:4.1f}°)")

    # ==============================================================================
    # 심층 분석 5: 모호성 Level별 오분류 통계
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  [심층 원인 5: 모호성 레벨(Level 1~3)별 오분류 분포]")
    print("=" * 90)

    lvl_stats = {}
    for s in samples:
        lvl = s['level']
        if lvl not in lvl_stats:
            lvl_stats[lvl] = {
                'total': 0, 'TN': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'Uncertain': 0
            }
        lvl_stats[lvl]['total'] += 1
        lvl_stats[lvl][s['category_2s']] += 1
        if s['pred_3s'] == 2:
            lvl_stats[lvl]['Uncertain'] += 1

    for lvl in sorted(lvl_stats.keys()):
        ls = lvl_stats[lvl]
        acc = (ls['TN'] + ls['TP']) / ls['total'] * 100.0
        print(f"  * Level {lvl} | Total: {ls['total']:3d}개 | TN:{ls['TN']:3d} | TP:{ls['TP']:3d} | "
              f"FP:{ls['FP']:3d} | FN:{ls['FN']:3d} | 정확도: {acc:5.2f}% | 보류율: {ls['Uncertain']/ls['total']*100:5.2f}%")

    # 대표적 오분류 케이스 10선 추출
    top_fp = sorted(fp_samples, key=lambda x: x['prob'], reverse=True)[:5]
    top_fn = sorted(fn_samples, key=lambda x: x['prob'])[:5]

    # JSON 저장
    output_data = {
        'total_valid': N_valid,
        'summary': {
            'TN': n_tn, 'TP': n_tp, 'FP': n_fp, 'FN': n_fn,
            'uncertain': n_uncertain,
            'accuracy_2state': float((n_tn + n_tp) / N_valid),
            'specificity_2state': float(n_tn / (n_tn + n_fp)),
            'sensitivity_2state': float(n_tp / (n_tp + n_fn))
        },
        'key_statistics': key_stats,
        'category_feature_statistics': cat_feature_stats,
        'view_statistics': view_stats,
        'subject_statistics': sub_stats,
        'level_statistics': lvl_stats,
        'representative_fp_cases': top_fp,
        'representative_fn_cases': top_fn
    }

    with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 오분류 심층 분석 결과 JSON 저장 완료: {OUT_JSON_PATH}")

if __name__ == '__main__':
    analyze()

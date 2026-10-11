# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c1_v11_a.py
Purpose: c1(척추의 중립 위반) v11_a (실험 코드 E0: Nadir ±8F 국소 윈도우 피처 38D) 모델
         - 2,605개 마스터 데이터셋 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 및 검증
         - v10(28D) 대비 OOF 성능 향상 감사
         - 5대 시점별 세부 F1/AUC/Acc 감사
         - 3-State 선택적 거부(Selective Abstention) Risk-Coverage 비교
"""

import os
import sys
import time
import json
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_a import (
    extract_c1_v11_a_features_from_seq,
    FEATURE_NAMES_38_C1,
    BabelSquatC1ModelV11A
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
JSON_DIR = r"d:\Data\01_raw_data\labels_json"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_38d_feats.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v11_a_weights.pkl"
RESULTS_JSON_PATH = r"d:\src\models\second_models\babel_squat\c1_v11_a_evaluation_results.json"


def interpolate_to_64f(arr: np.ndarray) -> np.ndarray:
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)


def build_spine_cache(base_sids: np.ndarray, json_map: dict) -> dict:
    """521개 base action에 대해 SpinePose 9대 분절 (64, 9, 3) 3D 좌표 사전 복원"""
    print(f"[*] [1/4] 521개 고유 base action에 대한 SpinePose 9대 분절 3D 척추 복원 시작...", flush=True)
    t0 = time.time()
    spine_cache = {}
    z_src = np.array([0.0, 0.25, 0.60, 0.85], dtype=np.float32)
    z_dst = np.array([0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00], dtype=np.float32)

    for i, b in enumerate(base_sids, 1):
        jp = json_map.get(b)
        if jp is None or not os.path.exists(jp):
            spine_cache[b] = None
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            jd = json.load(f)
        frames = jd.get('frames', [])
        pts_list = [f.get('pts', {}) for f in frames]
        neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in pts_list], dtype=np.float32)
        back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in pts_list], dtype=np.float32)
        waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in pts_list], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in pts_list], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in pts_list], dtype=np.float32)

        neck = interpolate_to_64f(neck_raw)
        back = interpolate_to_64f(back_raw)
        waist = interpolate_to_64f(waist_raw)
        pelvis = interpolate_to_64f((lh_raw + rh_raw) / 2.0)

        T = 64
        s9 = np.zeros((T, 9, 3), dtype=np.float32)
        for t in range(T):
            pts_4 = np.array([pelvis[t], waist[t], back[t], neck[t]], dtype=np.float32)
            fx = PchipInterpolator(z_src, pts_4[:, 0], extrapolate=True)
            fy = PchipInterpolator(z_src, pts_4[:, 1], extrapolate=True)
            fz = PchipInterpolator(z_src, pts_4[:, 2], extrapolate=True)
            s9[t, :, 0] = fx(z_dst)
            s9[t, :, 1] = fy(z_dst)
            s9[t, :, 2] = fz(z_dst)
        spine_cache[b] = s9

        if i % 100 == 0 or i == len(base_sids):
            print(f"    - 진행: {i}/{len(base_sids)} ({i/len(base_sids)*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    print(f"[*] SpinePose 9대 분절 캐시 완성! (총 {len(spine_cache)}개, 소요: {time.time()-t0:.2f}초)", flush=True)
    return spine_cache


def extract_or_load_all_features(seq_ids, base_sids, json_map):
    """2,605개 시퀀스로부터 v11_a 38D 피처 추출 또는 캐시 로드"""
    if os.path.exists(FEATS_CACHE_PATH):
        print(f"[*] 기존 피처 캐시 파일 발견: {FEATS_CACHE_PATH}", flush=True)
        cached = np.load(FEATS_CACHE_PATH, allow_pickle=True)
        if len(cached['X_c1_v11_a']) == len(seq_ids):
            print(f"[*] 캐시 데이터 로드 완료: shape={cached['X_c1_v11_a'].shape}", flush=True)
            return cached['X_c1_v11_a']

    spine_cache = build_spine_cache(np.unique(base_sids), json_map)

    print(f"[*] [2/4] 2,605개 전 시퀀스에 대한 c1 v11_a 38D (기존 28D + E0 10D) 피처 추출 시작...", flush=True)
    t0 = time.time()
    feats_list = []
    N = len(seq_ids)

    for i, (sid, bsid) in enumerate(zip(seq_ids, base_sids), 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']
        s9 = spine_cache.get(bsid)

        feat_vec = extract_c1_v11_a_features_from_seq(kps_33, s9)
        feats_list.append(feat_vec)

        if i % 500 == 0 or i == N:
            print(f"    - 피처 추출: {i}/{N} ({i/N*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    X_c1_v11_a = np.array(feats_list, dtype=np.float32)
    print(f"[*] 피처 추출 완료: shape={X_c1_v11_a.shape} (소요: {time.time()-t0:.2f}초)", flush=True)

    # 캐시 저장
    os.makedirs(os.path.dirname(FEATS_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        FEATS_CACHE_PATH,
        X_c1_v11_a=X_c1_v11_a,
        feature_names=FEATURE_NAMES_38_C1,
        sequence_ids=seq_ids
    )
    print(f"[*] 피처 캐시 저장 완료: {FEATS_CACHE_PATH}", flush=True)
    return X_c1_v11_a


def evaluate_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2)
    }


def main():
    raise RuntimeError(
        "CRITICAL ERROR: train_and_evaluate_c1_v11_a.py is permanently DEPRECATED and PROHIBITED! "
        "This script extracts features from AI-Hub 3D JSON ground truth files, which violates the "
        "vidio-health-trainer Strict Data Isolation rule. Use extract_true_spinepose_64f.py instead."
    )

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    base_sids = master['base_sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c1 = master['Y_defects_golden'][:, 0].astype(int)
    N = len(seq_ids)

    # JSON map 생성
    json_map = {}
    for jf in glob.glob(os.path.join(JSON_DIR, '**', '*_3D.json'), recursive=True):
        bn = os.path.basename(jf).replace('_3D.json', '')
        json_map[bn] = jf

    X = extract_or_load_all_features(seq_ids, base_sids, json_map)

    print(f"\n[*] [3/4] 5-Fold StratifiedGroupKFold 교차 검증 및 앙상블 학습 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(N, dtype=np.float32)
    cb_models = []
    lgb_models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_c1, groups=subjects), 1):
        X_train, y_train = X[train_idx], y_c1[train_idx]
        X_val, y_val = X[val_idx], y_c1[val_idx]
        val_sub = np.unique(subjects[val_idx])

        # CatBoost
        cb = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            thread_count=-1,
            verbose=False
        )
        cb.fit(X_train, y_train)

        # LightGBM
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            colsample_bytree=0.85,
            subsample=0.85,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        p_cb_val = cb.predict_proba(X_val)[:, 1]
        p_lgb_val = clf.predict_proba(X_val)[:, 1]
        p_ens_val = 0.60 * p_cb_val + 0.40 * p_lgb_val

        oof_probs[val_idx] = p_ens_val
        cb_models.append(cb)
        lgb_models.append(clf)

        m_fold = evaluate_metrics(y_val, p_ens_val)
        fold_metrics.append(m_fold)
        print(f"    - Fold {fold} (검증 피험자 {val_sub}): F1={m_fold['f1']:.2f}%, AUC={m_fold['auc']:.2f}%, Rec={m_fold['rec']:.2f}%, Prec={m_fold['prec']:.2f}%", flush=True)

    # v11_a 모델 객체 생성 및 가중치 저장
    model_v11_a = BabelSquatC1ModelV11A()
    model_v11_a.cb_models = cb_models
    model_v11_a.lgb_models = lgb_models
    model_v11_a.is_fitted = True
    model_v11_a.save_weights(WEIGHTS_PATH)

    # 전체 OOF 성능 산출
    m_v11_a = evaluate_metrics(y_c1, oof_probs)

    # v10 (v9 가중치 기준) 성능 기준치: F1 79.73%, AUC 88.33%, Prec 80.00%, Rec 79.46%
    v10_f1 = 79.73
    v10_auc = 88.33
    v10_prec = 80.00
    v10_rec = 79.46

    print("\n" + "=" * 115, flush=True)
    print("  [★ c1 v11_a (E0: 38D) vs v10 (28D) 5-Fold 실측 성능 직접 대조 (N=2,605)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'지표 (Metric)':<18} | {'v10 (28D 베이스)':<20} | {'v11_a (E0 38D)':<20} | {'변화량 (Delta)':<18}", flush=True)
    print("-" * 115, flush=True)
    for name, v10_val, v11_val in [
        ("F1-Score", v10_f1, m_v11_a['f1']),
        ("ROC-AUC", v10_auc, m_v11_a['auc']),
        ("Precision", v10_prec, m_v11_a['prec']),
        ("Recall", v10_rec, m_v11_a['rec']),
        ("Accuracy", 80.38, m_v11_a['acc'])
    ]:
        delta = v11_val - v10_val
        delta_str = f"+{delta:.2f}%p" if delta >= 0 else f"{delta:.2f}%p"
        print(f"{name:<18} | {v10_val:>18.2f}% | {v11_val:>18.2f}% | {delta_str:>18}", flush=True)

    # 5대 시점별 세부 성능 감사
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (후방 180°)', 'view2 (우측 45°)', 'view3 (정면 0°)', 'view4 (좌측 45°)', 'view5 (우측 후방 45°)']
    v10_view_f1s = {'view1': 76.12, 'view2': 83.21, 'view3': 74.30, 'view4': 81.85, 'view5': 83.18}

    print("\n" + "=" * 115, flush=True)
    print("  [5대 시점별 세부 성능 감사 (Multi-View Performance Audit)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (카메라)':<24} | {'표본수':<8} | {'v10 F1':<12} | {'v11_a F1':<14} | {'F1 변화량':<12} | {'v11_a AUC':<12} | {'v11_a Acc'}", flush=True)
    print("-" * 115, flush=True)

    view_results = {}
    for v, vn in zip(view_list, view_names):
        mask = (views == v)
        mv = evaluate_metrics(y_c1[mask], oof_probs[mask])
        v10_vf1 = v10_view_f1s.get(v, 0.0)
        d_f1 = mv['f1'] - v10_vf1
        d_f1_str = f"+{d_f1:.2f}%p" if d_f1 >= 0 else f"{d_f1:.2f}%p"
        print(f"{vn:<24} | {int(np.sum(mask)):<8} | {v10_vf1:>10.2f}% | {mv['f1']:>12.2f}% | {d_f1_str:>10} | {mv['auc']:>10.2f}% | {mv['acc']:>8.2f}%", flush=True)
        view_results[v] = {
            'view_name': vn,
            'count': int(np.sum(mask)),
            'v10_f1': v10_vf1,
            'v11_a_f1': mv['f1'],
            'delta_f1': round(d_f1, 2),
            'v11_a_auc': mv['auc'],
            'v11_a_acc': mv['acc']
        }

    # 3-State 선택적 거부(Selective Classification) 평가
    print("\n" + "=" * 115, flush=True)
    print("  [3-State 선택적 거부 (Selective Abstention) Risk-Coverage 비교]", flush=True)
    print("=" * 115, flush=True)

    threshold_pairs = [
        (0.50, 0.50, "Full Decision (강제 이진)"),
        (0.40, 0.60, "Balanced Abstention (표준 추천)"),
        (0.35, 0.65, "Standard Selective"),
        (0.30, 0.70, "High Confidence")
    ]
    print(f"{'임계치 설정':<32} | {'Coverage':<10} | {'Abstention':<11} | {'Selective F1':<13} | {'Selective Acc':<14} | {'Risk (오류율)':<12}", flush=True)
    print("-" * 115, flush=True)

    selective_results = {}
    for t_norm, t_def, desc in threshold_pairs:
        results = model_v11_a.predict_selective_batch(
            X,
            tau_normal=t_norm,
            tau_defect=t_def,
            disagreement_tol=0.35,
            fold_std_tol=0.25,
            min_rom_threshold=0.0004
        )
        statuses = np.array([r['status'] for r in results])
        det_mask = (statuses != "UNDETERMINED")
        n_det = int(np.sum(det_mask))
        cov = n_det / N * 100.0
        abs_rate = 100.0 - cov

        if n_det > 0:
            y_det_true = y_c1[det_mask]
            y_det_pred = (statuses[det_mask] == "DEFECT").astype(int)
            sel_f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
            sel_acc = accuracy_score(y_det_true, y_det_pred) * 100.0
            sel_risk = (1.0 - accuracy_score(y_det_true, y_det_pred)) * 100.0
        else:
            sel_f1, sel_acc, sel_risk = 0.0, 0.0, 100.0

        label_str = f"[{t_norm:.2f}, {t_def:.2f}] {desc}"
        print(f"{label_str:<32} | {cov:>9.1f}% | {abs_rate:>10.1f}% | {sel_f1:>12.2f}% | {sel_acc:>13.2f}% | {sel_risk:>11.2f}%", flush=True)
        selective_results[f"{t_norm}_{t_def}"] = {
            'desc': desc, 'coverage': round(cov, 1), 'abstention': round(abs_rate, 1),
            'selective_f1': round(sel_f1, 2), 'selective_acc': round(sel_acc, 2), 'risk': round(sel_risk, 2)
        }

    # 결과 JSON 저장
    output_summary = {
        'model_name': 'c1_v11_a',
        'experiment_code': 'E0',
        'description': 'Nadir-focused local window feature engineering (38D)',
        'overall_metrics': m_v11_a,
        'baseline_v10_metrics': {'f1': v10_f1, 'auc': v10_auc, 'prec': v10_prec, 'rec': v10_rec, 'acc': 80.38},
        'fold_metrics': fold_metrics,
        'view_metrics': view_results,
        'selective_metrics': selective_results
    }
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 실험 결과 JSON 저장 완료: {RESULTS_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()

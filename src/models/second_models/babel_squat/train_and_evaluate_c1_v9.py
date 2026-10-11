# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c1_v9.py
Purpose: c1(척추의 중립 위반) v9.0 28D 공식 모델 2,605개 마스터 데이터셋 5-Fold 교차 검증 및 학습/가중치 배포
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
from src.models.second_models.babel_squat.c1_model_v9 import (
    extract_c1_v9_features_from_seq,
    FEATURE_NAMES_28_C1,
    BabelSquatC1ModelV9
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
JSON_DIR = r"d:\Data\01_raw_data\labels_json"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c1_v9_28d_feats.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v9_weights.pkl"
OOF_PATH = r"D:\Data\피처_데이터\oof_c1_v9_predictions.npz"
PREV_OOF_PATH = r"D:\Data\피처_데이터\oof_master_64f_predictions.npz"


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
        jp = json_map[b]
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
    """2,605개 시퀀스로부터 v9 28D 피처 추출 또는 캐시 로드"""
    if os.path.exists(FEATS_CACHE_PATH):
        print(f"[*] 기존 피처 캐시 파일 발견: {FEATS_CACHE_PATH}", flush=True)
        cached = np.load(FEATS_CACHE_PATH, allow_pickle=True)
        if len(cached['X_c1_v9']) == len(seq_ids):
            print(f"[*] 캐시 데이터 로드 완료: shape={cached['X_c1_v9'].shape}", flush=True)
            return cached['X_c1_v9']

    spine_cache = build_spine_cache(np.unique(base_sids), json_map)

    print(f"[*] [2/4] 2,605개 전 시퀀스에 대한 c1 v9 28D 생체역학 피처 추출 시작...", flush=True)
    t0 = time.time()
    feats_list = []
    N = len(seq_ids)

    for i, (sid, bsid) in enumerate(zip(seq_ids, base_sids), 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']
        s9 = spine_cache[bsid]

        feat_vec = extract_c1_v9_features_from_seq(kps_33, s9)
        feats_list.append(feat_vec)

        if i % 500 == 0 or i == N:
            print(f"    - 피처 추출: {i}/{N} ({i/N*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    X_c1_v9 = np.array(feats_list, dtype=np.float32)
    print(f"[*] 피처 추출 완료: shape={X_c1_v9.shape} (소요: {time.time()-t0:.2f}초)", flush=True)

    # 캐시 저장
    os.makedirs(os.path.dirname(FEATS_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        FEATS_CACHE_PATH,
        X_c1_v9=X_c1_v9,
        feature_names=FEATURE_NAMES_28_C1,
        sequence_ids=seq_ids
    )
    print(f"[*] 피처 캐시 저장 완료: {FEATS_CACHE_PATH}", flush=True)
    return X_c1_v9


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
        "CRITICAL ERROR: train_and_evaluate_c1_v9.py is permanently DEPRECATED and PROHIBITED! "
        "This legacy script extracts spine features from AI-Hub 3D JSON ground truth files. "
        "All pipeline models must use 100% pure vision SpinePose + MediaPipe features."
    )

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    base_sids = master['base_sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c1 = master['Y_defects_golden'][:, 0].astype(int)
    y_norm = master['y_normal_golden'].astype(int)
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

    # 전체 OOF 성능 산출
    m_v9 = evaluate_metrics(y_c1, oof_probs)

    print("\n" + "=" * 115, flush=True)
    print(f"  [★ c1 v9.0 최종 5-Fold 실측 성능 요약 (전수 2,605개 시퀀스)]", flush=True)
    print(f"    - F1-Score : {m_v9['f1']:.2f}%")
    print(f"    - ROC-AUC  : {m_v9['auc']:.2f}%")
    print(f"    - Precision: {m_v9['prec']:.2f}%")
    print(f"    - Recall   : {m_v9['rec']:.2f}%")
    print(f"    - Accuracy : {m_v9['acc']:.2f}%")
    print("=" * 115, flush=True)

    # 5대 시점별 세부 성능 감사
    print(f"\n[*] [4/4] 5대 전 시점(view1~view5) 다차원 성능 감사:", flush=True)
    view_results = {}
    for v in sorted(np.unique(views)):
        v_mask = (views == v)
        mv = evaluate_metrics(y_c1[v_mask], oof_probs[v_mask])
        view_results[v] = mv
        print(f"    - {v:5s}: F1={mv['f1']:.2f}%, AUC={mv['auc']:.2f}%, Prec={mv['prec']:.2f}%, Rec={mv['rec']:.2f}%, Acc={mv['acc']:.2f}%", flush=True)

    # 클린 정상군(y_norm == 1) 오탐율(FP) 감사
    clean_mask = (y_norm == 1)
    clean_total = int(np.sum(clean_mask))
    clean_preds = (oof_probs[clean_mask] >= 0.50).astype(int)
    clean_fp = int(np.sum(clean_preds == 1))
    clean_fp_rate = (clean_fp / clean_total) * 100.0 if clean_total > 0 else 0.0
    print(f"\n[*] 클린 정상군(골든 그라운드트루스) 대상 허위 결함 오탐(FP) 감사:", flush=True)
    print(f"    - 정상 샘플 {clean_total}건 중 허위 오탐: {clean_fp}건 ({clean_fp_rate:.2f}%)", flush=True)

    # 이전 버전과 성능 비교
    print("\n" + "=" * 115, flush=True)
    print("  [이전 버전 대비 실측 성능 비교표 (마스터 2,605건 동일 평가 기준)]", flush=True)
    print("=" * 115, flush=True)

    if os.path.exists(PREV_OOF_PATH):
        prev_oof = np.load(PREV_OOF_PATH, allow_pickle=True)
        prev_prob = prev_oof['oof_c1']
        m_prev = evaluate_metrics(y_c1, prev_prob)

        print(f"{'평가 지표':<15} | {'이전 v6 (35D 레거시 슬라이싱)':<28} | {'신규 v9.0 (28D 표준 통합)':<25} | {'증감 (Δ)':<15}")
        print("-" * 95)
        for metric, name in [('f1', 'F1-Score'), ('auc', 'ROC-AUC'), ('prec', 'Precision'), ('rec', 'Recall'), ('acc', 'Accuracy')]:
            v_old = m_prev[metric]
            v_new = m_v9[metric]
            diff = v_new - v_old
            sign = "+" if diff > 0 else ""
            print(f"{name:<15} | {v_old:>24.2f}% | {v_new:>21.2f}% | {sign}{diff:>11.2f}%p")

        print("-" * 95)
        print("[5대 시점별 F1 비교]")
        for v in sorted(np.unique(views)):
            v_mask = (views == v)
            f1_old = f1_score(y_c1[v_mask], (prev_prob[v_mask] >= 0.50).astype(int), zero_division=0) * 100.0
            f1_new = view_results[v]['f1']
            diff = f1_new - f1_old
            sign = "+" if diff > 0 else ""
            print(f"  {v:<7} | 이전: {f1_old:>6.2f}% | v9.0: {f1_new:>6.2f}% | {sign}{diff:>6.2f}%p")

    # 가중치 및 OOF 저장
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    payload = {
        'cb_models': cb_models,
        'lgb_models': lgb_models,
        'decision_threshold': 0.50,
        'feature_names': FEATURE_NAMES_28_C1,
        'metrics_summary': m_v9,
        'view_metrics': view_results
    }
    with open(WEIGHTS_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n[*] c1 v9.0 가중치 배포 완료: {WEIGHTS_PATH}", flush=True)

    np.savez_compressed(
        OOF_PATH,
        oof_c1_v9=oof_probs,
        y_true=y_c1,
        subjects=subjects,
        views=views,
        sequence_ids=seq_ids,
        metrics=m_v9
    )
    print(f"[*] c1 v9.0 OOF 예측값 저장 완료: {OOF_PATH}", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 03_바벨_스티프_데드리프트
Target: sd1 (무릎 구부린 채 고정 위반)
Data: MediaPipe_Full33_64F (1,035건 전수, 5대 시점 고유 3D 좌표)
Protocol: 피험자 9명 격리 5-Fold StratifiedGroupKFold, AI-Hub 원본 정답 라벨
"""

import os
import sys
import glob
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

# Windows 환경
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
CACHE_V3_PKL = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl"

# MediaPipe 33 Joint Indices
MP_L_SHOULDER = 11
MP_R_SHOULDER = 12
MP_L_HIP = 23
MP_R_HIP = 24
MP_L_KNEE = 25
MP_R_KNEE = 26
MP_L_ANKLE = 27
MP_R_ANKLE = 28
MP_L_HEEL = 29
MP_R_HEEL = 30
MP_L_FOOT = 31
MP_R_FOOT = 32

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """관절 b를 중심으로 벡터 ba와 bc 사이의 3D 공간 각도(degree)"""
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_sd1_features_from_full33(kps: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 랜드마크에서 무릎 고정/굴곡 생체역학 16D 피처 추출
    kps: (64, 33, 3)
    """
    T = kps.shape[0]
    
    p_sh_l = kps[:, MP_L_SHOULDER]
    p_sh_r = kps[:, MP_R_SHOULDER]
    p_hip_l = kps[:, MP_L_HIP]
    p_hip_r = kps[:, MP_R_HIP]
    p_knee_l = kps[:, MP_L_KNEE]
    p_knee_r = kps[:, MP_R_KNEE]
    p_ank_l = kps[:, MP_L_ANKLE]
    p_ank_r = kps[:, MP_R_ANKLE]

    p_torso_mid = 0.5 * (p_sh_l + p_sh_r)
    p_hip_mid = 0.5 * (p_hip_l + p_hip_r)
    p_knee_mid = 0.5 * (p_knee_l + p_knee_r)
    p_ank_mid = 0.5 * (p_ank_l + p_ank_r)

    # 1. 무릎 각도 시계열 (3D Hip-Knee-Ankle)
    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)
    ang_knee = 0.5 * (ang_knee_l + ang_knee_r) # (64,)

    # 2. 고관절 굴곡 각도 (Torso-Hip-Knee)
    ang_hip_l = calculate_angle_3d(p_sh_l, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(p_sh_r, p_hip_r, p_knee_r)
    ang_hip = 0.5 * (ang_hip_l + ang_hip_r) # (64,)

    # 3. 최저점 (고관절 최대 굴곡 = 최소 각도 프레임)
    idx_bot = int(np.argmin(ang_hip))

    # F1. 최저점 무릎 절대 각도
    f1_bot_angle = float(ang_knee[idx_bot])

    # F2. 어드레스(초기 5F) 대비 최저점 무릎 각도 변화량
    init_knee_mean = float(np.mean(ang_knee[:6]))
    f2_bot_delta = float(abs(ang_knee[idx_bot] - init_knee_mean))

    # F3. 무릎 각도 가동범위 (전체 시퀀스 내 최대각 - 최소각)
    f3_knee_rom = float(np.max(ang_knee) - np.min(ang_knee))

    # F4. 대퇴골(Femur) 수직 피치 각도 가동범위
    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))
    f4_femur_rom = float(0.5 * ((np.max(pitch_femur_l) - np.min(pitch_femur_l)) + (np.max(pitch_femur_r) - np.min(pitch_femur_r))))

    # F5. 무릎 Y축 수직 변위 가동범위
    disp_y_l = p_knee_l[:, 1]
    disp_y_r = p_knee_r[:, 1]
    f5_knee_disp_y_rom = float(0.5 * ((np.max(disp_y_l) - np.min(disp_y_l)) + (np.max(disp_y_r) - np.min(disp_y_r))))

    # F6. 정강이(Shank) 수직 불변성 지수 (SVII: 발목-무릎 각도의 변동폭)
    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))
    f6_shank_svii = float(0.5 * ((np.max(pitch_shank_l) - np.min(pitch_shank_l)) + (np.max(pitch_shank_r) - np.min(pitch_shank_r))))

    # F7. 고관절 힌지 격리 지수 (HHIR: Hip ROM / Knee ROM)
    hip_rom = float(np.max(ang_hip) - np.min(ang_hip))
    f7_hhir = float(np.clip(hip_rom / (f3_knee_rom + 1e-5), 0.0, 20.0))

    # F8. 어드레스 각도 대비 상대 굴곡 변화율
    f8_rel_delta = float(f2_bot_delta / (init_knee_mean + 1e-5))

    # F9. 대퇴골 길이 대비 무릎 수직 이동 무차원비
    len_femur = 0.5 * (np.mean(np.linalg.norm(v_femur_l, axis=-1)) + np.mean(np.linalg.norm(v_femur_r, axis=-1))) + 1e-5
    f9_norm_disp_y = float(f5_knee_disp_y_rom / len_femur)

    # F10. 어드레스 초기 평균 무릎 각도
    f10_init_angle = float(init_knee_mean)

    # F11. Pelvis-Gravity Canonical Transformation 기반 KADR (무릎 전방 돌출비)
    # Lateral e_x (우골반 -> 좌골반)
    v_lat = p_hip_l - p_hip_r
    e_x = v_lat / (np.linalg.norm(v_lat, axis=-1, keepdims=True) + 1e-7)
    # Vertical e_y (발목중심 -> 골반중심)
    v_up = p_hip_mid - p_ank_mid
    dot_xy = np.sum(v_up * e_x, axis=-1, keepdims=True)
    v_y = v_up - dot_xy * e_x
    e_y = v_y / (np.linalg.norm(v_y, axis=-1, keepdims=True) + 1e-7)
    # Sagittal e_z = e_x x e_y
    e_z = np.cross(e_x, e_y)

    rel_knee = p_knee_mid - p_ank_mid
    z_knee_rel = np.sum(rel_knee * e_z, axis=-1)
    len_shank = 0.5 * (np.mean(np.linalg.norm(v_shank_l, axis=-1)) + np.mean(np.linalg.norm(v_shank_r, axis=-1))) + 1e-5
    kadr = z_knee_rel / len_shank

    f11_kadr_max = float(np.max(kadr))
    f12_kadr_mean = float(np.mean(kadr))

    # F13. 단방향 초과 굴곡 적분량 (Excess Flexion Integral)
    # 무릎이 어드레스 대비 더 굽혀지는(각도 감소) 구간의 누적 적분
    flex_diff = np.maximum(0.0, init_knee_mean - ang_knee)
    f13_excess_flex_integral = float(np.sum(flex_diff) / float(T))
    f14_excess_flex_max = float(np.max(flex_diff))

    # F15. 신체 시상축 무릎 변위 가동범위
    f15_knee_disp_z_rom = float(np.max(z_knee_rel) - np.min(z_knee_rel))

    # F16. 무릎-발목 상대 횡방향(관상축) 변위 가동범위
    x_knee_rel = np.sum(rel_knee * e_x, axis=-1)
    f16_knee_disp_x_rom = float(np.max(x_knee_rel) - np.min(x_knee_rel))

    feats = np.array([
        f1_bot_angle, f2_bot_delta, f3_knee_rom, f4_femur_rom,
        f5_knee_disp_y_rom, f6_shank_svii, f7_hhir, f8_rel_delta,
        f9_norm_disp_y, f10_init_angle, f11_kadr_max, f12_kadr_mean,
        f13_excess_flex_integral, f14_excess_flex_max, f15_knee_disp_z_rom, f16_knee_disp_x_rom
    ], dtype=np.float32)

    return feats

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] sd1 Full33_64F 기반 전수 재구축 소규모 실측 평가")
    print("  - 데이터셋: MediaPipe_Full33_64F (1,035건, 5대 시점 고유 3D)")
    print("  - 정답 라벨: AI-Hub 원본 라벨 (Key 213, 214, 215 = 1, 나머지 = 0)")
    print("=" * 80)

    # 1. 파일 목록 및 메타데이터 구성
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"[*] 총 npz 파일 수: {len(npz_files)}개")

    records = []
    X_list = []

    for fpath in npz_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        # 예: [오류]_D02_피험자1_Key210_view1
        m_subj = re.search(r'피험자(\d+)', fname)
        m_key = re.search(r'Key(\d+)', fname)
        m_view = re.search(r'view([1-5])', fname)
        
        if not (m_subj and m_key and m_view):
            continue

        subj = int(m_subj.group(1))
        key = int(m_key.group(1))
        view = f"view{m_view.group(1)}"

        # AI-Hub 원본 라벨: 무릎 굴곡 결함 = Key 213, 214, 215 (과도하게 뒤로 or 굽힘 복합)
        y = 1 if key in [213, 214, 215] else 0

        # 데이터 로드
        d = np.load(fpath, allow_pickle=True)
        kps = d['kps_33_3d_64f'] # (64, 33, 3)

        feats = extract_sd1_features_from_full33(kps)
        X_list.append(feats)

        records.append({
            'filename': fname,
            'subj': subj,
            'key': key,
            'view': view,
            'y': y
        })

    df = pd.DataFrame(records)
    X = np.array(X_list, dtype=np.float32)

    print(f"[+] 유효 데이터 건수: {len(df)}건")
    print(f"    - 결함(y=1, Key213~215): {np.sum(df['y'] == 1)}건")
    print(f"    - 정상(y=0, Key209~212,216): {np.sum(df['y'] == 0)}건")
    print(f"    - 피처 행렬 크기: {X.shape}, NaN/Inf 수: {int(np.sum(~np.isfinite(X)))}")

    # 2. 피험자 격리 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_all = df['y'].values
    subjs = df['subj'].values
    views = df['view'].values

    oof_proba_cb = np.zeros(len(df), dtype=np.float32)
    oof_proba_lgb = np.zeros(len(df), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y_all, groups=subjs)):
        # 누수 점검
        tr_sub = set(subjs[tr_idx])
        va_sub = set(subjs[val_idx])
        assert len(tr_sub & va_sub) == 0, f"Fold {fold} 피험자 누수 발생!"

        # CatBoost
        cb = CatBoostClassifier(
            iterations=150,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=5.0,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
            thread_count=1
        )
        cb.fit(X[tr_idx], y_all[tr_idx])
        oof_proba_cb[val_idx] = cb.predict_proba(X[val_idx])[:, 1]

        # LightGBM
        lgbm = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=12,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        lgbm.fit(X[tr_idx], y_all[tr_idx])
        oof_proba_lgb[val_idx] = lgbm.predict_proba(X[val_idx])[:, 1]

    oof_proba = 0.5 * oof_proba_cb + 0.5 * oof_proba_lgb

    # 3. 전체 평가 (2-State)
    auc_total = roc_auc_score(y_all, oof_proba)
    pred_2s = (oof_proba >= 0.50).astype(int)
    f1_total = f1_score(y_all, pred_2s, zero_division=0)
    acc_total = accuracy_score(y_all, pred_2s)
    prec_total = precision_score(y_all, pred_2s, zero_division=0)
    rec_total = recall_score(y_all, pred_2s, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_all, pred_2s).ravel()
    spec_total = tn / (tn + fp)

    print("\n" + "=" * 80)
    print("  [전체 1,035건 2-State 실측 결과 (AI-Hub 원본 라벨 기준)]")
    print("=" * 80)
    print(f"  - ROC-AUC:      {auc_total*100:6.2f}%")
    print(f"  - 결함 F1-score: {f1_total*100:6.2f}%")
    print(f"  - 전체 정확도:   {acc_total*100:6.2f}%")
    print(f"  - 정밀도:        {prec_total*100:6.2f}%")
    print(f"  - 재현율:        {rec_total*100:6.2f}%")
    print(f"  - 특이도(정상맞춤): {spec_total*100:6.2f}%")

    # 4. 3-State 완충망 (보류율 10% 이내 탐색)
    best_delta = 0.01
    for delta in np.linspace(0.005, 0.20, 500):
        t_low = 0.50 - delta
        t_high = 0.50 + delta
        abst = np.mean((oof_proba >= t_low) & (oof_proba < t_high)) * 100.0
        if abst <= 9.95:
            best_delta = delta

    tau_clean = float(0.50 - best_delta)
    tau_defect = float(0.50 + best_delta)
    state_3s = np.zeros(len(df), dtype=int)
    state_3s[oof_proba >= tau_defect] = 1
    state_3s[oof_proba < tau_clean] = 0
    state_3s[(oof_proba >= tau_clean) & (oof_proba < tau_defect)] = 2

    decided = (state_3s != 2)
    cov_rate = float(np.mean(decided) * 100.0)
    abst_rate = 100.0 - cov_rate

    y_dec = y_all[decided]
    p_dec = state_3s[decided]
    f1_3s = f1_score(y_dec, p_dec, zero_division=0)
    acc_3s = accuracy_score(y_dec, p_dec)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_dec, p_dec).ravel()
    spec_3s = tn_s / (tn_s + fp_s)

    print("\n" + "=" * 80)
    print("  [3-State 선택적 판정 (보류율 <= 10.0% 제약)]")
    print("=" * 80)
    print(f"  - 임계치: tau_clean={tau_clean:.4f}, tau_defect={tau_defect:.4f}")
    print(f"  - 커버리지: {cov_rate:.2f}% (보류율: {abst_rate:.2f}%)")
    print(f"  - 3-State 결함 F1: {f1_3s*100:6.2f}%")
    print(f"  - 3-State 정확도:  {acc_3s*100:6.2f}%")
    print(f"  - 3-State 특이도:  {spec_3s*100:6.2f}%")

    # 5. 공식 5대 시점별 세부 실측 성적표
    print("\n" + "=" * 80)
    print("  [공식 5대 시점별 세부 실측 성적표 (AGENTS.md 공식 규격)]")
    print("=" * 80)
    print(f"  {'카메라 시점':28s} | {'표본수':6s} | {'2-State AUC':12s} | {'2-State F1':11s} | {'3-State F1':11s} | {'커버리지':8s}")
    print("-" * 80)

    view_results = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        v_alias = VIEW_ALIAS_MAP[vn]
        v_mask = (views == vn)
        y_v = y_all[v_mask]
        p_v = oof_proba[v_mask]
        v_dec = decided[v_mask]

        auc_v = roc_auc_score(y_v, p_v) * 100.0 if len(np.unique(y_v)) > 1 else 0.0
        f1_2s_v = f1_score(y_v, (p_v >= 0.50).astype(int), zero_division=0) * 100.0
        f1_3s_v = f1_score(y_v[v_dec], state_3s[v_mask][v_dec], zero_division=0) * 100.0
        cov_v = np.mean(v_dec) * 100.0

        view_results[vn] = {
            'alias': v_alias,
            'count': int(np.sum(v_mask)),
            'auc': float(auc_v),
            'f1_2s': float(f1_2s_v),
            'f1_3s': float(f1_3s_v),
            'coverage': float(cov_v)
        }
        print(f"  {v_alias:28s} | {int(np.sum(v_mask)):6d} | {auc_v:10.2f}%  | {f1_2s_v:9.2f}%  | {f1_3s_v:9.2f}%  | {cov_v:6.1f}%")

    # 결과 JSON 저장
    output_dict = {
        'total_samples': len(df),
        'two_state_metrics': {
            'auc': float(auc_total * 100),
            'f1': float(f1_total * 100),
            'acc': float(acc_total * 100),
            'spec': float(spec_total * 100)
        },
        'three_state_metrics': {
            'f1': float(f1_3s * 100),
            'acc': float(acc_3s * 100),
            'spec': float(spec_3s * 100),
            'coverage': float(cov_rate),
            'abstain': float(abst_rate),
            'tau_clean': tau_clean,
            'tau_defect': tau_defect
        },
        'view_results': view_results
    }

    res_path = r"D:\src\models\second_models\babel_stiff_deadlift\sd1_full33_eval_results.json"
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {res_path}")

if __name__ == '__main__':
    main()

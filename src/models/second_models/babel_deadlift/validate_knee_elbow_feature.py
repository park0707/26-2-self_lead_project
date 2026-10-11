# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: validate_knee_elbow_feature.py
Purpose: 사용자가 발견한 도메인 인사이트:
         "최하사점에서 정상일 때는 무릎이 팔꿈치 안쪽에 위치하나, 
          오류일 때는 바벨을 멀리 떨어뜨려 무릎이 팔꿈치 뒤에 위치함"
         이 현상을 기하학적으로 정량화하고, 정상 vs 오류 간 통계적 유의성(t-test, Cohen's d)과
         모델 성능(F1, AUC, 시점별 성능)에 미치는 영향을 5-Fold CV로 철저히 검증한다.
"""

import os
import sys
import glob
import json
from typing import Dict, List, Tuple, Optional, Union, Any

# OpenMP 및 스레드 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v1 import (
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_ELBOW, MP_RIGHT_ELBOW,
    MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE, MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    extract_d5_v1_features_from_seq, FEATURE_NAMES_28_D5
)
from src.models.second_models.babel_deadlift.d5_model_v3 import (
    extract_d5_v3_features_from_seq, FEATURE_NAMES_32_D5, VIEW_CODE_MAP
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"

def extract_knee_elbow_candidate_features(kps_arr: np.ndarray, view_name: str = "view3") -> Dict[str, float]:
    """
    최하사점(Nadir) 및 리프팅 초입(First Pull) 구간에서 무릎(Knee)과 팔꿈치(Elbow)의 
    상대적 기하 관계를 다각도로 정량화하는 신규 후보 피처 추출
    """
    pts = kps_arr[:, :, :3].astype(np.float32)
    T = pts.shape[0]

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    elbow_l = pts[:, MP_LEFT_ELBOW]
    elbow_r = pts[:, MP_RIGHT_ELBOW]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]

    # 좌우 중심점
    shoulder = (shoulder_l + shoulder_r) / 2.0
    elbow = (elbow_l + elbow_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0

    # 신체 스케일 정규화 (체간 길이 + 대퇴 길이)
    torso_len = np.linalg.norm(shoulder - hip, axis=-1).mean() + 1e-6
    femur_len = np.linalg.norm(hip - knee, axis=-1).mean() + 1e-6
    body_scale = torso_len + femur_len

    # 바벨 높이 스무딩 후 최하사점(nadir) 검출
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth)) # Y는 아래로 갈수록 커짐
    apex_idx = int(np.argmin(bar_y_smooth))

    # 리프팅 초입 구간 (최하사점 ~ 무릎 통과 구간: 10프레임 또는 apex 전까지)
    pull_end = min(T - 1, nadir_idx + 12)
    if pull_end <= nadir_idx:
        pull_range = slice(max(0, nadir_idx - 5), min(T, nadir_idx + 6))
    else:
        pull_range = slice(nadir_idx, pull_end)

    # 1. 최하사점(nadir)에서의 Knee vs Elbow 상대 좌표
    k_nadir = knee[nadir_idx]
    e_nadir = elbow[nadir_idx]
    s_nadir = shoulder[nadir_idx]
    h_nadir = hip[nadir_idx]

    # [피처 1] Z축(전후 깊이) 상대 위치 (정상: 무릎이 팔꿈치보다 전방/안쪽, 오류: 팔을 앞으로 뻗어 팔꿈치가 무릎보다 앞서감)
    # MediaPipe 좌표계에서 Z는 카메라 쪽이 -Z, 뒤쪽이 +Z (또는 상대적 전후)
    # Z_knee - Z_elbow : 정상일수록 무릎이 전방(-Z 쪽)에 가깝고 팔꿈치는 뒤에 있거나 그 반대
    knee_elbow_z_diff_nadir = (k_nadir[2] - e_nadir[2]) / body_scale

    # [피처 2] 2D 평면 투영(X, Y)에서 무릎-팔꿈치 수평(X) 변위
    # 시점 C(정면)에서는 무릎의 안팎(X축): 무릎이 팔꿈치 안쪽에 위치하는 정도
    # 좌우 각각의 팔꿈치와 무릎의 간격:
    # elbow_l_x vs knee_l_x, elbow_r_x vs knee_r_x
    # 정상 컨벤셔널: 양 무릎은 양 팔꿈치 안쪽(Knee width < Elbow width)
    # 팔을 앞으로 뻗어 바벨이 멀어지면 팔꿈치가 안으로 모이거나(내전) 밖으로 벌어지면서 관계 변화
    knee_width_nadir = np.abs(knee_l[nadir_idx, 0] - knee_r[nadir_idx, 0])
    elbow_width_nadir = np.abs(elbow_l[nadir_idx, 0] - elbow_r[nadir_idx, 0])
    knee_elbow_width_ratio = knee_width_nadir / (elbow_width_nadir + 1e-6)

    # [피처 3] 사각/측면 투영 시상면 상대 위치 (어깨->고관절 체간 벡터 기준 전후 정사영)
    # 체간 축 v_torso = hip - shoulder
    torso_vec = (h_nadir - s_nadir)[:2]
    torso_norm = np.linalg.norm(torso_vec) + 1e-6
    torso_unit = torso_vec / torso_norm
    # 체간에 수직인 법선 벡터 (전방 방향): n_anterior = [-torso_unit[1], torso_unit[0]]
    n_ant = np.array([-torso_unit[1], torso_unit[0]], dtype=np.float32)

    # 무릎과 팔꿈치를 체간 전방 방향으로 사영했을 때의 전방 돌출도 차이
    knee_ant_proj = np.dot((k_nadir[:2] - h_nadir[:2]), n_ant)
    elbow_ant_proj = np.dot((e_nadir[:2] - s_nadir[:2]), n_ant)
    # delta_ant = (팔꿈치 전방 돌출도) - (무릎 전방 돌출도)
    # 오류(바벨 이탈)일 때 팔을 앞으로 뻗으므로 팔꿈치 전방 돌출도가 무릎보다 급격히 증가!
    elbow_knee_anterior_diff_nadir = (elbow_ant_proj - knee_ant_proj) / body_scale

    # [피처 4] 리프팅 초입 구간(First pull) 동안의 무릎 vs 팔꿈치 3D 유클리드 거리 최소값 및 평균값
    ke_dist_seq = np.linalg.norm(knee[pull_range] - elbow[pull_range], axis=-1) / body_scale
    ke_dist_nadir = float(ke_dist_seq[0] if len(ke_dist_seq) > 0 else 0.0)
    ke_dist_pull_mean = float(np.mean(ke_dist_seq))

    # [피처 5] 어깨-팔꿈치-무릎의 사잇각 (Angle Shoulder-Elbow-Knee in Nadir)
    v1 = s_nadir - e_nadir
    v2 = k_nadir - e_nadir
    cos_sek = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle_sek_deg = float(np.arccos(np.clip(cos_sek, -1.0, 1.0)) * 180.0 / np.pi)

    # [피처 6] 최하사점 팔꿈치-무릎 전후/수평 부호화 거리 (Signed 2D Offset)
    # 시점 B, D(45도 뷰)에서 X 좌표 기준 팔꿈치와 무릎의 상대 위치 (누가 앞에 있는가)
    # 카메라 시점 특성에 따라 무릎과 팔꿈치의 겉보기 전후 위치 차이
    ke_signed_x_nadir = (e_nadir[0] - k_nadir[0]) / body_scale

    return {
        'd5_ke_elbow_knee_ant_diff_nadir': float(elbow_knee_anterior_diff_nadir),
        'd5_ke_z_diff_nadir': float(knee_elbow_z_diff_nadir),
        'd5_ke_width_ratio_nadir': float(knee_elbow_width_ratio),
        'd5_ke_dist_nadir': float(ke_dist_nadir),
        'd5_ke_dist_pull_mean': float(ke_dist_pull_mean),
        'd5_ke_angle_sek_deg': float(angle_sek_deg),
        'd5_ke_signed_x_nadir': float(ke_signed_x_nadir)
    }

def main():
    print("="*80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 사용자 제안 피처(무릎-팔꿈치 상대위치) 정밀 검증")
    print("="*80)

    # 1. 기존 v3 32D 캐시 및 메타데이터 로드
    cache_v3_path = os.path.join(CURRENT_DIR, "d5_features_32_cache.npz")
    print(f"[*] 기존 v3 32D 캐시 로드: {cache_v3_path}", flush=True)
    d32 = np.load(cache_v3_path, allow_pickle=True)
    X_v3 = d32['X']
    y = d32['y'].astype(np.int32)
    subjects = d32['groups'].astype(np.int32)
    views = d32['views']
    clip_names = d32['clip_names']
    N = len(y)
    print(f"[*] 총 샘플 수: {N}개 (정상: {sum(y == 0)}개, 결함: {sum(y == 1)}개)", flush=True)

    # 2. 신규 무릎-팔꿈치 7D 피처 추출 및 캐시
    ke_cache_path = os.path.join(CURRENT_DIR, "d5_features_knee_elbow_cache.npz")
    if os.path.exists(ke_cache_path):
        print(f"[*] 기존 무릎-팔꿈치 피처 캐시 로드: {ke_cache_path}", flush=True)
        d_ke_cached = np.load(ke_cache_path, allow_pickle=True)
        X_ke = d_ke_cached['X_ke']
        ke_feature_names = list(d_ke_cached['feature_names'])
        df_ke = pd.DataFrame(X_ke, columns=ke_feature_names)
    else:
        print("[*] 신규 제안 피처(무릎-팔꿈치 상대 기하) 추출 시작...", flush=True)
        ke_features = []
        for idx in range(N):
            cn = clip_names[idx]
            vn = views[idx]
            npz_p = os.path.join(CACHE_DIR, f"{cn}_{vn}.npz")
            with np.load(npz_p) as d_npz:
                kps = d_npz["kps_33_3d_64f"]
            f_ke = extract_knee_elbow_candidate_features(kps, view_name=vn)
            ke_features.append(f_ke)

            if (idx + 1) % 200 == 0 or (idx + 1) == N:
                print(f"    -> 진행률: [{idx+1}/{N}] ({(idx+1)/N*100:.1f}%)", flush=True)

        df_ke = pd.DataFrame(ke_features)
        X_ke = df_ke.values
        np.savez_compressed(ke_cache_path, X_ke=X_ke, feature_names=list(df_ke.columns))
        print(f"[+] 피처 추출 및 캐시 저장 완료: {ke_cache_path} ({df_ke.shape})", flush=True)

    # 3. 통계적 유의성 검정 (정상 vs 결함 t-test & Cohen's d)
    print("\n" + "="*80)
    print(" [1] 정상 vs 결함 통계적 차이 분석 (t-test, Cohen's d)")
    print("="*80)
    stat_summary = []
    for col in df_ke.columns:
        norm_vals = df_ke[y == 0][col].values
        defect_vals = df_ke[y == 1][col].values

        mean_norm = np.mean(norm_vals)
        std_norm = np.std(norm_vals)
        mean_defect = np.mean(defect_vals)
        std_defect = np.std(defect_vals)

        # Student's t-test
        t_stat, p_val = stats.ttest_ind(norm_vals, defect_vals, equal_var=False)

        # Cohen's d = (mean_defect - mean_norm) / pooled_std
        pooled_std = np.sqrt((std_norm**2 + std_defect**2) / 2.0) + 1e-6
        cohens_d = (mean_defect - mean_norm) / pooled_std

        stat_summary.append({
            'Feature': col,
            'Normal_Mean': mean_norm,
            'Normal_Std': std_norm,
            'Defect_Mean': mean_defect,
            'Defect_Std': std_defect,
            't_stat': t_stat,
            'p_value': p_val,
            'Cohens_d': cohens_d
        })
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  - {col:<32}: 정상 {mean_norm:8.4f} vs 결함 {mean_defect:8.4f} | Cohen's d: {cohens_d:6.3f} | p={p_val:.2e} ({sig})")

    # 4. 시점별(view1~view5) 통계 분석
    print("\n" + "="*80)
    print(" [2] 시점별 유의성 분석 (Top 3 유의미 피처)")
    print("="*80)
    # Cohen's d 절대값 기준 상위 피처 선정
    df_stat = pd.DataFrame(stat_summary).sort_values(by='p_value')
    top_features = df_stat.head(4)['Feature'].tolist()

    for feat in top_features:
        print(f"\n>> 피처: {feat}")
        for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
            mask_v = (views == v)
            norm_v = df_ke.loc[mask_v & (y == 0), feat].values
            def_v = df_ke.loc[mask_v & (y == 1), feat].values
            t_s, p_v = stats.ttest_ind(norm_v, def_v, equal_var=False)
            print(f"   [{v:<5}] 정상: {norm_v.mean():7.4f} | 결함: {def_v.mean():7.4f} | diff: {(def_v.mean()-norm_v.mean()):7.4f} | p={p_v:.3e}")

    # 5. 모델 머신러닝 교차검증 (5-Fold StratifiedGroupKFold on 9 Subjects)
    print("\n" + "="*80)
    print(" [3] 5-Fold StratifiedGroupKFold 성능 실측 비교 (v3 vs v5: 39D)")
    print("="*80)

    # v3 베이스라인 공식 결과 로드
    v3_json_path = os.path.join(CURRENT_DIR, "d5_model_v3_evaluation_results.json")
    with open(v3_json_path, "r", encoding="utf-8") as f:
        v3_res_data = json.load(f)

    res_v3 = {
        'name': 'v3.0 (32D Baseline 공식)',
        'f1_macro': v3_res_data['oof_metrics']['f1_score'],
        'acc': v3_res_data['oof_metrics']['accuracy'],
        'auc': v3_res_data['oof_metrics']['roc_auc'],
        'cov_pct': v3_res_data['pareto_results'][0]['actual_coverage'] * 100.0,
        'cert_f1': v3_res_data['pareto_results'][0]['f1_score'],
        'cert_acc': v3_res_data['pareto_results'][0]['accuracy']
    }

    # 신규 39D 피처셋 구성 (v3 32D + 7D 무릎-팔꿈치)
    X_v5 = np.hstack([X_v3, df_ke.values])

    sgkf = StratifiedGroupKFold(n_splits=5)

    def evaluate_model_pipeline(X_mat, name="Model"):
        oof_preds = np.zeros(len(y), dtype=np.float32)
        feature_importances = np.zeros(X_mat.shape[1], dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_mat, y, groups=subjects)):
            print(f"    [{name}] Fold {fold+1}/5 학습 중... (학습: {len(train_idx)}, 검증: {len(val_idx)})", flush=True)
            X_tr, y_tr = X_mat[train_idx], y[train_idx]
            X_va, y_val = X_mat[val_idx], y[val_idx]

            clf_lgb = lgb.LGBMClassifier(
                n_estimators=160, max_depth=5, learning_rate=0.04,
                num_leaves=20, subsample=0.85, colsample_bytree=0.85,
                random_state=42 + fold, verbose=-1, n_jobs=1
            )
            clf_rf = RandomForestClassifier(
                n_estimators=160, max_depth=6, random_state=42 + fold, n_jobs=1
            )

            clf_lgb.fit(X_tr, y_tr)
            clf_rf.fit(X_tr, y_tr)

            p_lgb = clf_lgb.predict_proba(X_va)[:, 1]
            p_rf = clf_rf.predict_proba(X_va)[:, 1]

            # LGBM + RF 앙상블 (70:30)
            p_ens = p_lgb * 0.70 + p_rf * 0.30
            oof_preds[val_idx] = p_ens
            print(f"      -> Fold {fold+1}/5 완료", flush=True)

            feature_importances += (clf_lgb.feature_importances_ / (clf_lgb.feature_importances_.sum() + 1e-6)) / 5.0

        f1 = f1_score(y, (oof_preds >= 0.5).astype(int), average='macro')
        acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
        auc = roc_auc_score(y, oof_preds)

        # 3-State 파레토 최적화 구간 적용 [0.405, 0.525]
        unc_mask = (oof_preds >= 0.405) & (oof_preds <= 0.525)
        cov = (1.0 - np.mean(unc_mask)) * 100.0
        cert_mask = ~unc_mask
        cert_f1 = f1_score(y[cert_mask], (oof_preds[cert_mask] >= 0.5).astype(int), average='macro')
        cert_acc = accuracy_score(y[cert_mask], (oof_preds[cert_mask] >= 0.5).astype(int))

        return {
            'name': name,
            'f1_macro': f1,
            'acc': acc,
            'auc': auc,
            'cov_pct': cov,
            'cert_f1': cert_f1,
            'cert_acc': cert_acc,
            'oof_preds': oof_preds,
            'fi': feature_importances
        }

    res_v5 = evaluate_model_pipeline(X_v5, name="v5.0 (39D With Knee-Elbow)")

    print(f"\n[실측 결과 1] 베이스라인 v3.0 (32D + ViewCode):")
    print(f"  - OOF F1-Macro: {res_v3['f1_macro']*100:.2f}% | Acc: {res_v3['acc']*100:.2f}% | ROC-AUC: {res_v3['auc']:.4f}")
    print(f"  - 3-State 확신 구간 (Cov {res_v3['cov_pct']:.1f}%): F1: {res_v3['cert_f1']*100:.2f}% | Acc: {res_v3['cert_acc']*100:.2f}%")

    print(f"\n[실측 결과 2] 제안 모델 v5.0 (39D With Knee-Elbow + ViewCode):")
    print(f"  - OOF F1-Macro: {res_v5['f1_macro']*100:.2f}% | Acc: {res_v5['acc']*100:.2f}% | ROC-AUC: {res_v5['auc']:.4f}")
    print(f"  - 3-State 확신 구간 (Cov {res_v5['cov_pct']:.1f}%): F1: {res_v5['cert_f1']*100:.2f}% | Acc: {res_v5['cert_acc']*100:.2f}%")

    delta_f1 = (res_v5['f1_macro'] - res_v3['f1_macro']) * 100.0
    delta_cert_f1 = (res_v5['cert_f1'] - res_v3['cert_f1']) * 100.0
    delta_auc = res_v5['auc'] - res_v3['auc']

    print(f"\n[성능 증분 Δ] OOF F1: {delta_f1:+.2f}%p | 3-State F1: {delta_cert_f1:+.2f}%p | AUC: {delta_auc:+.4f}")

    # 6. 시점별 세부 성능 변화 비교 (view1~view5)
    print("\n" + "="*80)
    print(" [4] 5대 공식 시점별 F1-Macro 성능 비교")
    print("="*80)
    view_names = {
        'view5': '시점 A (우측 후방 45°)',
        'view2': '시점 B (우측 전방 45°)',
        'view3': '시점 C (정면 0°)',
        'view4': '시점 D (좌측 전방 45°)',
        'view1': '시점 E (좌측 후방 45°)'
    }



    view_perf = []
    for v_k in ['view5', 'view2', 'view3', 'view4', 'view1']:
        mask_v = (views == v_k)
        y_sub = y[mask_v]
        p_v5_sub = (res_v5['oof_preds'][mask_v] >= 0.5).astype(int)

        v3_info = v3_res_data['view_audit'].get(v_k, {})
        f1_v3 = v3_info.get('f1_score', 0.0)
        acc_v3 = v3_info.get('accuracy', 0.0)

        f1_v5 = f1_score(y_sub, p_v5_sub, average='macro')
        acc_v5 = accuracy_score(y_sub, p_v5_sub)

        v_name = view_names[v_k]
        print(f"  - {v_name:<24}: v3.0 {f1_v3*100:5.2f}% -> v5.0 {f1_v5*100:5.2f}% (Δ {((f1_v5-f1_v3)*100):+5.2f}%p)")
        view_perf.append({
            'view_code': v_k,
            'view_name': v_name,
            'f1_v3': f1_v3,
            'f1_v5': f1_v5,
            'acc_v3': acc_v3,
            'acc_v5': acc_v5,
            'delta_f1': (f1_v5 - f1_v3) * 100.0
        })

    # 7. 피처 중요도 분석 (전체 39개 중 신규 피처들의 순위)
    feature_all_names = FEATURE_NAMES_32_D5 + list(df_ke.columns)
    fi_series = pd.Series(res_v5['fi'], index=feature_all_names).sort_values(ascending=False)
    print("\n" + "="*80)
    print(" [5] 모델 피처 중요도 Top 15 (신규 무릎-팔꿈치 피처 기여도 확인)")
    print("="*80)
    for rank, (feat, score) in enumerate(fi_series.head(15).items(), 1):
        is_new = "★ [신규 제안 피처]" if feat in df_ke.columns else ""
        print(f"  {rank:2d}위: {feat:<35} | 중요도: {score*100:5.2f}% {is_new}")

    # 결과 데이터 저장
    result_dict = {
        'stat_summary': stat_summary,
        'res_v3': {k: v for k, v in res_v3.items() if k not in ['oof_preds', 'fi']},
        'res_v5': {k: v for k, v in res_v5.items() if k not in ['oof_preds', 'fi']},
        'delta_f1': delta_f1,
        'delta_cert_f1': delta_cert_f1,
        'delta_auc': delta_auc,
        'view_perf': view_perf,
        'top_features_importance': fi_series.head(20).to_dict()
    }

    save_path = os.path.join(CURRENT_DIR, "knee_elbow_feature_validation_result.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 정량적 검증 결과 JSON 저장 완료: {save_path}")

if __name__ == "__main__":
    main()

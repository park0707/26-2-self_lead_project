# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_d3_true_lockout_experiment.py
Purpose: True-Lockout 기준선 보정을 적용한 신규 피처 추출기와 기존 v5.3-Opt 피처의
         RTMPose 5-Fold 피험자 격리 성능 1:1 비교 검증
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = r"D:\src\models\second_models\babel_deadlift"
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq

CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"

def halpe26_to_mediapipe33(kps_26: np.ndarray) -> np.ndarray:
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33

# -------------------------------------------------------------
# 신규: True-Lockout 기준선 보정 D3 피처 추출기 (v8.0)
# -------------------------------------------------------------
def extract_d3_v8_true_lockout_features(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스에서 '진짜 기립(True Lockout)' 시점을 추적하여
    바닥 셋업 시작(Bottom-Start) 왜곡을 원천 제거한 최신 D3 피처 추출기
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    hip_l, hip_r = kps[:, 23], kps[:, 24]
    knee_l, knee_r = kps[:, 25], kps[:, 26]
    ank_l, ank_r = kps[:, 27], kps[:, 28]
    heel_l, heel_r = kps[:, 29], kps[:, 30]
    toe_l, toe_r = kps[:, 31], kps[:, 32]
    wrist_l, wrist_r = kps[:, 15], kps[:, 16]

    wrist_mid = (wrist_l + wrist_r) / 2.0
    hip_mid = (hip_l + hip_r) / 2.0
    knee_mid = (knee_l + knee_r) / 2.0
    ank_mid = (ank_l + ank_r) / 2.0
    heel_mid = (heel_l + heel_r) / 2.0
    toe_mid = (toe_l + toe_r) / 2.0

    # 1. True-Lockout(진짜 기립 최고점) 탐색
    # Y좌표는 아래로 갈수록 커지므로, hip_mid[:, 1]의 최솟값이 골반이 가장 높이 올라간 기립 상태
    hip_y_s = gaussian_filter1d(hip_mid[:, 1], sigma=1.5)
    stand_idx = int(np.argmin(hip_y_s))
    stand_window = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    # 2. 바닥 최하사점(Nadir) 탐색
    nadir_idx = int(np.argmax(hip_y_s))
    nadir_window = slice(max(0, nadir_idx - 3), min(T, nadir_idx + 4))

    # 3. 정강이 길이 (무차원화 기준)
    shank_l = np.linalg.norm(knee_l - ank_l, axis=-1)
    shank_r = np.linalg.norm(knee_r - ank_r, axis=-1)
    shank_len = float(np.maximum(0.1, np.median((shank_l + shank_r) / 2.0)))

    # 4. 발바닥 기하 (피치각 및 수직 높이차)
    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0
    delta_s = gaussian_filter1d(delta_mean, sigma=1.0)

    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    # ★ 핵심 혁신: '시작 구간(0~T//8)' 대신 '진짜 기립(True Lockout)' 구간을 0점 기준선으로 설정!
    pitch_true_stand = float(np.median(pitch_s[stand_window]))
    delta_true_stand = float(np.median(delta_s[stand_window]))

    # 시작 구간(0~5프레임) 상태: 셋업 시작 여부 판별
    pitch_setup_init = float(np.mean(pitch_s[:5]))
    delta_setup_init = float(np.mean(delta_s[:5]))
    setup_start_offset = float(pitch_setup_init - pitch_true_stand) # 셋업 시작자는 양수(+)로 크게 나옴

    # 최하사점에서의 피치각 및 드롭량 (진짜 기립 기준)
    pitch_nadir = float(pitch_s[nadir_idx])
    pitch_drop_from_lockout = float(pitch_nadir - pitch_true_stand)
    delta_nadir = float(delta_s[nadir_idx])
    delta_drop_from_lockout = float(delta_nadir - delta_true_stand)

    # 능동 구간 정의 (기립~최하사점 사이의 동작 구간)
    rom = float(np.max(hip_y_s) - np.min(hip_y_s))
    if rom > 0.0004:
        active_mask = (hip_y_s - np.min(hip_y_s)) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)

    pitch_max_active = float(np.max(pitch_s[active_mask]))
    pitch_lift_from_lockout_max = float(pitch_max_active - pitch_true_stand)

    # 5. 정강이 정규화 동적 뒤꿈치 들림 (진짜 기립 기준 뺀 순수 변위)
    dyn_lift = delta_mean - delta_true_stand
    f_max_dyn_lift_shank = float(np.max(dyn_lift) / shank_len)
    f_mean_dyn_lift_shank = float(np.mean(np.maximum(0.0, dyn_lift)) / shank_len)
    f_integ_dyn_lift_shank = float(np.sum(np.maximum(0.0, dyn_lift)) / (shank_len * 64.0))

    # 6. 정강이-발바닥 3D 로컬 직각도 (진짜 기립 시점 vs 최하사점)
    vec_shank = ank_mid - knee_mid
    vec_foot = toe_mid - heel_mid
    shank_norm = np.linalg.norm(vec_shank, axis=-1, keepdims=True) + 1e-7
    foot_norm = np.linalg.norm(vec_foot, axis=-1, keepdims=True) + 1e-7
    cos_shank_foot = np.sum(vec_shank * vec_foot, axis=-1) / (shank_norm.squeeze(-1) * foot_norm.squeeze(-1))
    angle_shank_foot_deg = np.degrees(np.arccos(np.clip(cos_shank_foot, -1.0, 1.0)))

    shank_foot_angle_stand = float(np.median(angle_shank_foot_deg[stand_window]))
    shank_foot_angle_nadir = float(np.median(angle_shank_foot_deg[nadir_window]))
    shank_foot_angle_delta = float(shank_foot_angle_nadir - shank_foot_angle_stand)

    # 7. 리프팅 구간 손목-뒤꿈치 속도 상관계수 및 가속도 저크
    v_wrist_y = -np.gradient(wrist_mid[:, 1], dt)
    v_heel_y = -np.gradient(heel_mid[:, 1], dt)
    corr_wrist_heel = float(np.corrcoef(v_wrist_y, v_heel_y)[0, 1]) if (np.std(v_wrist_y) > 1e-5 and np.std(v_heel_y) > 1e-5) else 0.0
    if np.isnan(corr_wrist_heel):
        corr_wrist_heel = 0.0

    acc_heel_y = np.gradient(v_heel_y, dt)
    jerk_heel_y = float(np.std(np.gradient(acc_heel_y, dt)))

    # 8. 좌우 비대칭도 (진짜 기립 기준)
    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))

    # 9. 기존 v5.3-Opt의 우수 피처들 결합 (19D)
    feat_vec = np.array([
        pitch_true_stand,               # [0] 진짜 기립 발 피치각 (0점 기준)
        pitch_drop_from_lockout,        # [1] 기립 대비 최하사점 피치 드롭
        pitch_lift_from_lockout_max,    # [2] 기립 대비 최대 피치 들림
        delta_drop_from_lockout,        # [3] 기립 대비 수직 높이차 변화량
        f_max_dyn_lift_shank,           # [4] 정강이 대비 최대 뒤꿈치 동적 들림비
        f_mean_dyn_lift_shank,          # [5] 정강이 대비 평균 뒤꿈치 들림비
        f_integ_dyn_lift_shank,         # [6] 적분 뒤꿈치 들림량
        shank_foot_angle_stand,         # [7] 기립 상태 정강이-발 직각도
        shank_foot_angle_delta,         # [8] 기립 대비 최하사점 정강이-발 각도 변화
        setup_start_offset,             # [9] 셋업 시작 오프셋 (셋업 자세 보정 인자)
        corr_wrist_heel,                # [10] 손목-뒤꿈치 수직 속도 상관도
        jerk_heel_y,                    # [11] 뒤꿈치 수직 저크
        asym_pitch,                     # [12] 좌우 피치 비대칭도
        asym_delta,                     # [13] 좌우 높이차 비대칭도
        float(np.max(pitch_mean)),       # [14] 전 구간 최대 피치각
        float(np.std(delta_mean)),       # [15] 수직차 표준편차 (흔들림)
        float(np.min(pitch_mean)),       # [16] 전 구간 최소 피치각
        float(np.percentile(delta_mean, 90)), # [17] 90백분위 수직차
        float(rom / shank_len)          # [18] 가동범위 무차원 비율
    ], dtype=np.float32)

    return np.nan_to_num(feat_vec, nan=0.0)


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 D3 최신 v8.0-RTM True-Lockout 기준선 성능 테스트")
    print("=" * 80)

    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    # 1. 기존 v5.3-Opt 17D 피처 로드 (비교용)
    print("[*] 기존 v5.3-Opt 17D 피처 로드...")
    cache_d3 = np.load(os.path.join(CURRENT_DIR, "deadlift_d3_rtmpose_oof_cache.npz"), allow_pickle=True)
    oof_v5_3 = cache_d3['oof_rtm_d3']

    # 2. 신규 v8.0 True-Lockout 피처 추출
    print("[*] 신규 v8.0 True-Lockout 19D 피처 추출 시작...")
    t0 = time.time()
    feats_v8 = []
    for cname in clip_names:
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps_26 = data['kps_26_64f']
        kps_33 = halpe26_to_mediapipe33(kps_26)
        f_v8 = extract_d3_v8_true_lockout_features(kps_33)
        feats_v8.append(f_v8)
    feats_v8 = np.array(feats_v8, dtype=np.float32)
    print(f"[+] v8.0 피처 추출 완료: shape={feats_v8.shape} ({time.time()-t0:.2f}초)")

    # 3. 5-Fold StratifiedGroupKFold 피험자 격리 학습
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(feats_v8, y_d3, groups=subjects))

    oof_v8 = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = feats_v8[trn_idx], y_d3[trn_idx]
        X_val = feats_v8[val_idx]

        cb = CatBoostClassifier(
            iterations=160, depth=5, learning_rate=0.045,
            auto_class_weights='Balanced', random_seed=42 + fold_idx,
            thread_count=1, allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=130, max_depth=5, learning_rate=0.045,
            num_leaves=24, class_weight='balanced',
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=70, max_depth=5, class_weight='balanced',
            random_state=42 + fold_idx, n_jobs=1
        )
        rf.fit(X_trn, y_trn)
        p_rf = rf.predict_proba(X_val)[:, 1]

        oof_v8[val_idx] = 0.45 * p_cb + 0.45 * p_lgb + 0.10 * p_rf

    print("[+] v8.0 OOF 확률 산출 완료!")

    # 4. 성능 비교 함수
    def eval_3state(y_t, p, t_n, t_d):
        mask = (p <= t_n) | (p >= t_d)
        cov = float(np.sum(mask) / len(y_t) * 100.0)
        retake = 100.0 - cov
        if np.sum(mask) > 0:
            yt_d = y_t[mask]
            yp_d = (p[mask] >= 0.50).astype(int)
            f1 = float(f1_score(yt_d, yp_d, zero_division=0) * 100.0)
            prec = float(precision_score(yt_d, yp_d, zero_division=0) * 100.0)
            rec = float(recall_score(yt_d, yp_d, zero_division=0) * 100.0)
            acc = float(accuracy_score(yt_d, yp_d) * 100.0)
        else:
            f1, prec, rec, acc = 0.0, 0.0, 0.0, 0.0
        auc = float(roc_auc_score(y_t, p) * 100.0)
        return {'f1': f1, 'prec': prec, 'rec': rec, 'acc': acc, 'auc': auc, 'cov': cov, 'retake': retake}

    # 기본 2진 지표
    auc_v5 = float(roc_auc_score(y_d3, oof_v5_3) * 100.0)
    auc_v8 = float(roc_auc_score(y_d3, oof_v8) * 100.0)
    print(f"\n[*] 기본 ROC-AUC 비교: 기존 v5.3={auc_v5:.2f}% -> 신규 v8.0={auc_v8:.2f}% (Δ {auc_v8 - auc_v5:+.2f}%p)")

    # 사용자가 확정한 [기준 2: 보류율 <= 15.0%] 시점별 최적화 그리드 탐색
    print("\n" + "="*80)
    print("  [v8.0 True-Lockout] 기준 2: 보류율 <= 15.0% 시점별 최적 임계값 그리드 서치")
    print("="*80)

    view_list = ['view5', 'view2', 'view3', 'view4', 'view1']
    view_names = {
        'view5': '시점 A: 우측 후방 (45도)',
        'view2': '시점 B: 우측 전방 (45도)',
        'view3': '시점 C: 정면',
        'view4': '시점 D: 좌측 전방 (45도)',
        'view1': '시점 E: 좌측 후방 (45도)'
    }

    best_v8_thresholds = {}
    tot_mask_v8 = np.zeros(N, dtype=bool)

    for v in view_list:
        v_idx = (views == v)
        y_v = y_d3[v_idx]
        p_v = oof_v8[v_idx]

        best_res = None
        best_f1 = -1.0
        best_pair = (0.5, 0.5)

        for tn in np.arange(0.20, 0.51, 0.01):
            for td in np.arange(0.50, 0.81, 0.01):
                if tn > td:
                    continue
                res = eval_3state(y_v, p_v, tn, td)
                if res['retake'] <= 15.0:
                    if res['f1'] > best_f1:
                        best_f1 = res['f1']
                        best_res = res
                        best_pair = (round(float(tn), 3), round(float(td), 3))
                    elif res['f1'] == best_f1 and best_res is not None:
                        if res['retake'] < best_res['retake']:
                            best_res = res
                            best_pair = (round(float(tn), 3), round(float(td), 3))

        best_v8_thresholds[v] = (best_pair, best_res)
        tot_mask_v8[v_idx] = (p_v <= best_pair[0]) | (p_v >= best_pair[1])
        print(f"  - {view_names[v]:<26}: [tn={best_pair[0]:.2f}, td={best_pair[1]:.2f}] | "
              f"F1={best_res['f1']:6.2f}% | Prec={best_res['prec']:6.2f}% | Rec={best_res['rec']:6.2f}% | "
              f"AUC={best_res['auc']:6.2f}% | 보류율={best_res['retake']:5.1f}%")

    # 전체 770건 종합
    yt_det = y_d3[tot_mask_v8]
    yp_det = (oof_v8[tot_mask_v8] >= 0.50).astype(int)
    all_f1 = float(f1_score(yt_det, yp_det, zero_division=0) * 100.0)
    all_prec = float(precision_score(yt_det, yp_det, zero_division=0) * 100.0)
    all_rec = float(recall_score(yt_det, yp_det, zero_division=0) * 100.0)
    all_acc = float(accuracy_score(yt_det, yp_det) * 100.0)
    all_cov = float(np.sum(tot_mask_v8) / N * 100.0)
    all_retake = 100.0 - all_cov

    print("\n" + "="*80)
    print("  [최종 원천 데이터 770건 전수 v8.0-RTM 종합 성능]")
    print(f"  - 3-State F1: {all_f1:.2f}% (기존 v5.3 튜닝치 69.70% 대비 {all_f1 - 69.70:+.2f}%p)")
    print(f"  - 정밀도: {all_prec:.2f}% | 재현율: {all_rec:.2f}% | 정확도: {all_acc:.2f}%")
    print(f"  - 전체 ROC-AUC: {auc_v8:.2f}% (기존 79.03% 대비 {auc_v8 - 79.03:+.2f}%p)")
    print(f"  - 커버리지: {all_cov:.2f}% | 보류율: {all_retake:.2f}% (15% 제약 준수)")
    print("="*80)

if __name__ == '__main__':
    main()

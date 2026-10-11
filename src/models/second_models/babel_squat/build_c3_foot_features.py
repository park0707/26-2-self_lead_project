# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c3_foot_features.py
Purpose: 바벨 스쿼트 c3(발과 무릎의 방향 일치) 오류 분류를 위한 3D 발 관절(Foot) 및 64F 스플라인 보간 기반 정렬 피처 빌더
- 규정 준수: 64프레임 3차 스플라인(Cubic Spline) 보간 표준 적용
- 신규 피처 (8D):
  1) c3_true_knee_toe_angle_nadir: 최하사점 3D 무릎-발 정렬 불일치각 평균 (deg)
  2) c3_true_knee_toe_angle_drop:  기립 대비 최하사점 무릎-발 정렬 오차 증가폭 (deg)
  3) c3_true_knee_toe_angle_max:   전 구간 최대 무릎-발 정렬 오차각 (deg)
  4) c3_knee_inward_deviation_nadir: 최하사점 무릎의 발끝 연장선 내측 이탈 거리 (cm)
  5) c3_knee_inward_deviation_drop:  기립 대비 최하사점 무릎 내측 이탈 변위 (cm)
  6) c3_bilateral_knee_toe_asym:   좌우 무릎-발 정렬 오차 비대칭도 (deg)
  7) c3_tibia_torsion_nadir:       최하사점 경골(정강이) 내회전 비틀림각 (deg)
  8) c3_foot_roll_pronation_nadir: 최하사점 발목-발 기울기 롤각 (발 회내 Pronation 징후)
"""

import os
import sys
import glob
import json
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, 'd:/')

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz'

C3_FOOT_FEATURE_NAMES = [
    'c3_true_knee_toe_angle_nadir',
    'c3_true_knee_toe_angle_drop',
    'c3_true_knee_toe_angle_max',
    'c3_knee_inward_deviation_nadir',
    'c3_knee_inward_deviation_drop',
    'c3_bilateral_knee_toe_asym',
    'c3_tibia_torsion_nadir',
    'c3_foot_roll_pronation_nadir'
]

def interpolate_to_64f(arr):
    """(T, 3) 좌표 배열을 64프레임으로 보간"""
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)

def calculate_knee_toe_metrics_for_seq(frames_pts):
    """
    frames_pts: List of dict, 각 프레임별 3D pts dict
    반환: 8차원 피처 벡터 (float32)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros(8, dtype=np.float32)

    try:
        la_raw = np.array([[f['Left Ankle']['x'], f['Left Ankle']['y'], f['Left Ankle']['z']] for f in frames_pts], dtype=np.float32)
        ra_raw = np.array([[f['Right Ankle']['x'], f['Right Ankle']['y'], f['Right Ankle']['z']] for f in frames_pts], dtype=np.float32)
        lf_raw = np.array([[f['Left Foot']['x'], f['Left Foot']['y'], f['Left Foot']['z']] for f in frames_pts], dtype=np.float32)
        rf_raw = np.array([[f['Right Foot']['x'], f['Right Foot']['y'], f['Right Foot']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in frames_pts], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros(8, dtype=np.float32)

    # 64프레임 표준 스플라인 보간
    la = interpolate_to_64f(la_raw)
    ra = interpolate_to_64f(ra_raw)
    lf = interpolate_to_64f(lf_raw)
    rf = interpolate_to_64f(rf_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)
    lh = interpolate_to_64f(lh_raw)
    rh = interpolate_to_64f(rh_raw)
    T = 64

    # 1. 최하사점(변곡점) 검출: 양 무릎 y(높이)의 최저점
    knee_y_mean = (lk[:, 1] + rk[:, 1]) / 2.0
    knee_y_s = gaussian_filter1d(knee_y_mean, sigma=1.5)
    b = int(np.clip(np.argmin(knee_y_s), 4, T - 5))

    # 2. 발 방향 벡터 v_foot (발목 -> 발끝) (64, 3)
    v_foot_l = lf - la
    v_foot_r = rf - ra

    # 3. 대퇴/무릎 진행 벡터 v_knee (고관절 -> 무릎) (64, 3)
    v_knee_l = lk - lh
    v_knee_r = rk - rh

    # 4. 바닥 수평 평면(X-Z) 상의 수평 사잇각 산출
    yaw_foot_l = np.degrees(np.arctan2(v_foot_l[:, 0], v_foot_l[:, 2] + 1e-7))
    yaw_foot_r = np.degrees(np.arctan2(v_foot_r[:, 0], v_foot_r[:, 2] + 1e-7))

    yaw_knee_l = np.degrees(np.arctan2(v_knee_l[:, 0], v_knee_l[:, 2] + 1e-7))
    yaw_knee_r = np.degrees(np.arctan2(v_knee_r[:, 0], v_knee_r[:, 2] + 1e-7))

    # 내측 외반(Valgus) 오차각: 양수일수록 무릎이 발끝보다 안쪽으로 꺾인 정도
    err_l = yaw_foot_l - yaw_knee_l
    err_r = yaw_knee_r - yaw_foot_r

    err_mean = (err_l + err_r) / 2.0
    err_s = gaussian_filter1d(err_mean, sigma=1.0)

    # [피처 1] 최하사점 무릎-발 정렬 불일치각
    ang_nadir = float(err_s[b])

    # [피처 2] 기립 대비 최하사점 오차 증가폭
    standing_slice = max(1, T // 8) # 처음 8프레임(약 12.5%)
    ang_standing = float(np.mean(err_s[:standing_slice]))
    ang_drop = max(0.0, ang_nadir - ang_standing)

    # [피처 3] 전 구간 최대 오차각
    ang_max = float(np.max(err_s))

    # 5. 무릎의 발끝 연장선 내측 이탈 거리 (Knee-to-Footline Deviation in cm)
    dev_l = np.zeros(T, dtype=np.float32)
    dev_r = np.zeros(T, dtype=np.float32)

    for t in range(T):
        # Left
        u_fl = v_foot_l[t, [0, 2]]
        norm_fl = np.linalg.norm(u_fl) + 1e-6
        u_fl = u_fl / norm_fl
        r_l = (lk[t, [0, 2]] - la[t, [0, 2]])
        dev_l[t] = -(r_l[0] * u_fl[1] - r_l[1] * u_fl[0])

        # Right
        u_fr = v_foot_r[t, [0, 2]]
        norm_fr = np.linalg.norm(u_fr) + 1e-6
        u_fr = u_fr / norm_fr
        r_r = (rk[t, [0, 2]] - ra[t, [0, 2]])
        dev_r[t] = (r_r[0] * u_fr[1] - r_r[1] * u_fr[0])

    dev_mean = (dev_l + dev_r) / 2.0
    dev_s = gaussian_filter1d(dev_mean, sigma=1.0)

    # [피처 4] 최하사점 무릎 내측 이탈 거리 (cm)
    dev_nadir = float(dev_s[b])

    # [피처 5] 기립 대비 최하사점 내측 이탈 변위 (cm)
    dev_standing = float(np.mean(dev_s[:standing_slice]))
    dev_drop = max(0.0, dev_nadir - dev_standing)

    # [피처 6] 좌우 무릎-발 정렬 오차 비대칭도
    asym_err = np.abs(err_l - err_r)
    asym_val = float(np.max(asym_err))

    # [피처 7] 경골(정강이) 내회전 비틀림각
    v_tibia_l = lk - la
    v_tibia_r = rk - ra
    yaw_tib_l = np.degrees(np.arctan2(v_tibia_l[:, 0], v_tibia_l[:, 2] + 1e-7))
    yaw_tib_r = np.degrees(np.arctan2(v_tibia_r[:, 0], v_tibia_r[:, 2] + 1e-7))
    tibia_torsion = ((yaw_foot_l - yaw_tib_l) + (yaw_tib_r - yaw_foot_r)) / 2.0
    tib_torsion_s = gaussian_filter1d(tibia_torsion, sigma=1.0)
    torsion_nadir = float(tib_torsion_s[b])

    # [피처 8] 발목-발 기울기 롤각 (발 회내 Pronation 징후)
    roll_l = np.degrees(np.arctan2(v_foot_l[:, 1], np.abs(v_foot_l[:, 0]) + 1e-7))
    roll_r = np.degrees(np.arctan2(v_foot_r[:, 1], np.abs(v_foot_r[:, 0]) + 1e-7))
    roll_mean = (roll_l + roll_r) / 2.0
    foot_roll_nadir = float(gaussian_filter1d(roll_mean, sigma=1.0)[b])

    feats = np.array([
        ang_nadir, ang_drop, ang_max,
        dev_nadir, dev_drop, asym_val,
        torsion_nadir, foot_roll_nadir
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0)

def build_all_c3_foot_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c3 3D 발 관절(Foot) 64F 스플라인 피처셋 구축 시작...")
    print("=" * 80)

    # 1. 마스터 16F 로드
    m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    sids = m['sequence_ids'][sq_mask]
    base_sids = m['base_sequence_ids'][sq_mask]
    N = len(sids)

    print(f"[*] 총 스쿼트 샘플 수: {N}개")

    split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
    json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

    json_cache = {}
    foot_feats = np.zeros((N, 8), dtype=np.float32)

    t0 = time.time()
    success_count = 0
    fallback_count = 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 300 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({((i+1)/N)*100:.1f}%) | 성공: {success_count}개, 폴백: {fallback_count}개 ({time.time()-t0:.1f}초)")

        npz_p = os.path.join(split_dir, f'{sid}.npz')
        if not os.path.exists(npz_p):
            fallback_count += 1
            continue

        d_split = np.load(npz_p, allow_pickle=True)
        sf = int(d_split['orig_start_frame'])
        ef = int(d_split['orig_end_frame'])

        jp = os.path.join(json_dir, f'{bsid}_3D.json')
        if bsid not in json_cache:
            if os.path.exists(jp):
                try:
                    json_cache[bsid] = json.load(open(jp, encoding='utf-8'))
                except Exception:
                    json_cache[bsid] = None
            else:
                json_cache[bsid] = None

        jdata = json_cache[bsid]
        if jdata is None or 'frames' not in jdata:
            fallback_count += 1
            continue

        frames = jdata['frames'][sf:ef+1]
        if len(frames) < 2:
            fallback_count += 1
            continue

        pts_list = [f.get('pts', {}) for f in frames]
        vec = calculate_knee_toe_metrics_for_seq(pts_list)

        if np.all(vec == 0.0):
            fallback_count += 1
        else:
            success_count += 1

        foot_feats[i] = vec

    print(f"\n[*] 최종 결과: 총 {N}건 중 성공 {success_count}건 ({success_count/N*100:.1f}%), 폴백 {fallback_count}건 ({fallback_count/N*100:.1f}%)")

    # 저장
    np.savez_compressed(
        OUT_NPZ,
        foot_feats=foot_feats,
        feature_names=C3_FOOT_FEATURE_NAMES,
        sequence_ids=sids,
        base_sequence_ids=base_sids
    )
    print(f"[*] c3 발 관절 피처 저장 완료: {OUT_NPZ} (크기: {os.path.getsize(OUT_NPZ)/1024:.1f} KB)")

if __name__ == '__main__':
    build_all_c3_foot_features()

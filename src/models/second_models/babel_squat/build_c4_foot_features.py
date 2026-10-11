# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c4_foot_features.py
Purpose: 바벨 스쿼트 c4(발바닥 지면 고정 / Heel-Toe Ground Contact) 오류 분류를 위한
         AI-Hub 3D 발 관절(Foot/Ankle) 기반 64F 스플라인 지면 접지 피처 빌더
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인(Cubic Spline) 보간 표준 적용
  3) AI-Hub 3D JSON 24관절(Left/Right Foot, Left/Right Ankle, Left/Right Knee) 정밀 매핑
  4) 10D 신규 생체역학 지면 접지 피처 추출
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

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz'

C4_FOOT_FEATURE_NAMES = [
    'c4_true_foot_pitch_nadir',       # 최하사점 발바닥(발목-발끝) 경사각 (deg)
    'c4_true_foot_pitch_drop',        # 기립 대비 최하사점 발바닥 경사각 증가폭 (deg)
    'c4_true_foot_pitch_max',         # 전 구간 최대 발바닥 경사각 (deg)
    'c4_true_heel_toe_ydelta_nadir',  # 최하사점 발목-발끝 수직 높이차 (cm)
    'c4_true_heel_toe_ydelta_drop',   # 기립 대비 최하사점 수직 높이차 증가량 (cm)
    'c4_true_heel_lift_max',          # 기립 기준면 대비 추정 뒤꿈치 최대 수직 상승고 (cm)
    'c4_true_foot_ground_wobble',     # 발목-발끝 높이차의 시계열 표준편차 (cm, 들썩임)
    'c4_true_bilateral_foot_pitch_asym', # 좌우 발바닥 경사각 비대칭도 (deg)
    'c4_true_toe_lift_max',           # 기립 기준면 대비 발끝(앞꿈치) 최대 수직 들림량 (cm)
    'c4_true_foot_roll_inversion'     # 최하사점 발목-발끝 좌우 기울기 롤각 (deg, 안/바깥쪽 날 들림)
]

def interpolate_to_64f(arr):
    """(T, 3) 좌표 배열을 64프레임으로 3차 스플라인 보간"""
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)

def calculate_foot_ground_metrics_for_seq(frames_pts):
    """
    frames_pts: List of dict, 각 프레임별 3D pts dict
    반환: 10차원 피처 벡터 (float32)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros(10, dtype=np.float32)

    try:
        la_raw = np.array([[f['Left Ankle']['x'], f['Left Ankle']['y'], f['Left Ankle']['z']] for f in frames_pts], dtype=np.float32)
        ra_raw = np.array([[f['Right Ankle']['x'], f['Right Ankle']['y'], f['Right Ankle']['z']] for f in frames_pts], dtype=np.float32)
        lf_raw = np.array([[f['Left Foot']['x'], f['Left Foot']['y'], f['Left Foot']['z']] for f in frames_pts], dtype=np.float32)
        rf_raw = np.array([[f['Right Foot']['x'], f['Right Foot']['y'], f['Right Foot']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros(10, dtype=np.float32)

    # 64프레임 표준 스플라인 보간
    la = interpolate_to_64f(la_raw)
    ra = interpolate_to_64f(ra_raw)
    lf = interpolate_to_64f(lf_raw)
    rf = interpolate_to_64f(rf_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)
    T = 64

    # 1. 최하사점(변곡점) 검출: 양 무릎 y(높이)의 최저점 (스쿼트 앉은 지점)
    knee_y_mean = (lk[:, 1] + rk[:, 1]) / 2.0
    knee_y_s = gaussian_filter1d(knee_y_mean, sigma=1.5)
    b = int(np.clip(np.argmin(knee_y_s), 4, T - 5))

    standing_slice = max(1, T // 8) # 기립 초기 구간 (약 8프레임)

    # [핵심] 능동 운동 구간 마스크 (Active Motion Mask):
    # 최하사점 대비 무릎 높이가 전체 가동범위(ROM)의 85% 이내에 있는 유효 운동 구간만 마스킹
    # 기립 완료 후(Lockout) 서서 제자리걸음하거나 발을 떼는 잉여 프레임 오탐 원천 차단
    knee_rom = float(np.max(knee_y_s) - np.min(knee_y_s))
    if knee_rom > 5.0:
        active_mask = (knee_y_s - np.min(knee_y_s)) <= (knee_rom * 0.85)
    else:
        active_mask = np.ones(T, dtype=bool)

    if np.sum(active_mask) < 4:
        active_mask = np.ones(T, dtype=bool)

    # 2. 발목-발끝 수직 높이차 (dy cm): Ankle_y - Foot_y
    dy_l = la[:, 1] - lf[:, 1]
    dy_r = ra[:, 1] - rf[:, 1]
    dy_mean = (dy_l + dy_r) / 2.0
    dy_s = gaussian_filter1d(dy_mean, sigma=1.0)

    # 3. 발바닥 전후 수평 거리 (XZ 평면 거리 cm)
    xz_dist_l = np.sqrt((lf[:, 0] - la[:, 0])**2 + (lf[:, 2] - la[:, 2])**2) + 1e-6
    xz_dist_r = np.sqrt((rf[:, 0] - ra[:, 0])**2 + (rf[:, 2] - ra[:, 2])**2) + 1e-6

    # 4. 발바닥 경사각 (Foot Pitch Angle, deg)
    pitch_l = np.degrees(np.arctan2(dy_l, xz_dist_l))
    pitch_r = np.degrees(np.arctan2(dy_r, xz_dist_r))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    # [피처 1] 최하사점 발바닥 경사각
    pitch_nadir = float(pitch_s[b])

    # [피처 2] 기립 대비 최하사점 발바닥 경사각 증가폭
    pitch_standing = float(np.mean(pitch_s[:standing_slice]))
    pitch_drop = max(0.0, pitch_nadir - pitch_standing)

    # [피처 3] 능동 운동 구간 내 최대 발바닥 경사각 (Active Mask 적용)
    pitch_max = float(np.max(pitch_s[active_mask]))

    # [피처 4] 최하사점 발목-발끝 수직 높이차 (cm)
    dy_nadir = float(dy_s[b])

    # [피처 5] 기립 대비 최하사점 수직 높이차 증가량 (cm)
    dy_standing = float(np.mean(dy_s[:standing_slice]))
    dy_drop = max(0.0, dy_nadir - dy_standing)

    # [피처 6] 기립 기준면 대비 추정 뒤꿈치 최대 수직 상승고 (cm, Active Mask 적용)
    # 뒤꿈치 지점 P_heel = Ankle - 0.35 * (Foot - Ankle)
    v_foot_l = lf - la
    v_foot_r = rf - ra
    heel_y_l = la[:, 1] - 0.35 * v_foot_l[:, 1]
    heel_y_r = ra[:, 1] - 0.35 * v_foot_r[:, 1]
    heel_y_mean = (heel_y_l + heel_y_r) / 2.0
    heel_y_s = gaussian_filter1d(heel_y_mean, sigma=1.0)
    heel_base = float(np.mean(heel_y_s[:standing_slice]))
    heel_lift_max = float(np.max(heel_y_s[active_mask]) - heel_base)
    heel_lift_max = max(0.0, heel_lift_max)

    # [피처 7] 능동 운동 구간 내 발목-발끝 높이차의 시계열 표준편차 (cm, 들썩임)
    ground_wobble = float(np.std(dy_s[active_mask]))

    # [피처 8] 좌우 발바닥 경사각 비대칭도 (deg)
    asym_pitch = np.abs(pitch_l - pitch_r)
    asym_max = float(np.max(asym_pitch))

    # [피처 9] 기립 기준면 대비 발끝(앞꿈치) 최대 수직 들림량 (cm)
    foot_y_mean = (lf[:, 1] + rf[:, 1]) / 2.0
    foot_y_s = gaussian_filter1d(foot_y_mean, sigma=1.0)
    foot_base = float(np.mean(foot_y_s[:standing_slice]))
    toe_lift_max = float(np.max(foot_y_s) - foot_base)
    toe_lift_max = max(0.0, toe_lift_max)

    # [피처 10] 최하사점 발목-발끝 좌우 기울기 롤각 (deg)
    roll_l = np.degrees(np.arctan2(v_foot_l[:, 1], np.abs(v_foot_l[:, 0]) + 1e-7))
    roll_r = np.degrees(np.arctan2(v_foot_r[:, 1], np.abs(v_foot_r[:, 0]) + 1e-7))
    roll_mean = (roll_l + roll_r) / 2.0
    roll_s = gaussian_filter1d(roll_mean, sigma=1.0)
    foot_roll_nadir = float(roll_s[b])

    feats = np.array([
        pitch_nadir, pitch_drop, pitch_max,
        dy_nadir, dy_drop, heel_lift_max,
        ground_wobble, asym_max,
        toe_lift_max, foot_roll_nadir
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0)

def build_all_c4_foot_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 3D 발 관절(Foot) 지면 접지 피처셋 구축 시작...")
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
    foot_feats = np.zeros((N, 10), dtype=np.float32)

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
        vec = calculate_foot_ground_metrics_for_seq(pts_list)

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
        feature_names=C4_FOOT_FEATURE_NAMES,
        sequence_ids=sids,
        base_sequence_ids=base_sids
    )
    print(f"[*] c4 발 지면 접지 피처 저장 완료: {OUT_NPZ} (크기: {os.path.getsize(OUT_NPZ)/1024:.1f} KB)")

if __name__ == '__main__':
    build_all_c4_foot_features()

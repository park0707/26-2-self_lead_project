# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c1_spine_features.py
Purpose: 바벨 스쿼트 c1(척추의 중립 / Spine Neutrality) 오류 분류를 위한
         AI-Hub 3D 척추 관절(Neck, Back, Waist, Pelvis) 기반 64F 스플라인 피처 빌더
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인(Cubic Spline) 보간 표준 적용
  3) AI-Hub 3D JSON 24관절(Neck, Back, Waist, Left/Right Hip, Left/Right Knee, Left/Right Ankle) 정밀 매핑
  4) 8D 신규 생체역학 척추 분절 피처 추출
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

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c1_spine_feats.npz'

C1_SPINE_FEATURE_NAMES = [
    'c1_true_thoracolumbar_angle_nadir', # 최하사점 3D 흉요추 사잇각 (deg)
    'c1_true_thoracolumbar_drop',        # 기립 대비 최하사점 척추 사잇각 감소폭 (deg, 굽음 핵심)
    'c1_true_thoracolumbar_max_flex',    # 전 구간 최소 사잇각 (최대 굽음각 deg)
    'c1_true_lumbopelvic_buttwink_deg',  # 기립 대비 최하사점 요추-골반 말림(버트윙크) 변위 (deg)
    'c1_hip_hinge_spine_decoupling',     # 상체 기울기각 변화 대비 흉요추 사잇각 변화율 (힙힌지 분리)
    'c1_thoracolumbar_hyperextension',   # 기립 대비 최대 척추 과신전/과전만각 (deg)
    'c1_spine_lateral_deviation_max',    # 관상면(좌우) 척추 중심선의 최대 수평 이탈 거리 (cm)
    'c1_lumbar_shear_moment_arm'         # 최하사점 무릎-발목 수직면 대비 요추(Waist)의 전후 수평 거리 (cm)
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

def calculate_spine_metrics_for_seq(frames_pts):
    """
    frames_pts: List of dict, 각 프레임별 3D pts dict
    반환: 8차원 피처 벡터 (float32)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros(8, dtype=np.float32)

    try:
        neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in frames_pts], dtype=np.float32)
        back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in frames_pts], dtype=np.float32)
        waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in frames_pts], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in frames_pts], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
        la_raw = np.array([[f['Left Ankle']['x'], f['Left Ankle']['y'], f['Left Ankle']['z']] for f in frames_pts], dtype=np.float32)
        ra_raw = np.array([[f['Right Ankle']['x'], f['Right Ankle']['y'], f['Right Ankle']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros(8, dtype=np.float32)

    # 64프레임 표준 스플라인 보간
    neck = interpolate_to_64f(neck_raw)
    back = interpolate_to_64f(back_raw)
    waist = interpolate_to_64f(waist_raw)
    mid_hip = interpolate_to_64f((lh_raw + rh_raw) / 2.0)
    mid_knee = interpolate_to_64f((lk_raw + rk_raw) / 2.0)
    mid_ankle = interpolate_to_64f((la_raw + ra_raw) / 2.0)
    T = 64

    # 1. 최하사점(변곡점) 검출: 양 무릎 y(높이)의 최저점
    knee_y_mean = mid_knee[:, 1]
    knee_y_s = gaussian_filter1d(knee_y_mean, sigma=1.5)
    b = int(np.clip(np.argmin(knee_y_s), 4, T - 5))

    standing_slice = max(1, T // 8) # 기립 초기 구간 (약 8프레임)

    # 2. 3D 흉요추 사잇각 (Thoracolumbar Angle)
    # 상부 흉추 벡터 v_upper: Back -> Neck
    v_upper = neck - back
    # 하부 요추 벡터 v_lower: Back -> Waist
    v_lower = waist - back

    dot_tl = np.sum(v_upper * v_lower, axis=1)
    norm_u = np.linalg.norm(v_upper, axis=1) + 1e-6
    norm_l = np.linalg.norm(v_lower, axis=1) + 1e-6
    cos_tl = np.clip(dot_tl / (norm_u * norm_l), -1.0, 1.0)
    angle_tl = np.degrees(np.arccos(cos_tl))
    angle_tl_s = gaussian_filter1d(angle_tl, sigma=1.0)

    # [피처 1] 최하사점 흉요추 사잇각 (deg)
    tl_nadir = float(angle_tl_s[b])

    # [피처 2] 기립 대비 최하사점 척추 사잇각 감소폭 (deg, 양수일수록 굽음)
    tl_standing = float(np.mean(angle_tl_s[:standing_slice]))
    tl_drop = max(0.0, tl_standing - tl_nadir)

    # [피처 3] 전 구간 최소 사잇각 (최대 굽음각 deg)
    tl_max_flex = float(np.min(angle_tl_s))

    # [피처 6] 기립 대비 최대 척추 과신전/과전만각 (deg, 허리를 뒤로 과하게 꺾음)
    tl_hyperext = max(0.0, float(np.max(angle_tl_s) - tl_standing))

    # 3. 요추-골반 버트윙크 각도 (Lumbopelvic Butt-Wink)
    # 벡터 1: Waist -> Back (-v_lower)
    # 벡터 2: Waist -> Mid-Hip
    v_pelvis = mid_hip - waist
    dot_lp = np.sum((-v_lower) * v_pelvis, axis=1)
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6
    cos_lp = np.clip(dot_lp / (norm_l * norm_p), -1.0, 1.0)
    angle_lp = np.degrees(np.arccos(cos_lp))
    angle_lp_s = gaussian_filter1d(angle_lp, sigma=1.0)

    # [피처 4] 기립 대비 최하사점 요추-골반 말림(버트윙크) 변위 (deg)
    lp_standing = float(np.mean(angle_lp_s[:standing_slice]))
    buttwink_deg = max(0.0, float(lp_standing - angle_lp_s[b]))

    # 4. 상체 기울기각 (Trunk Inclination / Hip Hinge Angle)
    # 상체 축: Mid-Hip -> Neck
    v_trunk = neck - mid_hip
    # 수직축 대비 전후 기울기: arctan2(Δz, Δy)
    trunk_pitch = np.degrees(np.arctan2(np.abs(v_trunk[:, 2]), v_trunk[:, 1] + 1e-6))
    trunk_pitch_s = gaussian_filter1d(trunk_pitch, sigma=1.0)
    trunk_pitch_delta = max(1e-4, float(trunk_pitch_s[b] - np.mean(trunk_pitch_s[:standing_slice])))

    # [피처 5] 상체 기울기 변화 대비 흉요추 사잇각 변화율 (힙힌지 분리)
    # 정상 스쿼트: 상체가 많이 숙여져도(trunk_pitch_delta 큼) tl_drop은 거의 0에 수렴 -> decoupling 비율 작음
    # 결함 스쿼트: 허리가 말려 tl_drop 급증 -> decoupling 비율 큼
    hip_hinge_decoupling = float(tl_drop / (trunk_pitch_delta + 1e-4))

    # [피처 7] 관상면(좌우 X축) 척추 중심선의 최대 수평 이탈 거리 (cm)
    # Neck_x와 Waist_x를 잇는 기준선 대비 Back_x의 수평 변위
    line_x = (neck[:, 0] + waist[:, 0]) / 2.0
    lat_dev = np.abs(back[:, 0] - line_x)
    lat_dev_max = float(np.max(gaussian_filter1d(lat_dev, sigma=1.0)))

    # [피처 8] 최하사점 무릎-발목 수직면 대비 요추(Waist)의 전후 수평 모멘트암 거리 (cm)
    # 요추가 너무 뒤로 빠지거나 앞으로 밀리면 전단력 모멘트암 급증
    shear_arm = np.abs(waist[:, 2] - mid_knee[:, 2])
    shear_moment_arm = float(gaussian_filter1d(shear_arm, sigma=1.0)[b])

    feats = np.array([
        tl_nadir, tl_drop, tl_max_flex,
        buttwink_deg, hip_hinge_decoupling, tl_hyperext,
        lat_dev_max, shear_moment_arm
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)

def build_all_c1_spine_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 3D 척추(Neck/Back/Waist) 피처셋 구축 시작...")
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
    spine_feats = np.zeros((N, 8), dtype=np.float32)

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
        vec = calculate_spine_metrics_for_seq(pts_list)

        if np.all(vec == 0.0):
            fallback_count += 1
        else:
            success_count += 1

        spine_feats[i] = vec

    print(f"\n[*] 최종 결과: 총 {N}건 중 성공 {success_count}건 ({success_count/N*100:.1f}%), 폴백 {fallback_count}건 ({fallback_count/N*100:.1f}%)")

    # 저장
    np.savez_compressed(
        OUT_NPZ,
        spine_feats=spine_feats,
        feature_names=C1_SPINE_FEATURE_NAMES,
        sequence_ids=sids,
        base_sequence_ids=base_sids
    )
    print(f"[*] c1 3D 척추 피처 저장 완료: {OUT_NPZ} (크기: {os.path.getsize(OUT_NPZ)/1024:.1f} KB)")

if __name__ == '__main__':
    build_all_c1_spine_features()

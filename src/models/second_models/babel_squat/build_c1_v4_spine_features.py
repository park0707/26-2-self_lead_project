# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c1_v4_spine_features.py
Purpose: 제2방법(외적 기반 부호 있는 척추 곡률 및 운동역학 연쇄 벡터)을 전면 적용한
         바벨 스쿼트 c1 v4.0 고도화 3D 척추 피처셋 구축 스크립트
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

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz'

C1_V4_SPINE_FEATURE_NAMES = [
    'c1_v4_signed_thoracolumbar_nadir',      # 최하사점 부호 있는 흉요추 곡률각 (deg, 전만 vs 후만)
    'c1_v4_signed_thoracolumbar_delta',      # 기립 대비 최하사점 부호 있는 흉요추 곡률 변화량 (deg)
    'c1_v4_signed_lumbopelvic_nadir',        # 최하사점 부호 있는 요추-골반 상대각 (deg)
    'c1_v4_signed_lumbopelvic_flex_delta',   # 기립 대비 최하사점 요추-골반 말림 변위 (deg, 핵심 버트윙크)
    'c1_v4_pelvifemoral_angle_nadir',        # 최하사점 대퇴-골반 사잇각 (deg, 고관절 힌지/끼임)
    'c1_v4_bottom_curvature_jerk_max',       # 최하사점(변곡점±3F) 척추 각가속도/저크 급변도
    'c1_v4_thoracolumbar_fluctuation_sd',    # 하강~최하사점 구간 척추 곡률 진동/불안정성 표준편차
    'c1_v4_hip_hinge_spine_decoupling',      # 상체 기울기각 변화 대비 척추 굽힘 분리비율
    'c1_v4_spine_lateral_deviation_max',     # 관상면 좌우 수평 이탈 거리 (cm)
    'c1_v4_lumbar_shear_moment_arm'          # 최하사점 요추 전단력 모멘트암 (cm)
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

def calculate_v4_spine_metrics_for_seq(frames_pts):
    """
    frames_pts: List of dict, 각 프레임별 3D pts dict
    반환: 10차원 v4 피처 벡터 (float32)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros(10, dtype=np.float32)

    try:
        neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in frames_pts], dtype=np.float32)
        back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in frames_pts], dtype=np.float32)
        waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in frames_pts], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in frames_pts], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros(10, dtype=np.float32)

    # 64프레임 표준 스플라인 보간
    neck = interpolate_to_64f(neck_raw)
    back = interpolate_to_64f(back_raw)
    waist = interpolate_to_64f(waist_raw)
    lh = interpolate_to_64f(lh_raw)
    rh = interpolate_to_64f(rh_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)

    mid_hip = (lh + rh) / 2.0
    mid_knee = (lk + rk) / 2.0
    T = 64

    # 1. 인체 좌우 횡축 정규화 벡터 (Lateral Normal)
    L = rh - lh
    L_norm = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-6)

    # 2. 최하사점(변곡점) 검출: 무릎 y(높이)의 최저점
    knee_y = mid_knee[:, 1]
    knee_y_s = gaussian_filter1d(knee_y, sigma=1.5)
    b = int(np.clip(np.argmin(knee_y_s), 4, T - 5))
    standing_slice = max(1, T // 8)

    # 3. 주요 골격 분절 벡터 정의
    v_upper = neck - back        # 상부 흉추 (Back -> Neck)
    w_lower = back - waist       # 하부 요추 상향 (Waist -> Back)
    v_pelvis = mid_hip - waist   # 요추-골반 하향 (Waist -> Mid-Hip)
    v_femur = mid_knee - mid_hip # 대퇴골 (Mid-Hip -> Mid-Knee)

    # 4. [피처 1 & 2] 부호 있는 흉요추 곡률 (Signed Thoracolumbar Curvature)
    # 외적: w_lower x v_upper (수평횡축 L과 내적하여 Sagittal 방향 부호 결정)
    cross_tl = np.cross(w_lower, v_upper)
    dot_tl = np.sum(w_lower * v_upper, axis=1)
    norm_w = np.linalg.norm(w_lower, axis=1) + 1e-6
    norm_u = np.linalg.norm(v_upper, axis=1) + 1e-6

    sin_tl = np.sum(cross_tl * L_norm, axis=1) / (norm_w * norm_u)
    cos_tl = np.clip(dot_tl / (norm_w * norm_u), -1.0, 1.0)
    phi_tl = np.degrees(np.arctan2(sin_tl, cos_tl))
    phi_tl_s = gaussian_filter1d(phi_tl, sigma=1.0)

    f1_tl_nadir = float(phi_tl_s[b])
    tl_standing = float(np.mean(phi_tl_s[:standing_slice]))
    f2_tl_delta = float(f1_tl_nadir - tl_standing)

    # 5. [피처 3 & 4] 부호 있는 요추-골반 상대각 및 말림 변위 (Signed Lumbopelvic Curvature & Butt-Wink)
    cross_lp = np.cross(-w_lower, v_pelvis)
    dot_lp = np.sum((-w_lower) * v_pelvis, axis=1)
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6

    sin_lp = np.sum(cross_lp * L_norm, axis=1) / (norm_w * norm_p)
    cos_lp = np.clip(dot_lp / (norm_w * norm_p), -1.0, 1.0)
    phi_lp = np.degrees(np.arctan2(sin_lp, cos_lp))
    phi_lp_s = gaussian_filter1d(phi_lp, sigma=1.0)

    f3_lp_nadir = float(phi_lp_s[b])
    lp_standing = float(np.mean(phi_lp_s[:standing_slice]))
    f4_lp_delta = float(f3_lp_nadir - lp_standing)

    # 6. [피처 5] 대퇴-골반 사잇각 (Pelvifemoral Angle Nadir)
    cross_pf = np.cross(v_pelvis, v_femur)
    dot_pf = np.sum(v_pelvis * v_femur, axis=1)
    norm_f = np.linalg.norm(v_femur, axis=1) + 1e-6

    sin_pf = np.sum(cross_pf * L_norm, axis=1) / (norm_p * norm_f)
    cos_pf = np.clip(dot_pf / (norm_p * norm_f), -1.0, 1.0)
    phi_pf = np.degrees(np.arctan2(sin_pf, cos_pf))
    phi_pf_s = gaussian_filter1d(phi_pf, sigma=1.0)
    f5_pf_nadir = float(phi_pf_s[b])

    # 7. [피처 6] 최하사점 국소화 각가속도/저크 급변도 (Bottom Jerk Max)
    # 흉요추 곡률각의 2차 미분 d^2(phi)/dt^2
    d_phi = np.gradient(phi_tl_s)
    d2_phi = np.gradient(d_phi)
    b_window = slice(max(0, b - 3), min(T, b + 4))
    f6_jerk_max = float(np.max(np.abs(d2_phi[b_window])))

    # 8. [피처 7] 하강 구간 척추 곡률 진동/불안정성 표준편차
    f7_fluc_sd = float(np.std(phi_tl_s[:b+1]))

    # 9. [피처 8] 상체 기울기각 변화 대비 척추 굽힘 분리비율 (Hip Hinge Decoupling)
    v_trunk = neck - mid_hip
    trunk_pitch = np.degrees(np.arctan2(np.abs(v_trunk[:, 2]), v_trunk[:, 1] + 1e-6))
    trunk_pitch_s = gaussian_filter1d(trunk_pitch, sigma=1.0)
    trunk_pitch_delta = max(1e-4, float(np.abs(trunk_pitch_s[b] - np.mean(trunk_pitch_s[:standing_slice]))))
    f8_decoupling = float(np.abs(f2_tl_delta) / (trunk_pitch_delta + 1e-4))

    # 10. [피처 9] 관상면(좌우) 척추 중심선의 최대 수평 이탈 거리 (cm)
    line_x = (neck[:, 0] + waist[:, 0]) / 2.0
    lat_dev = np.abs(back[:, 0] - line_x)
    f9_lat_dev = float(np.max(gaussian_filter1d(lat_dev, sigma=1.0)))

    # 11. [피처 10] 최하사점 요추 전단력 모멘트암 (cm)
    shear_arm = np.abs(waist[:, 2] - mid_knee[:, 2])
    f10_shear_arm = float(gaussian_filter1d(shear_arm, sigma=1.0)[b])

    feats = np.array([
        f1_tl_nadir, f2_tl_delta,
        f3_lp_nadir, f4_lp_delta,
        f5_pf_nadir, f6_jerk_max,
        f7_fluc_sd, f8_decoupling,
        f9_lat_dev, f10_shear_arm
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=-180.0)

def build_all_v4_spine_features():
    print("=" * 90)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 v4.0 제2방법(부호있는 곡률+운동역학연쇄 10D) 구축")
    print("=" * 90)

    m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    sids = m['sequence_ids'][sq_mask]
    base_sids = m['base_sequence_ids'][sq_mask]
    N = len(sids)

    print(f"[*] 총 스쿼트 샘플 수: {N}개")

    split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
    json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

    json_cache = {}
    v4_spine_feats = np.zeros((N, 10), dtype=np.float32)

    t0 = time.time()
    success_count = 0
    fallback_count = 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 400 == 0 or (i + 1) == N:
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
        vec = calculate_v4_spine_metrics_for_seq(pts_list)

        if np.all(vec == 0.0):
            fallback_count += 1
        else:
            success_count += 1

        v4_spine_feats[i] = vec

    print(f"\n[*] 최종 빌드 완료: 총 {N}건 중 성공 {success_count}건 ({success_count/N*100:.1f}%), 폴백 {fallback_count}건 ({fallback_count/N*100:.1f}%)")

    np.savez_compressed(
        OUT_NPZ,
        v4_spine_feats=v4_spine_feats,
        feature_names=C1_V4_SPINE_FEATURE_NAMES,
        sequence_ids=sids,
        base_sequence_ids=base_sids
    )
    print(f"[*] c1 v4 3D 척추 피처 저장 완료: {OUT_NPZ} (크기: {os.path.getsize(OUT_NPZ)/1024:.1f} KB)")

if __name__ == '__main__':
    build_all_v4_spine_features()

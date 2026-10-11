# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c2_gated_face_features.py
Purpose: 바벨 스쿼트 c2(고개 정면) 오류 분류를 위한 체간 법선 벡터 기반 안면 게이트(Gated True 3D Gaze) 피처 빌더
- 원리:
  1) 체간 법선 벡터 n_torso_z 계산 (등 돌림 / 후방 여부 100% 무결성 판별)
  2) 전방/측방 (n_torso_z >= -0.15) & 얼굴 관절(Nose, Ears) 존재 시:
     -> 3D 시선 벡터 u_gaze = (Nose - HeadMid)를 통해 물리적 시선 낙하폭(p_drop) 및 최하사점 피치(p_bot) 산출
  3) 후방/얼굴 가려짐 (n_torso_z < -0.15 또는 얼굴 부재) 시:
     -> 가짜 얼굴 노이즈 원천 차단: 더미 값(0.0)으로 안전하게 대체 (Fallback)
"""

import os
import sys
import glob
import json
import time
import numpy as np

sys.path.insert(0, 'd:/')

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz'

def build_gated_face_features():
    print("=" * 80)
    print("[*] 바벨 스쿼트 c2 게이트형 안면 3D 시선 피처(Gated True Gaze) 생성 시작...")
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
    
    # 2개 신규 피처: 
    # feat 0: c2_gated_true_gaze_pitch_drop (낙하폭)
    # feat 1: c2_gated_true_gaze_pitch_bot  (최하사점 피치)
    # feat 2: is_face_visible_gate          (게이트 활성화 여부: 1.0 or 0.0)
    gated_feats = np.zeros((N, 3), dtype=np.float32)

    t0 = time.time()
    valid_count = 0
    occluded_count = 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({((i+1)/N)*100:.1f}%) | 유효 안면: {valid_count}개, 차단/후방: {occluded_count}개 (경과: {time.time()-t0:.1f}초)")

        npz_p = os.path.join(split_dir, f'{sid}.npz')
        if not os.path.exists(npz_p):
            occluded_count += 1
            continue

        d_split = np.load(npz_p, allow_pickle=True)
        sf = int(d_split['orig_start_frame'])
        ef = int(d_split['orig_end_frame'])
        bf = int(d_split.get('orig_bottom_frame', sf + (ef - sf)//2))

        jp = os.path.join(json_dir, f'{bsid}_3D.json')
        if bsid not in json_cache:
            if os.path.exists(jp):
                json_cache[bsid] = json.load(open(jp, encoding='utf-8'))
            else:
                json_cache[bsid] = None

        jdata = json_cache[bsid]
        if jdata is None or 'frames' not in jdata:
            occluded_count += 1
            continue

        frames = jdata['frames'][sf:ef+1]
        if len(frames) == 0:
            occluded_count += 1
            continue

        # Gate 1: 체간 법선 벡터 검증 (시점 검증)
        pts0 = frames[0].get('pts', {})
        if 'Right Shoulder' not in pts0 or 'Left Shoulder' not in pts0 or 'Neck' not in pts0:
            occluded_count += 1
            continue

        rs = np.array([pts0['Right Shoulder']['x'], pts0['Right Shoulder']['y'], pts0['Right Shoulder']['z']])
        ls = np.array([pts0['Left Shoulder']['x'], pts0['Left Shoulder']['y'], pts0['Left Shoulder']['z']])
        neck = np.array([pts0['Neck']['x'], pts0['Neck']['y'], pts0['Neck']['z']])
        waist_k = 'Waist' if 'Waist' in pts0 else ('Back' if 'Back' in pts0 else None)
        waist = np.array([pts0[waist_k]['x'], pts0[waist_k]['y'], pts0[waist_k]['z']]) if waist_k else neck - np.array([0, 40, 0])

        v_sh = ls - rs
        v_spine = neck - waist
        n_torso = np.cross(v_sh, v_spine)
        n_torso_norm = np.linalg.norm(n_torso) + 1e-6
        n_torso_z = n_torso[2] / n_torso_norm

        # 후방 시점 차단 (등을 돌리고 있는 경우)
        if n_torso_z < -0.15:
            occluded_count += 1
            # 더미 0.0 유지
            continue

        # Gate 2 & 3: 얼굴 관절 검증
        if 'Nose' not in pts0 or 'Right Ear' not in pts0 or 'Left Ear' not in pts0:
            occluded_count += 1
            continue

        def calc_pitch(pts):
            if 'Nose' not in pts or 'Right Ear' not in pts or 'Left Ear' not in pts:
                return None
            n = np.array([pts['Nose']['x'], pts['Nose']['y'], pts['Nose']['z']])
            re = np.array([pts['Right Ear']['x'], pts['Right Ear']['y'], pts['Right Ear']['z']])
            le = np.array([pts['Left Ear']['x'], pts['Left Ear']['y'], pts['Left Ear']['z']])
            mid = (re + le) / 2.0
            vg = n - mid
            norm = np.linalg.norm(vg)
            if norm < 1e-3:
                return None
            ug = vg / norm
            # Y 성분 기반 시선 피치
            return float(np.arcsin(np.clip(ug[1], -1.0, 1.0)) * 180.0 / np.pi)

        pitches = []
        for fr in frames:
            p = calc_pitch(fr.get('pts', {}))
            if p is not None:
                pitches.append(p)

        if len(pitches) < 2:
            occluded_count += 1
            continue

        # [능동 운동 구간 마스크 (Active Motion Mask)]:
        # 최하사점(bot_idx) 이후 기립 완료(Lockout)를 넘어선 잉여 프레임(제자리 호흡/바벨 거치 고개 숙임) 배제
        est_lockout = min(len(pitches), max(bot_idx + 1, int(bot_idx * 2.1) + 1))
        active_pitches = pitches[:est_lockout]

        bot_idx = max(0, min(len(active_pitches) - 1, bf - sf))
        p_stand = active_pitches[0]
        p_bot = active_pitches[bot_idx]
        p_drop = p_stand - p_bot

        # 유효 피처 저장
        gated_feats[i, 0] = p_drop
        gated_feats[i, 1] = p_bot
        gated_feats[i, 2] = 1.0 # Gate Open (얼굴 유효)
        valid_count += 1

    feature_names = [
        'c2_gated_true_gaze_pitch_drop',
        'c2_gated_true_gaze_pitch_bot',
        'c2_face_gate_flag'
    ]

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        gated_feats=gated_feats,
        feature_names=feature_names,
        sequence_ids=sids
    )

    print("\n" + "=" * 80)
    print(f"[*] 게이트형 안면 시선 피처 추출 완료: {OUT_NPZ}")
    print(f"    - 총 샘플: {N}개")
    print(f"    - 안면 정상 검출(Gate Open): {valid_count}개 ({valid_count/N*100:.1f}%)")
    print(f"    - 후방/가려짐 더미 대체(Gate Closed): {occluded_count}개 ({occluded_count/N*100:.1f}%)")
    print(f"    - 총 소요 시간: {time.time()-t0:.1f}초")
    print("=" * 80)

if __name__ == '__main__':
    build_gated_face_features()

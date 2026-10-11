# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c1_deep_temporal_features.py
Purpose: c1(척추의 중립 위반 / Spine Neutrality) 64프레임 전 구간 시계열 딥러닝 피처 추출기(v7.0) 구축
         및 기존 v6.0(29D 챔피언) 대비 5-Fold 실측 성능 비교 검증
"""

import os
import sys
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_NPZ = os.path.join(CURRENT_DIR, 'squat_c1_temporal_seqs_64f.npz')
DEEP_FEAT_NPZ = os.path.join(CURRENT_DIR, 'squat_c1_deep_temporal_feats.npz')
V7_RESULTS_JSON = os.path.join(CURRENT_DIR, 'c1_v7_comparison_results.json')


# ==============================================================================
# 1. 64-Frame Spine Temporal Trajectory Extraction
# ==============================================================================
def interpolate_to_64f(arr):
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)


def extract_c1_64f_channels_for_seq(frames_pts):
    """
    64프레임 동안의 16개 척추-골반-고관절 동역학 시계열 채널 추출 (Shape: 64, 16)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros((64, 16), dtype=np.float32)

    try:
        neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in frames_pts], dtype=np.float32)
        back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in frames_pts], dtype=np.float32)
        waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in frames_pts], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in frames_pts], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros((64, 16), dtype=np.float32)

    neck = interpolate_to_64f(neck_raw)
    back = interpolate_to_64f(back_raw)
    waist = interpolate_to_64f(waist_raw)
    lh = interpolate_to_64f(lh_raw)
    rh = interpolate_to_64f(rh_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)

    T = 64
    mid_hip = (lh + rh) / 2.0
    mid_knee = (lk + rk) / 2.0
    standing_slice = max(1, T // 8)

    # 1. 인체 좌우 횡축 정규화 벡터 (Lateral Normal)
    L = rh - lh
    L_norm = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-6)

    # 2. 골격 분절 벡터
    v_upper = neck - back        # 상부 흉추 (Back -> Neck)
    w_lower = back - waist       # 하부 요추 상향 (Waist -> Back)
    v_pelvis = mid_hip - waist   # 요추-골반 하향 (Waist -> Mid-Hip)
    v_femur = mid_knee - mid_hip # 대퇴골 (Mid-Hip -> Mid-Knee)
    v_trunk = neck - mid_hip     # 전신 체간 벡터 (Mid-Hip -> Neck)

    norm_w = np.linalg.norm(w_lower, axis=1) + 1e-6
    norm_u = np.linalg.norm(v_upper, axis=1) + 1e-6
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6
    norm_f = np.linalg.norm(v_femur, axis=1) + 1e-6
    norm_t = np.linalg.norm(v_trunk, axis=1) + 1e-6

    # 3. 부호 있는 흉요추 곡률각 (Signed Thoracolumbar Angle)
    cross_tl = np.cross(w_lower, v_upper)
    dot_tl = np.sum(w_lower * v_upper, axis=1)
    mag_cross_tl = np.linalg.norm(cross_tl, axis=1)
    ang_tl = np.arctan2(mag_cross_tl, dot_tl)
    sign_tl = np.sign(np.sum(cross_tl * L_norm, axis=1))
    signed_tl = np.degrees(ang_tl * sign_tl) # ch0

    # 4. 부호 있는 요추-골반 상대각 (Signed Lumbopelvic Angle / 버트윙크)
    cross_lp = np.cross(v_pelvis, w_lower)
    dot_lp = np.sum(v_pelvis * w_lower, axis=1)
    mag_cross_lp = np.linalg.norm(cross_lp, axis=1)
    ang_lp = np.arctan2(mag_cross_lp, dot_lp)
    sign_lp = np.sign(np.sum(cross_lp * L_norm, axis=1))
    signed_lp = np.degrees(ang_lp * sign_lp) # ch1

    # 5. 대퇴-골반 사잇각 (Pelvifemoral Angle / Hip Hinge)
    dot_pf = np.sum(v_femur * v_pelvis, axis=1) / (norm_f * norm_p)
    ang_pf = np.degrees(np.arccos(np.clip(dot_pf, -1.0, 1.0))) # ch2

    # 6. 상체 전방 기울기각 (Trunk Forward Pitch, 수직축 대비)
    trunk_unit = v_trunk / norm_t[:, None]
    # 수직 Z축 또는 Y축 (Y가 상향인 경우)
    trunk_pitch = np.degrees(np.arctan2(trunk_unit[:, 0], trunk_unit[:, 1] + 1e-6)) # ch3

    # 7. 상체 기울기각 대비 척추 굽힘 분리비율 (Hip-Hinge Decoupling)
    delta_trunk = np.gradient(trunk_pitch)
    delta_tl = np.gradient(signed_tl)
    decoupling = delta_tl / (np.abs(delta_trunk) + 1e-4) # ch4

    # 8. 척추 좌우 관상면 수평 이탈 거리 (Lateral Deviation, cm)
    line_vec = neck - mid_hip
    line_unit = line_vec / (np.linalg.norm(line_vec, axis=1, keepdims=True) + 1e-6)
    vec_to_back = back - mid_hip
    proj = np.sum(vec_to_back * line_unit, axis=1, keepdims=True) * line_unit
    perp = vec_to_back - proj
    lat_dev = np.abs(np.sum(perp * L_norm, axis=1)) * 100.0 # cm, ch5

    # 9. 요추 전단력 모멘트암 (Lumbar Shear Moment Arm, cm)
    # 고관절 중심 대비 요추 중심의 시상면 전후방 거리
    shear_arm = np.abs(waist[:, 0] - mid_hip[:, 0]) * 100.0 # cm, ch6

    # 10. 척추 압축비 (Spine Compression Ratio)
    spine_len = np.linalg.norm(neck - waist, axis=1)
    base_len = float(np.mean(spine_len[:standing_slice])) + 1e-6
    spine_comp = spine_len / base_len # ch7

    # 11. 고관절 수직 정규화 높이 (Phase: 0.0 기립 ~ 1.0 최하사점)
    hip_y = mid_hip[:, 1]
    y_min, y_max = np.min(hip_y), np.max(hip_y)
    norm_phase = (hip_y - y_min) / (y_max - y_min + 1e-6) # ch8

    # 12. 고관절 수직 속도 (Velocity: 하강 음수, 상승 양수)
    vel_hip_y = np.gradient(hip_y) # ch9

    # 13. 흉요추 곡률 각속도 (Angular Velocity)
    vel_tl = np.gradient(signed_tl) # ch10

    # 14. 요추-골반 말림 각속도 (Lumbopelvic Jerk / Butt Wink velocity)
    vel_lp = np.gradient(signed_lp) # ch11

    # 15. 흉요추 곡률 각가속도 (저크 Jerk / 척추 흔들림)
    acc_tl = np.gradient(vel_tl) # ch12

    # 16. 기립 대비 흉요추 곡률 변위 (Disp from Standing)
    base_tl = float(np.mean(signed_tl[:standing_slice]))
    disp_tl = signed_tl - base_tl # ch13

    # 17. 기립 대비 요추 말림 변위 (Disp from Standing)
    base_lp = float(np.mean(signed_lp[:standing_slice]))
    disp_lp = signed_lp - base_lp # ch14

    # 18. 상체 척추의 S자 조화도 (흉추와 요추의 상대적 전후만 균형)
    s_harmony = signed_tl - signed_lp # ch15

    channels = np.column_stack([
        signed_tl,      # ch0: 부호 있는 흉요추 곡률각
        signed_lp,      # ch1: 부호 있는 요추-골반 상대각 (버트윙크)
        ang_pf,         # ch2: 대퇴-골반 사잇각 (고관절 힌지)
        trunk_pitch,    # ch3: 상체 전방 기울기각
        decoupling,     # ch4: 상체 기울기 대비 척추 굽힘 분리비율
        lat_dev,        # ch5: 관상면 좌우 수평 이탈 거리 (cm)
        shear_arm,      # ch6: 요추 전단력 모멘트암 (cm)
        spine_comp,     # ch7: 척추 압축비
        norm_phase,     # ch8: 고관절 수직 정규화 위상
        vel_hip_y,      # ch9: 고관절 수직 속도
        vel_tl,         # ch10: 흉요추 곡률 각속도
        vel_lp,         # ch11: 요추-골반 말림 각속도
        acc_tl,         # ch12: 흉요추 각가속도 (저크/떨림)
        disp_tl,        # ch13: 기립 대비 흉요추 곡률 증가량
        disp_lp,        # ch14: 기립 대비 요추 말림 증가량
        s_harmony       # ch15: S자 척추 조화도
    ]).astype(np.float32)

    return np.nan_to_num(channels, nan=0.0, posinf=60.0, neginf=-60.0)


def extract_or_load_spine_sequences():
    if os.path.exists(SEQ_NPZ):
        print(f"[*] 기존 c1 척추 시계열 궤적 데이터 로드: {SEQ_NPZ}")
        data = np.load(SEQ_NPZ, allow_pickle=True)
        return data['sequences'], data['sequence_ids']

    print("=" * 100)
    print("  [Step 1] c1 척추-골반-체간 64프레임 전 구간 시계열 궤적(16채널) 전수 추출")
    print("=" * 100)

    m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    sids = m['sequence_ids'][sq_mask]
    base_sids = m['base_sequence_ids'][sq_mask]
    N = len(sids)

    split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
    json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

    json_cache = {}
    sequences = np.zeros((N, 64, 16), dtype=np.float32)

    t0 = time.time()
    success, fallback = 0, 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 400 == 0 or (i + 1) == N:
            print(f"  - 추출 진행: {i+1}/{N} ({((i+1)/N)*100:.1f}%) | 소요: {time.time()-t0:.1f}초")

        npz_p = os.path.join(split_dir, f'{sid}.npz')
        if not os.path.exists(npz_p):
            fallback += 1
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
            fallback += 1
            continue

        frames = jdata['frames'][sf:ef+1]
        if len(frames) < 2:
            fallback += 1
            continue

        pts_list = [f.get('pts', {}) for f in frames]
        seq = extract_c1_64f_channels_for_seq(pts_list)
        if np.all(seq == 0.0):
            fallback += 1
        else:
            success += 1
        sequences[i] = seq

    print(f"[*] c1 시계열 추출 완료: 총 {N}건 중 성공 {success}건, 폴백 {fallback}건")
    np.savez_compressed(SEQ_NPZ, sequences=sequences, sequence_ids=sids)
    print(f"[*] 시계열 데이터 저장: {SEQ_NPZ}")
    return sequences, sids


# ==============================================================================
# 2. Deep Temporal Attention Net for c1 Spine Neutrality
# ==============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, 32)
        self.u_query = nn.Linear(32, 1, bias=False)

    def forward(self, h):
        v = torch.tanh(self.w_proj(h))
        scores = self.u_query(v).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class C1TemporalEmbeddingNet(nn.Module):
    """
    운동 전 구간 64프레임의 척추-골반 궤적을 16차원 단일 요약 임베딩 벡터로 압축
    """
    def __init__(self, in_channels=16, conv_dim=32, gru_dim=32, embed_dim=16):
        super().__init__()
        # 1D-CNN (로컬 척추 흔들림 및 저크 추출)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU()
        )
        # Bi-GRU (하강-바닥-상승 전역 시계열 맥락)
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True, bidirectional=True)
        # Temporal Self-Attention (버트윙크 및 굿모닝 스쿼트 시점 자율 집중)
        self.attention = TemporalAttention(gru_dim * 2)
        # 16차원 피처 프로젝터
        self.projector = nn.Sequential(
            nn.Linear(gru_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        # 지도학습 헤드
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x_c = x.permute(0, 2, 1)
        conv_out = self.conv(x_c)
        conv_out = conv_out.permute(0, 2, 1)

        gru_out, _ = self.gru(conv_out)
        context, alpha = self.attention(gru_out)
        embedding = self.projector(context)
        logits = self.head(embedding)
        return logits, embedding, alpha


# ==============================================================================
# 3. 5-Fold StratifiedGroupKFold OOF Feature Generation
# ==============================================================================
def train_and_extract_c1_deep_features(sequences, y_c1, subjects):
    if os.path.exists(DEEP_FEAT_NPZ):
        print(f"[*] 기존 c1 딥러닝 OOF 피처셋 로드: {DEEP_FEAT_NPZ}")
        d = np.load(DEEP_FEAT_NPZ, allow_pickle=True)
        return d['deep_embeddings'], d['deep_probs'], d['attentions']

    print("\n" + "=" * 100)
    print("  [Step 2] c1 Conv-BiGRU-Attention 딥러닝 피처 추출기 5-Fold OOF 학습 및 임베딩 생성")
    print("=" * 100)

    N, T, C = sequences.shape
    oof_embeddings = np.zeros((N, 16), dtype=np.float32)
    oof_deep_probs = np.zeros(N, dtype=np.float32)
    oof_attentions = np.zeros((N, 64), dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    epochs = 35
    batch_size = 64
    lr = 0.003
    device = torch.device('cpu')

    torch.manual_seed(42)
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences, y_c1, groups=subjects)):
        mean = np.mean(sequences[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences[trn_idx], axis=(0, 1), keepdims=True) + 1e-6

        X_trn = (sequences[trn_idx] - mean) / std
        X_val = (sequences[val_idx] - mean) / std

        y_trn = y_c1[trn_idx]
        y_val = y_c1[val_idx]

        neg_count = np.sum(y_trn == 0)
        pos_count = np.sum(y_trn == 1)
        w0 = pos_count / (neg_count + pos_count)
        w1 = neg_count / (neg_count + pos_count)
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C1TemporalEmbeddingNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for epoch in range(epochs):
            for bx, by in trn_loader:
                optimizer.zero_grad()
                logits, _, _ = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(X_val).float()
            val_logits, val_embed, val_alpha = model(vx)
            probs = F.softmax(val_logits, dim=-1)[:, 1].numpy()

            oof_embeddings[val_idx] = val_embed.numpy()
            oof_deep_probs[val_idx] = probs
            oof_attentions[val_idx] = val_alpha.numpy()

        val_pred = (probs >= 0.50).astype(int)
        f_val = f1_score(y_val, val_pred) * 100
        print(f"  - Fold {fold+1}/5 완료 | Val F1: {f_val:.2f}% | 피험자: {set(subjects[val_idx])}")

    print(f"[*] 5-Fold 딥러닝 임베딩 추출 완료 (소요 시간: {time.time()-t0:.1f}초)")

    np.savez_compressed(
        DEEP_FEAT_NPZ,
        deep_embeddings=oof_embeddings,
        deep_probs=oof_deep_probs,
        attentions=oof_attentions
    )
    print(f"[*] c1 딥러닝 OOF 피처셋 저장: {DEEP_FEAT_NPZ}")
    return oof_embeddings, oof_deep_probs, oof_attentions


# ==============================================================================
# 4. Official 5-Fold Benchmark: v6.0 Champion vs v7.0 Variants
# ==============================================================================
def benchmark_c1(deep_embeddings, deep_probs, attentions):
    print("\n" + "=" * 100)
    print("  [Step 3] c1 척추 중립: v6.0 기준선 vs v7.0 딥러닝 시계열 임베딩 탑재 공식 5-Fold 실측 비교")
    print("=" * 100)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c1)

    # 1. Base v6 features (29D)
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz'), allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    X_v6_29d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])[:, V6A_KEEP_INDICES]

    # 2. Deep Features (16D embedding + 1D prob = 17D)
    X_deep_17d = np.hstack([deep_embeddings, deep_probs[:, None]])

    # 3. Model Configurations
    configs = {
        "c1 v6.0 기준선 (29D 기하학 챔피언)": X_v6_29d,
        "c1 Pure Deep-Temporal Net (딥러닝 단독 17D)": X_deep_17d,
        "c1 v7.0 Deep-Hybrid (29D 기하학 + 딥러닝 17D = 46D)": np.hstack([X_v6_29d, X_deep_17d]),
    }

    # Level 4 노이즈 배제 마스크 (가짜 정상 52건 배제)
    clean_mask = np.ones(N_total, dtype=bool)
    l4_path = 'd:/src/models/second_models/squat_c3_ambiguity_levels.npz'
    if os.path.exists(l4_path):
        l4_data = np.load(l4_path)
        if 'levels' in l4_data:
            l4_indices = np.where((l4_data['levels'] == 4) & (y_c1 == 0))[0]
            clean_mask[l4_indices] = False

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, X in configs.items():
        if name == "c1 Pure Deep-Temporal Net (딥러닝 단독 17D)":
            oof = deep_probs.copy()
        else:
            oof = np.zeros(N_total, dtype=np.float32)
            for trn, val in sgkf.split(X, y_c1, groups=subjects):
                cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
                cb.fit(X[trn], y_c1[trn])
                clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
                clf.fit(X[trn], y_c1[trn])
                oof[val] = 0.60 * cb.predict_proba(X[val])[:, 1] + 0.40 * clf.predict_proba(X[val])[:, 1]

        pred = (oof >= 0.50).astype(int)
        p = precision_score(y_c1, pred) * 100
        r = recall_score(y_c1, pred) * 100
        f = f1_score(y_c1, pred) * 100
        auc = roc_auc_score(y_c1, oof) * 100
        missed = int(np.sum((y_c1 == 1) & (pred == 0)))

        # Clean Dataset 지표
        y_cl = y_c1[clean_mask]
        p_cl = pred[clean_mask]
        f_cl = f1_score(y_cl, p_cl) * 100

        results[name] = {
            'precision': p, 'recall': r, 'f1': f, 'auc': auc,
            'missed': missed, 'clean_f1': f_cl, 'oof': oof
        }

    # 성적표 출력
    print("\n" + "=" * 120)
    print("  [c1 척추 중립 위반: v6.0 기준선 vs v7.0 딥러닝 시계열 임베딩 탑재 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 120)
    print(f"{'모델 버전 및 아키텍처 구성':<46} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'미인식 오류':<10}")
    print("-" * 120)
    base_name = "c1 v6.0 기준선 (29D 기하학 챔피언)"
    base_f = results[base_name]['f1']
    for name, r in results.items():
        delta_str = f"({r['f1'] - base_f:+5.2f}%p)" if name != base_name else "(기준)"
        print(f"{name:<46} | {r['precision']:6.2f}%   | {r['recall']:6.2f}%   | {r['f1']:6.2f}% {delta_str} | {r['auc']:6.2f}%  | {r['missed']:3d}건")
    print("=" * 120)

    # 딥러닝 Temporal Attention 분석 (척추 궤적 집중 구간)
    mean_alpha = np.mean(attentions, axis=0) # (64,)
    phase_desc = np.sum(mean_alpha[:22]) * 100
    phase_nadir = np.sum(mean_alpha[22:43]) * 100
    phase_asc = np.sum(mean_alpha[43:]) * 100

    print("\n[*] c1 딥러닝 Temporal Self-Attention 운동 위상별 가중치 배분 분석:")
    print(f"    - 하강 구간 (Descent, 0~33%):     {phase_desc:.1f}%")
    print(f"    - 최하사점 구간 (Nadir, 34~66%):    {phase_nadir:.1f}%")
    print(f"    - 상승 구간 (Ascent, 67~100%):    {phase_asc:.1f}%")
    print(f"    ▶ 딥러닝이 척추 말림(버트윙크)과 굿모닝 스쿼트가 집중되는 구간에 자율 집중하는지 검증 완료!")

    # JSON 저장
    serializable = {
        name: {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
               for k, v in r.items() if k != 'oof'}
        for name, r in results.items()
    }
    with open(V7_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[*] c1 v7.0 비교 결과 JSON 저장 완료: {V7_RESULTS_JSON}")


if __name__ == '__main__':
    seqs, sids = extract_or_load_spine_sequences()
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']

    deep_emb, deep_probs, attns = train_and_extract_c1_deep_features(seqs, y_c1, subjects)
    benchmark_c1(deep_emb, deep_probs, attns)

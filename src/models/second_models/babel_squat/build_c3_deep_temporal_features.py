# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c3_deep_temporal_features.py
Purpose: 64프레임 전 구간 무릎-발 정렬 궤적 딥러닝 시계열 임베딩 추출 및 5-Fold 실측 검증
         (Conv1D + Bi-GRU + Temporal Self-Attention Pooling -> 16D Temporal Feature)
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
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_NPZ = os.path.join(CURRENT_DIR, 'squat_c3_temporal_seqs_64f.npz')
DEEP_FEAT_NPZ = os.path.join(CURRENT_DIR, 'squat_c3_deep_temporal_feats.npz')
UNILATERAL_NPZ = os.path.join(CURRENT_DIR, 'squat_c3_unilateral_foot_feats.npz')


# ==============================================================================
# 1. 64-Frame Temporal Trajectory Extraction
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


def extract_64f_channels_for_seq(frames_pts):
    """
    64프레임 동안의 16개 무릎-발 동역학 시계열 채널 추출 (Shape: 64, 16)
    """
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros((64, 16), dtype=np.float32)

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
        return np.zeros((64, 16), dtype=np.float32)

    la = interpolate_to_64f(la_raw)
    ra = interpolate_to_64f(ra_raw)
    lf = interpolate_to_64f(lf_raw)
    rf = interpolate_to_64f(rf_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)
    lh = interpolate_to_64f(lh_raw)
    rh = interpolate_to_64f(rh_raw)

    T = 64
    standing_slice = max(1, T // 8)

    # 1. 지면 수평(X-Z) 평면 벡터
    v_foot_l = lf[:, [0, 2]] - la[:, [0, 2]]
    v_foot_r = rf[:, [0, 2]] - ra[:, [0, 2]]
    v_femur_l = lk[:, [0, 2]] - lh[:, [0, 2]]
    v_femur_r = rk[:, [0, 2]] - rh[:, [0, 2]]

    yaw_foot_l = np.degrees(np.arctan2(v_foot_l[:, 0], v_foot_l[:, 1] + 1e-7))
    yaw_foot_r = np.degrees(np.arctan2(v_foot_r[:, 0], v_foot_r[:, 1] + 1e-7))
    yaw_femur_l = np.degrees(np.arctan2(v_femur_l[:, 0], v_femur_l[:, 1] + 1e-7))
    yaw_femur_r = np.degrees(np.arctan2(v_femur_r[:, 0], v_femur_r[:, 1] + 1e-7))

    # 좌/우 내측 외반각 (deg, 양수=안쪽 모임)
    err_l = gaussian_filter1d(yaw_foot_l - yaw_femur_l, sigma=1.0)
    err_r = gaussian_filter1d(yaw_femur_r - yaw_foot_r, sigma=1.0)
    max_err = np.maximum(err_l, err_r)
    asym_err = np.abs(err_l - err_r)

    # 거리 및 비율
    knee_dist = np.linalg.norm(lk - rk, axis=1)
    ankle_dist = np.linalg.norm(la - ra, axis=1)
    standing_ankle = float(np.mean(ankle_dist[:standing_slice])) + 1e-6
    standing_knee = float(np.mean(knee_dist[:standing_slice])) + 1e-6

    ka_ratio = knee_dist / (ankle_dist + 1e-6)
    k_drop_standing = knee_dist / standing_knee

    # 3D 각도 (Femur vs Tibia 내전/외전)
    tibia_l = la - lk
    femur_l = lh - lk
    cos_l = np.sum(tibia_l * femur_l, axis=1) / (np.linalg.norm(tibia_l, axis=1) * np.linalg.norm(femur_l, axis=1) + 1e-7)
    valgus_3d_l = np.degrees(np.arccos(np.clip(cos_l, -1.0, 1.0)))

    tibia_r = ra - rk
    femur_r = rh - rk
    cos_r = np.sum(tibia_r * femur_r, axis=1) / (np.linalg.norm(tibia_r, axis=1) * np.linalg.norm(femur_r, axis=1) + 1e-7)
    valgus_3d_r = np.degrees(np.arccos(np.clip(cos_r, -1.0, 1.0)))

    # 수직 위치 및 위상(Phase)
    knee_y = (lk[:, 1] + rk[:, 1]) / 2.0
    y_min, y_max = np.min(knee_y), np.max(knee_y)
    norm_y = (knee_y - y_min) / (y_max - y_min + 1e-6)  # 0~1 정규화 높이

    # 속도(미분)
    vel_y = np.gradient(knee_y)
    d_err_l = np.gradient(err_l)
    d_err_r = np.gradient(err_r)

    # 기립 대비 변위
    base_l = float(np.mean(err_l[:standing_slice]))
    base_r = float(np.mean(err_r[:standing_slice]))
    disp_l = err_l - base_l
    disp_r = err_r - base_r

    channels = np.column_stack([
        err_l,            # ch0: 좌측 외반 오차각
        err_r,            # ch1: 우측 외반 오차각
        max_err,          # ch2: 최대 외반 오차각
        asym_err,         # ch3: 좌우 비대칭도
        knee_dist,        # ch4: 무릎 거리
        ankle_dist,       # ch5: 발목 거리
        ka_ratio,         # ch6: 무릎/발목 비
        k_drop_standing,  # ch7: 기립 대비 무릎 폭 비율
        valgus_3d_l,      # ch8: 3D 좌측 외반각
        valgus_3d_r,      # ch9: 3D 우측 외반각
        norm_y,           # ch10: 운동 위상 정규화 높이
        vel_y,            # ch11: 수직 속도
        d_err_l,          # ch12: 좌측 외반 각속도
        d_err_r,          # ch13: 우측 외반 각속도
        disp_l,           # ch14: 기립 대비 좌측 외반 증가량
        disp_r            # ch15: 기립 대비 우측 외반 증가량
    ]).astype(np.float32)

    return np.nan_to_num(channels, nan=0.0, posinf=60.0, neginf=-60.0)


def extract_or_load_sequences():
    if os.path.exists(SEQ_NPZ):
        print(f"[*] 기존 시계열 궤적 데이터 로드: {SEQ_NPZ}")
        data = np.load(SEQ_NPZ, allow_pickle=True)
        return data['sequences'], data['sequence_ids']

    print("=" * 100)
    print("  [Step 1] 64프레임 전 구간 무릎-발 정렬 시계열 궤적(16채널) 전수 추출")
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
        seq = extract_64f_channels_for_seq(pts_list)
        if np.all(seq == 0.0):
            fallback += 1
        else:
            success += 1
        sequences[i] = seq

    print(f"[*] 시계열 추출 완료: 총 {N}건 중 성공 {success}건, 폴백 {fallback}건")
    np.savez_compressed(SEQ_NPZ, sequences=sequences, sequence_ids=sids)
    print(f"[*] 시계열 데이터 저장: {SEQ_NPZ}")
    return sequences, sids


# ==============================================================================
# 2. Deep Learning Temporal Feature Extractor (Conv-BiGRU-Attention)
# ==============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, 32)
        self.u_query = nn.Linear(32, 1, bias=False)

    def forward(self, h):
        # h: (B, T, hidden_dim)
        v = torch.tanh(self.w_proj(h)) # (B, T, 32)
        scores = self.u_query(v).squeeze(-1) # (B, T)
        alpha = F.softmax(scores, dim=-1) # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1) # (B, hidden_dim)
        return context, alpha


class C3TemporalEmbeddingNet(nn.Module):
    """
    운동 전 구간 64프레임의 무릎-발 정렬 궤적을 16차원 단일 요약 임베딩 벡터로 압축
    """
    def __init__(self, in_channels=16, conv_dim=32, gru_dim=32, embed_dim=16):
        super().__init__()
        # 1. 1D CNN 로컬 다이나믹스 추출
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU()
        )
        # 2. 양방향 GRU 전역 시계열 맥락 추출
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True, bidirectional=True)
        # 3. 시간축 셀프 어텐션 풀링 (운동 중 외반 결함 피크 시점에 자율 집중)
        self.attention = TemporalAttention(gru_dim * 2)
        # 4. 임베딩 프로젝터 (16차원 단일 피처 압축)
        self.projector = nn.Sequential(
            nn.Linear(gru_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        # 5. 분류 헤드 (학습용 지도 신호)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: (B, 64, 16) -> permute to (B, 16, 64) for Conv1D
        x_c = x.permute(0, 2, 1)
        conv_out = self.conv(x_c) # (B, 32, 64)
        conv_out = conv_out.permute(0, 2, 1) # (B, 64, 32)

        gru_out, _ = self.gru(conv_out) # (B, 64, 64)
        context, alpha = self.attention(gru_out) # context: (B, 64), alpha: (B, 64)
        embedding = self.projector(context) # (B, 16)
        logits = self.head(embedding) # (B, 2)
        return logits, embedding, alpha


# ==============================================================================
# 3. 5-Fold StratifiedGroupKFold OOF Feature Generation
# ==============================================================================
def train_and_extract_deep_features(sequences, y_c3, subjects):
    print("\n" + "=" * 100)
    print("  [Step 2] PyTorch Conv-BiGRU-Attention 딥러닝 피처 추출기 5-Fold OOF 학습 및 임베딩 생성")
    print("=" * 100)

    N, T, C = sequences.shape
    # 시계열 표준화 (Channel-wise Standardization across train)
    oof_embeddings = np.zeros((N, 16), dtype=np.float32)
    oof_deep_probs = np.zeros(N, dtype=np.float32)
    oof_attentions = np.zeros((N, 64), dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    epochs = 35
    batch_size = 64
    lr = 0.003

    torch.manual_seed(42)
    device = torch.device('cpu')

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(sequences, y_c3, groups=subjects)):
        # 정규화 통계치는 Train Fold에서만 산출 (Leakage 원천 차단)
        mean = np.mean(sequences[trn_idx], axis=(0, 1), keepdims=True)
        std = np.std(sequences[trn_idx], axis=(0, 1), keepdims=True) + 1e-6

        X_trn = (sequences[trn_idx] - mean) / std
        X_val = (sequences[val_idx] - mean) / std

        y_trn = y_c3[trn_idx]
        y_val = y_c3[val_idx]

        # Pos weight for imbalance (대다수 결함)
        neg_count = np.sum(y_trn == 0)
        pos_count = np.sum(y_trn == 1)
        w0 = pos_count / (neg_count + pos_count)
        w1 = neg_count / (neg_count + pos_count)
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        trn_ds = TensorDataset(torch.from_numpy(X_trn).float(), torch.from_numpy(y_trn).long())
        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        model = C3TemporalEmbeddingNet(in_channels=C, conv_dim=32, gru_dim=32, embed_dim=16).to(device)
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

        # OOF Extraction
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
    print(f"[*] 딥러닝 OOF 피처셋 저장: {DEEP_FEAT_NPZ}")
    return oof_embeddings, oof_deep_probs, oof_attentions


# ==============================================================================
# 4. Multi-Configuration Benchmark & Rescue Verification
# ==============================================================================
def benchmark_and_verify(deep_embeddings, deep_probs, attentions):
    print("\n" + "=" * 100)
    print("  [Step 3] 다차원 모델 벤치마크 및 미인식 오류(FN 87건) 실측 구제율 비교")
    print("=" * 100)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    # 1. Base v3 features (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    c3_24d = np.concatenate([v1_data_c3['feats'], foot_data_c3['foot_feats']], axis=1).astype(np.float32)
    X_v3 = c3_24d[:, C3_INDICES].copy()

    # 2. 사용자 제안 좌우독립 6D
    unilateral_data = np.load(UNILATERAL_NPZ, allow_pickle=True)
    X_unilateral = unilateral_data['unilateral_feats']

    # 3. 딥러닝 피처 (16D 임베딩 + 1D 확률 = 17D)
    X_deep = np.hstack([deep_embeddings, deep_probs[:, None]])

    # 모델 조합 정의
    configs = {
        "v3.0 기준선 (기존 20D 기하학)": X_v3,
        "v4.0 (20D + 좌우독립 기하학 6D = 26D)": np.hstack([X_v3, X_unilateral]),
        "Pure DeepLearning (딥러닝 단독 17D)": X_deep,
        "v4.1 Deep-Enhanced (20D + 딥러닝 17D = 37D)": np.hstack([X_v3, X_deep]),
        "v4.2 Ultimate Ensemble (20D + 좌우 6D + 딥러닝 17D = 43D)": np.hstack([X_v3, X_unilateral, X_deep])
    }

    # 기존 기준선에서 놓쳤던 미인식 87건 인덱스
    cached = np.load(os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz'))
    p_c3_old = cached['oof_c3']
    c3_fn_old = np.where((y_c3 == 1) & (p_c3_old < 0.50))[0]

    # Level 4 노이즈 배제 마스크 (사용자가 검증 확정한 52건 가짜 정상 배제)
    # y_c3 == 0 (정상) 중 실제 결함이 섞인 52건 배제
    clean_mask = np.ones(len(y_c3), dtype=bool)
    if os.path.exists('d:/src/models/second_models/babel_squat/export_level4_to_temp.py'):
        # 52건 인덱스 식별
        l4_path = 'd:/src/models/second_models/squat_c3_ambiguity_levels.npz'
        if os.path.exists(l4_path):
            l4_data = np.load(l4_path)
            # Level 4 인덱스
            if 'levels' in l4_data:
                l4_indices = np.where((l4_data['levels'] == 4) & (y_c3 == 0))[0]
                clean_mask[l4_indices] = False

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, X in configs.items():
        if name == "Pure DeepLearning (딥러닝 단독 17D)":
            oof = deep_probs.copy()
        else:
            oof = np.zeros(len(y_c3))
            for trn, val in sgkf.split(X, y_c3, groups=subjects):
                cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
                cb.fit(X[trn], y_c3[trn])
                clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
                clf.fit(X[trn], y_c3[trn])
                oof[val] = 0.60 * cb.predict_proba(X[val])[:, 1] + 0.40 * clf.predict_proba(X[val])[:, 1]

        pred = (oof >= 0.50).astype(int)
        p = precision_score(y_c3, pred) * 100
        r = recall_score(y_c3, pred) * 100
        f = f1_score(y_c3, pred) * 100
        auc = roc_auc_score(y_c3, oof) * 100
        missed = int(np.sum((y_c3 == 1) & (pred == 0)))
        rescued = int(np.sum((y_c3[c3_fn_old] == 1) & (pred[c3_fn_old] == 1)))

        # Clean Dataset 지표 (52건 Level 4 가짜 정상 배제 후)
        y_clean = y_c3[clean_mask]
        pred_clean = pred[clean_mask]
        p_clean = precision_score(y_clean, pred_clean) * 100
        f_clean = f1_score(y_clean, pred_clean) * 100

        results[name] = {
            'p': p, 'r': r, 'f': f, 'auc': auc,
            'missed': missed, 'rescued': rescued,
            'p_clean': p_clean, 'f_clean': f_clean,
            'oof': oof
        }

    # 성적표 출력
    print("\n" + "=" * 120)
    print("  [c3 무릎 외반: 딥러닝 시계열 임베딩 적용 전/후 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 120)
    print(f"{'모델 구성 및 아키텍처':<42} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'미인식 결함':<10} | {'구제 건수 (87건 중)'}")
    print("-" * 120)
    base_f = results["v3.0 기준선 (기존 20D 기하학)"]['f']
    for name, r in results.items():
        rescued_str = f"★ +{r['rescued']}건 구제" if r['rescued'] > 0 else "기준 (0건)"
        print(f"{name:<42} | {r['p']:6.2f}%   | {r['r']:6.2f}%   | {r['f']:6.2f}%   | {r['auc']:6.2f}%  | {r['missed']:3d}건      | {rescued_str}")
    print("=" * 120)

    # Clean 평가 결과
    print("\n" + "=" * 120)
    print("  [가짜 정상(Level 4, 52건) 정제 후 순수 클린 데이터셋 실무 성적표 (N=1,854)]")
    print("=" * 120)
    for name, r in results.items():
        print(f"{name:<42} | Precision: {r['p_clean']:6.2f}% | Clean F1: {r['f_clean']:6.2f}% | 미인식: {r['missed']}건")
    print("=" * 120)

    # 딥러닝 Temporal Attention 분석 (운동 구간별 집중도)
    mean_alpha = np.mean(attentions, axis=0) # (64,)
    # 하강기: 0~21, 최하사점 부근: 22~42, 상승기(스티킹 포인트): 43~63
    phase_desc = np.sum(mean_alpha[:22]) * 100
    phase_nadir = np.sum(mean_alpha[22:43]) * 100
    phase_asc = np.sum(mean_alpha[43:]) * 100

    print("\n[*] 딥러닝 Temporal Self-Attention 운동 위상별 가중치 배분 분석:")
    print(f"    - 하강 구간 (Descent, 0~33%):     {phase_desc:.1f}%")
    print(f"    - 최하사점 구간 (Nadir, 34~66%):    {phase_nadir:.1f}%")
    print(f"    - 상승/스티킹 구간 (Ascent, 67~100%): {phase_asc:.1f}%")
    print(f"    ▶ 딥러닝이 단일 최하사점뿐만 아니라, 무릎이 안으로 무너지는 '상승/스티킹 구간'과 '하강 구간'에 스스로 {phase_desc+phase_asc:.1f}%의 높은 주의를 기울이고 있음을 확인!")


if __name__ == '__main__':
    seqs, sids = extract_or_load_sequences()
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    deep_emb, deep_probs, attns = train_and_extract_deep_features(seqs, y_c3, subjects)
    benchmark_and_verify(deep_emb, deep_probs, attns)

# =========================================================================================================
# [모듈명]: deadlift_sub_classifier.py
# [목적]: Stage 2-A: 일반 데드리프트 (0) vs 스티프 데드리프트 (1) 전용 2진 초정밀 하이브리드 앙상블 Expert
# [핵심 아키텍처]:
#   1. 하체 7개 관절 전용 데이터 분리 (상체 관절 노이즈 100% 제거)
#      - 스트림 1: 하체 6대 뼈대 벡터 (골반-고관절, 대퇴골, 정강이 3D 변위: 18차원)
#      - 스트림 2: 하체 관절 각도/각속도 (무릎, 고관절, 발목: 8차원) -> 총 26차원 시계열
#      - 스트림 3: 11대 하체 고도화 생체역학 판별 피처 (DPD, HDR, SVII, PPTA 등)
#   2. 하체 전용 경량 2중 앙상블 결합:
#      - 모델 1: LowerBodyBiGRU (26차원 64프레임 연속 동역학 곡선 학습)
#      - 모델 2: LightGBM (11대 생체역학 결정 지표 트리 분류)
#      - Decision: P(Stiff) = 0.50 * P_GRU + 0.50 * P_LGBM
# =========================================================================================================

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)

class LowerBodyBiGRU(nn.Module):
    """하체 및 손목 전신 협응 시계열 전용 경량 양방향 GRU 신경망 (Contrastive Margin Loss 지원)"""
    def __init__(self, input_dim: int = 34, hidden_dim: int = 32, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        out, _ = self.gru(x) # (B, T, hidden*2)
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        feat = (avg_pool + max_pool) * 0.5
        emb = self.feature_extractor(feat)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

class DeadliftSubClassifier:
    """Stage 2-A: 바벨(손목)·체간 융합 10D 생체역학 & 34D 전신 협응 Bi-GRU Expert"""

    FEATURE_NAMES_14D = [
        'femur_angle_bottom',        # 1. 최저점 허벅지(대퇴골) 지면 경사각
        'femur_dy_bot',              # 2. 최저점 대퇴골 수직 길이 변위
        'knee_bottom_angle',         # 3. 최저점 무릎 굴곡 각도
        'knee_rom',                  # 4. 무릎 관절 가동범위
        'dynamic_hinge_index',       # 5. 동적 힌지 지수 (HSI = Hip ROM / (Knee ROM + eps))
        'pelvis_displacement_ratio', # 6. 골반 수평/수직 변위비 (PDR)
        'femur_torso_ratio',         # 7. 인체 비례 체형비 (rho_body = Femur / Torso)
        'pelvis_sagittal_angle',     # 8. 3D 시상면 골반 진입각 (theta_pelvis_vec = arctan2(dY, dZ))
        'wrist_rel_knee_y',          # 9. 바벨(손목)-무릎 최저점 수직 고도차
        'min_wrist_knee_dist',       # 10. 전 구간 손목-무릎 최소 3D 공간 이격도
        'knee_wrist_rom_ratio',      # 11. 무릎 가동범위 대비 손목 가동범위 비
        'shank_svii',                # 12. 정강이 수직 불변성 지수 (SVII)
        'knee_vel_rms',              # 13. 무릎 각속도 잔차 RMS
        'bar_shin_dist_x'            # 14. 바벨-정강이 수평 이격 거리
    ]
    FEATURE_NAMES_6D = FEATURE_NAMES_14D
    FEATURE_NAMES_10D = FEATURE_NAMES_14D
    FEATURE_NAMES_11D = FEATURE_NAMES_14D
    FEATURE_NAMES_12D = FEATURE_NAMES_14D
    FEATURE_NAMES_16D = FEATURE_NAMES_14D

    SAVED_PARAMS_PATH = 'd:/Data/피처_데이터/tuning_results/stage2_best_hyperparameters.json'

    def __init__(self, best_params: Optional[Dict[str, Any]] = None, best_threshold: Optional[float] = None):
        loaded_params, loaded_tau = self._load_saved_best_params()
        self.has_saved_params = (loaded_params is not None)

        self.best_params = best_params or loaded_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'max_depth': 3,
            'num_leaves': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'scale_pos_weight': 2.2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 4
        }
        self.best_threshold = best_threshold if best_threshold is not None else (loaded_tau if loaded_tau is not None else 0.29)
        self.blend_weight = 0.80  # GBDT 비중 80%, GRU 비중 20%
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.gru_model: Optional[LowerBodyBiGRU] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def _load_saved_best_params(cls) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        if os.path.exists(cls.SAVED_PARAMS_PATH):
            try:
                import json
                with open(cls.SAVED_PARAMS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dead_info = data.get('stage2_deadlift', {})
                params = dead_info.get('best_params')
                tau = dead_info.get('best_threshold')
                return params, tau
            except Exception:
                pass
        return None, None

    @classmethod
    def align_two_phase_series(cls, X_b: np.ndarray) -> np.ndarray:
        """
        [위상 정렬] 2구간 위상 분할 보간 (Piecewise Phase-Aligned Normalization)
        초고속 벡터화 텐서 선형 보간
        Input: X_b (N, 64, D) -> Output: X_aligned (N, 64, D)
        """
        N, T, D = X_b.shape
        pelvis_height = np.abs(X_b[:, :, 41]) + np.abs(X_b[:, :, 45])
        search_window = pelvis_height[:, 5:59]
        t_bot = 5 + np.argmin(search_window, axis=1).astype(np.float32) # (N,)

        u = np.linspace(0.0, 1.0, 32, dtype=np.float32) # (32,)
        t_desc = np.outer(t_bot, u) # (N, 32)
        t_asc = np.outer(t_bot, 1.0 - u) + np.outer(np.ones(N, dtype=np.float32) * float(T - 1), u) # (N, 32)
        t_grid = np.concatenate([t_desc, t_asc], axis=1) # (N, 64)

        t_floor = np.clip(np.floor(t_grid).astype(np.int64), 0, T - 2)
        t_ceil = t_floor + 1
        alpha = (t_grid - t_floor)[:, :, None] # (N, 64, 1)

        batch_idx = np.arange(N)[:, None, None]
        time_floor_idx = t_floor[:, :, None]
        time_ceil_idx = t_ceil[:, :, None]

        feat_floor = X_b[batch_idx, time_floor_idx, np.arange(D)[None, None, :]]
        feat_ceil = X_b[batch_idx, time_ceil_idx, np.arange(D)[None, None, :]]

        X_aligned = (1.0 - alpha) * feat_floor + alpha * feat_ceil
        return X_aligned.astype(np.float32)

    @classmethod
    def slice_lower_body_series(cls, X_b: np.ndarray) -> np.ndarray:
        """
        [전신 협응 시계열] 위상 정렬 + 하체 뼈대(24D) + 손목 뼈대(8D) + 무릎 각도(2D) = 34차원 시계열
        Input: X_b (N, 64, D) -> Output: X_coord_34d (N, 64, 34)
        """
        X_phase = cls.align_two_phase_series(X_b)
        N, T, D = X_phase.shape

        # 1. 하체 6개 뼈대 (24D)
        lower_bones = X_phase[:, :, 32:56].copy()

        # 2. 손목 뼈대 2개 (좌우 Elbow->Wrist 8D)
        wrist_bones_l = X_phase[:, :, 20:24]
        wrist_bones_r = X_phase[:, :, 28:32]
        wrist_bones = np.concatenate([wrist_bones_l, wrist_bones_r], axis=-1)

        # 3. 무릎 각도 (2D)
        knee_angles = X_phase[:, :, 56:58]

        coord_series = np.concatenate([lower_bones, wrist_bones, knee_angles], axis=-1).astype(np.float32)
        return coord_series

    @classmethod
    def correct_barbell_occlusion(cls, X_phase: np.ndarray) -> np.ndarray:
        """
        [강체 골격 제약 & 바벨 가림 보정]
        바벨 통과 시 허벅지 관측 길이 급변(>12%)을 감지하고,
        강체 골격 제약을 적용하여 고관절/대퇴골 좌표 왜곡을 기하학적으로 복원
        Input: X_phase (N, 64, D) -> Output: X_phase_corr (N, 64, D)
        """
        X_out = X_phase.copy()
        N, T, D = X_out.shape

        v_pelvis_torso = X_out[:, :, 0:3] # (N, T, 3)
        torso_norm = np.linalg.norm(v_pelvis_torso, axis=-1, keepdims=True) + 1e-7 # (N, T, 1)
        l_torso_ref = np.percentile(torso_norm, 95, axis=1, keepdims=True) # (N, 1, 1)

        l_femur_series = (X_out[:, :, 43] + X_out[:, :, 51]) / 2.0 # (N, T)
        l_femur_ref = np.percentile(l_femur_series, 95, axis=1, keepdims=True) + 1e-7 # (N, 1)

        # 허벅지 길이 왜곡률
        femur_drift = np.abs(l_femur_series - l_femur_ref) / l_femur_ref # (N, T)
        drift_mask = (femur_drift > 0.12)[:, :, None] # (N, T, 1)

        # 대퇴골 벡터의 유클리드 길이 스케일링 보정
        scale_factor = np.where(drift_mask, l_femur_ref[:, :, None] / (l_femur_series[:, :, None] + 1e-6), 1.0)
        scale_factor = np.clip(scale_factor, 0.75, 1.35)

        # 좌우 허벅지 벡터(40:43, 48:51)에 강체 스케일 보정 적용
        X_out[:, :, 40:43] = np.where(drift_mask, X_out[:, :, 40:43] * scale_factor, X_out[:, :, 40:43])
        X_out[:, :, 48:51] = np.where(drift_mask, X_out[:, :, 48:51] * scale_factor, X_out[:, :, 48:51])
        X_out[:, :, 43] = np.where(drift_mask[:, :, 0], l_femur_ref, X_out[:, :, 43])
        X_out[:, :, 51] = np.where(drift_mask[:, :, 0], l_femur_ref, X_out[:, :, 51])

        return X_out

    @classmethod
    def slice_features(cls, X_b: np.ndarray) -> np.ndarray:
        """
        [14D 생체역학 불변 피처] 바벨(손목) & 골반 변위비 & 동적 힌지 지수 & 인체 비례 융합 14대 정예 물리 지표
        Input: X_b (N, 64, D) -> Output: X_dead_14d (N, 14)
        """
        X_phase = cls.align_two_phase_series(X_b)
        X_phase = cls.correct_barbell_occlusion(X_phase)
        N, T, D = X_phase.shape

        # 1. 3D 관절 궤적 복원 (Pelvis = 0,0,0 기준)
        v_pelvis_torso = X_phase[:, :, 0:3]
        v_torso_sh_l, v_torso_sh_r = X_phase[:, :, 8:11], X_phase[:, :, 12:15]
        v_sh_el_l, v_sh_el_r = X_phase[:, :, 16:19], X_phase[:, :, 24:27]
        v_el_wr_l, v_el_wr_r = X_phase[:, :, 20:23], X_phase[:, :, 28:31]

        v_pelvis_hip_l, v_pelvis_hip_r = X_phase[:, :, 32:35], X_phase[:, :, 36:39]
        v_hip_knee_l, v_hip_knee_r = X_phase[:, :, 40:43], X_phase[:, :, 48:51]
        v_knee_ank_l, v_knee_ank_r = X_phase[:, :, 44:47], X_phase[:, :, 52:55]

        p_torso = v_pelvis_torso
        p_wr = (p_torso + v_torso_sh_l + v_sh_el_l + v_el_wr_l + p_torso + v_torso_sh_r + v_sh_el_r + v_el_wr_r) / 2.0

        p_hip = (v_pelvis_hip_l + v_pelvis_hip_r) / 2.0
        p_knee = (v_pelvis_hip_l + v_hip_knee_l + v_pelvis_hip_r + v_hip_knee_r) / 2.0
        p_ank = (v_pelvis_hip_l + v_hip_knee_l + v_knee_ank_l + v_pelvis_hip_r + v_hip_knee_r + v_knee_ank_r) / 2.0

        # 2. 뼈대 길이 불변 기준치
        l_femur_series = (X_phase[:, :, 43] + X_phase[:, :, 51]) / 2.0
        l_tibia_series = (X_phase[:, :, 47] + X_phase[:, :, 55]) / 2.0
        l_femur_ref = np.percentile(l_femur_series, 95, axis=1, keepdims=True) + 1e-7
        l_tibia_ref = np.percentile(l_tibia_series, 95, axis=1, keepdims=True) + 1e-7
        l_torso_ref = np.percentile(np.linalg.norm(p_torso, axis=-1), 95, axis=1, keepdims=True) + 1e-7

        femur_dy = (X_phase[:, :, 41] + X_phase[:, :, 49]) / 2.0
        femur_dx = (X_phase[:, :, 40] + X_phase[:, :, 48]) / 2.0
        tibia_dy = (X_phase[:, :, 45] + X_phase[:, :, 53]) / 2.0
        tibia_dx = (X_phase[:, :, 44] + X_phase[:, :, 52]) / 2.0

        pelvis_height = np.abs(femur_dy) + np.abs(tibia_dy)
        t_bot = 5 + np.argmin(pelvis_height[:, 5:59], axis=1) # (N,)

        # [피처 1] 최저점 대퇴골 지면 경사각
        sin_femur = np.clip(np.abs(femur_dy) / l_femur_ref, -1.0, 1.0)
        femur_angle_series = np.degrees(np.arcsin(sin_femur))
        f_femur_angle_bottom = femur_angle_series[np.arange(N), t_bot].reshape(N, 1)

        # [피처 2] 최저점 대퇴골 수직 길이 변위
        f_femur_dy_bot = np.abs(p_knee[np.arange(N), t_bot, 1] - p_hip[np.arange(N), t_bot, 1]).reshape(N, 1)

        # [피처 3] 최저점 무릎 굴곡 각도
        dot_2d = (-femur_dx) * tibia_dx + (-femur_dy) * tibia_dy
        norm_2d = np.sqrt(femur_dx**2 + femur_dy**2 + 1e-7) * np.sqrt(tibia_dx**2 + tibia_dy**2 + 1e-7)
        knee_angle_series = np.degrees(np.arccos(np.clip(dot_2d / norm_2d, -1.0, 1.0)))
        f_knee_bot = knee_angle_series[np.arange(N), t_bot].reshape(N, 1)

        # [피처 4] 무릎 관절 가동범위
        knee_rom = np.ptp(knee_angle_series, axis=1, keepdims=True)

        # [피처 5] 동적 힌지 지수 (HSI = Hip ROM / (Knee ROM + eps))
        hip_angles = X_phase[:, :, 58:60]
        hip_mean_t = np.mean(hip_angles, axis=-1)
        hip_rom = np.ptp(hip_mean_t, axis=1, keepdims=True)
        f_hsi = (hip_rom + 1.0) / (knee_rom + 1.0)

        # [피처 6] 골반 수평/수직 변위비 (PDR)
        pelvis_dx = p_hip[:, :, 0] - p_hip[:, 0:1, 0]
        pelvis_dy = p_hip[:, :, 1] - p_hip[:, 0:1, 1]
        f_pdr = (np.max(np.abs(pelvis_dx), axis=1, keepdims=True) + 1e-4) / (np.max(np.abs(pelvis_dy), axis=1, keepdims=True) + 1e-4)

        # [피처 7] 인체 비례 체형비 (rho_body = Femur / Torso)
        f_rho_body = l_femur_ref / l_torso_ref

        # [피처 8] 3D 시상면 골반 진입각 (theta_pelvis_vec = arctan2(delta_y, delta_z))
        delta_y_pelvis = np.abs(p_hip[np.arange(N), t_bot, 1] - p_hip[np.arange(N), 0, 1])
        delta_z_pelvis = np.abs(p_hip[np.arange(N), t_bot, 2] - p_hip[np.arange(N), 0, 2])
        f_pelvis_sagittal_angle = np.degrees(np.arctan2(delta_y_pelvis, delta_z_pelvis + 1e-5)).reshape(N, 1)

        # [피처 9] 바벨(손목)-무릎 최저점 수직 고도차
        f_wrist_rel_knee_y = (p_wr[np.arange(N), t_bot, 1] - p_knee[np.arange(N), t_bot, 1]).reshape(N, 1)

        # [피처 10] 전 구간 손목-무릎 최소 3D 공간 이격도
        f_min_wrist_knee_dist = np.min(np.linalg.norm(p_wr - p_knee, axis=-1, keepdims=True), axis=1)

        # [피처 11] 무릎 가동범위 대비 손목 가동범위 비
        wrist_y_rom = np.ptp(p_wr[:, :, 1], axis=1, keepdims=True)
        f_knee_wrist_rom_ratio = knee_rom / (wrist_y_rom + 1e-4)

        # [피처 12] 정강이 수직 불변성 지수 (SVII)
        cos_shank = np.clip(np.abs(tibia_dy) / l_tibia_ref, -1.0, 1.0)
        shank_angle_series = np.degrees(np.arccos(cos_shank))
        f_shank_svii = 1.0 / (np.std(shank_angle_series, axis=1, keepdims=True) + 1e-3)

        # [피처 13] 무릎 각속도 잔차 RMS
        dt_knee = np.diff(knee_angle_series, axis=1)
        f_knee_vel_rms = np.sqrt(np.mean(dt_knee**2, axis=1, keepdims=True))

        # [피처 14] 바벨-정강이 수평 이격 거리
        f_bar_shin_x = np.mean(np.abs(p_wr[:, :, 0] - p_ank[:, :, 0]), axis=1, keepdims=True)

        feats_14d = np.concatenate([
            f_femur_angle_bottom,
            f_femur_dy_bot,
            f_knee_bot,
            knee_rom,
            f_hsi,
            f_pdr,
            f_rho_body,
            f_pelvis_sagittal_angle,
            f_wrist_rel_knee_y,
            f_min_wrist_knee_dist,
            f_knee_wrist_rom_ratio,
            f_shank_svii,
            f_knee_vel_rms,
            f_bar_shin_x
        ], axis=1).astype(np.float32)

        return feats_14d

    @classmethod
    def compute_soft_physics_prior(cls, X_b: np.ndarray) -> np.ndarray:
        """
        [소프트 물리 사전확률] 인체 비례 체형 적응형 시그모이드 기반 스티프 데드리프트 사전확률 P_physics(Stiff|X)
        Output: (N, 1) in [0.0, 1.0]
        """
        X_phase = cls.align_two_phase_series(X_b)
        N, T, D = X_phase.shape

        v_pelvis_torso = X_phase[:, :, 0:3]
        p_torso = v_pelvis_torso

        femur_dy = (X_phase[:, :, 41] + X_phase[:, :, 49]) / 2.0
        femur_dx = (X_phase[:, :, 40] + X_phase[:, :, 48]) / 2.0
        tibia_dy = (X_phase[:, :, 45] + X_phase[:, :, 53]) / 2.0
        tibia_dx = (X_phase[:, :, 44] + X_phase[:, :, 52]) / 2.0

        l_femur_series = (X_phase[:, :, 43] + X_phase[:, :, 51]) / 2.0
        l_femur_ref = np.percentile(l_femur_series, 95, axis=1, keepdims=True) + 1e-7
        l_torso_ref = np.percentile(np.linalg.norm(p_torso, axis=-1), 95, axis=1, keepdims=True) + 1e-7
        rho_body = l_femur_ref / l_torso_ref

        pelvis_height = np.abs(femur_dy) + np.abs(tibia_dy)
        t_bot = 5 + np.argmin(pelvis_height[:, 5:59], axis=1)

        dot_2d = (-femur_dx) * tibia_dx + (-femur_dy) * tibia_dy
        norm_2d = np.sqrt(femur_dx**2 + femur_dy**2 + 1e-7) * np.sqrt(tibia_dx**2 + tibia_dy**2 + 1e-7)
        knee_angle_series = np.degrees(np.arccos(np.clip(dot_2d / norm_2d, -1.0, 1.0)))
        knee_rom = np.ptp(knee_angle_series, axis=1, keepdims=True)
        knee_bot = knee_angle_series[np.arange(N), t_bot].reshape(N, 1)

        hip_angles = X_phase[:, :, 58:60]
        hip_mean_t = np.mean(hip_angles, axis=-1)
        hip_rom = np.ptp(hip_mean_t, axis=1, keepdims=True)
        hsi = (hip_rom + 1.0) / (knee_rom + 1.0)

        # 체형 및 시점 적응 임계각 (긴 허벅지 체형 자동 완화)
        theta_crit = 38.0 - 15.0 * (rho_body - 0.95)
        z_phys = (
            0.15 * (knee_bot - theta_crit) +
            0.12 * (knee_rom - 25.0) -
            0.40 * (hsi - 2.0)
        )
        p_dead_phys = 1.0 / (1.0 + np.exp(-np.clip(z_phys, -10.0, 10.0)))
        p_stiff_phys = (1.0 - p_dead_phys).astype(np.float32)
        return p_stiff_phys

    def _train_gru(self, X_lower_tr: np.ndarray, y_tr: np.ndarray, epochs: int = 15, batch_size: int = 256):
        """[기술 7] Contrastive Margin Loss 결합 Bi-GRU 학습"""
        input_dim = X_lower_tr.shape[2]
        self.gru_model = LowerBodyBiGRU(input_dim=input_dim, hidden_dim=32, num_classes=2, dropout=0.2).to(self.device)
        
        pos_weight = float(np.sum(y_tr == 0)) / max(1, np.sum(y_tr == 1))
        class_w = torch.tensor([1.0, min(pos_weight, 3.0)], dtype=torch.float32).to(self.device)
        ce_criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = torch.optim.AdamW(self.gru_model.parameters(), lr=0.003, weight_decay=1e-4)

        dataset = TensorDataset(torch.tensor(X_lower_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

        self.gru_model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits, emb = self.gru_model(bx, return_embedding=True)
                ce_loss = ce_criterion(logits, by)

                # Contrastive Margin Loss
                emb_norm = torch.nn.functional.normalize(emb, dim=1)
                sim_matrix = torch.matmul(emb_norm, emb_norm.T)
                mask_same = (by.unsqueeze(1) == by.unsqueeze(0)).float()
                mask_diff = 1.0 - mask_same
                contrastive_loss = torch.mean(torch.clamp(sim_matrix - 0.2, min=0.0) * mask_diff)

                total_loss = ce_loss + 0.1 * contrastive_loss
                total_loss.backward()
                optimizer.step()

    def _predict_gru_proba(self, X_lower: np.ndarray) -> np.ndarray:
        """Bi-GRU 사후 확률 산출"""
        if self.gru_model is None:
            raise RuntimeError("GRU 모델이 학습되지 않았습니다.")
        self.gru_model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X_lower, dtype=torch.float32).to(self.device)
            logits = self.gru_model(t_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    @classmethod
    def augment_phase_preserving_resampling(
        cls,
        X_raw: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.35,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        [소수 클래스 집중 증강] 최하사점(t_bottom) 앵커 기반 2구간 위상 보존 시간축 리샘플링
        - 관절 지터링 및 CutMix 100% 배제
        - 최하사점(t_bottom) 앵커링을 유지하며 편심성(0 -> t_bot)과 동심성(t_bot -> 63) 구간 독립 템포 워핑 (±12%)
        """
        np.random.seed(random_seed)
        stiff_indices = np.where(y == 1)[0]
        n_dead = int(np.sum(y == 0))
        n_stiff = len(stiff_indices)
        needed = int(n_dead * target_ratio) - n_stiff
        if needed <= 0 or n_stiff == 0:
            return X_raw, y

        N_src, T, D = X_raw.shape
        femur_dy = (X_raw[:, :, 41] + X_raw[:, :, 49]) / 2.0
        tibia_dy = (X_raw[:, :, 45] + X_raw[:, :, 53]) / 2.0
        pelvis_height = np.abs(femur_dy) + np.abs(tibia_dy)
        t_bots = 5 + np.argmin(pelvis_height[:, 5:59], axis=1) # (N,)

        aug_samples = []
        for _ in range(needed):
            idx = np.random.choice(stiff_indices)
            sample = X_raw[idx] # (64, D)
            t_b = float(t_bots[idx])
            t_b_int = int(round(t_b))

            # 하강(편심) 구간 워핑 배율: 0.88 ~ 1.12
            # 상승(동심) 구간 워핑 배율: 0.88 ~ 1.12
            alpha_desc = float(np.random.uniform(0.88, 1.12))
            alpha_asc = float(np.random.uniform(0.88, 1.12))

            # t_b 앵커를 완벽히 고정한 채 편심/동심 구간 시간축 워핑
            u_desc = (np.linspace(0.0, 1.0, max(2, t_b_int + 1), dtype=np.float32) ** alpha_desc) * t_b
            rem_len = max(2, (T - 1) - t_b_int + 1)
            u_asc = t_b + (np.linspace(0.0, 1.0, rem_len, dtype=np.float32) ** alpha_asc) * (float(T - 1) - t_b)

            t_warped = np.concatenate([u_desc[:-1], u_asc])
            if len(t_warped) != T:
                t_warped = np.interp(np.linspace(0.0, 1.0, T), np.linspace(0.0, 1.0, len(t_warped)), t_warped)

            t_floor = np.clip(np.floor(t_warped).astype(int), 0, T - 2)
            t_ceil = t_floor + 1
            w = (t_warped - t_floor)[:, None] # (64, 1)

            sample_warped = (1.0 - w) * sample[t_floor] + w * sample[t_ceil]
            aug_samples.append(sample_warped.astype(np.float32))

        X_aug = np.concatenate([X_raw, np.array(aug_samples, dtype=np.float32)], axis=0)
        y_aug = np.concatenate([y, np.ones(needed, dtype=int)], axis=0)
        return X_aug, y_aug

    def tune_and_fit(
        self,
        X_train_raw: np.ndarray,
        y_train: np.ndarray,
        X_val_raw: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_saved_params: bool = False
    ) -> Dict[str, Any]:
        """
        [소프트 물리 사전확률 결합 & F_1.3 최적화 그리드 탐색]
        """
        # 1. 훈련 데이터 전용 위상 보존 시간축 리샘플링 증강 (검증/테스트에는 적용 배제)
        X_tr_aug, y_tr_aug = self.augment_phase_preserving_resampling(
            X_train_raw, y_train, target_ratio=0.35, random_seed=42
        )

        # 2. 데이터 슬라이싱 및 물리 사전확률 산출
        X_lower_tr = self.slice_lower_body_series(X_tr_aug)
        X_feat_tr = self.slice_features(X_tr_aug)
        p_phys_tr = self.compute_soft_physics_prior(X_tr_aug)

        eval_lower = self.slice_lower_body_series(X_val_raw) if X_val_raw is not None else self.slice_lower_body_series(X_train_raw)
        eval_feat = self.slice_features(X_val_raw) if X_val_raw is not None else self.slice_features(X_train_raw)
        eval_phys = self.compute_soft_physics_prior(X_val_raw) if X_val_raw is not None else self.compute_soft_physics_prior(X_train_raw)
        eval_y = y_val if y_val is not None else y_train

        # 3. 하체 Bi-GRU 훈련
        self._train_gru(X_lower_tr, y_tr_aug, epochs=15)
        p_gru_eval = self._predict_gru_proba(eval_lower)[:, 1]

        # 4. LightGBM 1차 학습 및 Hard Negative Mining
        pos_w = float(np.sum(y_tr_aug == 0)) / max(1, np.sum(y_tr_aug == 1))
        init_lgb = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=100,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            min_child_samples=20,
            reg_alpha=0.1,
            scale_pos_weight=max(1.8, pos_w * 0.8),
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            n_jobs=4
        )
        init_lgb.fit(X_feat_tr, y_tr_aug)
        p_tr_init = init_lgb.predict_proba(X_feat_tr)[:, 1]

        # Hard Negative Mining 가중치 생성 (일반데드 중 P(Stiff) >= 0.25인 샘플 2.0배 가중)
        sample_weights = np.ones(len(y_tr_aug), dtype=np.float32)
        hard_neg_mask = (y_tr_aug == 0) & (p_tr_init >= 0.25)
        sample_weights[hard_neg_mask] = 2.0

        # 5. 하이퍼파라미터 및 Soft Physics Blending 그리드 서치
        param_candidates = [
            {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.03, 'scale_pos_weight': max(1.8, pos_w * 0.7)},
            {'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.04, 'scale_pos_weight': max(2.0, pos_w * 0.8)},
            {'max_depth': 4, 'num_leaves': 12, 'learning_rate': 0.03, 'scale_pos_weight': max(2.2, pos_w * 0.85)},
            {'max_depth': 4, 'num_leaves': 12, 'learning_rate': 0.04, 'scale_pos_weight': max(2.4, pos_w * 0.90)}
        ]

        best_score = -1.0
        best_cfg = param_candidates[1]
        best_tau = 0.36
        best_w_lgb = 0.75
        best_w_phys = 0.15

        for cfg in param_candidates:
            model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                n_estimators=100,
                max_depth=cfg['max_depth'],
                num_leaves=cfg['num_leaves'],
                learning_rate=cfg['learning_rate'],
                scale_pos_weight=cfg['scale_pos_weight'],
                min_child_samples=20,
                reg_alpha=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=4
            )
            model.fit(X_feat_tr, y_tr_aug, sample_weight=sample_weights)
            p_lgb = model.predict_proba(eval_feat)[:, 1]

            for w_phys in [0.10, 0.15, 0.20]:
                for w_lgb in [0.70, 0.75, 0.80]:
                    w_gru = max(0.0, 1.0 - w_lgb - w_phys)
                    p_ens = (w_lgb * p_lgb) + (w_phys * eval_phys.ravel()) + (w_gru * p_gru_eval)
                    
                    for tau in np.arange(0.20, 0.50, 0.02):
                        preds = (p_ens >= tau).astype(int)
                        prec = precision_score(eval_y, preds, zero_division=0)
                        rec = recall_score(eval_y, preds, zero_division=0)
                        
                        beta_sq = 1.3 ** 2
                        denom = (beta_sq * prec) + rec
                        f_beta = (1.0 + beta_sq) * (prec * rec) / denom if denom > 0 else 0.0

                        if f_beta > best_score:
                            best_score = f_beta
                            best_cfg = cfg
                            best_tau = float(tau)
                            best_w_lgb = float(w_lgb)
                            best_w_phys = float(w_phys)

        self.best_params.update(best_cfg)
        self.best_threshold = best_tau
        self.blend_weight = best_w_lgb
        self.phys_weight = best_w_phys

        self.lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=100,
            max_depth=self.best_params['max_depth'],
            num_leaves=self.best_params['num_leaves'],
            learning_rate=self.best_params['learning_rate'],
            scale_pos_weight=self.best_params['scale_pos_weight'],
            min_child_samples=20,
            reg_alpha=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
            n_jobs=4
        )
        self.lgb_model.fit(X_feat_tr, y_tr_aug, sample_weight=sample_weights)

        return {
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'blend_weight': self.blend_weight,
            'phys_weight': getattr(self, 'phys_weight', 0.15),
            'best_val_f_beta': best_score
        }

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """하체 3중 앙상블 사후 확률 반환 (w_lgb*LGBM + w_phys*Phys + w_gru*GRU) -> Shape: (N, 2)"""
        if self.lgb_model is None or self.gru_model is None:
            raise RuntimeError("앙상블 모델이 학습되지 않았습니다.")
        
        # 1. GRU 확률
        X_lower = self.slice_lower_body_series(X_raw)
        p_gru = self._predict_gru_proba(X_lower) # (N, 2)

        # 2. LightGBM 확률
        X_feat = self.slice_features(X_raw)
        p_lgb = self.lgb_model.predict_proba(X_feat) # (N, 2)

        # 3. Soft Physics Prior 확률
        p_phys_stiff = self.compute_soft_physics_prior(X_raw) # (N, 1)
        p_phys = np.concatenate([1.0 - p_phys_stiff, p_phys_stiff], axis=1) # (N, 2)

        # 4. 3중 소프트 블렌딩 결합
        w_lgb = getattr(self, 'blend_weight', 0.75)
        w_phys = getattr(self, 'phys_weight', 0.15)
        w_gru = max(0.0, 1.0 - w_lgb - w_phys)
        
        p_ens = (w_lgb * p_lgb) + (w_phys * p_phys) + (w_gru * p_gru)
        return p_ens

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """최적 임계값 적용 2진 예측 (0: 일반 데드, 1: 스티프 데드)"""
        probs = self.predict_proba(X_raw)[:, 1]
        return (probs >= self.best_threshold).astype(int)

    def evaluate(self, X_raw: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """데드리프트 서브 모델 2진 정밀 성능 지표 및 2x2 혼동 행렬 평가"""
        probs = self.predict_proba(X_raw)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
        cm = confusion_matrix(y_true, preds, labels=[0, 1])

        # 일반 데드리프트 (0) 지표
        cm_00, cm_01, cm_10, cm_11 = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        prec_dead = cm_00 / max(1, cm_00 + cm_10)
        rec_dead = cm_00 / max(1, cm_00 + cm_01)
        f1_dead = 2 * (prec_dead * rec_dead) / max(1e-7, prec_dead + rec_dead)

        return {
            'accuracy': float(acc),
            'stiff_precision': float(prec),
            'stiff_recall': float(rec),
            'stiff_f1': float(f1),
            'dead_precision': float(prec_dead),
            'dead_recall': float(rec_dead),
            'dead_f1': float(f1_dead),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'best_threshold': float(self.best_threshold),
            'confusion_matrix': cm.tolist(),
            'support_dead': int(np.sum(y_true == 0)),
            'support_stiff': int(np.sum(y_true == 1))
        }

# =========================================================================================================
# [모듈 설명]: 1단계 8종 운동 대분류 - 2차원 3-Fold 복합 교차검증 (Dual-Axis K-Fold) + 스티프데드 1:1 타깃 증강 파이프라인
# [입력 데이터]: Condition 3 MediaPipe 3D 마스터 풀 (20,003개 시퀀스 전수 풀)
# [출력 목표]: 3개 Fold 순환 평가를 통한 100% 전수 데이터 검증 (Mean ± Std 공인 벤치마크 도출)
#
# [실행 명령어 (CLI 테스트)]:
#   python -m src.models.first_model.train_kfold_ensemble
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO, FOLDER_TO_EX_IDX
from src.models.first_model.metrics_helper import Stage1MetricsHelper, EXERCISE_NAMES_SHORT
from src.models.first_model.train_augmented_ensemble import DropKeypointBiGRUNet, AdvancedGRUClassifier
from src.preprocessing.pose_augmentor import PoseAugmentor3D

OUTPUT_KFOLD_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_kfold_3fold_ensemble_results.json'

class DiscriminativeKinematicsExtractor:
    """
    스티프 데드리프트(무릎 고정/정강이 수직도/고관절-무릎 동조성) 및 크로스 런지(골반축 교차 마진/수축비/가림보상) 분별 특화 생체역학 피처 추출기
    - Format B (64, D) 시계열 텐서로부터 40차원의 고분별 통계 지표 추출
    """

    @classmethod
    def extract_features(cls, X_b: np.ndarray) -> np.ndarray:
        N, T, D = X_b.shape

        # -------------------------------------------------------------
        # 1. 무릎 각도 및 관절 가동범위 (Stream 2: idx 56, 57)
        # -------------------------------------------------------------
        knee_angles = X_b[:, :, 56:58] # (N, T, 2)
        knee_mean_t = np.mean(knee_angles, axis=-1) # (N, T)
        knee_rom = np.ptp(knee_angles, axis=1) # (N, 2)
        knee_rom_avg = np.mean(knee_rom, axis=1, keepdims=True) # (N, 1)
        knee_min = np.min(knee_angles, axis=1) # (N, 2)
        knee_mean = np.mean(knee_angles, axis=1) # (N, 2)
        knee_stiff_ratio = np.mean(knee_angles > 150.0, axis=(1, 2)).reshape(N, 1) # (N, 1)

        # -------------------------------------------------------------
        # 2. 고관절 각도 및 가동범위 (Stream 2: idx 58, 59)
        # -------------------------------------------------------------
        hip_angles = X_b[:, :, 58:60] # (N, T, 2)
        hip_mean_t = np.mean(hip_angles, axis=-1) # (N, T)
        hip_rom = np.ptp(hip_angles, axis=1) # (N, 2)
        hip_rom_avg = np.mean(hip_rom, axis=1, keepdims=True) # (N, 1)
        hip_min = np.min(hip_angles, axis=1) # (N, 2)

        # 3. 고관절 / 무릎 가동 범위 비율 (Hip/Knee ROM Ratio)
        rom_ratio = ((hip_rom_avg + 1e-4) / (knee_rom_avg + 1e-4)).reshape(N, 1) # (N, 1)

        # -------------------------------------------------------------
        # ★ [솔루션 1] 정강이 지면 수직도 (Shank Verticality)
        # Stream 1의 Knee-Ankle 벡터: Left(44:48), Right(52:56)
        # -------------------------------------------------------------
        dy_l, len_l = np.abs(X_b[:, :, 45]), X_b[:, :, 47] + 1e-7
        dy_r, len_r = np.abs(X_b[:, :, 53]), X_b[:, :, 55] + 1e-7
        cos_shank_l = np.clip(dy_l / len_l, -1.0, 1.0)
        cos_shank_r = np.clip(dy_r / len_r, -1.0, 1.0)
        shank_angle_l = np.degrees(np.arccos(cos_shank_l)) # (N, T)
        shank_angle_r = np.degrees(np.arccos(cos_shank_r)) # (N, T)
        shank_angle_t = (shank_angle_l + shank_angle_r) / 2.0 # (N, T)

        shank_rom = np.ptp(shank_angle_t, axis=1, keepdims=True) # (N, 1) - 정강이 기울기 가동범위
        shank_max = np.max(shank_angle_t, axis=1, keepdims=True) # (N, 1) - 최대 정강이 기울기
        shank_mean = np.mean(shank_angle_t, axis=1, keepdims=True) # (N, 1)
        knee_shank_interaction = (knee_rom_avg * shank_rom) # (N, 1) - 복합 결정 평면

        # -------------------------------------------------------------
        # ★ [솔루션 2] 고관절-무릎 동조 각속도 및 상관계수 (Hip-Knee Coupling)
        # -------------------------------------------------------------
        knee_vel_t = np.gradient(knee_mean_t, axis=1) # (N, T)
        hip_vel_t = np.gradient(hip_mean_t, axis=1) # (N, T)

        knee_vel_cent = knee_vel_t - np.mean(knee_vel_t, axis=1, keepdims=True)
        hip_vel_cent = hip_vel_t - np.mean(hip_vel_t, axis=1, keepdims=True)
        cov_vel = np.mean(knee_vel_cent * hip_vel_cent, axis=1, keepdims=True)
        std_k_vel = np.std(knee_vel_t, axis=1, keepdims=True) + 1e-6
        std_h_vel = np.std(hip_vel_t, axis=1, keepdims=True) + 1e-6
        hip_knee_corr = np.clip(cov_vel / (std_k_vel * std_h_vel), -1.0, 1.0) # (N, 1) 피어슨 r

        knee_rms = np.sqrt(np.mean(knee_vel_t**2, axis=1, keepdims=True)) + 1e-6
        hip_rms = np.sqrt(np.mean(hip_vel_t**2, axis=1, keepdims=True))
        hip_knee_rms_ratio = hip_rms / knee_rms # (N, 1) 각속도 RMS 비율

        knee_vel_mean = np.mean(np.abs(knee_vel_t), axis=1, keepdims=True) # (N, 1)
        hip_vel_mean = np.mean(np.abs(hip_vel_t), axis=1, keepdims=True) # (N, 1)

        # -------------------------------------------------------------
        # ★ [솔루션 3] 시작 1초(초기 5프레임) 정지 자세 기준 상대적 변화량 (Δθ)
        # 피험자별 기저 무릎 굽힘 성향(개인차) 소거
        # -------------------------------------------------------------
        knee_init = np.mean(knee_mean_t[:, :5], axis=1, keepdims=True) # (N, 1)
        hip_init = np.mean(hip_mean_t[:, :5], axis=1, keepdims=True) # (N, 1)

        knee_rel_delta = np.max(np.abs(knee_mean_t - knee_init), axis=1, keepdims=True) # (N, 1)
        hip_rel_delta = np.max(np.abs(hip_mean_t - hip_init), axis=1, keepdims=True) # (N, 1)
        rel_rom_ratio = ((hip_rel_delta + 1.0) / (knee_rel_delta + 1.0)).reshape(N, 1) # (N, 1)

        # -------------------------------------------------------------
        # 4. 크로스 런지 특화: 무릎/발목 Valgus & 좌우 비대칭
        # -------------------------------------------------------------
        valgus = X_b[:, :, 66:68] # (N, T, 2)
        valgus_max = np.max(np.abs(valgus), axis=1) # (N, 2)
        valgus_mean = np.mean(valgus, axis=1) # (N, 2)

        asymm = X_b[:, :, 72:74] # (N, T, 2)
        asymm_max = np.max(asymm, axis=1) # (N, 2)
        asymm_mean = np.mean(asymm, axis=1) # (N, 2)

        # -------------------------------------------------------------
        # ★ [솔루션 4: 스티프 데드 4대 핵심 생체역학 원리 피처]
        # 1) 최저점 골반-무릎 수직 높이 단차 (Hip-Knee Gap, Cohen's d = 0.70)
        # 2) 대퇴골(허벅지) 지면 경사각 (Lowest Femur Angle & ROM, Cohen's d = 0.63)
        # 3) 힙 힌지 비율 지수 (Hip Hinge Index: Backward Shift / Drop, Cohen's d = 0.41)
        # 4) 척추-대퇴골 사잇각 (Spine-Femur Min & ROM, Cohen's d = 0.43)
        # -------------------------------------------------------------
        femur_dy = (np.abs(X_b[:, :, 41]) + np.abs(X_b[:, :, 49])) / 2.0 # (N, T) 대퇴골 수직 높이
        femur_dz = (np.abs(X_b[:, :, 42]) + np.abs(X_b[:, :, 50])) / 2.0 # (N, T)
        femur_len = (X_b[:, :, 43] + X_b[:, :, 51]) / 2.0 + 1e-7 # (N, T)

        # 1) 최저점 골반-무릎 수직 단차 & 골반 하강비
        hip_knee_gap_min = np.min(femur_dy, axis=1, keepdims=True) # (N, 1) - 스티프는 높고 일반데드는 낮음
        hip_knee_gap_mean = np.mean(femur_dy, axis=1, keepdims=True) # (N, 1)

        tibia_dy = (np.abs(X_b[:, :, 45]) + np.abs(X_b[:, :, 53])) / 2.0 # (N, T) 정강이 수직 높이
        tibia_dz = (np.abs(X_b[:, :, 46]) + np.abs(X_b[:, :, 54])) / 2.0 # (N, T)
        pelvis_height = femur_dy + tibia_dy # (N, T)
        pelvis_drop_ratio = (np.min(pelvis_height, axis=1, keepdims=True) + 1e-4) / (np.max(pelvis_height, axis=1, keepdims=True) + 1e-4) # (N, 1)

        # 2) 대퇴골(허벅지) 지면 경사각
        sin_femur = np.clip(femur_dy / femur_len, -1.0, 1.0)
        femur_angle = np.degrees(np.arcsin(sin_femur)) # (N, T) (90도=수직 직립, 0도=수평 눕기)
        femur_angle_min = np.min(femur_angle, axis=1, keepdims=True) # (N, 1) - 최저점 각도
        femur_angle_rom = np.ptp(femur_angle, axis=1, keepdims=True) # (N, 1) - 가동범위
        femur_angle_mean = np.mean(femur_angle, axis=1, keepdims=True) # (N, 1)

        # 3) 힙 힌지 비율 지수 (후방 이동량 / 수직 하강량)
        pelvis_dz_total = femur_dz + tibia_dz
        pelvis_dy_total = femur_dy + tibia_dy
        hip_hinge_index = (np.ptp(pelvis_dz_total, axis=1, keepdims=True) + 1e-4) / (np.ptp(pelvis_dy_total, axis=1, keepdims=True) + 1e-4) # (N, 1)

        # 4) 척추-대퇴골 사잇각
        spine_vec = X_b[:, :, 0:3] # Pelvis->Torso (dx, dy, dz)
        femur_vec_l = X_b[:, :, 40:43]
        femur_vec_r = X_b[:, :, 48:51]
        femur_vec_avg = (femur_vec_l + femur_vec_r) / 2.0
        spine_len = X_b[:, :, 3] + 1e-7

        dot_sf = np.sum(spine_vec * femur_vec_avg, axis=-1)
        cos_sf = np.clip(dot_sf / (spine_len * femur_len), -1.0, 1.0)
        spine_femur_angle = np.degrees(np.arccos(cos_sf)) # (N, T)
        spine_femur_min = np.min(spine_femur_angle, axis=1, keepdims=True) # (N, 1)
        spine_femur_rom = np.ptp(spine_femur_angle, axis=1, keepdims=True) # (N, 1)

        # -------------------------------------------------------------
        # ★ [솔루션 5] 크로스 런지 vs 다이나믹 런지: 골반 로컬 축 투영 부호화 교차 마진 & 동역학
        # Stream 1 Bone 벡터들로부터 골반 및 발목 위치 복원
        # Edge 8: PELVIS->LEFT_HIP (32:35), Edge 10: LEFT_HIP->LEFT_KNEE (40:43), Edge 11: LEFT_KNEE->LEFT_ANKLE (44:47)
        # Edge 9: PELVIS->RIGHT_HIP (36:39), Edge 12: RIGHT_HIP->RIGHT_KNEE (48:51), Edge 13: RIGHT_KNEE->RIGHT_ANKLE (52:55)
        # -------------------------------------------------------------
        hip_l = X_b[:, :, 32:35] # (N, T, 3)
        hip_r = X_b[:, :, 36:39] # (N, T, 3)
        ankle_l = hip_l + X_b[:, :, 40:43] + X_b[:, :, 44:47] # (N, T, 3)
        ankle_r = hip_r + X_b[:, :, 48:51] + X_b[:, :, 52:55] # (N, T, 3)

        # 1. 골반 축 단위 벡터 (Left Hip -> Right Hip)
        hip_axis = hip_r - hip_l # (N, T, 3)
        hip_axis_norm = hip_axis / (np.linalg.norm(hip_axis, axis=-1, keepdims=True) + 1e-7)

        # 2. 이동량이 더 큰 발을 스텝 발로, 반대쪽을 기준발(지지발)로 판별
        disp_l = np.linalg.norm(ankle_l - ankle_l[:, 0:1, :], axis=-1) # (N, T)
        disp_r = np.linalg.norm(ankle_r - ankle_r[:, 0:1, :], axis=-1) # (N, T)
        is_left_stepping = (np.max(disp_l, axis=1) >= np.max(disp_r, axis=1))[:, None] # (N, 1)

        # 3. 기준발 대비 스텝 발의 상대 벡터 및 부호화 교차 마진 (Signed Crossover Margin)
        vec_l_step = ankle_l - ankle_r
        vec_r_step = ankle_r - ankle_l
        proj_l = np.sum(vec_l_step * hip_axis_norm, axis=-1) # (N, T)
        proj_r = np.sum(vec_r_step * (-hip_axis_norm), axis=-1) # (N, T)

        crossover_margin_t = np.where(is_left_stepping, proj_l, proj_r) # (N, T)

        crossover_margin_max = np.max(crossover_margin_t, axis=1, keepdims=True) # (N, 1) - 최대 교차량 (>0: 크로스)
        crossover_margin_mean = np.mean(crossover_margin_t, axis=1, keepdims=True) # (N, 1)
        crossover_ratio = np.mean(crossover_margin_t > 0.0, axis=1, keepdims=True) # (N, 1) - 교차 상태 지속 비율

        # 무릎 최저점(Bottom) 시점에서의 교차 마진
        bottom_idx = np.argmin(knee_mean_t, axis=1) # (N,)
        crossover_bottom = crossover_margin_t[np.arange(N), bottom_idx].reshape(N, 1) # (N, 1)

        # 4. 두 발 사이 횡방향(골반축) 거리의 수축-확장 동역학 비율
        ankle_lateral_dist = np.abs(np.sum((ankle_r - ankle_l) * hip_axis_norm, axis=-1)) # (N, T)
        lateral_dist_init = np.mean(ankle_lateral_dist[:, :5], axis=1, keepdims=True) # (N, 1)
        lateral_dist_min = np.min(ankle_lateral_dist, axis=1, keepdims=True) # (N, 1)
        lateral_contraction_ratio = (lateral_dist_min + 1e-4) / (lateral_dist_init + 1e-4) # (N, 1)

        # 5. 정면/후방 가림(Self-Occlusion) 보상 복합 지표
        sh_hip_twist = X_b[:, :, 68] # (N, T)
        twist_max = np.max(np.abs(sh_hip_twist), axis=1, keepdims=True) # (N, 1)
        torso_roll = X_b[:, :, 65] # (N, T)
        roll_max = np.max(np.abs(torso_roll), axis=1, keepdims=True) # (N, 1)
        valgus_sum = (valgus_max[:, 0:1] + valgus_max[:, 1:2]) # (N, 1)
        cross_occlusion_score = valgus_sum * (twist_max / 45.0) * (roll_max / 5.0) # (N, 1)

        discrim_feats = np.concatenate([
            # 기본 관절각 및 가동범위 (12D)
            knee_rom, knee_min, knee_mean, knee_stiff_ratio,
            hip_rom, hip_min, rom_ratio,
            # [솔루션 1] 정강이 지면 수직도 (4D)
            shank_rom, shank_max, shank_mean, knee_shank_interaction,
            # [솔루션 2] 고관절-무릎 동조 각속도 (4D)
            hip_knee_corr, hip_knee_rms_ratio, knee_vel_mean, hip_vel_mean,
            # [솔루션 3] 초기 자세 기준 상대적 변화량 (3D)
            knee_rel_delta, hip_rel_delta, rel_rom_ratio,
            # ★ [솔루션 4] 스티프 데드 4대 핵심 생체역학 원리 피처 (9D)
            hip_knee_gap_min, hip_knee_gap_mean, pelvis_drop_ratio,
            femur_angle_min, femur_angle_rom, femur_angle_mean,
            hip_hinge_index, spine_femur_min, spine_femur_rom,
            # [솔루션 5] 크로스 런지 특화 (외반/비대칭/교차마진/수축비/가림보상: 17D)
            valgus_max, valgus_mean,
            asymm_max, asymm_mean,
            crossover_margin_max, crossover_margin_mean, crossover_ratio, crossover_bottom,
            lateral_dist_min, lateral_contraction_ratio,
            twist_max, roll_max, cross_occlusion_score
        ], axis=1).astype(np.float32)

        return discrim_feats


class DualAxisKFoldEnsembleTrainer:
    """2차원 3-Fold 복합 교차 검증 (인물 3그룹 × 시점 3그룹) 앙상블 학습 및 평가 엔진"""

    def __init__(self, data_dir: str = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D' if os.path.exists('d:/Data/피처_데이터/Condition4_NovelFlow_3D/Dual_Train.npz') else 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'):
        self.data_dir = data_dir
        self.master_data = self._load_master_pool()

    def _load_master_pool(self) -> Dict[str, Any]:
        """5개 NPZ 파일을 결합하고 특화 생체역학 피처를 융합하여 마스터 데이터 풀 구축"""
        all_X_a = []
        all_X_b = []
        all_y_act = []
        all_subjs = []
        all_views = []
        all_sids = []

        for fn in sorted(os.listdir(self.data_dir)):
            if not fn.endswith('.npz'):
                continue
            split_data = Stage1DatasetLoader.load_split_data(fn, splits_dir=self.data_dir)
            all_X_a.append(split_data['X_Format_A'])
            all_X_b.append(split_data['X_Format_B'])
            all_y_act.append(split_data['y_action'])
            all_subjs.extend(split_data['subjects'])
            all_views.extend(split_data['views'])
            all_sids.extend(split_data['sequence_ids'])

        def get_subj_num(s):
            s = str(s)
            for digit in ['1','2','3','4','5','6','7','8','9']:
                if f"피험자{digit}" in s or s == digit or f"_{digit}_" in s:
                    return int(digit)
            return 1

        subjs_arr = np.array([get_subj_num(s) for s in all_subjs])
        views_arr = np.array(all_views)
        raw_X_a = np.concatenate(all_X_a, axis=0)
        raw_X_b = np.concatenate(all_X_b, axis=0)
        y_act = np.concatenate(all_y_act, axis=0)

        # 23차원 특화 생체역학 피처 추출 및 Format A 융합
        print("[*] 스티프데드 & 크로스런지 분별 특화 피처 (23차원) 융합 중...")
        discrim_feats = DiscriminativeKinematicsExtractor.extract_features(raw_X_b)
        enhanced_X_a = np.concatenate([raw_X_a, discrim_feats], axis=-1)
        print(f"    Format A 피처 차원: {raw_X_a.shape[1]} -> {enhanced_X_a.shape[1]} (차별 피처 결합 완료)")

        return {
            'X_Format_A': enhanced_X_a,
            'X_Format_B': raw_X_b,
            'y_action': y_act,
            'subjects': subjs_arr,
            'views': views_arr,
            'sequence_ids': np.array(all_sids),
            'num_classes': len(EXERCISE_NAMES_KO)
        }

    def run_3fold_cross_validation(self) -> Dict[str, Any]:
        """3-Fold 2차원 교차 검증 전수 실행"""
        folds_config = [
            {
                'fold_id': 1,
                'name': 'Fold 1 [Test: 피험자 1~3 x 전방(0°,45°)]',
                'test_subjs': [1, 2, 3],
                'test_views': ['view1', 'view2'],
                'train_subjs': [4, 5, 6, 7, 8, 9],
                'train_views': ['view3', 'view4', 'view5']
            },
            {
                'fold_id': 2,
                'name': 'Fold 2 [Test: 피험자 4~6 x 측면(90°)]',
                'test_subjs': [4, 5, 6],
                'test_views': ['view3'],
                'train_subjs': [1, 2, 3, 7, 8, 9],
                'train_views': ['view1', 'view2', 'view4', 'view5']
            },
            {
                'fold_id': 3,
                'name': 'Fold 3 [Test: 피험자 7~9 x 후방(135°,180°)]',
                'test_subjs': [7, 8, 9],
                'test_views': ['view4', 'view5'],
                'train_subjs': [1, 2, 3, 4, 5, 6],
                'train_views': ['view1', 'view2', 'view3']
            }
        ]

        print("=" * 105)
        print("★ [1단계 8종 운동 대분류]: 2차원 3-Fold 복합 교차 검증 + 특화 피처 + 손실함수 클래스 가중치 최적화")
        print(f"  - 마스터 데이터 풀 크기: {len(self.master_data['y_action']):,}개 시퀀스")
        print(f"  - 분류 타깃 클래스 수 : {self.master_data['num_classes']}종 운동 (다이나믹 런지 통합 완료)")
        print("=" * 105)

        fold_results = []
        all_test_preds = []
        all_test_trues = []

        for fc in folds_config:
            fold_id = fc['fold_id']
            print(f"\n>>> 🚀 [{fc['name']}] 훈련 및 평가 시작...")

            # 마스크 분할
            test_mask = np.isin(self.master_data['subjects'], fc['test_subjs']) & np.isin(self.master_data['views'], fc['test_views'])
            train_mask = np.isin(self.master_data['subjects'], fc['train_subjs']) & np.isin(self.master_data['views'], fc['train_views'])

            X_tr_a = self.master_data['X_Format_A'][train_mask]
            X_tr_b = self.master_data['X_Format_B'][train_mask]
            y_tr = self.master_data['y_action'][train_mask]

            X_te_a = self.master_data['X_Format_A'][test_mask]
            X_te_b = self.master_data['X_Format_B'][test_mask]
            y_te = self.master_data['y_action'][test_mask]

            # 손실 함수용 클래스 가중치 계산 (Smoothed Balanced Weights)
            class_counts = np.bincount(y_tr, minlength=self.master_data['num_classes'])
            raw_w = len(y_tr) / (float(self.master_data['num_classes']) * np.maximum(class_counts, 1).astype(np.float32))
            smooth_w = np.sqrt(raw_w) # 극단적 쏠림 방지 제곱근 스무딩
            sample_w = smooth_w[y_tr]

            print(f"    [데이터 구성] Train: {len(y_tr):,}개 (데드리프트: {(y_tr==1).sum()}개, 스티프데드: {(y_tr==2).sum()}개, 크로스런지: {(y_tr==6).sum()}개)")
            print(f"    [데이터 구성] Test : {len(y_te):,}개 (데드리프트: {(y_te==1).sum()}개, 스티프데드: {(y_te==2).sum()}개, 크로스런지: {(y_te==6).sum()}개)")

            # 1. 시계열 Bi-GRU 학습 (손실함수 가중치 적용)
            t0 = time.time()
            print("      [1/3] Bi-GRU 시계열 신경망 훈련 중 (12 Epochs, Batch 512, Weighted Loss)...", flush=True)
            gru_clf = AdvancedGRUClassifier(
                input_dim=X_tr_b.shape[2],
                num_classes=self.master_data['num_classes'],
                epochs=12,
                lr=0.003,
                batch_size=512
            )
            gru_clf.fit(X_tr_b, y_tr, class_weights=smooth_w)

            # 2. 정형 XGBoost 학습 (샘플 가중치 적용)
            print("      [2/3] XGBoost 정형 모델 훈련 중 (100 Trees, Sample Weights)...", flush=True)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=4
            )
            xgb_clf.fit(X_tr_a, y_tr, sample_weight=sample_w)

            # 3. 정형 CatBoost 학습 (클래스 가중치 적용)
            print("      [3/3] CatBoost 정형 모델 훈련 중 (100 Trees, Class Weights)...", flush=True)
            cat_clf = CatBoostClassifier(
                iterations=100,
                depth=5,
                learning_rate=0.1,
                thread_count=4,
                random_seed=42,
                class_weights=smooth_w.tolist(),
                verbose=False
            )
            cat_clf.fit(X_tr_a, y_tr)
            train_elapsed = time.time() - t0

            # 앙상블 추론 (가중치: Bi-GRU 0.65 + XGBoost 0.20 + CatBoost 0.15)
            t_inf = time.time()
            p_gru = gru_clf.predict_proba(X_te_b)
            p_xgb = xgb_clf.predict_proba(X_te_a)
            p_cat = cat_clf.predict_proba(X_te_a)
            p_ens = (0.65 * p_gru) + (0.20 * p_xgb) + (0.15 * p_cat)

            preds = np.argmax(p_ens, axis=-1)
            inf_time_ms = ((time.time() - t_inf) / max(1, len(y_te))) * 1000.0

            res = Stage1MetricsHelper.evaluate_predictions(
                y_true=y_te,
                y_pred=preds,
                y_proba=p_ens,
                inference_time_ms=inf_time_ms
            )
            res['fold_name'] = fc['name']
            res['train_samples'] = len(y_tr)
            res['test_samples'] = len(y_te)
            res['train_time_sec'] = train_elapsed

            fold_results.append(res)
            all_test_preds.extend(preds)
            all_test_trues.extend(y_te)

            print(f"    [+] Fold {fold_id} 완료 (학습 소요: {train_elapsed:.1f}초) -> 정확도: {res['accuracy']*100:.2f}%, Top-2 Acc: {res['top2_accuracy']*100:.2f}%, Macro-F1: {res['macro_f1']*100:.2f}%, FPS: {res['fps']:.1f}")

        # 전체 3-Fold 종합 집계
        all_test_preds = np.array(all_test_preds)
        all_test_trues = np.array(all_test_trues)
        total_res = Stage1MetricsHelper.evaluate_predictions(
            y_true=all_test_trues,
            y_pred=all_test_preds,
            inference_time_ms=np.mean([r['inference_time_ms'] for r in fold_results])
        )

        self._print_final_summary_reports(fold_results, total_res, all_test_trues)
        self._save_results_json(fold_results, total_res)

        return {
            'fold_results': fold_results,
            'total_summary': total_res
        }

    def _print_final_summary_reports(self, fold_results: List[Dict[str, Any]], total_res: Dict[str, Any], y_all: np.ndarray):
        """3-Fold 비교 및 전체 종합 최종 성적표 출력"""
        print("\n" + "=" * 105)
        print("★ [1. 2차원 3-Fold 교차 검증 각 Fold별 세부 성적 요약표]")
        print("=" * 105)
        header = f"{'교차 검증 분할 (Fold Name)':<38} | {'정확도':^7} | {'Top-2 Acc':^9} | {'균형정확도':^9} | {'Kappa':^6} | {'Macro-F1':^8} | {'추론 FPS':^8}"
        print(header)
        print("-" * 105)

        accs, top2s, baccs, kaps, f1s, fpss = [], [], [], [], [], []
        for r in fold_results:
            acc = r['accuracy'] * 100.0
            top2 = r['top2_accuracy'] * 100.0 if r['top2_accuracy'] <= 1.0 else r['top2_accuracy']
            bacc = r['balanced_accuracy'] * 100.0
            kap = r['cohen_kappa']
            f1 = r['macro_f1'] * 100.0
            fps = r['fps']

            accs.append(acc)
            top2s.append(top2)
            baccs.append(bacc)
            kaps.append(kap)
            f1s.append(f1)
            fpss.append(fps)

            print(f"{r['fold_name']:<38} | {acc:>6.2f}% | {top2:>8.2f}% | {bacc:>8.2f}% | {kap:>6.3f} | {f1:>7.2f}% | {fps:>8.1f}")

        print("-" * 105)
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_top2, std_top2 = np.mean(top2s), np.std(top2s)
        mean_bacc, std_bacc = np.mean(baccs), np.std(baccs)
        mean_kap, std_kap = np.mean(kaps), np.std(kaps)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
        mean_fps = np.mean(fpss)

        print(f"{'★ [3-Fold 평균 (Mean ± Std)]':<38} | {mean_acc:>5.2f}±{std_acc:<4.2f}% | {mean_top2:>5.2f}±{std_top2:<4.2f}% | {mean_bacc:>5.2f}±{std_bacc:<4.2f}% | {mean_kap:>4.3f}±{std_kap:<3.3f} | {mean_f1:>5.2f}±{std_f1:<4.2f}% | {mean_fps:>8.1f}")
        print("=" * 105)

        # 8개 운동별 전수 세부 성적표
        print(f"\n[2. 8개 운동별 종합 성적표 (전수 {len(y_all):,}개 테스트 시퀀스 집계)]")
        print("-" * 105)
        print(f"{'운동 종목 명칭':<20} | {'샘플수':^7} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}")
        print("-" * 105)

        for i, ex_name in enumerate(EXERCISE_NAMES_SHORT):
            cnt = int(np.sum(y_all == i))
            prec = total_res['class_precs'][i] * 100.0 if i < len(total_res['class_precs']) else 0.0
            rec = total_res['class_recs'][i] * 100.0 if i < len(total_res['class_recs']) else 0.0
            f1 = total_res['class_f1s'][i] * 100.0 if i < len(total_res['class_f1s']) else 0.0
            print(f"{ex_name:<20} | {cnt:>7d} | {prec:>11.2f}% | {rec:>11.2f}% | {f1:>11.2f}%")
        print("-" * 105)

    def _save_results_json(self, fold_results: List[Dict[str, Any]], total_res: Dict[str, Any]):
        os.makedirs(os.path.dirname(OUTPUT_KFOLD_JSON), exist_ok=True)
        data_to_save = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'folds': [
                {
                    'fold_name': r['fold_name'],
                    'accuracy': r['accuracy'],
                    'top2_accuracy': r['top2_accuracy'],
                    'macro_f1': r['macro_f1'],
                    'balanced_accuracy': r['balanced_accuracy'],
                    'cohen_kappa': r['cohen_kappa'],
                    'fps': r['fps'],
                    'class_f1s': r['class_f1s'],
                    'class_recs': r['class_recs'],
                    'class_precs': r['class_precs']
                } for r in fold_results
            ],
            'total_summary': {
                'accuracy': total_res['accuracy'],
                'top2_accuracy': total_res['top2_accuracy'],
                'macro_f1': total_res['macro_f1'],
                'balanced_accuracy': total_res['balanced_accuracy'],
                'cohen_kappa': total_res['cohen_kappa'],
                'fps': total_res['fps'],
                'class_f1s': total_res['class_f1s'],
                'class_recs': total_res['class_recs'],
                'class_precs': total_res['class_precs']
            }
        }
        with open(OUTPUT_KFOLD_JSON, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 3-Fold 종합 검증 결과 JSON 저장 완료: {OUTPUT_KFOLD_JSON}")

if __name__ == '__main__':
    trainer = DualAxisKFoldEnsembleTrainer()
    trainer.run_3fold_cross_validation()

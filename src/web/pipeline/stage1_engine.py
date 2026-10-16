# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/stage1_engine.py
Purpose: 웹 서빙 전용 1단계 8종 운동 분류 실시간 단일 추론 엔진 (Stage1InferenceEngine)
Input: (64, 15, 3) or (64, 45) or (64, 33, 3) 관절 시퀀스
Output: {
    'exercise_id': int,
    'exercise_name': str,
    'confidence': float,
    'probabilities': Dict[str, float],
    'macro_group': str,
    'inference_time_ms': float
}
"""

import os
import sys
import time
import pickle
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.features.kinematics_v6 import KinematicsEngineV6
from src.features.kinematics_3d import KinematicsExtractor3D
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.core.skeleton2d import Joint15, SKELETON_EDGES
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

DEFAULT_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "src", "web", "weights", "stage1_v6_production_bundle.pkl")

from src.models.first_model.train_augmented_ensemble import DropKeypointBiGRUNet


class Stage1InferenceEngine:
    """1단계 8종 운동 분류 실시간 초고속 단일 추론 엔진 (웹 서비스 전용)"""

    _instance = None  # 싱글톤 캐시 지원

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH, device: str = 'cpu'):
        self.bundle_path = bundle_path
        self.device = torch.device(device)
        self._load_bundle()

    def _load_bundle(self):
        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(
                f"[Stage1InferenceEngine] 가중치 번들 파일을 찾을 수 없습니다: {self.bundle_path}\n"
                f"먼저 `python -m src.web.pipeline.train_and_export_stage1` 을 실행하여 가중치를 생성해 주세요."
            )

        class _NumpyCompatibleUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    alt_mod = module.replace("numpy._core", "numpy.core")
                    try:
                        return super().find_class(module, name)
                    except (ModuleNotFoundError, AttributeError):
                        return super().find_class(alt_mod, name)
                elif module.startswith("numpy.core"):
                    alt_mod = module.replace("numpy.core", "numpy._core")
                    try:
                        return super().find_class(module, name)
                    except (ModuleNotFoundError, AttributeError):
                        return super().find_class(alt_mod, name)
                return super().find_class(module, name)

        t0 = time.time()
        with open(self.bundle_path, 'rb') as f:
            try:
                self.bundle = pickle.load(f)
            except (ModuleNotFoundError, AttributeError):
                f.seek(0)
                self.bundle = _NumpyCompatibleUnpickler(f).load()

        # 1. Bi-GRU 신경망 로드 (체크포인트 텐서로부터 차원 동적 추론)
        sd = self.bundle['gru_model_state_dict']
        input_dim = sd['conv1.weight'].shape[1]
        conv_filters = sd['conv1.weight'].shape[0]
        hidden_dim = sd['gru.weight_ih_l0'].shape[0] // 3
        num_classes = sd['fc.3.weight'].shape[0] if 'fc.3.weight' in sd else 6

        self.gru_model = DropKeypointBiGRUNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            conv_filters=conv_filters,
            num_layers=2,
            dropout=0.0,
            channel_drop_prob=0.0
        ).to(self.device)
        self.gru_model.load_state_dict(sd)
        self.gru_model.eval()

        # 2. GBDT 트리 모델 및 가중치 로드
        self.lgb_macro = self.bundle['lgb_macro']
        self.xgb_macro = self.bundle['xgb_macro']
        self.weights = self.bundle['macro_ensemble_weights']  # [0.50, 0.30, 0.20]

        # 3. Stage 2 세부 Expert 모델 로드
        self.sub_dead_clf = self.bundle['sub_dead_clf']
        self.sub_lunge_clf = self.bundle['sub_lunge_clf']
        self.lunge_best_threshold = self.bundle.get('lunge_best_threshold', 0.50)
        self.dead_decision_threshold = self.bundle.get('dead_decision_threshold', 0.50)

        # 4. 메타데이터
        self.exercise_names_ko = self.bundle['exercise_names_ko']
        self.exercise_names_short = self.bundle['exercise_names_short']
        self.macro_group_names = self.bundle['macro_group_names']

        print(f"[Stage1InferenceEngine] 번들 로드 완료 ({time.time()-t0:.3f}초, 버전: {self.bundle.get('version')})")

    @classmethod
    def get_instance(cls, bundle_path: str = DEFAULT_BUNDLE_PATH, device: str = 'cpu'):
        """싱글톤 패턴 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(bundle_path=bundle_path, device=device)
        return cls._instance

    def _normalize_input(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        다양한 포맷의 입력을 (1, 64, 98) Format B 텐서와 (1, 64, 15, 3) 3D 관절 좌표로 변환
        지원 규격:
        1. (64, 98) 또는 (1, 64, 98): Format B 시계열 -> 15관절 역복원
        2. (64, 45) 또는 (1, 64, 45): 15관절 평탄화 -> (64, 15, 3) 복원 후 처리
        3. (T, 33, 3): MediaPipe 33관절 -> 15관절 매핑 + 64F 스플라인 + 기하정규화 + Format B (98D)
        4. (T, 26, 3): RTMPose 26관절 -> 15관절 매핑 + 64F 스플라인 + 기하정규화 + Format B (98D)
        5. (T, 15, 3): 15관절 -> 64F 스플라인 + 기하정규화 + Format B (98D)
        """
        seq = np.asarray(seq, dtype=np.float32)

        # 1. 2D 배열 입력 처리
        if seq.ndim == 2:
            if seq.shape == (64, 98):
                X_b = seq[np.newaxis, ...]  # (1, 64, 98)
                kps_3d = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(X_b)
                return X_b, kps_3d
            elif seq.shape == (64, 45):
                seq_3d = seq.reshape(64, 15, 3)
                return self._process_kps_15(seq_3d)
            elif seq.shape[1] == 98:
                # T 프레임 Format B -> 64 프레임 스플라인
                seq_64 = SkeletalTopologyAdapter.resample_to_64_frames(seq)
                X_b = seq_64[np.newaxis, ...]
                kps_3d = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(X_b)
                return X_b, kps_3d
            else:
                raise ValueError(f"지원하지 않는 2D 텐서 규격입니다: {seq.shape}. 기대 규격: (64, 98) 또는 (64, 45)")

        # 2. 3D 배열 입력 처리
        if seq.ndim == 3:
            # (1, 64, 98)인 경우
            if seq.shape == (1, 64, 98):
                kps_3d = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(seq)
                return seq, kps_3d
            elif seq.shape == (1, 64, 45):
                seq_3d = seq[0].reshape(64, 15, 3)
                return self._process_kps_15(seq_3d)

            T, J, C = seq.shape
            if J == 33:
                kps_15 = SkeletalTopologyAdapter.mediapipe_33_to_15(seq[:, :, :3])
                return self._process_kps_15(kps_15)
            elif J == 26:
                kps_15 = SkeletalTopologyAdapter.rtmpose_26_to_15(seq[:, :, :3])
                return self._process_kps_15(kps_15)
            elif J == 15:
                return self._process_kps_15(seq[:, :, :3])
            else:
                raise ValueError(f"지원하지 않는 관절 수입니다: J={J}. 지원: 15, 26, 33")

        raise ValueError(f"지원하지 않는 텐서 차원입니다: ndim={seq.ndim}")

    def _process_kps_15(self, kps_15: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(T, 15, 3) 관절 시퀀스를 64F 스플라인, 3D 기하 정규화 및 Format B (1, 64, 98)로 변환"""
        # 1. 64 프레임 3차 스플라인 리샘플링 (T != 64 인 경우)
        if len(kps_15) != 64:
            kps_15 = SkeletalTopologyAdapter.resample_to_64_frames(kps_15)

        # 2. 3D 기하 정규화 (골반 원점, 회전 정렬, 체간 스케일링)
        norm_res = GeometricNormalizer3D.process_sequence_3d(kps_15)
        kps_norm = norm_res['keypoints_3d_norm'] if norm_res is not None else kps_15

        # 3. Format B 92차원 시계열 피처 추출
        fb_raw = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm)  # (64, 92)

        # 4. 98차원 규격 제로 패딩
        if fb_raw.shape[-1] < 98:
            pad = np.zeros((64, 98 - fb_raw.shape[-1]), dtype=np.float32)
            fb_98 = np.concatenate([fb_raw, pad], axis=-1)
        else:
            fb_98 = fb_raw[:, :98]

        X_b = fb_98[np.newaxis, ...].astype(np.float32)  # (1, 64, 98)
        kps_3d = kps_norm[np.newaxis, ...].astype(np.float32)  # (1, 64, 15, 3)
        return X_b, kps_3d

    def predict(self, sequence: np.ndarray) -> dict:
        """
        단일 시퀀스(Single Repetition 64프레임)에 대한 8종 운동 분류 수행
        """
        t0 = time.time()

        # 1. 입력 정규화
        X_b, kps_3d = self._normalize_input(sequence)

        # 2. 59D Model F Direct KCF 피처 추출
        X_kcf = KinematicsEngineV6.build_model_f_direct_kcf(kps_3d)  # (1, 59)

        # 3. Stage 1: 6대 상위 그룹 3중 앙상블 추론
        # (A) Bi-GRU
        with torch.no_grad():
            inp_t = torch.from_numpy(X_b).to(self.device)
            logits = self.gru_model(inp_t)
            P_gru = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (6,)

        # (B) LightGBM
        P_lgb = self.lgb_macro.predict_proba(X_kcf)[0]  # (6,)

        # (C) XGBoost
        P_xgb = self.xgb_macro.predict_proba(X_kcf)[0]  # (6,)

        # 3중 볼록 결합
        P_macro = self.weights[0] * P_gru + self.weights[1] * P_lgb + self.weights[2] * P_xgb
        macro_idx = int(np.argmax(P_macro))

        # 4. Stage 2 세부 Expert 조건부 실행
        # (A) 데드리프트 패밀리 (Group 1: 일반 vs 스티프)
        P_dead_stiff = float(self.sub_dead_clf.predict_proba(X_b)[0, 1])

        # (B) 무빙 런지 패밀리 (Group 3: 다이나믹 vs 크로스)
        X_lunge_sub = LungeSubClassifier.slice_features(X_b)
        P_lunge_cross = float(self.sub_lunge_clf.predict_proba(X_lunge_sub)[0, 1])

        # 5. 소프트 베이지안 확률 결합 (8개 세부 종목)
        P_final = np.zeros(8, dtype=np.float32)
        P_final[0] = P_macro[0]  # 01. 바벨 스쿼트
        P_final[3] = P_macro[2]  # 04. 바벨 런지 (단독)
        P_final[5] = P_macro[4]  # 06. 사이드 런지
        P_final[7] = P_macro[5]  # 08. 굿모닝

        P_dead_macro = P_macro[1]
        P_final[1] = P_dead_macro * (1.0 - P_dead_stiff)  # 02. 바벨 데드리프트
        P_final[2] = P_dead_macro * P_dead_stiff          # 03. 바벨 스티프 데드리프트

        P_lunge_macro = P_macro[3]
        P_final[4] = P_lunge_macro * (1.0 - P_lunge_cross)  # 05. 다이나믹 런지
        P_final[6] = P_lunge_macro * P_lunge_cross          # 07. 크로스 런지

        # 최종 클래스 결정
        final_id = int(np.argmax(P_final))

        # 세부 Expert 임계값 보정
        if macro_idx == 3:  # 무빙 런지 패밀리
            final_id = 6 if P_lunge_cross >= self.lunge_best_threshold else 4
        elif macro_idx == 1:  # 데드리프트 패밀리
            final_id = 2 if P_dead_stiff >= self.dead_decision_threshold else 1

        elapsed_ms = (time.time() - t0) * 1000.0

        # 반환 딕셔너리 생성
        prob_dict = {}
        for idx, name in enumerate(self.exercise_names_ko):
            prob_dict[name] = round(float(P_final[idx]), 4)

        return {
            'exercise_id': final_id,
            'exercise_name': self.exercise_names_ko[final_id],
            'exercise_short': self.exercise_names_short[final_id],
            'confidence': round(float(P_final[final_id]), 4),
            'macro_group': self.macro_group_names[macro_idx],
            'probabilities': prob_dict,
            'inference_time_ms': round(elapsed_ms, 2)
        }

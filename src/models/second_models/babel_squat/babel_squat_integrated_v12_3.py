# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v12_3.py
Purpose: v12.3 [전체 해결책 통합 마스터] 통합 진단 엔진
         - SpinePose 순수 비전 척추 추출 및 생체역학 노이즈 제거 파이프라인 공식 결합
         - Layer 1: 4대 부위별 3-State 독립 진단 (c1 v11.1 Distilled SpinePose, c2 v8.0, c3 v6.0, c4 v5.1)
         - Layer 2: 39D View-Conditioned MoE 메타 신경망 (FiLM + Energy Head)
         - Layer 3: Model-B 맞춤형 5대 시점 16D 벡터 스케일러 및 온도 보정
         - Layer 4: 시계열 물리 필터링 (Wobble, Jerk, F0-Rate) + 위험도 기반 동적 게이트
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.features.skeletal_normalizer import normalize_skeletal_lengths
from src.features.spinepose_denoiser import denoise_spinepose_sequence
from src.models.second_models.babel_squat.c1_spinepose_extractor import BarbellSquatC1SpinePoseExtractor
from src.models.second_models.babel_squat.stgcn_transformer_net import STGCNTransformerResidualNet
from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c2_model_v8 import BabelSquatC2ModelV8
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1
from src.models.second_models.babel_squat.joint_meta_16way_v12_3_net import JointMeta16WayMoENetV12_3
from src.models.second_models.babel_squat.joint_meta_16way_net import decode_hybrid

VIEW_ALIAS_MAP = {
    '시점 A': 'view5', '시점A': 'view5', 'A': 'view5', 'view5': 'view5',
    '시점 B': 'view2', '시점B': 'view2', 'B': 'view2', 'view2': 'view2',
    '시점 C': 'view3', '시점C': 'view3', 'C': 'view3', 'view3': 'view3',
    '시점 D': 'view4', '시점D': 'view4', 'D': 'view4', 'view4': 'view4',
    '시점 E': 'view1', '시점E': 'view1', 'E': 'view1', 'view1': 'view1',
}

VIEW_METADATA = {
    'view5': {'code': '시점 A', 'name': '우측 후방 (45도)', 'desc': 'Right Back 45°'},
    'view2': {'code': '시점 B', 'name': '우측 전방 (45도)', 'desc': 'Right Front 45°'},
    'view3': {'code': '시점 C', 'name': '정면 (0도)', 'desc': 'Front View 0°'},
    'view4': {'code': '시점 D', 'name': '좌측 전방 (45도)', 'desc': 'Left Front 45°'},
    'view1': {'code': '시점 E', 'name': '좌측 후방 (45도)', 'desc': 'Left Back 45°'},
}

DEFECT_METADATA = {
    'c1': {
        'code': 'c1',
        'name': '척추의 중립 (Spine Neutrality)',
        'description': '요추 전만/후만(버트윙크) 및 과도한 상체 숙임',
        'normal_feedback': '척추의 중립 정렬이 바르고 안정적으로 유지되었습니다.',
        'defect_feedback': '상체가 지나치게 앞으로 숙여지거나 허리(요추)가 말려 척추 중립이 위반되었습니다.',
        'defect_action': '복압(코어)을 단단히 유지하고 시선을 고정한 채 엉덩이가 말리지 않는 깊이까지만 수행하세요.',
        'abstain_feedback': '촬영 각도상 척추 곡률이 가려지거나 움직임이 모호하여 척추 중립 판정을 보류합니다. (대각선 45도 측면 촬영 권장)'
    },
    'c2': {
        'code': 'c2',
        'name': '고개 정면 및 시선 (Head & Gaze)',
        'description': '고개 숙임/들림 및 경추 과신전/굴곡',
        'normal_feedback': '고개와 경추가 정면을 향해 안정적으로 일직선을 유지했습니다.',
        'defect_feedback': '고개를 지나치게 숙여 바닥을 보거나 목이 뒤로 꺾여 경추 정렬이 무너졌습니다.',
        'defect_action': '턱을 가볍게 당기고 정면 거울이나 시선 정면 한 점을 응시하며 동작을 수행하세요.',
        'abstain_feedback': '촬영 각도상 얼굴이 가려지거나 왜곡되어 정확한 시선 판정이 어렵습니다. (정면 또는 45도 측면 촬영 권장)'
    },
    'c3': {
        'code': 'c3',
        'name': '발과 무릎의 방향 일치 (Knee Valgus)',
        'description': '하강 및 상승 시 무릎 내측 모임(외반)',
        'normal_feedback': '스쿼트 동작 중 무릎과 발끝의 방향이 완벽하게 일치하였습니다.',
        'defect_feedback': '스쿼트 하강/상승 시 무릎이 안쪽으로 모이거나 흔들리는 외반(Valgus)이 발생했습니다.',
        'defect_action': '발끝 방향(약 15~30도)과 동일하게 무릎을 바깥쪽으로 벌리며 앉으세요.',
        'abstain_feedback': '무릎 및 발끝의 정렬 상태가 경계 구간에 있어 판정을 보류합니다. (정면 또는 45도 측면 권장)'
    },
    'c4': {
        'code': 'c4',
        'name': '발바닥 지면 고정 (Foot Ground Fixation)',
        'description': '스쿼트 중 발뒤꿈치 또는 앞꿈치 들림',
        'normal_feedback': '발바닥 전체가 지면에 안정적으로 밀착되어 완벽한 접지력을 보였습니다.',
        'defect_feedback': '스쿼트 동작 중 발뒤꿈치나 앞꿈치가 지면에서 들려 중심이 무너졌습니다.',
        'defect_action': '발바닥 삼각점(엄지발가락, 새끼발가락, 뒤꿈치)에 체중을 고르게 분산시키고 지면을 꽉 딛으세요.',
        'abstain_feedback': '발 접지면 랜드마크가 카메라 프레임 하단에 가려져 발 접지 판정을 보류합니다. (발끝까지 보이도록 촬영 권장)'
    }
}

COMPONENT_ORDER = ['c1', 'c2', 'c3', 'c4']


class BabelSquatIntegratedEngineV12_3:
    """
    v12.3 [전체 해결책 통합 마스터] 바벨 스쿼트 통합 진단 엔진
    - 순수 비전 SpinePose 실시간 연동 및 Model-B 맞춤형 재교정 반영
    """
    VERSION = "v12.3"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        enable_selective: bool = True,
        q_min: float = 0.05,
        deadband_low: float = 0.30,
        deadband_high: float = 0.36,
        margin_retake_threshold: float = 0.05,
        risk_threshold: float = 0.70,
        wobble_threshold: float = 0.008,
        jerk_threshold: float = 220.0,
        f0_rate_threshold: float = 0.35,
        device: Optional[str] = None
    ):
        self.weights_dir = weights_dir or CURRENT_DIR
        self.enable_selective = enable_selective
        self.q_min = q_min
        self.deadband_low = deadband_low
        self.deadband_high = deadband_high
        self.margin_retake_threshold = margin_retake_threshold
        self.risk_threshold = risk_threshold
        self.wobble_threshold = wobble_threshold
        self.jerk_threshold = jerk_threshold
        self.f0_rate_threshold = f0_rate_threshold
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # 1. 시점별 16D 벡터 스케일러 로드
        scaler_path = os.path.join(self.weights_dir, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}

        # 2. 14D ST-GCN + Transformer 잔차 신경망 로드
        self.stgcn_weights_path = os.path.join(self.weights_dir, "stgcn_transformer_residual_weights.pth")
        self.stgcn_model = None
        self.g_front = 0.08
        self.g_rear  = 0.00
        if os.path.exists(self.stgcn_weights_path):
            try:
                self.stgcn_model = STGCNTransformerResidualNet().to(self.device)
                self.stgcn_model.load_state_dict(torch.load(self.stgcn_weights_path, map_location=self.device, weights_only=True))
                self.stgcn_model.eval()
            except Exception as e:
                warnings.warn(f"Failed to load STGCNTransformerResidualNet: {e}")
                self.stgcn_model = None

        # 3. 서브모델 초기화 (Layer 1)
        w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')

        self.c1_model = BabelSquatC1ModelV11_1(weights_path=w_c1 if os.path.exists(w_c1) else None)
        self.c2_model = BabelSquatC2ModelV8(weights_path=w_c2 if os.path.exists(w_c2) else None)
        self.c3_model = BabelSquatC3ModelV6(weights_path=w_c3 if os.path.exists(w_c3) else None)
        self.c4_model = BabelSquatC4ModelV5_1(weights_path=w_c4 if os.path.exists(w_c4) else None)

        # 4. 온도 파라미터 로드
        temp_param_file = os.path.join(self.weights_dir, "temperature_params.json")
        if os.path.exists(temp_param_file):
            with open(temp_param_file, 'r', encoding='utf-8') as f:
                t_cfg = json.load(f)
            self.T_c1 = t_cfg['c1']['global_T']
            self.T_c3 = t_cfg['c3']['global_T']
            self.T_c4 = t_cfg['c4']['global_T']
            self.view_T_c2 = t_cfg['c2']['view_T']
        else:
            self.T_c1, self.T_c3, self.T_c4 = 0.456, 1.112, 1.001
            self.view_T_c2 = {'view5': 0.89, 'view2': 0.90, 'view3': 0.93, 'view4': 0.92, 'view1': 0.96}

        # 5. 39D View-Conditioned MoE 로드
        self.joint_model = JointMeta16WayMoENetV12_3(in_features=39, hidden_dim=48, dropout_rate=0.20).to(self.device)
        weights_v12_3 = os.path.join(self.weights_dir, "joint_meta_16way_v12_3_weights.pth")
        if os.path.exists(weights_v12_3):
            self.joint_model.load_state_dict(torch.load(weights_v12_3, map_location=self.device, weights_only=True))
        self.joint_model.eval()

        self.tau_clean = 0.36
        self.eta = 0.05

    @staticmethod
    def normalize_view(view: str) -> str:
        v = str(view).strip()
        return VIEW_ALIAS_MAP.get(v, VIEW_ALIAS_MAP.get(f"시점 {v}", 'view3'))

    def compute_12d_quality(self, kps_3d: np.ndarray) -> np.ndarray:
        sh_mid = (kps_3d[:, 11, :] + kps_3d[:, 12, :]) / 2.0
        hip_mid = (kps_3d[:, 23, :] + kps_3d[:, 24, :]) / 2.0
        torso_len = np.linalg.norm(sh_mid - hip_mid, axis=1)
        t_mean = float(np.mean(torso_len)) + 1e-6
        t_rel_std = float(np.std(torso_len) / t_mean)

        femur_l = np.linalg.norm(kps_3d[:, 23, :] - kps_3d[:, 25, :], axis=1)
        femur_r = np.linalg.norm(kps_3d[:, 24, :] - kps_3d[:, 26, :], axis=1)
        femur_rel_std = float((np.std(femur_l) / (np.mean(femur_l) + 1e-6) + np.std(femur_r) / (np.mean(femur_r) + 1e-6)) / 2.0)

        tibia_l = np.linalg.norm(kps_3d[:, 25, :] - kps_3d[:, 27, :], axis=1)
        tibia_r = np.linalg.norm(kps_3d[:, 26, :] - kps_3d[:, 28, :], axis=1)
        tibia_rel_std = float((np.std(tibia_l) / (np.mean(tibia_l) + 1e-6) + np.std(tibia_r) / (np.mean(tibia_r) + 1e-6)) / 2.0)

        bone_instability = float((t_rel_std + femur_rel_std + tibia_rel_std) / 3.0)
        q_bone = float(np.exp(-10.0 * bone_instability))

        acc = np.diff(hip_mid, n=2, axis=0)
        jitter = float(np.mean(np.linalg.norm(acc, axis=1)) / t_mean)
        q_jitter = float(np.exp(-20.0 * jitter))
        q_overall = float(min(q_bone, q_jitter))

        hip_y = hip_mid[:, 1]
        nadir_idx = int(np.argmax(hip_y))
        nadir_norm = float(nadir_idx / 63.0)
        rom_depth = float((np.max(hip_y) - np.min(hip_y)) / t_mean)

        sh_w = float(np.mean(np.linalg.norm(kps_3d[:, 11, :] - kps_3d[:, 12, :], axis=1)) / t_mean)
        hip_w = float(np.mean(np.linalg.norm(kps_3d[:, 23, :] - kps_3d[:, 24, :], axis=1)) / t_mean)
        z_knee_diff = float(np.mean(np.abs(kps_3d[:, 25, 2] - kps_3d[:, 26, 2])) / t_mean)
        z_ank_diff = float(np.mean(np.abs(kps_3d[:, 27, 2] - kps_3d[:, 28, 2])) / t_mean)

        ear_mid = (kps_3d[:, 7, :] + kps_3d[:, 8, :]) / 2.0
        head_dist = np.linalg.norm(kps_3d[:, 0, :] - ear_mid, axis=1)
        head_vis = float(np.mean(head_dist) / t_mean)

        return np.array([
            q_bone, q_jitter, q_overall, bone_instability, jitter,
            nadir_norm, rom_depth, sh_w, hip_w, z_knee_diff, z_ank_diff, head_vis
        ], dtype=np.float32)

    def compute_9d_kinetic(self, kps_3d: np.ndarray) -> np.ndarray:
        hip_mid = (kps_3d[:, 23, :] + kps_3d[:, 24, :]) / 2.0
        knee_mid = (kps_3d[:, 25, :] + kps_3d[:, 26, :]) / 2.0
        ank_mid = (kps_3d[:, 27, :] + kps_3d[:, 28, :]) / 2.0
        sh_mid = (kps_3d[:, 11, :] + kps_3d[:, 12, :]) / 2.0

        v_thigh = hip_mid - knee_mid
        v_shank = knee_mid - ank_mid
        v_trunk = sh_mid - hip_mid

        def safe_c(a, b):
            m_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-7
            m_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-7
            dot = np.sum((a / m_a) * (b / m_b), axis=1)
            return float(np.mean(dot))

        c_fk = safe_c(v_thigh, v_shank)
        c_kt = safe_c(v_shank, v_trunk)
        c_tn = safe_c(v_trunk, v_thigh)
        c_ft = safe_c(v_thigh, v_trunk)

        w_knee = np.linalg.norm(kps_3d[:, 25, :] - kps_3d[:, 26, :], axis=1)
        w_ank = np.linalg.norm(kps_3d[:, 27, :] - kps_3d[:, 28, :], axis=1) + 1e-6
        d_valg = float(np.max(w_knee / w_ank) - np.min(w_knee / w_ank))

        n_idx = int(np.argmax(hip_mid[:, 1]))
        n_tilt = float(np.abs(v_trunk[n_idx, 0]) / (np.linalg.norm(v_trunk[n_idx]) + 1e-6))
        n_valg = float(w_knee[n_idx] / w_ank[n_idx])

        a_heel = float(np.max(np.abs(kps_3d[:, 29, 1] - kps_3d[:, 30, 1])))
        p_lag = float(np.argmax(v_thigh[:, 1]) - np.argmax(v_trunk[:, 1])) / 63.0

        return np.array([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag], dtype=np.float32)

    def predict_integrated(
        self,
        kps_sequence_3d: np.ndarray,
        camera_view: str,
        spine_9_sequence_3d: Optional[np.ndarray] = None,
        raw_video_frames: Optional[List[np.ndarray]] = None,
        c4_raw_feats: Optional[np.ndarray] = None,
        quality_raw_feats: Optional[np.ndarray] = None,
        kinetic_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        통합 진단 메인 엔트리포인트 (v12.3 Master)
        - spine_9_sequence_3d가 주어지면 직접 주입
        - 없으면 raw_video_frames로부터 SpinePose 추출 + One-Euro 노이즈 제거 후 c1에 투입
        """
        norm_view = self.normalize_view(camera_view)

        # SpinePose 자동 추출 및 노이즈 제거 브릿지
        if spine_9_sequence_3d is None and raw_video_frames is not None:
            try:
                extractor = BarbellSquatC1SpinePoseExtractor()
                spine_raw = extractor.extract_from_frames(raw_video_frames)
                spine_9_sequence_3d = denoise_spinepose_sequence(spine_raw)
            except Exception:
                spine_9_sequence_3d = None

        stream_a, stream_b, norm_metrics = normalize_skeletal_lengths(kps_sequence_3d)

        sub_results = {}
        # c1: 청정 SpinePose 9개 관절 주입
        sub_results['c1'] = self.c1_model.predict_selective(stream_b, spine_9=spine_9_sequence_3d)
        # c2: 고개
        sub_results['c2'] = self.c2_model.predict_selective(kps_sequence_3d, camera_view=norm_view)

        # c3: 무릎
        res_c3_raw = self.c3_model.predict_selective(stream_b)
        res_c3_norm = self.c3_model.predict_selective(stream_a)
        p3_r = float(res_c3_raw.get('defect_probability') or 0.5)
        p3_n = float(res_c3_norm.get('defect_probability') or 0.5)
        p3_fused = float(np.clip(0.70 * p3_r + 0.30 * p3_n, 0.0, 1.0))
        res_c3 = res_c3_raw.copy()
        res_c3['defect_probability'] = p3_fused
        res_c3['status'] = 'NORMAL' if p3_fused <= 0.40 else ('DEFECT' if p3_fused >= 0.60 else 'UNDETERMINED')
        sub_results['c3'] = res_c3

        # c4: 발바닥
        sub_results['c4'] = self.c4_model.predict_selective(stream_b, camera_view=norm_view)

        if c4_raw_feats is None:
            c4_raw_feats = self.c4_model.extract_features(stream_b)
        if quality_raw_feats is None and kps_sequence_3d.shape[0] == 64:
            quality_raw_feats = self.compute_12d_quality(stream_b)
        if kinetic_raw_feats is None and kps_sequence_3d.shape[0] == 64:
            kinetic_raw_feats = self.compute_9d_kinetic(kps_sequence_3d)

        # 확률 수집
        probs_raw = np.array([
            sub_results['c1'].get('defect_probability') or 0.5,
            sub_results['c2'].get('defect_probability') or 0.5,
            sub_results['c3'].get('defect_probability') or 0.5,
            sub_results['c4'].get('defect_probability') or 0.5,
        ], dtype=np.float32)

        eps = 1e-7
        p_clip = np.clip(probs_raw, eps, 1.0 - eps)
        logits_raw = np.log(p_clip / (1.0 - p_clip))

        # 온도 보정
        T_v2 = self.view_T_c2.get(norm_view, 0.90)
        p_cal = np.array([
            1.0 / (1.0 + np.exp(-logits_raw[0] / self.T_c1)),
            1.0 / (1.0 + np.exp(-logits_raw[1] / T_v2)),
            1.0 / (1.0 + np.exp(-logits_raw[2] / self.T_c3)),
            1.0 / (1.0 + np.exp(-logits_raw[3] / self.T_c4)),
        ], dtype=np.float32)
        margins_cal = np.abs(p_cal - 0.5)

        # 물리 피처
        wobble = c4_raw_feats[3]; jerk = c4_raw_feats[15]; f0_rate = c4_raw_feats[24]
        jerk_norm   = np.clip(jerk   / 200.0, 0.0, 2.5)
        wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)
        ydelta_norm = np.clip(c4_raw_feats[6] / 0.020, -1.0, 2.5)
        toe_norm    = np.clip(c4_raw_feats[1] / 0.020,  0.0, 2.5)
        f0_norm     = np.clip(f0_rate / 0.50,       0.0, 2.0)
        phys_feats  = np.array([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm], dtype=np.float32)

        view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
        view_onehot = np.eye(5, dtype=np.float32)[view_map.get(norm_view, 2)]

        # 품질 피처 정규화
        q_bone     = quality_raw_feats[0]; q_jitter = quality_raw_feats[1]; q_overall_f = quality_raw_feats[2]
        bone_instab= np.clip(quality_raw_feats[3]   / 0.20,          0.0, 2.5)
        jitter_s   = np.clip(quality_raw_feats[4]   / 0.001,          0.0, 2.5)
        nadir_s    = quality_raw_feats[5]
        rom_s      = np.clip(quality_raw_feats[6]   / 0.015,          0.0, 2.5)
        sh_w_s     = np.clip((quality_raw_feats[7]  - 0.65) / 0.15,  -2.5, 2.5)
        hip_w_s    = np.clip((quality_raw_feats[8]  - 0.42) / 0.10,  -2.5, 2.5)
        z_knee_s   = np.clip(quality_raw_feats[9]   / 0.40,           0.0, 2.5)
        z_ank_s    = np.clip(quality_raw_feats[10]  / 0.45,           0.0, 2.5)
        head_vis_s = np.clip((quality_raw_feats[11] - 0.27) / 0.05,  -2.5, 2.5)
        qual_feats = np.array([q_bone, q_jitter, q_overall_f, bone_instab, jitter_s,
                               nadir_s, rom_s, sh_w_s, hip_w_s, z_knee_s, z_ank_s, head_vis_s], dtype=np.float32)

        # 운동사슬 피처 정규화
        c_fk   = kinetic_raw_feats[0]; c_kt = kinetic_raw_feats[1]
        c_tn   = kinetic_raw_feats[2]; c_ft = kinetic_raw_feats[3]
        d_valg = np.clip(kinetic_raw_feats[4]  / 0.40,          -2.5, 2.5)
        n_tilt = np.clip(kinetic_raw_feats[5]  / 0.50,           0.0, 2.5)
        n_valg = np.clip((kinetic_raw_feats[6] - 1.40) / 0.35,  -2.5, 2.5)
        a_heel = np.clip(kinetic_raw_feats[7]  / 0.60,           0.0, 2.5)
        p_lag  = np.clip(kinetic_raw_feats[8]  / 0.40,          -2.5, 2.5)
        kin_feats = np.array([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag], dtype=np.float32)

        X_39d = np.concatenate([p_cal, margins_cal, view_onehot, phys_feats, qual_feats, kin_feats]).reshape(1, 39)

        # MoE 신경망 추론
        with torch.no_grad():
            raw_l16, raw_l4, _ = self.joint_model(torch.from_numpy(X_39d).float().to(self.device))
            z_16 = raw_l16.cpu().numpy()[0]
            p4_pred = torch.sigmoid(raw_l4).cpu().numpy()[0]

        # 16D 맞춤형 벡터 스케일러 적용
        if norm_view in self.view_vector_scalers:
            sc = np.array(self.view_vector_scalers[norm_view]['scale'], dtype=np.float32)
            bi = np.array(self.view_vector_scalers[norm_view]['bias'], dtype=np.float32)
            z_16 = z_16 * sc + bi

        exp_z = np.exp(z_16 - np.max(z_16))
        probs_16 = exp_z / np.sum(exp_z)

        # 위험도 계산
        H_max = np.log(16.0)
        log_p = np.log(np.clip(probs_16, 1e-8, 1.0))
        ent = -np.sum(probs_16 * log_p)
        sp = np.sort(probs_16)
        margin = sp[-1] - sp[-2]
        risk = float((ent / H_max) * (1.0 - margin))

        # 하이브리드 디코딩
        s_idx, pred_defs, is_clean, p_clean = decode_hybrid(probs_16, p4_pred, eta=self.eta, tau_clean=self.tau_clean)

        # 물리 필터링 보정
        if is_clean and ((jerk > 220.0 and wobble > 0.008) or f0_rate > 0.35):
            is_clean = False
            pred_defs = np.array([0, 0, 0, 1], dtype=int)
        elif (not is_clean) and np.max(p_cal) <= 0.50 and (wobble <= 0.008 and jerk <= 180.0):
            is_clean = True
            pred_defs = np.array([0, 0, 0, 0], dtype=int)

        # 3-State 선택적 판정 상태 결정
        is_retake = False
        retake_reason = None
        if self.enable_selective:
            if q_overall_f < self.q_min:
                is_retake = True
                retake_reason = "신체 관측 품질 미달 (관절 가림 또는 급격한 떨림)"
            elif risk > self.risk_threshold and not is_clean:
                is_retake = True
                retake_reason = "다중 결함 패턴 모호 (카메라 각도 재조정 권고)"

        final_status = "RETAKE_RECOMMENDED" if is_retake else ("CLEAN" if is_clean else "DEFECT")

        component_reports = {}
        for idx, k in enumerate(COMPONENT_ORDER):
            meta = DEFECT_METADATA[k]
            st = "UNDETERMINED" if is_retake else ("DEFECT" if pred_defs[idx] == 1 else "NORMAL")
            fb = meta['normal_feedback'] if st == 'NORMAL' else (meta['defect_feedback'] if st == 'DEFECT' else meta['abstain_feedback'])
            component_reports[k] = {
                'code': k,
                'name': meta['name'],
                'status': st,
                'probability': float(p_cal[idx]),
                'feedback': fb
            }

        return {
            'session_status': final_status,
            'is_clean': bool(is_clean),
            'clean_probability': float(probs_16[0]),
            'risk_score': round(risk, 4),
            'camera_view': norm_view,
            'view_metadata': VIEW_METADATA.get(norm_view, {}),
            'retake_recommended': is_retake,
            'retake_reason': retake_reason,
            'component_reports': component_reports,
            'probabilities_16way': probs_16.tolist()
        }


BabelSquatIntegratedEngineV12_Master = BabelSquatIntegratedEngineV12_3
BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV12_3

if __name__ == '__main__':
    print("=" * 90)
    print("  [vidio-health-trainer] BabelSquatIntegratedEngineV12_3 단위 테스트 가동")
    print("=" * 90)
    engine = BabelSquatIntegratedEngineV12_3()
    dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
    dummy_spine = np.random.randn(64, 9, 3).astype(np.float32)
    res = engine.predict_integrated(dummy_kps, camera_view="시점 C", spine_9_sequence_3d=dummy_spine)
    print(f"[+] 통합 진단 성공! Status: {res['session_status']} | Clean Prob: {res['clean_probability']:.4f} | Risk: {res['risk_score']}")
    print(f"    - c1: {res['component_reports']['c1']['status']} ({res['component_reports']['c1']['probability']:.2f})")
    print("=" * 90)

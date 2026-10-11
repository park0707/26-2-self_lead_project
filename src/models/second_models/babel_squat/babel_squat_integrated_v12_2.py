# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v12_2.py
Purpose: v12.2 [해결책 1 + 해결책 2 + 시점별 벡터 스케일러 재조정] 통합 진단 엔진
         - [해결책 1]: Factorized Pairwise Energy Head (Unary 4D + Pairwise 6D MLP)
         - [해결책 2]: Soft-EMR + Soft-Clean F1 지표 정렬 복합 손실
         - [사후 보정]: v12.2 전용 5대 시점별 16D 벡터 스케일러 (view_vector_scaler_params_v12_2.json)
         - [시공간 잔차]: 14D ST-GCN + Temporal Transformer 잔차 Late Fusion (g=0.08)
         - [선택적 판정]: 3-State 동적 위험도 게이트 (NORMAL / DEFECT VECTOR / RETAKE)
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
from src.models.second_models.babel_squat.stgcn_transformer_net import STGCNTransformerResidualNet
from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c2_model_v8 import BabelSquatC2ModelV8
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1
from src.models.second_models.babel_squat.joint_meta_16way_v12_2_net import JointMeta16WayMoENetV12_2
from src.models.second_models.babel_squat.joint_meta_16way_net import (
    decode_hybrid,
    idx_to_defect_tuple,
    defect_tuple_to_idx
)

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
        'abstain_feedback': '촬영 각도상 얼굴이 가려지거나 왜곡되어 정확한 시선 판정이 어렵습니다. (정면 또는 45도 측면 촬영 권장)',
        'not_evaluated_feedback': '후방 촬영 각도로 인해 얼굴이 가려져 이번 세션에서는 시선 평가를 제외하였습니다.'
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


class BabelSquatIntegratedEngineV12_2:
    """
    v12.2 [해결책 1 + 해결책 2] 바벨 스쿼트 통합 진단 엔진
    """
    VERSION = "v12.2"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        enable_selective: bool = True,
        q_min: float = 0.05,
        deadband_low: float = 0.30,
        deadband_high: float = 0.36,
        margin_retake_threshold: float = 0.05,
        risk_threshold: float = 0.45,
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

        # [v12.2]: v12.2 전용 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(self.weights_dir, "view_vector_scaler_params_v12_2.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}

        # 14D ST-GCN + Temporal Transformer 잔차 신경망 로드
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

        # 서브모델 초기화 (Layer 1)
        w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')

        self.c1_model = BabelSquatC1ModelV11_1(weights_path=w_c1 if os.path.exists(w_c1) else None)
        self.c2_model = BabelSquatC2ModelV8(weights_path=w_c2 if os.path.exists(w_c2) else None)
        self.c3_model = BabelSquatC3ModelV6(weights_path=w_c3 if os.path.exists(w_c3) else None)
        self.c4_model = BabelSquatC4ModelV5_1(weights_path=w_c4 if os.path.exists(w_c4) else None)

        # 온도 파라미터 로드
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

        # 39D View-Conditioned MoE v12.2 로드
        self.joint_model = JointMeta16WayMoENetV12_2(in_features=39, hidden_dim=48, dropout_rate=0.20).to(self.device)
        weights_v12_2 = os.path.join(self.weights_dir, "joint_meta_16way_v12_2_weights.pth")
        if os.path.exists(weights_v12_2):
            self.joint_model.load_state_dict(torch.load(weights_v12_2, map_location=self.device))
        self.joint_model.eval()

    @staticmethod
    def extract_14d_tensor(kps_64: np.ndarray) -> np.ndarray:
        mid_hip = (kps_64[:, 23:24, :] + kps_64[:, 24:25, :]) * 0.5
        rel_pos = kps_64 - mid_hip
        vel = np.gradient(rel_pos, axis=0)
        acc = np.gradient(vel, axis=0)
        parents = [
            0, 0, 1, 2, 0, 4, 5, 3, 6, 0, 0,
            12, 11, 11, 12, 13, 14, 15, 16, 15, 16, 15, 16,
            11, 12, 23, 24, 25, 26, 27, 28, 27, 28
        ]
        parent_pos = rel_pos[:, parents, :]
        bone_vec = rel_pos - parent_pos
        bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
        bone_dir = bone_vec / (bone_len + 1e-7)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        comb = np.concatenate([rel_pos, vel, acc, bone_dir, bone_len, speed], axis=-1)
        return np.transpose(comb, (2, 0, 1)).astype(np.float32)

    @staticmethod
    def normalize_view(view: str) -> str:
        v = view.strip()
        if v in VIEW_ALIAS_MAP:
            return VIEW_ALIAS_MAP[v]
        return 'view3'


BabelSquatIntegratedEngineV12 = BabelSquatIntegratedEngineV12_2
BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV12_2

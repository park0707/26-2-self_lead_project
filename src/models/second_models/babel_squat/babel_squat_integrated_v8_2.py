# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v8_2.py
Purpose: 2단계 계층적 확률 보정 (Level 1 서브모델 로짓 온도 보정 + Level 2 메타 소프트맥스 온도 보정)
         및 16-Way Joint Head 융합 기반 차세대 통합 진단 엔진 v8.2
         (Layer 1: 4대 부위별 3-State 진단 + Layer 2: 2단계 온도 보정 16-Way 메타 신경망 + Layer 3: 시간적 일관성 & 이상치 필터링)

[공식 5대 카메라 시점 명명 규격 (Official 5-View Taxonomy)]
- 시점 A: 우측 후방 (45도) [Right Back 45° / view5]
- 시점 B: 우측 전방 (45도) [Right Front 45° / view2]
- 시점 C: 정면 (0도) [Front View 0° / view3]
- 시점 D: 좌측 전방 (45도) [Left Front 45° / view4]
- 시점 E: 좌측 후방 (45도) [Left Back 45° / view1]

핵심 설계 철학:
1. 베이스라인(v8.1) 코드 및 4대 서브모델 인터페이스 100% 온전 계승:
   - c1: 척추 중립 위반 (v11.1 고신뢰 정상 분산 면제)
   - c2: 고개 정면 및 시선 (v7.1 사선 시점 원근 왜곡 보정)
   - c3: 발과 무릎 방향 일치 (v6.0 20D 관상면 무릎-발목 비율)
   - c4: 발바닥 지면 고정 (v5.1 지면 기준선 중앙값 추정 및 롤 특이점 보정)
2. Layer 2 [해결책 2 혁신]: 2단계 계층적 확률 보정 (Hierarchical 2-Level Temperature Scaling):
   - [Level 1 보정]: 4대 서브모델 로짓에 대해 라벨별 및 5대 시점별 최적 온도(T_k, T_2,v)를 적용하여 ECE 오차를 1.6% 이하로 정렬
   - [Level 2 보정]: 16-Way 메타 소프트맥스에 메타 온도(T_meta = 1.20)를 적용하여 클래스 과신을 완화하고 전역 사후확률 극대화
3. Layer 3 [시간적 일관성 & 이상치 필터링]:
   - 단발성 튐 정상 표본 구제 및 시계열 고저크 결함 오승인 차단
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

from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c2_model_v7_1 import BabelSquatC2ModelV7_1
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1
from src.models.second_models.babel_squat.joint_meta_16way_net import (
    JointMeta16WayNet,
    decode_hybrid,
    idx_to_defect_tuple,
    defect_tuple_to_idx
)

# 5대 공인 시점 매핑 딕셔너리
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
        'normal_feedback': '발바닥 전체(뒤꿈치 및 앞꿈치)가 지면에 단단히 고정되었습니다.',
        'defect_feedback': '동작 중 발뒤꿈치 또는 앞꿈치가 바닥에서 들려 체중 지지가 불안정했습니다.',
        'defect_action': '체중을 발뒤꿈치와 발바닥 삼각점(엄지·새끼발가락·뒤꿈치)에 고르게싣고 지면을 누르세요.',
        'abstain_feedback': '바닥 그림자 또는 각도로 인해 발바닥 접지 경계의 판정 신뢰도가 부족하여 판정을 보류합니다.'
    }
}

COMPONENT_ORDER = ['c1', 'c2', 'c3', 'c4']


class BabelSquatIntegratedEngineV8_2:
    """
    바벨 스쿼트 2단계 계층적 확률 보정(Temperature Scaling) & 16-Way 메타 통합 진단 엔진 v8.2
    """

    VERSION = "v8.2"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        T_meta: float = 1.20,
        eta: float = 0.10,
        tau_clean: float = 0.38,
        jerk_threshold: float = 220.0,
        wobble_threshold: float = 0.008,
        f0_rate_threshold: float = 0.35
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.T_meta = T_meta
        self.eta = eta
        self.tau_clean = tau_clean
        self.jerk_threshold = jerk_threshold
        self.wobble_threshold = wobble_threshold
        self.f0_rate_threshold = f0_rate_threshold

        self.w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        self.w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        self.w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        self.w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')
        self.w_joint = os.path.join(self.weights_dir, 'joint_meta_16way_v8_2_weights.pth')
        self.w_temp_json = os.path.join(self.weights_dir, 'temperature_params.json')
        self.w_meta_json = os.path.join(self.weights_dir, 'joint_meta_16way_v8_2_meta.json')

        # Level 1 온도 파라미터 로드
        self.T_c1 = 0.456
        self.T_c3 = 1.112
        self.T_c4 = 1.001
        self.view_T_c2 = {'view1': 0.903, 'view2': 0.898, 'view3': 0.963, 'view4': 0.890, 'view5': 0.893}

        if os.path.exists(self.w_temp_json):
            try:
                with open(self.w_temp_json, 'r', encoding='utf-8') as f:
                    t_cfg = json.load(f)
                    self.T_c1 = t_cfg.get('c1', {}).get('global_T', self.T_c1)
                    self.T_c3 = t_cfg.get('c3', {}).get('global_T', self.T_c3)
                    self.T_c4 = t_cfg.get('c4', {}).get('global_T', self.T_c4)
                    self.view_T_c2 = t_cfg.get('c2', {}).get('view_T', self.view_T_c2)
            except Exception:
                pass

        # Level 2 메타 파라미터 로드
        if os.path.exists(self.w_meta_json):
            try:
                with open(self.w_meta_json, 'r', encoding='utf-8') as f:
                    m_cfg = json.load(f)
                    self.T_meta = m_cfg.get('T_meta', self.T_meta)
                    self.eta = m_cfg.get('eta', self.eta)
                    self.tau_clean = m_cfg.get('tau_clean', self.tau_clean)
            except Exception:
                pass

        # 4대 챔피언 서브모델 초기화 (c1 v11.1, c2 v7.1, c3 v6.0, c4 v5.1)
        self.model_c1 = BabelSquatC1ModelV11_1(weights_path=self.w_c1)
        self.model_c2 = BabelSquatC2ModelV7_1(weights_path=self.w_c2)
        self.model_c3 = BabelSquatC3ModelV6(weights_path=self.w_c3)
        self.model_c4 = BabelSquatC4ModelV5_1(weights_path=self.w_c4)

        # 16-Way Joint 메타 모델 로드
        self.joint_model = JointMeta16WayNet(in_features=18, hidden_dim=64, dropout_rate=0.20)
        if os.path.exists(self.w_joint):
            state_dict = torch.load(self.w_joint, map_location=torch.device('cpu'), weights_only=True)
            self.joint_model.load_state_dict(state_dict)
        self.joint_model.eval()

        self.is_loaded = True

    def normalize_view(self, camera_view: Optional[str]) -> str:
        """사용자 입력 시점 명칭을 5대 표준 코드(view1~view5)로 정규화"""
        if not camera_view:
            return 'view3'  # 기본값: 시점 C (정면)
        view_str = str(camera_view).strip()
        return VIEW_ALIAS_MAP.get(view_str, VIEW_ALIAS_MAP.get(view_str.upper(), 'view3'))

    def diagnose_components(
        self,
        c1_input: Any,
        c2_input: Any,
        c3_input: Any,
        c4_input: Any,
        camera_view: Optional[str] = None,
        c4_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """4대 서브모델 추론 및 v8.2 2단계 온도 보정 16-Way Joint 메타 추론 종합 리포트 생성"""
        norm_view = self.normalize_view(camera_view)

        r1 = self.model_c1.predict_selective(c1_input)
        r2 = self.model_c2.predict_selective(c2_input, camera_view=norm_view)
        r3 = self.model_c3.predict_selective(c3_input)
        r4 = self.model_c4.predict_selective(c4_input, camera_view=norm_view)

        sub_results = {'c1': r1, 'c2': r2, 'c3': r3, 'c4': r4}
        return self._synthesize_diagnosis(sub_results, norm_view, c4_raw_feats=c4_raw_feats)

    def diagnose_sequence(
        self,
        sequence_64f: np.ndarray,
        camera_view: Optional[str] = None
    ) -> Dict[str, Any]:
        """(64, 33, 3) MediaPipe 시퀀스 직접 진단"""
        norm_view = self.normalize_view(camera_view)
        c4_feats = self.model_c4.extract_features(sequence_64f)
        return self.diagnose_components(
            c1_input=sequence_64f,
            c2_input=sequence_64f,
            c3_input=sequence_64f,
            c4_input=sequence_64f,
            camera_view=norm_view,
            c4_raw_feats=c4_feats
        )

    def _synthesize_diagnosis(
        self,
        sub_results: Dict[str, Dict[str, Any]],
        camera_view: str,
        c4_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        3-Layer 다차원 퓨전 (v8.2):
        - Layer 1: 부위별 3-State 진단
        - Layer 2: 2단계 계층적 확률 보정 (Level 1 로짓 온도 보정 + Level 2 메타 소프트맥스 온도 보정)
        - Layer 3: 시간적 일관성 및 시퀀스 무결성 필터링 (v8.0 온전 계승)
        """
        norm_view = self.normalize_view(camera_view)
        view_meta = VIEW_METADATA.get(norm_view, VIEW_METADATA['view3'])

        component_reports = {}
        component_states = {}
        component_codes = {}
        component_probs = {}

        state_to_code = {'NORMAL': 0, 'DEFECT': 1, 'UNDETERMINED': -1, 'NOT_EVALUATED': -2}
        is_rear = norm_view in ['view1', 'view5']    # 시점 E (view1), 시점 A (view5)
        is_oblique = norm_view in ['view2', 'view4']  # 시점 B (view2), 시점 D (view4)
        is_front = (norm_view == 'view3')            # 시점 C (view3)

        for k in COMPONENT_ORDER:
            res = sub_results[k]
            st = res.get('status', 'UNDETERMINED')
            prob = res.get('defect_probability')
            if prob is None:
                prob = 0.5
            meta = DEFECT_METADATA[k]

            raw_reason = res.get('reason_code', 'CLEAR_DECISION')
            if raw_reason in ['INSUFFICIENT_ROM', 'LOW_QUALITY']:
                std_reason = 'LOW_INPUT_QUALITY'
            elif raw_reason in ['FOLD_PREDICTION_UNSTABLE', 'ENSEMBLE_DISAGREEMENT', 'REAR_FOLD_UNSTABLE', 'REAR_DISAGREEMENT', 'PREDICTION_VARIANCE']:
                std_reason = 'PREDICTION_VARIANCE'
            elif raw_reason in ['LOW_CONFIDENCE', 'LOW_CONFIDENCE_MARGIN', 'REAR_VIEW_LOW_CONFIDENCE', 'MODEL_UNCERTAINTY']:
                std_reason = 'MODEL_UNCERTAINTY'
            elif raw_reason in ['NOT_OBSERVABLE', 'PHYSICAL_INVISIBILITY']:
                std_reason = 'PHYSICAL_INVISIBILITY'
            elif raw_reason in ['GROUND_SINGULARITY_COMPENSATED', 'TEMPORAL_SMOOTH_COMPENSATED']:
                std_reason = 'TEMPORAL_CONSISTENCY_CLEAN'
            else:
                std_reason = 'CLEAR_DECISION'

            if st == 'NORMAL':
                fb = meta['normal_feedback']
                act = None
            elif st == 'DEFECT':
                fb = meta['defect_feedback']
                act = meta['defect_action']
            else:
                fb = res.get('message') or meta['abstain_feedback']
                act = res.get('action') or '카메라 위치 및 각도를 조정한 후 다시 시도해 주세요.'

            component_states[k] = st
            component_codes[k] = state_to_code.get(st, -1)
            component_probs[k] = float(prob)
            component_reports[k] = {
                'code': k,
                'name': meta['name'],
                'status': st,
                'probability': prob,
                'feedback': fb,
                'action': act,
                'reason_code': std_reason,
                'raw_reason': raw_reason
            }

        p1 = component_probs['c1']
        p2 = component_probs['c2']
        p3 = component_probs['c3']
        p4 = component_probs['c4']

        # -------------------------------------------------------------
        # [Level 1 보정]: 4대 서브모델 로짓 온도 스케일링
        # -------------------------------------------------------------
        eps = 1e-7
        p_raw = np.clip([p1, p2, p3, p4], eps, 1.0 - eps)
        z_raw = np.log(p_raw / (1.0 - p_raw))

        T_c2_v = self.view_T_c2.get(norm_view, 0.90)

        p1_cal = float(1.0 / (1.0 + np.exp(-z_raw[0] / self.T_c1)))
        p2_cal = float(1.0 / (1.0 + np.exp(-z_raw[1] / T_c2_v)))
        p3_cal = float(1.0 / (1.0 + np.exp(-z_raw[2] / self.T_c3)))
        p4_cal = float(1.0 / (1.0 + np.exp(-z_raw[3] / self.T_c4)))

        # c4 물리 동역학 피처 추출
        toe_lift_norm = 0.0
        wobble = 0.005
        ydelta = 0.0
        jerk = 120.0
        f0_rate = 0.10

        if c4_raw_feats is not None:
            c4_f = np.asarray(c4_raw_feats).ravel()
            if len(c4_f) >= 25:
                toe_lift_norm = float(c4_f[1])
                wobble = float(c4_f[3])
                ydelta = float(c4_f[6])
                jerk = float(c4_f[15])
                f0_rate = float(c4_f[24])

        # -------------------------------------------------------------
        # Layer 2 [v8.2]: 18D 16-Way Joint Meta Head + Level 2 메타 온도 보정
        # -------------------------------------------------------------
        view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
        v_idx = view_map.get(norm_view, 2)
        v_onehot = np.zeros(5, dtype=np.float32)
        v_onehot[v_idx] = 1.0

        p_cal_vec = np.array([p1_cal, p2_cal, p3_cal, p4_cal], dtype=np.float32)
        margin_cal_vec = np.abs(p_cal_vec - 0.5)

        jerk_n = float(np.clip(jerk / 200.0, 0.0, 2.5))
        wobble_n = float(np.clip(wobble / 0.015, 0.0, 2.5))
        ydelta_n = float(np.clip(ydelta / 0.020, -1.0, 2.5))
        toe_n = float(np.clip(toe_lift_norm / 0.020, 0.0, 2.5))
        f0_n = float(np.clip(f0_rate / 0.50, 0.0, 2.0))
        phys_vec = np.array([toe_n, wobble_n, ydelta_n, jerk_n, f0_n], dtype=np.float32)

        x_18d = np.concatenate([p_cal_vec, margin_cal_vec, v_onehot, phys_vec])[None, :]
        x_tensor = torch.from_numpy(x_18d).float()

        with torch.no_grad():
            l16, l4 = self.joint_model(x_tensor)
            # Level 2 Meta Temperature Scaling
            p16_pred = torch.softmax(l16 / self.T_meta, dim=-1)
            p4_pred = torch.sigmoid(l4)

        p16_arr = p16_pred[0].cpu().numpy()
        p4_arr = p4_pred[0].cpu().numpy()

        best_state, defects_tuple, joint_is_clean, state_conf = decode_hybrid(
            p16_arr, p4_arr, eta=self.eta, tau_clean=self.tau_clean
        )

        is_verified_clean = False
        is_observable_clean = False
        clean_tier = 'NONE'

        if joint_is_clean:
            session_status = 'CLEAN'
            if is_rear:
                clean_tier = 'OBSERVABLE'
                is_observable_clean = True
                component_states['c2'] = 'NOT_EVALUATED'
                component_codes['c2'] = -2
                component_reports['c2']['status'] = 'NOT_EVALUATED'
                component_reports['c2']['reason_code'] = 'PHYSICAL_INVISIBILITY'
                component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
            else:
                clean_tier = 'VERIFIED'
                is_verified_clean = True
        elif is_rear and (p1_cal <= 0.40) and (p3_cal <= 0.40) and (p4_cal <= 0.40) and (p2_cal < 0.70) and ((1.0 - p1_cal) * (1.0 - p3_cal) * (1.0 - p4_cal) >= 0.55) and (p1_cal < 0.50 and p3_cal < 0.50 and p4_cal < 0.50):
            session_status = 'CLEAN'
            clean_tier = 'OBSERVABLE'
            is_observable_clean = True
            component_states['c2'] = 'NOT_EVALUATED'
            component_codes['c2'] = -2
            component_reports['c2']['status'] = 'NOT_EVALUATED'
            component_reports['c2']['reason_code'] = 'PHYSICAL_INVISIBILITY'
            component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
        else:
            session_status = 'DEFECT'
            # 16-way 판정에 따른 4개 결함 상태 동기화
            c_keys = ['c1', 'c2', 'c3', 'c4']
            for idx_k, k_name in enumerate(c_keys):
                if defects_tuple[idx_k] == 1:
                    component_states[k_name] = 'DEFECT'
                    component_codes[k_name] = 1
                    component_reports[k_name]['status'] = 'DEFECT'
                    component_reports[k_name]['feedback'] = DEFECT_METADATA[k_name]['defect_feedback']
                    component_reports[k_name]['action'] = DEFECT_METADATA[k_name]['defect_action']
                else:
                    if is_rear and k_name == 'c2':
                        component_states['c2'] = 'NOT_EVALUATED'
                        component_codes['c2'] = -2
                        component_reports['c2']['status'] = 'NOT_EVALUATED'
                        component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
                    else:
                        component_states[k_name] = 'NORMAL'
                        component_codes[k_name] = 0
                        component_reports[k_name]['status'] = 'NORMAL'
                        component_reports[k_name]['feedback'] = DEFECT_METADATA[k_name]['normal_feedback']

        # -------------------------------------------------------------
        # Layer 3 [시간적 일관성 & 이상치 필터링: v8.0 온전 계승]
        # -------------------------------------------------------------
        # 1) 시간적 안정성 기반 단발성 지터 정상 표본 구제 (Recovery)
        if session_status != 'CLEAN':
            max_p_cal = max(p1_cal, p2_cal, p3_cal, p4_cal)
            is_smooth = (wobble <= self.wobble_threshold) and (jerk <= 180.0)
            if max_p_cal <= 0.50 and is_smooth:
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED' if not is_rear else 'OBSERVABLE'
                if is_rear:
                    is_observable_clean = True
                    component_states['c2'] = 'NOT_EVALUATED'
                    component_codes['c2'] = -2
                    component_reports['c2']['status'] = 'NOT_EVALUATED'
                else:
                    is_verified_clean = True

        # 2) 고저크/고불안정 결함 오승인 차단 (Temporal Anomaly Rejection)
        if session_status == 'CLEAN':
            is_anomaly = (jerk > self.jerk_threshold and wobble > self.wobble_threshold) or (f0_rate > self.f0_rate_threshold)
            if is_anomaly:
                session_status = 'DEFECT'
                is_verified_clean = False
                is_observable_clean = False
                clean_tier = 'NONE'

        is_clean = (session_status == 'CLEAN')
        has_defect = (session_status == 'DEFECT')
        is_retake = (session_status == 'RETAKE')

        k_undetermined = sum(1 for s in component_states.values() if s in ['UNDETERMINED', 'NOT_EVALUATED'])
        n_defects = sum(1 for s in component_states.values() if s == 'DEFECT')
        n_normals = sum(1 for s in component_states.values() if s == 'NORMAL')

        coaching_sentences = []
        view_display = f"{view_meta['code']}: {view_meta['name']}"
        if is_observable_clean:
            coaching_sentences.append(f"[종합 판정 (관측 가능 정상) | {view_display}]: 후방 촬영으로 인해 시선(고개)은 평가에서 제외되었으나, 관측 가능한 척추·무릎·발 정렬은 모두 안정적인 정상입니다!")
        elif is_verified_clean:
            coaching_sentences.append(f"[종합 판정 (전수 검증 정상) | {view_display}]: 척추, 시선, 무릎, 발의 모든 정렬이 완벽하게 일치하는 이상적인 무결점 정상 스쿼트입니다!")

        for k in COMPONENT_ORDER:
            c = component_reports[k]
            if c['status'] == 'NORMAL':
                coaching_sentences.append(f"[{c['name']}]: {c['feedback']}")
            elif c['status'] == 'DEFECT':
                coaching_sentences.append(f"[{c['name']} (주의)]: {c['feedback']} -> {c['action']}")
            elif c['status'] == 'NOT_EVALUATED':
                coaching_sentences.append(f"[{c['name']} (평가 제외)]: {c['feedback']}")
            else:
                coaching_sentences.append(f"[{c['name']} (판단 보류)]: {c['feedback']}")

        summary_text = "\n".join(coaching_sentences)

        return {
            'version': self.VERSION,
            'camera_view': norm_view,
            'view_metadata': view_meta,
            'calibrated_probs': {
                'c1': round(p1_cal, 4),
                'c2': round(p2_cal, 4),
                'c3': round(p3_cal, 4),
                'c4': round(p4_cal, 4)
            },
            'joint_meta': {
                'predicted_state': int(best_state),
                'state_confidence': round(float(state_conf), 4),
                'p_clean_16way': round(float(p16_arr[0]), 4),
                'defects_tuple': defects_tuple
            },
            'session_summary': {
                'status': session_status,
                'is_clean': is_clean,
                'clean_tier': clean_tier,
                'is_verified_clean': is_verified_clean,
                'is_observable_clean': is_observable_clean,
                'is_retake': is_retake,
                'has_defect': has_defect,
                'soft_risk_score': round(float(1.0 - p16_arr[0]), 4),
                'n_normals': n_normals,
                'n_defects': n_defects,
                'n_undetermined': k_undetermined,
                'summary_text': summary_text
            },
            'component_reports': component_reports,
            'component_probs': component_probs,
            'component_codes': component_codes
        }


BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV8_2

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v8.py
Purpose: 바벨 스쿼트 세션 레벨 시간적 일관성 및 시퀀스 무결성 필터링 기반 최종 마스터 통합 진단 엔진 v8.0
         (Layer 1: 4대 부위별 3-State 진단 + Layer 2: 5대 공인 시점(A~E) 동적 퓨전 + Layer 3: 시간적 일관성 & 이상치 필터링)

[공식 5대 카메라 시점 명명 규격 (Official 5-View Taxonomy)]
- 시점 A: 우측 후방 (45도) [Right Back 45° / view5]
- 시점 B: 우측 전방 (45도) [Right Front 45° / view2]
- 시점 C: 정면 (0도) [Front View 0° / view3]
- 시점 D: 좌측 전방 (45도) [Left Front 45° / view4]
- 시점 E: 좌측 후방 (45도) [Left Back 45° / view1]

핵심 설계 철학:
1. 베이스라인(v7.5) 코드 및 4대 서브모델 인터페이스 100% 온전 계승:
   - c1: 척추 중립 위반 (v11.1 고신뢰 정상 분산 면제)
   - c2: 고개 정면 및 시선 (v7.1 사선 시점 원근 왜곡 보정)
   - c3: 발과 무릎 방향 일치 (v6.0 20D 관상면 무릎-발목 비율)
   - c4: 발바닥 지면 고정 (v5.1 지면 기준선 중앙값 추정 및 롤 특이점 보정)
2. Layer 3 [해결책 7 최종 혁신]: 세션 레벨 시간적 일관성 및 이상치 필터링 (Temporal Consistency Filtering):
   - 1) 단발성 튐 정상 표본 구제 (Spurious Peak Rejection):
     전체 시계열 흔들림(wobble <= 0.008)과 지면 저크(jerk <= 180.0)가 극도로 안정적인 스무드 표본 중 단발성 지터로
     경계선(p <= 0.50)에 걸려 탈락하던 골든 정상 표본을 시간적 일관성에 기반해 안전하게 추가 구제.
   - 2) 시계열 고저크 결함 오승인 차단 (Temporal Anomaly Rejection):
     지면 가속도 저크(jerk > 220.0 & wobble > 0.008) 또는 상태 전이 이상(f0_rate > 0.35) 표본을 격리 차단.
3. 3-Tier 세션 정상 판정 체계 및 5대 가드레일 엄격 유지:
   - VERIFIED_CLEAN, OBSERVABLE_CLEAN, DEFECT_DETECTED, RETAKE_RECOMMENDED
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

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
        'defect_action': '체중을 발뒤꿈치와 발바닥 삼각점(엄지·새끼발가락·뒤꿈치)에 고르게 싣고 지면을 누르세요.',
        'abstain_feedback': '바닥 그림자 또는 각도로 인해 발바닥 접지 경계의 판정 신뢰도가 부족하여 판정을 보류합니다.'
    }
}

COMPONENT_ORDER = ['c1', 'c2', 'c3', 'c4']


class BabelSquatIntegratedEngineV8:
    """
    바벨 스쿼트 5대 공인 시점(A~E) 동적 퓨전 및 시간적 일관성 필터링 최종 마스터 엔진 v8.0
    """

    VERSION = "v8.0"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_clean_4p: float = 0.38,
        tau_clean_3p: float = 0.55,
        tau_safe_normal: float = 0.40,
        tau_defect: float = 0.60,
        tau_oblique_c2_max: float = 0.70,
        tau_rear_c2_max: float = 0.70,
        jerk_threshold: float = 220.0,
        wobble_threshold: float = 0.008,
        f0_rate_threshold: float = 0.35
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.tau_clean_4p = tau_clean_4p
        self.tau_clean_3p = tau_clean_3p
        self.tau_safe_normal = tau_safe_normal
        self.tau_defect = tau_defect
        self.tau_oblique_c2_max = tau_oblique_c2_max
        self.tau_rear_c2_max = tau_rear_c2_max
        self.jerk_threshold = jerk_threshold
        self.wobble_threshold = wobble_threshold
        self.f0_rate_threshold = f0_rate_threshold

        self.w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        self.w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        self.w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        self.w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')

        # 4대 챔피언 서브모델 초기화 (c1 v11.1, c2 v7.1, c3 v6.0, c4 v5.1)
        self.model_c1 = BabelSquatC1ModelV11_1(weights_path=self.w_c1)
        self.model_c2 = BabelSquatC2ModelV7_1(weights_path=self.w_c2)
        self.model_c3 = BabelSquatC3ModelV6(weights_path=self.w_c3)
        self.model_c4 = BabelSquatC4ModelV5_1(weights_path=self.w_c4)

        self.is_loaded = True

    def normalize_view(self, camera_view: Optional[str]) -> str:
        """사용자 입력 시점 명칭을 5대 표준 코드(view1~view5)로 정규화"""
        if not camera_view:
            return 'view3' # 기본값: 시점 C (정면)
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
        """4대 서브모델 추론 및 v8.0 시간적 일관성 필터링 세션 종합 리포트 생성"""
        norm_view = self.normalize_view(camera_view)

        r1 = self.model_c1.predict_selective(c1_input)
        r2 = self.model_c2.predict_selective(c2_input, camera_view=norm_view)
        r3 = self.model_c3.predict_selective(c3_input, camera_view=norm_view)
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
        3-Layer 다차원 퓨전:
        - Layer 1: 부위별 3-State 진단
        - Layer 2: 5대 시점(A~E) 동적 임계치 매트릭스
        - Layer 3: 시간적 일관성 및 시퀀스 무결성 필터링 (v8.0)
        """
        norm_view = self.normalize_view(camera_view)
        view_meta = VIEW_METADATA.get(norm_view, VIEW_METADATA['view3'])

        component_reports = {}
        component_states = {}
        component_codes = {}
        component_probs = {}

        state_to_code = {'NORMAL': 0, 'DEFECT': 1, 'UNDETERMINED': -1, 'NOT_EVALUATED': -2}
        is_rear = norm_view in ['view1', 'view5']   # 시점 E (view1), 시점 A (view5)
        is_oblique = norm_view in ['view2', 'view4'] # 시점 B (view2), 시점 D (view4)
        is_front = (norm_view == 'view3')           # 시점 C (view3)

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

        p_clean_4p = float((1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4))
        p_clean_3p = float((1.0 - p1) * (1.0 - p3) * (1.0 - p4))
        has_clear_defect = any(p >= self.tau_defect for p in [p1, p2, p3, p4])

        # -------------------------------------------------------------
        # Layer 2: 5대 공인 시점(A~E) 동적 퓨전 판정
        # -------------------------------------------------------------
        is_verified_clean = False
        is_observable_clean = False
        clean_tier = 'NONE'

        # c4 시간적 동역학 피처 추출
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

        foot_ground_ok = (ydelta <= 0.0) and (toe_lift_norm <= 0.010) and (wobble <= 0.008) and (p4 <= 0.70)
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean_for_ground = (p2 < self.tau_rear_c2_max) if is_rear else (p2 <= 0.35)
        ground_compensation_pass = upper_body_clean and gaze_clean_for_ground and foot_ground_ok

        if is_rear:
            # [시점 A, 시점 E]: 후방 동적 가드레일 (p2 < 0.70 & p_cln_3 >= 0.55)
            cond_3p_safe = (p1 <= self.tau_safe_normal) and (p3 <= self.tau_safe_normal) and (p4 <= self.tau_safe_normal)
            if cond_3p_safe and (p2 < self.tau_rear_c2_max) and (p_clean_3p >= self.tau_clean_3p) and (p1 < 0.50 and p3 < 0.50 and p4 < 0.50):
                session_status = 'CLEAN'
                clean_tier = 'OBSERVABLE'
                is_observable_clean = True
                component_states['c2'] = 'NOT_EVALUATED'
                component_codes['c2'] = -2
                component_reports['c2']['status'] = 'NOT_EVALUATED'
                component_reports['c2']['reason_code'] = 'PHYSICAL_INVISIBILITY'
                component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
            elif ground_compensation_pass:
                session_status = 'CLEAN'
                clean_tier = 'OBSERVABLE'
                is_observable_clean = True
                component_states['c2'] = 'NOT_EVALUATED'
                component_codes['c2'] = -2
                component_reports['c2']['status'] = 'NOT_EVALUATED'
                component_reports['c2']['reason_code'] = 'PHYSICAL_INVISIBILITY'
                component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
                component_states['c4'] = 'NORMAL'
                component_reports['c4']['status'] = 'NORMAL'
                component_reports['c4']['reason_code'] = 'GROUND_SINGULARITY_COMPENSATED'
            elif has_clear_defect:
                session_status = 'DEFECT'
            else:
                session_status = 'RETAKE'

        elif is_oblique:
            # [시점 B, 시점 D]: 사선 동적 적응형 퓨전
            cond_3p_very_safe = (p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25)
            if cond_3p_very_safe and (p2 < self.tau_oblique_c2_max):
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif ground_compensation_pass:
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
                component_states['c4'] = 'NORMAL'
                component_reports['c4']['status'] = 'NORMAL'
                component_reports['c4']['reason_code'] = 'GROUND_SINGULARITY_COMPENSATED'
            elif (p_clean_4p >= 0.40) and all(p < 0.50 for p in [p1, p2, p3, p4]):
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif has_clear_defect:
                session_status = 'DEFECT'
            else:
                session_status = 'RETAKE'

        else:
            # [시점 C]: 정면 전수 검증 미세 최적화
            if (p_clean_4p >= self.tau_clean_4p) and all(p < 0.50 for p in [p1, p2, p3, p4]):
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif ground_compensation_pass:
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
                component_states['c4'] = 'NORMAL'
                component_reports['c4']['status'] = 'NORMAL'
                component_reports['c4']['reason_code'] = 'GROUND_SINGULARITY_COMPENSATED'
            elif has_clear_defect:
                session_status = 'DEFECT'
            else:
                session_status = 'RETAKE'

        # -------------------------------------------------------------
        # Layer 3 [v8.0 신규]: 시간적 일관성 및 시퀀스 무결성 필터링
        # -------------------------------------------------------------
        # 1) 시간적 안정성 기반 단발성 지터 정상 표본 구제 (Recovery)
        if session_status != 'CLEAN':
            max_p = max(p1, p2, p3, p4)
            n_above_40 = sum(p >= 0.40 for p in [p1, p2, p3, p4])
            is_smooth = (wobble <= self.wobble_threshold) and (jerk <= 180.0)
            if max_p <= 0.50 and n_above_40 <= 1 and is_smooth:
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
        m_valid = 4 - k_undetermined
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
            'session_summary': {
                'status': session_status,
                'is_clean': is_clean,
                'clean_tier': clean_tier,
                'is_verified_clean': is_verified_clean,
                'is_observable_clean': is_observable_clean,
                'is_retake': is_retake,
                'has_defect': has_defect,
                'soft_risk_score': round(float(p_clean_4p), 4),
                'n_normals': n_normals,
                'n_defects': n_defects,
                'n_undetermined': k_undetermined,
                'summary_text': summary_text
            },
            'component_reports': component_reports,
            'component_probs': component_probs,
            'component_codes': component_codes
        }


BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV8

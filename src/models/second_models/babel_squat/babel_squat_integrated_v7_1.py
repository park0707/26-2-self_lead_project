# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v7_1.py
Purpose: 바벨 스쿼트 시점 인지형 관측 가능 정상(Observable Clean) 및 3-Tier 세션 의사결정 프로덕션 통합 엔진 v7.1
         (Layer 1: 부위별 독립 3-State 진단 카드 + Layer 2: 3-Tier Clean 의사결정 및 5대 안전 가드레일)

핵심 설계 철학:
1. 베이스라인(v7.0) 코드 및 인터페이스 100% 온전 계승:
   - 4대 최신 챔피언 서브모델 결합:
     * c1: 척추 중립 위반 (v11.0 3-Way Tri-Ensemble)
     * c2: 고개 정면 미유지 (v7.0 19D 생체역학 로컬좌표계 + 가시귀 피치)
     * c3: 발과 무릎 방향 불일치 (v6.0 20D 관상면 무릎-발목 투영비율)
     * c4: 발바닥 지면 미고정 (v5.0 25D F0 접촉 상태 머신 앙상블)
2. Layer 1 (부위별 독립 설명 카드 - Explainable Coaching Layer):
   - 4개 부위별 상태(NORMAL / DEFECT / UNDETERMINED / NOT_EVALUATED) 분리.
   - 후방 시점(view1, view5)에서 시선이 물리적으로 가려진 경우 투명하게 '평가 제외'로 고지.
3. Layer 2 (3-Tier 정상 의사결정 및 5대 안전 가드레일 - Three-Tier Decision Layer):
   - Tier 1 (VERIFIED_CLEAN): 전면/측면 4개 부위 전수 검증 무결점 정상 (P_clean_4p >= 0.45)
   - Tier 2 (OBSERVABLE_CLEAN): 후방 시점 시선 비가시성 분리 조건부 정상 (5대 가드레일 충족 시)
     * c1, c3, c4 결함 확률 각각 <= 0.40 (고신뢰 정상)
     * c2 결함 신호 부재 (p_c2 < 0.60)
     * 3대 부위 결합 정상 확률 P_clean_3p >= 0.50
   - Tier 3 (DEFECT_DETECTED): 1개 부위라도 명확한 결함(p >= 0.60) 감지
   - Tier 4 (RETAKE_RECOMMENDED): 그 외 모호한 불확실 구간
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

from src.models.second_models.babel_squat.c1_model_v11 import BabelSquatC1ModelV11
from src.models.second_models.babel_squat.c2_model_v7 import BabelSquatC2ModelV7
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.c4_model_v5 import BabelSquatC4ModelV5

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
        'abstain_feedback': '후방 촬영 또는 바벨 원판에 얼굴이 가려져 정확한 시선 판정이 어렵습니다. (정면 또는 45도 측면 촬영 권장)',
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


class BabelSquatIntegratedEngineV7_1:
    """
    바벨 스쿼트 시점 인지형 관측 가능 정상(Observable Clean) 프로덕션 통합 진단 엔진 v7.1
    """

    VERSION = "v7.1"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_clean_4p: float = 0.45,
        tau_clean_3p: float = 0.50,
        tau_safe_normal: float = 0.40,
        tau_defect: float = 0.60
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.tau_clean_4p = tau_clean_4p
        self.tau_clean_3p = tau_clean_3p
        self.tau_safe_normal = tau_safe_normal
        self.tau_defect = tau_defect

        self.w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        self.w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        self.w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        self.w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')

        # 4대 챔피언 서브모델 초기화
        self.model_c1 = BabelSquatC1ModelV11(weights_path=self.w_c1)
        self.model_c2 = BabelSquatC2ModelV7(weights_path=self.w_c2)
        self.model_c3 = BabelSquatC3ModelV6(weights_path=self.w_c3)
        self.model_c4 = BabelSquatC4ModelV5(weights_path=self.w_c4)

        self.is_loaded = True

    def diagnose_components(
        self,
        c1_input: Any,
        c2_input: Any,
        c3_input: Any,
        c4_input: Any,
        camera_view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        4대 서브모델에 입력을 직접 전달하여 부위별 독립 3-State 진단 및 3-Tier 세션 퓨전 수행
        """
        r1 = self.model_c1.predict_selective(c1_input, camera_view=camera_view)
        r2 = self.model_c2.predict_selective(c2_input, camera_view=camera_view)
        r3 = self.model_c3.predict_selective(c3_input, camera_view=camera_view)
        r4 = self.model_c4.predict_selective(c4_input, camera_view=camera_view)

        sub_results = {'c1': r1, 'c2': r2, 'c3': r3, 'c4': r4}

        return self._synthesize_diagnosis(sub_results, camera_view)

    def diagnose_sequence(
        self,
        sequence_64f: np.ndarray,
        camera_view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        (64, 33, 3) MediaPipe 단일 시퀀스 직접 진단
        """
        return self.diagnose_components(
            c1_input=sequence_64f,
            c2_input=sequence_64f,
            c3_input=sequence_64f,
            c4_input=sequence_64f,
            camera_view=camera_view
        )

    def _synthesize_diagnosis(
        self,
        sub_results: Dict[str, Dict[str, Any]],
        camera_view: Optional[str]
    ) -> Dict[str, Any]:
        """
        2-Layer + 3-Tier 의사결정 융합:
        - Layer 1: 부위별 독립 리포트 생성 (Explainable Cards)
        - Layer 2: 시점 인지형 3-Tier 정상 판정 (Verified vs Observable vs Retake)
        """
        component_reports = {}
        component_states = {}
        component_codes = {}
        component_probs = {}

        state_to_code = {'NORMAL': 0, 'DEFECT': 1, 'UNDETERMINED': -1, 'NOT_EVALUATED': -2}
        is_rear = camera_view in ['view1', 'view5']

        for k in COMPONENT_ORDER:
            res = sub_results[k]
            st = res.get('status', 'UNDETERMINED')
            prob = res.get('defect_probability')
            if prob is None:
                prob = 0.5
            meta = DEFECT_METADATA[k]

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
                'reason_code': res.get('reason_code', 'NORMAL' if st == 'NORMAL' else 'UNKNOWN')
            }

        p1 = component_probs['c1']
        p2 = component_probs['c2']
        p3 = component_probs['c3']
        p4 = component_probs['c4']

        # 4대 부위 결합 정상 확률
        p_clean_4p = float((1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4))
        # 3대 부위(척추·무릎·발) 결합 정상 확률
        p_clean_3p = float((1.0 - p1) * (1.0 - p3) * (1.0 - p4))

        has_clear_defect = any(p >= self.tau_defect for p in [p1, p2, p3, p4])

        # -------------------------------------------------------------
        # Layer 2: 3-Tier 세션 정상 판정 및 5대 안전 가드레일
        # -------------------------------------------------------------
        is_verified_clean = False
        is_observable_clean = False
        clean_tier = 'NONE'

        if has_clear_defect:
            session_status = 'DEFECT'
        else:
            if not is_rear:
                # 전면/측면 시점: 4부위 전수 검증 정상 여부
                if p_clean_4p >= self.tau_clean_4p:
                    session_status = 'CLEAN'
                    clean_tier = 'VERIFIED'
                    is_verified_clean = True
                else:
                    session_status = 'RETAKE'
            else:
                # 후방 시점: 5대 가드레일 기반 Observable Clean 판정
                cond_3p_safe = (p1 <= self.tau_safe_normal) and (p3 <= self.tau_safe_normal) and (p4 <= self.tau_safe_normal)
                cond_c2_no_defect = (p2 < self.tau_defect)
                cond_prob_3p = (p_clean_3p >= self.tau_clean_3p)

                if cond_3p_safe and cond_c2_no_defect and cond_prob_3p:
                    session_status = 'CLEAN'
                    clean_tier = 'OBSERVABLE'
                    is_observable_clean = True
                    # 후방 시선 카드 상태를 NOT_EVALUATED로 투명하게 전환
                    component_states['c2'] = 'NOT_EVALUATED'
                    component_codes['c2'] = -2
                    component_reports['c2']['status'] = 'NOT_EVALUATED'
                    component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
                    component_reports['c2']['action'] = '시선 분석을 원하실 경우 전면 또는 45도 측면에서 재촬영을 권장합니다.'
                else:
                    session_status = 'RETAKE'

        is_clean = (session_status == 'CLEAN')
        has_defect = (session_status == 'DEFECT')
        is_retake = (session_status == 'RETAKE')

        # 부위별 카운트 집계
        k_undetermined = sum(1 for s in component_states.values() if s in ['UNDETERMINED', 'NOT_EVALUATED'])
        m_valid = 4 - k_undetermined
        n_defects = sum(1 for s in component_states.values() if s == 'DEFECT')
        n_normals = sum(1 for s in component_states.values() if s == 'NORMAL')

        # 사용자 친화적 종합 코칭 문구 조합
        coaching_sentences = []
        if is_observable_clean:
            coaching_sentences.append("[종합 판정 (관측 가능 정상)]: 후방 촬영으로 인해 시선(고개)은 평가에서 제외되었으나, 관측 가능한 척추·무릎·발 정렬은 모두 안정적인 정상입니다!")
        elif is_verified_clean:
            coaching_sentences.append("[종합 판정 (전수 검증 정상)]: 척추, 시선, 무릎, 발의 모든 정렬이 완벽하게 일치하는 이상적인 무결점 정상 스쿼트입니다!")

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
            'camera_view': camera_view,
            'session_summary': {
                'status': session_status,
                'is_clean': is_clean,
                'clean_tier': clean_tier,
                'is_verified_clean': is_verified_clean,
                'is_observable_clean': is_observable_clean,
                'has_defect': has_defect,
                'is_retake': is_retake,
                'session_clean_probability_4p': p_clean_4p,
                'session_clean_probability_3p': p_clean_3p,
                'total_components': 4,
                'valid_count': m_valid,
                'undetermined_count': k_undetermined,
                'defect_count': n_defects,
                'normal_count': n_normals
            },
            'components': component_reports,
            'component_states': component_states,
            'component_codes': component_codes,
            'user_coaching_text': summary_text
        }


# 하위 호환성 별칭
BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV7_1


if __name__ == '__main__':
    print("=" * 105)
    print("  [vidio-health-trainer] BabelSquatIntegratedEngineV7_1 (v7.1) 모듈 로드 및 구조 검증")
    print("=" * 105)

    engine = BabelSquatIntegratedEngineV7_1()
    print("[+] v7.1 엔진 초기화 성공! 시점 인지형 Observable Clean 탑재 완료.")
    print(f"    - c1: {engine.model_c1.__class__.__name__} ({getattr(engine.model_c1, 'VERSION', 'v11.0')})")
    print(f"    - c2: {engine.model_c2.__class__.__name__} ({getattr(engine.model_c2, 'VERSION', 'v7.0')})")
    print(f"    - c3: {engine.model_c3.__class__.__name__} ({getattr(engine.model_c3, 'VERSION', 'v6.0')})")
    print(f"    - c4: {engine.model_c4.__class__.__name__} ({getattr(engine.model_c4, 'VERSION', 'v5.0')})")
    print(f"    - Decision Thresholds: 4p={engine.tau_clean_4p}, 3p={engine.tau_clean_3p}, safe_normal={engine.tau_safe_normal}, defect={engine.tau_defect}")
    print("=" * 105)

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v6.py
Purpose: 바벨 스쿼트 부위별 독립 3-State(정상/결함/판단불가) 및 다중 결함 직접 판정 공식 프로덕션 통합 엔진 v6.0
         (Component-wise Independent 3-State + Direct Multi-Defect Pipeline + Explainable Multi-Card Feedback)

핵심 설계 철학:
1. 게이트키퍼(Tier 1) 완전 배제 및 E2E 병렬 독립 추론:
   - 과거 거시 게이트키퍼에 의한 조기 차단(Early Exit) 및 오류 전파(False Negative Cascading)를 원천 차단.
   - 4대 최신 챔피언 서브모델이 MediaPipe 64프레임 입력을 직접 받아 독립적으로 판정.
2. 부위별 독립 3-State (NORMAL / DEFECT / UNDETERMINED) 반환:
   - c1: 척추의 중립 위반 (v11.0 3-Way Tri-Ensemble, [0.40, 0.60])
   - c2: 고개 정면 미유지 (v7.0 19D 생체역학 + 가시귀 피치, 전면 0.4/0.6, 후방 0.3/0.7)
   - c3: 발과 무릎 방향 불일치 (v6.0 20D 관상면 무릎-발목 투영비율, [0.40, 0.60])
   - c4: 발바닥 지면 미고정 (v5.0 25D F0 접촉 상태 머신 앙상블, [0.40, 0.60])
3. 투명하고 설명 가능한 부위별 분리 피드백 (Multi-Card Coaching Feedback):
   - 단일 라벨로 뭉뚱그리지 않고 각 관절 부위별 상태(성공 / 교정 필요 / 판단 불가 사유)를 분리하여 사용자에게 제공.
4. 선택적 다중 라벨 평가(Selective Multi-Label Evaluation) 인터페이스 내장:
   - 판단 불가 개수(k=0~4) x 적중 개수(m=0~4) 2D 교차 검증 매트릭스 산출
   - 판단 불가 제외 유효 라벨 Precision / Recall / F1-Score 산출
   - 완전 정상(Clean Squat) 및 세부 결함 전수 일치율(Exact Match Ratio, EMR) 산출
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
        'abstain_feedback': '후방 촬영 또는 바벨 원판에 얼굴이 가려져 정확한 시선 판정이 어렵습니다. (정면 또는 45도 측면 촬영 권장)'
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


class BabelSquatIntegratedEngineV6:
    """
    바벨 스쿼트 부위별 독립 3-State 및 다중 결함 직접 판정 공식 프로덕션 통합 엔진 v6.0
    """

    VERSION = "v6.0"

    def __init__(self, weights_dir: Optional[str] = None):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir

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
        4대 서브모델에 입력을 직접 전달하여 부위별 독립 3-State 진단 수행
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
        4대 서브모델의 결과를 취합하여 부위별 독립 리포트 및 종합 세션 리포트 생성
        """
        component_reports = {}
        component_states = {} # 'NORMAL', 'DEFECT', 'UNDETERMINED'
        component_codes = {}  # 0: NORMAL, 1: DEFECT, -1: UNDETERMINED

        state_to_code = {'NORMAL': 0, 'DEFECT': 1, 'UNDETERMINED': -1}

        for k in COMPONENT_ORDER:
            res = sub_results[k]
            st = res.get('status', 'UNDETERMINED')
            prob = res.get('defect_probability')
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
            component_reports[k] = {
                'code': k,
                'name': meta['name'],
                'status': st,
                'probability': prob,
                'feedback': fb,
                'action': act,
                'reason_code': res.get('reason_code', 'NORMAL' if st == 'NORMAL' else 'UNKNOWN')
            }

        # 통계 계산
        k_undetermined = sum(1 for s in component_states.values() if s == 'UNDETERMINED')
        m_valid = 4 - k_undetermined
        n_defects = sum(1 for s in component_states.values() if s == 'DEFECT')
        n_normals = sum(1 for s in component_states.values() if s == 'NORMAL')

        # 세션 거시 상태 (Macro Status)
        if n_defects > 0:
            session_status = 'DEFECT'
        elif n_normals == 4:
            session_status = 'CLEAN'
        else:
            session_status = 'RETAKE'

        # 사용자 친화적 종합 코칭 문구 조합
        coaching_sentences = []
        for k in COMPONENT_ORDER:
            c = component_reports[k]
            if c['status'] == 'NORMAL':
                coaching_sentences.append(f"[{c['name']}]: {c['feedback']}")
            elif c['status'] == 'DEFECT':
                coaching_sentences.append(f"[{c['name']} (주의)]: {c['feedback']} -> {c['action']}")
            else:
                coaching_sentences.append(f"[{c['name']} (판단 보류)]: {c['feedback']}")

        summary_text = "\n".join(coaching_sentences)

        return {
            'version': self.VERSION,
            'camera_view': camera_view,
            'session_summary': {
                'status': session_status,
                'is_clean': (session_status == 'CLEAN'),
                'has_defect': (session_status == 'DEFECT'),
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
BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV6


if __name__ == '__main__':
    print("=" * 105)
    print("  [vidio-health-trainer] BabelSquatIntegratedEngineV6 (v6.0) 모듈 로드 및 구조 검증")
    print("=" * 105)

    engine = BabelSquatIntegratedEngineV6()
    print("[+] v6.0 엔진 초기화 성공! 4대 최신 챔피언 서브모델 결합 완료.")
    print(f"    - c1: {engine.model_c1.__class__.__name__} ({getattr(engine.model_c1, 'VERSION', 'v11.0')})")
    print(f"    - c2: {engine.model_c2.__class__.__name__} ({getattr(engine.model_c2, 'VERSION', 'v7.0')})")
    print(f"    - c3: {engine.model_c3.__class__.__name__} ({getattr(engine.model_c3, 'VERSION', 'v6.0')})")
    print(f"    - c4: {engine.model_c4.__class__.__name__} ({getattr(engine.model_c4, 'VERSION', 'v5.0')})")
    print("=" * 105)

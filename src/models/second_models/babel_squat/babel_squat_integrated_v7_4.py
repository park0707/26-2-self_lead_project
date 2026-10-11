# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: babel_squat_integrated_v7_4.py
Purpose: 바벨 스쿼트 c4 지면 기준선 중앙값 추정 및 롤 특이점 보정(v5.1) 기반 통합 진단 엔진 v7.4
         (Layer 1: 4대 부위별 정밀 3-State 진단 카드 + Layer 2: 3-Tier 세션 다차원 적응형 퓨전 엔진)

핵심 설계 철학:
1. 베이스라인(v7.3) 코드 및 인터페이스 100% 온전 계승:
   - c1: 척추 중립 위반 (v11.1 고신뢰 정상 분산 면제 규칙)
   - c2: 고개 정면 및 시선 (v7.1 사선 시점 원근 왜곡 보정 모델)
   - c3: 발과 무릎 방향 일치 (v6.0 20D 관상면 무릎-발목 투영비율)
   - c4: 발바닥 지면 고정 (신규 v5.1 지면 기준선 중앙값 추정 및 롤 특이점 보정 모델 장착)
2. Layer 2 [해결책 5 혁신]: 지면 접촉 정밀 검증 퓨전 (Ground Contact Precision Verification Fusion):
   - 척추(c1 <= 0.35)와 무릎(c3 <= 0.20)이 완벽한 정상이고,
   - 시선(c2)이 안전 범위(후방 p2 < 0.50, 사선/정면 p2 <= 0.35)이며,
   - 발바닥 지면 접촉이 기구학적으로 확인된 표본(뒤꿈치 들림 0cm, 앞꿈치 들림 극소, 상승기 흔들림 극소, p4 <= 0.70)에 대해,
     단안 2D 키포인트 롤 아크탄젠트 특이점으로 인한 거짓 결함을 정상으로 최종 구제 승인.
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


class BabelSquatIntegratedEngineV7_4:
    """
    바벨 스쿼트 c4 지면 기준선 중앙값 추정 및 롤 특이점 보정 통합 진단 엔진 v7.4
    """

    VERSION = "v7.4"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_clean_4p: float = 0.40,
        tau_clean_3p: float = 0.50,
        tau_safe_normal: float = 0.40,
        tau_defect: float = 0.60,
        tau_oblique_c2_max: float = 0.70
    ):
        if weights_dir is None:
            weights_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = weights_dir
        self.tau_clean_4p = tau_clean_4p
        self.tau_clean_3p = tau_clean_3p
        self.tau_safe_normal = tau_safe_normal
        self.tau_defect = tau_defect
        self.tau_oblique_c2_max = tau_oblique_c2_max

        self.w_c1 = os.path.join(self.weights_dir, 'c1_model_v11_weights.pkl')
        self.w_c2 = os.path.join(self.weights_dir, 'c2_model_v7_weights.pkl')
        self.w_c3 = os.path.join(self.weights_dir, 'c3_model_v6_weights.pkl')
        self.w_c4 = os.path.join(self.weights_dir, 'c4_model_v5_weights.pkl')

        # 4대 챔피언 서브모델 초기화 (c1 v11.1, c2 v7.1, c4 v5.1 장착)
        self.model_c1 = BabelSquatC1ModelV11_1(weights_path=self.w_c1)
        self.model_c2 = BabelSquatC2ModelV7_1(weights_path=self.w_c2)
        self.model_c3 = BabelSquatC3ModelV6(weights_path=self.w_c3)
        self.model_c4 = BabelSquatC4ModelV5_1(weights_path=self.w_c4)

        self.is_loaded = True

    def diagnose_components(
        self,
        c1_input: Any,
        c2_input: Any,
        c3_input: Any,
        c4_input: Any,
        camera_view: Optional[str] = None,
        c4_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """4대 서브모델 추론 및 v7.4 지면 특이점 보정 적응형 퓨전 세션 종합 리포트 생성"""
        r1 = self.model_c1.predict_selective(c1_input)
        r2 = self.model_c2.predict_selective(c2_input, camera_view=camera_view)
        r3 = self.model_c3.predict_selective(c3_input, camera_view=camera_view)
        r4 = self.model_c4.predict_selective(c4_input, camera_view=camera_view)

        sub_results = {'c1': r1, 'c2': r2, 'c3': r3, 'c4': r4}
        return self._synthesize_diagnosis(sub_results, camera_view, c4_raw_feats=c4_raw_feats)

    def diagnose_sequence(
        self,
        sequence_64f: np.ndarray,
        camera_view: Optional[str] = None
    ) -> Dict[str, Any]:
        """(64, 33, 3) MediaPipe 시퀀스 직접 진단"""
        c4_feats = self.model_c4.extract_features(sequence_64f)
        return self.diagnose_components(
            c1_input=sequence_64f,
            c2_input=sequence_64f,
            c3_input=sequence_64f,
            c4_input=sequence_64f,
            camera_view=camera_view,
            c4_raw_feats=c4_feats
        )

    def _synthesize_diagnosis(
        self,
        sub_results: Dict[str, Dict[str, Any]],
        camera_view: Optional[str],
        c4_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        2-Layer + 3-Tier + 사선 원근 적응형 + 지면 접촉 정밀 검증 퓨전
        """
        component_reports = {}
        component_states = {}
        component_codes = {}
        component_probs = {}

        state_to_code = {'NORMAL': 0, 'DEFECT': 1, 'UNDETERMINED': -1, 'NOT_EVALUATED': -2}
        is_rear = camera_view in ['view1', 'view5']
        is_oblique = camera_view in ['view2', 'view4']

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
            elif raw_reason in ['GROUND_SINGULARITY_COMPENSATED']:
                std_reason = 'GROUND_SINGULARITY_COMPENSATED'
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
        # Layer 2: 3-Tier 세션 정상 판정 및 다차원 적응형 퓨전
        # -------------------------------------------------------------
        is_verified_clean = False
        is_observable_clean = False
        clean_tier = 'NONE'

        # [해결책 5] 지면 접촉 정밀 감사 가드레일 (Ground Contact Singularity Guardrail)
        foot_ground_ok = False
        if c4_raw_feats is not None:
            c4_f = np.asarray(c4_raw_feats).ravel()
            if len(c4_f) >= 25:
                toe_lift_norm = float(c4_f[1])
                wobble = float(c4_f[3])
                ydelta = float(c4_f[6])
                foot_ground_ok = (ydelta <= 0.0) and (toe_lift_norm <= 0.010) and (wobble <= 0.008) and (p4 <= 0.70)

        # 상체/체간의 완벽한 정상 여부
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean_for_ground = (p2 < 0.50) if is_rear else (p2 <= 0.35)
        ground_compensation_pass = upper_body_clean and gaze_clean_for_ground and foot_ground_ok

        if is_rear:
            # === 후방 뷰: 5대 가드레일 Observable Clean ===
            cond_3p_safe = (p1 <= self.tau_safe_normal) and (p3 <= self.tau_safe_normal) and (p4 <= self.tau_safe_normal)
            cond_c2_safe = (p2 < self.tau_defect)
            cond_prob_3p = (p_clean_3p >= self.tau_clean_3p)

            if cond_3p_safe and cond_c2_safe and cond_prob_3p and (not has_clear_defect):
                session_status = 'CLEAN'
                clean_tier = 'OBSERVABLE'
                is_observable_clean = True
                component_states['c2'] = 'NOT_EVALUATED'
                component_codes['c2'] = -2
                component_reports['c2']['status'] = 'NOT_EVALUATED'
                component_reports['c2']['reason_code'] = 'PHYSICAL_INVISIBILITY'
                component_reports['c2']['feedback'] = DEFECT_METADATA['c2']['not_evaluated_feedback']
            elif ground_compensation_pass:
                # [해결책 5] 후방 지면 롤 특이점 구제
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
            # === 사선 45도 뷰: 사선 투영 적응형 퓨전 + 지면 특이점 보정 ===
            cond_3p_very_safe = (p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25)
            cond_c2_oblique_ok = (p2 < self.tau_oblique_c2_max)

            if cond_3p_very_safe and cond_c2_oblique_ok:
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif ground_compensation_pass:
                # [해결책 5] 사선 뷰 지면 롤 특이점 구제
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
                component_states['c4'] = 'NORMAL'
                component_reports['c4']['status'] = 'NORMAL'
                component_reports['c4']['reason_code'] = 'GROUND_SINGULARITY_COMPENSATED'
            elif (p_clean_4p >= self.tau_clean_4p) and all(p < 0.50 for p in [p1, p2, p3, p4]):
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif has_clear_defect:
                session_status = 'DEFECT'
            else:
                session_status = 'RETAKE'

        else:
            # === 정면 뷰 (view3): 4부위 전수 검증 + 지면 특이점 보정 ===
            if (p_clean_4p >= self.tau_clean_4p) and all(p < 0.50 for p in [p1, p2, p3, p4]):
                session_status = 'CLEAN'
                clean_tier = 'VERIFIED'
                is_verified_clean = True
            elif ground_compensation_pass:
                # [해결책 5] 정면 뷰 지면 롤 특이점 구제
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

        is_clean = (session_status == 'CLEAN')
        has_defect = (session_status == 'DEFECT')
        is_retake = (session_status == 'RETAKE')

        k_undetermined = sum(1 for s in component_states.values() if s in ['UNDETERMINED', 'NOT_EVALUATED'])
        m_valid = 4 - k_undetermined
        n_defects = sum(1 for s in component_states.values() if s == 'DEFECT')
        n_normals = sum(1 for s in component_states.values() if s == 'NORMAL')

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


BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV7_4

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/model_registry.py
Purpose: 8대 운동 33종 개별 세부 결함 모델 싱글톤 레지스트리 (DefectModelRegistry)
         - 마스터 클래스 없이 개별 세부 결함 모델 직접 호출
         - 서버 기동 시 사전 워밍(Pre-warming) 지원으로 실시간 추론 지연시간 20~50ms 달성
         - 관절 토폴로지(MediaPipe 33J / RTMPose 26J) 자동 매핑 및 변환
         - 모든 모델 출력 규격: {defect_code, defect_name, probability, state, state_label, feedback: '피드백 자리'}
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
FEATURES_DIR = os.path.join(SRC_ROOT, "features")

for p in [PROJECT_ROOT, SRC_ROOT, FEATURES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.web.core.config import EXERCISE_CATALOG, VIEW_ALIAS_MAP, WEIGHTS_DIR
from src.web.core.schemas import DefectPrediction
from src.web.core.topology_adapter import SkeletalTopologyAdapter

logger = logging.getLogger("model_registry")


class DefectModelRegistry:
    """8대 운동 33종 세부 결함 모델 싱글톤 레지스트리"""

    _instance = None

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._is_warmed: Dict[int, bool] = {i: False for i in range(8)}
        self._load_all_registered = False

    @classmethod
    def get_instance(cls) -> "DefectModelRegistry":
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # 1. 운동별 서브 모델 지연 로딩 (Lazy-Loading with Cache)
    # =========================================================================

    def _get_squat_models(self) -> Dict[str, Any]:
        """01_바벨_스쿼트 4대 결함 (c1, c2: MediaPipe 33J, c3, c4: RTMPose 26J) 로드"""
        if 'squat_c1' not in self._models:
            from src.models.second_models.babel_squat.c1_model_v11_a import BabelSquatC1ModelV11A
            from src.models.second_models.babel_squat.c2_model_v7 import BabelSquatC2ModelV7
            from src.models.second_models.babel_squat.c3_model_v7_rtmp import BabelSquatC3ModelV7RTMP
            from src.models.second_models.babel_squat.c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP

            c1_weights = os.path.join(PROJECT_ROOT, "src/models/second_models/babel_squat/c1_model_v11_a_weights.pkl")
            c2_weights = os.path.join(PROJECT_ROOT, "src/models/second_models/babel_squat/c2_model_v7_weights.pkl")

            c1_model = BabelSquatC1ModelV11A(weights_path=c1_weights if os.path.exists(c1_weights) else None)
            c2_model = BabelSquatC2ModelV7(weights_path=c2_weights if os.path.exists(c2_weights) else None)
            c3_model = BabelSquatC3ModelV7RTMP()
            c4_model = BabelSquatC4ModelV6RTMP()

            self._models['squat_c1'] = c1_model
            self._models['squat_c2'] = c2_model
            self._models['squat_c3'] = c3_model
            self._models['squat_c4'] = c4_model
        return {
            'c1': self._models['squat_c1'],
            'c2': self._models['squat_c2'],
            'c3': self._models['squat_c3'],
            'c4': self._models['squat_c4']
        }

    def _get_deadlift_models(self) -> Dict[str, Any]:
        """02_바벨_데드리프트 5대 결함 (d1, d2, d3, d4, d5) 로드"""
        if 'deadlift_d1' not in self._models:
            from src.models.second_models.babel_deadlift.d1_model_rtmpose_confirmed import DeadliftD1ModelRTMPoseConfirmed
            from src.models.second_models.babel_deadlift.d2_model_rtmpose_confirmed import DeadliftD2ModelRTMConfirmed
            from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import BabelDeadliftD3ModelV5_3Opt
            from src.models.second_models.babel_deadlift.d4_model_v6_rtmp import BabelDeadliftD4ModelV6RTMPose
            from src.models.second_models.babel_deadlift.d5_model_v7_two_tier import infer_d5_v7_sequence

            d3_model = BabelDeadliftD3ModelV5_3Opt()
            d3_weights = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "babel_deadlift", "d3_model_v5_3_opt_weights.pkl")
            if os.path.exists(d3_weights):
                d3_model.load_weights(d3_weights)

            self._models['deadlift_d1'] = DeadliftD1ModelRTMPoseConfirmed()
            self._models['deadlift_d2'] = DeadliftD2ModelRTMConfirmed()
            self._models['deadlift_d3'] = d3_model
            self._models['deadlift_d4'] = BabelDeadliftD4ModelV6RTMPose()
            self._models['deadlift_d5'] = infer_d5_v7_sequence

        return {
            'd1': self._models['deadlift_d1'],
            'd2': self._models['deadlift_d2'],
            'd3': self._models['deadlift_d3'],
            'd4': self._models['deadlift_d4'],
            'd5': self._models['deadlift_d5']
        }

    def _get_stiff_deadlift_models(self) -> Dict[str, Any]:
        """03_바벨_스티프_데드리프트 3대 결함 (sd1, sd2, sd3) 로드"""
        if 'stiff_sd1' not in self._models:
            from src.models.second_models.babel_stiff_deadlift.sd1_rtmp_1_classifier import SD1RTMP1Classifier
            from src.models.second_models.babel_stiff_deadlift.sd2_model_v3_rtmp import SD2RTMPClassifier
            from src.models.second_models.babel_stiff_deadlift.sd3_model_v4_rtmp import SD3RTMPClassifier

            self._models['stiff_sd1'] = SD1RTMP1Classifier()
            self._models['stiff_sd2'] = SD2RTMPClassifier()
            self._models['stiff_sd3'] = SD3RTMPClassifier()
        return {
            'sd1': self._models['stiff_sd1'],
            'sd2': self._models['stiff_sd2'],
            'sd3': self._models['stiff_sd3']
        }

    def _get_babel_lunge_models(self) -> Dict[str, Any]:
        """04_바벨_런지 5대 결함 (bl1, bl2, bl3, bl4, bl5) 로드"""
        if 'lunge_bl1' not in self._models:
            import joblib
            from src.models.second_models.babel_lunge.bl1_final import BL1Classifier
            from src.models.second_models.babel_lunge.bl2_final import predict_babel_lunge_c2
            from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_14d_features_v4
            from src.models.second_models.babel_lunge.bl4_v4b import BL4V4bClassifier
            from src.models.second_models.babel_lunge.bl5_final import BL5Classifier

            bl3_path = os.path.join(WEIGHTS_DIR, "bl3_v4_champion_bundle.joblib")
            if not os.path.exists(bl3_path):
                bl3_path = r"D:\Data\피처_데이터\tuning_results\bl3_v4_champion_bundle.joblib"
            bl3_bundle = joblib.load(bl3_path) if os.path.exists(bl3_path) else None

            self._models['lunge_bl1'] = BL1Classifier()
            self._models['lunge_bl2'] = predict_babel_lunge_c2
            self._models['lunge_bl3'] = bl3_bundle
            self._models['lunge_bl3_extractor'] = extract_lunge_bl3_14d_features_v4
            self._models['lunge_bl4'] = BL4V4bClassifier()
            self._models['lunge_bl5'] = BL5Classifier()

        return {
            'bl1': self._models['lunge_bl1'],
            'bl2': self._models['lunge_bl2'],
            'bl3': self._models['lunge_bl3'],
            'bl4': self._models['lunge_bl4'],
            'bl5': self._models['lunge_bl5']
        }

    def _get_dynamic_lunge_models(self) -> Dict[str, Any]:
        """05_다이나믹_런지 5대 결함 (dl1, dl2, dl3, dl4, dl5) 로드"""
        if 'dlunge_dl1' not in self._models:
            from src.models.second_models.dynamic_lunge.dl1_rtmp_1 import DL1RTMP1Classifier
            from src.models.second_models.dynamic_lunge.dl2_rtmp_1 import predict_dynamic_lunge_c2
            from src.models.second_models.dynamic_lunge.dl3_rtmp_1 import DL3RTMP1Classifier
            from src.models.second_models.dynamic_lunge.dl4_rtmp_1 import DL4RTMP1Classifier
            from src.models.second_models.dynamic_lunge.dl5_final import DL5Classifier

            self._models['dlunge_dl1'] = DL1RTMP1Classifier()
            self._models['dlunge_dl2'] = predict_dynamic_lunge_c2
            self._models['dlunge_dl3'] = DL3RTMP1Classifier()
            self._models['dlunge_dl4'] = DL4RTMP1Classifier()
            self._models['dlunge_dl5'] = DL5Classifier()

        return {
            'dl1': self._models['dlunge_dl1'],
            'dl2': self._models['dlunge_dl2'],
            'dl3': self._models['dlunge_dl3'],
            'dl4': self._models['dlunge_dl4'],
            'dl5': self._models['dlunge_dl5']
        }

    def _get_side_lunge_models(self) -> Dict[str, Any]:
        """06_사이드_런지 5대 결함 (sl1, sl2, sl3, sl4, sl5) 로드"""
        if 'side_sl1' not in self._models:
            side_lunge_dir = os.path.join(PROJECT_ROOT, "src", "models", "second_models", "side_lunge")
            if side_lunge_dir not in sys.path:
                sys.path.insert(0, side_lunge_dir)

            from src.models.second_models.side_lunge.sl1_rtmp_1 import SL1RTMP1Classifier
            from src.models.second_models.side_lunge.sl2_rtmp_1 import SL2RTMP1Classifier
            from src.models.second_models.side_lunge.sl3_rtmp_1 import SL3RTMP1Classifier
            from src.models.second_models.side_lunge.sl4_rtmp_1 import SL4RTMP1Classifier
            from src.models.second_models.side_lunge.sl5_rtmp_1 import SL5RTMP1Classifier

            from src.features.side_lunge_c1_features_v11 import extract_side_lunge_c1_feature_vector_v11
            from src.models.second_models.side_lunge.side_lunge_c2_v7_features import extract_side_lunge_c2_v7_features_51d
            from src.features.side_lunge_c3_features_v3 import extract_side_lunge_c3_features_v3_28d
            from src.features.side_lunge_c4_features_v4 import extract_side_lunge_c4_features_v4
            from src.features.side_lunge_c5_features_v7 import extract_side_lunge_c5_v7_feature_vector

            self._models['side_sl1'] = SL1RTMP1Classifier()
            self._models['side_sl2'] = SL2RTMP1Classifier()
            self._models['side_sl3'] = SL3RTMP1Classifier()
            self._models['side_sl4'] = SL4RTMP1Classifier()
            self._models['side_sl5'] = SL5RTMP1Classifier()

            self._models['ext_sl1'] = extract_side_lunge_c1_feature_vector_v11
            self._models['ext_sl2'] = extract_side_lunge_c2_v7_features_51d
            self._models['ext_sl3'] = extract_side_lunge_c3_features_v3_28d
            self._models['ext_sl4'] = extract_side_lunge_c4_features_v4
            self._models['ext_sl5'] = extract_side_lunge_c5_v7_feature_vector

        return {
            'sl1': self._models['side_sl1'],
            'sl2': self._models['side_sl2'],
            'sl3': self._models['side_sl3'],
            'sl4': self._models['side_sl4'],
            'sl5': self._models['side_sl5'],
            'ext_sl1': self._models['ext_sl1'],
            'ext_sl2': self._models['ext_sl2'],
            'ext_sl3': self._models['ext_sl3'],
            'ext_sl4': self._models['ext_sl4'],
            'ext_sl5': self._models['ext_sl5']
        }

    def _get_cross_lunge_models(self) -> Dict[str, Any]:
        """07_크로스_런지 3대 결함 (cl1, cl2, cl3) 로드"""
        if 'cross_cl1' not in self._models:
            from src.models.second_models.cross_lunge.cl1_rtmp_1 import CL1RTMP1Classifier
            from src.models.second_models.cross_lunge.cl2_rtmp_1 import CL2RTMP1Classifier
            from src.models.second_models.cross_lunge.cl3_rtmp_1 import CL3RTMP1Classifier

            self._models['cross_cl1'] = CL1RTMP1Classifier()
            self._models['cross_cl2'] = CL2RTMP1Classifier()
            self._models['cross_cl3'] = CL3RTMP1Classifier()

        return {
            'cl1': self._models['cross_cl1'],
            'cl2': self._models['cross_cl2'],
            'cl3': self._models['cross_cl3']
        }

    def _get_good_morning_models(self) -> Dict[str, Any]:
        """08_굿모닝 3대 결함 (gm1, gm2, gm3) 로드"""
        if 'gm_gm1' not in self._models:
            from src.models.second_models.babel_good_morning.gm1_rtmp_1 import BabelGoodMorningGM1RTMP1
            from src.models.second_models.babel_good_morning.gm2_rtmp_1 import BabelGoodMorningGM2RTMP1
            from src.models.second_models.babel_good_morning.gm3_rtmp_1 import BabelGoodMorningGM3RTMP1

            self._models['gm_gm1'] = BabelGoodMorningGM1RTMP1()
            self._models['gm_gm2'] = BabelGoodMorningGM2RTMP1()
            self._models['gm_gm3'] = BabelGoodMorningGM3RTMP1()

        return {
            'gm1': self._models['gm_gm1'],
            'gm2': self._models['gm_gm2'],
            'gm3': self._models['gm_gm3']
        }

    # =========================================================================
    # 2. 사전 워밍 (Pre-warming)
    # =========================================================================

    def warm_up(self, exercise_id: Optional[int] = None) -> float:
        """지정된 운동(또는 8대 운동 전체)의 모델들을 메모리에 사전 상주시키고 더미 추론 수행"""
        t0 = time.time()
        targets = [exercise_id] if exercise_id is not None else list(range(8))

        dummy_mp = np.zeros((64, 33, 3), dtype=np.float32)
        dummy_rtm = np.zeros((64, 26, 3), dtype=np.float32)

        for eid in targets:
            if self._is_warmed.get(eid, False):
                continue
            try:
                self.evaluate_exercise_defects(eid, dummy_mp, dummy_rtm, view_code='view3')
                self._is_warmed[eid] = True
            except Exception as e:
                logger.warning(f"[Prewarmer] 운동 ID {eid} 사전 워밍 중 경고 발생: {e}")
                self._is_warmed[eid] = True

        elapsed = time.time() - t0
        return elapsed

    # =========================================================================
    # 3. 8대 운동 33종 결함 직접 단일 추론 (Decoupled Direct Dispatch)
    # =========================================================================

    def evaluate_exercise_defects(
        self,
        exercise_id: int,
        mp33_seq: np.ndarray,
        rtm26_seq: np.ndarray,
        view_code: str = 'view3',
        spine_seq: Optional[np.ndarray] = None
    ) -> List[DefectPrediction]:
        """
        마스터 클래스 없이 8대 운동별 개별 세부 결함 모델을 직접 호출하여
        표준화된 DefectPrediction 리스트로 반환
        """
        # 관절 데이터 규격 보증 (T == 64)
        if len(mp33_seq) != 64:
            mp33_seq = SkeletalTopologyAdapter.resample_to_64_frames(mp33_seq)
        if len(rtm26_seq) != 64:
            rtm26_seq = SkeletalTopologyAdapter.resample_to_64_frames(rtm26_seq)

        results: List[DefectPrediction] = []

        # -------------------------------------------------------------
        # [01. 바벨 스쿼트: c1, c2 (MediaPipe 33J) / c3, c4 (RTMPose 26J)]
        # -------------------------------------------------------------
        if exercise_id == 0:
            models = self._get_squat_models()

            # c1: 척추의 중립 위반 (MediaPipe 33J + SpinePose 9J 정밀 기반 v11_a)
            try:
                res_c1 = models['c1'].predict_selective(mp33_seq, spine_9=spine_seq)
                prob_c1 = float(res_c1.get('defect_probability') or 0.0)
                st_c1 = str(res_c1.get('status', 'NORMAL'))
                s_c1 = 1 if st_c1 == 'DEFECT' else (2 if st_c1 == 'UNDETERMINED' else 0)
                lbl_c1 = "결함" if s_c1 == 1 else ("주의/보류" if s_c1 == 2 else "정상")
                fb_c1 = res_c1.get('message', '피드백 자리')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] squat c1 실패: {e}")
                prob_c1, s_c1, lbl_c1, fb_c1 = 0.0, 0, "정상", "피드백 자리"
            results.append(DefectPrediction(
                defect_code='c1',
                defect_name='척추의 중립 위반 (Spine Neutrality)',
                probability=round(prob_c1, 4),
                state=s_c1,
                state_label=lbl_c1,
                feedback=fb_c1 or "피드백 자리"
            ))

            # c2: 고개 정면 및 시선 (MediaPipe 33J 기반 v7.0)
            try:
                res_c2 = models['c2'].predict_selective(mp33_seq)
                prob_c2 = float(res_c2.get('defect_probability') or 0.0)
                st_c2 = str(res_c2.get('status', 'NORMAL'))
                s_c2 = 1 if st_c2 == 'DEFECT' else (2 if st_c2 == 'UNDETERMINED' else 0)
                lbl_c2 = "결함" if s_c2 == 1 else ("주의/보류" if s_c2 == 2 else "정상")
                fb_c2 = res_c2.get('message', '피드백 자리')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] squat c2 실패: {e}")
                prob_c2, s_c2, lbl_c2, fb_c2 = 0.0, 0, "정상", "피드백 자리"
            results.append(DefectPrediction(
                defect_code='c2',
                defect_name='고개 정면 및 시선 (Head & Gaze Orientation)',
                probability=round(prob_c2, 4),
                state=s_c2,
                state_label=lbl_c2,
                feedback=fb_c2 or "피드백 자리"
            ))

            # c3: 무릎-발 방향 일치 (RTMPose 26J 기반 v7_rtmp - F1 94.82%)
            try:
                res_c3 = models['c3'].predict_sequence(rtm26_seq if rtm26_seq is not None else mp33_seq, view_code)
                prob_c3 = float(res_c3.get('calibrated_probability', res_c3.get('raw_probability', 0.0)))
                s_c3 = int(res_c3.get('state_code', 0))
                lbl_c3 = "결함" if s_c3 == 1 else ("주의/보류" if s_c3 == 2 else "정상")
                fb_c3 = res_c3.get('message', res_c3.get('feedback'))
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] squat c3 실패: {e}")
                prob_c3, s_c3, lbl_c3, fb_c3 = 0.0, 0, "정상", None
            results.append(DefectPrediction(
                defect_code='c3',
                defect_name='무릎-발 방향 일치 (Knee-Toe Alignment)',
                probability=round(prob_c3, 4),
                state=s_c3,
                state_label=lbl_c3,
                feedback=fb_c3 or "피드백 자리"
            ))

            # c4: 발바닥 지면 고정 (RTMPose 26J 기반 v6_rtmp - F1 88.12%)
            try:
                res_c4 = models['c4'].predict_sequence(rtm26_seq if rtm26_seq is not None else mp33_seq, view_code)
                prob_c4 = float(res_c4.get('calibrated_probability', res_c4.get('raw_probability', 0.0)))
                s_c4 = int(res_c4.get('state_code', 0))
                lbl_c4 = "결함" if s_c4 == 1 else ("주의/보류" if s_c4 == 2 else "정상")
                fb_c4 = res_c4.get('message', res_c4.get('feedback'))
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] squat c4 실패: {e}")
                prob_c4, s_c4, lbl_c4, fb_c4 = 0.0, 0, "정상", None
            results.append(DefectPrediction(
                defect_code='c4',
                defect_name='발바닥 지면 고정 (Foot Grounding)',
                probability=round(prob_c4, 4),
                state=s_c4,
                state_label=lbl_c4,
                feedback=fb_c4 or "피드백 자리"
            ))

        # -------------------------------------------------------------
        # [02. 바벨 데드리프트: d1, d2, d3, d4, d5]
        # -------------------------------------------------------------
        elif exercise_id == 1:
            models = self._get_deadlift_models()

            # d1: 척추의 중립 위반 (SpinePose 9J + RTMPose 26J 확정 모델)
            try:
                res_d1 = models['d1'].diagnose_3state(rtm26_seq, spine_9_64f=spine_seq)
                p_d1 = float(res_d1.get('defect_probability', 0.0))
                s_d1 = int(res_d1.get('state_code', 0))
                lbl_d1 = "결함" if s_d1 == 1 else ("주의/보류" if s_d1 == 2 else "정상")
                fb_d1 = res_d1.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] deadlift d1 실패: {e}")
                p_d1, s_d1, lbl_d1, fb_d1 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(
                defect_code='d1',
                defect_name='척추의 중립 위반 (Spine Neutrality)',
                probability=round(p_d1, 4),
                state=s_d1,
                state_label=lbl_d1,
                feedback=fb_d1 or "피드백 자리"
            ))

            # d2
            try:
                res_d2 = models['d2'].predict_sequence(rtm26_seq)
                p_d2 = float(res_d2.get('defect_prob', 0.0))
                s_d2 = 1 if p_d2 >= models['d2'].tau_def else (2 if p_d2 > models['d2'].tau_norm else 0)
                lbl_d2 = "결함" if s_d2 == 1 else ("주의/보류" if s_d2 == 2 else "정상")
                fb_d2 = res_d2.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] deadlift d2 실패: {e}")
                p_d2, s_d2, lbl_d2, fb_d2 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='d2', defect_name='무릎-발목 정렬 모임/외반 (Knee Medial Collapse)', probability=round(p_d2, 4), state=s_d2, state_label=lbl_d2, feedback=fb_d2 or "피드백 자리"))

            # d3
            try:
                res_d3 = models['d3'].predict_single_sequence(mp33_seq, view_code=view_code)
                p_d3 = float(res_d3.get('prob', 0.0))
                dec_d3 = int(res_d3.get('state', 0))
                s_d3 = 1 if dec_d3 == 1 else (2 if dec_d3 == -1 else 0)
                lbl_d3 = "결함" if s_d3 == 1 else ("주의/보류" if s_d3 == 2 else "정상")
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] deadlift d3 실패: {e}")
                p_d3, s_d3, lbl_d3 = 0.0, 0, "정상"
            results.append(DefectPrediction(defect_code='d3', defect_name='발바닥 지면 고정 / 힙 슈트업 (Foot Ground Fixation)', probability=round(p_d3, 4), state=s_d3, state_label=lbl_d3, feedback="피드백 자리"))

            # d4
            try:
                res_d4 = models['d4'].predict_sequence(mp33_seq, view_code)
                p_d4 = float(res_d4.get('calibrated_probability', res_d4.get('defect_probability', 0.0)))
                s_d4 = int(res_d4.get('state_code', 0))
                lbl_d4 = "결함" if s_d4 == 1 else ("주의/보류" if s_d4 == 2 else "정상")
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] deadlift d4 실패: {e}")
                p_d4, s_d4, lbl_d4 = 0.0, 0, "정상"
            results.append(DefectPrediction(defect_code='d4', defect_name='무릎-골반 동시 신전 불일치 (Knee-Hip Desynchronization)', probability=round(p_d4, 4), state=s_d4, state_label=lbl_d4, feedback="피드백 자리"))

            # d5
            try:
                res_d5 = models['d5'](mp33_seq, view_name=view_code)
                p_d5 = float(res_d5.get('defect_prob', 0.0))
                pred_d5 = int(res_d5.get('prediction', 0))
                is_dec_d5 = bool(res_d5.get('is_decided', True))
                s_d5 = 1 if pred_d5 == 1 else (0 if is_dec_d5 else 2)
                lbl_d5 = "결함" if s_d5 == 1 else ("주의/보류" if s_d5 == 2 else "정상")
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] deadlift d5 실패: {e}")
                p_d5, s_d5, lbl_d5 = 0.0, 0, "정상"
            results.append(DefectPrediction(defect_code='d5', defect_name='바벨 신체 밀착 실패 (Barbell Path Proximity)', probability=round(p_d5, 4), state=s_d5, state_label=lbl_d5, feedback="피드백 자리"))

        # -------------------------------------------------------------
        # [03. 바벨 스티프 데드리프트: sd1, sd2, sd3]
        # -------------------------------------------------------------
        elif exercise_id == 2:
            models = self._get_stiff_deadlift_models()

            # sd1
            try:
                res_sd1 = models['sd1'].diagnose(rtm26_seq, view_name=view_code)
                p_sd1 = float(res_sd1.get('defect_probability', 0.0))
                s_str_sd1 = res_sd1.get('decision_state', 'CLEAN')
                s_sd1 = 1 if s_str_sd1 == 'DEFECT' else (2 if s_str_sd1 == 'RETAKE' else 0)
                lbl_sd1 = "결함" if s_sd1 == 1 else ("주의/보류" if s_sd1 == 2 else "정상")
                fb_sd1 = res_sd1.get('feedback_message', res_sd1.get('feedback', ''))
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] stiff_deadlift sd1 실패: {e}")
                p_sd1, s_sd1, lbl_sd1, fb_sd1 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='sd1', defect_name='무릎 구부린 채 고정 위반 (Knee Fixation Violation)', probability=round(p_sd1, 4), state=s_sd1, state_label=lbl_sd1, feedback=fb_sd1 or "피드백 자리"))

            # sd2
            try:
                res_sd2 = models['sd2'].diagnose(rtm26_seq)
                p_sd2 = float(res_sd2.get('probability', 0.0))
                s_sd2 = int(res_sd2.get('state', 0))
                lbl_sd2 = "결함" if s_sd2 == 1 else ("주의/보류" if s_sd2 == 2 else "정상")
                fb_sd2 = res_sd2.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] stiff_deadlift sd2 실패: {e}")
                p_sd2, s_sd2, lbl_sd2, fb_sd2 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='sd2', defect_name='척추의 중립 위반 (Spine Neutrality)', probability=round(p_sd2, 4), state=s_sd2, state_label=lbl_sd2, feedback=fb_sd2 or "피드백 자리"))

            # sd3
            try:
                res_sd3 = models['sd3'].diagnose(rtm26_seq, view_name=view_code)
                p_sd3 = float(res_sd3.get('probability', 0.0))
                s_sd3 = int(res_sd3.get('state', 0))
                lbl_sd3 = "결함" if s_sd3 == 1 else ("주의/보류" if s_sd3 == 2 else "정상")
                fb_sd3 = res_sd3.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] stiff_deadlift sd3 실패: {e}")
                p_sd3, s_sd3, lbl_sd3, fb_sd3 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='sd3', defect_name='바벨 궤적과 몸 밀착 위반 (Barbell Path Proximity)', probability=round(p_sd3, 4), state=s_sd3, state_label=lbl_sd3, feedback=fb_sd3 or "피드백 자리"))

        # -------------------------------------------------------------
        # [04. 바벨 런지: bl1, bl2, bl3, bl4, bl5]
        # -------------------------------------------------------------
        elif exercise_id == 3:
            models = self._get_babel_lunge_models()

            # bl1
            try:
                res_bl1 = models['bl1'].predict_sequence(mp33_seq)
                p_bl1 = float(res_bl1.get('defect_probability', res_bl1.get('prob_defect', 0.0)))
                s_bl1 = int(res_bl1.get('state', 0))
                lbl_bl1 = "결함" if s_bl1 == 1 else ("주의/보류" if s_bl1 == 2 else "정상")
                fb_bl1 = res_bl1.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] babel_lunge bl1 실패: {e}")
                p_bl1, s_bl1, lbl_bl1, fb_bl1 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='bl1', defect_name='앞다리 무릎 각도 90도 (Lead Knee 90°)', probability=round(p_bl1, 4), state=s_bl1, state_label=lbl_bl1, feedback=fb_bl1 or "피드백 자리"))

            # bl2
            try:
                res_bl2 = models['bl2'](mp33_seq, rtm26_seq, camera_view=view_code)
                p_bl2 = float(res_bl2.get('probability', 0.0))
                s_str_bl2 = str(res_bl2.get('state', 'CLEAN'))
                s_bl2 = 1 if s_str_bl2 == 'DEFECT' else (2 if s_str_bl2 == 'RETAKE' else 0)
                lbl_bl2 = "결함" if s_bl2 == 1 else ("주의/보류" if s_bl2 == 2 else "정상")
                fb_bl2 = res_bl2.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] babel_lunge bl2 실패: {e}")
                p_bl2, s_bl2, lbl_bl2, fb_bl2 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='bl2', defect_name='몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)', probability=round(p_bl2, 4), state=s_bl2, state_label=lbl_bl2, feedback=fb_bl2 or "피드백 자리"))

            # bl3
            try:
                f14 = self._models['lunge_bl3_extractor'](mp33_seq[:, :, :2]).reshape(1, -1)
                p_bl3 = float(self._models['lunge_bl3']['cb_model'].predict_proba(f14)[0, 1])
                tau_h = float(self._models['lunge_bl3'].get('tau_high', 0.55))
                tau_l = float(self._models['lunge_bl3'].get('tau_low', 0.35))
                s_bl3 = 1 if p_bl3 >= tau_h else (2 if p_bl3 > tau_l else 0)
                lbl_bl3 = "결함" if s_bl3 == 1 else ("주의/보류" if s_bl3 == 2 else "정상")
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] babel_lunge bl3 실패: {e}")
                p_bl3, s_bl3, lbl_bl3 = 0.0, 0, "정상"
            results.append(DefectPrediction(defect_code='bl3', defect_name='뒤다리 무릎 각도 90도 (Rear Knee 90°)', probability=round(p_bl3, 4), state=s_bl3, state_label=lbl_bl3, feedback="피드백 자리"))

            # bl4
            try:
                res_bl4 = models['bl4'].predict_3state(mp33_seq)
                p_bl4 = float(models['bl4'].predict_proba(mp33_seq))
                s_bl4 = int(res_bl4.get('state_code', 0))
                lbl_bl4 = "결함" if s_bl4 == 1 else ("주의/보류" if s_bl4 == 2 else "정상")
                fb_bl4 = res_bl4.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] babel_lunge bl4 실패: {e}")
                p_bl4, s_bl4, lbl_bl4, fb_bl4 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='bl4', defect_name='척추의 중립 위반 (Spine Neutrality)', probability=round(p_bl4, 4), state=s_bl4, state_label=lbl_bl4, feedback=fb_bl4 or "피드백 자리"))

            # bl5
            try:
                res_bl5 = models['bl5'].predict_sequence(mp33_seq, rtm26_seq, view=view_code)
                p_bl5 = float(res_bl5.get('defect_prob', 0.0))
                s_bl5 = int(res_bl5.get('state_code', 0))
                lbl_bl5 = "결함" if s_bl5 == 1 else ("주의/보류" if s_bl5 == 2 else "정상")
                fb_bl5 = res_bl5.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] babel_lunge bl5 실패: {e}")
                p_bl5, s_bl5, lbl_bl5, fb_bl5 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='bl5', defect_name='상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)', probability=round(p_bl5, 4), state=s_bl5, state_label=lbl_bl5, feedback=fb_bl5 or "피드백 자리"))

        # -------------------------------------------------------------
        # [05. 다이나믹 런지: dl1, dl2, dl3, dl4, dl5]
        # -------------------------------------------------------------
        elif exercise_id == 4:
            models = self._get_dynamic_lunge_models()

            # dl1
            try:
                res_dl1 = models['dl1'].predict_sequence(rtm26_seq, view_code=view_code)
                p_dl1 = float(res_dl1.get('defect_probability', 0.0))
                s_dl1 = int(res_dl1.get('state', 0))
                lbl_dl1 = "결함" if s_dl1 == 1 else ("주의/보류" if s_dl1 == 2 else "정상")
                fb_dl1 = res_dl1.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] dynamic_lunge dl1 실패: {e}")
                p_dl1, s_dl1, lbl_dl1, fb_dl1 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='dl1', defect_name='앞다리 무릎 각도 90도 (Lead Knee 90°)', probability=round(p_dl1, 4), state=s_dl1, state_label=lbl_dl1, feedback=fb_dl1 or "피드백 자리"))

            # dl2
            try:
                res_dl2 = models['dl2'](mp33_seq, rtm26_seq, camera_view=view_code)
                p_dl2 = float(res_dl2.get('probability', res_dl2.get('prob', 0.0)))
                s3_dl2 = str(res_dl2.get('state', 'CLEAN'))
                s_dl2 = 1 if s3_dl2 == 'DEFECT' else (2 if s3_dl2 == 'RETAKE' else 0)
                lbl_dl2 = "결함" if s_dl2 == 1 else ("주의/보류" if s_dl2 == 2 else "정상")
                fb_dl2 = res_dl2.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] dynamic_lunge dl2 실패: {e}")
                p_dl2, s_dl2, lbl_dl2, fb_dl2 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='dl2', defect_name='몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)', probability=round(p_dl2, 4), state=s_dl2, state_label=lbl_dl2, feedback=fb_dl2 or "피드백 자리"))

            # dl3
            try:
                res_dl3 = models['dl3'].predict_sequence(rtm26_seq)
                p_dl3 = float(res_dl3.get('defect_probability', 0.0))
                s_dl3 = int(res_dl3.get('state', 0))
                lbl_dl3 = "결함" if s_dl3 == 1 else ("주의/보류" if s_dl3 == 2 else "정상")
                fb_dl3 = res_dl3.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] dynamic_lunge dl3 실패: {e}")
                p_dl3, s_dl3, lbl_dl3, fb_dl3 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='dl3', defect_name='뒤다리 무릎 각도 90도 (Rear Knee 90°)', probability=round(p_dl3, 4), state=s_dl3, state_label=lbl_dl3, feedback=fb_dl3 or "피드백 자리"))

            # dl4
            try:
                res_dl4 = models['dl4'].predict_3state(rtm26_seq, view=view_code)
                p_dl4 = float(res_dl4.get('probability', 0.0))
                s_dl4 = int(res_dl4.get('state', 0))
                lbl_dl4 = "결함" if s_dl4 == 1 else ("주의/보류" if s_dl4 == 2 else "정상")
                fb_dl4 = res_dl4.get('feedback', '')
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] dynamic_lunge dl4 실패: {e}")
                p_dl4, s_dl4, lbl_dl4, fb_dl4 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='dl4', defect_name='척추의 중립 위반 (Spine Neutrality)', probability=round(p_dl4, 4), state=s_dl4, state_label=lbl_dl4, feedback=fb_dl4 or "피드백 자리"))

            # dl5
            try:
                res_dl5 = models['dl5'].predict_sequence(mp33_seq, rtm26_seq, view=view_code)
                p_dl5 = float(res_dl5.get('defect_prob', 0.0))
                s_dl5 = int(res_dl5.get('state_code', 0))
                lbl_dl5 = "결함" if s_dl5 == 1 else ("주의/보류" if s_dl5 == 2 else "정상")
                raw_fb_dl5 = res_dl5.get('feedback', '')
                fb_dl5 = " ".join(raw_fb_dl5) if isinstance(raw_fb_dl5, list) else str(raw_fb_dl5)
            except Exception as e:
                logger.error(f"[evaluate_exercise_defects] dynamic_lunge dl5 실패: {e}")
                p_dl5, s_dl5, lbl_dl5, fb_dl5 = 0.0, 0, "정상", ""
            results.append(DefectPrediction(defect_code='dl5', defect_name='상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)', probability=round(p_dl5, 4), state=s_dl5, state_label=lbl_dl5, feedback=fb_dl5 or "피드백 자리"))

        # -------------------------------------------------------------
        # [06. 사이드 런지: sl1, sl2, sl3, sl4, sl5]
        # -------------------------------------------------------------
        elif exercise_id == 5:
            models = self._get_side_lunge_models()
            side_lunge_defs = [
                ('sl1', '앞다리 무릎 각도 90도 (Lead Knee 90°)', 'sl1', 'ext_sl1'),
                ('sl2', '몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)', 'sl2', 'ext_sl2'),
                ('sl3', '안 굽히는 다리 구부정 / 신전 결함 (Trail Leg Knee Extension)', 'sl3', 'ext_sl3'),
                ('sl4', '척추의 중립 위반 (Spine Neutrality)', 'sl4', 'ext_sl4'),
                ('sl5', '상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)', 'sl5', 'ext_sl5')
            ]
            for code, name, m_key, ext_key in side_lunge_defs:
                try:
                    if code == 'sl2':
                        f_vec = models[ext_key](mp33_seq, view_code=view_code)
                    else:
                        f_vec = models[ext_key](mp33_seq)
                    res = models[m_key].predict_3state(f_vec, view_code=view_code)
                    prob = float(res.get('calibrated_prob', res.get('probability', 0.0)))
                    state = int(res.get('state', 0))
                    state_lbl = "결함" if state == 1 else ("주의/보류" if state == 2 else "정상")
                    fb = res.get('feedback', res.get('message', ''))
                except Exception as e:
                    logger.error(f"[evaluate_exercise_defects] side_lunge {code} 실패: {e}")
                    prob, state, state_lbl, fb = 0.0, 0, "정상", ""

                results.append(DefectPrediction(
                    defect_code=code,
                    defect_name=name,
                    probability=round(prob, 4),
                    state=state,
                    state_label=state_lbl,
                    feedback=fb or "피드백 자리"
                ))

        # -------------------------------------------------------------
        # [07. 크로스 런지: cl1, cl2, cl3]
        # -------------------------------------------------------------
        elif exercise_id == 6:
            models = self._get_cross_lunge_models()
            for code, name, m_key in [
                ('cl1', '앞다리 무릎 각도 90도 (Lead Knee 90°)', 'cl1'),
                ('cl2', '앞다리 무릎 안쪽 무너짐 (Lead Knee Valgus Collapse)', 'cl2'),
                ('cl3', '교차 다리 궤적 이탈 / 과도한 교차 (Cross Alignment Deficit)', 'cl3')
            ]:
                try:
                    res = models[m_key].predict(rtm26_seq, camera_view=view_code)
                    prob = float(res.get('defect_probability', res.get('raw_probability', 0.0)))
                    s3 = str(res.get('prediction_3state', 'NORMAL'))
                    state = 1 if s3 == 'DEFECT' else (2 if s3 == 'UNCERTAIN' else 0)
                    state_lbl = "결함" if state == 1 else ("주의/보류" if state == 2 else "정상")
                    fb = res.get('feedback', res.get('message', ''))
                except Exception as e:
                    logger.error(f"[evaluate_exercise_defects] cross_lunge {code} 실패: {e}")
                    prob, state, state_lbl, fb = 0.0, 0, "정상", ""

                results.append(DefectPrediction(
                    defect_code=code,
                    defect_name=name,
                    probability=round(prob, 4),
                    state=state,
                    state_label=state_lbl,
                    feedback=fb or "피드백 자리"
                ))

        # -------------------------------------------------------------
        # [08. 굿모닝: gm1, gm2, gm3]
        # -------------------------------------------------------------
        elif exercise_id == 7:
            models = self._get_good_morning_models()
            for code, name, m_key in [
                ('gm1', '무릎 구부린 채 고정 위반 (Knee Fixed Angle Violation)', 'gm1'),
                ('gm2', '척추의 중립 위반 (Spine Neutrality)', 'gm2'),
                ('gm3', '고관절 힌지 깊이 미흡 (Hip Hinge Depth Deficit)', 'gm3')
            ]:
                try:
                    res = models[m_key].predict_sequence(rtm26_seq, view_code)
                    prob = float(res.get('calibrated_probability', res.get('raw_probability', 0.0)))
                    state = int(res.get('state_code', 0))
                    state_lbl = "결함" if state == 1 else ("주의/보류" if state == 2 else "정상")
                    fb = res.get('feedback', res.get('message', ''))
                except Exception as e:
                    logger.error(f"[evaluate_exercise_defects] good_morning {code} 실패: {e}")
                    prob, state, state_lbl, fb = 0.0, 0, "정상", ""

                results.append(DefectPrediction(
                    defect_code=code,
                    defect_name=name,
                    probability=round(prob, 4),
                    state=state,
                    state_label=state_lbl,
                    feedback=fb or "피드백 자리"
                ))

        # -------------------------------------------------------------
        # [3단계 생체역학 코칭 피드백 자동 주입 (Stage 3 Feedback Engine)]
        # -------------------------------------------------------------
        from src.web.core.feedback_engine import FeedbackEngine
        fb_engine = FeedbackEngine.get_instance()
        for dp in results:
            item = fb_engine.get_feedback_def(dp.defect_code)
            if item:
                dp.target_joint = item.target_joint
                dp.action_cue = item.action_cue
                dp.corrective_action = item.corrective_action
                dp.biomechanical_reason = item.biomechanical_reason
                dp.optimal_view = item.optimal_view
                # 기존 하드코딩된 플레이스홀더('피드백 자리')를 실질적 조치 문구로 자동 치환
                if not dp.feedback or dp.feedback == "피드백 자리":
                    dp.feedback = fb_engine.format_single_feedback(
                        dp.defect_code, dp.state, dp.probability, view_code
                    )

        return results

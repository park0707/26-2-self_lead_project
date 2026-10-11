# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 스텝 포워드 & 스텝 백워드)
Module: dl3.py
Purpose: 스포츠 생체역학(IEEE TBME, Journal of Biomechanics) 선행 연구 기반
         원천 라벨(y_orig)을 100% 원본 보존한 상태에서
         다이나믹 런지 세 번째 결함 '뒤다리 무릎 각도 90도 (dl3)' 전담 3-State 초정밀 전문가 분류기 (v1.0 Official Champion)
         
         모델 사양:
         - 피처 구성: 14D 정예 고급 기구학 피처 (bl3_v4 베이스라인 구조 계승)
           * f1 ~ f11: 직교 정예 기구학 피처 (무릎 안정성, 하지 대칭비, 대퇴/정강이 각도, 보폭, 각속도 감속도 등)
           * f12: knee_hysteresis_area (무릎 각도-각속도 위상 평면 히스테리시스 루프 적분 면적)
           * f13: femur_tibia_proj_ratio (최하사점 대퇴/정강이 2D 투영 길이비 - 원근 단축 보정)
           * f14: crp_hip_knee_mean (고관절-슬관절 연속 상대 위상 CRP: 하체 협응 대칭성)
         - 모델 아키텍처: 14D CatBoost / Tri-Ensemble Classifier
         - 학습 및 평가 데이터셋: AI-Hub 다이나믹 런지 1,959개 전수 시퀀스 (05 포워드 1,071개 + 06 백워드 888개)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.features.lunge_dl3_features import extract_lunge_dl3_14d_features, FEATURE_NAMES_14D

MODEL_DIR = r"D:\Data\피처_데이터\tuning_results"
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "dl3_champion_bundle.joblib")

class DL3Classifier:
    """다이나믹 런지 세 번째 결함 (뒤다리 무릎 각도 90도, dl3) 전담 14D 정예 기구학 3-State 전문가 분류기"""
    
    EXERCISE_NAME = "다이나믹 런지"
    DEFECT_NAME = "뒤다리 무릎 각도 90도"
    MODEL_CODE = "dl3"
    VERSION = "v1.0"
    
    # 기본 하이퍼파라미터 (bl3 v4 계승)
    BEST_PARAMS = {
        'iterations': 250,
        'learning_rate': 0.035,
        'depth': 5,
        'l2_leaf_reg': 4.0,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 4
    }
    
    # 기본 임계값 (번들 로드 시 자동 업데이트)
    OPTIMAL_TAU = 0.45
    TAU_LOW = 0.30
    TAU_HIGH = 0.60
    
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or MODEL_BUNDLE_PATH
        self.cb_model: Optional[CatBoostClassifier] = None
        self.bundle_data: Optional[Dict[str, Any]] = None
        self._load_or_init_model()
        
    def _load_or_init_model(self):
        """저장된 번들 파일에서 챔피언 모델 로드"""
        if os.path.exists(self.bundle_path):
            try:
                bundle = joblib.load(self.bundle_path)
                self.bundle_data = bundle
                self.cb_model = bundle.get('cb_model')
                self.OPTIMAL_TAU = float(bundle.get('optimal_tau', self.OPTIMAL_TAU))
                self.TAU_LOW = float(bundle.get('tau_low', self.TAU_LOW))
                self.TAU_HIGH = float(bundle.get('tau_high', self.TAU_HIGH))
                return
            except Exception:
                pass
                
        self.cb_model = CatBoostClassifier(**self.BEST_PARAMS)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """14D 정예 피처셋 전체 학습 및 모델 번들 저장"""
        self.cb_model = CatBoostClassifier(**self.BEST_PARAMS)
        self.cb_model.fit(X, y)
        return self
        
    def extract_features(self, kps: np.ndarray) -> np.ndarray:
        """단일 64프레임 관절 시퀀스 (64, 15, 2)로부터 14D 정예 고급 생체역학 피처 추출"""
        return extract_lunge_dl3_14d_features(kps)
        
    def predict_single(self, kps: np.ndarray) -> Dict[str, Any]:
        """
        단일 시퀀스(64, 15, 2)에 대해 dl3 3-State 판정 및 맞춤형 생체역학 코칭 메시지 반환
        """
        feat_14d = self.extract_features(kps).reshape(1, -1)
        
        # 모델 결측 시 번들 재확인
        if self.cb_model is None:
            self._load_or_init_model()
            
        prob_defect = float(self.cb_model.predict_proba(feat_14d)[0, 1])
        
        # 3-State 판정 로직: 0 (정상), 1 (결함), -1 (보류/재확인)
        if prob_defect < self.TAU_LOW:
            state = 0
            state_label = "정상 (Normal)"
            confidence = (self.TAU_LOW - prob_defect) / self.TAU_LOW
        elif prob_defect > self.TAU_HIGH:
            state = 1
            state_label = "결함 (Defect: 뒤다리 무릎 각도 불량)"
            confidence = (prob_defect - self.TAU_HIGH) / (1.0 - self.TAU_HIGH)
        else:
            state = -1
            state_label = "판정보류 (Undetermined / Re-check)"
            confidence = 1.0 - abs(prob_defect - self.OPTIMAL_TAU) / (self.TAU_HIGH - self.TAU_LOW + 1e-6)
            
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # 기구학 핵심 피처 추출 (코칭용)
        # feat_14d 인덱스 매핑:
        # 10: rear_knee_angle_bottom, 9: lead_knee_angle_bottom, 6: rear_knee_target_dev_bottom
        rear_knee_bot = float(feat_14d[0, 10])
        lead_knee_bot = float(feat_14d[0, 9])
        dev_bot = float(feat_14d[0, 6])
        
        # 맞춤형 코칭 피드백 생성
        feedback = self._generate_feedback(state, rear_knee_bot, lead_knee_bot, dev_bot)
        
        return {
            'exercise': self.EXERCISE_NAME,
            'defect_code': self.MODEL_CODE,
            'defect_name': self.DEFECT_NAME,
            'state': state,
            'state_label': state_label,
            'defect_probability': prob_defect,
            'confidence': confidence,
            'kinematics': {
                'rear_knee_angle_bottom_deg': round(rear_knee_bot, 2),
                'lead_knee_angle_bottom_deg': round(lead_knee_bot, 2),
                'target_90deg_deviation': round(dev_bot, 2)
            },
            'coaching_feedback': feedback
        }
        
    def _generate_feedback(self, state: int, rear_knee_deg: float, lead_knee_deg: float, dev_deg: float) -> str:
        """생체역학적 근거 기반 실시간 운동 처방 메시지 생성"""
        if state == 0:
            return f"완벽한 다이나믹 런지 자세입니다! 뒤다리 무릎 각도가 {rear_knee_deg:.1f}°로 90도 표준에 정확히 부합하며 하체 굴곡 협응력이 안정적입니다."
        elif state == 1:
            if rear_knee_deg > 105.0:
                return f"[주의] 런지 시 뒤다리가 충분히 굽혀지지 않았습니다 (최하사점 각도 {rear_knee_deg:.1f}°, 90도 대비 +{dev_deg:.1f}° 편차). 뒷무릎이 지면에 수직으로 약 90도가 되도록 골반을 충분히 낮춰주세요."
            elif rear_knee_deg < 75.0:
                return f"[주의] 뒤다리 무릎이 과도하게 좁게 굽혀졌습니다 (최하사점 각도 {rear_knee_deg:.1f}°, 90도 대비 -{dev_deg:.1f}° 편차). 앞뒤 보폭을 조금 더 넓혀 뒷무릎에 가해지는 슬개건 압박을 줄여주세요."
            else:
                return f"[주의] 다이나믹 런지 하강 전환기에서 뒤다리 무릎 제어가 불안정합니다 (각도 {rear_knee_deg:.1f}°). 스텝 착지 후 골반 중심을 수직으로 통제하며 동작을 완수해주세요."
        else:
            return f"[판정 보류] 프레임 가림 또는 전환 동작 구간입니다. 카메라 정면이나 측면 45도 각도에서 뒤다리 무릎이 잘 보이도록 유지하며 1회 반복을 수행해주세요."

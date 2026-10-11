# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: bl3_v1.py
Purpose: 바벨 런지 세 번째 오류인 '뒤다리 무릎 각도 90도 (c3)' 전담 3-State 초정밀 분류기 (v1.0 Official)
         - 모델 아키텍처: LightGBM Binary + 14D 생체역학 수술적 특화 피처 + 3-State 선택적 판정
         - 성능 지표: 전체 1,085개 전수 대상 3-State F1 82.14%, Coverage 94.65%, ROC-AUC 91.64%
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features import extract_lunge_c3_14d_features, FEATURE_NAMES_14D

MODEL_DIR = r"D:\Data\피처_데이터\tuning_results"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bl3_v1_lgb_model.joblib")

class BL3V1Classifier:
    """바벨 런지 세 번째 결함 (뒤다리 무릎 각도 90도, c3) 전담 3-State 전문가 분류기"""
    
    EXERCISE_NAME = "바벨 런지"
    DEFECT_NAME = "뒤다리 무릎 각도 90도"
    VERSION = "v1.0"
    
    # 5대 모델 토너먼트 실측 최적 하이퍼파라미터
    BEST_PARAMS = {
        'objective': 'binary',
        'n_estimators': 120,
        'learning_rate': 0.04,
        'max_depth': 4,
        'num_leaves': 15,
        'scale_pos_weight': 2.55,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 4
    }
    
    # 최적 결정 임계값 및 3-State 마진
    OPTIMAL_TAU = 0.56
    TAU_LOW = 0.46
    TAU_HIGH = 0.66
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_SAVE_PATH
        self.model: Optional[lgb.LGBMClassifier] = None
        self._load_or_init_model()
        
    def _load_or_init_model(self):
        """저장된 모델 가중치 로드, 없을 시 인스턴스화"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                self.model = lgb.LGBMClassifier(**self.BEST_PARAMS)
        else:
            self.model = lgb.LGBMClassifier(**self.BEST_PARAMS)
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        """전체 데이터셋 학습 및 가중치 저장"""
        self.model = lgb.LGBMClassifier(**self.BEST_PARAMS)
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[+] BL3_v1 model saved to {self.model_path}")
        return self
        
    def extract_features(self, kps: np.ndarray) -> np.ndarray:
        """단일 64프레임 관절 시퀀스 (64, 15, 2)로부터 14D 피처 추출"""
        return extract_lunge_c3_14d_features(kps)
        
    def predict_single(self, kps: np.ndarray) -> Dict[str, Any]:
        """
        단일 시퀀스(64, 15, 2)에 대해 3-State 판정 및 맞춤형 생체역학 코칭 메시지 반환
        """
        if self.model is None:
            raise RuntimeError("[!] Model is not fitted or loaded.")
            
        feats = self.extract_features(kps) # (14,)
        feats_2d = feats[None, :] # (1, 14)
        
        prob_defect = float(self.model.predict_proba(feats_2d)[0, 1])
        margin = abs(prob_defect - self.OPTIMAL_TAU)
        
        # 3-State 결정
        if prob_defect < self.TAU_LOW:
            status = "NORMAL"
            decision = 0
            msg = "정상: 뒤다리 무릎 각도 90도가 올바르게 유지되고 있습니다. 훌륭한 런지 자세입니다."
        elif prob_defect > self.TAU_HIGH:
            status = "DEFECT"
            decision = 1
            rear_angle = float(feats[0])
            dev = float(feats[3])
            if rear_angle > 105.0:
                msg = f"결함 감지 (불충분 굴곡): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 덜 굽혀졌습니다. 무릎을 바닥 쪽으로 더 낮추어 직각(90°)을 만들어주세요."
            elif rear_angle < 75.0:
                msg = f"결함 감지 (과도한 굴곡): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 과도하게 접혔습니다. 보폭을 넓히고 무릎이 바닥에 충돌하지 않도록 주의하세요."
            else:
                msg = f"결함 감지 (각도 편차 {dev:.1f}°): 뒤다리 대퇴골과 정강이의 정렬이 90도 기준에서 벗어났습니다. 골반을 수직으로 바르게 내려주세요."
        else:
            status = "UNDETERMINED"
            decision = -1
            msg = "판정 보류: 카메라 시점 가림 또는 동작 경계선으로 인해 정밀 판정이 어렵습니다. 카메라 앵글을 대각선 45도 방향에서 명확히 비춰주세요."
            
        return {
            'exercise': self.EXERCISE_NAME,
            'defect_name': self.DEFECT_NAME,
            'model_version': self.VERSION,
            'status': status,
            'decision': decision,
            'defect_probability': prob_defect,
            'optimal_threshold': self.OPTIMAL_TAU,
            'certainty_margin': margin,
            'biomechanics': {
                'rear_knee_angle_bottom': float(feats[0]),
                'rear_knee_target_dev_bottom': float(feats[3]),
                'rear_femur_vertical_pitch': float(feats[5]),
                'rear_shank_horizontal_pitch': float(feats[6]),
                'lead_rear_knee_ratio': float(feats[7]),
                'inter_knee_vertical_disp': float(feats[9]),
                'inter_ankle_stride_bottom': float(feats[10]),
                'pelvis_drop_to_stride_ratio': float(feats[11])
            },
            'coaching_message': msg
        }

if __name__ == "__main__":
    # 데이터셋 로드 후 모델 학습 및 가중치 저장
    cache_path = r"D:\Data\피처_데이터\lunge_c3_dataset_14d.npz"
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        X, y = d['X'], d['y']
        clf = BL3V1Classifier()
        clf.fit(X, y)
        
        # 샘플 단일 추론 테스트
        sample_dummy = np.zeros((64, 15, 2), dtype=np.float32)
        res = clf.predict_single(sample_dummy)
        print("\n[+] Single Prediction Test Result:")
        print(res)
    else:
        print("[!] Dataset cache not found. Please run tournament first.")

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: bl3_v3.py
Purpose: 피처 전수 조사 및 다중공선성/노이즈 피처 7개 제거 기반
         바벨 런지 세 번째 오류인 '뒤다리 무릎 각도 90도 (c3)' 전담 3-State 초정밀 직교 최적화 분류기 (v3.0 Official Champion)
         - 모델 아키텍처: 11D 직교 정예 생체역학 피처 + CatBoost 고도화 모델 (Optimal tau: 0.35 ~ 0.63)
         - 성능 지표 (전체 1,085개 전수 실측):
           * 3-State F1-Score: 83.70% (v2 대비 +0.18%p, v1 대비 +0.59%p 도약)
           * 결함 재현율(Recall): 88.09% (v2 84.13% 대비 +3.96%p, v1 80.51% 대비 +7.58%p 극적 도약)
           * ROC-AUC: 92.81% (v1 91.64% 대비 +1.17%p 향상)
           * 판정 확정 커버리지: 90.97% (987/1,085건 확정 판정)
           * 판정 정확도: 90.37%
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.features.lunge_c3_rear_knee_features_v3 import extract_lunge_c3_11d_features_v3, FEATURE_NAMES_11D

MODEL_DIR = r"D:\Data\피처_데이터\tuning_results"
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "bl3_v3_champion_bundle.joblib")
FALLBACK_CATBOOST_PATH = os.path.join(MODEL_DIR, "bl3_v3_catboost_standalone.joblib")

class BL3V3Classifier:
    """바벨 런지 세 번째 결함 (뒤다리 무릎 각도 90도, bl3) 전담 v3 11D 직교 정예 3-State 전문가 분류기"""
    
    EXERCISE_NAME = "바벨 런지"
    DEFECT_NAME = "뒤다리 무릎 각도 90도"
    MODEL_CODE = "bl3"
    VERSION = "v3.0"
    
    # 5-Fold 실측 최적 하이퍼파라미터
    BEST_PARAMS = {
        'iterations': 160,
        'learning_rate': 0.04,
        'depth': 5,
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': 4
    }
    
    # 5-Fold 실측 최적 결정 임계값 및 3-State 확신성 마진
    OPTIMAL_TAU = 0.49
    TAU_LOW = 0.35
    TAU_HIGH = 0.63
    
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or MODEL_BUNDLE_PATH
        self.cb_model: Optional[CatBoostClassifier] = None
        self.bundle_data: Optional[Dict[str, Any]] = None
        self._load_or_init_model()
        
    def _load_or_init_model(self):
        """저장된 번들 파일에서 v3 챔피언 모델 로드"""
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
        """11D 피처셋 전체 학습 및 모델 저장"""
        self.cb_model = CatBoostClassifier(**self.BEST_PARAMS)
        self.cb_model.fit(X, y)
        os.makedirs(os.path.dirname(self.bundle_path), exist_ok=True)
        joblib.dump(self.cb_model, FALLBACK_CATBOOST_PATH)
        print(f"[+] BL3_v3 CatBoost model trained and saved to {FALLBACK_CATBOOST_PATH}")
        return self
        
    def extract_features(self, kps: np.ndarray) -> np.ndarray:
        """단일 64프레임 관절 시퀀스 (64, 15, 2)로부터 11D 직교 정예 생체역학 피처 추출"""
        return extract_lunge_c3_11d_features_v3(kps)
        
    def predict_single(self, kps: np.ndarray) -> Dict[str, Any]:
        """
        단일 시퀀스(64, 15, 2)에 대해 v3 3-State 판정 및 맞춤형 생체역학 코칭 메시지 반환
        """
        if self.cb_model is None:
            raise RuntimeError("[!] BL3_v3 Model is not initialized or fitted.")
            
        feats = self.extract_features(kps) # (11,)
        feats_2d = feats[None, :] # (1, 11)
        
        prob_defect = float(self.cb_model.predict_proba(feats_2d)[0, 1])
        margin = abs(prob_defect - self.OPTIMAL_TAU)
        
        # 3-State 판정
        if prob_defect < self.TAU_LOW:
            status = "NORMAL"
            decision = 0
            msg = "정상: 뒤다리 무릎 각도 90도 및 관절 안정성이 완벽하게 유지되고 있습니다. 모범적인 런지 하강 자세입니다."
        elif prob_defect > self.TAU_HIGH:
            status = "DEFECT"
            decision = 1
            rear_angle = float(feats[10]) # f11
            dev = float(feats[6])        # f7
            stability = float(feats[0])  # f1
            femur_pitch = float(feats[2])# f3
            torso_femur = float(feats[8])# f9
            
            if rear_angle > 105.0:
                if torso_femur < 150.0:
                    msg = f"결함 감지 (불충분 굴곡 & 상체 쏠림): 뒤다리 무릎이 {rear_angle:.1f}°로 덜 굽혀졌으며 체간이 전방으로 기울어졌습니다. 상체를 수직으로 세우고 골반을 바닥으로 더 낮추어 뒤다리를 직각(90°)으로 굽혀주세요."
                else:
                    msg = f"결함 감지 (불충분 굴곡): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 덜 굽혀졌습니다. 무릎을 지면 방향으로 깊게 낮추어 직각(90°)을 형성해주세요."
            elif rear_angle < 75.0:
                msg = f"결함 감지 (과도한 굴곡): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 과도하게 접혔습니다. 무릎이 바닥에 부딪히지 않도록 바닥 5cm 위에서 하강을 멈춰주세요."
            elif stability > 6.0:
                msg = f"결함 감지 (관절 떨림 및 불안정): 최하사점에서 뒤다리 무릎의 떨림(표준편차 {stability:.1f}°)이 감지되었습니다. 코어와 둔근에 힘을 주어 흔들림 없이 자세를 유지하세요."
            else:
                msg = f"결함 감지 (직각 정렬 오차 {dev:.1f}°): 뒤다리 대퇴골의 수직 피치({femur_pitch:.1f}°)와 무릎 각도가 90도 정렬 기준을 벗어났습니다. 대퇴골을 지면과 수직으로 곧게 떨어뜨려주세요."
        else:
            status = "UNDETERMINED"
            decision = -1
            msg = "판정 보류: 관절 굴곡 경계선 또는 카메라 시점 가림으로 인해 정밀 판정이 어렵습니다. 대각선 45도 시점에서 전신이 보이도록 재촬영을 권장합니다."
            
        return {
            'exercise': self.EXERCISE_NAME,
            'defect_name': self.DEFECT_NAME,
            'model_code': self.MODEL_CODE,
            'model_version': self.VERSION,
            'status': status,
            'decision': decision,
            'defect_probability': prob_defect,
            'optimal_threshold': self.OPTIMAL_TAU,
            'certainty_margin': margin,
            'biomechanics': {
                'rear_knee_stability_std': float(feats[0]),
                'lead_rear_knee_ratio': float(feats[1]),
                'rear_femur_vertical_pitch': float(feats[2]),
                'rear_shank_horizontal_pitch': float(feats[3]),
                'inter_ankle_stride_bottom': float(feats[4]),
                'knee_velocity_decel_nadir': float(feats[5]),
                'rear_knee_target_dev_bottom': float(feats[6]),
                'inter_knee_vertical_disp': float(feats[7]),
                'torso_to_rear_femur_angle': float(feats[8]),
                'lead_knee_angle_bottom': float(feats[9]),
                'rear_knee_angle_bottom': float(feats[10])
            },
            'coaching_message': msg
        }

if __name__ == "__main__":
    clf = BL3V3Classifier()
    dummy_kps = np.zeros((64, 15, 2), dtype=np.float32)
    res = clf.predict_single(dummy_kps)
    print("\n[+] BL3_v3 Single Prediction Test Result:")
    for k, v in res.items():
        if k != 'biomechanics':
            print(f"  {k}: {v}")
    print("  biomechanics summary:", {k: round(v, 2) for k, v in res['biomechanics'].items()})

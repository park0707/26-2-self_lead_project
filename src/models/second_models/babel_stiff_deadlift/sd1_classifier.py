# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 1: sd1 (무릎 구부린 채 고정 위반 / Knee Fixed-Angle Defect)
Version: v7.0 (Official Production Standard)
Module: sd1_classifier.py
Purpose: 스티프 데드리프트 1번 고유 결함 판별을 위한 공식 프로덕션 추론기
         (Pelvis-Gravity Canonical 3D + 생체역학 12D + CatBoost 5-Fold 앙상블, v6 절제 최적화 버전)
         ※ v6 계승 이력: sd1_v6_classifier.py / weights/sd1_v6_bundle.pkl 보존
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sd1_v7_classifier import SD1V7Classifier

class SD1Classifier(SD1V7Classifier):
    """
    vidio-health-trainer 스티프 데드리프트 1번 결함(sd1) 공식 프로덕션 분류기
    - 공식 버전: v7.0 (v6 절제 최적화: 중복 정규화 피처 2개 제거, 12D CatBoost 앙상블)
    - LOSO 9-Fold(시드 10개 평균) 기준: 결함 F1 49.9%(정제 라벨) / 35.8%(원본 라벨), AUC 81.9% / 57.4%
    - 평가 보류율: 약 9.9% (Conformal Prediction 10% Bounded 준수)
    """
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            default_path = os.path.join(CURRENT_DIR, "weights", "sd1_bundle.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "weights", "sd1_v7_bundle.pkl")
            bundle_path = default_path
        super().__init__(bundle_path=bundle_path)

if __name__ == '__main__':
    clf = SD1Classifier()
    print("[+] vidio-health-trainer sd1 공식 프로덕션 분류기(v7.0) 로딩 완료!")
    print(f"  - 임계값: tau_lower={clf.tau_lower:.4f}, tau_upper={clf.tau_upper:.4f}")
    dummy_seq = np.random.randn(64, 98).astype(np.float32)
    diag = clf.diagnose(dummy_seq)
    print(f"  - 단일 시퀀스 진단 테스트: {diag['state_label']} (확률: {diag['probability']:.4f})")

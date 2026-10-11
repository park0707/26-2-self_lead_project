# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2 (척추의 중립 위반 / Spine Neutrality Defect)
Version: v3.0 RTMPose Optimized (Official Production Standard)
Module: sd2_classifier.py
Purpose: 스티프 데드리프트 2번 고유 결함 판별을 위한 공식 최종 프로덕션 추론기
         - RTMPose-m (Halpe26) 2D 직교 기저계 기반 30D 생체역학 피처
         - Two-Tier Tri-Ensemble (24D TCN + LightGBM + CatBoost)
         - 3-State 선택적 진단 (CLEAN, DEFECT, RETAKE)
         ※ Halpe26(26관절) 및 MediaPipe(33관절) 64프레임 시퀀스 완전 호환 지원
         ※ 레거시 버전 보존: sd2_v3_classifier.py
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd2_model_v3_rtmp import SD2RTMPClassifier


class SD2Classifier(SD2RTMPClassifier):
    """
    vidio-health-trainer 스티프 데드리프트 2번 결함(sd2) 공식 최종 프로덕션 분류기
    - 공식 버전: v3.0 RTMPose Optimized (최종 승격 버전)
    - 2D 평면 체간 기저계 30D 피처 + TCN-LightGBM-CatBoost Tri-Ensemble
    - 5-Fold StratifiedGroupKFold 기준 3-State F1 89.86%, ROC-AUC 81.25%, 선택적 재현율 100.0%
    """
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            default_path = os.path.join(CURRENT_DIR, "sd2_model_v3_rtmp_weights.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "weights", "sd2_model_v3_rtmp_weights.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")
            bundle_path = default_path
        super().__init__(bundle_path=bundle_path)


if __name__ == '__main__':
    clf = SD2Classifier()
    print("[+] vidio-health-trainer sd2 공식 프로덕션 분류기(v3.0 RTMPose Optimized) 로딩 완료!")
    print(f"  - Model: {clf.model_name} ({clf.defect_name})")
    print(f"  - Binary Threshold: {clf.binary_threshold:.3f}")
    print(f"  - Selective Thresholds: tau_clean={clf.tau_clean:.3f}, tau_defect={clf.tau_defect:.3f}")

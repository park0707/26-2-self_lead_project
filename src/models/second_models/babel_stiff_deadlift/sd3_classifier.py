# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect)
Version: v4.0 RTMPose Optimized (Official Production Standard)
Module: sd3_classifier.py
Purpose: 스티프 데드리프트 3번 고유 결함 판별을 위한 공식 최종 프로덕션 추론기
         - RTMPose-m (Halpe26) 2D 기하 36D GBDT + Dilated-TCN 2-Stream
         - Multi-View Dynamic Meta Fusion (5대 시점 적응형 가중 결합)
         - 3-State 선택적 진단 (CLEAN, DEFECT, RETAKE)
         ※ Halpe26(26관절) 및 MediaPipe(33관절) 64프레임 시퀀스 완전 호환 지원
         ※ 레거시 버전 보존: sd3_v1_classifier.py, sd3_v2_classifier.py, sd3_v4_classifier.py
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd3_model_v4_rtmp import SD3RTMPClassifier


class SD3Classifier(SD3RTMPClassifier):
    """
    vidio-health-trainer 스티프 데드리프트 3번 결함(sd3) 공식 최종 프로덕션 분류기
    - 공식 버전: v4.0 RTMPose Optimized (최종 승격 버전)
    - Multi-View Dynamic Meta Fusion (36D GBDT Tabular + Dilated-TCN 2-Stream)
    - 5-Fold StratifiedGroupKFold 기준 3-State F1 80.39%, ROC-AUC 89.18%
    - 대각/후방 4대 시점 F1 83.87% ~ 91.97% SOTA 달성
    """
    def __init__(self, bundle_path: str = None):
        if bundle_path is None:
            default_path = os.path.join(CURRENT_DIR, "sd3_model_v4_rtmp_weights.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "weights", "sd3_model_v4_rtmp_weights.pkl")
            if not os.path.exists(default_path):
                default_path = os.path.join(CURRENT_DIR, "weights", "sd3_bundle.pkl")
            bundle_path = default_path
        super().__init__(bundle_path=bundle_path)


if __name__ == '__main__':
    clf = SD3Classifier()
    print("[+] vidio-health-trainer sd3 공식 프로덕션 분류기(v4.0 RTMPose Optimized) 로딩 완료!")
    print(f"  - Model: {clf.model_name} ({clf.defect_name})")
    print(f"  - Binary Threshold: {clf.binary_threshold:.3f}")
    print(f"  - View Thresholds: {clf.view_thresholds}")

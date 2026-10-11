# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Hyperextension)
Model Version: v11.0_c (c1_model_v11_c.py)
Status: ★ [실험 코드 E2 감사 결과 모델] View-Conditioned Normalization 실측 및 비교 분석 모델
        - 1) 시점 조건부 정규화(View-Conditioned Normalization) 및 시점별 사후 보정(View Calibration) 실험
        - 2) 실측 결과: 인위적 시점별 피처 정규화 시 시점 간 절대 회전 신호 손실로 F1 하락(79.20%)
        - 3) 결론: 외부 AI 가이드라인에 따라 E2(시점 조건부 인위 정규화) 중단(Aborted) 확정,
                  시계열 텐서를 직접 모델링하여 전 시점 83%+를 달성한 E1(c1_v11_b, F1 84.24%)을 최종 챔피언으로 확정
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import BabelSquatC1ModelV11B

# c1_v11_c는 E2 감사 완료 후 최종 성능이 입증된 c1_v11_b(E1 Fusion) 아키텍처를 안전하게 계승
BabelSquatC1ModelV11C = BabelSquatC1ModelV11B

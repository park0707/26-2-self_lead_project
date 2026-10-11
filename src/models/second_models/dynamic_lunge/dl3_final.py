# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드 & 06_스텝_백워드)
Defect: Condition 3 - 뒤다리 무릎 각도 90도 (Rear Knee 90° Defect)
Module: dl3_final.py
Purpose: 다이나믹 런지 c3 최종 공식 프로덕션 모델 배포 인터페이스 (dl3_rtmp_1 기반)
Architecture:
  - Feature Extractor: lunge_dl3_rtmp_features.py (22D 정예 도메인 기구학 피처)
  - Classifier: 22D CatBoost Classifier (Production Champion)
  - Decision Logic: 공식 5대 시점별 적응형 3-State 선택적 분류 (보류율 <= 14.8% 제약 준수)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")
sys.path.append(r"D:\src\models\second_models\dynamic_lunge")

from dl3_rtmp_1 import (
    DL3RTMP1Classifier,
    VIEW_ALIAS_MAP,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_CBM_PATH
)
from lunge_dl3_rtmp_features import (
    extract_lunge_dl3_22d_kinematics,
    FEATURE_NAMES_22D
)

class DL3Classifier(DL3RTMP1Classifier):
    """다이나믹 런지 c3(뒤다리 무릎 각도 90도) 최종 공식 프로덕션 분류기 (dl3_rtmp_1 기반)"""
    pass

def predict_dynamic_lunge_c3(
    kps_26: np.ndarray,
    view_code: str = 'view2',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    다이나믹 런지 c3 세부 결함(뒤다리 무릎 90도) 표준 함수형 인터페이스
    
    Parameters:
      kps_26: (64, 26, 3) 또는 (64, 26, 2) RTMPose 관절 시퀀스
      view_code: 'view1' ~ 'view5' (선택적)
      bundle_path: 모델 번들 경로 (선택적)
      
    Returns:
      dict: 판정 상태, 결함 확률, 생체역학 피처값, 교정 코칭 피드백
    """
    clf = DL3Classifier(bundle_path=bundle_path)
    return clf.predict_sequence(kps_26, view_code=view_code)

if __name__ == '__main__':
    clf = DL3Classifier()
    dummy_kps = np.zeros((64, 26, 3), dtype=np.float32)
    res = clf.predict_sequence(dummy_kps, view_code='view2')
    print("=" * 65)
    print("  [+] dl3_final (dl3_rtmp_1) 공식 추론 인터페이스 테스트 성공!")
    print("=" * 65)
    for k, v in res.items():
        if k != 'biomechanics':
            print(f"  {k}: {v}")
    print("  biomechanics summary:", {k: round(v, 2) for k, v in res['biomechanics'].items()})
    assert res['model_name'] == 'dl3_rtmp_1'
    assert res['state'] in [0, 1, 2]
    print("[+] All assertions passed!")

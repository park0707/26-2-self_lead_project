# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/config.py
Purpose: 웹 서비스 전역 설정, 공식 5대 시점 매핑, 8대 운동 메타데이터 및 기본 임계값
"""

import os
import sys

# NumPy 2.x <-> 1.x Pickle 호환성 Shim (클라우드 서빙 환경 무결성 보장)
try:
    import numpy as np
    if not hasattr(np, '_core') and hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = getattr(np.core, 'numeric', np.core)
        sys.modules['numpy._core.multiarray'] = getattr(np.core, 'multiarray', np.core)
        sys.modules['numpy._core.umath'] = getattr(np.core, 'umath', np.core)
    elif hasattr(np, '_core') and not hasattr(np, 'core'):
        sys.modules['numpy.core'] = np._core
        sys.modules['numpy.core.numeric'] = getattr(np._core, 'numeric', np._core)
        sys.modules['numpy.core.multiarray'] = getattr(np._core, 'multiarray', np._core)
        sys.modules['numpy.core.umath'] = getattr(np._core, 'umath', np._core)
except Exception:
    pass

# 웹 서비스 루트 디렉터리
WEB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.join(WEB_ROOT, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))

WEIGHTS_DIR = os.path.join(WEB_ROOT, "weights")
STAGE1_BUNDLE_PATH = os.path.join(WEIGHTS_DIR, "stage1_v6_production_bundle.pkl")

# 공식 5대 카메라 시점 체계 (AGENTS.md 제4조)
VIEW_ALIAS_MAP = {
    "시점 A": "view5", "시점A": "view5", "A": "view5", "view5": "view5",
    "시점 B": "view2", "시점B": "view2", "B": "view2", "view2": "view2",
    "시점 C": "view3", "시점C": "view3", "C": "view3", "view3": "view3",
    "시점 D": "view4", "시점D": "view4", "D": "view4", "view4": "view4",
    "시점 E": "view1", "시점E": "view1", "E": "view1", "view1": "view1"
}
VIEW_CODE_MAP = VIEW_ALIAS_MAP


VIEW_METADATA = {
    "view5": {"code": "시점 A", "name": "우측 후방 (45도)", "desc": "Right Back 45°", "best_for": ["바벨 스쿼트", "사이드 런지"]},
    "view2": {"code": "시점 B", "name": "우측 전방 (45도)", "desc": "Right Front 45°", "best_for": ["사이드 런지 sl2", "바벨 데드리프트"]},
    "view3": {"code": "시점 C", "name": "정면 (0도)", "desc": "Front View 0°", "best_for": ["무릎 외반", "골반 수평", "발끝 정렬"]},
    "view4": {"code": "시점 D", "name": "좌측 전방 (45도)", "desc": "Left Front 45°", "best_for": ["런지 앞무릎 각도", "바벨 데드리프트"]},
    "view1": {"code": "시점 E", "name": "좌측 후방 (45도)", "desc": "Left Back 45°", "best_for": ["척추 중립성", "바벨 궤적"]}
}

# 8대 공식 운동 종목 메타데이터
EXERCISE_CATALOG = {
    0: {"id": 0, "code": "01_squat", "name": "바벨 스쿼트", "short": "스쿼트", "defects_count": 4},
    1: {"id": 1, "code": "02_deadlift", "name": "바벨 데드리프트", "short": "데드리프트", "defects_count": 5},
    2: {"id": 2, "code": "03_stiff_deadlift", "name": "바벨 스티프 데드리프트", "short": "스티프데드", "defects_count": 3},
    3: {"id": 3, "code": "04_babel_lunge", "name": "바벨 런지", "short": "바벨런지", "defects_count": 5},
    4: {"id": 4, "code": "05_dynamic_lunge", "name": "다이나믹 런지", "short": "다이나믹런지", "defects_count": 5},
    5: {"id": 5, "code": "06_side_lunge", "name": "사이드 런지", "short": "사이드런지", "defects_count": 5},
    6: {"id": 6, "code": "07_cross_lunge", "name": "크로스 런지", "short": "크로스런지", "defects_count": 3},
    7: {"id": 7, "code": "08_good_morning", "name": "굿모닝", "short": "굿모닝", "defects_count": 3}
}

# 3-State 상태 레이블
STATE_LABELS = {
    0: "정상 (Clean)",
    1: "결함 (Defect)",
    2: "주의/보류 (Caution)"
}

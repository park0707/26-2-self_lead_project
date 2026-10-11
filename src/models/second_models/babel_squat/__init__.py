# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트(Babel Squat) 세부 결함 전문 분류 모델 패키지
- c1: 척추의 중립 (Spine Neutrality) -> BabelSquatC1ModelV12RTMP (BabelSquatC1Model)
- c2: 고개 정면 (Head Forward / Gaze) -> BabelSquatC2ModelV9RTMP (BabelSquatC2Model)
- c3: 발과 무릎의 방향 일치 (Knee Valgus / Alignment) -> BabelSquatC3ModelV7RTMP (BabelSquatC3Model)
- c4: 발바닥 지면 고정 (Foot Ground Fixation) -> BabelSquatC4ModelV6RTMP (BabelSquatC4Model)
- 통합 진단 챔피언 엔진: BabelSquatSubmodelsRTMP (BabelSquatIntegratedEngine, BabelSquatEngine)
"""

# 최신 RTMPose 공식 챔피언 모델 직접 임포트
from .c1_model_v12_rtmp import BabelSquatC1ModelV12RTMP, BabelSquatC1ModelV12RTMP as BabelSquatC1Model
from .c2_model_v9_rtmp import BabelSquatC2ModelV9RTMP, BabelSquatC2ModelV9RTMP as BabelSquatC2Model
from .c3_model_v7_rtmp import BabelSquatC3ModelV7RTMP, BabelSquatC3ModelV7RTMP as BabelSquatC3Model
from .c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP, BabelSquatC4ModelV6RTMP as BabelSquatC4Model

from .squat_submodels_rtmp import (
    BabelSquatSubmodelsRTMP,
    BabelSquatSubmodelsRTMP as BabelSquatIntegratedEngine,
    BabelSquatSubmodelsRTMP as BabelSquatEngine
)

# 레거시 모델 지연 로딩 매핑 (초기 임포트 지연 및 메모리 낭비 원천 차단)
_LEGACY_IMPORT_MAP = {
    'BabelSquatC1ModelV1': ('.c1_model_v1', 'BabelSquatC1ModelV1'),
    'BabelSquatC1ModelV10': ('.c1_model_v10', 'BabelSquatC1ModelV10'),
    'BabelSquatC2ModelV3': ('.c2_model_v3', 'BabelSquatC2ModelV3'),
    'BabelSquatC2ModelV6': ('.c2_model_v6', 'BabelSquatC2ModelV6'),
    'BabelSquatC3ModelV2': ('.c3_model_v2', 'BabelSquatC3ModelV2'),
    'BabelSquatC3ModelV3': ('.c3_model_v3', 'BabelSquatC3ModelV3'),
    'BabelSquatC3ModelV6': ('.c3_model_v6', 'BabelSquatC3ModelV6'),
    'BabelSquatC4ModelV1': ('.c4_model_v1', 'BabelSquatC4ModelV1'),
    'BabelSquatC4ModelV2': ('.c4_model_v2', 'BabelSquatC4ModelV2'),
    'BabelSquatC4ModelV3': ('.c4_model_v3', 'BabelSquatC4ModelV3'),
    'BabelSquatC4ModelV4': ('.c4_model_v4', 'BabelSquatC4ModelV4'),
    'BabelSquatIntegratedEngineV11': ('.babel_squat_integrated_v11', 'BabelSquatIntegratedEngineV11'),
}

__all__ = [
    # c1
    'BabelSquatC1ModelV12RTMP', 'BabelSquatC1Model', 'BabelSquatC1ModelV1', 'BabelSquatC1ModelV10',
    # c2
    'BabelSquatC2ModelV9RTMP', 'BabelSquatC2Model', 'BabelSquatC2ModelV3', 'BabelSquatC2ModelV6',
    # c3
    'BabelSquatC3ModelV7RTMP', 'BabelSquatC3Model', 'BabelSquatC3ModelV2', 'BabelSquatC3ModelV3', 'BabelSquatC3ModelV6',
    # c4
    'BabelSquatC4ModelV6RTMP', 'BabelSquatC4Model', 'BabelSquatC4ModelV1', 'BabelSquatC4ModelV2', 'BabelSquatC4ModelV3', 'BabelSquatC4ModelV4',
    # Integrated Champions
    'BabelSquatSubmodelsRTMP', 'BabelSquatIntegratedEngine', 'BabelSquatEngine',
    # Legacy V11
    'BabelSquatIntegratedEngineV11'
]


def __getattr__(name: str):
    """
    레거시 모델에 대한 지연 로딩(Lazy Import)을 지원하여
    패키지 임포트 속도를 극대화하고 하위 호환성을 100% 보장합니다.
    """
    if name in _LEGACY_IMPORT_MAP:
        mod_name, attr_name = _LEGACY_IMPORT_MAP[name]
        import importlib
        module = importlib.import_module(mod_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

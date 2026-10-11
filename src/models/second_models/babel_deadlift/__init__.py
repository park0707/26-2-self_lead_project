"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Package: src.models.second_models.babel_deadlift
Official Production Model:
  - d1 (척추의 중립 위반): BabelDeadliftD1Model (v4.3 Diet - 25D 정예 다이어트 Tri-Ensemble)
  - d2 (발과 무릎의 방향 일치): BabelDeadliftD2Model (v6.0 Two-Tier Final - 2계층 모호성 정제 10D 초정예 Dual-GBDT + Tiny TCN)
  - d3 (발바닥 지면 고정): BabelDeadliftD3Model (v5.3-Opt - 17D 정예 다이어트 Tri-Ensemble + 57D ST-SeqNet)
"""

from .d1_model_v4_3_diet import (
    BabelDeadliftD1Model,
    BabelDeadliftD1ModelV4_3,
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_25_D1_V4_3
)

from .d2_model_v6_twotier import (
    BabelDeadliftD2ModelV6
)

from .d2_model_v5_final import (
    BabelDeadliftD2Model,
    BabelDeadliftD2ModelV5,
    BabelDeadliftD2ModelV5_2,
    extract_d2_v5_final_features,
    FEATURE_NAMES_10_D2,
    VIEW_ALIAS_MAP as D2_VIEW_ALIAS_MAP
)


from .d3_model_v7_0 import (
    BabelDeadliftD3ModelV7_0,
    FEATURE_NAMES_36_D3,
    OFFICIAL_V7_VIEW_THRESHOLDS as D3_OFFICIAL_VIEW_THRESHOLDS
)
BabelDeadliftD3Model = BabelDeadliftD3ModelV7_0

__all__ = [
    # d1
    "BabelDeadliftD1Model",
    "BabelDeadliftD1ModelV4_3",
    "extract_d1_v4_3_diet_features_25d",
    "extract_morphology_and_crosstalk_ratios",
    "MorphologyResidualizerV4_3",
    "extract_temporal_24d_with_spine",
    "FEATURE_NAMES_25_D1_V4_3",
    # d2
    "BabelDeadliftD2Model",
    "BabelDeadliftD2ModelV5",
    "BabelDeadliftD2ModelV5_2",
    "BabelDeadliftD2ModelV6",
    "extract_d2_v5_final_features",
    "FEATURE_NAMES_10_D2",
    "D2_VIEW_ALIAS_MAP",
    # d3
    "BabelDeadliftD3Model",
    "BabelDeadliftD3ModelV7_0",
    "FEATURE_NAMES_36_D3",
    "D3_OFFICIAL_VIEW_THRESHOLDS"
]



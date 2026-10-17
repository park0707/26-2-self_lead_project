# -*- coding: utf-8 -*-
"""
vidio-health-trainer: Domain Feature Hub (src.features)
"""

__all__ = ["FeatureRegistry"]


def __getattr__(name: str):
    if name == "FeatureRegistry":
        from src.features.registry import FeatureRegistry
        return FeatureRegistry
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


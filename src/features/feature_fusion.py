import numpy as np
from typing import Dict, List, Tuple, Any

class FeatureFusionEngine:
    """AI Hub 2D 정밀 라벨과 MediaPipe 3D 공간 피처를 결합하는 융합 엔진 (연구 계획서 조건 2)"""

    @staticmethod
    def fuse_format_a_features(
        feat_a_2d: np.ndarray,
        feat_a_3d: np.ndarray,
        feat_names_2d: List[str],
        feat_names_3d: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Format A 융합 피처 생성: 97D (2D) + 161D (3D) = 258D
        """
        fused_vals = np.concatenate([feat_a_2d, feat_a_3d], axis=-1).astype(np.float32)
        
        prefixed_2d = [f"2d_{name}" for name in feat_names_2d]
        prefixed_3d = [f"3d_{name}" for name in feat_names_3d]
        fused_names = prefixed_2d + prefixed_3d
        
        return fused_vals, fused_names

    @staticmethod
    def fuse_format_b_timeseries(
        feat_b_2d: np.ndarray,
        feat_b_3d: np.ndarray
    ) -> np.ndarray:
        """
        Format B 융합 시계열 텐서 생성: (64, 69) + (64, 92) = (64, 161)
        """
        fused_tensor = np.concatenate([feat_b_2d, feat_b_3d], axis=-1).astype(np.float32)
        return fused_tensor

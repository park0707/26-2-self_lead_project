import numpy as np
from typing import Dict, Tuple, List
from src.features.joint_angles import JointAngleExtractor
from src.features.alignment_metrics import AlignmentMetricsExtractor
from src.core.types import KeypointSequence

class FeatureAggregator:
    """Format A (정적 요약 통계량) 및 Format B (시계열 텐서) 빌더"""

    @classmethod
    def build_features(cls, seq: KeypointSequence) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """단일 KeypointSequence에서 Format A (1D 통계 벡터)와 Format B (C, T 시계열) 동시 추출
        
        Returns:
            format_a: np.ndarray shape (num_summary_features,)
            format_b: np.ndarray shape (num_channels, T)
            feature_names: List of string names for format_a
        """
        kps = seq.keypoints # (T, 15, 3)
        
        # 1. 관절 각도 추출
        j_angles = JointAngleExtractor.extract_all_joint_angles(kps)
        
        # 2. 정렬 및 비대칭 지표 추출
        alignments = AlignmentMetricsExtractor.extract_all_alignments(kps, j_angles)
        
        # 모든 시계열 채널 결합 (Dictionary)
        all_signals = {**j_angles, **alignments}
        channel_names = sorted(list(all_signals.keys()))
        
        # Format B: (C, T) 시계열 텐서
        T = kps.shape[0]
        C = len(channel_names)
        format_b = np.zeros((C, T), dtype=np.float32)
        for c_idx, name in enumerate(channel_names):
            format_b[c_idx, :] = all_signals[name]
            
        # Format A: 클립 수준 통계 요약 (Min, Max, Mean, Std, Range)
        stat_values = []
        feature_names = []
        for name in channel_names:
            sig = all_signals[name]
            stat_min = float(np.min(sig))
            stat_max = float(np.max(sig))
            stat_mean = float(np.mean(sig))
            stat_std = float(np.std(sig))
            stat_range = stat_max - stat_min
            
            stat_values.extend([stat_min, stat_max, stat_mean, stat_std, stat_range])
            feature_names.extend([
                f"{name}_min", f"{name}_max", f"{name}_mean", f"{name}_std", f"{name}_range"
            ])
            
        format_a = np.array(stat_values, dtype=np.float32)
        return format_a, format_b, feature_names

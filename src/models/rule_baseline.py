import numpy as np
from typing import Dict, Any

class RuleBasedPostureClassifier:
    """생체역학 임계값 기반 규칙 분류기 (머신러닝 도입 효과 정량화를 위한 하한 대조군 Baseline)"""
    
    def __init__(self, exercise_name: str = '바벨 스쿼트'):
        self.exercise_name = exercise_name
        
    def predict(self, format_a_features: np.ndarray, feature_names: list) -> int:
        """주요 역학적 임계값 초과 시 오류(0), 만족 시 정상(1) 판정"""
        feat_dict = dict(zip(feature_names, format_a_features))
        
        # 스쿼트/런지 기준 규칙:
        # 1. 무릎 각도 최소값이 너무 얕거나(>120도) 과도하지 않아야 함
        # 2. 체간 경사각 최대값이 과도하게 숙여지면(>60도) 오류
        # 3. Knee Valgus(안쪽 모임)이 심하면(<160도) 오류
        
        torso_max = feat_dict.get('torso_inclination_max', 30.0)
        valgus_l_min = feat_dict.get('knee_valgus_l_min', 175.0)
        valgus_r_min = feat_dict.get('knee_valgus_r_min', 175.0)
        
        if torso_max > 55.0: # 허리 과도한 숙임
            return 0
        if valgus_l_min < 165.0 or valgus_r_min < 165.0: # 무릎 안쪽 모임
            return 0
            
        return 1

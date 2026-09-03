import os
from typing import Dict, Any, Optional
import numpy as np

DUAL_SPLITS_DIR = 'd:/Data/피처_데이터/Dual_Splits'

class Stage2DatasetLoader:
    """2단계: 정상(1) vs 오류(0) 판별용 2차원 복합 분할(Dual_Splits) 데이터 로더"""

    @classmethod
    def load_split_data(cls, split_filename: str) -> Dict[str, Any]:
        npz_path = os.path.join(DUAL_SPLITS_DIR, split_filename)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"분할 파일을 찾을 수 없습니다: {npz_path}")

        d = np.load(npz_path, allow_pickle=True)
        X_a = d['X_Format_A']       # (N, 97)
        X_b = d['X_Format_B']       # (N, 64, 69)
        y = d['y'].astype(np.int64) # (N,) - 0 (오류), 1 (정상)
        sequence_ids = d['sequence_ids']
        subjects = d['subjects']
        views = d['views']
        feat_names = d['feature_names'] if 'feature_names' in d else []

        # 불균형 비율 계산
        normal_cnt = int(np.sum(y == 1))
        error_cnt = int(np.sum(y == 0))
        scale_pos_weight = float(error_cnt) / max(1.0, float(normal_cnt))

        return {
            'X_Format_A': X_a,
            'X_Format_B': X_b,
            'y': y,
            'sequence_ids': sequence_ids,
            'subjects': subjects,
            'views': views,
            'feature_names': feat_names,
            'normal_count': normal_cnt,
            'error_count': error_cnt,
            'scale_pos_weight': scale_pos_weight
        }

    @classmethod
    def load_all_dual_splits(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'train': cls.load_split_data('Dual_Train.npz'),
            'valid': cls.load_split_data('Dual_Valid.npz'),
            'test_strict': cls.load_split_data('Dual_Test_Strict.npz'),
            'test_unseen_subj': cls.load_split_data('Dual_Test_Unseen_Subj.npz'),
            'test_unseen_view': cls.load_split_data('Dual_Test_Unseen_View.npz')
        }

import os
import glob
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

RAW_LABELS_BASE = 'd:/Data/01_raw_data/labels_json'
DUAL_SPLITS_DIR = 'd:/Data/피처_데이터/Dual_Splits'

EXERCISE_NAMES_KO = [
    '01_바벨_스쿼트',
    '02_바벨_데드리프트',
    '03_바벨_스티프_데드리프트',
    '04_바벨_런지',
    '05_스텝_포워드_다이나믹_런지',
    '06_스텝_백워드_다이나믹_런지',
    '07_사이드_런지',
    '08_크로스_런지',
    '09_굿모닝'
]

class Stage1DatasetLoader:
    """1단계 9종 운동 분류용 2차원 복합 분할(Dual_Splits) 데이터 로더"""

    _seq_to_ex_idx: Optional[Dict[str, int]] = None

    @classmethod
    def _get_seq_to_ex_map(cls) -> Dict[str, int]:
        """시퀀스 ID -> 9개 운동 인덱스(0~8) 매핑 테이블 캐싱"""
        if cls._seq_to_ex_idx is None:
            cls._seq_to_ex_idx = {}
            if os.path.exists(RAW_LABELS_BASE):
                folders = sorted(os.listdir(RAW_LABELS_BASE))
                for idx, folder in enumerate(folders):
                    p = os.path.join(RAW_LABELS_BASE, folder, '2D')
                    if os.path.isdir(p):
                        for fn in os.listdir(p):
                            sid = os.path.splitext(fn)[0]
                            cls._seq_to_ex_idx[sid] = idx
        return cls._seq_to_ex_idx

    @classmethod
    def load_split_data(cls, split_filename: str) -> Dict[str, Any]:
        """특정 분할 파일 로드 및 9종 운동 정수 라벨(y_action) 매핑 반환"""
        npz_path = os.path.join(DUAL_SPLITS_DIR, split_filename)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"분할 파일을 찾을 수 없습니다: {npz_path}")

        d = np.load(npz_path, allow_pickle=True)
        X_a = d['X_Format_A']       # (N, 97)
        X_b = d['X_Format_B']       # (N, 64, 69)
        y_binary = d['y']           # (N,) - 정상(1)/오류(0)
        sequence_ids = d['sequence_ids']
        subjects = d['subjects']
        views = d['views']
        feat_names = d['feature_names'] if 'feature_names' in d else []

        ex_map = cls._get_seq_to_ex_map()
        y_action = []

        for sid in sequence_ids:
            base_sid = '_'.join(str(sid).split('_')[:-1]) # remove _viewX
            if base_sid in ex_map:
                y_action.append(ex_map[base_sid])
            else:
                # fallback parsing
                matched = 0
                for ex_i, ex_name in enumerate(EXERCISE_NAMES_KO):
                    if ex_name in str(sid):
                        matched = ex_i
                        break
                y_action.append(matched)

        y_action = np.array(y_action, dtype=np.int64)

        return {
            'X_Format_A': X_a,
            'X_Format_B': X_b,
            'y_action': y_action,
            'y_binary': y_binary,
            'sequence_ids': sequence_ids,
            'subjects': subjects,
            'views': views,
            'feature_names': feat_names,
            'exercise_names': EXERCISE_NAMES_KO,
            'num_classes': len(EXERCISE_NAMES_KO)
        }

    @classmethod
    def load_all_dual_splits(cls) -> Dict[str, Dict[str, Any]]:
        """2차원 복합 분할의 5대 데이터셋(Train, Valid, Test_Strict, Unseen_Subj, Unseen_View) 일괄 로드"""
        splits = {
            'train': cls.load_split_data('Dual_Train.npz'),
            'valid': cls.load_split_data('Dual_Valid.npz'),
            'test_strict': cls.load_split_data('Dual_Test_Strict.npz'),
            'test_unseen_subj': cls.load_split_data('Dual_Test_Unseen_Subj.npz'),
            'test_unseen_view': cls.load_split_data('Dual_Test_Unseen_View.npz')
        }
        return splits

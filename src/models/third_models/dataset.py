import os
import glob
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

RAW_LABELS_BASE = 'd:/Data/01_raw_data/labels_json'
DUAL_SPLITS_DIR = 'd:/Data/피처_데이터/Dual_Splits'

class Stage3DatasetLoader:
    """3단계: 세부 결함(Conditions) 다중 라벨 진단용 데이터 로더"""

    _seq_info_map: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def _build_conditions_cache(cls):
        if cls._seq_info_map is None:
            cls._seq_info_map = {}
            if os.path.exists(RAW_LABELS_BASE):
                folders = sorted(os.listdir(RAW_LABELS_BASE))
                for folder in folders:
                    folder_path = os.path.join(RAW_LABELS_BASE, folder, '2D')
                    if not os.path.isdir(folder_path):
                        continue
                    for fn in os.listdir(folder_path):
                        if not fn.endswith('.json'):
                            continue
                        sid = os.path.splitext(fn)[0]
                        jf_path = os.path.join(folder_path, fn)
                        try:
                            with open(jf_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            t_info = data.get('type_info', {})
                            raw_conds = t_info.get('conditions', [])
                            cond_names = []
                            cond_vals = []
                            for c in raw_conds:
                                if isinstance(c, dict):
                                    cond_names.append(c.get('condition', ''))
                                    # False = 감점 발생(결함), True = 정상 수행
                                    # 결함 진단이므로 1=결함(Defect), 0=정상(No defect)으로 통일
                                    is_defect = 0 if bool(c.get('value', False)) else 1
                                    cond_vals.append(is_defect)
                                elif isinstance(c, bool):
                                    is_defect = 0 if c else 1
                                    cond_vals.append(is_defect)

                            cls._seq_info_map[sid] = {
                                'folder': folder,
                                'exercise': t_info.get('exercise', ''),
                                'cond_names': cond_names,
                                'cond_vals': cond_vals
                            }
                        except Exception:
                            pass

    @classmethod
    def load_exercise_split(cls, exercise_folder: str = '01_바벨_스쿼트', split_filename: str = 'Dual_Train.npz') -> Dict[str, Any]:
        """특정 운동 폴더에 대한 다중 라벨 결함 데이터 로드"""
        cls._build_conditions_cache()
        npz_path = os.path.join(DUAL_SPLITS_DIR, split_filename)
        d = np.load(npz_path, allow_pickle=True)

        X_a = d['X_Format_A']
        X_b = d['X_Format_B']
        sequence_ids = d['sequence_ids']

        indices = []
        Y_list = []
        cond_names_ref = None

        for idx, sid in enumerate(sequence_ids):
            base_sid = '_'.join(str(sid).split('_')[:-1])
            info = cls._seq_info_map.get(base_sid)
            if info and info['folder'] == exercise_folder:
                indices.append(idx)
                Y_list.append(info['cond_vals'])
                if cond_names_ref is None and info['cond_names']:
                    cond_names_ref = info['cond_names']

        indices = np.array(indices, dtype=np.int64)
        X_a_sub = X_a[indices] if len(indices) > 0 else np.zeros((0, 97))
        X_b_sub = X_b[indices] if len(indices) > 0 else np.zeros((0, 64, 69))
        Y_sub = np.array(Y_list, dtype=np.int64) if len(Y_list) > 0 else np.zeros((0, len(cond_names_ref or [])))

        return {
            'exercise': exercise_folder,
            'cond_names': cond_names_ref or [],
            'num_conditions': len(cond_names_ref or []),
            'X_Format_A': X_a_sub,
            'X_Format_B': X_b_sub,
            'Y_multi': Y_sub,
            'num_samples': len(indices)
        }

    @classmethod
    def load_all_splits_for_exercise(cls, exercise_folder: str = '01_바벨_스쿼트') -> Dict[str, Dict[str, Any]]:
        return {
            'train': cls.load_exercise_split(exercise_folder, 'Dual_Train.npz'),
            'valid': cls.load_exercise_split(exercise_folder, 'Dual_Valid.npz'),
            'test_strict': cls.load_exercise_split(exercise_folder, 'Dual_Test_Strict.npz'),
            'test_unseen_subj': cls.load_exercise_split(exercise_folder, 'Dual_Test_Unseen_Subj.npz'),
            'test_unseen_view': cls.load_exercise_split(exercise_folder, 'Dual_Test_Unseen_View.npz')
        }

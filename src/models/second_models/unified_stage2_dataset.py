# =========================================================================================================
# [모듈 설명]: 8종 운동 통합 1-Stage 데이터셋 로더 (unified_stage2_dataset.py)
# [데이터셋]: Condition 5 단일 횟수(16F) 22,562개 샘플 + conditions_cache.pkl 전수 100% 매핑
# [기능]:
#   - 8개 운동(스쿼트, 데드리프트, 스티프, 바벨런지, 다이나믹런지, 사이드런지, 크로스런지, 굿모닝)
#   - 런지 계열 통합(05_다이나믹_런지 = 포워드+백워드)
#   - Format A 161차원 피처, subjects(피험자 격리용), y_normal(정상=1, 오류=0), Y_defects(K개 결함)
# =========================================================================================================

import os
import glob
import pickle
from typing import Dict, Any, List
import numpy as np

MASTER_NPZ_PATH = 'd:/Data/전처리_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz'
CACHE_PKL_PATH = 'd:/Data/피처_데이터/Dual_Splits/conditions_cache.pkl'

class UnifiedStage2DatasetLoader:
    """8종 운동 통합 1-Stage 데이터셋 로더"""

    _cache_data = None
    _master_data = None

    @classmethod
    def _init_master_and_cache(cls):
        if cls._master_data is None:
            if not os.path.exists(MASTER_NPZ_PATH):
                matches = glob.glob('d:/Data/*/Condition5_Repetition_16F/*.npz')
                if not matches:
                    raise FileNotFoundError("Condition5 마스터 데이터셋을 찾을 수 없습니다.")
                p_npz = matches[0]
            else:
                p_npz = MASTER_NPZ_PATH
            cls._master_data = np.load(p_npz, allow_pickle=True)

        if cls._cache_data is None:
            if not os.path.exists(CACHE_PKL_PATH):
                raise FileNotFoundError(f"Conditions 캐시를 찾을 수 없습니다: {CACHE_PKL_PATH}")
            with open(CACHE_PKL_PATH, 'rb') as f:
                cls._cache_data = pickle.load(f)

    @classmethod
    def get_exercise_names(cls) -> List[str]:
        cls._init_master_and_cache()
        return list(cls._master_data['exercise_names'])

    @classmethod
    def load_exercise_data(cls, exercise_idx: int) -> Dict[str, Any]:
        """특정 운동(0~7)에 대한 피처, 정상라벨, K개 세부결함 라벨 로드"""
        cls._init_master_and_cache()
        d = cls._master_data
        ex_names = list(d['exercise_names'])
        ex_name = ex_names[exercise_idx]

        mask = (d['y_action'] == exercise_idx)
        X_sub = d['X_Format_A'][mask].astype(np.float32)
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

        subjects = d['subjects'][mask].astype(np.int32)
        y_normal = d['y_binary'][mask].astype(np.int32) # 1=정상, 0=오류
        base_sids = d['base_sequence_ids'][mask]

        # 해당 운동의 조건 명칭 목록 추출
        cond_names = None
        for b in base_sids:
            b_str = str(b)
            if b_str in cls._cache_data:
                cond_names = cls._cache_data[b_str]['cond_names']
                break

        if cond_names is None:
            raise ValueError(f"운동 [{ex_name}]의 조건을 캐시에서 찾을 수 없습니다.")

        K = len(cond_names)
        Y_defects = np.zeros((len(base_sids), K), dtype=np.int32)

        for i, b in enumerate(base_sids):
            b_str = str(b)
            if b_str in cls._cache_data:
                item = cls._cache_data[b_str]
                c_vals = item.get('cond_vals', [])
                for j in range(min(K, len(c_vals))):
                    Y_defects[i, j] = int(c_vals[j])

        return {
            'exercise_index': exercise_idx,
            'exercise_name': ex_name,
            'X': X_sub,
            'subjects': subjects,
            'y_normal': y_normal,
            'Y_defects': Y_defects,
            'defect_names': cond_names,
            'num_defects': K,
            'total_samples': len(X_sub),
            'normal_count': int((y_normal == 1).sum()),
            'error_count': int((y_normal == 0).sum())
        }

if __name__ == '__main__':
    names = UnifiedStage2DatasetLoader.get_exercise_names()
    print(f"통합 로더 검증 - 총 {len(names)}종 운동:")
    for i, name in enumerate(names):
        data = UnifiedStage2DatasetLoader.load_exercise_data(i)
        print(f"  [{i}] {name:<20}: 샘플 {data['total_samples']:>5}개 (정상 {data['normal_count']:>4}, 오류 {data['error_count']:>4}) | 결함 {data['num_defects']}개: {data['defect_names']}")

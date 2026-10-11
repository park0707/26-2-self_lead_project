# =========================================================================================================
# [모듈 설명]: 바벨 스쿼트 1-Stage 통합 파일럿 전용 데이터셋 로더
# [데이터셋]: Condition 5 마스터 단일 횟수(16F) 데이터셋 + AI-Hub labels_json 4대 세부 결함 매핑
# [출력 구성]:
#   - X: Format A 161차원 기하 통계 요약 피처 (1,974 x 161)
#   - subjects: 피험자 ID (피험자 격리 5-Fold용)
#   - y_normal: 완전 정상(1, 129개) vs 오류(0, 1,845개) [불균형 1:14.3]
#   - Y_defects: 4대 세부 결함 이진 행렬 (1,974 x 4) [각각 ~50% 균형]
#     * c1: 척추의 중립 위반 (48.43%)
#     * c2: 고개 정면 위반 (50.35%)
#     * c3: 발과 무릎 방향 일치 위반 (49.95%)
#     * c4: 발바닥 지면 고정 위반 (49.34%)
# =========================================================================================================

import os
import glob
import json
import pickle
from typing import Dict, Any, Tuple
import numpy as np

DEFECT_NAMES = [
    '척추의 중립',
    '고개 정면',
    '발과 무릎의 방향 일치',
    '발바닥 지면 고정'
]

class SquatPilotDatasetLoader:
    """바벨 스쿼트 1-Stage 통합 진단 전용 데이터 로더"""

    @classmethod
    def _find_master_npz(cls) -> str:
        matches = glob.glob('d:/Data/*/Condition5_Repetition_16F/*.npz')
        if matches:
            return matches[0]
        matches2 = glob.glob('d:/Data/*/*/*Condition5*/*.npz')
        if matches2:
            return matches2[0]
        raise FileNotFoundError("Condition5 마스터 데이터셋을 찾을 수 없습니다.")

    @classmethod
    def _find_squat_json_dir(cls) -> str:
        base = 'd:/Data/01_raw_data/labels_json'
        if os.path.exists(base):
            for f in os.listdir(base):
                if '스쿼트' in f or '01' in f:
                    p2d = os.path.join(base, f, '2D')
                    if os.path.exists(p2d):
                        return p2d
        # glob fallback
        matches = glob.glob('d:/Data/01_raw_data/labels_json/*01*/*/2D')
        if matches:
            return matches[0]
        matches2 = glob.glob('d:/Data/01_raw_data/labels_json/*01*/2D')
        if matches2:
            return matches2[0]
        raise FileNotFoundError("01_바벨_스쿼트 2D JSON 디렉토리를 찾을 수 없습니다.")

    @classmethod
    def load_dataset(cls) -> Dict[str, Any]:
        """바벨 스쿼트 1,974개 샘플 및 4대 결함 라벨 로드"""
        npz_file = cls._find_master_npz()
        d = np.load(npz_file, allow_pickle=True)
        sq_mask = (d['y_action'] == 0)

        X_all = d['X_Format_A'][sq_mask].astype(np.float32)
        subjects = d['subjects'][sq_mask].astype(np.int32)
        y_normal = d['y_binary'][sq_mask].astype(np.int32)
        base_sids = d['base_sequence_ids'][sq_mask]
        feature_names = d['feature_names']

        defect_map = cls._get_defect_map()

        Y_defects = np.zeros((len(base_sids), len(DEFECT_NAMES)), dtype=np.int32)
        for i, bsid in enumerate(base_sids):
            cmap = defect_map.get(str(bsid), {})
            for j, cname in enumerate(DEFECT_NAMES):
                Y_defects[i, j] = cmap.get(cname, 0)

        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'X': X_all,
            'subjects': subjects,
            'y_normal': y_normal,
            'Y_defects': Y_defects,
            'defect_names': DEFECT_NAMES,
            'feature_names': feature_names,
            'base_sids': base_sids,
            'total_samples': len(X_all),
            'normal_count': int((y_normal == 1).sum()),
            'error_count': int((y_normal == 0).sum())
        }

    @classmethod
    def _get_defect_map(cls) -> Dict[str, Dict[str, int]]:
        cache_path = os.path.join(os.path.dirname(cls._find_master_npz()), 'squat_defects_cache.pkl')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    if data and len(data) > 100:
                        return data
            except Exception:
                pass

        json_dir = cls._find_squat_json_dir()
        defect_map = {}
        for fn in os.listdir(json_dir):
            if not fn.endswith('.json'):
                continue
            stem = os.path.splitext(fn)[0]
            jf_path = os.path.join(json_dir, fn)
            try:
                with open(jf_path, 'r', encoding='utf-8') as f:
                    data = json.load(jf)
                conds = data.get('type_info', {}).get('conditions', [])
                cmap = {}
                for c in conds:
                    cname = c.get('condition', '')
                    cval = bool(c.get('value', False))
                    cmap[cname] = 0 if cval else 1
                defect_map[stem] = cmap
            except Exception:
                pass

        if len(defect_map) > 100:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(defect_map, f)
            except Exception:
                pass

        return defect_map

if __name__ == '__main__':
    ds = SquatPilotDatasetLoader.load_dataset()
    print("SquatPilotDatasetLoader 로드 완료:")
    print(f"  - X shape: {ds['X'].shape}")
    print(f"  - y_normal: 정상 {ds['normal_count']} / 오류 {ds['error_count']}")
    print(f"  - Y_defects shape: {ds['Y_defects'].shape}")
    for j, cname in enumerate(ds['defect_names']):
        pos_cnt = ds['Y_defects'][:, j].sum()
        print(f"    * [{cname}] 결함 {pos_cnt}개 ({pos_cnt/ds['total_samples']*100:.2f}%)")

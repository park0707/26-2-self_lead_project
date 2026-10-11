# -*- coding: utf-8 -*-
"""
NovelFlow 2단계 (Stage 2: 정상 vs 오류 판별) 횟수별 분할(Repetition) 데이터 로더
- 데이터 소스: d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz
- 단위: 단일 횟수(Repetition, 16프레임) 단위 샘플 (총 22,562개)
- 타깃 라벨 y: 1 (정상 자세) vs 0 (오류 자세)
"""

import os
from typing import Dict, Any, Optional, List
import numpy as np

REP_MASTER_NPZ = 'd:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz'

class Stage2RepDatasetLoader:
    """2단계 자세 오류 판정 전용 16F 횟수별 마스터 데이터 로더"""

    @classmethod
    def load_master_data(
        cls,
        npz_path: str = REP_MASTER_NPZ,
        exercise_idx_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        16프레임 횟수별 마스터 데이터 로드
        :param npz_path: 마스터 npz 경로
        :param exercise_idx_filter: 특정 운동 종목만 필터링하고자 할 경우 (0~7)
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"횟수별 마스터 데이터를 찾을 수 없습니다: {npz_path}")

        d = np.load(npz_path, allow_pickle=True)
        X_a = d['X_Format_A']       # (N, 161)
        X_b = d['X_Format_B']       # (N, 16, 92)
        y_bin = d['y_binary'].astype(np.int64) # (N,) - 1(정상), 0(오류)
        y_act = d['y_action'].astype(np.int64) # (N,) - 0~7
        sids = d['sequence_ids']
        base_sids = d['base_sequence_ids'] if 'base_sequence_ids' in d else sids
        subjects = d['subjects']
        views = d['views']
        feat_names = d['feature_names'] if 'feature_names' in d else []

        # 특정 운동 필터링
        if exercise_idx_filter is not None:
            mask = (y_act == exercise_idx_filter)
            X_a = X_a[mask]
            X_b = X_b[mask]
            y_bin = y_bin[mask]
            y_act = y_act[mask]
            sids = sids[mask]
            base_sids = base_sids[mask]
            subjects = subjects[mask]
            views = views[mask]

        normal_cnt = int(np.sum(y_bin == 1))
        error_cnt = int(np.sum(y_bin == 0))
        scale_pos_weight = float(error_cnt) / max(1.0, float(normal_cnt))

        return {
            'X_Format_A': X_a,
            'X_Format_B': X_b,
            'y': y_bin,
            'y_action': y_act,
            'sequence_ids': sids,
            'base_sequence_ids': base_sids,
            'subjects': subjects,
            'views': views,
            'feature_names': feat_names,
            'normal_count': normal_cnt,
            'error_count': error_cnt,
            'scale_pos_weight': scale_pos_weight
        }

import os
import json
import random
from typing import List, Dict, Tuple
from src.core.types import KeypointSequence

class SubjectWiseSplitter:
    """피험자(Subject) 기반 데이터 누수 방지 Train/Val/Test 분할 생성기"""
    
    @staticmethod
    def split_sequences(
        sequences: List[KeypointSequence],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[KeypointSequence], List[KeypointSequence], List[KeypointSequence]]:
        """피험자 ID 단위로 그룹 분할 수행"""
        random.seed(seed)
        
        # 피험자별 시퀀스 그룹화
        subject_to_seqs: Dict[str, List[KeypointSequence]] = {}
        for seq in sequences:
            subj = seq.subject_id if seq.subject_id else 'Unknown_Subj'
            if subj not in subject_to_seqs:
                subject_to_seqs[subj] = []
            subject_to_seqs[subj].append(seq)
            
        subjects = list(subject_to_seqs.keys())
        random.shuffle(subjects)
        
        num_subjects = len(subjects)
        num_train = max(1, int(num_subjects * train_ratio))
        num_val = max(1, int(num_subjects * val_ratio))
        
        train_subjs = set(subjects[:num_train])
        val_subjs = set(subjects[num_train:num_train + num_val])
        test_subjs = set(subjects[num_train + num_val:])
        
        train_seqs = [s for subj in train_subjs for s in subject_to_seqs[subj]]
        val_seqs = [s for subj in val_subjs for s in subject_to_seqs[subj]]
        test_seqs = [s for subj in test_subjs for s in subject_to_seqs[subj]]
        
        return train_seqs, val_seqs, test_seqs

    @staticmethod
    def save_split_manifest(
        output_dir: str,
        exercise_name: str,
        train_seqs: List[KeypointSequence],
        val_seqs: List[KeypointSequence],
        test_seqs: List[KeypointSequence]
    ):
        """분할 메타데이터 manifest JSON 저장"""
        os.makedirs(output_dir, exist_ok=True)
        manifest = {
            'exercise_name': exercise_name,
            'train_count': len(train_seqs),
            'val_count': len(val_seqs),
            'test_count': len(test_seqs),
            'train_ids': [s.sequence_id for s in train_seqs],
            'val_ids': [s.sequence_id for s in val_seqs],
            'test_ids': [s.sequence_id for s in test_seqs],
            'train_labels': [s.label for s in train_seqs],
            'val_labels': [s.label for s in val_seqs],
            'test_labels': [s.label for s in test_seqs]
        }
        filename = f"{exercise_name.replace(' ', '_')}_split.json"
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Split manifest saved to {path}")

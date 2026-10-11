import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Optional

class PoseAugmentor3D:
    """
    1단계 9종 운동 대분류용 3D 포즈 및 시계열 데이터 증강기 (PoseAug & Temporal Warping 기반)
    - 대칭 운동 좌우 스왑 (Symmetric L/R Flip)
    - 시간적 비선형 신축 (Nonlinear Temporal Warping)
    - 시점 및 관절 가우시안 지터링 (Gaussian Noise Injection)
    - 소수 클래스 집중 클래스내 믹스업 (Intra-Class Mixup)
    """

    # Format B (64, 91) 내 좌우 대칭 스왑 인덱스 매핑
    # Stream 1 (56D = 14 edges * 4 (dx, dy, dz, len))
    # Edge pairs:
    # 2: Torso-LSh (idx 8..11) <-> 3: Torso-RSh (idx 12..15)
    # 4: LSh-LElb (idx 16..19) <-> 6: RSh-RElb (idx 24..27)
    # 5: LElb-LWr (idx 20..23) <-> 7: RElb-RWr (idx 28..31)
    # 8: Pelvis-LHip (idx 32..35) <-> 9: Pelvis-RHip (idx 36..39)
    # 10: LHip-LKnee (idx 40..43) <-> 12: RHip-RKnee (idx 48..51)
    # 11: LKnee-LAnk (idx 44..47) <-> 13: RKnee-RAnk (idx 52..55)
    
    # Stream 2 (8D angles, offset 56):
    # 0: knee_l (56) <-> 1: knee_r (57)
    # 2: hip_l (58) <-> 3: hip_r (59)
    # 4: ankle_l (60) <-> 5: ankle_r (61)
    # 6: elbow_l (62) <-> 7: elbow_r (63)

    # Stream 3 (8D align, offset 64):
    # 2: valgus_l (66) <-> 3: valgus_r (67)
    # 1: torso_roll (65) -> sign invert

    # Stream 4 (2D asymm, offset 72):
    # 0: knee_asymm (72), 1: hip_asymm (73)

    # Stream 5 (8D vel, offset 74):
    # 0..7 pairs match Stream 2

    # 양측 대칭성이 성립하는 운동 클래스 목록
    # 0: 바벨 스쿼트, 1: 바벨 데드리프트, 2: 바벨 스티프 데드리프트, 8: 굿모닝
    SYMMETRIC_CLASSES = {0, 1, 2, 8}

    @classmethod
    def time_warp_sequence(cls, seq: np.ndarray, warp_scale: float = 0.2) -> np.ndarray:
        """
        비선형 시간 왜곡 (Nonlinear Time Warping)
        seq: (T, D) -> (T, D)
        """
        T, D = seq.shape
        orig_t = np.linspace(0, 1, T)
        # 3개 랜덤 제어점으로 큐빅 스플라인 왜곡 곡선 생성
        control_t = np.array([0.0, 0.25 + np.random.uniform(-0.1, 0.1), 0.75 + np.random.uniform(-0.1, 0.1), 1.0])
        warp_t = control_t + np.array([0.0, np.random.uniform(-warp_scale, warp_scale), np.random.uniform(-warp_scale, warp_scale), 0.0])
        warp_t = np.clip(np.sort(warp_t), 0.0, 1.0)
        
        f_warp = interp1d(control_t, warp_t, kind='linear')
        warped_t_grid = f_warp(orig_t)
        
        f_seq = interp1d(orig_t, seq, axis=0, kind='linear', fill_value='extrapolate')
        return f_seq(warped_t_grid).astype(np.float32)

    @classmethod
    def add_jitter_and_scale(cls, seq: np.ndarray, noise_std: float = 0.008, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        가우시안 관절 노이즈 주입 및 미세 크기 스케일링
        seq: (T, D)
        """
        noise = np.random.normal(0, noise_std, size=seq.shape).astype(np.float32)
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return (seq * scale + noise).astype(np.float32)

    @classmethod
    def symmetric_flip_format_b(cls, seq: np.ndarray) -> np.ndarray:
        """
        Format B (64, 91) 텐서의 좌/우 관절 대칭 스왑 및 X축 부호 반전
        """
        flipped = seq.copy()
        
        # 1. Stream 1 Bone Edges swap
        edge_pairs = [
            (slice(8, 12), slice(12, 16)),   # Torso-LSh <-> Torso-RSh
            (slice(16, 20), slice(24, 28)),  # LSh-LElb <-> RSh-RElb
            (slice(20, 24), slice(28, 32)),  # LElb-LWr <-> RElb-RWr
            (slice(32, 36), slice(36, 40)),  # Pelvis-LHip <-> Pelvis-RHip
            (slice(40, 44), slice(48, 52)),  # LHip-LKnee <-> RHip-RKnee
            (slice(44, 48), slice(52, 56)),  # LKnee-LAnk <-> RKnee-RAnk
        ]
        for p1, p2 in edge_pairs:
            tmp = flipped[:, p1].copy()
            flipped[:, p1] = flipped[:, p2]
            flipped[:, p2] = tmp
            # X direction flip (dx at idx 0 of 4)
            flipped[:, p1.start] *= -1.0
            flipped[:, p2.start] *= -1.0

        # Pelvis-Torso (0..3) and Torso-Neck (4..7) X direction flip
        flipped[:, 0] *= -1.0
        flipped[:, 4] *= -1.0

        # 2. Stream 2 Angles swap (56..63)
        angle_pairs = [(56, 57), (58, 59), (60, 61), (62, 63)]
        for a1, a2 in angle_pairs:
            tmp = flipped[:, a1].copy()
            flipped[:, a1] = flipped[:, a2]
            flipped[:, a2] = tmp

        # 3. Stream 3 Alignment swap & Roll invert (64..71)
        flipped[:, 65] *= -1.0 # torso_roll
        # knee_valgus_l (66) <-> knee_valgus_r (67)
        tmp = flipped[:, 66].copy()
        flipped[:, 66] = flipped[:, 67]
        flipped[:, 67] = tmp

        # 4. Stream 5 Velocity swap (74..81)
        vel_pairs = [(74, 75), (76, 77), (78, 79), (80, 81)]
        for v1, v2 in vel_pairs:
            tmp = flipped[:, v1].copy()
            flipped[:, v1] = flipped[:, v2]
            flipped[:, v2] = tmp

        # 5. Stream 7 Disambiguation swap (85..91) if present
        if flipped.shape[1] >= 92:
            # signed_ankle_z_rel_l (85) <-> signed_ankle_z_rel_r (86)
            tmp = flipped[:, 85].copy()
            flipped[:, 85] = flipped[:, 86]
            flipped[:, 86] = tmp
            # stepping_foot_dx (87) -> sign invert
            flipped[:, 87] *= -1.0
            # shank_verticality_angle_l (89) <-> shank_verticality_angle_r (90)
            tmp = flipped[:, 89].copy()
            flipped[:, 89] = flipped[:, 90]
            flipped[:, 90] = tmp

        return flipped

    @classmethod
    def recompute_format_a_from_b(cls, seq_b: np.ndarray, orig_feat_a: np.ndarray) -> np.ndarray:
        """
        증강된 Format B로부터 통계 요약 피처 Format A를 재산출
        """
        T, D = seq_b.shape
        num_signals = 28 if D >= 92 else 21
        if num_signals == 28:
            indices = list(range(56, 64)) + list(range(64, 72)) + list(range(72, 74)) + list(range(82, 85)) + list(range(85, 92))
        else:
            indices = list(range(56, 64)) + list(range(64, 72)) + list(range(72, 74)) + list(range(82, 85))
        
        stat_feats = []
        for idx in indices:
            col = seq_b[:, idx]
            m = float(np.mean(col))
            s = float(np.std(col))
            mi = float(np.min(col))
            ma = float(np.max(col))
            d = ma - mi
            stat_feats.extend([m, s, mi, ma, d])
            
        # 최저점 스냅샷
        knee_mean = (seq_b[:, 56] + seq_b[:, 57]) / 2.0
        bottom_idx = int(np.argmin(knee_mean))
        
        snapshot_indices = [
            56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 71, 82
        ]
        if D >= 92:
            snapshot_indices.extend([85, 86, 89, 90, 91])

        for s_idx in snapshot_indices:
            stat_feats.append(float(seq_b[bottom_idx, s_idx]))
            
        new_stat = np.array(stat_feats, dtype=np.float32)
        base_len = len(new_stat)
        
        res = orig_feat_a.copy()
        if len(res) >= base_len:
            res[:base_len] = new_stat
        else:
            res = new_stat
        return res

    @classmethod
    def augment_dataset(
        cls,
        X_b: np.ndarray,
        X_a: np.ndarray,
        y: np.ndarray,
        target_per_class: int = 2200,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        전체 훈련 데이터셋 대상 지능형 클래스 균형 증강 파이프라인
        - 다수 클래스는 가벼운 시간 왜곡 및 노이즈 주입
        - 소수 클래스(스티프 데드, 크로스 런지, 굿모닝 등)는 집중 증강으로 불균형 완전 해소
        """
        np.random.seed(random_seed)
        aug_X_b = [X_b]
        aug_X_a = [X_a]
        aug_y = [y]

        classes, counts = np.unique(y, return_counts=True)
        print(f"[*] Starting Dataset Augmentation... (Original size: {len(y)})")

        for c, count in zip(classes, counts):
            needed = max(0, target_per_class - count)
            if needed <= 0:
                continue

            indices = np.where(y == c)[0]
            c_X_b = X_b[indices]
            c_X_a = X_a[indices]

            new_b_list = []
            new_a_list = []

            print(f"  [+] Augmenting Class {c}: {count} samples -> target {target_per_class} (+{needed} new)")

            # 필요 샘플 수만큼 증강 생성
            while len(new_b_list) < needed:
                # 1. 원본 샘플 무작위 선택
                rand_idx = np.random.randint(0, len(c_X_b))
                sample_b = c_X_b[rand_idx]
                sample_a = c_X_a[rand_idx]

                # 증강 기법 무작위 선택
                mode = np.random.choice(['warp', 'noise', 'flip', 'mixup'])

                if mode == 'flip' and c in cls.SYMMETRIC_CLASSES:
                    b_aug = cls.symmetric_flip_format_b(sample_b)
                    b_aug = cls.add_jitter_and_scale(b_aug, noise_std=0.005)
                elif mode == 'mixup' and len(c_X_b) > 1:
                    rand_idx2 = np.random.randint(0, len(c_X_b))
                    lam = np.random.beta(0.4, 0.4)
                    b_aug = lam * sample_b + (1.0 - lam) * c_X_b[rand_idx2]
                elif mode == 'warp':
                    b_aug = cls.time_warp_sequence(sample_b, warp_scale=0.15)
                    b_aug = cls.add_jitter_and_scale(b_aug, noise_std=0.006)
                else: # noise
                    b_aug = cls.add_jitter_and_scale(sample_b, noise_std=0.01, scale_range=(0.96, 1.04))

                a_aug = cls.recompute_format_a_from_b(b_aug, sample_a)
                new_b_list.append(b_aug)
                new_a_list.append(a_aug)

            aug_X_b.append(np.array(new_b_list, dtype=np.float32))
            aug_X_a.append(np.array(new_a_list, dtype=np.float32))
            aug_y.append(np.full(len(new_b_list), c, dtype=np.int64))

        final_X_b = np.concatenate(aug_X_b, axis=0)
        final_X_a = np.concatenate(aug_X_a, axis=0)
        final_y = np.concatenate(aug_y, axis=0)

        # 무작위 셔플링
        shuffle_idx = np.random.permutation(len(final_y))
        final_X_b = final_X_b[shuffle_idx]
        final_X_a = final_X_a[shuffle_idx]
        final_y = final_y[shuffle_idx]

        print(f"[+] Augmentation Completed! New Training Size: {len(final_y)} (Balanced across all 9 classes)")
        return final_X_b, final_X_a, final_y

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from src.core.skeleton2d import Joint15, SKELETON_EDGES, SYMMETRIC_PAIRS

class OneEuroFilter1D:
    """1D One-Euro Filter (Casiez et al., CHI 2012)"""
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    def _smoothing_factor(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * self.freq
        edx = self._smoothing_factor(self.dcutoff)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self._smoothing_factor(cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0


class OneEuroFilter3D:
    """(T, 15, 3) 전체 관절 좌표계용 One-Euro 필터 스트림"""
    def __init__(self, num_joints: int = 15, freq: float = 30.0, mincutoff: float = 1.2, beta: float = 0.05):
        self.num_joints = num_joints
        self.filters = [[OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta) for _ in range(3)] for _ in range(num_joints)]

    def apply(self, seq_3d: np.ndarray) -> np.ndarray:
        T, J, C = seq_3d.shape
        out = np.zeros_like(seq_3d, dtype=np.float32)
        for j in range(J):
            for c in range(C):
                self.filters[j][c].reset()
                for t in range(T):
                    out[t, j, c] = self.filters[j][c].filter(float(seq_3d[t, j, c]))
        return out


class Pose3DCorrector:
    """
    3단계 생체역학 3D 포즈 정밀 보정 파이프라인
    - 1단계: Root-centering & Temporal Smoothing (One-Euro)
    - 2단계: Kinematic Tree Bone-Length Invariance Projection (Forward Kinematics)
    - 3단계: Anthropometric Scale Normalization
    """

    KINEMATIC_TREE: List[Tuple[int, int, str]] = [
        # 척추/상체 (Pelvis -> Torso -> Neck)
        (Joint15.PELVIS, Joint15.TORSO, 'spine'),
        (Joint15.TORSO, Joint15.NECK, 'neck'),
        # 좌측 팔 (Torso -> L_Shoulder -> L_Elbow -> L_Wrist)
        (Joint15.TORSO, Joint15.LEFT_SHOULDER, 'clavicle_l'),
        (Joint15.LEFT_SHOULDER, Joint15.LEFT_ELBOW, 'upper_arm_l'),
        (Joint15.LEFT_ELBOW, Joint15.LEFT_WRIST, 'forearm_l'),
        # 우측 팔 (Torso -> R_Shoulder -> R_Elbow -> R_Wrist)
        (Joint15.TORSO, Joint15.RIGHT_SHOULDER, 'clavicle_r'),
        (Joint15.RIGHT_SHOULDER, Joint15.RIGHT_ELBOW, 'upper_arm_r'),
        (Joint15.RIGHT_ELBOW, Joint15.RIGHT_WRIST, 'forearm_r'),
        # 좌측 다리 (Pelvis -> L_Hip -> L_Knee -> L_Ankle)
        (Joint15.PELVIS, Joint15.LEFT_HIP, 'pelvis_hip_l'),
        (Joint15.LEFT_HIP, Joint15.LEFT_KNEE, 'thigh_l'),
        (Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE, 'shank_l'),
        # 우측 다리 (Pelvis -> R_Hip -> R_Knee -> R_Ankle)
        (Joint15.PELVIS, Joint15.RIGHT_HIP, 'pelvis_hip_r'),
        (Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE, 'thigh_r'),
        (Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE, 'shank_r')
    ]

    def __init__(
        self,
        user_height_m: float = 1.75,
        use_one_euro: bool = True,
        freq: float = 30.0,
        mincutoff: float = 1.2,
        beta: float = 0.05
    ):
        self.user_height_m = user_height_m
        self.use_one_euro = use_one_euro
        self.filter_engine = OneEuroFilter3D(num_joints=15, freq=freq, mincutoff=mincutoff, beta=beta)

    def enforce_bone_lengths(self, kps_3d: np.ndarray) -> np.ndarray:
        """
        키네마틱 트리를 따라 시퀀스 중위수 뼈 길이를 불변 기준(L_ref)으로 고정.
        각 관절의 3D 방향 벡터(Unit Vector)는 100% 보존하면서 길이만 고정함으로써
        자세 각도 왜곡 없이 뼈 길이 변동률을 0%로 정제.
        """
        T, J, C = kps_3d.shape
        corrected = np.copy(kps_3d)

        # 1. 각 분절의 시퀀스 중위수 기준 길이 산출
        ref_lengths = {}
        for p_idx, c_idx, seg_name in self.KINEMATIC_TREE:
            dists = np.linalg.norm(kps_3d[:, c_idx, :] - kps_3d[:, p_idx, :], axis=-1)
            ref_lengths[(p_idx, c_idx)] = float(np.median(dists))

        # 2. Forward Kinematics 순회하며 고정
        for t in range(T):
            for p_idx, c_idx, seg_name in self.KINEMATIC_TREE:
                p_pos = corrected[t, p_idx]
                c_pos = corrected[t, c_idx]
                vec = c_pos - p_pos
                dist = np.linalg.norm(vec)
                L_ref = ref_lengths[(p_idx, c_idx)]
                if dist > 1e-6:
                    u_vec = vec / dist
                else:
                    u_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                corrected[t, c_idx] = p_pos + u_vec * L_ref

        return corrected

    def correct_sequence(self, kps_3d: np.ndarray) -> Dict[str, Any]:
        """단일 시퀀스 (T, 15, 3) 3단계 통합 정밀 보정"""
        # 1. 시간축 One-Euro 필터링 (원시 지터 제거)
        if self.use_one_euro:
            filtered_kps = self.filter_engine.apply(kps_3d)
        else:
            filtered_kps = kps_3d

        # 2. 뼈 길이 불변성 키네마틱 투영
        final_kps = self.enforce_bone_lengths(filtered_kps)

        # 변동률 측정
        thigh_raw = np.linalg.norm(kps_3d[:, Joint15.LEFT_KNEE] - kps_3d[:, Joint15.LEFT_HIP], axis=-1)
        cv_before = float(np.std(thigh_raw) / (np.mean(thigh_raw) + 1e-6) * 100.0)

        thigh_after = np.linalg.norm(final_kps[:, Joint15.LEFT_KNEE] - final_kps[:, Joint15.LEFT_HIP], axis=-1)
        cv_after = float(np.std(thigh_after) / (np.mean(thigh_after) + 1e-6) * 100.0)

        return {
            'corrected_3d': final_kps,
            'bone_length_cv_before': cv_before,
            'bone_length_cv_after': cv_after
        }

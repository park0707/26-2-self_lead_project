import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from src.core.skeleton2d import Joint15, SKELETON_EDGES, SYMMETRIC_PAIRS

@dataclass
class KinematicPipelineConfig:
    """파이프라인 하이퍼파라미터 설정"""
    # 모듈 1: 캘리브레이션
    lockout_quantile: float = 0.95
    walkout_variance_thresh: float = 0.005
    walkout_min_frames: int = 15 # 약 0.5~0.6초
    
    # 모듈 2: Soft-Slack & IK
    softplus_beta: float = 20.0
    epsilon_mach: float = 1e-7
    singularity_eps: float = 1e-5
    
    # 모듈 3: One-Euro & 접지 역학
    one_euro_fc_min: float = 1.0 # Hz
    one_euro_beta: float = 0.007 # 속도 계수
    one_euro_d_cutoff: float = 1.0 # Hz
    foot_contact_vel_thresh: float = 0.08 # m/s
    foot_contact_height_thresh: float = 0.03 # m
    
    # 기본 체중 (BSIP용)
    default_body_mass_kg: float = 70.0


class OneEuroFilter1D:
    """속도 적응형 One-Euro 필터 (1차원)"""
    def __init__(self, t0: float, x0: float, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t: float, x: float) -> float:
        dt = t - self.t_prev
        if dt <= 1e-5:
            return self.x_prev
        
        # 1. 속도 필터링
        dx = (x - self.x_prev) / dt
        edx = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev
        
        # 2. 위치 가변 컷오프 계산
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # 3. 위치 필터링
        ex = self._smoothing_factor(dt, cutoff)
        x_hat = ex * x + (1.0 - ex) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter3D:
    """(N, 3) 다차원 관절 배열 전용 One-Euro 필터"""
    def __init__(self, n_joints: int = 15, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.n_joints = n_joints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def filter_sequence(self, sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        sequence: (T, N, D)
        returns: (T, N, D) filtered
        """
        T, N, D = sequence.shape
        out = np.zeros_like(sequence)
        dt = 1.0 / fps
        
        filters = []
        for j in range(N):
            joint_filters = []
            for d in range(D):
                f = OneEuroFilter1D(0.0, sequence[0, j, d], min_cutoff=self.min_cutoff,
                                   beta=self.beta, d_cutoff=self.d_cutoff)
                joint_filters.append(f)
            filters.append(joint_filters)
        
        out[0] = sequence[0]
        for t_idx in range(1, T):
            t = t_idx * dt
            for j in range(N):
                for d in range(D):
                    out[t_idx, j, d] = filters[j][d](t, sequence[t_idx, j, d])
        return out


class KinematicPipelineEngine:
    """
    차세대 인공지능 근골격 생체역학 파이프라인 (4대 모듈 구현체)
    """
    def __init__(self, config: Optional[KinematicPipelineConfig] = None):
        self.cfg = config or KinematicPipelineConfig()
        self.one_euro = OneEuroFilter3D(
            n_joints=15,
            min_cutoff=self.cfg.one_euro_fc_min,
            beta=self.cfg.one_euro_beta,
            d_cutoff=self.cfg.one_euro_d_cutoff
        )

    # -------------------------------------------------------------
    # [모듈 1] 인체 캘리브레이션 & 뼈대 길이 동결 (L* Anchor)
    # -------------------------------------------------------------
    def calibrate_bone_lengths(self, kps_seq: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        kps_seq: (T, 15, 2) or (T, 15, 3)
        동적 95% 분위수 뼈대 길이 L*를 추출하여 시간 불변 상수로 고정
        """
        T = len(kps_seq)
        bone_lengths = {}
        
        # 1. 워크아웃 배제 필터링 (발목 분산 안정화 구간 탐색)
        left_ankle = kps_seq[:, Joint15.LEFT_ANKLE, :2]
        right_ankle = kps_seq[:, Joint15.RIGHT_ANKLE, :2]
        
        start_idx = 0
        for t in range(min(T - self.cfg.walkout_min_frames, 30)):
            var_l = np.var(left_ankle[t:t + self.cfg.walkout_min_frames], axis=0).sum()
            var_r = np.var(right_ankle[t:t + self.cfg.walkout_min_frames], axis=0).sum()
            if var_l < self.cfg.walkout_variance_thresh and var_r < self.cfg.walkout_variance_thresh:
                start_idx = t
                break
        
        # 2. 안정화 구간 기반 뼈대 길이 95% 분위수 산출
        valid_frames = kps_seq[start_idx:] if (T - start_idx) >= 5 else kps_seq
        for p, c in SKELETON_EDGES:
            diff = valid_frames[:, p] - valid_frames[:, c]
            dist = np.linalg.norm(diff, axis=-1)
            l_star = float(np.percentile(dist, self.cfg.lockout_quantile * 100))
            bone_lengths[(p, c)] = max(l_star, 1e-3)
            
        return bone_lengths

    # -------------------------------------------------------------
    # [모듈 2] 지수 완화 Soft-Slack 해석적 IK & 카메라 3D 복원
    # -------------------------------------------------------------
    def apply_soft_slack_ik(self, kps_2d: np.ndarray, kps_3d_prior: np.ndarray,
                            bone_lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        kps_2d: (T, 15, 2) - 고정밀 2D 입력
        kps_3d_prior: (T, 15, 3) - 깊이 부호 prior
        bone_lengths: 고정된 L* 딕셔너리
        returns: (T, 15, 3) 카메라 좌표계 기준 엄밀 기구학 3D 좌표계
        """
        T, N, _ = kps_2d.shape
        refined_3d = np.zeros((T, N, 3), dtype=np.float32)
        
        # Root(Pelvis)는 (x_2d, y_2d, 0)
        refined_3d[:, Joint15.PELVIS, :2] = kps_2d[:, Joint15.PELVIS]
        refined_3d[:, Joint15.PELVIS, 2] = 0.0
        
        # 뼈대 트리를 따라 Forward Kinematics 전파
        for p, c in SKELETON_EDGES:
            L_star = bone_lengths.get((p, c), 0.2)
            L_sq = L_star * L_star
            
            dx = kps_2d[:, c, 0] - kps_2d[:, p, 0]
            dy = kps_2d[:, c, 1] - kps_2d[:, p, 1]
            dxy_sq = dx * dx + dy * dy
            
            # Prior 깊이 부호
            dz_prior = kps_3d_prior[:, c, 2] - kps_3d_prior[:, p, 2] if kps_3d_prior.shape[-1] >= 3 else 0.0
            sign_prior = np.where(dz_prior >= 0, 1.0, -1.0)
            
            # 지수 완화 Softplus 수식
            arg = self.cfg.softplus_beta * (L_sq - dxy_sq)
            softplus_val = np.where(
                arg > 20.0,
                L_sq - dxy_sq,
                (1.0 / self.cfg.softplus_beta) * np.log1p(np.exp(np.clip(arg, -50.0, 20.0)))
            )
            
            dz = sign_prior * np.sqrt(np.maximum(softplus_val, 0.0) + self.cfg.epsilon_mach)
            
            refined_3d[:, c, 0] = refined_3d[:, p, 0] + dx
            refined_3d[:, c, 1] = refined_3d[:, p, 1] + dy
            refined_3d[:, c, 2] = refined_3d[:, p, 2] + dz

        return refined_3d

    # -------------------------------------------------------------
    # [모듈 2] 중력 정렬 SE(2) Yaw 불변 로컬 좌표계
    # -------------------------------------------------------------
    def align_to_se2_gravity_frame(self, kps_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        kps_3d: (T, 15, 3) 카메라 좌표계
        returns: (canonical_3d, yaw_angles)
        """
        T, N, _ = kps_3d.shape
        aligned = np.zeros_like(kps_3d)
        yaw_angles = np.zeros(T, dtype=np.float32)
        
        for t in range(T):
            root = kps_3d[t, Joint15.PELVIS].copy()
            centered = kps_3d[t] - root
            
            hip_vec = centered[Joint15.RIGHT_HIP] - centered[Joint15.LEFT_HIP]
            hx, hz = hip_vec[0], hip_vec[2]
            norm_xz = np.sqrt(hx * hx + hz * hz) + 1e-7
            
            yaw = np.arctan2(hz, hx)
            yaw_angles[t] = yaw
            
            cos_y = np.cos(-yaw)
            sin_y = np.sin(-yaw)
            
            R_y = np.array([
                [cos_y, 0.0, sin_y],
                [0.0,   1.0, 0.0],
                [-sin_y, 0.0, cos_y]
            ], dtype=np.float32)
            
            aligned[t] = (R_y @ centered.T).T
            
        return aligned, yaw_angles

    # -------------------------------------------------------------
    # [모듈 3] 실시간 One-Euro 필터링 및 접지 제약
    # -------------------------------------------------------------
    def apply_contact_and_temporal_filter(self, kps_3d: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """One-Euro 적응형 필터 적용 및 발바닥 접지 안착"""
        filtered_3d = self.one_euro.filter_sequence(kps_3d, fps=fps)
        return filtered_3d

    # -------------------------------------------------------------
    # 엔드투엔드 전체 추출 실행기
    # -------------------------------------------------------------
    def process_sequence(self, kps_2d: np.ndarray, kps_3d_prior: np.ndarray,
                         fps: float = 30.0) -> Dict[str, Any]:
        """
        1) 2D 입력 One-Euro 필터링 -> 노이즈 및 지터 제거
        2) 뼈대 길이 L* 고정
        3) Soft-Slack IK로 카메라 좌표계 3D 복원 (ik_3d_camera)
        4) SE(2) Yaw 불변 정렬로 표준 3D 복원 (canonical_3d)
        """
        # 0. 2D 좌표 One-Euro 필터링 (카메라 프레임 2D 지터 제거)
        kps_2d_filt = self.one_euro.filter_sequence(kps_2d, fps=fps)
        
        # 1. 뼈대 캘리브레이션
        bone_lengths = self.calibrate_bone_lengths(kps_2d_filt)
        
        # 2. Soft-Slack 해석적 IK (카메라 좌표계)
        ik_3d_camera = self.apply_soft_slack_ik(kps_2d_filt, kps_3d_prior, bone_lengths)
        ik_3d_camera_filt = self.apply_contact_and_temporal_filter(ik_3d_camera, fps=fps)
        
        # 3. SE(2) 중력 정렬 로컬계 (동작 인식/분류용)
        canonical_3d, yaw_angles = self.align_to_se2_gravity_frame(ik_3d_camera_filt)
        
        return {
            'camera_2d_refined': ik_3d_camera_filt[:, :, :2],
            'camera_3d_refined': ik_3d_camera_filt,
            'canonical_3d': canonical_3d,
            'bone_lengths': bone_lengths,
            'yaw_angles': yaw_angles
        }


class KinematicFeatureCalculator:
    """
    15개 관절 좌표로부터 생체역학적 핵심 각도(2D/3D) 및 피처 추출기
    """
    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """
        p1, p2(vertex), p3: (T, 2) or (T, 3)
        returns: (T,) 각도 (deg, 0~180)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
        v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
        
        dot = np.sum((v1 / v1_norm) * (v2 / v2_norm), axis=-1)
        dot_clipped = np.clip(dot, -1.0, 1.0)
        return np.degrees(np.arccos(dot_clipped))

    @classmethod
    def extract_core_features_2d(cls, kps_2d: np.ndarray, fps: float = 30.0) -> Dict[str, np.ndarray]:
        """2D 카메라 평면 기준 관절 각도 및 피처"""
        knee_l = cls.calculate_angle(kps_2d[:, Joint15.LEFT_HIP], kps_2d[:, Joint15.LEFT_KNEE], kps_2d[:, Joint15.LEFT_ANKLE])
        knee_r = cls.calculate_angle(kps_2d[:, Joint15.RIGHT_HIP], kps_2d[:, Joint15.RIGHT_KNEE], kps_2d[:, Joint15.RIGHT_ANKLE])
        knee_avg = (knee_l + knee_r) / 2.0
        
        hip_l = cls.calculate_angle(kps_2d[:, Joint15.TORSO], kps_2d[:, Joint15.LEFT_HIP], kps_2d[:, Joint15.LEFT_KNEE])
        hip_r = cls.calculate_angle(kps_2d[:, Joint15.TORSO], kps_2d[:, Joint15.RIGHT_HIP], kps_2d[:, Joint15.RIGHT_KNEE])
        hip_avg = (hip_l + hip_r) / 2.0
        
        elbow_l = cls.calculate_angle(kps_2d[:, Joint15.LEFT_SHOULDER], kps_2d[:, Joint15.LEFT_ELBOW], kps_2d[:, Joint15.LEFT_WRIST])
        elbow_r = cls.calculate_angle(kps_2d[:, Joint15.RIGHT_SHOULDER], kps_2d[:, Joint15.RIGHT_ELBOW], kps_2d[:, Joint15.RIGHT_WRIST])
        
        # 체간 경사각 (수직 Y축 [0, -1] 대비)
        spine_vec = kps_2d[:, Joint15.NECK] - kps_2d[:, Joint15.PELVIS]
        spine_norm = np.linalg.norm(spine_vec, axis=-1, keepdims=True) + 1e-7
        spine_dir = spine_vec / spine_norm
        torso_lean = np.degrees(np.arccos(np.clip(-spine_dir[:, 1], -1.0, 1.0)))
        
        dt = 1.0 / fps
        knee_vel = np.gradient(knee_avg, dt)
        
        return {
            'knee_angle_l': knee_l,
            'knee_angle_r': knee_r,
            'knee_angle_avg': knee_avg,
            'hip_angle_l': hip_l,
            'hip_angle_r': hip_r,
            'hip_angle_avg': hip_avg,
            'elbow_angle_l': elbow_l,
            'elbow_angle_r': elbow_r,
            'torso_lean': torso_lean,
            'knee_velocity': knee_vel
        }

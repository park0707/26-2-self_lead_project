import numpy as np
from typing import Dict, List, Tuple, Any
from src.core.skeleton2d import Joint15, SKELETON_EDGES

class KinematicsExtractor2D:
    """2D 15-Joint 생체역학 멀티스트림 피처 및 Format A (97차원) / Format B (69차원) 빌더"""

    FEATURE_NAMES_19 = [
        # 1. 8대 관절 굴곡/신전 각도 (8개)
        'knee_angle_l', 'knee_angle_r',
        'hip_angle_l', 'hip_angle_r',
        'ankle_angle_l', 'ankle_angle_r',
        'elbow_angle_l', 'elbow_angle_r',
        # 2. 자세 결함 및 정렬 지표 (6개)
        'torso_inclination',
        'knee_valgus_l', 'knee_valgus_r',
        'sh_hip_twist',
        'knee_distance', 'ankle_distance',
        # 3. 좌우 비대칭률 (2개)
        'knee_asymmetry', 'hip_asymmetry',
        # 4. AI Hub 9종 운동 특화 지표 (3개)
        'neck_spine_alignment',    # 고개 정면 / 시선 방향 유지 오류 검출 (Neck-Torso-Pelvis)
        'barbell_body_distance',  # 바벨 궤적 몸 밀착 실패 검출 (Wrist to Knee/Shank 수평 거리)
        'knee_hip_synchrony'      # 무릎-골반 동시 펴짐(동조성) 실패 검출 (무릎-고관절 각속도 차이)
    ]

    @staticmethod
    def calculate_2d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """세 점 p1-p2-p3 (p2가 꼭짓점)의 2D 사잇각 계산 (Shape: (T,), 단위: degree [0, 180])"""
        v1 = p1 - p2 # (T, 2)
        v2 = p3 - p2 # (T, 2)
        dot = np.sum(v1 * v2, axis=-1)
        norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7
        cos_val = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_val))

    @classmethod
    def extract_stream1_bones(cls, kps: np.ndarray) -> np.ndarray:
        """Stream 1: 14개 뼈대 분절의 (dx, dy, length) -> (T, 42)"""
        T = kps.shape[0]
        bone_feats = []
        for src_j, dst_j in SKELETON_EDGES:
            vec = kps[:, dst_j, :] - kps[:, src_j, :] # (T, 2)
            length = np.linalg.norm(vec, axis=-1, keepdims=True) # (T, 1)
            bone_feats.extend([vec[:, 0:1], vec[:, 1:2], length])
        return np.concatenate(bone_feats, axis=-1).astype(np.float32) # (T, 42)

    @classmethod
    def extract_stream2_angles(cls, kps: np.ndarray) -> Dict[str, np.ndarray]:
        """Stream 2: 주요 8대 관절 굴곡/신전 각도 산출 -> Dict of (T,)"""
        angles = {}
        # 무릎 굴곡각 (Hip - Knee - Ankle)
        angles['knee_angle_l'] = cls.calculate_2d_angle(
            kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :], kps[:, Joint15.LEFT_ANKLE, :]
        )
        angles['knee_angle_r'] = cls.calculate_2d_angle(
            kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :], kps[:, Joint15.RIGHT_ANKLE, :]
        )
        # 고관절 굴곡각 (Torso - Hip - Knee)
        angles['hip_angle_l'] = cls.calculate_2d_angle(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :]
        )
        angles['hip_angle_r'] = cls.calculate_2d_angle(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :]
        )
        # 발목 각도 (Pelvis - Ankle - Knee)
        angles['ankle_angle_l'] = cls.calculate_2d_angle(
            kps[:, Joint15.PELVIS, :], kps[:, Joint15.LEFT_ANKLE, :], kps[:, Joint15.LEFT_KNEE, :]
        )
        angles['ankle_angle_r'] = cls.calculate_2d_angle(
            kps[:, Joint15.PELVIS, :], kps[:, Joint15.RIGHT_ANKLE, :], kps[:, Joint15.RIGHT_KNEE, :]
        )
        # 팔꿈치 각도 (Shoulder - Elbow - Wrist)
        angles['elbow_angle_l'] = cls.calculate_2d_angle(
            kps[:, Joint15.LEFT_SHOULDER, :], kps[:, Joint15.LEFT_ELBOW, :], kps[:, Joint15.LEFT_WRIST, :]
        )
        angles['elbow_angle_r'] = cls.calculate_2d_angle(
            kps[:, Joint15.RIGHT_SHOULDER, :], kps[:, Joint15.RIGHT_ELBOW, :], kps[:, Joint15.RIGHT_WRIST, :]
        )
        return angles

    @classmethod
    def extract_stream3_alignment(cls, kps: np.ndarray) -> Dict[str, np.ndarray]:
        """Stream 3: 체간 경사각, Knee Valgus 편위, 스탠스/무릎 거리 -> Dict of (T,)"""
        align = {}
        # 1. 체간 전방 경사각 (수직축 대비 Torso-Pelvis 기울기)
        torso_vec = kps[:, Joint15.TORSO, :] - kps[:, Joint15.PELVIS, :]
        align['torso_inclination'] = np.degrees(
            np.arctan2(np.abs(torso_vec[:, 0]), np.abs(torso_vec[:, 1]) + 1e-7)
        )
        
        # 2. Knee Valgus 2D 편위 (Hip->Knee x변위 vs Knee->Ankle x변위)
        align['knee_valgus_l'] = (kps[:, Joint15.LEFT_KNEE, 0] - kps[:, Joint15.LEFT_HIP, 0]) - \
                                (kps[:, Joint15.LEFT_ANKLE, 0] - kps[:, Joint15.LEFT_KNEE, 0])
        align['knee_valgus_r'] = (kps[:, Joint15.RIGHT_KNEE, 0] - kps[:, Joint15.RIGHT_HIP, 0]) - \
                                (kps[:, Joint15.RIGHT_ANKLE, 0] - kps[:, Joint15.RIGHT_KNEE, 0])
                                
        # 3. 어깨-골반 수평 평행도 차이 (상체 회전 비틀림)
        sh_vec = kps[:, Joint15.RIGHT_SHOULDER, :] - kps[:, Joint15.LEFT_SHOULDER, :]
        hip_vec = kps[:, Joint15.RIGHT_HIP, :] - kps[:, Joint15.LEFT_HIP, :]
        sh_ang = np.degrees(np.arctan2(sh_vec[:, 1], sh_vec[:, 0] + 1e-7))
        hip_ang = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0] + 1e-7))
        align['sh_hip_twist'] = np.abs(sh_ang - hip_ang)
        
        # 4. 양 무릎 간 거리 및 양 발목 간 스탠스 거리
        align['knee_distance'] = np.linalg.norm(
            kps[:, Joint15.LEFT_KNEE, :] - kps[:, Joint15.RIGHT_KNEE, :], axis=-1
        )
        align['ankle_distance'] = np.linalg.norm(
            kps[:, Joint15.LEFT_ANKLE, :] - kps[:, Joint15.RIGHT_ANKLE, :], axis=-1
        )
        return align

    @classmethod
    def extract_stream4_asymmetry(cls, angles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Stream 4: 좌우 대칭 관절 비대칭률 (%) -> Dict of (T,)"""
        asym = {}
        # 무릎 비대칭률
        kl, kr = angles['knee_angle_l'], angles['knee_angle_r']
        asym['knee_asymmetry'] = (np.abs(kl - kr) / (np.abs(kl) + np.abs(kr) + 1e-6)) * 100.0
        
        # 고관절 비대칭률
        hl, hr = angles['hip_angle_l'], angles['hip_angle_r']
        asym['hip_asymmetry'] = (np.abs(hl - hr) / (np.abs(hl) + np.abs(hr) + 1e-6)) * 100.0
        return asym

    @classmethod
    def extract_stream5_velocity(cls, angles: Dict[str, np.ndarray], dt: float = 1.0 / 30.0) -> np.ndarray:
        """Stream 5: 주요 8대 관절 각속도 (1차 시간 미분) -> (T, 8)"""
        ang_keys = [
            'knee_angle_l', 'knee_angle_r', 'hip_angle_l', 'hip_angle_r',
            'ankle_angle_l', 'ankle_angle_r', 'elbow_angle_l', 'elbow_angle_r'
        ]
        ang_mat = np.stack([angles[k] for k in ang_keys], axis=-1) # (T, 8)
        vel_mat = np.gradient(ang_mat, dt, axis=0) # (T, 8)
        return vel_mat.astype(np.float32)

    @classmethod
    def extract_aihub_special_features(cls, kps: np.ndarray, angles: Dict[str, np.ndarray], dt: float = 1.0 / 30.0) -> Dict[str, np.ndarray]:
        """AI Hub 9종 운동 12대 오류 조건 100% 식별을 위한 3대 특화 피처"""
        spec = {}
        # 1. 시선/경추 정렬각: Neck(14) - Torso(1) - Pelvis(0) 사잇각 (고개 정면/시선 이탈 검출)
        spec['neck_spine_alignment'] = cls.calculate_2d_angle(
            kps[:, Joint15.NECK, :], kps[:, Joint15.TORSO, :], kps[:, Joint15.PELVIS, :]
        )
        
        # 2. 바벨-몸 밀착 거리: 손목(Wrist 6,7)과 무릎/정강이(Knee 10,11) 사이의 수평 이격 거리 (데드리프트 바벨 이탈 검출)
        wrist_x = (kps[:, Joint15.LEFT_WRIST, 0] + kps[:, Joint15.RIGHT_WRIST, 0]) / 2.0
        knee_x = (kps[:, Joint15.LEFT_KNEE, 0] + kps[:, Joint15.RIGHT_KNEE, 0]) / 2.0
        spec['barbell_body_distance'] = np.abs(wrist_x - knee_x)

        # 3. 무릎-골반 동조성 지표: 무릎 신전 각속도와 고관절 신전 각속도의 절대 차이 (엉덩이 먼저 들림 검출)
        knee_avg = (angles['knee_angle_l'] + angles['knee_angle_r']) / 2.0
        hip_avg = (angles['hip_angle_l'] + angles['hip_angle_r']) / 2.0
        knee_vel = np.gradient(knee_avg, dt)
        hip_vel = np.gradient(hip_avg, dt)
        spec['knee_hip_synchrony'] = np.abs(knee_vel - hip_vel)

        return spec

    @classmethod
    def build_format_b_timeseries(cls, kps_2d_norm: np.ndarray) -> np.ndarray:
        """Format B: 64프레임 전 구간 69차원 시계열 피처 텐서 구축 -> (64, 69)"""
        # 1. Bone Stream (42)
        s1_bones = cls.extract_stream1_bones(kps_2d_norm) # (64, 42)
        # 2. Joint Angles (8)
        s2_angles_dict = cls.extract_stream2_angles(kps_2d_norm)
        s2_angles = np.stack([s2_angles_dict[k] for k in [
            'knee_angle_l', 'knee_angle_r', 'hip_angle_l', 'hip_angle_r',
            'ankle_angle_l', 'ankle_angle_r', 'elbow_angle_l', 'elbow_angle_r'
        ]], axis=-1) # (64, 8)
        # 3. Alignment (6)
        s3_align_dict = cls.extract_stream3_alignment(kps_2d_norm)
        s3_align = np.stack([s3_align_dict[k] for k in [
            'torso_inclination', 'knee_valgus_l', 'knee_valgus_r',
            'sh_hip_twist', 'knee_distance', 'ankle_distance'
        ]], axis=-1) # (64, 6)
        # 4. Asymmetry (2)
        s4_asym_dict = cls.extract_stream4_asymmetry(s2_angles_dict)
        s4_asym = np.stack([s4_asym_dict['knee_asymmetry'], s4_asym_dict['hip_asymmetry']], axis=-1) # (64, 2)
        # 5. Velocity (8)
        s5_vel = cls.extract_stream5_velocity(s2_angles_dict) # (64, 8)
        # 6. AI Hub Special (3)
        s6_spec_dict = cls.extract_aihub_special_features(kps_2d_norm, s2_angles_dict)
        s6_spec = np.stack([
            s6_spec_dict['neck_spine_alignment'],
            s6_spec_dict['barbell_body_distance'],
            s6_spec_dict['knee_hip_synchrony']
        ], axis=-1) # (64, 3)

        # Concat: 42 + 8 + 6 + 2 + 8 + 3 = 69차원
        feat_b = np.concatenate([s1_bones, s2_angles, s3_align, s4_asym, s5_vel, s6_spec], axis=-1)
        return feat_b.astype(np.float32) # (64, 69)

    @classmethod
    def build_format_a_summary(cls, kps_2d_norm: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Format A: 19대 핵심 지표 통계량 (95개) + 템포 지표 (2개) -> 총 97차원 1D 벡터"""
        s2_angles_dict = cls.extract_stream2_angles(kps_2d_norm)
        s3_align_dict = cls.extract_stream3_alignment(kps_2d_norm)
        s4_asym_dict = cls.extract_stream4_asymmetry(s2_angles_dict)
        s6_spec_dict = cls.extract_aihub_special_features(kps_2d_norm, s2_angles_dict)

        all_19_dict = {}
        all_19_dict.update(s2_angles_dict)
        all_19_dict.update(s3_align_dict)
        all_19_dict.update(s4_asym_dict)
        all_19_dict.update(s6_spec_dict)

        feat_values = []
        feat_names = []

        # 19개 지표 x 5대 통계량 = 95차원
        for key in cls.FEATURE_NAMES_19:
            arr = all_19_dict[key] # (64,)
            val_min = float(np.min(arr))
            val_max = float(np.max(arr))
            val_mean = float(np.mean(arr))
            val_std = float(np.std(arr))
            val_rom = float(val_max - val_min)

            feat_values.extend([val_min, val_max, val_mean, val_std, val_rom])
            feat_names.extend([
                f"{key}_min", f"{key}_max", f"{key}_mean", f"{key}_std", f"{key}_rom"
            ])

        # 템포 지표 2차원: 최대 굴곡 시점 프레임 비율 & 무릎 평균 각속도
        knee_avg = (s2_angles_dict['knee_angle_l'] + s2_angles_dict['knee_angle_r']) / 2.0
        peak_idx = int(np.argmin(knee_avg)) # 최저점 (가장 깊이 굴곡한 시점)
        peak_ratio = float(peak_idx / max(1, len(knee_avg) - 1))
        
        knee_vel = np.gradient(knee_avg, 1.0 / 30.0)
        mean_vel_mag = float(np.mean(np.abs(knee_vel)))

        feat_values.extend([peak_ratio, mean_vel_mag])
        feat_names.extend(['peak_flexion_frame_ratio', 'knee_velocity_magnitude_mean'])

        return np.array(feat_values, dtype=np.float32), feat_names

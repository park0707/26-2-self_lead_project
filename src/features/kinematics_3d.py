import numpy as np
from typing import Dict, List, Tuple, Any
from src.core.skeleton2d import Joint15, SKELETON_EDGES

class KinematicsExtractor3D:
    """3D 15-Joint 3차원 생체역학 멀티스트림 피처 및 Format A (121차원) / Format B (85차원) 빌더"""

    FEATURE_NAMES_28_3D = [
        # 1. 8대 3차원 관절 굴곡/신전 각도 (8개)
        'knee_angle_l_3d', 'knee_angle_r_3d',
        'hip_angle_l_3d', 'hip_angle_r_3d',
        'ankle_angle_l_3d', 'ankle_angle_r_3d',
        'elbow_angle_l_3d', 'elbow_angle_r_3d',
        # 2. 3차원 자세 결함 및 정렬 지표 (8개)
        'torso_pitch_3d', 'torso_roll_3d',
        'knee_valgus_l_3d', 'knee_valgus_r_3d',
        'sh_hip_twist_3d',
        'knee_distance_3d', 'ankle_distance_3d',
        'ankle_z_spread_3d',
        # 3. 좌우 3차원 비대칭률 (2개)
        'knee_asymmetry_3d', 'hip_asymmetry_3d',
        # 4. 3D 특화 동적 지표 (3개)
        'neck_spine_alignment_3d',
        'barbell_body_distance_3d',
        'knee_hip_synchrony_3d',
        # 5. [신규] 런지 및 스티프데드 취약점 분별 특화 지표 (7개)
        'signed_ankle_z_rel_l_3d', 'signed_ankle_z_rel_r_3d',
        'stepping_foot_dx_3d', 'stepping_foot_dz_3d',
        'shank_verticality_angle_l_3d', 'shank_verticality_angle_r_3d',
        'hip_to_knee_rom_ratio_3d'
    ]
    # 하위 호환성 유지
    FEATURE_NAMES_21_3D = FEATURE_NAMES_28_3D

    @staticmethod
    def calculate_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """세 점 p1-p2-p3 (p2가 꼭짓점)의 3차원 공간 사잇각 계산 (Shape: (T,), 단위: degree [0, 180])"""
        v1 = p1 - p2 # (T, 3)
        v2 = p3 - p2 # (T, 3)
        dot = np.sum(v1 * v2, axis=-1)
        norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7
        cos_val = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_val))

    @classmethod
    def extract_stream1_bones_3d(cls, kps: np.ndarray) -> np.ndarray:
        """Stream 1: 14개 뼈대 분절의 (dx, dy, dz, length_3d) -> (T, 56)"""
        T = kps.shape[0]
        bone_feats = []
        for src_j, dst_j in SKELETON_EDGES:
            vec = kps[:, dst_j, :] - kps[:, src_j, :] # (T, 3)
            length = np.linalg.norm(vec, axis=-1, keepdims=True) # (T, 1)
            bone_feats.extend([vec[:, 0:1], vec[:, 1:2], vec[:, 2:3], length])
        return np.concatenate(bone_feats, axis=-1).astype(np.float32) # (T, 56)

    @classmethod
    def extract_stream2_angles_3d(cls, kps: np.ndarray) -> Dict[str, np.ndarray]:
        """Stream 2: 주요 8대 3차원 관절 사잇각 산출 -> Dict of (T,)"""
        angles = {}
        # 무릎 굴곡각 (Hip - Knee - Ankle)
        angles['knee_angle_l_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :], kps[:, Joint15.LEFT_ANKLE, :]
        )
        angles['knee_angle_r_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :], kps[:, Joint15.RIGHT_ANKLE, :]
        )
        # 고관절 굴곡각 (Torso - Hip - Knee)
        angles['hip_angle_l_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :]
        )
        angles['hip_angle_r_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :]
        )
        # 발목 굴곡각 (Knee - Ankle - Foot_Ref(Ankle_Z+0.2))
        ankle_ref_l = kps[:, Joint15.LEFT_ANKLE, :].copy()
        ankle_ref_l[:, 2] += 0.2
        ankle_ref_r = kps[:, Joint15.RIGHT_ANKLE, :].copy()
        ankle_ref_r[:, 2] += 0.2

        angles['ankle_angle_l_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.LEFT_KNEE, :], kps[:, Joint15.LEFT_ANKLE, :], ankle_ref_l
        )
        angles['ankle_angle_r_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.RIGHT_KNEE, :], kps[:, Joint15.RIGHT_ANKLE, :], ankle_ref_r
        )
        # 팔꿈치 각도 (Shoulder - Elbow - Wrist)
        angles['elbow_angle_l_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.LEFT_SHOULDER, :], kps[:, Joint15.LEFT_ELBOW, :], kps[:, Joint15.LEFT_WRIST, :]
        )
        angles['elbow_angle_r_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.RIGHT_SHOULDER, :], kps[:, Joint15.RIGHT_ELBOW, :], kps[:, Joint15.RIGHT_WRIST, :]
        )
        return angles

    @classmethod
    def extract_stream3_alignment_3d(cls, kps: np.ndarray) -> Dict[str, np.ndarray]:
        """Stream 3: 3차원 자세 결함 및 정렬 지표 (8개)"""
        alignments = {}
        # 1. 체간 Pitch (Sagittal 전방 경사각): Z-Y 평면 상의 척추 경사
        spine_vec = kps[:, Joint15.NECK, :] - kps[:, Joint15.PELVIS, :] # (T, 3)
        alignments['torso_pitch_3d'] = np.degrees(np.arctan2(spine_vec[:, 2], np.abs(spine_vec[:, 1]) + 1e-7))
        
        # 2. 체간 Roll (Coronal 좌우 기움각): X-Y 평면 상의 척추 경사
        alignments['torso_roll_3d'] = np.degrees(np.arctan2(spine_vec[:, 0], np.abs(spine_vec[:, 1]) + 1e-7))

        # 3. 무릎 내측 편위 (Knee Valgus): 고관절-발목 연결선 대비 무릎의 X방향 내측 변위
        hip_ankle_mid_l = (kps[:, Joint15.LEFT_HIP, :] + kps[:, Joint15.LEFT_ANKLE, :]) / 2.0
        hip_ankle_mid_r = (kps[:, Joint15.RIGHT_HIP, :] + kps[:, Joint15.RIGHT_ANKLE, :]) / 2.0
        alignments['knee_valgus_l_3d'] = kps[:, Joint15.LEFT_KNEE, 0] - hip_ankle_mid_l[:, 0]
        alignments['knee_valgus_r_3d'] = hip_ankle_mid_r[:, 0] - kps[:, Joint15.RIGHT_KNEE, 0]

        # 4. 어깨-골반 3차원 비틀림각 (Transverse Twist)
        sh_vec = kps[:, Joint15.RIGHT_SHOULDER, :] - kps[:, Joint15.LEFT_SHOULDER, :]
        hip_vec = kps[:, Joint15.RIGHT_HIP, :] - kps[:, Joint15.LEFT_HIP, :]
        # 수평 평면(X-Z) 상의 회전 각도차
        sh_angle_xz = np.degrees(np.arctan2(sh_vec[:, 2], sh_vec[:, 0] + 1e-7))
        hip_angle_xz = np.degrees(np.arctan2(hip_vec[:, 2], hip_vec[:, 0] + 1e-7))
        alignments['sh_hip_twist_3d'] = np.abs(sh_angle_xz - hip_angle_xz)

        # 5. 무릎/발목 3차원 유클리드 거리
        alignments['knee_distance_3d'] = np.linalg.norm(
            kps[:, Joint15.LEFT_KNEE, :] - kps[:, Joint15.RIGHT_KNEE, :], axis=-1
        )
        alignments['ankle_distance_3d'] = np.linalg.norm(
            kps[:, Joint15.LEFT_ANKLE, :] - kps[:, Joint15.RIGHT_ANKLE, :], axis=-1
        )

        # 6. 발목 전후 깊이 분리 거리 (런지/스쿼트 측면 식별의 핵심)
        alignments['ankle_z_spread_3d'] = np.abs(
            kps[:, Joint15.LEFT_ANKLE, 2] - kps[:, Joint15.RIGHT_ANKLE, 2]
        )

        return alignments

    @classmethod
    def extract_stream4_asymmetry_3d(cls, angles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Stream 4: 좌우 3차원 관절각 비대칭률 (2개)"""
        asymm = {}
        denom_k = np.maximum(angles['knee_angle_l_3d'], angles['knee_angle_r_3d']) + 1e-7
        asymm['knee_asymmetry_3d'] = np.abs(angles['knee_angle_l_3d'] - angles['knee_angle_r_3d']) / denom_k

        denom_h = np.maximum(angles['hip_angle_l_3d'], angles['hip_angle_r_3d']) + 1e-7
        asymm['hip_asymmetry_3d'] = np.abs(angles['hip_angle_l_3d'] - angles['hip_angle_r_3d']) / denom_h
        return asymm

    @classmethod
    def extract_stream5_velocity_3d(cls, angle_matrix: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """Stream 5: 3D 관절 각속도 (T, 8)"""
        vel = np.gradient(angle_matrix, dt, axis=0) # (T, 8)
        return vel.astype(np.float32)

    @classmethod
    def extract_stream6_special_3d(cls, kps: np.ndarray, angles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Stream 6: 3D 특화 지표 (3개)"""
        special = {}
        # 1. 목-척추-골반 3D 직선성 정렬 (Neck - Torso - Pelvis 각도)
        special['neck_spine_alignment_3d'] = cls.calculate_3d_angle(
            kps[:, Joint15.NECK, :], kps[:, Joint15.TORSO, :], kps[:, Joint15.PELVIS, :]
        )
        # 2. 바벨-체간 3D 유클리드 최소 거리
        wrist_mid = (kps[:, Joint15.LEFT_WRIST, :] + kps[:, Joint15.RIGHT_WRIST, :]) / 2.0
        knee_mid = (kps[:, Joint15.LEFT_KNEE, :] + kps[:, Joint15.RIGHT_KNEE, :]) / 2.0
        special['barbell_body_distance_3d'] = np.linalg.norm(wrist_mid - knee_mid, axis=-1)

        # 3. 무릎-골반 펴짐 동조성 (각속도 차이)
        knee_mean_angle = (angles['knee_angle_l_3d'] + angles['knee_angle_r_3d']) / 2.0
        hip_mean_angle = (angles['hip_angle_l_3d'] + angles['hip_angle_r_3d']) / 2.0
        knee_vel = np.gradient(knee_mean_angle, axis=0)
        hip_vel = np.gradient(hip_mean_angle, axis=0)
        special['knee_hip_synchrony_3d'] = np.abs(knee_vel - hip_vel)

        return special

    @classmethod
    def extract_stream7_disambiguation_3d(cls, kps: np.ndarray, angles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Stream 7: [취약점 보완 핵심] 포워드/백워드 런지 및 스티프/일반 데드리프트 분별 특화 지표 (7개)
        1. 골반 기준 좌/우 발목 부호화 Z축 변위 (signed_ankle_z_rel_l/r_3d)
        2. 스텝 발 3D 이동 변위 (stepping_foot_dx/dz_3d)
        3. 정강이 뼈대 지면 수직 각도 (shank_verticality_angle_l/r_3d)
        4. 고관절 대 무릎 가동범위 비율 (hip_to_knee_rom_ratio_3d)
        """
        disam = {}
        T = kps.shape[0]

        # 1. 골반 기준 좌/우 발목 부호화 Z축 변위 (전진: 음수, 후진: 양수)
        pelvis_z = kps[:, Joint15.PELVIS, 2]
        disam['signed_ankle_z_rel_l_3d'] = kps[:, Joint15.LEFT_ANKLE, 2] - pelvis_z
        disam['signed_ankle_z_rel_r_3d'] = kps[:, Joint15.RIGHT_ANKLE, 2] - pelvis_z

        # 2. 스텝 발 골반 로컬 축 투영 변위 및 부호화 교차 궤적
        hip_axis = kps[:, Joint15.RIGHT_HIP, :] - kps[:, Joint15.LEFT_HIP, :] # (T, 3)
        hip_axis_norm = hip_axis / (np.linalg.norm(hip_axis, axis=-1, keepdims=True) + 1e-7)

        p_l0 = kps[0, Joint15.LEFT_ANKLE, :]
        p_r0 = kps[0, Joint15.RIGHT_ANKLE, :]
        disp_l = np.linalg.norm(kps[:, Joint15.LEFT_ANKLE, :] - p_l0, axis=-1)
        disp_r = np.linalg.norm(kps[:, Joint15.RIGHT_ANKLE, :] - p_r0, axis=-1)

        # 전체 시퀀스 중 이동량이 더 큰 발을 스텝 발로 자동 선택
        if np.max(disp_l) >= np.max(disp_r):
            step_vec = kps[:, Joint15.LEFT_ANKLE, :] - p_l0 # (T, 3)
            # 기준발(오른발) 대비 스텝발(왼발)의 골반 축 방향 투영 (오른쪽으로 넘어가면 양수)
            rel_plant_to_step = kps[:, Joint15.LEFT_ANKLE, :] - kps[:, Joint15.RIGHT_ANKLE, :]
            crossover_lateral = np.sum(rel_plant_to_step * hip_axis_norm, axis=-1)
        else:
            step_vec = kps[:, Joint15.RIGHT_ANKLE, :] - p_r0 # (T, 3)
            # 기준발(왼발) 대비 스텝발(오른발)의 골반 축 방향 투영 (왼쪽으로 넘어가면 양수)
            rel_plant_to_step = kps[:, Joint15.RIGHT_ANKLE, :] - kps[:, Joint15.LEFT_ANKLE, :]
            crossover_lateral = np.sum(rel_plant_to_step * (-hip_axis_norm), axis=-1)

        disam['stepping_foot_dx_3d'] = crossover_lateral # 골반 정렬 부호화 횡방향 교차 변위 (>0: 크로스)
        disam['stepping_foot_dz_3d'] = step_vec[:, 2]

        # 3. 정강이 뼈대 지면 수직 각도 (발목->무릎 벡터와 Y축 수직선 [0, 1, 0]의 사잇각)
        tibia_l = kps[:, Joint15.LEFT_KNEE, :] - kps[:, Joint15.LEFT_ANKLE, :] # (T, 3)
        tibia_r = kps[:, Joint15.RIGHT_KNEE, :] - kps[:, Joint15.RIGHT_ANKLE, :] # (T, 3)

        norm_l = np.linalg.norm(tibia_l, axis=-1) + 1e-7
        norm_r = np.linalg.norm(tibia_r, axis=-1) + 1e-7

        # Y축 성분의 코사인 각도 (0도가 완전 수직)
        cos_l = np.clip(np.abs(tibia_l[:, 1]) / norm_l, -1.0, 1.0)
        cos_r = np.clip(np.abs(tibia_r[:, 1]) / norm_r, -1.0, 1.0)

        disam['shank_verticality_angle_l_3d'] = np.degrees(np.arccos(cos_l))
        disam['shank_verticality_angle_r_3d'] = np.degrees(np.arccos(cos_r))

        # 4. 순간 고관절 대 무릎 가동범위(변화량) 비율
        knee_mean = (angles['knee_angle_l_3d'] + angles['knee_angle_r_3d']) / 2.0
        hip_mean = (angles['hip_angle_l_3d'] + angles['hip_angle_r_3d']) / 2.0

        d_knee = np.abs(knee_mean - knee_mean[0])
        d_hip = np.abs(hip_mean - hip_mean[0])

        disam['hip_to_knee_rom_ratio_3d'] = (d_hip + 1.0) / (d_knee + 1.0)

        return disam

    @classmethod
    def build_format_b_timeseries_3d(cls, kps_norm_3d: np.ndarray) -> np.ndarray:
        """
        Format B: 64개 프레임 전 구간에 걸친 92차원 시계열 텐서 생성 -> Shape: (64, 92)
        - Stream 1 (Bones 3D): 14 edges * 4 (dx, dy, dz, length_3d) = 56D
        - Stream 2 (Angles 3D): 8 joint angles = 8D
        - Stream 3 (Alignments 3D): 8 defect/alignment metrics = 8D
        - Stream 4 (Asymmetry 3D): 2 asymmetry metrics = 2D
        - Stream 5 (Angular Velocity 3D): 8 angular velocities = 8D
        - Stream 6 (Special 3D): 3 exercise-specific metrics = 3D
        - Stream 7 (Disambiguation 3D): 7 취약점 분별 특화 채널 = 7D
        Total = 56 + 8 + 8 + 2 + 8 + 3 + 7 = 92D
        """
        T = kps_norm_3d.shape[0] # 64
        # Stream 1
        s1 = cls.extract_stream1_bones_3d(kps_norm_3d) # (64, 56)

        # Stream 2
        s2_dict = cls.extract_stream2_angles_3d(kps_norm_3d)
        angle_keys = ['knee_angle_l_3d', 'knee_angle_r_3d', 'hip_angle_l_3d', 'hip_angle_r_3d',
                      'ankle_angle_l_3d', 'ankle_angle_r_3d', 'elbow_angle_l_3d', 'elbow_angle_r_3d']
        s2_list = [s2_dict[k][:, None] for k in angle_keys]
        s2 = np.concatenate(s2_list, axis=-1) # (64, 8)

        # Stream 3
        s3_dict = cls.extract_stream3_alignment_3d(kps_norm_3d)
        align_keys = ['torso_pitch_3d', 'torso_roll_3d', 'knee_valgus_l_3d', 'knee_valgus_r_3d',
                      'sh_hip_twist_3d', 'knee_distance_3d', 'ankle_distance_3d', 'ankle_z_spread_3d']
        s3_list = [s3_dict[k][:, None] for k in align_keys]
        s3 = np.concatenate(s3_list, axis=-1) # (64, 8)

        # Stream 4
        s4_dict = cls.extract_stream4_asymmetry_3d(s2_dict)
        s4 = np.concatenate([s4_dict['knee_asymmetry_3d'][:, None], s4_dict['hip_asymmetry_3d'][:, None]], axis=-1) # (64, 2)

        # Stream 5
        s5 = cls.extract_stream5_velocity_3d(s2) # (64, 8)

        # Stream 6
        s6_dict = cls.extract_stream6_special_3d(kps_norm_3d, s2_dict)
        spec_keys = ['neck_spine_alignment_3d', 'barbell_body_distance_3d', 'knee_hip_synchrony_3d']
        s6_list = [s6_dict[k][:, None] for k in spec_keys]
        s6 = np.concatenate(s6_list, axis=-1) # (64, 3)

        # Stream 7: 취약점 분별 특화
        s7_dict = cls.extract_stream7_disambiguation_3d(kps_norm_3d, s2_dict)
        disam_keys = ['signed_ankle_z_rel_l_3d', 'signed_ankle_z_rel_r_3d',
                      'stepping_foot_dx_3d', 'stepping_foot_dz_3d',
                      'shank_verticality_angle_l_3d', 'shank_verticality_angle_r_3d',
                      'hip_to_knee_rom_ratio_3d']
        s7_list = [s7_dict[k][:, None] for k in disam_keys]
        s7 = np.concatenate(s7_list, axis=-1) # (64, 7)

        format_b = np.concatenate([s1, s2, s3, s4, s5, s6, s7], axis=-1).astype(np.float32) # (64, 92)
        return format_b

    @classmethod
    def compute_active_motion_mask(cls, kps_norm_3d: np.ndarray, s2_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        골반(Pelvis) 높이 변위 및 무릎 굴곡 각도 기반 유효 운동 프레임(Active Motion Mask) 산출
        - 1회, 2회, 3회 등 모든 반복 횟수의 하강/최하사점/상승 구간을 100% 포용
        - 서 있는 정적 대기 구간 및 운동 종료 후 고개 떨굼/발 움직임 잉여 구간만 배제
        """
        T = kps_norm_3d.shape[0]
        knee_mean = (s2_dict['knee_angle_l_3d'] + s2_dict['knee_angle_r_3d']) / 2.0

        # 1. 무릎 굴곡 기준: 165도 미만 (완전히 서 있을 때는 170~180도)
        knee_active = (knee_mean < 165.0)

        # 2. 골반 수직 하강 기준: 전체 ROM의 15% 이상 하강한 구간
        pelvis_y = kps_norm_3d[:, Joint15.PELVIS, 1]
        rom = np.max(pelvis_y) - np.min(pelvis_y)
        pelvis_active = (pelvis_y - np.min(pelvis_y)) >= (rom * 0.15) if rom > 0.05 else np.zeros(T, dtype=bool)

        active_mask = knee_active | pelvis_active

        # [안전 가드 Fallback]: 거의 움직임이 없는 극단적 영상인 경우 전체 프레임 보존
        if np.sum(active_mask) < 2:
            active_mask = np.ones(T, dtype=bool)

        return active_mask

    @classmethod
    def build_format_a_summary_3d(cls, kps_norm_3d: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Format A: 3D 기하 및 생체역학 요약 통계 피처 -> Shape: (161,)
        - 28개 3D 지표 통계치 (mean, std, min, max, delta): 28 * 5 = 140D
        - 최대 굴곡 시점(최저점) 21개 핵심 3D 스냅샷 = 21D
        Total = 161D
        """
        # 1. 28개 지표 추출
        s2_dict = cls.extract_stream2_angles_3d(kps_norm_3d)
        s3_dict = cls.extract_stream3_alignment_3d(kps_norm_3d)
        s4_dict = cls.extract_stream4_asymmetry_3d(s2_dict)
        s6_dict = cls.extract_stream6_special_3d(kps_norm_3d, s2_dict)
        s7_dict = cls.extract_stream7_disambiguation_3d(kps_norm_3d, s2_dict)

        all_signals = {}
        all_signals.update(s2_dict)
        all_signals.update(s3_dict)
        all_signals.update(s4_dict)
        all_signals.update(s6_dict)
        all_signals.update(s7_dict)

        feat_vals = []
        feat_names = []

        # 유효 운동 프레임 동적 마스크 산출 (운동 중 프레임만 통계량 집계)
        active_mask = cls.compute_active_motion_mask(kps_norm_3d, s2_dict)

        # 28개 지표 각각 5가지 요약 통계 (28 * 5 = 140D)
        for name in cls.FEATURE_NAMES_28_3D:
            sig = all_signals[name] # (T,)
            active_sig = sig[active_mask] # 유효 운동 프레임만 슬라이싱
            mean_v = float(np.mean(active_sig))
            std_v = float(np.std(active_sig))
            min_v = float(np.min(active_sig))
            max_v = float(np.max(active_sig))
            delta_v = max_v - min_v

            feat_vals.extend([mean_v, std_v, min_v, max_v, delta_v])
            feat_names.extend([
                f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max", f"{name}_delta"
            ])

        # 2. 최대 굴곡 시점 (Lowest/Bottom Point) 스냅샷 피처 (21D)
        knee_mean = (s2_dict['knee_angle_l_3d'] + s2_dict['knee_angle_r_3d']) / 2.0
        bottom_idx = int(np.argmin(knee_mean)) # 무릎 각도가 가장 작아지는 순간

        snapshot_keys = [
            'knee_angle_l_3d', 'knee_angle_r_3d', 'hip_angle_l_3d', 'hip_angle_r_3d',
            'ankle_angle_l_3d', 'ankle_angle_r_3d', 'elbow_angle_l_3d', 'elbow_angle_r_3d',
            'torso_pitch_3d', 'torso_roll_3d', 'knee_valgus_l_3d', 'knee_valgus_r_3d',
            'sh_hip_twist_3d', 'knee_distance_3d', 'ankle_z_spread_3d', 'neck_spine_alignment_3d',
            'signed_ankle_z_rel_l_3d', 'signed_ankle_z_rel_r_3d',
            'shank_verticality_angle_l_3d', 'shank_verticality_angle_r_3d',
            'hip_to_knee_rom_ratio_3d'
        ]

        for k in snapshot_keys:
            feat_vals.append(float(all_signals[k][bottom_idx]))
            feat_names.append(f"{k}_bottom_snapshot")

        format_a = np.array(feat_vals, dtype=np.float32)
        return format_a, feat_names

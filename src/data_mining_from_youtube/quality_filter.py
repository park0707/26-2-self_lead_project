# -*- coding: utf-8 -*-
"""
4단계 자동 품질 게이팅 및 카메라 뷰포인트 분류기
"""
import numpy as np
from typing import Tuple, Dict, Any
from src.core.skeleton2d import Joint15

class YouTubeQualityFilter:
    """수집된 3D 스켈레톤 시퀀스의 품질 적합성 자동 평가 모듈"""

    @staticmethod
    def evaluate(
        kps_3d: np.ndarray,
        visibilities: np.ndarray,
        min_frames: int = 30,
        min_visibility: float = 0.50,
        max_occluded_ratio: float = 0.50,
        min_motion_rom: float = 0.08
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        3D 관절 및 가시성 시계열을 평가하여 데이터셋 적합 여부 판단
        반환: (is_valid, reject_reason, metadata_dict)
        """
        T = kps_3d.shape[0]
        meta: Dict[str, Any] = {'total_frames': T}

        # 1. 최소 프레임 검사
        if T < min_frames:
            return False, f"Too short sequence (frames={T} < {min_frames})", meta

        # 2. 신체 시인성 (Visibility) 검사
        # 측면/사선 뷰에서는 카메라 반대편 관절이 자연 가림(self-occlusion)되므로, 카메라를 향한 주 시인성 측(Dominant Side)을 기준 평가
        left_joints = [Joint15.LEFT_HIP, Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE, Joint15.LEFT_SHOULDER]
        right_joints = [Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE, Joint15.RIGHT_SHOULDER]
        vis_l = np.mean(visibilities[:, left_joints], axis=-1)
        vis_r = np.mean(visibilities[:, right_joints], axis=-1)
        dominant_vis_per_frame = np.maximum(vis_l, vis_r)
        
        mean_vis = float(np.mean(dominant_vis_per_frame))
        meta['mean_visibility'] = mean_vis

        # 영상 전체 평균 시인성이 0.35 미만이면 인물 미인식/비정상 영상으로 판정
        if mean_vis < 0.35:
            return False, f"Low human visibility (mean_vis={mean_vis:.2f} < 0.35)", meta

        # 3. 가동범위 / 정적 영상 검사 (Motion Dynamic ROM)
        # MediaPipe pose_world_landmarks는 Pelvis가 원점(0,0,0)에 고정되므로,
        # Pelvis 대비 주요 가동 관절(발목, 무릎, 손목, 어깨 등)의 최대 수직 변위 진폭을 평가
        joint_roms_y = [float(np.ptp(kps_3d[:, j, 1])) for j in range(15)]
        max_joint_rom_y = float(max(joint_roms_y))
        ankle_rom_y = float(max(np.ptp(kps_3d[:, Joint15.LEFT_ANKLE, 1]), np.ptp(kps_3d[:, Joint15.RIGHT_ANKLE, 1])))
        wrist_rom_y = float(max(np.ptp(kps_3d[:, Joint15.LEFT_WRIST, 1]), np.ptp(kps_3d[:, Joint15.RIGHT_WRIST, 1])))
        knee_rom_y = float(max(np.ptp(kps_3d[:, Joint15.LEFT_KNEE, 1]), np.ptp(kps_3d[:, Joint15.RIGHT_KNEE, 1])))

        meta['max_joint_rom_y'] = max_joint_rom_y
        meta['ankle_rom_y'] = ankle_rom_y
        meta['wrist_rom_y'] = wrist_rom_y
        meta['knee_rom_y'] = knee_rom_y

        if max_joint_rom_y < min_motion_rom:
            return False, f"Static pose / insufficient motion (max_joint_rom_y={max_joint_rom_y:.3f}m < {min_motion_rom}m)", meta

        # 4. 카메라 뷰포인트 분류 (Sagittal / Coronal / Oblique)
        sh_vec = kps_3d[:, Joint15.RIGHT_SHOULDER, :] - kps_3d[:, Joint15.LEFT_SHOULDER, :] # (T, 3)
        mean_sh = np.mean(sh_vec, axis=0)
        # 어깨 벡터와 X축(정면 방향) 간의 사잇각
        angle_to_cam = np.degrees(np.arctan2(np.abs(mean_sh[2]), np.abs(mean_sh[0]) + 1e-7))

        if angle_to_cam > 55.0:
            viewpoint = "sagittal" # 측면
        elif angle_to_cam < 30.0:
            viewpoint = "coronal" # 정면
        else:
            viewpoint = "oblique" # 사선(대각선)

        meta['viewpoint'] = viewpoint
        meta['view_angle_deg'] = float(angle_to_cam)

        return True, "Passed all quality gates", meta

# =========================================================================================================
# [스크립트명]: kinematic_coordination.py
# [위치]: src/features/kinematic_coordination.py
# [목적]: 생체역학적 협응성 및 운동 경향성(Kinematic Coordination Features, KCF) 고분별 피처 추출 엔진
# [핵심 원리]:
#   - 절대적 각도 스냅샷(절대 무릎각, 체간 기울기 등)에 의한 도메인 과적합 완벽 탈피
#   - Continuous Relative Phase (CRP), Modified Vector Coding (4사분면 협응 패턴),
#     Angle-Angle Cyclograms (Green 정리 면적, 주축 배향각), 무차원 가동범위/각속도 시너지 비율 추출
#   - 해부학적 관절 사잇각 및 뼈대 축 정렬(팔꿈치 굴곡, 골반-발목 축 평행도) 결합
# =========================================================================================================

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class KinematicCoordinationExtractor:
    """
    3D 시계열 텐서 Format B (N, T=64, D>=91)로부터
    개인차 및 카메라 화각에 불변인 운동 경향성/협응성 지표(KCF)를 벡터화하는 고분별 추출기
    """

    FEATURE_NAMES: List[str] = [
        # 1. 무릎-고관절 Continuous Relative Phase (CRP)
        "crp_knee_hip_in_phase_ratio",
        "crp_knee_hip_anti_phase_ratio",
        "crp_knee_hip_marp",
        "crp_knee_hip_dp",
        # 2. 정강이-고관절 CRP
        "crp_shank_hip_in_phase_ratio",
        "crp_shank_hip_anti_phase_ratio",
        "crp_shank_hip_marp",
        "crp_shank_hip_dp",
        # 3. 무릎-고관절 Modified Vector Coding (4개 협응 사분면 비율)
        "vc_knee_hip_q1_in_phase",
        "vc_knee_hip_q2_hip_dominant",
        "vc_knee_hip_q3_anti_phase",
        "vc_knee_hip_q4_knee_dominant",
        "vc_knee_hip_circ_std",
        # 4. 정강이-무릎 Modified Vector Coding
        "vc_shank_knee_q1_in_phase",
        "vc_shank_knee_q2_shank_dominant",
        "vc_shank_knee_q3_anti_phase",
        "vc_shank_knee_q4_knee_dominant",
        "vc_shank_knee_circ_std",
        # 5. Angle-Angle Cyclograms (무릎-고관절 폐곡선 기하 특성)
        "cyclogram_knee_hip_hysteresis_area",
        "cyclogram_knee_hip_norm_area",
        "cyclogram_knee_hip_orientation_deg",
        "cyclogram_knee_hip_aspect_ratio",
        "cyclogram_knee_hip_perimeter",
        # 6. Angle-Angle Cyclograms (정강이-무릎 폐곡선 기하 특성)
        "cyclogram_shank_knee_hysteresis_area",
        "cyclogram_shank_knee_norm_area",
        "cyclogram_shank_knee_orientation_deg",
        # 7. 무차원 가동범위(ROM) 비율 및 지표
        "rom_knee_mean",
        "rom_hip_mean",
        "rom_shank_mean",
        "rom_torso_pitch",
        "rom_ratio_knee_to_hip",
        "rom_ratio_hip_to_knee",
        "rom_ratio_shank_to_knee",
        "rom_ratio_shank_to_hip",
        "rom_ratio_torso_to_hip",
        "rom_ratio_torso_to_knee",
        # 8. 각속도 동역학 및 시너지 지표
        "vel_ratio_max_knee_to_hip",
        "vel_ratio_mean_knee_to_hip",
        "vel_knee_hip_correlation",
        "angle_knee_hip_correlation",
        "phase_lag_knee_hip_bottom",
        "time_knee_bottom_ratio",
        "time_hip_bottom_ratio",
        # 9. 좌우 대칭성 및 지지기반 안정성 지표
        "lai_knee_rom",
        "lai_hip_rom",
        "ankle_distance_rom",
        "pelvis_ankle_cos_sim",
        # 10. 해부학적 불변 지표 (상하체 협응 및 바벨 견착/파지 분별)
        "elbow_angle_min",
        "elbow_angle_mean",
        "arm_vertical_hang_mean",
        "rom_elbow_mean",
        "rom_ratio_upper_to_lower",
        "barbell_dist_min",
        "barbell_body_dist_rom",
        "knee_valgus_rom_mean"
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return list(cls.FEATURE_NAMES)

    @classmethod
    def extract_features(cls, X_b: np.ndarray) -> np.ndarray:
        """
        Format B 텐서 (N, T=64, D>=91)로부터 고분별 KCF 피처 매트릭스 산출
        """
        N, T, D = X_b.shape
        eps = 1e-6

        # 관절 각도 추출
        knee_l, knee_r = X_b[:, :, 56], X_b[:, :, 57]
        hip_l, hip_r = X_b[:, :, 58], X_b[:, :, 59]
        ankle_l, ankle_r = X_b[:, :, 60], X_b[:, :, 61]
        elbow_l, elbow_r = X_b[:, :, 62], X_b[:, :, 63]

        knee_mean = (knee_l + knee_r) / 2.0   # (N, T)
        hip_mean = (hip_l + hip_r) / 2.0     # (N, T)
        elbow_mean = (elbow_l + elbow_r) / 2.0 # (N, T)

        # 체간 기울기 및 정렬
        torso_pitch = X_b[:, :, 64] # (N, T)
        valgus_l, valgus_r = X_b[:, :, 66], X_b[:, :, 67]
        ankle_dist = X_b[:, :, 70]

        # 정강이 지면 각도
        if D >= 91:
            shank_l = X_b[:, :, 89]
            shank_r = X_b[:, :, 90]
        else:
            dy_l = np.abs(X_b[:, :, 45])
            len_l = X_b[:, :, 47] + eps
            dy_r = np.abs(X_b[:, :, 53])
            len_r = X_b[:, :, 55] + eps
            shank_l = np.degrees(np.arccos(np.clip(dy_l / len_l, -1.0, 1.0)))
            shank_r = np.degrees(np.arccos(np.clip(dy_r / len_r, -1.0, 1.0)))
        shank_mean = (shank_l + shank_r) / 2.0 # (N, T)

        barbell_dist = X_b[:, :, 83] if D >= 84 else np.zeros((N, T), dtype=np.float32)

        # =========================================================================
        # 1 & 2. Continuous Relative Phase (CRP) 산출 함수
        # =========================================================================
        def compute_phase_angles(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a_min = np.min(angles, axis=1, keepdims=True)
            a_max = np.max(angles, axis=1, keepdims=True)
            a_norm = 2.0 * (angles - a_min) / (a_max - a_min + eps) - 1.0 # [-1, 1]

            vel = np.gradient(angles, axis=1) # (N, T)
            v_max = np.max(np.abs(vel), axis=1, keepdims=True) + eps
            v_norm = vel / v_max

            phi = np.arctan2(v_norm, a_norm) # [-pi, pi]
            return phi, vel

        phi_knee, vel_knee = compute_phase_angles(knee_mean)
        phi_hip, vel_hip = compute_phase_angles(hip_mean)
        phi_shank, vel_shank = compute_phase_angles(shank_mean)

        def compute_crp_metrics(phi1: np.ndarray, phi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            diff = np.abs(np.degrees(phi1 - phi2))
            crp = np.minimum(diff, 360.0 - diff) # (N, T)

            in_phase = np.mean(crp < 35.0, axis=1) # 동위상 비율 (<35도)
            anti_phase = np.mean(crp > 115.0, axis=1) # 역위상 비율 (>115도)
            marp = np.mean(crp, axis=1) # Mean Absolute Relative Phase
            dp = np.std(crp, axis=1)    # Deviation Phase (협응 안정성)
            return in_phase, anti_phase, marp, dp

        crp_kh_in, crp_kh_anti, crp_kh_marp, crp_kh_dp = compute_crp_metrics(phi_knee, phi_hip)
        crp_sh_in, crp_sh_anti, crp_sh_marp, crp_sh_dp = compute_crp_metrics(phi_shank, phi_hip)

        # =========================================================================
        # 3 & 4. Modified Vector Coding (Needham et al.)
        # =========================================================================
        def compute_vector_coding_bins(ang1: np.ndarray, ang2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            d1 = np.diff(ang1, axis=1)
            d2 = np.diff(ang2, axis=1)
            gamma = (np.degrees(np.arctan2(d2, d1)) + 360.0) % 360.0

            q1 = ((gamma >= 22.5) & (gamma < 67.5)) | ((gamma >= 202.5) & (gamma < 247.5))
            q2 = ((gamma >= 67.5) & (gamma < 112.5)) | ((gamma >= 247.5) & (gamma < 292.5))
            q3 = ((gamma >= 112.5) & (gamma < 157.5)) | ((gamma >= 292.5) & (gamma < 337.5))
            q4 = ((gamma >= 157.5) & (gamma < 202.5)) | (gamma >= 337.5) | (gamma < 22.5)

            p_q1 = np.mean(q1, axis=1)
            p_q2 = np.mean(q2, axis=1)
            p_q3 = np.mean(q3, axis=1)
            p_q4 = np.mean(q4, axis=1)

            rad_gamma = 2.0 * np.radians(gamma)
            r_bar = np.sqrt(np.mean(np.cos(rad_gamma), axis=1)**2 + np.mean(np.sin(rad_gamma), axis=1)**2)
            r_bar = np.clip(r_bar, 1e-4, 1.0)
            circ_std = np.sqrt(-2.0 * np.log(r_bar)) / 2.0

            return p_q1, p_q2, p_q3, p_q4, circ_std

        vc_kh_q1, vc_kh_q2, vc_kh_q3, vc_kh_q4, vc_kh_cstd = compute_vector_coding_bins(knee_mean, hip_mean)
        vc_sk_q1, vc_sk_q2, vc_sk_q3, vc_sk_q4, vc_sk_cstd = compute_vector_coding_bins(knee_mean, shank_mean)

        # =========================================================================
        # 5 & 6. Angle-Angle Cyclograms (Green 정리 면적 및 주축 배향각)
        # =========================================================================
        def compute_cyclogram_features(ang_x: np.ndarray, ang_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_t = ang_x[:, :-1]
            x_next = ang_x[:, 1:]
            y_t = ang_y[:, :-1]
            y_next = ang_y[:, 1:]

            last_term = ang_x[:, -1] * ang_y[:, 0] - ang_x[:, 0] * ang_y[:, -1]
            shoelace = np.sum(x_t * y_next - x_next * y_t, axis=1) + last_term
            hyst_area = 0.5 * np.abs(shoelace)

            rom_x = np.ptp(ang_x, axis=1) + eps
            rom_y = np.ptp(ang_y, axis=1) + eps
            norm_area = hyst_area / (rom_x * rom_y + eps)

            x_c = ang_x - np.mean(ang_x, axis=1, keepdims=True)
            y_c = ang_y - np.mean(ang_y, axis=1, keepdims=True)

            var_x = np.var(ang_x, axis=1)
            var_y = np.var(ang_y, axis=1)
            cov_xy = np.mean(x_c * y_c, axis=1)

            orient_deg = 0.5 * np.degrees(np.arctan2(2.0 * cov_xy, var_x - var_y + eps))

            term_diff = var_x - var_y
            term_sqrt = np.sqrt(term_diff**2 + 4.0 * cov_xy**2)
            lambda_1 = 0.5 * (var_x + var_y + term_sqrt)
            lambda_2 = 0.5 * (var_x + var_y - term_sqrt)
            aspect_ratio = np.sqrt(np.maximum(lambda_1, 0.0) / (np.maximum(lambda_2, 0.0) + eps))

            dx = x_next - x_t
            dy = y_next - y_t
            perimeter = np.sum(np.sqrt(dx**2 + dy**2), axis=1)

            return hyst_area, norm_area, orient_deg, aspect_ratio, perimeter

        cyc_kh_area, cyc_kh_norm, cyc_kh_orient, cyc_kh_aspect, cyc_kh_perim = compute_cyclogram_features(knee_mean, hip_mean)
        cyc_sk_area, cyc_sk_norm, cyc_sk_orient, _, _ = compute_cyclogram_features(knee_mean, shank_mean)

        # =========================================================================
        # 7. 무차원 가동범위(ROM) 지표 및 시너지 비율
        # =========================================================================
        rom_knee = np.ptp(knee_mean, axis=1)
        rom_hip = np.ptp(hip_mean, axis=1)
        rom_shank = np.ptp(shank_mean, axis=1)
        rom_torso = np.ptp(torso_pitch, axis=1)
        rom_elbow = np.ptp(elbow_mean, axis=1)

        rom_ratio_kh = rom_knee / (rom_hip + eps)
        rom_ratio_hk = rom_hip / (rom_knee + eps)
        rom_ratio_sk = rom_shank / (rom_knee + eps)
        rom_ratio_sh = rom_shank / (rom_hip + eps)
        rom_ratio_th = rom_torso / (rom_hip + eps)
        rom_ratio_tk = rom_torso / (rom_knee + eps)

        # =========================================================================
        # 8. 각속도 동역학 및 시너지 지표
        # =========================================================================
        max_vel_k = np.max(np.abs(vel_knee), axis=1)
        max_vel_h = np.max(np.abs(vel_hip), axis=1)
        mean_vel_k = np.mean(np.abs(vel_knee), axis=1)
        mean_vel_h = np.mean(np.abs(vel_hip), axis=1)

        vel_ratio_max_kh = max_vel_k / (max_vel_h + eps)
        vel_ratio_mean_kh = mean_vel_k / (mean_vel_h + eps)

        def compute_corr(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
            s1_c = s1 - np.mean(s1, axis=1, keepdims=True)
            s2_c = s2 - np.mean(s2, axis=1, keepdims=True)
            cov = np.mean(s1_c * s2_c, axis=1)
            std1 = np.std(s1, axis=1) + eps
            std2 = np.std(s2, axis=1) + eps
            return np.clip(cov / (std1 * std2), -1.0, 1.0)

        corr_vel_kh = compute_corr(vel_knee, vel_hip)
        corr_ang_kh = compute_corr(knee_mean, hip_mean)

        t_bottom_knee = np.argmin(knee_mean, axis=1) / float(T)
        t_bottom_hip = np.argmin(hip_mean, axis=1) / float(T)
        lag_kh_bottom = t_bottom_knee - t_bottom_hip

        # =========================================================================
        # 9. 좌우 대칭성 및 지지기반 안정성 지표
        # =========================================================================
        rom_kl = np.ptp(knee_l, axis=1)
        rom_kr = np.ptp(knee_r, axis=1)
        lai_knee = np.abs(rom_kl - rom_kr) / (rom_kl + rom_kr + eps)

        rom_hl = np.ptp(hip_l, axis=1)
        rom_hr = np.ptp(hip_r, axis=1)
        lai_hip = np.abs(rom_hl - rom_hr) / (rom_hl + rom_hr + eps)

        ankle_dist_rom = np.ptp(ankle_dist, axis=1)

        # 해부학적 골반축-발목축 평행도 (1.0 = 스쿼트 양발 정렬, 0.45 = 런지 전후 분리)
        hip_vec_x = X_b[:, :, 36] - X_b[:, :, 32]
        hip_vec_z = X_b[:, :, 38] - X_b[:, :, 34]
        l_ankle_x = X_b[:, :, 32] + X_b[:, :, 40] + X_b[:, :, 44]
        l_ankle_z = X_b[:, :, 34] + X_b[:, :, 42] + X_b[:, :, 46]
        r_ankle_x = X_b[:, :, 36] + X_b[:, :, 48] + X_b[:, :, 52]
        r_ankle_z = X_b[:, :, 38] + X_b[:, :, 50] + X_b[:, :, 54]
        ankle_vec_x = r_ankle_x - l_ankle_x
        ankle_vec_z = r_ankle_z - l_ankle_z
        dot = hip_vec_x * ankle_vec_x + hip_vec_z * ankle_vec_z
        norm_h = np.sqrt(hip_vec_x**2 + hip_vec_z**2) + eps
        norm_a = np.sqrt(ankle_vec_x**2 + ankle_vec_z**2) + eps
        cos_sim = np.clip(dot / (norm_h * norm_a), -1.0, 1.0)
        pelvis_ankle_cos_sim = np.mean(cos_sim, axis=1)

        # =========================================================================
        # 10. 해부학적 불변 지표 (상하체 협응 및 바벨 견착/파지 분별)
        # =========================================================================
        elbow_angle_min = np.min(elbow_mean, axis=1)
        elbow_angle_mean = np.mean(elbow_mean, axis=1)

        # 어깨-손목 수직 하강도 (데드리프트 시 팔이 수직으로 늘어짐)
        arm_dy = (X_b[:, :, 17] + X_b[:, :, 21] + X_b[:, :, 25] + X_b[:, :, 29]) / 2.0
        arm_vertical_hang_mean = np.mean(arm_dy, axis=1)

        rom_upper_to_lower = (rom_elbow + eps) / (rom_knee + rom_hip + eps)
        barbell_dist_min = np.min(barbell_dist, axis=1)
        barbell_dist_rom = np.ptp(barbell_dist, axis=1)
        valgus_rom_mean = 0.5 * (np.ptp(valgus_l, axis=1) + np.ptp(valgus_r, axis=1))

        # =========================================================================
        # 55차원 최종 정밀 KCF 피처 매트릭스 결합
        # =========================================================================
        features_list = [
            crp_kh_in, crp_kh_anti, crp_kh_marp, crp_kh_dp,
            crp_sh_in, crp_sh_anti, crp_sh_marp, crp_sh_dp,
            vc_kh_q1, vc_kh_q2, vc_kh_q3, vc_kh_q4, vc_kh_cstd,
            vc_sk_q1, vc_sk_q2, vc_sk_q3, vc_sk_q4, vc_sk_cstd,
            cyc_kh_area, cyc_kh_norm, cyc_kh_orient, cyc_kh_aspect, cyc_kh_perim,
            cyc_sk_area, cyc_sk_norm, cyc_sk_orient,
            rom_knee, rom_hip, rom_shank, rom_torso,
            rom_ratio_kh, rom_ratio_hk, rom_ratio_sk, rom_ratio_sh, rom_ratio_th, rom_ratio_tk,
            vel_ratio_max_kh, vel_ratio_mean_kh, corr_vel_kh, corr_ang_kh,
            lag_kh_bottom, t_bottom_knee, t_bottom_hip,
            lai_knee, lai_hip, ankle_dist_rom, pelvis_ankle_cos_sim,
            elbow_angle_min, elbow_angle_mean, arm_vertical_hang_mean,
            rom_elbow, rom_upper_to_lower, barbell_dist_min, barbell_dist_rom, valgus_rom_mean
        ]

        feature_matrix = np.column_stack(features_list).astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e4, neginf=-1e4)

        return feature_matrix

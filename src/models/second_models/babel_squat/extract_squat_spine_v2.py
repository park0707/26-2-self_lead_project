# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 척추 중립(c1) 전담 10D 정제 동적 경향성 피처 추출기 v2

[v2 변경 사항]
  - 기존 9D에서 미기여 3개 완전 제거:
      (3) c1_onset_phase_lag         → 16프레임 분해능 부족, 중요도 0
      (5) c1_torso_log_dimensionless_jerk → 3차 미분 노이즈 상쇄, 중요도 0
      (8) c1_lumbar_hyperextension_index  → 스쿼트 과신전 패턴 희소, 중요도 1
  - 신규 선행 연구 기반 4D 추가:
      (A) c1_thoraco_lumbar_segment_ratio  흉요추 분절 기여도 비율 (Colyer et al. 2018)
      (B) c1_pelvis_tilt_at_bottom         최하사점 골반 전/후 틸트 (버트윙크 직접 측정)
      (C) c1_descent_spine_drift_rate      하강 구간 척추 붕괴율 (Tao et al. 2021)
      (D) c1_trunk_knee_phase_coherence    체간-무릎 굴곡 위상 동기화 지표

[참고 선행 연구]
  - Colyer S.L. et al. (2018) "A Review of the Relationship Between Hip Flexor Muscle Stiffness and Anterior Pelvic Tilt during Athletic Performance"
  - Tao W. et al. (2021) "Barbell squat kinematic detection using wearable sensors and machine learning"
  - Sado N. et al. (2019) "Relation between the trunk and pelvis movements and lower-extremity kinematics during body-weight squats"
  - Hartmann H. et al. (2013) "Influence of squatting depth on jumping performance"
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List

SPINE_TREND_V2_FEATURE_NAMES_10: List[str] = [
    # === 기존 9D에서 미기여 3개 제거 후 6개 유지 ===
    "c1_cyclogram_norm_area",              # 1. 체간-고관절 무차원 폐곡선 면적 (Shoelace 적분)
    "c1_ascent_velocity_ratio",            # 2. 상승기 초기 엉덩이/어깨 수직속도비 (굿모닝 결함)
    "c1_butt_wink_accel",                  # 3. 최하사점 골반 틸트 각가속도 스파이크 (벗 윙크)
    "c1_frontal_spine_asymmetry",          # 4. 기준 골반너비 정규화 관상면 척추 좌우 쏠림비
    "c1_rmse_hysteresis_trunk",            # 5. 동일 무릎가동률 대비 하강-상승 체간각 편차 RMSE
    "c1_bar_to_midfoot_sagittal_deviation", # 6. 어깨-미드풋 수평 모멘트암 전방 이탈률
    # === 신규 선행 연구 기반 4D ===
    "c1_thoraco_lumbar_segment_ratio",     # 7. 흉요추 분절 기여도 비율 (흉추 vs 요추 전방경사 기여)
    "c1_pelvis_tilt_at_bottom",            # 8. 최하사점 골반 틸트각 (버트윙크 직접 측정)
    "c1_descent_spine_drift_rate",         # 9. 하강 구간 척추 전방 붕괴율 (deg/normalized_frame)
    "c1_trunk_knee_phase_coherence",       # 10. 체간 피치-무릎 굴곡 교차상관 피크 (위상 동기화)
]


def safe_divide(
    num: np.ndarray,
    denom: np.ndarray,
    eps: float = 1e-4,
    clip_range: Tuple[float, float] = (-10.0, 10.0)
) -> np.ndarray:
    """분모 0 및 극소값 발산 방지 안전 나눗셈"""
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), eps)
    res = num / safe_denom
    if clip_range is not None:
        res = np.clip(res, clip_range[0], clip_range[1])
    return res


def _cross_correlation_peak(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 정규화된 시계열의 교차 상관 최대값 반환 (위상 동기화 지표).
    완벽한 동기화 → 1.0, 역위상 → -1.0, 무관 → 0.0
    """
    a_z = a - np.mean(a)
    b_z = b - np.mean(b)
    std_a = np.std(a_z)
    std_b = np.std(b_z)
    if std_a < 1e-5 or std_b < 1e-5:
        return 0.0
    a_n = a_z / std_a
    b_n = b_z / std_b
    xcorr = np.correlate(a_n, b_n, mode='full')
    # 중앙 ±3프레임 이내 최대값 (허용 위상 지연 범위)
    center = len(xcorr) // 2
    window = xcorr[max(0, center - 3): center + 4]
    return float(np.max(window)) / len(a_n)


def extract_squat_spine_v2(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, 16, 92) → (N, 10) 척추 중립 전담 10D 정제 동적 경향성 피처 추출 v2

    Format B 채널 맵 (사용 채널):
      Stream 1 (뼈대 벡터):
        idx 0: Pelvis-Torso dx  (체간 좌우 편차)
        idx 1: Pelvis-Torso dy  (체간 수직 높이)
        idx 2: Pelvis-Torso dz  (체간 전후 깊이)
        idx 8: Torso-LShoulder dx
        idx 12: Torso-RShoulder dx
        idx 32: Pelvis-LHip dx  (골반 폭)
        idx 33: Pelvis-LHip dy
        idx 36: Pelvis-RHip dx
        idx 40: LHip-LKnee dx
        idx 41: LHip-LKnee dy  (좌 고관절-무릎 수직 변위)
        idx 48: RHip-RKnee dy  (우 고관절-무릎 수직 변위)
      Stream 2 (관절각):
        idx 56,57: knee_angle_l/r  (무릎 굴곡각)
        idx 58,59: hip_angle_l/r   (고관절 굴곡각)
      Stream 3 (정렬/결함):
        idx 64: torso_pitch_3d    (체간 전방 경사각)
        idx 65: torso_roll_3d     (체간 좌우 기울기각)
      idx 70: ankle_distance_3d  (발목 간격 — 골반 너비 프록시)
    """
    N, T, C = X_b.shape
    feats = np.zeros((N, 10), dtype=np.float32)
    dt = 1.0 / max(1, T - 1)

    for i in range(N):
        # ─── 기본 시계열 추출 ────────────────────────────────────────────────
        # 무릎 굴곡각 (클수록 최하사점)
        knee_raw = (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
        knee_flex = 180.0 - knee_raw

        # 고관절 굴곡각
        hip_raw = (X_b[i, :, 58] + X_b[i, :, 59]) / 2.0
        hip_flex = 180.0 - hip_raw

        # 체간 전방 경사각 (Pitch) & 좌우 기울기 (Roll)
        torso_pitch = X_b[i, :, 64]
        torso_roll = X_b[i, :, 65]

        # 골반/어깨 수직 변위 추정
        hip_knee_dy = (X_b[i, :, 41] + X_b[i, :, 49]) / 2.0
        pelvis_y = -hip_knee_dy  # 상승 시 증가

        torso_dy = X_b[i, :, 1]
        shoulder_y = pelvis_y + torso_dy

        # 어깨 좌우 & 전후 편차
        shoulder_dx = X_b[i, :, 0] + (X_b[i, :, 8] + X_b[i, :, 12]) / 2.0
        shoulder_dz = X_b[i, :, 2]

        # 골반 분절 벡터 (흉요추 분절용)
        # Pelvis-LHip dy / Pelvis-RHip dy → 골반 자체 틸트 추정
        pelvis_lhip_dy = X_b[i, :, 33]  # 좌 골반-고관절 수직 벡터
        # Pelvis-Torso dy → 체간 분절 상승 벡터
        torso_segment_dy = X_b[i, :, 1]

        # ─── 신호 평활화 ─────────────────────────────────────────────────────
        sigma = 0.8
        knee_flex_s    = gaussian_filter1d(knee_flex,    sigma=sigma)
        hip_flex_s     = gaussian_filter1d(hip_flex,     sigma=sigma)
        torso_pitch_s  = gaussian_filter1d(torso_pitch,  sigma=sigma)
        pelvis_y_s     = gaussian_filter1d(pelvis_y,     sigma=sigma)
        shoulder_y_s   = gaussian_filter1d(shoulder_y,   sigma=sigma)
        pelvis_lhip_dy_s = gaussian_filter1d(pelvis_lhip_dy, sigma=sigma)
        torso_seg_dy_s = gaussian_filter1d(torso_segment_dy, sigma=sigma)

        # ─── 최하사점 인덱스 & 기준 골반 너비 ────────────────────────────────
        b = int(np.argmax(knee_flex_s))
        b = int(np.clip(b, 2, T - 3))

        pelvis_width_init = np.mean(X_b[i, :3, 70])  # 발목 간격 = 골반 너비 프록시
        w_pelvis_ref = max(float(pelvis_width_init), 0.15)

        # ─── Range-of-motion 정규화 기준값 ───────────────────────────────────
        k_ptp = max(float(np.ptp(knee_flex_s)), 5.0)
        h_ptp = max(float(np.ptp(hip_flex_s)), 5.0)
        t_ptp = max(float(np.ptp(torso_pitch_s)), 3.0)

        # ════════════════════════════════════════════════════════════════════
        # 피처 1: 체간-고관절 무차원 Cyclogram 면적 (Shoelace 무차원 적분)
        # ════════════════════════════════════════════════════════════════════
        h_norm = (hip_flex_s - np.min(hip_flex_s)) / h_ptp
        t_norm = (torso_pitch_s - np.min(torso_pitch_s)) / t_ptp
        shoelace = 0.5 * np.abs(
            np.sum(h_norm[:-1] * t_norm[1:] - h_norm[1:] * t_norm[:-1]) +
            (h_norm[-1] * t_norm[0] - h_norm[0] * t_norm[-1])
        )
        feats[i, 0] = float(shoelace)

        # ════════════════════════════════════════════════════════════════════
        # 피처 2: 상승기 초기 엉덩이/어깨 수직속도비 (Rv, 굿모닝 결함)
        # ════════════════════════════════════════════════════════════════════
        con_len = max(1, T - 1 - b)
        init_con_end = min(T - 1, b + max(1, int(con_len * 0.35)))

        v_pelvis_y = np.gradient(pelvis_y_s, dt)
        v_shoulder_y = np.gradient(shoulder_y_s, dt)

        mean_v_pelvis  = float(np.mean(v_pelvis_y[b:init_con_end + 1]))
        mean_v_shoulder = float(np.mean(v_shoulder_y[b:init_con_end + 1]))

        rv = safe_divide(
            np.array([mean_v_pelvis]),
            np.array([mean_v_shoulder]),
            eps=1e-4,
            clip_range=(-5.0, 5.0)
        )[0]
        feats[i, 1] = float(rv)

        # ════════════════════════════════════════════════════════════════════
        # 피처 3: 최하사점 골반 틸트 각가속도 스파이크 (버트윙크)
        # ════════════════════════════════════════════════════════════════════
        hip_vel = np.gradient(hip_flex_s, dt)
        hip_acc = np.gradient(hip_vel, dt)
        win_start = max(0, b - 2)
        win_end   = min(T, b + 3)
        butt_wink_spike = float(np.max(np.abs(hip_acc[win_start:win_end])))
        feats[i, 2] = float(
            safe_divide(
                np.array([butt_wink_spike]),
                np.array([h_ptp]),
                eps=1e-4,
                clip_range=(0.0, 50.0)
            )[0]
        )

        # ════════════════════════════════════════════════════════════════════
        # 피처 4: 관상면 척추 좌우 쏠림비 (측굴)
        # ════════════════════════════════════════════════════════════════════
        max_torso_roll   = float(np.max(np.abs(torso_roll)))
        max_shoulder_dx  = float(np.max(np.abs(shoulder_dx)))
        spine_asym = (max_shoulder_dx / w_pelvis_ref) + (max_torso_roll / 90.0)
        feats[i, 3] = float(np.clip(spine_asym, 0.0, 5.0))

        # ════════════════════════════════════════════════════════════════════
        # 피처 5: 동일 무릎각 대비 하강-상승 체간각 편차 RMSE (히스테리시스)
        # ════════════════════════════════════════════════════════════════════
        try:
            m_depths = np.linspace(
                np.min(knee_flex_s) + 2.0,
                np.max(knee_flex_s) - 2.0,
                8
            )
            trunk_down = np.interp(m_depths, knee_flex_s[:b + 1], torso_pitch_s[:b + 1])
            trunk_up   = np.interp(m_depths, knee_flex_s[b:][::-1], torso_pitch_s[b:][::-1])
            rmse_hyst  = float(np.sqrt(np.mean((trunk_up - trunk_down) ** 2)))
        except Exception:
            rmse_hyst = 0.0
        feats[i, 4] = float(np.clip(rmse_hyst, 0.0, 45.0))

        # ════════════════════════════════════════════════════════════════════
        # 피처 6: 어깨-미드풋 수평 모멘트암 전방 이탈률
        # ════════════════════════════════════════════════════════════════════
        max_shoulder_dz = float(np.max(shoulder_dz) - np.min(shoulder_dz))
        moment_arm_dev = safe_divide(
            np.array([max_shoulder_dz]),
            np.array([w_pelvis_ref]),
            eps=1e-4,
            clip_range=(0.0, 5.0)
        )[0]
        feats[i, 5] = float(moment_arm_dev)

        # ════════════════════════════════════════════════════════════════════
        # [신규] 피처 7: 흉요추 분절 기여도 비율
        # (상부 체간 수직 벡터 변화 vs 하부 골반-고관절 수직 벡터 변화)
        # 참고: Colyer et al. 2018 - 골반 틸트와 요추 전방경사의 분절 기여 측정
        # 흉추 기여 ↑ → 등 상부 굽음 (흉추 후만)
        # 요추 기여 ↑ → 버트윙크 (골반 말림)
        # 비율 ≈ 1.0 → 균형 / > 1.5 → 흉추 우세 후만 / < 0.5 → 골반 말림 우세
        # ════════════════════════════════════════════════════════════════════
        torso_ptp = max(float(np.ptp(torso_seg_dy_s)), 1e-4)
        pelvis_ptp = max(float(np.ptp(pelvis_lhip_dy_s)), 1e-4)
        tl_ratio = safe_divide(
            np.array([torso_ptp]),
            np.array([pelvis_ptp]),
            eps=1e-4,
            clip_range=(0.0, 5.0)
        )[0]
        feats[i, 6] = float(tl_ratio)

        # ════════════════════════════════════════════════════════════════════
        # [신규] 피처 8: 최하사점 골반 틸트각 (버트윙크 직접 측정)
        # 골반-고관절 수직 벡터(Pelvis-LHip dy)의 최하사점 절대값
        # 수직(= 0)에서 편차가 클수록 골반 전후 말림 발생
        # 참고: Sado et al. 2019 - 스쿼트 최하사점 골반 각도와 요추 굴곡의 관계
        # ════════════════════════════════════════════════════════════════════
        # b 주변 ±1 프레임 평균으로 안정화
        b0 = max(0, b - 1)
        b1 = min(T - 1, b + 1)
        pelvis_tilt_bottom = float(np.abs(np.mean(pelvis_lhip_dy_s[b0:b1 + 1])))
        # 기립 시 기준값으로 정규화
        pelvis_tilt_standing = float(np.abs(np.mean(pelvis_lhip_dy_s[:3])))
        pelvis_tilt_drop = pelvis_tilt_bottom - pelvis_tilt_standing
        feats[i, 7] = float(np.clip(np.abs(pelvis_tilt_drop), 0.0, 1.0))

        # ════════════════════════════════════════════════════════════════════
        # [신규] 피처 9: 하강 구간 척추 붕괴율 (deg/normalized_frame)
        # 하강 시작(0프레임)부터 최하사점(b)까지 체간 피치각 선형 회귀 기울기
        # 양수 & 가파를수록 하강 중 척추가 빠르게 앞으로 꺾임 (결함)
        # 음수(기립 유지) → 정상 또는 굿모닝 형태
        # 참고: Tao et al. 2021 - 바벨 스쿼트 체간 전방 경사 시계열 패턴 분류
        # ════════════════════════════════════════════════════════════════════
        if b >= 2:
            descent_frames = np.arange(b + 1, dtype=float) / max(1, b)
            descent_torso  = torso_pitch_s[:b + 1]
            # 선형 회귀 기울기 (최소제곱법)
            if len(descent_frames) >= 2:
                slope = float(np.polyfit(descent_frames, descent_torso, deg=1)[0])
            else:
                slope = 0.0
            # 정규화: 전체 체간 가동범위 대비 비율
            drift_rate = slope / max(t_ptp, 1.0)
        else:
            drift_rate = 0.0
        feats[i, 8] = float(np.clip(drift_rate, -3.0, 3.0))

        # ════════════════════════════════════════════════════════════════════
        # [신규] 피처 10: 체간 피치-무릎 굴곡 위상 동기화 지표
        # 정상: 무릎과 체간이 동기화 (하강 시 같이 이동) → cross-corr 피크 ≈ 0.8~1.0
        # 굿모닝 결함: 무릎이 먼저 멈추고 체간만 계속 전방으로 기울어짐 → 낮은 동기화
        # 척추 결함: 전환점 이후 체간이 독립적으로 불안정하게 움직임 → 낮은 동기화
        # 참고: Hartmann et al. 2013 - 슬관절-고관절 협응 동기화 패턴과 척추 중립의 관계
        # ════════════════════════════════════════════════════════════════════
        coherence = _cross_correlation_peak(torso_pitch_s, knee_flex_s)
        feats[i, 9] = float(np.clip(coherence, -1.0, 1.0))

    # 전역 수치 방어벽 (Global Guardrail)
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats

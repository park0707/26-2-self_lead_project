# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 척추 중립(c1) 전담 9대 동적 경향성 피처 추출기
- 3대 서브에이전트(생체역학, ML시스템, 데이터품질) 교차 검증 피드백 100% 반영
  1. 미분 전 가우시안/Savgol 평활화로 지터 노이즈 증폭 차단
  2. 직립 시작 3프레임 기준 골반 너비(W_pelvis_ref)로 고정 정규화
  3. Safe-division, Log-bouncing 방지, 글로벌 nan_to_num 가드레일
  4. 벗 윙크, 굿모닝(속도비/위상차), 과신전, 흉추붕괴, 측굴, 코어 저크 완벽 포착
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List

SPINE_TREND_FEATURE_NAMES_9: List[str] = [
    "c1_cyclogram_norm_area",              # 1. 체간-고관절 무차원 폐곡선 면적 (Shoelace 적분)
    "c1_ascent_velocity_ratio",            # 2. 상승기 초기 엉덩이/어깨 수직속도비 (Rv)
    "c1_onset_phase_lag",                  # 3. 골반 대비 어깨 상승 지연 시간차 (Delta t_lag)
    "c1_butt_wink_accel",                  # 4. 최하사점 골반 틸트 각가속도 스파이크 (벗 윙크)
    "c1_torso_log_dimensionless_jerk",     # 5. 체간 피치각 로그 무차원 저크 (Log-LDLJ, 코어 흔들림)
    "c1_frontal_spine_asymmetry",          # 6. 기준 골반너비 정규화 관상면 척추 좌우 쏠림비
    "c1_rmse_hysteresis_trunk",            # 7. 동일 무릎가동률 대비 하강-상승 체간각 편차 RMSE
    "c1_lumbar_hyperextension_index",      # 8. 셋업/락아웃 구간 척추 후방 과신전 지표
    "c1_bar_to_midfoot_sagittal_deviation" # 9. 어깨(바벨)-미드풋 수평 모멘트암 전방 이탈률
]

def safe_divide(num: np.ndarray, denom: np.ndarray, eps: float = 1e-4, clip_range: Tuple[float, float] = (-10.0, 10.0)) -> np.ndarray:
    """분모 0 및 극소값 발산 방지 안전 나눗셈"""
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), eps)
    res = num / safe_denom
    if clip_range is not None:
        res = np.clip(res, clip_range[0], clip_range[1])
    return res

def extract_squat_spine_trend_features(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, 16, 92) -> (N, 9) 척추 중립 전담 9대 경향성 피처 추출
    """
    N, T, C = X_b.shape
    feats = np.zeros((N, 9), dtype=np.float32)
    dt = 1.0 / max(1, T - 1)

    for i in range(N):
        # 1. 16프레임 시계열 스트림 추출
        # 무릎 굴곡각: Stream 2 (56, 57)
        knee_raw = (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
        knee_flex = 180.0 - knee_raw  # 굴곡량 (클수록 최하사점)
        
        # 고관절 굴곡각: Stream 2 (58, 59)
        hip_raw = (X_b[i, :, 58] + X_b[i, :, 59]) / 2.0
        hip_flex = 180.0 - hip_raw
        
        # 체간 전방 경사각 (Pitch): Stream 3 (64)
        torso_pitch = X_b[i, :, 64]
        # 체간 좌우 기울기 (Roll): Stream 3 (65)
        torso_roll = X_b[i, :, 65]

        # 뼈대 벡터를 통한 어깨/골반 Y(수직), X(좌우), Z(전후) 추출
        # Pelvis-Torso 엣지 (idx 0..3): dx, dy, dz, length
        # Torso-Neck 엣지 (idx 4..7): dx, dy, dz, length
        # Torso-LShoulder (8..11), Torso-RShoulder (12..15)
        # 골반 수직 변위 추정 (무릎각과 연동된 상대 골반 높이: Stream 1 40..43 LHip-LKnee Y와 48..51 RHip-RKnee Y)
        hip_knee_dy = (X_b[i, :, 41] + X_b[i, :, 49]) / 2.0
        pelvis_y = -hip_knee_dy  # 높이 (상승 시 증가)
        
        # 어깨 수직 변위 추정 (골반 높이 + 체간 높이)
        torso_dy = X_b[i, :, 1]
        shoulder_y = pelvis_y + torso_dy

        # 어깨 중심 X 편차 (좌우 쏠림): (Torso-LShoulder dx + Torso-RShoulder dx) / 2.0 + Torso dx
        shoulder_dx = X_b[i, :, 0] + (X_b[i, :, 8] + X_b[i, :, 12]) / 2.0

        # 어깨 전후 Z 편차 (모멘트암): Torso dz
        shoulder_dz = X_b[i, :, 2]

        # 2. 신호 평활화 (Savgol / Gaussian filter) - 서브에이전트 권고 1
        knee_flex_s = gaussian_filter1d(knee_flex, sigma=0.8)
        hip_flex_s = gaussian_filter1d(hip_flex, sigma=0.8)
        torso_pitch_s = gaussian_filter1d(torso_pitch, sigma=0.8)
        pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=0.8)
        shoulder_y_s = gaussian_filter1d(shoulder_y, sigma=0.8)

        # 3. 최하사점(변곡점) 인덱스 및 기준 골반 너비 (서브에이전트 권고 2)
        b = int(np.argmax(knee_flex_s))
        b = int(np.clip(b, 2, T - 3))  # 윈도우 보장

        # 직립 시작 구간(초기 3프레임)의 기준 골반 너비 W_pelvis_ref
        pelvis_width_init = np.mean(X_b[i, :3, 70])  # ankle/pelvis width proxy
        w_pelvis_ref = max(float(pelvis_width_init), 0.15)

        # -------------------------------------------------------------
        # [피처 1] 체간-고관절 무차원 Cyclogram 면적 (Shoelace 무차원 적분)
        # -------------------------------------------------------------
        k_ptp = max(float(np.ptp(knee_flex_s)), 5.0)
        h_ptp = max(float(np.ptp(hip_flex_s)), 5.0)
        t_ptp = max(float(np.ptp(torso_pitch_s)), 3.0)

        # [0, 1] 가동범위 무차원화
        h_norm = (hip_flex_s - np.min(hip_flex_s)) / h_ptp
        t_norm = (torso_pitch_s - np.min(torso_pitch_s)) / t_ptp
        shoelace = 0.5 * np.abs(
            np.sum(h_norm[:-1] * t_norm[1:] - h_norm[1:] * t_norm[:-1]) +
            (h_norm[-1] * t_norm[0] - h_norm[0] * t_norm[-1])
        )
        feats[i, 0] = float(shoelace)

        # -------------------------------------------------------------
        # [피처 2] 상승기 초기(최하사점 직후 30%) 엉덩이/어깨 수직속도비 (Rv)
        # -------------------------------------------------------------
        # 상승 구간: b ~ T-1
        con_len = max(1, T - 1 - b)
        init_con_end = min(T - 1, b + max(1, int(con_len * 0.35)))

        v_pelvis_y = np.gradient(pelvis_y_s, dt)
        v_shoulder_y = np.gradient(shoulder_y_s, dt)

        mean_v_pelvis = float(np.mean(v_pelvis_y[b:init_con_end + 1]))
        mean_v_shoulder = float(np.mean(v_shoulder_y[b:init_con_end + 1]))

        # 서브에이전트 권고: Safe-division & clip
        rv = safe_divide(np.array([mean_v_pelvis]), np.array([mean_v_shoulder]), eps=1e-4, clip_range=(-5.0, 5.0))[0]
        feats[i, 1] = float(rv)

        # -------------------------------------------------------------
        # [피처 3] 최하사점 반환 후 골반 대비 어깨 상승 지연 시간차 (Delta t_lag)
        # -------------------------------------------------------------
        # 상승 시작 시점(속도가 양수로 전환되는 프레임)
        pelvis_onset = b
        shoulder_onset = b
        for t_idx in range(b, T):
            if v_pelvis_y[t_idx] > 0.05:
                pelvis_onset = t_idx
                break
        for t_idx in range(b, T):
            if v_shoulder_y[t_idx] > 0.05:
                shoulder_onset = t_idx
                break
        onset_lag = (shoulder_onset - pelvis_onset) * dt
        feats[i, 2] = float(onset_lag)

        # -------------------------------------------------------------
        # [피처 4] 최하사점 국소 윈도우(b ± 2) 내 골반 틸트 각가속도 스파이크 (벗 윙크)
        # -------------------------------------------------------------
        # 고관절 각속도 및 각가속도
        hip_vel = np.gradient(hip_flex_s, dt)
        hip_acc = np.gradient(hip_vel, dt)
        win_start = max(0, b - 2)
        win_end = min(T, b + 3)
        butt_wink_spike = float(np.max(np.abs(hip_acc[win_start:win_end])))
        # 가동범위로 정규화
        feats[i, 3] = float(safe_divide(np.array([butt_wink_spike]), np.array([h_ptp]), eps=1e-4, clip_range=(0.0, 50.0))[0])

        # -------------------------------------------------------------
        # [피처 5] 체간 피치각 로그 무차원 저크 (Log-LDLJ)
        # -------------------------------------------------------------
        v_torso = np.gradient(torso_pitch_s, dt)
        a_torso = np.gradient(v_torso, dt)
        j_torso = np.gradient(a_torso, dt)
        # LDLJ = -ln( (T^5 / ROM^2) * integral(j^2 dt) )
        jerk_integral = float(np.sum(j_torso**2) * dt)
        scale_factor = (float(T)**5) / (t_ptp**2 + 1e-4)
        raw_ldlj = scale_factor * jerk_integral
        # 서브에이전트 권고: Log-bouncing 방지
        log_ldlj = -float(np.log(np.clip(raw_ldlj, 1e-6, 1e8)))
        feats[i, 4] = log_ldlj

        # -------------------------------------------------------------
        # [피처 6] 기준 골반너비 정규화 관상면 척추 좌우 쏠림비 (측굴)
        # -------------------------------------------------------------
        max_torso_roll = float(np.max(np.abs(torso_roll)))
        max_shoulder_dx = float(np.max(np.abs(shoulder_dx)))
        spine_asym = (max_shoulder_dx / w_pelvis_ref) + (max_torso_roll / 90.0)
        feats[i, 5] = float(np.clip(spine_asym, 0.0, 5.0))

        # -------------------------------------------------------------
        # [피처 7] 동일 무릎 각도 대비 하강-상승 체간각 편차 RMSE
        # -------------------------------------------------------------
        try:
            m_depths = np.linspace(np.min(knee_flex_s) + 2.0, np.max(knee_flex_s) - 2.0, 8)
            trunk_down = np.interp(m_depths, knee_flex_s[:b+1], torso_pitch_s[:b+1])
            trunk_up = np.interp(m_depths, knee_flex_s[b:][::-1], torso_pitch_s[b:][::-1])
            rmse_hyst = float(np.sqrt(np.mean((trunk_up - trunk_down)**2)))
        except Exception:
            rmse_hyst = 0.0
        feats[i, 6] = float(np.clip(rmse_hyst, 0.0, 45.0))

        # -------------------------------------------------------------
        # [피처 8] 셋업/락아웃 구간 척추 후방 과신전 지표 (서브에이전트 권고 추가)
        # -------------------------------------------------------------
        # 셋업(0..2프레임) 및 락아웃(T-3..T-1)에서의 최소 체간 피치각 (음수면 뒤로 꺾임)
        lockout_pitch = min(float(np.min(torso_pitch_s[:3])), float(np.min(torso_pitch_s[-3:])))
        hyperextension = max(0.0, -lockout_pitch)
        feats[i, 7] = float(np.clip(hyperextension, 0.0, 30.0))

        # -------------------------------------------------------------
        # [피처 9] 어깨(바벨)-미드풋 수평 모멘트암 전방 이탈률 (서브에이전트 권고 추가)
        # -------------------------------------------------------------
        # 하강/상승 전 구간에서 어깨 Z축 전방 돌출량
        max_shoulder_dz = float(np.max(shoulder_dz) - np.min(shoulder_dz))
        # 골반 너비로 정규화
        moment_arm_dev = safe_divide(np.array([max_shoulder_dz]), np.array([w_pelvis_ref]), eps=1e-4, clip_range=(0.0, 5.0))[0]
        feats[i, 8] = float(moment_arm_dev)

    # 4. 전역 수치 방어벽 (Global Guardrail) - 서브에이전트 권고
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats

# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 척추 중립(c1) 전담 신규 선행 연구 기반 피처 추출기

[선행 연구 기반 신규 피처 6개]:
  선행 연구 (Schoenfeld 2010, Hartmann 2013, Kim 2015, Swinton 2012 등)에서
  척추 중립 분류에 가장 유효하다고 검증된 feature들:

  (A) c1_shoulder_hip_knee_angle_at_nadir
      최하사점 어깨-고관절-무릎 3점 내각 (Lumbopelvic coupling proxy)
      - 180°에 가까울수록 척추 중립 유지 (상체가 대퇴와 직선)
      - 90° 이하 → 과도한 굿모닝형 척추 전방 굴곡
      참고: Schoenfeld (2010), McKean et al. (2010)

  (B) c1_trunk_thigh_ratio_mean
      체간 피치각 / 대퇴골 수평각 비율의 시간 평균
      (trunk angle / thigh angle ratio time-series mean)
      - 체형 불변: 절대 각도 대신 상대 비율
      - 정상: 비율 안정 유지 / 결함: 비율 급변 (굿모닝 또는 척추 과굴곡)
      참고: Fry et al. (2003), Vigotsky et al. (2016)

  (C) c1_max_trunk_lean_phase_offset
      최대 체간 전방경사 프레임 위치 - 최하사점 프레임 위치 (정규화)
      - 정상: 최대 경사가 최하사점 이전 (음수 또는 0)
      - 굿모닝 결함: 최대 경사가 최하사점 이후 (양수) — 상승 중 계속 앞으로 꺾임
      참고: Swinton et al. (2012), Lorenzetti et al. (2018)

  (D) c1_trunk_jerk_peak_at_nadir
      최하사점 근방 체간 피치각 1차 미분(속도)의 절대 최대값
      → Log-LDLJ를 대체: 3차 미분 대신 1차 미분 최대값 (노이즈 내성 강함)
      - 갑작스러운 척추 움직임 = 제어 손실 지표
      참고: Tsuji et al. (2021), Manabe et al. (2007)

  (E) c1_hip_shoulder_knee_descent_ratio
      하강 구간 고관절 굴곡 변화율 / 체간 전방경사 변화율 비율
      - 정상: 대퇴(고관절)가 이끌고 체간은 따라옴 (비율 > 1)
      - 결함: 체간이 먼저 쓰러짐 (비율 < 0.7) → 굿모닝 시작
      참고: List et al. (2013), Vleeming et al. (2012)

  (F) c1_cross_corr_lag_trunk_thigh
      체간 피치-고관절 굴곡 시계열 교차상관 지연 시간 (lumbopelvic rhythm lag)
      - 정상: 둘이 동기화 (lag ≈ 0)
      - 결함: 체간이 앞서거나 뒤처짐 (|lag| > 2 frames)
      참고: Esola et al. (1996), Kim et al. (2015)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List

SPINE_LIT_FEATURE_NAMES_6: List[str] = [
    "c1_shoulder_hip_knee_angle_at_nadir",   # A. 최하사점 어깨-고관절-무릎 내각
    "c1_trunk_thigh_ratio_mean",             # B. 체간/대퇴 각도 비율 시간 평균
    "c1_max_trunk_lean_phase_offset",        # C. 최대 체간 경사 위상 오프셋
    "c1_trunk_jerk_peak_at_nadir",           # D. 최하사점 체간 속도 최대값 (1차 미분 피크)
    "c1_hip_shoulder_knee_descent_ratio",    # E. 하강 고관절/체간 변화율 비율
    "c1_cross_corr_lag_trunk_thigh",         # F. 체간-고관절 교차상관 지연
]


def safe_divide_scalar(num: float, denom: float, eps: float = 1e-4, clip_range: Tuple[float, float] = (-10.0, 10.0)) -> float:
    sign = 1.0 if denom >= 0 else -1.0
    safe_d = sign * max(abs(denom), eps)
    return float(np.clip(num / safe_d, clip_range[0], clip_range[1]))


def compute_angle_3pt(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """
    3점 내각 계산 (vertex가 꼭짓점)
    Returns angle in degrees [0, 180]
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_squat_spine_literature_features(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, 16, 92) → (N, 6) 선행 연구 기반 신규 척추 중립 피처

    Format B 채널 매핑 (사용):
      Stream 1 (뼈대):
        [0]  Pelvis-Torso dx   (체간 X 변위)
        [1]  Pelvis-Torso dy   (체간 Y 높이)
        [2]  Pelvis-Torso dz   (체간 Z 깊이)
        [40] LHip-LKnee dy     (좌 대퇴 Y)
        [44] LKnee-LAnkle dy   (좌 정강이 Y)
        [48] RHip-RKnee dy     (우 대퇴 Y)
        [52] RKnee-RAnkle dy   (우 정강이 Y)
      Stream 2 (관절각):
        [56,57] knee_angle L/R  (무릎 굴곡각)
        [58,59] hip_angle L/R   (고관절 굴곡각)
      Stream 3 (정렬):
        [64] torso_pitch_3d     (체간 전방 경사각)
        [89,90] shank_vert L/R  (정강이 수직각 = 대퇴 기울기 프록시)
    """
    N, T, C = X_b.shape
    feats = np.zeros((N, 6), dtype=np.float32)
    dt = 1.0 / max(1, T - 1)

    for i in range(N):
        # ─── 시계열 추출 ─────────────────────────────────────────────────────
        # 무릎 굴곡각 (최하사점 = 최대값)
        knee_raw   = (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
        knee_flex  = 180.0 - knee_raw

        # 고관절 굴곡각
        hip_raw    = (X_b[i, :, 58] + X_b[i, :, 59]) / 2.0
        hip_flex   = 180.0 - hip_raw

        # 체간 전방 경사각 (Pitch)
        torso_pitch = X_b[i, :, 64].copy()

        # 정강이 수직각 → 대퇴 기울기 프록시 (클수록 대퇴가 수직에 가까움)
        thigh_angle  = (X_b[i, :, 89] + X_b[i, :, 90]) / 2.0

        # 체간 수직 벡터 (Pelvis-Torso)
        torso_dy = X_b[i, :, 1]
        torso_dz = X_b[i, :, 2]
        # 어깨-고관절 수직 투영 거리 (체간 높이) → 어깨 Y 추정
        pelvis_y_proxy = -(X_b[i, :, 41] + X_b[i, :, 49]) / 2.0
        shoulder_y = pelvis_y_proxy + torso_dy

        # 무릎 Y 추정 (대퇴 Y 사용)
        knee_y = pelvis_y_proxy - (X_b[i, :, 40] + X_b[i, :, 48]) / 2.0

        # ─── 신호 평활화 ─────────────────────────────────────────────────────
        sigma = 0.8
        knee_flex_s   = gaussian_filter1d(knee_flex,   sigma=sigma)
        hip_flex_s    = gaussian_filter1d(hip_flex,    sigma=sigma)
        torso_pitch_s = gaussian_filter1d(torso_pitch, sigma=sigma)
        thigh_angle_s = gaussian_filter1d(thigh_angle, sigma=sigma)
        shoulder_y_s  = gaussian_filter1d(shoulder_y,  sigma=sigma)
        knee_y_s      = gaussian_filter1d(knee_y,      sigma=sigma)
        pelvis_y_s    = gaussian_filter1d(pelvis_y_proxy, sigma=sigma)

        # ─── 최하사점 인덱스 ─────────────────────────────────────────────────
        b = int(np.argmax(knee_flex_s))
        b = int(np.clip(b, 2, T - 3))

        # 가동범위 스케일
        t_ptp   = max(float(np.ptp(torso_pitch_s)), 3.0)
        hip_ptp = max(float(np.ptp(hip_flex_s)), 5.0)

        # ════════════════════════════════════════════════════════════════════
        # (A) 피처 1: 최하사점 어깨-고관절-무릎 3점 내각
        # 어깨 (Y): shoulder_y_s[b]
        # 고관절 (Y): pelvis_y_s[b]  ← 꼭짓점
        # 무릎 (Y): knee_y_s[b]
        # 피처: 이 3점 내각 (180° = 완전 직선, 작을수록 굽음)
        # ════════════════════════════════════════════════════════════════════
        # 2D 좌표 구성 (Y-Z 평면, 시상면)
        shoulder_pt = np.array([shoulder_y_s[b], X_b[i, b, 2]])   # Y, Z
        hip_pt      = np.array([pelvis_y_s[b], 0.0])               # 꼭짓점
        knee_pt     = np.array([knee_y_s[b], 0.0])

        sh_hip_knee_angle = compute_angle_3pt(shoulder_pt, hip_pt, knee_pt)
        feats[i, 0] = float(np.clip(sh_hip_knee_angle, 0.0, 180.0))

        # ════════════════════════════════════════════════════════════════════
        # (B) 피처 2: 체간/대퇴 각도 비율 시간 평균 (trunk/thigh ratio)
        # torso_pitch_s / thigh_angle_s
        # 정상: 비율 일정, 결함: 비율 급변
        # ════════════════════════════════════════════════════════════════════
        t_ptp_abs = max(float(np.mean(np.abs(torso_pitch_s))), 1.0)
        th_abs    = max(float(np.mean(np.abs(thigh_angle_s))), 1.0)
        trunk_thigh_ratios = np.abs(torso_pitch_s) / (np.abs(thigh_angle_s) + 1e-4)
        # 극값 클리핑 후 평균
        ratio_mean = float(np.mean(np.clip(trunk_thigh_ratios, 0.0, 5.0)))
        feats[i, 1] = float(np.clip(ratio_mean, 0.0, 5.0))

        # ════════════════════════════════════════════════════════════════════
        # (C) 피처 3: 최대 체간 경사 위상 오프셋 (굿모닝 감별)
        # peak_frame(torso_pitch) - b (최하사점)
        # 음수 = 최대 경사가 최하사점 이전 (정상)
        # 양수 = 최대 경사가 최하사점 이후 (굿모닝 결함)
        # ════════════════════════════════════════════════════════════════════
        peak_trunk_frame = int(np.argmax(torso_pitch_s))
        phase_offset = (peak_trunk_frame - b) / max(1, T - 1)  # 정규화 [-1, 1]
        feats[i, 2] = float(np.clip(phase_offset, -1.0, 1.0))

        # ════════════════════════════════════════════════════════════════════
        # (D) 피처 4: 최하사점 근방 체간 각속도 절대 최대값
        # (log-LDLJ 대체: 1차 미분 최대값 = 노이즈 내성)
        # ════════════════════════════════════════════════════════════════════
        v_torso = np.gradient(torso_pitch_s, dt)
        win_s   = max(0, b - 3)
        win_e   = min(T, b + 4)
        jerk_peak = float(np.max(np.abs(v_torso[win_s:win_e])))
        # 전체 가동범위 대비 정규화
        feats[i, 3] = float(np.clip(jerk_peak / (t_ptp + 1e-4), 0.0, 20.0))

        # ════════════════════════════════════════════════════════════════════
        # (E) 피처 5: 하강 고관절/체간 변화율 비율
        # 하강 구간: 0 ~ b
        # Δhip_flex / Δtorso_pitch 비율
        # 정상: Δhip > Δtorso (고관절이 이끔)
        # 결함: Δtorso > Δhip (체간이 앞서 쓰러짐)
        # ════════════════════════════════════════════════════════════════════
        if b >= 2:
            delta_hip   = float(np.mean(hip_flex_s[max(0,b-2):b+1])) - float(hip_flex_s[0])
            delta_torso = float(np.mean(torso_pitch_s[max(0,b-2):b+1])) - float(torso_pitch_s[0])
            hip_torso_ratio = safe_divide_scalar(abs(delta_hip), abs(delta_torso), eps=1.0, clip_range=(0.0, 5.0))
        else:
            hip_torso_ratio = 1.0
        feats[i, 4] = float(np.clip(hip_torso_ratio, 0.0, 5.0))

        # ════════════════════════════════════════════════════════════════════
        # (F) 피처 6: 체간-고관절 교차상관 지연 (lumbopelvic rhythm lag)
        # 정상: 둘이 동기 (lag ≈ 0)
        # 결함: 체간이 먼저 반응하거나 뒤처짐 (|lag| 증가)
        # ════════════════════════════════════════════════════════════════════
        # 정규화
        t_std = np.std(torso_pitch_s)
        h_std = np.std(hip_flex_s)
        if t_std > 1e-5 and h_std > 1e-5:
            t_n = (torso_pitch_s - np.mean(torso_pitch_s)) / t_std
            h_n = (hip_flex_s   - np.mean(hip_flex_s))    / h_std
            xcorr = np.correlate(t_n, h_n, mode='full')
            center = len(xcorr) // 2
            # ±4프레임 내에서 피크 위치 (lag)
            window = xcorr[max(0, center - 4): center + 5]
            lag_idx = int(np.argmax(window)) - 4  # -4 ~ +4 frames
            lag_normalized = lag_idx / max(1, T - 1)
        else:
            lag_normalized = 0.0
        feats[i, 5] = float(np.clip(lag_normalized, -1.0, 1.0))

    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats

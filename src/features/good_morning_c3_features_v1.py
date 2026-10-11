# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 諛붾꺼 ?ㅽ떚???곕뱶由ы봽??(Barbell Stiff-leg Deadlift)
Module: stiff_spine_features_v3.py
Purpose: ?ㅽ떚???곕뱶由ы봽??泥숈텛 以묐┰(sd2 v3) ?꾩슜 怨좊룄???앹껜??븰 ?쇱쿂 異붿텧湲?  1. ??援쏀옒 (Thoracic & Lumbar Flexion / Round Back) ?뱁솕 ?쇱쿂
  2. ?덈━ 怨쇱떊??(Lumbar Hyperextension / Lockout Posterior Lean / Pelvic Anterior Sway) ?뱁솕 ?쇱쿂
  3. 諛붾꺼 諛?泥닿컙 湲곗슱??(Barbell Tilt Roll, Shoulder/Hip Roll, Lateral Barbell Asymmetry) ?뱁솕 ?쇱쿂
  4. CoRe (Contrastive Regression) 怨⑤뱶 ?뺤긽 ?쒗뵆由??곷? ?몄감 ?쇱쿂 吏??"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

FEATURE_NAMES_V3: List[str] = [
    # [1~6] ?쒖긽硫?泥숈텛 留뚭끝 諛??붿텛 ?숈뿭??(??援쏀옒 援닿끝 媛먯?)
    "stiff_lumbar_poly_a",                # [ 1] ?붿텛 2李??щЪ??怨〓쪧 怨꾩닔 ?됯퇏 (援닿끝 ?뺣룄)
    "stiff_lumbar_stiffness_sd",          # [ 2] ??援ш컙 ?붿텛 媛곷룄 蹂?숈꽦 (泥숈텛 媛뺤꽦 ?덉젙??
    "stiff_spine_lateral_dev_max",        # [ 3] 泥숈텛 以묒떖???〓갑???댄깉 理쒕?移?(醫뚯슦 鍮꾨?移?
    "stiff_trunk_flex_speed_max",         # [ 4] ?곸껜 援닿끝 ?띾룄 理쒕? ?쇳겕
    "stiff_cervical_min",                 # [ 5] 寃쎌텛-?됱텛 理쒖냼 媛곷룄 (?쒖꽑 諛?寃쎌텛 ?뺣젹)
    "stiff_active_rom",                   # [ 6] ?먮ぉ 湲곗? ?섏쭅 媛?숇쾾??(ROM)
    "stiff_cobb_mean",                    # [ 7] ?붿텛 Cobb媛??됯퇏
    "stiff_shoulder_hip_twist_max",       # [ 8] ?닿묠??怨⑤컲??3D 鍮꾪?由?媛곷룄
    "stiff_shear_stress_proxy",           # [ 9] ?곸껜 紐⑤찘?몄븫 湲곕컲 ?붿텛 ?꾨떒???꾨줉??    "stiff_pitch_rom",                    # [10] ?곸껜 援닿끝 媛곷룄 吏꾪룺 (Max - Min)
    "stiff_view_occlusion",               # [11] ?꾨갑 ?쒖젏 媛由?吏??(泥닿컙 Y異?Z?ъ쁺)
    "stiff_view_comp_curv",               # [12] 3D 怨듦컙 留뚭끝 蹂댁젙 怨〓쪧 (Chord-Displacement)
    "stiff_ratio_arm_to_torso",           # [13] ??湲몄씠 ? 泥닿컙 湲몄씠 鍮?    "stiff_ratio_femur_to_tibia",         # [14] ???? ?섑눜 湲몄씠 鍮?    "stiff_ratio_leg_to_torso",           # [15] ?ㅻ━ ?꾩껜 ? 泥닿컙 湲몄씠 鍮?    "stiff_knee_fixation_sd",             # [16] 臾대쫷 愿??媛곷룄 怨좎젙 ?덉젙??    "stiff_torso_pelvis_lateral_offset",  # [17] 泥닿컙 以묒떖怨?怨⑤컲 以묒떖???〓갑???곷? ?몄감
    "stiff_sagittal_spine_stiffness",     # [18] ?섍컯-?곸듅 ?꾪솚 ??泥숈텛 ?뺥깭 蹂댁〈 媛뺤꽦 吏??
    # [19~23] ??諛붾꺼 諛?泥닿컙 醫뚯슦 湲곗슱???뱁솕 ?쇱쿂 (Key 211, 212, 213, 214 ???
    "stiff_barbell_roll_max",             # [19] ???먮ぉ(諛붾꺼) ?섑룊 湲곗슱湲?理쒕? 媛곷룄 (Roll Max)
    "stiff_barbell_roll_mean",            # [20] ???먮ぉ(諛붾꺼) ?섑룊 湲곗슱湲??됯퇏 媛곷룄
    "stiff_barbell_lateral_drift",        # [21] 怨⑤컲 以묒떖 ?鍮?諛붾꺼(???먮ぉ 以묒젏) ?〓갑??鍮꾨?移??몄감
    "stiff_shoulder_roll_max",            # [22] ?닿묠???섑룊 湲곗슱湲?理쒕? ?몄감 (Shoulder Roll)
    "stiff_hip_roll_max",                 # [23] 怨⑤컲???섑룊 湲곗슱湲?理쒕? ?몄감 (Hip Roll)

    # [24~27] ???덈━ 怨쇱떊??Hyperextension / Lockout Lean) ?뱁솕 ?쇱쿂 (Key 215 ???
    "stiff_lockout_posterior_lean",       # [24] 湲곕┰ ?쎌븘?????곸껜 ?꾨갑 怨쇱떊??寃쎌궗媛?(?섏쭅???꾨갑 ?뽱옒)
    "stiff_lockout_pelvic_sway",          # [25] 湲곕┰ ?쎌븘????諛쒕ぉ ?鍮?怨⑤컲 ?꾨갑 ?ㅼ썾???꾨갑 諛由? 蹂??    "stiff_lumbar_lordosis_peak",         # [26] ?붿텛 ?꾨쭔 怨쇱떊???뚯닔 ?쇳겕 (Lordosis Over-extension)
    "stiff_lockout_reversal_accel",       # [27] ?쎌븘???뺤젏 ?꾨떖 ??諛섎? 諛⑺뼢 ??컧??異⑷꺽??
    # [28~30] ????援쏀옒(Round Back / Thoracic Kyphosis) ?뱁솕 ?쇱쿂 (Key 216 ???
    "stiff_thoracic_kyphosis_peak",       # [28] ?곷? ?됱텛 怨〓쪧 諛??닿묠 ?꾨갑 留먮┝ ?쇳겕 (Protraction)
    "stiff_thoracolumbar_c-curve_depth",  # [29] C7-S1 吏곸꽑 ?鍮??됱슂異?留뚭끝 理쒕? ?⑤ぐ 源딆씠 (C-Curve Depth)
    "stiff_roundback_inflexion_integral"  # [30] ?섍컯 理쒖? 蹂怨≪젏 遺洹쇱쓽 ?됱텛 援닿끝 ?쒓컙 ?곷텇媛?]


def extract_stiff_spine_features_v3(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None,
    gold_template_kps: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    ?⑥씪 64?꾨젅???쒗?ㅻ줈遺??30D ?ㅽ떚???곕뱶由ы봽??泥숈텛 ?뺤삁 v3 ?쇱쿂 異붿텧
    kps_33_64f: (64, 33, 3)
    spine_9_64f: (64, 9, 3)
    """
    T = kps_33_64f.shape[0]  # 64
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, MP['LSH'], :]
    r_sh = kps_33_64f[:, MP['RSH'], :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, MP['LHIP'], :]
    r_hip = kps_33_64f[:, MP['RHIP'], :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, MP['LKNE'], :]
    r_knee = kps_33_64f[:, MP['RKNE'], :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, MP['LANK'], :]
    r_ank = kps_33_64f[:, MP['RANK'], :]
    mid_ank = (l_ank + r_ank) / 2.0

    l_wrist = kps_33_64f[:, MP['LWR'], :]
    r_wrist = kps_33_64f[:, MP['RWR'], :]
    mid_wrist = (l_wrist + r_wrist) / 2.0

    # 1. 泥닿컙 濡쒖뺄 ?뺢퇋 吏곴탳 湲곗?怨?    v_torso = mid_sh - mid_hip
    torso_len = np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7
    y_hat = v_torso / torso_len

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. SpinePose 9遺꾩젅 3D 醫뚰몴
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    # 3. ?붿텛 2李??ㅽ빆 怨〓쪧 poly_a
    z_s1 = np.sum((s_pts[:, 0] - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((s_pts[:, 1] - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((s_pts[:, 2] - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((s_pts[:, 3] - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((s_pts[:, 4] - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        p = np.polyfit(x_coords, lumbar_curves[t_idx], deg=2)
        poly_a_series[t_idx] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    # 4. 泥숈텛 ?〓갑???댄깉
    x_s_disp = np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1)
    f3_lat_dev = float(np.max(np.abs(x_s_disp)))

    # 5. ?곸껜 援닿끝 ?띾룄
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))

    # 6. 寃쎌텛-?됱텛 理쒖냼 媛곷룄
    v_cerv = s_pts[:, 8] - s_pts[:, 6]
    v_thor = s_pts[:, 5] - s_pts[:, 4]
    dot_ct = np.sum(v_cerv * v_thor, axis=-1)
    norm_ct = (np.linalg.norm(v_cerv, axis=-1) * np.linalg.norm(v_thor, axis=-1)) + 1e-7
    ct_angles = np.degrees(np.arccos(np.clip(dot_ct / norm_ct, -1.0, 1.0)))
    f5_cervical_min = float(np.min(ct_angles)) if spine_9_64f is not None else 150.0

    # 7. 媛?숇쾾??    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f6_active_rom = float(np.ptp(wrist_y_s))

    # 8. Cobb 媛곷룄
    v_l5_s1 = s_pts[:, 0] - s_pts[:, 1]
    v_l3_l1 = s_pts[:, 3] - s_pts[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    cobb_series = gaussian_filter1d(np.abs(cobb_angles), sigma=1.0)
    f7_cobb_mean = float(np.mean(cobb_series))

    # 9. ?닿묠-怨⑤컲 鍮꾪?由?    v_hip = l_hip - r_hip
    twist = np.degrees(np.arccos(np.clip(
        np.sum(v_sh * v_hip, axis=-1) / (np.linalg.norm(v_sh, axis=-1) * np.linalg.norm(v_hip, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f8_twist_max = float(np.max(twist))

    # 10. ?붿텛 ?꾨떒???꾨줉??    f9_shear = float(np.mean(np.sin(np.radians(torso_pitch))))

    # 11. 泥닿컙 媛곷룄 吏꾪룺
    f10_pitch_rom = float(np.ptp(torso_pitch))

    # 12. ?꾨갑 ?쒖젏 媛由?吏??    f11_view_occlusion = float(np.mean(np.abs(y_hat[:, 2])))

    # 13. 3D 怨듦컙 留뚭끝 蹂댁젙 怨〓쪧
    v_c7_s1 = s_pts[:, 6] - s_pts[:, 0]
    chord_len = np.linalg.norm(v_c7_s1, axis=-1) + 1e-6
    l3_disp = np.linalg.norm(np.cross(v_c7_s1, s_pts[:, 2] - s_pts[:, 0]), axis=-1) / chord_len
    f12_view_comp_curv = float(np.mean(l3_disp))

    # 14~16. ?좎껜 鍮꾩쑉
    mean_torso = float(np.mean(torso_len))
    arm_len = float(np.mean(np.linalg.norm(mid_sh - mid_wrist, axis=-1)))
    femur_len = float(np.mean(np.linalg.norm(mid_hip - mid_knee, axis=-1)))
    tibia_len = float(np.mean(np.linalg.norm(mid_knee - mid_ank, axis=-1)))
    f13_arm_torso = arm_len / max(1e-4, mean_torso)
    f14_femur_tibia = femur_len / max(1e-4, tibia_len)
    f15_leg_torso = (femur_len + tibia_len) / max(1e-4, mean_torso)

    # 17. 臾대쫷 怨좎젙 ?덉젙??    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f16_knee_fixation_sd = float(np.std(knee_angle))

    # 18. 泥닿컙-怨⑤컲 ?〓갑???곷? ?ㅽ봽??    torso_hip_dx = np.abs(mid_sh[:, 0] - mid_hip[:, 0])
    f17_torso_pelvis_offset = float(np.mean(torso_hip_dx))

    # 19. 泥숈텛 媛뺤꽦 ?좎? 吏??    bottom_idx = int(np.argmax(wrist_y_s))
    curv_at_bottom = poly_a_series[bottom_idx]
    f18_sagittal_stiffness = float(curv_at_bottom - f1_poly_a)

    # -------------------------------------------------------------
    # ??[20~24] 諛붾꺼 諛?泥닿컙 醫뚯슦 湲곗슱???뱁솕 (Key 211, 212, 213, 214 ???
    # -------------------------------------------------------------
    v_wrist = l_wrist - r_wrist  # ???먮ぉ = 諛붾꺼 ?섑룊??    wrist_horiz_dx = v_wrist[:, 0]
    wrist_horiz_dy = v_wrist[:, 1]
    barbell_roll = np.degrees(np.arctan2(wrist_horiz_dy, np.abs(wrist_horiz_dx) + 1e-7))
    f19_barbell_roll_max = float(np.max(np.abs(barbell_roll)))
    f20_barbell_roll_mean = float(np.mean(np.abs(barbell_roll)))

    barbell_center_x = mid_wrist[:, 0]
    pelvis_center_x = mid_hip[:, 0]
    f21_barbell_lateral_drift = float(np.max(np.abs(barbell_center_x - pelvis_center_x)))

    sh_roll = np.degrees(np.arctan2(v_sh[:, 1], np.abs(v_sh[:, 0]) + 1e-7))
    f22_shoulder_roll_max = float(np.max(np.abs(sh_roll)))

    hip_roll = np.degrees(np.arctan2(v_hip[:, 1], np.abs(v_hip[:, 0]) + 1e-7))
    f23_hip_roll_max = float(np.max(np.abs(hip_roll)))

    # -------------------------------------------------------------
    # ??[25~28] ?덈━ 怨쇱떊??Hyperextension / Lockout Lean) ?뱁솕 (Key 215 ???
    # -------------------------------------------------------------
    # ?쎌븘??湲곕┰ 吏곷┰) 援ш컙: ?곸쐞 10% 理쒖? 媛?숇쾾???꾨젅?꾨뱾
    standing_frames = np.where(wrist_y_s < (np.min(wrist_y_s) + 0.15 * np.ptp(wrist_y_s)))[0]
    if len(standing_frames) == 0:
        standing_frames = np.array([0, T - 1])

    # ?쎌븘????泥닿컙???섏쭅 湲곗? ?ㅻ줈 ?뽱?吏?媛곷룄 (Posterior Lean)
    vertical_vec = np.array([0.0, -1.0, 0.0])  # ?꾩そ 諛⑺뼢
    torso_vectors = (mid_sh - mid_hip)[standing_frames]
    torso_z_projections = torso_vectors[:, 2]  # Z異??꾨갑 ?뚯텧
    # ?섏쭅 ?鍮??꾨갑 湲곗슱湲?    posterior_lean_angles = np.degrees(np.arctan2(torso_z_projections, -torso_vectors[:, 1] + 1e-7))
    f24_lockout_posterior_lean = float(np.max(np.maximum(0.0, posterior_lean_angles)))

    # 怨⑤컲??諛쒕ぉ蹂대떎 ?욎쑝濡??ㅼ썾??諛由???蹂??(Anterior Pelvic Sway)
    pelvic_forward_disp = (mid_hip - mid_ank)[standing_frames, 2]
    f25_lockout_pelvic_sway = float(np.max(pelvic_forward_disp))

    # ?붿텛 ?꾨쭔 怨쇱떊???뚯닔 ?쇳겕 (Lordosis Over-extension)
    f26_lordosis_peak = float(np.min(poly_a_series[standing_frames]))

    # ?쎌븘???뺤젏 媛?띾룄 ??쟾 異⑷꺽??    standing_vel = torso_pitch_vel[standing_frames]
    f27_lockout_reversal = float(np.max(standing_vel) if len(standing_vel) > 0 else 0.0)

    # -------------------------------------------------------------
    # ??[29~31] ??援쏀옒(Round Back / Thoracic Kyphosis) ?뱁솕 (Key 216 ???
    # -------------------------------------------------------------
    # ?됱텛 ?뚯텧 諛??닿묠 ?꾨갑 留먮┝ (Kyphosis Protraction)
    v_c7_midsh = mid_sh - s_pts[:, 6]
    thoracic_forward_disp = np.sum(v_c7_midsh * z_hat, axis=-1)
    f28_thoracic_kyphosis = float(np.max(thoracic_forward_disp))

    # C7-S1 吏곸꽑 ?鍮?泥숈텛 以묎컙??T8/L1) ?⑤ぐ 源딆씠 (C-Curve Depth)
    v_c7_s1_full = s_pts[:, 6] - s_pts[:, 0]
    line_norm = np.linalg.norm(v_c7_s1_full, axis=-1, keepdims=True) + 1e-7
    mid_spine = s_pts[:, 3]  # L1
    proj = s_pts[:, 0] + np.sum((mid_spine - s_pts[:, 0]) * (v_c7_s1_full / line_norm), axis=-1, keepdims=True) * (v_c7_s1_full / line_norm)
    c_curve_dist = np.linalg.norm(mid_spine - proj, axis=-1)
    f29_c_curve_depth = float(np.max(c_curve_dist))

    # 理쒖? 蹂怨≪젏(?섍컯 ?꾪솚湲? ?됱텛 援닿끝 ?쒓컙 ?곷텇媛?    inflexion_window = max(2, int(T * 0.15))
    start_inf = max(0, bottom_idx - inflexion_window)
    end_inf = min(T, bottom_idx + inflexion_window)
    f30_roundback_inflexion = float(np.mean(thoracic_forward_disp[start_inf:end_inf]))

    feats_30d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min,
        f6_active_rom, f7_cobb_mean, f8_twist_max, f9_shear, f10_pitch_rom,
        f11_view_occlusion, f12_view_comp_curv, f13_arm_torso, f14_femur_tibia, f15_leg_torso,
        f16_knee_fixation_sd, f17_torso_pelvis_offset, f18_sagittal_stiffness,
        f19_barbell_roll_max, f20_barbell_roll_mean, f21_barbell_lateral_drift,
        f22_shoulder_roll_max, f23_hip_roll_max,
        f24_lockout_posterior_lean, f25_lockout_pelvic_sway, f26_lordosis_peak, f27_lockout_reversal,
        f28_thoracic_kyphosis, f29_c_curve_depth, f30_roundback_inflexion
    ], dtype=np.float32)

    return feats_30d

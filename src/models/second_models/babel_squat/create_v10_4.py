# -*- coding: utf-8 -*-
"""
Helper script to clone babel_squat_integrated_v10_3.py into v10_4.py
with View-Conditioned Vector Scaling & Dynamic Risk Gate integration.
"""

src_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_3.py"
dst_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_4.py"

with open(src_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update version identifiers
content = content.replace("BabelSquatIntegratedEngineV10_3", "BabelSquatIntegratedEngineV10_4")
content = content.replace("v10.3", "v10.4")
content = content.replace("v10_3", "v10_4")
content = content.replace('VERSION = "v10.3"', 'VERSION = "v10.4"')

# 2. Add scaler loader in __init__
init_old = """        self.deadband_low = float(meta_cfg.get('deadband_low', 0.30))
        self.deadband_high = float(meta_cfg.get('deadband_high', 0.36))
        self.margin_retake_threshold = float(meta_cfg.get('margin_retake_threshold', 0.04))"""

init_new = """        self.deadband_low = float(meta_cfg.get('deadband_low', 0.30))
        self.deadband_high = float(meta_cfg.get('deadband_high', 0.36))
        self.margin_retake_threshold = float(meta_cfg.get('margin_retake_threshold', 0.04))
        self.risk_threshold = float(meta_cfg.get('risk_threshold', 0.45))

        # [v10.4 신규]: 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}"""

if init_old in content:
    content = content.replace(init_old, init_new)
    print("[SUCCESS] Replaced __init__ scaler config!")
else:
    print("[WARNING] init_old not matched directly, checking CRLF...")
    init_old_crlf = init_old.replace("\n", "\r\n")
    if init_old_crlf in content:
        content = content.replace(init_old_crlf, init_new)
        print("[SUCCESS] Replaced __init__ with CRLF!")
    else:
        print("[ERROR] Could not match init_old")

# 3. Add dynamic risk gate in Layer 4
layer4_old = """        if self.enable_selective:
            is_quality_deficient = (q_overall_val < self.q_min)
            in_clean_deadband = (p_clean_16 >= self.deadband_low) and (p_clean_16 < self.deadband_high)
            margin_deficient = (top_margin < self.margin_retake_threshold)

            if is_quality_deficient:
                session_status = 'RETAKE'
                is_retake = True
                clean_tier = 'NONE'
                is_verified_clean = False
                is_observable_clean = False
                retake_reason = 'POSE_TRACKING_JITTER_LOW_QUALITY'
            elif session_status != 'CLEAN' and (in_clean_deadband or margin_deficient):
                session_status = 'RETAKE'
                is_retake = True
                clean_tier = 'NONE'
                is_verified_clean = False
                is_observable_clean = False
                retake_reason = 'BOUNDARY_UNCERTAINTY' if in_clean_deadband else 'LOW_MARGIN_CONFIDENCE'"""

layer4_new = """        if self.enable_selective:
            is_quality_deficient = (q_overall_val < self.q_min)

            # [v10.4 신규]: 시점별 벡터 스케일링 기반 동적 위험도(Risk-Aware) 계산
            scaler_cfg = getattr(self, 'view_vector_scalers', {}).get(norm_view, None)
            if scaler_cfg is not None:
                a_v = np.array(scaler_cfg['scale'], dtype=np.float32)
                b_v = np.array(scaler_cfg['bias'], dtype=np.float32)
                z_cal_16 = l16_arr * a_v + b_v
                p16_cal_vec = np.exp(z_cal_16 - np.max(z_cal_16))
                p16_cal_vec = p16_cal_vec / np.sum(p16_cal_vec)
            else:
                p16_cal_vec = p16_arr

            H_max = np.log(16.0)
            log_p_cal = np.log(np.clip(p16_cal_vec, 1e-8, 1.0))
            entropy_val = -float(np.sum(p16_cal_vec * log_p_cal))
            sorted_p = np.sort(p16_cal_vec)
            margin_gap = float(sorted_p[-1] - sorted_p[-2])
            risk_val = float((entropy_val / H_max) * (1.0 - margin_gap))
            is_risk_excessive = (risk_val > self.risk_threshold)

            if is_quality_deficient:
                session_status = 'RETAKE'
                is_retake = True
                clean_tier = 'NONE'
                is_verified_clean = False
                is_observable_clean = False
                retake_reason = 'POSE_TRACKING_JITTER_LOW_QUALITY'
            elif session_status != 'CLEAN' and is_risk_excessive:
                session_status = 'RETAKE'
                is_retake = True
                clean_tier = 'NONE'
                is_verified_clean = False
                is_observable_clean = False
                retake_reason = 'DYNAMIC_RISK_UNCERTAINTY'"""

if layer4_old in content:
    content = content.replace(layer4_old, layer4_new)
    print("[SUCCESS] Replaced Layer 4 selective prediction logic!")
else:
    print("[WARNING] layer4_old not matched directly, checking CRLF...")
    layer4_old_crlf = layer4_old.replace("\n", "\r\n")
    if layer4_old_crlf in content:
        content = content.replace(layer4_old_crlf, layer4_new)
        print("[SUCCESS] Replaced Layer 4 with CRLF!")
    else:
        print("[ERROR] Could not match layer4_old")

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[DONE] Saved v10.4 to {dst_path}")

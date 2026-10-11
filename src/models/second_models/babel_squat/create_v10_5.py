# -*- coding: utf-8 -*-
"""
Helper script to clone babel_squat_integrated_v10_4.py into v10_5.py
with 14D ST-GCN + Temporal Transformer Residual Late Fusion integration.
"""

src_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_4.py"
dst_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_5.py"

with open(src_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update version identifiers
content = content.replace("BabelSquatIntegratedEngineV10_4", "BabelSquatIntegratedEngineV10_5")
content = content.replace("v10.4", "v10.5")
content = content.replace("v10_4", "v10_5")
content = content.replace('VERSION = "v10.4"', 'VERSION = "v10.5"')

# 2. Add import for STGCNTransformerResidualNet
import_old = "from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1"
import_new = """from src.models.second_models.babel_squat.c4_model_v5_1 import BabelSquatC4ModelV5_1
from src.models.second_models.babel_squat.stgcn_transformer_net import STGCNTransformerResidualNet"""

if import_old in content:
    content = content.replace(import_old, import_new)
    print("[SUCCESS] Added STGCNTransformerResidualNet import!")

# 3. Add ST-GCN Transformer model loading in __init__
init_old = """        # [v10.4 신규]: 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}"""

init_new = """        # [v10.4]: 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}

        # [v10.5 신규]: 14D ST-GCN + Temporal Transformer 경량 잔차 신경망 로드
        self.stgcn_weights_path = os.path.join(CURRENT_DIR, "stgcn_transformer_residual_weights.pth")
        self.stgcn_model = None
        self.g_front = 0.08  # 전방 시점 (view2, view3, view4) 최적 잔차 게이트 가중치
        self.g_rear  = 0.00  # 후방 시점 (view1, view5) 보수적 제로 가중치 (안정성 보존)
        if os.path.exists(self.stgcn_weights_path):
            try:
                self.stgcn_model = STGCNTransformerResidualNet().to(self.device)
                self.stgcn_model.load_state_dict(torch.load(self.stgcn_weights_path, map_location=self.device, weights_only=True))
                self.stgcn_model.eval()
            except Exception as e:
                warnings.warn(f"Failed to load STGCNTransformerResidualNet: {e}")
                self.stgcn_model = None"""

if init_old in content:
    content = content.replace(init_old, init_new)
    print("[SUCCESS] Added ST-GCN Transformer initialization!")
else:
    init_old_crlf = init_old.replace("\n", "\r\n")
    if init_old_crlf in content:
        content = content.replace(init_old_crlf, init_new)
        print("[SUCCESS] Added ST-GCN Transformer initialization (CRLF)!")
    else:
        print("[ERROR] Could not match init_old")

# 4. Update predict_session signature to accept x_14d
pred_sig_old = """        c2_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        c4_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        quality_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        kinetic_raw_feats: Optional[Union[np.ndarray, List[float]]] = None"""

pred_sig_new = """        c2_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        c4_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        quality_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        kinetic_raw_feats: Optional[Union[np.ndarray, List[float]]] = None,
        x_14d: Optional[Union[np.ndarray, torch.Tensor]] = None"""

if pred_sig_old in content:
    content = content.replace(pred_sig_old, pred_sig_new)
    print("[SUCCESS] Updated predict_session signature!")
else:
    pred_sig_old_crlf = pred_sig_old.replace("\n", "\r\n")
    if pred_sig_old_crlf in content:
        content = content.replace(pred_sig_old_crlf, pred_sig_new)
        print("[SUCCESS] Updated predict_session signature (CRLF)!")
    else:
        print("[ERROR] Could not match pred_sig_old")

# 5. In predict_session, apply ST-GCN residual late fusion onto z_cal_16
fusion_old = """        with torch.no_grad():
            l16, l4, gate_w = self.joint_model(x_tensor)
            p16_pred = torch.softmax(l16 / self.T_meta, dim=-1)
            p4_pred = torch.sigmoid(l4)

        l16_arr = l16[0].cpu().numpy()
        p16_arr = p16_pred[0].cpu().numpy()
        p4_arr = p4_pred[0].cpu().numpy()
        gate_arr = gate_w[0].cpu().numpy()

        p_clean_16 = float(p16_arr[0])
        sorted_p16 = np.sort(p16_arr)
        top_margin = float(sorted_p16[-1] - sorted_p16[-2])

        best_state, defects_tuple, joint_is_clean, state_conf = decode_hybrid(
            p16_arr, p4_arr, eta=self.eta, tau_clean=self.tau_clean_safe
        )"""

fusion_new = """        with torch.no_grad():
            l16, l4, gate_w = self.joint_model(x_tensor)
            p16_pred = torch.softmax(l16 / self.T_meta, dim=-1)
            p4_pred = torch.sigmoid(l4)

        l16_arr = l16[0].cpu().numpy()
        p16_arr = p16_pred[0].cpu().numpy()
        p4_arr = p4_pred[0].cpu().numpy()
        gate_arr = gate_w[0].cpu().numpy()

        # [v10.4]: 시점별 벡터 스케일링 로짓
        scaler_cfg = getattr(self, 'view_vector_scalers', {}).get(norm_view, None)
        if scaler_cfg is not None:
            a_v = np.array(scaler_cfg['scale'], dtype=np.float32)
            b_v = np.array(scaler_cfg['bias'], dtype=np.float32)
            z_cal_16 = l16_arr * a_v + b_v
        else:
            z_cal_16 = l16_arr

        # [v10.5 신규]: ST-GCN + Transformer 잔차 결합 (Residual Late Fusion)
        delta_z16 = np.zeros(16, dtype=np.float32)
        if self.stgcn_model is not None and x_14d is not None:
            try:
                if isinstance(x_14d, np.ndarray):
                    if x_14d.ndim == 3:
                        x_14d_t = torch.from_numpy(x_14d[None, ...]).float().to(self.device)
                    else:
                        x_14d_t = torch.from_numpy(x_14d).float().to(self.device)
                else:
                    x_14d_t = x_14d if x_14d.ndim == 4 else x_14d.unsqueeze(0)
                with torch.no_grad():
                    dz16_t, _ = self.stgcn_model(x_14d_t)
                    delta_z16 = dz16_t[0].cpu().numpy()
            except Exception as e:
                warnings.warn(f"ST-GCN residual inference skipped: {e}")

        # 시점별 최적 잔차 게이팅
        is_front_view = norm_view in ['view2', 'view3', 'view4']
        g_v = self.g_front if is_front_view else self.g_rear
        z_final_16 = z_cal_16 + g_v * delta_z16

        # 최종 16-State 확률 분포 산출
        p16_final = np.exp(z_final_16 - np.max(z_final_16))
        p16_final = p16_final / np.sum(p16_final)

        p_clean_16 = float(p16_final[0])
        sorted_p16 = np.sort(p16_final)
        top_margin = float(sorted_p16[-1] - sorted_p16[-2])

        best_state, defects_tuple, joint_is_clean, state_conf = decode_hybrid(
            p16_final, p4_arr, eta=self.eta, tau_clean=self.tau_clean_safe
        )"""

if fusion_old in content:
    content = content.replace(fusion_old, fusion_new)
    print("[SUCCESS] Replaced inference and residual fusion logic!")
else:
    fusion_old_crlf = fusion_old.replace("\n", "\r\n")
    if fusion_old_crlf in content:
        content = content.replace(fusion_old_crlf, fusion_new)
        print("[SUCCESS] Replaced inference and residual fusion logic (CRLF)!")
    else:
        print("[ERROR] Could not match fusion_old")

# 6. Update layer 4 risk calculation to use p16_final
layer4_old = """            # [v10.4 신규]: 시점별 벡터 스케일링 기반 동적 위험도(Risk-Aware) 계산
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
            is_risk_excessive = (risk_val > self.risk_threshold)"""

layer4_new = """            # [v10.5]: 잔차 융합 최종 확률(p16_final) 기반 동적 위험도(Risk-Aware) 계산
            H_max = np.log(16.0)
            log_p_cal = np.log(np.clip(p16_final, 1e-8, 1.0))
            entropy_val = -float(np.sum(p16_final * log_p_cal))
            sorted_p = np.sort(p16_final)
            margin_gap = float(sorted_p[-1] - sorted_p[-2])
            risk_val = float((entropy_val / H_max) * (1.0 - margin_gap))
            is_risk_excessive = (risk_val > self.risk_threshold)"""

if layer4_old in content:
    content = content.replace(layer4_old, layer4_new)
    print("[SUCCESS] Updated Layer 4 dynamic risk gate!")
else:
    layer4_old_crlf = layer4_old.replace("\n", "\r\n")
    if layer4_old_crlf in content:
        content = content.replace(layer4_old_crlf, layer4_new)
        print("[SUCCESS] Updated Layer 4 dynamic risk gate (CRLF)!")
    else:
        print("[ERROR] Could not match layer4_old")

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[DONE] Successfully generated {dst_path}!")

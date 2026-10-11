# -*- coding: utf-8 -*-
"""
Script to construct babel_squat_integrated_v10_5.py cleanly from v10_4.py
"""

src_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_4.py"
dst_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_5.py"

with open(src_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update version identifiers
text = text.replace("BabelSquatIntegratedEngineV10_4", "BabelSquatIntegratedEngineV10_5")
text = text.replace('VERSION = "v10.4"', 'VERSION = "v10.5"')
text = text.replace("v10.4", "v10.5")
text = text.replace("v10_4", "v10_5")

# 2. Add import for STGCNTransformerResidualNet
target_import = "from src.features.skeletal_normalizer import normalize_skeletal_lengths"
new_import = """from src.features.skeletal_normalizer import normalize_skeletal_lengths
from src.models.second_models.babel_squat.stgcn_transformer_net import STGCNTransformerResidualNet"""
text = text.replace(target_import, new_import, 1)

# 3. Add 14D extraction method
extractor_code = '''
    @staticmethod
    def extract_14d_tensor(kps_64: np.ndarray) -> np.ndarray:
        """
        64프레임 33관절 시퀀스 (64, 33, 3)로부터 14D 시공간 텐서 (14, 64, 33) 고속 추출 (0.15ms 소요)
        """
        mid_hip = (kps_64[:, 23:24, :] + kps_64[:, 24:25, :]) * 0.5
        rel_pos = kps_64 - mid_hip
        vel = np.gradient(rel_pos, axis=0)
        acc = np.gradient(vel, axis=0)
        parents = [
            0, 0, 1, 2, 0, 4, 5, 3, 6, 0, 0,
            12, 11, 11, 12, 13, 14, 15, 16, 15, 16, 15, 16,
            11, 12, 23, 24, 25, 26, 27, 28, 27, 28
        ]
        parent_pos = rel_pos[:, parents, :]
        bone_vec = rel_pos - parent_pos
        bone_len = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
        bone_dir = bone_vec / (bone_len + 1e-7)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        comb = np.concatenate([rel_pos, vel, acc, bone_dir, bone_len, speed], axis=-1)
        return np.transpose(comb, (2, 0, 1)).astype(np.float32)
'''

# Place extractor before normalize_view
text = text.replace("    @staticmethod\n    def normalize_view(view: str) -> str:", extractor_code + "\n    @staticmethod\n    def normalize_view(view: str) -> str:")

# 4. In __init__, add stgcn_model loading
target_init = """        # [v10.5 신규]: 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(self.weights_dir, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}"""

new_init = """        # [v10.4]: 시점별 16D 벡터 스케일러 파라미터 로드
        scaler_path = os.path.join(self.weights_dir, "view_vector_scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.view_vector_scalers = json.load(f)
        else:
            self.view_vector_scalers = {}

        # [v10.5 신규]: 14D ST-GCN + Temporal Transformer 경량 잔차 신경망 로드
        self.stgcn_weights_path = os.path.join(self.weights_dir, "stgcn_transformer_residual_weights.pth")
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

text = text.replace(target_init, new_init)

# 5. In _synthesize_diagnosis, accept x_14d
target_synth_def = """    def _synthesize_diagnosis(
        self,
        sub_results: Dict[str, Dict[str, Any]],
        camera_view: str,
        c4_raw_feats: Optional[np.ndarray] = None,
        quality_raw_feats: Optional[np.ndarray] = None,
        kinetic_raw_feats: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:"""

new_synth_def = """    def _synthesize_diagnosis(
        self,
        sub_results: Dict[str, Dict[str, Any]],
        camera_view: str,
        c4_raw_feats: Optional[np.ndarray] = None,
        quality_raw_feats: Optional[np.ndarray] = None,
        kinetic_raw_feats: Optional[np.ndarray] = None,
        x_14d: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:"""

text = text.replace(target_synth_def, new_synth_def)

# 6. In _synthesize_diagnosis, apply residual fusion
target_fusion = """        with torch.no_grad():
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

new_fusion = """        with torch.no_grad():
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
                    x_14d_t = torch.from_numpy(x_14d[None, ...] if x_14d.ndim == 3 else x_14d).float().to(self.device)
                else:
                    x_14d_t = x_14d if x_14d.ndim == 4 else x_14d.unsqueeze(0)
                with torch.no_grad():
                    dz16_t, _ = self.stgcn_model(x_14d_t)
                    delta_z16 = dz16_t[0].cpu().numpy()
            except Exception as e:
                warnings.warn(f"ST-GCN residual inference skipped: {e}")

        # 시점별 최적 잔차 게이팅 (전방 0.08, 후방 0.00)
        is_front_view = norm_view in ['view2', 'view3', 'view4']
        g_v = self.g_front if is_front_view else self.g_rear
        z_final_16 = z_cal_16 + g_v * delta_z16

        p16_final = np.exp(z_final_16 - np.max(z_final_16))
        p16_final = p16_final / np.sum(p16_final)

        p_clean_16 = float(p16_final[0])
        sorted_p16 = np.sort(p16_final)
        top_margin = float(sorted_p16[-1] - sorted_p16[-2])

        best_state, defects_tuple, joint_is_clean, state_conf = decode_hybrid(
            p16_final, p4_arr, eta=self.eta, tau_clean=self.tau_clean_safe
        )"""

text = text.replace(target_fusion, new_fusion)

# 7. In Layer 4, update risk calculation to use p16_final
target_layer4 = """            # [v10.5 신규]: 시점별 벡터 스케일링 기반 동적 위험도(Risk-Aware) 계산
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

new_layer4 = """            # [v10.5]: 잔차 융합 최종 확률(p16_final) 기반 동적 위험도(Risk-Aware) 계산
            H_max = np.log(16.0)
            log_p_cal = np.log(np.clip(p16_final, 1e-8, 1.0))
            entropy_val = -float(np.sum(p16_final * log_p_cal))
            sorted_p = np.sort(p16_final)
            margin_gap = float(sorted_p[-1] - sorted_p[-2])
            risk_val = float((entropy_val / H_max) * (1.0 - margin_gap))
            is_risk_excessive = (risk_val > self.risk_threshold)"""

text = text.replace(target_layer4, new_layer4)

# 8. In predict_integrated, extract x_14d and pass to _synthesize_diagnosis
target_call = """        return self._synthesize_diagnosis(
            sub_results=sub_results,
            camera_view=camera_view,
            c4_raw_feats=c4_raw_feats,
            quality_raw_feats=quality_raw_feats,
            kinetic_raw_feats=kinetic_raw_feats
        )"""

new_call = """        # [v10.5]: 14D 관절 채널 시공간 텐서 추출
        x_14d = self.extract_14d_tensor(kps_sequence_3d)

        return self._synthesize_diagnosis(
            sub_results=sub_results,
            camera_view=camera_view,
            c4_raw_feats=c4_raw_feats,
            quality_raw_feats=quality_raw_feats,
            kinetic_raw_feats=kinetic_raw_feats,
            x_14d=x_14d
        )"""

text = text.replace(target_call, new_call)

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"[DONE] Successfully generated {dst_path}!")

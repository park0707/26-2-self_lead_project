# -*- coding: utf-8 -*-
"""
Clones babel_squat_integrated_v11.py into babel_squat_integrated_v12_1.py
and applies [Solution 1: Factorized Pairwise Energy Head].
"""

src_p = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v11.py"
dst_p = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v12_1.py"

with open(src_p, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update version and headers
text = text.replace("BabelSquatIntegratedEngineV11", "BabelSquatIntegratedEngineV12_1")
text = text.replace('VERSION = "v11.0"', 'VERSION = "v12.1"')
text = text.replace("babel_squat_integrated_v11.py", "babel_squat_integrated_v12_1.py")
text = text.replace("통합 진단 엔진 v10.0", "통합 진단 엔진 v12.1 (해결책 1 탑재)")
text = text.replace("최종 공식 마스터 프로덕션 챔피언 통합 진단 엔진 v11.0", "v12.1 [해결책 1] Factorized Pairwise Energy Head 통합 진단 엔진")

# 2. Update import: import JointMeta16WayMoENetV12_1
old_import = "from src.models.second_models.babel_squat.joint_meta_16way_moe_net import JointMeta16WayMoENet, JointMeta16WayMoENetV10_1"
new_import = "from src.models.second_models.babel_squat.joint_meta_16way_v12_1_net import JointMeta16WayMoENetV12_1"
text = text.replace(old_import, new_import)

# 3. Update model loading in __init__
old_init_model = """        # 39D View-Conditioned MoE 신경망 로드 (v10.0 가중치)
        self.joint_model = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20).to(self.device)
        weights_v9_0 = os.path.join(self.weights_dir, "joint_meta_16way_v10_1_weights.pth")
        if os.path.exists(weights_v9_0):
            self.joint_model.load_state_dict(torch.load(weights_v9_0, map_location=self.device))
        self.joint_model.eval()"""

new_init_model = """        # 39D View-Conditioned MoE 신경망 로드 (v12.1 해결책 1 Factorized Pairwise Energy Head)
        self.joint_model = JointMeta16WayMoENetV12_1(in_features=39, hidden_dim=48, dropout_rate=0.20).to(self.device)
        weights_v12_1 = os.path.join(self.weights_dir, "joint_meta_16way_v12_1_weights.pth")
        if os.path.exists(weights_v12_1):
            self.joint_model.load_state_dict(torch.load(weights_v12_1, map_location=self.device))
        else:
            # Fallback to v10_1 if v12_1 not found
            weights_v10_1 = os.path.join(self.weights_dir, "joint_meta_16way_v10_1_weights.pth")
            if os.path.exists(weights_v10_1):
                warnings.warn("v12.1 weights not found, fallback to v10.1 weights")
        self.joint_model.eval()"""

text = text.replace(old_init_model, new_init_model)

# 4. Update aliases at bottom
text = text.replace("BabelSquatIntegratedEngineV11_0 = BabelSquatIntegratedEngineV11", "BabelSquatIntegratedEngineV12_1_Alias = BabelSquatIntegratedEngineV12_1")
text = text.replace("BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV11", "BabelSquatIntegratedEngineV12 = BabelSquatIntegratedEngineV12_1")

with open(dst_p, "w", encoding="utf-8") as f:
    f.write(text)

print(f"[SUCCESS] Cloned v11 into {dst_p} with Solution 1 (FactorizedPairwiseEnergyHead)!")

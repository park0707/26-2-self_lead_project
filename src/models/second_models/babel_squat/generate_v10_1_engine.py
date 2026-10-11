# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_v10_1_engine.py
Purpose: babel_squat_integrated_v10_1.py 생성 및 확정 파라미터 동기화
"""

source_file = r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_0.py'
target_file = r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_1.py'

with open(source_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace references
text = text.replace('BabelSquatIntegratedEngineV10_0', 'BabelSquatIntegratedEngineV10_1')
text = text.replace('VERSION = "v10.0"', 'VERSION = "v10.1"')
text = text.replace('joint_meta_16way_v10_0_weights.pth', 'joint_meta_16way_v10_1_weights.pth')
text = text.replace('joint_meta_16way_v10_0_meta.json', 'joint_meta_16way_v10_1_meta.json')
text = text.replace('from src.models.second_models.babel_squat.joint_meta_16way_moe_net import JointMeta16WayMoENet', 
                    'from src.models.second_models.babel_squat.joint_meta_16way_moe_net import JointMeta16WayMoENet, JointMeta16WayMoENetV10_1')
text = text.replace('JointMeta16WayMoENet(in_features=39, hidden_dim=48, dropout_rate=0.20)', 
                    'JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)')

# Default parameters
text = text.replace('deadband_low: float = 0.32', 'deadband_low: float = 0.30')
text = text.replace('deadband_high: float = 0.38', 'deadband_high: float = 0.36')
text = text.replace('self.tau_clean_safe = 0.38', 'self.tau_clean_safe = 0.36')
text = text.replace('self.eta = 0.10', 'self.eta = 0.05')

# Bottom alias
text = text.replace('BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV10_0', 
                    'BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV10_1')

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Generated babel_squat_integrated_v10_1.py successfully!')

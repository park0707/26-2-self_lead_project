# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_v10_2_engine.py
Purpose: babel_squat_integrated_v10_2.py 생성 (c2 v8 불확실성 인식 모델 탑재)
"""

source_file = r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_1.py'
target_file = r'D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_2.py'

with open(source_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace version and class names
text = text.replace('BabelSquatIntegratedEngineV10_1', 'BabelSquatIntegratedEngineV10_2')
text = text.replace('VERSION = "v10.1"', 'VERSION = "v10.2"')

# Replace c2 import: c2_model_v7_1 -> c2_model_v8
text = text.replace('from src.models.second_models.babel_squat.c2_model_v7_1 import BabelSquatC2ModelV7_1',
                    'from src.models.second_models.babel_squat.c2_model_v8 import BabelSquatC2ModelV8')
text = text.replace('self.c2_model = BabelSquatC2ModelV7_1(weights_path=weights_c2)',
                    'self.c2_model = BabelSquatC2ModelV8(weights_path=weights_c2)')

# Replace rear condition in _synthesize_diagnosis:
# In rear views, allow c2 calibrated probability up to 0.38
text = text.replace('(p_cal[1] < 0.35)', '(p_cal[1] < 0.38)')

# Alias at bottom
text = text.replace('BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV10_1',
                    'BabelSquatIntegratedEngine = BabelSquatIntegratedEngineV10_2')

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Generated babel_squat_integrated_v10_2.py successfully!')

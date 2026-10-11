import os

path = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_weak_supervision.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "def get_angle(A, B, C):" in line:
        skip = True
        new_lines.append("    y = np.load(r'D:\\src\\models\\second_models\\babel_stiff_deadlift\\y_clean.npy')\n")
    if "y = np.array(y_clean, dtype=np.int32)" in line:
        skip = False
        continue
    
    if not skip:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Fixed.")

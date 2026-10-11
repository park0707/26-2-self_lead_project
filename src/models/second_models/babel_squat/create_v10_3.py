# -*- coding: utf-8 -*-
"""
Helper script to clone babel_squat_integrated_v10_2.py into v10_3.py
with Skeletal Normalization & Dual-Stream integration.
"""

src_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_2.py"
dst_path = r"D:\src\models\second_models\babel_squat\babel_squat_integrated_v10_3.py"

with open(src_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update version identifiers
content = content.replace("BabelSquatIntegratedEngineV10_2", "BabelSquatIntegratedEngineV10_3")
content = content.replace("v10.2", "v10.3")
content = content.replace("v10_2", "v10_3")

# 2. Add import for skeletal normalizer
import_line = "from src.features.skeletal_normalizer import normalize_skeletal_lengths\n"
content = import_line + content

# 3. Locate predict_integrated and insert dual-stream logic
old_block = """        sub_results = {}
        sub_results['c1'] = self.c1_model.predict_selective(kps_sequence_3d)
        sub_results['c2'] = self.c2_model.predict_selective(kps_sequence_3d, camera_view=norm_view)
        sub_results['c3'] = self.c3_model.predict_selective(kps_sequence_3d)
        sub_results['c4'] = self.c4_model.predict_selective(kps_sequence_3d, camera_view=norm_view)

        if c4_raw_feats is None:
            c4_raw_feats = self.c4_model.extract_features(kps_sequence_3d)

        if quality_raw_feats is None and kps_sequence_3d.shape[0] == 64 and kps_sequence_3d.shape[1] == 33:
            quality_raw_feats = self.compute_12d_quality(kps_sequence_3d)

        if kinetic_raw_feats is None and kps_sequence_3d.shape[0] == 64 and kps_sequence_3d.shape[1] == 33:
            kinetic_raw_feats = self.compute_9d_kinetic(kps_sequence_3d)"""

new_block = """        # [v10.3 신규]: Skeletal Length Normalization & Dual-Stream 분기
        stream_a, stream_b, norm_metrics = normalize_skeletal_lengths(kps_sequence_3d)

        sub_results = {}
        sub_results['c1'] = self.c1_model.predict_selective(stream_a)
        sub_results['c2'] = self.c2_model.predict_selective(kps_sequence_3d, camera_view=norm_view)
        sub_results['c3'] = self.c3_model.predict_selective(stream_a)
        sub_results['c4'] = self.c4_model.predict_selective(stream_b, camera_view=norm_view)

        if c4_raw_feats is None:
            c4_raw_feats = self.c4_model.extract_features(stream_b)

        if quality_raw_feats is None and kps_sequence_3d.shape[0] == 64 and kps_sequence_3d.shape[1] == 33:
            quality_raw_feats = self.compute_12d_quality(stream_b)

        if kinetic_raw_feats is None and kps_sequence_3d.shape[0] == 64 and kps_sequence_3d.shape[1] == 33:
            kinetic_raw_feats = self.compute_9d_kinetic(stream_a)"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("[SUCCESS] Matched and updated predict_integrated() with Dual-Stream!")
else:
    print("[WARNING] old_block not matched exactly! Checking line endings...")
    old_block_crlf = old_block.replace("\n", "\r\n")
    if old_block_crlf in content:
        content = content.replace(old_block_crlf, new_block)
        print("[SUCCESS] Matched with CRLF and updated!")
    else:
        print("[ERROR] Could not find old_block in content!")

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[DONE] Saved v10.3 to {dst_path}")

import os
import glob
import json
from collections import defaultdict

label_dir = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
print(f"Total json files: {len(files)}")

desc_counts = defaultdict(int)
mismatch_cases = []
pattern_summary = defaultdict(list)

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        t_info = data.get("type_info", {})
        desc = t_info.get("description", "")
        conds = t_info.get("conditions", [])
        c_vals = {c["condition"]: c["value"] for c in conds}
        fn = os.path.basename(f)
        is_error = "[오류]" in fn
        desc_counts[desc] += 1
        
        # conditions tuple
        cond_summary = tuple(sorted([(k, v) for k, v in c_vals.items()]))
        pattern_summary[(desc, is_error)].append((fn, c_vals))
    except Exception as e:
        print(f"Error {f}: {e}")

print("\n--- Description summary ---")
for d, cnt in sorted(desc_counts.items(), key=lambda x: -x[1]):
    print(f"[{cnt} files] : '{d}'")

print("\n--- Description vs Conditions Analysis ---")
for (desc, is_err), samples in sorted(pattern_summary.items(), key=lambda x: str(x[0])):
    print(f"\nDesc: '{desc}', is_error_folder: {is_err}, Count: {len(samples)}")
    # Sample condition values
    first_fn, first_c = samples[0]
    print(f"  Sample {first_fn}:")
    for k, v in first_c.items():
        print(f"    - {k}: {v}")

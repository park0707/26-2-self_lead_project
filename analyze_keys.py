import os
import glob
import json
from collections import defaultdict

label_dir = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

key_analysis = {}

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        t_info = data.get("type_info", {})
        key = t_info.get("key", "")
        desc = t_info.get("description", "")
        conds = t_info.get("conditions", [])
        c_vals = {c["condition"]: c["value"] for c in conds}
        fn = os.path.basename(f)
        is_err = "[오류]" in fn
        
        if key not in key_analysis:
            key_analysis[key] = {
                "key": key,
                "desc": desc,
                "is_error": is_err,
                "conditions": c_vals,
                "count": 0,
                "files": []
            }
        key_analysis[key]["count"] += 1
        key_analysis[key]["files"].append(fn)
    except Exception as e:
        print(f"Error {f}: {e}")

out_txt = r"D:\side_lunge_keys_detailed.txt"
with open(out_txt, "w", encoding="utf-8") as fp:
    fp.write(f"Total files: {len(files)}, Total Keys: {len(key_analysis)}\n\n")
    for key in sorted(key_analysis.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        item = key_analysis[key]
        fp.write(f"Key {key} (Count: {item['count']}, Error: {item['is_error']}):\n")
        fp.write(f"  Description: '{item['desc']}'\n")
        fp.write(f"  Conditions in JSON:\n")
        for c_name, c_val in item["conditions"].items():
            fp.write(f"    - {c_name}: {c_val}\n")
        fp.write("\n")

print(f"Key details written to {out_txt}")

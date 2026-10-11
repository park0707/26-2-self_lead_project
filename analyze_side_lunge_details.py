import os
import glob
import json
from collections import defaultdict

label_dir = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

print(f"Total json files: {len(files)}")

desc_summary = defaultdict(lambda: {"count": 0, "samples": [], "cond4_vals": defaultdict(int), "is_error_folder": defaultdict(int)})

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        t_info = data.get("type_info", {})
        desc = t_info.get("description", "")
        conds = t_info.get("conditions", [])
        c_vals = {c["condition"]: c["value"] for c in conds}
        fn = os.path.basename(f)
        is_err = "[오류]" in fn
        c4_val = c_vals.get("척추의 중립", None)
        
        info = desc_summary[desc]
        info["count"] += 1
        info["cond4_vals"][c4_val] += 1
        info["is_error_folder"][is_err] += 1
        if len(info["samples"]) < 2:
            info["samples"].append((fn, c_vals))
    except Exception as e:
        print(f"Error {f}: {e}")

# Save output to text file so we can view it cleanly
out_file = r"D:\side_lunge_label_analysis.txt"
with open(out_file, "w", encoding="utf-8") as out:
    out.write(f"Total files: {len(files)}\n\n")
    for desc, info in sorted(desc_summary.items(), key=lambda x: -x[1]["count"]):
        out.write(f"Description: '{desc}' | Total: {info['count']}\n")
        out.write(f"  Folder error flag: {dict(info['is_error_folder'])}\n")
        out.write(f"  척추의 중립 (True:정상, False:결함): {dict(info['cond4_vals'])}\n")
        first_fn, first_c = info["samples"][0]
        out.write(f"  Sample conditions ({first_fn}):\n")
        for k, v in first_c.items():
            out.write(f"    - {k}: {v}\n")
        out.write("\n")

print(f"Analysis saved to {out_file}")

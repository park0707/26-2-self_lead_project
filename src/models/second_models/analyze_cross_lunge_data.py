# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np

def analyze_cross_lunge_labels():
    label_dir = r"D:\Data\01_raw_data\labels_json\08_크로스_런지"
    json_files = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
    print(f"Total json files found: {len(json_files)}")

    key_dict = {}
    cond_counter = {}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                tinfo = d.get("type_info", {})
                k = str(tinfo.get("key", ""))
                if k not in key_dict:
                    key_dict[k] = {
                        "type": tinfo.get("type"),
                        "pose": tinfo.get("pose"),
                        "exercise": tinfo.get("exercise"),
                        "description": tinfo.get("description"),
                        "conditions": tinfo.get("conditions", []),
                        "count": 0
                    }
                key_dict[k]["count"] += 1
                for idx, c in enumerate(tinfo.get("conditions", [])):
                    cname = c.get("condition", "")
                    cval = c.get("value")
                    c_key = f"cond_{idx+1}: {cname}"
                    if c_key not in cond_counter:
                        cond_counter[c_key] = {"True": 0, "False": 0}
                    cond_counter[c_key][str(cval)] += 1
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    print("\n--- Keys and Conditions ---")
    for k in sorted(key_dict.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        info = key_dict[k]
        print(f"Key {k} (count={info['count']}): type={info['type']}, pose={info['pose']}, desc={info['description']}")
        for idx, c in enumerate(info["conditions"]):
            print(f"   [C{idx+1}] {c.get('condition')}: {c.get('value')}")

    print("\n--- Overall Condition Stats ---")
    for c_key, val_counts in cond_counter.items():
        print(f"{c_key} -> {val_counts}")

def analyze_cross_lunge_npz():
    dirs_to_check = [
        r"D:\Data\전처리_데이터\MediaPipe\08_크로스_런지",
        r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지",
        r"D:\Data\MediaPipe\전처리\08_크로스_런지"
    ]
    for d in dirs_to_check:
        if os.path.exists(d):
            npz_files = glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)
            print(f"\nDirectory: {d} -> found {len(npz_files)} npz files")
            if npz_files:
                sample = npz_files[0]
                print(f"Sample file: {sample}")
                try:
                    data = np.load(sample)
                    print(f"Keys in npz: {list(data.keys())}")
                    for k in data.keys():
                        arr = data[k]
                        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}")
                except Exception as e:
                    print(f"Error loading npz: {e}")
        else:
            print(f"\nDirectory NOT exists: {d}")

if __name__ == "__main__":
    analyze_cross_lunge_labels()
    analyze_cross_lunge_npz()

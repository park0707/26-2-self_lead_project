import json
import glob
import os

print("=== 01_정상인데_결함으로_분류_FP ===")
for p in sorted(glob.glob(r"D:\Data\temp\01_정상인데_결함으로_분류_FP\*\*.json")):
    if "sample_info" in p or "manifest" in p:
        continue
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    ti = d.get("type_info", {})
    conds = ti.get("conditions", [])
    c5 = conds[4] if len(conds) >= 5 else {}
    print(f"File: {os.path.basename(p)}")
    print(f"  Key: {ti.get('key')} | Desc: {ti.get('description')}")
    print(f"  Condition 5: {c5.get('condition')} -> {c5.get('value')} (Label={'결함(1)' if c5.get('value') is False else '정상(0)'})")
    for idx, c in enumerate(conds):
        print(f"    Cond {idx+1}: {c.get('condition')} = {c.get('value')}")
    print("-" * 50)

print("\n=== 02_결함인데_정상으로_분류_FN ===")
for p in sorted(glob.glob(r"D:\Data\temp\02_결함인데_정상으로_분류_FN\*\*.json")):
    if "sample_info" in p or "manifest" in p:
        continue
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    ti = d.get("type_info", {})
    conds = ti.get("conditions", [])
    c5 = conds[4] if len(conds) >= 5 else {}
    print(f"File: {os.path.basename(p)}")
    print(f"  Key: {ti.get('key')} | Desc: {ti.get('description')}")
    print(f"  Condition 5: {c5.get('condition')} -> {c5.get('value')} (Label={'결함(1)' if c5.get('value') is False else '정상(0)'})")
    for idx, c in enumerate(conds):
        print(f"    Cond {idx+1}: {c.get('condition')} = {c.get('value')}")
    print("-" * 50)

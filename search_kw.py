import os
import glob
import re

search_dirs = [r"D:\자주프 문서", r"D:\src", r"D:\Data"]
keywords = ["라벨과 설명", "설명과 라벨", "설명이 다른", "라벨이 다른", "불일치"]

results = []

for s_dir in search_dirs:
    for root, dirs, files in os.walk(s_dir):
        for f in files:
            if f.endswith(('.md', '.py', '.txt', '.json')):
                # skip large json or data files
                if f.endswith('.json') and ('MediaPipe' in root or 'labels_json' in root):
                    continue
                fp = os.path.join(root, f)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as fp_in:
                        lines = fp_in.readlines()
                        for i, line in enumerate(lines):
                            for kw in keywords:
                                if kw in line:
                                    results.append(f"{fp}:{i+1}: {line.strip()[:150]}")
                                    break
                except Exception:
                    pass

out_fp = r"D:\label_desc_search_results.txt"
with open(out_fp, "w", encoding="utf-8") as out:
    for r in results:
        out.write(r + "\n")

print(f"Found {len(results)} matches. Saved to {out_fp}")

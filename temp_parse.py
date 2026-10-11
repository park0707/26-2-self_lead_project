import os
import glob
import re
from collections import defaultdict

docs_dir = r'd:/자주프 문서'
pattern = re.compile(r'3-State F1.*?([0-9]{2}\.[0-9]{2})%', re.IGNORECASE)
pattern2 = re.compile(r'([0-9]{2}\.[0-9]{2})%.*?3-State F1', re.IGNORECASE)
model_pattern = re.compile(r'([a-z]{1,2}[1-5])')

results = defaultdict(list)

for root, _, files in os.walk(docs_dir):
    for f in files:
        if f.endswith('.md'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Search model in content or filename
                    model_name = None
                    m = model_pattern.search(f)
                    if m:
                        model_name = m.group(1)
                    
                    if model_name:
                        f1_matches = pattern.findall(content) + pattern2.findall(content)
                        if f1_matches:
                            for match in f1_matches:
                                results[model_name].append((float(match), path))
            except Exception as e:
                pass

final_results = {}
for model, scores in results.items():
    best_score = max(scores, key=lambda x: x[0])
    final_results[model] = best_score

print("=== 3-State F1 Scores ===")
for model in sorted(final_results.keys()):
    score, path = final_results[model]
    print(f"{model}: {score}% (Found in {os.path.basename(path)})")

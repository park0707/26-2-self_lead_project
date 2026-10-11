import os
import glob
import re

dirs_to_search = [r'd:/자주프 문서', r'd:/src', r'd:/기타']
pattern1 = re.compile(r'3-State F1.*?([0-9]{2}\.[0-9]{2})%', re.IGNORECASE)
pattern2 = re.compile(r'([0-9]{2}\.[0-9]{2})%.*?3-State F1', re.IGNORECASE)
model_pattern = re.compile(r'\b(sq[1-5]|d[1-5]|sd[1-3]|bl[1-5]|dl[1-5]|sl[1-5]|cl[1-5]|gm[1-3]|[cd][1-5])\b', re.IGNORECASE)

results = {}

for d in dirs_to_search:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.md') or f.endswith('.py') or f.endswith('.json'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        
                        f1_matches = pattern1.findall(content) + pattern2.findall(content)
                        if not f1_matches:
                            continue
                            
                        m = model_pattern.findall(f)
                        models_in_file = set([x.lower() for x in m])
                        
                        if not models_in_file:
                            m_content = model_pattern.findall(content)
                            models_in_file = set([x.lower() for x in m_content])
                            
                        best_f1_in_file = max([float(x) for x in f1_matches])
                        
                        for model in models_in_file:
                            if model not in results:
                                results[model] = best_f1_in_file
                            else:
                                results[model] = max(results[model], best_f1_in_file)
                                
                except Exception as e:
                    pass

print("=== 3-State F1 Scores ===")
for model in sorted(results.keys()):
    score = results[model]
    print(f"{model}: {score}%")

import os
import re

docs_dir = r'D:\자주프 문서'

pattern = re.compile(r'3-State F1.*?([0-9]{2}\.[0-9]{2})%', re.IGNORECASE)
pattern2 = re.compile(r'([0-9]{2}\.[0-9]{2})%.*?3-State F1', re.IGNORECASE)

print("Squat models:")
for f in os.listdir(r'D:\자주프 문서\5주차\바벨 스쿼트 오류 분류 모델'):
    if f.endswith('.md'):
        with open(os.path.join(r'D:\자주프 문서\5주차\바벨 스쿼트 오류 분류 모델', f), 'r', encoding='utf-8') as file:
            content = file.read()
            f1s = pattern.findall(content) + pattern2.findall(content)
            if f1s:
                print(f, max([float(x) for x in f1s]))

print("Good Morning models:")
for root, _, files in os.walk(docs_dir):
    for f in files:
        if '굿모닝' in f or 'good_morning' in f:
            if f.endswith('.md'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    f1s = pattern.findall(content) + pattern2.findall(content)
                    if f1s:
                        print(f, max([float(x) for x in f1s]))

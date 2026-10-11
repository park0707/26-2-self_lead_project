import re
with open(r'D:\자주프 문서\6주차\보고서\generate_week6_report.py', 'r', encoding='utf-8') as f:
    text = f.read()
    
matches = re.findall(r'\[([^\]]+)\]', text)
for m in matches:
    if '3-State F1' in m or 'F1' in m:
        pass
        
table_data = re.findall(r'\[\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*".*?\]', text)
for t in table_data:
    print(t)

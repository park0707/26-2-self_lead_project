import ast
import re

with open(r'D:\자주프 문서\6주차\보고서\generate_week6_report.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_table = False
table_lines = []
for line in lines:
    if "main_data = [" in line:
        in_table = True
        table_lines.append(line.split('=', 1)[1].strip())
    elif in_table:
        table_lines.append(line.strip())
        if line.strip() == "]" or line.strip() == "],":
            break

table_str = " ".join(table_lines)
if table_str.endswith(","): table_str = table_str[:-1]

try:
    data = ast.literal_eval(table_str)
    for row in data:
        print(f"{row[1]}: {row[6]}")
except Exception as e:
    print(e)

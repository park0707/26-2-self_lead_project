# -*- coding: utf-8 -*-
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r'C:\Users\phj20\.gemini\antigravity\brain\c42088db-b497-45a4-bd4e-de7c9da83845\.system_generated\logs\transcript.jsonl'

lines = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        lines.append(json.loads(line))

for step in [6, 24, 30, 38, 43, 160, 164]:
    if step < len(lines):
        d = lines[step]
        stype = d.get('type')
        print(f"=== Step {step} ({stype}) ===")
        text = d.get('content', '')
        print(text[:1800])
        print("...\n" + "-"*50)

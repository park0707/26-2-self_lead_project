# -*- coding: utf-8 -*-
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
p = r"C:\Users\phj20\.gemini\antigravity\brain\5c55688e-5a69-4acb-8f20-dd6a04766992\.system_generated\logs\transcript.jsonl"
with open(p, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        d = json.loads(line)
        content = d.get("content", "")
        if any(k in content for k in ["채워", "0으로", "피쳐가", "피처가", "비어", "누락", "결측"]):
            if d.get("type") == "USER_INPUT":
                print(f"[USER]: {content.strip()}")
            elif d.get("type") == "PLANNER_RESPONSE" and any(k in content for k in ["채워", "비어", "누락"]):
                print(f"[AGENT]: {content[:200].strip()}...")

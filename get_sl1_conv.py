import json

p = r"C:\Users\phj20\.gemini\antigravity\brain\e6a99e02-31e3-42fc-8974-4d0e1fb28ed3\.system_generated\logs\transcript.jsonl"
out_p = r"D:\sl1_conversation_summary.txt"

with open(p, 'r', encoding='utf-8', errors='ignore') as f, open(out_p, 'w', encoding='utf-8') as out:
    for line in f:
        d = json.loads(line)
        stype = d.get('type', '')
        source = d.get('source', '')
        content = d.get('content', '')
        if stype == 'USER_INPUT':
            out.write(f"\n[USER]: {content}\n")
        elif stype == 'PLANNER_RESPONSE' and not d.get('tool_calls'):
            out.write(f"\n[ASSISTANT]: {content[:400]}...\n")

print(f"Summary written to {out_p}")

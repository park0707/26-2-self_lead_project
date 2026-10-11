# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: extract_external_ai_responses.py
Purpose: D:\자주프 문서\외부ai 응답 폴더 내의 PDF 3개에서 텍스트를 추출하여 저장하고 분석한다.
"""

import os
import sys
import fitz  # PyMuPDF

TARGET_DIR = r"D:\자주프 문서\외부ai 응답"
OUT_DIR = r"D:\src\models\second_models\babel_deadlift\external_ai_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

files = [
    "바벨 데드리프트 d5(바벨 궤적 및 몸 밀착) 모델 최적화 및 80%+ 돌파를 위한 선행.pdf",
    "새 채팅 _ 아주LLM.pdf",
    "제미나이 답변.pdf"
]

for fn in files:
    fp = os.path.join(TARGET_DIR, fn)
    out_txt_p = os.path.join(OUT_DIR, fn.replace(".pdf", ".txt"))
    
    print(f"[*] 처리 중: {fn}")
    doc = fitz.open(fp)
    full_text = []
    for page_idx, page in enumerate(doc):
        text = page.get_text()
        full_text.append(f"--- [Page {page_idx+1}] ---\n" + text)
    
    combined = "\n\n".join(full_text)
    with open(out_txt_p, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"    -> 텍스트 저장 완료: {out_txt_p} ({len(doc)}페이지, {len(combined)}자)")

print("[+] 전수 추출 완료!")

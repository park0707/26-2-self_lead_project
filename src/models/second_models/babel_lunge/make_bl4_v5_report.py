# -*- coding: utf-8 -*-
"""bl4_v5 vs bl4_v4b 비교 보고서(.docx/.md) 생성."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

OUT = r"D:\자주프 문서\6주차\바벨_런지_c4_bl4_v5_SpinePose_성능비교보고서"

SECTIONS = [
    ("1. 목적과 결론", [
        "목적: 스쿼트 모델에서 우수한 성능을 보인 SpinePose 3D 척추 피처를 런지 척추 중립 결함(c4)에도 적용하여 성능 향상 여부를 검증한다 (bl4_v5).",
        "결론: SpinePose 피처 추가는 오히려 성능 저하를 유발하였다. v5(sp_cl)의 AUC는 82.80%로 기존 v4b(cl) 82.93% 대비 소폭 감소했고, TCN 서브모델의 AUC는 80.61%에서 76.53%로 크게 하락(-4.08%p)하였다.",
        "따라서 런지 모델에 SpinePose 피처를 도입하는 방안은 기각하며, 기존의 MediaPipe 24D 기반의 v3(또는 v4b) 구조를 최종으로 유지한다."
    ]),
    ("2. 성능 저하 원인 분석 (왜 스쿼트와 다른가?)", [
        "스쿼트는 양발이 평행하고 골반이 정면을 향하는 대칭 동작이므로, 2D에서 3D 척추 곡률을 추정하는 SpinePose 모델이 높은 정확도를 보인다.",
        "반면 런지는 앞굽이 자세(Split Stance)로 인해 골반이 비틀어지고 다리 교차에 의한 신체 가림(Occlusion)이 심하게 발생한다.",
        "런지 영상에 스쿼트/정상 기립 기반으로 학습된 가상 3D 복원 모델(SpinePose)을 강제 적용하면, 골반 비틀림이 척추의 3D 곡률 왜곡으로 이어져 기존의 순수 비전(24D) 피처보다 더 큰 노이즈를 주입하는 결과(TCN 성능 급락)를 초래했다."
    ]),
    ("3. 공정 비교 프로토콜", [
        "동일 5-Fold StratifiedGroupKFold(피험자 9명 격리, seed 42), 동일 Confident Learning + SCE (v4b 구조).",
        "추가된 피처: SpinePose 31D 도메인 피처 + 6ch 시계열 척추 곡률(스케일 정규화).",
        "평가는 원본 라벨 787건 전수, 동일 탐색 격자/커버리지 하한 적용."
    ]),
]

HDR = ["변형", "AUC", "고정 0.44/0.51 (cov / F1)", "cov>=95% F1", "cov>=90% F1"]
ROWS = [
    ["cl (v4b: 기존 24D)", "82.93%", "97.2% / 75.35%", "76.44%", "78.01%"],
    ["sp_cl (v5: SpinePose 추가)", "82.80%", "96.6% / 75.16%", "76.24%", "77.81%"],
    ["ref (v3: 기존 24D, CL미적용)", "81.30%", "95.9% / 76.21%", "76.54%", "77.38%"],
    ["sp_ref (v5: SpinePose 추가, CL미적용)", "81.19%", "96.2% / 75.63%", "76.08%", "77.32%"],
]
HDR2 = ["서브모델 (CL 적용 기준)", "cl (v4b) AUC", "sp_cl (v5) AUC", "차이"]
ROWS2 = [["TCN", "80.61%", "76.53%", "-4.08%p"], ["LightGBM", "81.25%", "80.82%", "-0.43%p"],
         ["XGBoost", "81.27%", "81.44%", "+0.17%p"], ["CatBoost", "81.68%", "81.45%", "-0.23%p"]]

TAIL = [
    ("4. 향후 계획", [
        "SpinePose는 스쿼트와 같이 좌우 대칭성이 강한 동작에만 한정하여 사용한다.",
        "런지 c4 모델은 SpinePose를 배제하고, 기존 v3 (또는 라벨 노이즈가 강건해진 v4b)를 최종 상용 버전으로 동결한다."
    ]),
]

def build():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = Document()
    st = doc.styles['Normal']; st.font.name = 'Malgun Gothic'; st.font.size = Pt(10)
    t = doc.add_paragraph(); r = t.add_run("vidio-health-trainer | 바벨 런지 c4 SpinePose(v5) 도입 검증 보고서")
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    def table(hdr, rows):
        tb = doc.add_table(rows=1, cols=len(hdr)); tb.style = 'Light Grid Accent 1'
        for i, h in enumerate(hdr):
            tb.rows[0].cells[i].text = h
        for row in rows:
            cells = tb.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = v
        doc.add_paragraph()

    def sec(title, items):
        h = doc.add_paragraph(); rr = h.add_run(title); rr.font.bold = True; rr.font.size = Pt(13)
        for it in items:
            doc.add_paragraph(it, style='List Bullet')

    for s in SECTIONS:
        sec(*s)
    h = doc.add_paragraph(); rr = h.add_run("4. 결과 (OOF, N=787, 피험자 격리 5-Fold)"); rr.font.bold = True; rr.font.size = Pt(13)
    table(HDR, ROWS); table(HDR2, ROWS2)
    for s in TAIL:
        sec(*s)
    doc.save(OUT + ".docx")

    md = ["# bl4_v5 SpinePose 도입 성능 검증\n"]
    def mdt(hdr, rows):
        md.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in rows))
    for title, items in SECTIONS:
        md.append(f"## {title}\n" + "".join(f"- {i}\n" for i in items))
    md.append("## 4. 결과\n"); mdt(HDR, ROWS); mdt(HDR2, ROWS2)
    for title, items in TAIL:
        md.append(f"## {title}\n" + "".join(f"- {i}\n" for i in items))
    open(OUT + ".md", 'w', encoding='utf-8').write("\n".join(md))
    print("saved", OUT)

if __name__ == '__main__':
    build()

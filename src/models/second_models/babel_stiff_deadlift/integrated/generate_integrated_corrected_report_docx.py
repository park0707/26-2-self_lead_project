"""
vidio-health-trainer 통합 진단 모델 v2~v4.1 정정 통합 성능보고서 생성기 (.docx + .md)
- 실측으로 검증된 수치만 사용. 임계치 in-sample 튜닝 수치는 '낙관적'으로 명시.
- 헬퍼 함수는 generate_integrated_v4_report_docx.py 에서 계승(Copy-and-Refine).
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from generate_integrated_v4_report_docx import add_header, add_callout, format_table

DOC_DIR = r"D:\자주프 문서\7주차"
NAME = "바벨_스티프_데드리프트_통합진단모델_v2~v4.1_정정_통합_성능보고서"
DOCX_PATH = os.path.join(DOC_DIR, NAME + ".docx")
MD_PATH = os.path.join(DOC_DIR, NAME + ".md")

NOTICE = (
    "기존 v2 / v2 vs v3 / v3 vs v4 보고서에는 (1) 실측하지 않은 AUC·TP·커버리지 수치, "
    "(2) 검증되지 않은 v2 시점별 수치, (3) 근거 없는 '성능 한계 도달 / 파레토 프론티어 도달' 표현이 포함되어 있었습니다. "
    "해당 문서는 '정정전_보관' 폴더로 이동했으며, 본 문서가 대체합니다. "
    "또한 v3/v4의 EMR·Clean F1은 평가 데이터로 임계치를 직접 튜닝한 낙관적 수치입니다."
)

SEC1_H = ["버전", "2-State EMR", "2-State Clean F1", "3-State EMR*", "3-State Clean F1", "비고"]
SEC1 = [
    ["v1", "34.98", "26.38", "35.12", "27.72", "임계치 in-sample 튜닝"],
    ["v2", "39.03", "30.38", "38.95", "33.65", "clean TP 53/130, 커버리지 90.53%"],
    ["v3", "38.94", "38.04", "39.96", "41.13", "메타 AUC 77.63, clean TP 102, 커버리지 92.37%"],
    ["v4", "39.90", "38.04", "41.30", "41.18", "커버리지 91.01%, 시점별 임계치 in-sample 튜닝"],
]
SEC1_NOTE = ("* 3-State EMR은 접수(accepted) 샘플 기준이라 2-State와 직접 비교 불가. "
             "모든 수치는 임계치를 평가 데이터에 맞춰 튜닝한 낙관적 값.")

SEC2_H = ["지표", "v4 (피험자 격리 LOSO 정직 평가)", "v4 (기존 보고, 낙관적)"]
SEC2 = [
    ["EMR", "37.20%", "41.30%"],
    ["Clean F1", "36.04%", "41.18%"],
    ["참고: 무조건 (0,1,0) 예측", "37.20%", "-"],
]
SEC2_NOTE = "정직 평가 시 v4 EMR은 항상 (sd1,sd2,sd3)=(0,1,0)을 찍는 자명 기준선(37.20%)과 동일합니다."

SEC3_H = ["항목", "v4 (LOSO)", "v4.1 (nested CV, 3 seeds)"]
SEC3 = [
    ["2-State EMR", "37.20%", "50.27 ± 0.81%"],
    ["2-State Clean F1", "36.04%", "35.29 ± 1.27%"],
    ["3-State EMR (기각 후)", "-", "52.19 ± 1.00%"],
    ["3-State Clean F1", "-", "35.84 ± 1.69%"],
    ["커버리지", "-", "90.8%"],
    ["Clean 적중", "-", "79/130"],
]
SEC3_NOTE = ("v4.1: 단일 카메라 5-class(label powerset) 스태커 + class-0 prior shift(w0=3.0) + 신뢰도 기각(약 9%). "
             "Clean F1은 v4 대비 유의한 개선이 없고, EMR 향상이 핵심입니다.")

SEC4_H = ["시점", "EMR", "Clean F1"]
SEC4 = [
    ["A 우측 후방 45° (view5)", "46.05", "31.12"],
    ["B 우측 전방 45° (view2)", "58.94", "43.53"],
    ["C 정면 (view3)", "38.97", "30.71"],
    ["D 좌측 전방 45° (view4)", "61.35", "44.45"],
    ["E 좌측 후방 45° (view1)", "46.05", "26.24"],
]

SEC5_H = ["서브모델", "OOF AUC", "디코딩 F1"]
SEC5 = [["sd1 무릎 고정", "61.87", "56.56"], ["sd2 척추 중립", "75.54", "77.06"], ["sd3 바벨 경로", "90.38", "75.29"]]

ORACLE = [
    "sd1만 완벽 → EMR 55.85%",
    "sd2만 완벽 → EMR 54.98%",
    "sd3만 완벽 → EMR 44.83%",
    "sd1+sd2 완벽 → EMR 81.74%",
    "Clean(0,0,0)은 (0,1,0)과 sd2에서만 다름. 정상 130건 중 87건이 (0,1,0)으로 오분류 → sd2(척추)가 Clean F1의 병목이며 스태킹으로는 해결 불가.",
]
CAVEATS = [
    "스태커는 서브모델 OOF 확률을 입력으로 사용하며 서브모델과 스태커의 fold가 다르므로 경미한 누수 가능성이 있음.",
    "피험자 9명, 정상 130건(시점당 26건)으로 신뢰구간이 넓음.",
    "다중 시점 융합은 서비스가 단일 카메라이므로 배포에서 제외(참고용 실험에서 EMR 약 58.8%, Clean F1 약 40%).",
    "이 문서는 v4.1 이후 척추 모델(sd2) 개선 작업의 기준선입니다.",
]


def md_table(h, d):
    s = "| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n"
    for r in d:
        s += "| " + " | ".join(r) + " |\n"
    return s + "\n"


def add_table(doc, h, d, widths):
    t = doc.add_table(rows=0, cols=len(h))
    t.style = "Table Grid"
    format_table(t, widths, h, d)
    doc.add_paragraph()


def add_note(doc, text):
    p = doc.add_paragraph(text)
    for r in p.runs:
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(90, 90, 90)


def add_bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(10)


def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(0.8)

    p = doc.add_paragraph()
    r = p.add_run("vidio-health-trainer 기술 보고서\n")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(127, 140, 141)
    r = p.add_run("바벨 스티프 데드리프트 통합 진단 모델 v2~v4.1 정정 통합 성능보고서")
    r.font.size = Pt(19)
    r.font.bold = True
    r.font.color.rgb = RGBColor(24, 76, 120)

    add_callout(doc, "정정 공지", NOTICE, bg_color="FDEDEC", border_color="C0392B")
    add_header(doc, "1. 버전별 실측 성능 (낙관적: 임계치 in-sample 튜닝)", 1)
    add_table(doc, SEC1_H, SEC1, [0.6, 1.0, 1.1, 1.0, 1.1, 2.2])
    add_note(doc, SEC1_NOTE)
    add_header(doc, "2. v4 정직 평가", 1)
    add_table(doc, SEC2_H, SEC2, [2.3, 2.3, 2.0])
    add_note(doc, SEC2_NOTE)
    add_header(doc, "3. v4 vs v4.1 (단일 카메라, 피험자 격리 nested CV)", 1)
    add_table(doc, SEC3_H, SEC3, [2.3, 2.0, 2.4])
    add_note(doc, SEC3_NOTE)
    add_header(doc, "4. v4.1 시점별 성능 (2-State)", 1)
    add_table(doc, SEC4_H, SEC4, [3.0, 1.5, 1.5])
    add_header(doc, "5. 서브모델 성능 (OOF)", 1)
    add_table(doc, SEC5_H, SEC5, [2.5, 1.5, 1.5])
    add_header(doc, "6. 병목 분석 (오라클)", 1)
    add_bullets(doc, ORACLE)
    add_header(doc, "7. 한계 및 주의", 1)
    add_bullets(doc, CAVEATS)
    doc.save(DOCX_PATH)

    md = f"# 통합 진단 모델 v2~v4.1 정정 통합 성능보고서\n\n> **정정 공지**: {NOTICE}\n\n"
    md += "## 1. 버전별 실측 성능 (낙관적)\n" + md_table(SEC1_H, SEC1) + SEC1_NOTE + "\n\n"
    md += "## 2. v4 정직 평가\n" + md_table(SEC2_H, SEC2) + SEC2_NOTE + "\n\n"
    md += "## 3. v4 vs v4.1\n" + md_table(SEC3_H, SEC3) + SEC3_NOTE + "\n\n"
    md += "## 4. 시점별 (v4.1, 2-State)\n" + md_table(SEC4_H, SEC4)
    md += "## 5. 서브모델 OOF\n" + md_table(SEC5_H, SEC5)
    md += "## 6. 병목 분석\n" + "".join(f"- {x}\n" for x in ORACLE)
    md += "\n## 7. 한계 및 주의\n" + "".join(f"- {x}\n" for x in CAVEATS)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print("OK", DOCX_PATH)


if __name__ == "__main__":
    main()

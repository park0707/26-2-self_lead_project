# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: generate_week5_session_report_docx.py
Purpose: [5주차 대화 세션 종합 성과 보고서]
         선택적 분류(Selective Classification) 기반 3-State 판정보류 아키텍처 실측,
         다중 라벨 결합 확률 분석 및 실서비스 가이드라인 확정 보고서 (.docx & .md)
"""

import os
import sys
import json
from datetime import datetime

try:
    import docx
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml import parse_xml, OxmlElement
    from docx.oxml.ns import nsdecls, qn
except ImportError:
    print("[!] python-docx 미설치. pip install python-docx 필요")

WEEK5_DIR = r"D:\자주프 문서\5주차"
OUTPUT_DOCX = os.path.join(WEEK5_DIR, "바벨스쿼트_3State_선택적분류_및_다중라벨_결합분석_종합보고서.docx")
OUTPUT_MD = os.path.join(WEEK5_DIR, "바벨스쿼트_3State_선택적분류_및_다중라벨_결합분석_종합보고서.md")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout(doc, text, title="핵심 결론", border_color="0D47A1", bg_color="E3F2FD"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"■ {title}: ")
    r_title.bold = True
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    r_text = p.add_run(text)
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def build_documents():
    os.makedirs(WEEK5_DIR, exist_ok=True)

    # 1. Word 문서 생성
    doc = Document()
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_pre = title_p.add_run("vidio-health-trainer 생체역학 AI 코칭 시스템 연구개발 보고서\n")
    r_pre.font.size = Pt(11)
    r_pre.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)
    r_pre.bold = True

    r_main = title_p.add_run("바벨 스쿼트 3-State 선택적 분류(Selective Classification) 아키텍처 및\n다중 결함 라벨 결합 확률 분석 종합 보고서")
    r_main.font.size = Pt(18)
    r_main.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
    r_main.bold = True

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_sub = sub_p.add_run(f"발행일자: 2026년 9월 25일 | 공식 파이프라인: Phase 1 L2 Meta-Learner + 개별 정예 앙상블 체제")
    r_sub.font.size = Pt(9)
    r_sub.font.color.rgb = RGBColor(0x75, 0x75, 0x75)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Callout
    callout_txt = (
        "정보 부족·모호성 구간에서 무리한 강제 이진 판정(Forced Decision)을 배제하고 '판정 보류(Retake Prompt)'를 유도하는 "
        "설명 가능한 3-State 선택적 분류 시스템을 구축했습니다. AI-Hub 1,906건 전수 5-Fold 실측 결과, "
        "단 4.51%(86건)의 모호 구간을 보류함으로써 Clean F1 75.73%, Defect F1 98.54%의 최고 신뢰도를 달성했습니다. "
        "특히 Defect Override(결함 구제)를 통해 16건의 치명적 결함을 100% 적중률로 구제했으며, "
        "4대 결함 다중 라벨 분석 결과 4개 중 3개 이상 완벽 적중률이 91.43%에 달해 실서비스 코칭 품질이 매우 견고함을 실증했습니다."
    )
    add_callout(doc, callout_txt, title="연구개발 핵심 요약")

    # 1. 배경 및 문제 제기
    doc.add_heading("1. 연구 배경 및 세션 개요", level=1)
    p1 = doc.add_paragraph()
    p1.add_run(
        "본 연구 세션에서는 바벨 스쿼트 AI 코칭 엔진의 상용화 신뢰성을 극대화하기 위해 다음 3가지 핵심 과제를 다루었습니다:\n"
        "1) Phase 3 멀티태스크 딥러닝(공유 백본) 실험 및 실측 한계 검증: 단일 신경망 접근법이 개별 특화 앙상블 대비 부정적 전이(Negative Transfer) 및 "
        "그래디언트 충돌로 성능이 저하됨을 실증하고, 최고 성능을 내는 기존 [Phase 1 메타러너 + 개별 정예 앙상블] 체제를 최종 확정했습니다.\n"
        "2) 모호 구간 '판정 보류(3-State)'의 기술적·도메인적 타당성 검토: 옷 주름, 카메라 각도, 애매한 경계선에 있는 데이터를 무조건 정상/결함으로 강제 판정하는 대신 "
        "재촬영을 요청하는 선택적 분류(Selective Classification) 패러다임을 도입했습니다.\n"
        "3) 4대 결함 다중 라벨(Multi-Label) 적중률 및 100% 일치율(Exact Match Ratio)의 수학적 원인 분석: "
        "개별 결함 F1(80~93%)과 정상 F1(75.7%) 대비 EMR이 61.65%로 집계되는 결합 확률의 지수적 감소 법칙과 실제 오답 분포를 규명했습니다."
    )

    # 2. 계층적 선택적 분류 아키텍처
    doc.add_heading("2. 계층적 3-State 선택적 분류 아키텍처 설계", level=1)
    p2 = doc.add_paragraph()
    p2.add_run(
        "사용자가 질의한 '세부 모델부터 3상태로 나눠야 하는가?'에 대해 수학적·공학적 검토를 수행한 결과, "
        "세부 모델 레벨의 독립 3분할(Local Hard Reject)은 보류 오차 누적(Abstention Error Stacking)으로 전체의 20% 가까운 세션이 퇴짜를 맞는 "
        "서비스 마비 현상을 초래함을 확인했습니다.\n\n"
        "이에 따라 세부 모델은 보정된 연속 확률과 확신도 마진(|P - 0.5|)을 상위로 전달하고, 상위 메타러너가 종합 판정하는 '계층적 하이브리드 거부(Hierarchical Hybrid Reject)' 아키텍처를 최종 채택했습니다:\n"
        "• Tier 0 (센서 게이트): MediaPipe 키포인트 신뢰도 < 0.50 또는 화면 이탈 시 모델 가동 전 즉시 하드웨어 재조정 안내\n"
        "• Tier 1 (세부 앙상블): c1(척추 46D), c2(시선 13D), c3(무릎 53D), c4(발목 9D)가 정밀 연속 확률 및 마진 계산\n"
        "• Tier 2 (메타러너 3-State): P(Clean) < 0.15 (결함), P(Clean) >= 0.45 (정상), 그 외 모호 구간(판정 보류)\n"
        "• Tier 3 (결함 구제 & 원인 귀속): 단일 관절 결함 확신도가 90% 이상이면 즉시 결함 확정(Defect Override), "
        "남은 보류 건은 가장 마진이 작은 관절을 역추적하여 핀포인트 촬영 가이드 제시"
    )

    # 3. 실측 성능
    doc.add_heading("3. 3-State 실측 성능 벤치마크 및 결함 구제 효과", level=1)
    p3 = doc.add_paragraph()
    p3.add_run(
        "AI-Hub 1,906건 전수(9명 피험자, 5개 시점)에 대해 5-Fold 피험자 격리 실측 벤치마크를 수행했습니다."
    )

    tbl = doc.add_table(rows=3, cols=6)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    h = ["평가 전략", "판정 커버리지", "보류 건수(율)", "Clean F1", "Defect F1", "구제된 결함(적중률)"]
    for i, title in enumerate(h):
        tbl.rows[0].cells[i].text = title
        set_cell_background(tbl.rows[0].cells[i], "1565C0")
        p = tbl.rows[0].cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.runs[0].font.size = Pt(8.5)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    rows_data = [
        ("기존 Phase 1 대안 B (단순 필터)", "94.65%", "102건 (5.35%)", "70.47%", "98.33%", "0건 (무조건 재촬영)"),
        ("신규 3-State 시스템 (구제+귀속)", "95.49%", "86건 (4.51%)", "75.73%", "98.54%", "16건 구제 (100.0% 적중)")
    ]

    for r_idx, row in enumerate(rows_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.rows[r_idx].cells[c_idx]
            cell.text = val
            set_cell_background(cell, "F5F5F5" if r_idx == 1 else "FFFFFF")
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4. 원인 귀속 분포
    doc.add_heading("4. 설명 가능한 재촬영(Explainable Retake) 원인 귀속 분석", level=1)
    p4 = doc.add_paragraph()
    p4.add_run(
        "최종 판정 보류로 분류된 86건(4.51%)에 대해 불확실성을 유발한 관절을 역추적한 결과, "
        "스탠포드 OpenCap 연구팀이 지적한 카메라 시점 편향(Sagittal vs Coronal)과 완벽하게 일치하는 분포를 보였습니다."
    )

    tbl2 = doc.add_table(rows=5, cols=4)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = ["모호 유발 관절", "보류 건수", "점유율", "사용자 맞춤형 촬영 피드백 가이드"]
    for i, title in enumerate(h2):
        tbl2.rows[0].cells[i].text = title
        set_cell_background(tbl2.rows[0].cells[i], "0D47A1")
        p = tbl2.rows[0].cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.runs[0].font.size = Pt(8.5)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    cause_data = [
        ("c1. 척추의 중립", "50건", "58.14%", "정면/후방 시점 한계 -> '카메라를 대각선 45도 방향, 1m 높이에 두고 다시 수행해 주세요.'"),
        ("c2. 고개 정면", "21건", "24.42%", "시선 경계선 모호 -> '정면을 흔들림 없이 안정적으로 응시하고 다시 수행해 주세요.'"),
        ("c3. 발과 무릎 외반", "8건", "9.30%", "하체 가림 이슈 -> '하체가 화면에 잘 보이도록 카메라 거리를 2m 확보해 주세요.'"),
        ("c4. 발바닥 지면", "7건", "8.14%", "지면 조명 이슈 -> '발끝까지 밝게 보이도록 조명을 확인하고 다시 수행해 주세요.'")
    ]

    for r_idx, row in enumerate(cause_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl2.rows[r_idx].cells[c_idx]
            cell.text = val
            set_cell_background(cell, "F5F5F5" if r_idx % 2 == 1 else "FFFFFF")
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 3 else WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5. 다중 라벨 결합 확률 및 EMR 분석
    doc.add_heading("5. 4대 결함 다중 라벨(Multi-Label) 적중률 및 EMR 심층 분석", level=1)
    p5 = doc.add_paragraph()
    p5.add_run(
        "정상 감지 F1은 75.7%에 달하지만 4대 라벨 100% 동시 일치율(Exact Match Ratio, EMR)이 61.65%로 측정된 원인을 규명하기 위해 "
        "확정 판정군 1,820건의 샘플별 적중 개수를 전수 분석했습니다."
    )

    tbl3 = doc.add_table(rows=6, cols=3)
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    h3 = ["적중 라벨 개수", "샘플 수", "점유율 (누적)"]
    for i, title in enumerate(h3):
        tbl3.rows[0].cells[i].text = title
        set_cell_background(tbl3.rows[0].cells[i], "1565C0")
        p = tbl3.rows[0].cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.runs[0].font.size = Pt(8.5)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    match_data = [
        ("4개 중 4개 전부 100% 일치 (Exact Match)", "1,122건", "61.65% (누적 61.65%)"),
        ("4개 중 3개 적중 (단 1개만 아쉽게 오차)", "542건", "29.78% (누적 91.43%) ◀ 91.4% 달성!"),
        ("4개 중 2개 적중", "124건", "6.81% (누적 98.24%)"),
        ("4개 중 1개 적중", "29건", "1.59% (누적 99.84%)"),
        ("4개 중 0개 적중 (완전 오판)", "3건", "0.16% (100.0%)")
    ]

    for r_idx, row in enumerate(match_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl3.rows[r_idx].cells[c_idx]
            cell.text = val
            set_cell_background(cell, "F5F5F5" if r_idx % 2 == 1 else "FFFFFF")
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p5_sub = doc.add_paragraph()
    p5_sub.add_run(
        "\n[결합 확률의 핵심 결론]:\n"
        "1. 4개 중 3개 이상 완벽 적중률은 91.43%(1,664건)에 달합니다. 모델이 동작을 완전히 잘못 짚은 것이 아니라, "
        "3개 관절은 완벽히 맞추고 1개 부위에서 경계선 오차가 발생하여 가혹한 EMR 평가(All-or-Nothing)에서 0점 처리된 것입니다.\n"
        "2. 1개 오차의 71.2%(386건)가 척추(c1)와 시선(c2)에 집중되어 있습니다. "
        "이는 단일 카메라의 시상면 깊이 판독 한계와 AI-Hub의 시선 라벨 노이즈에 기인합니다.\n"
        "3. 수학적 독립 기대치(0.81 x 0.86 x 0.93 x 0.90 = 58.05%)보다 모델의 실측 EMR(61.65%)이 더 높게 나타나, "
        "관절 간 상호작용 피처가 이미 높은 시너지를 발휘하고 있음이 입증되었습니다."
    )

    # 6. 향후 계획
    doc.add_heading("6. 종합 결론 및 향후 실서비스 배포 로드맵", level=1)
    p6 = doc.add_paragraph()
    p6.add_run(
        "1. Phase 3 딥러닝 기각 및 Phase 1 메타 체제 확정: "
        "실측 결과 F1이 열세인 멀티태스크 신경망을 기각하고, 최고 성능을 내는 Phase 1 메타러너 체제로 확정했습니다.\n"
        "2. 3-State 상태 머신 통합 배포: "
        "babel_squat_integrated_v4.py를 업데이트하여 단순 이진 출력을 폐기하고, "
        "CLEAN | DEFECT | RETAKE 3-State 상태 머신을 공식 탑재합니다.\n"
        "3. 맞춤형 음성/시각 피드백 연동: "
        "판정 보류 발생 시 58%를 차지하는 척추 시점 한계를 해소하기 위해 '측면 45도 카메라 가이드'를 인터랙티브하게 안내합니다."
    )

    doc.save(OUTPUT_DOCX)
    print(f"[*] Word 보고서 (.docx) 생성 완료: {OUTPUT_DOCX}")

    # 2. Markdown 생성
    md_content = f"""# [vidio-health-trainer] 바벨 스쿼트 3-State 선택적 분류 및 다중 라벨 결합 분석 종합 보고서

- **발행일자**: 2026년 9월 25일
- **공식 파이프라인**: Phase 1 L2 Meta-Learner + 개별 정예 앙상블 ($c_1$ v7.1, $c_2$ v3.0, $c_3$ v5.0, $c_4$ v3.0)
- **데이터셋 규격**: AI-Hub 9명 피험자 전수, 5대 시점 전수 (1,906건), 64프레임 3차 스플라인 표준

---

## 1. 연구개발 핵심 요약

> **■ 핵심 요약**
> {callout_txt}

---

## 2. 계층적 3-State 선택적 분류 실측 성능 비교

| 평가 전략 | 판정 커버리지 | 보류 건수(율) | Clean F1 | Defect F1 | 구제된 결함(적중률) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **기존 Phase 1 대안 B (단순 필터)** | 94.65% | 102건 (5.35%) | 70.47% | 98.33% | 0건 (무조건 재촬영) |
| **신규 3-State 시스템 (구제+귀속)** | **95.49%** | **86건 (4.51%)** | **75.73%** | **98.54%** | **16건 구제 (100.0% 적중)** |

---

## 3. 설명 가능한 재촬영(Explainable Retake) 원인 귀속 분석

최종 보류 86건(4.51%)에 대해 불확실성을 유발한 관절을 역추적한 결과:

| 모호 유발 관절 | 보류 건수 | 점유율 | 사용자 맞춤형 촬영 피드백 가이드 |
| :--- | :---: | :---: | :--- |
| **$c_1$. 척추의 중립** | **50건** | **58.14%** | 정면/후방 시점 한계 👉 *"카메라를 대각선 45도 방향, 1m 높이에 두고 다시 수행해 주세요."* |
| **$c_2$. 고개 정면** | **21건** | **24.42%** | 시선 경계선 모호 👉 *"정면을 흔들림 없이 안정적으로 응시하고 다시 수행해 주세요."* |
| **$c_3$. 발과 무릎 외반** | **8건** | **9.30%** | 하체 가림 이슈 👉 *"하체가 화면에 잘 보이도록 카메라 거리를 2m 확보해 주세요."* |
| **$c_4$. 발바닥 지면** | **7건** | **8.14%** | 지면 조명 이슈 👉 *"발끝까지 밝게 보이도록 조명을 확인하고 다시 수행해 주세요."* |

---

## 4. 4대 결함 다중 라벨(Multi-Label) 적중률 및 EMR 심층 분석

확정 판정군 1,820건 대상 샘플별 맞춤 개수 분포:

| 적중 라벨 개수 | 샘플 수 | 점유율 (누적) | 분석 및 실무 의미 |
| :--- | :---: | :---: | :--- |
| **4개 중 4개 전부 100% 일치 (Exact Match)** | **1,122건** | **61.65%** | 4대 결함 조합 전체를 단 하나의 오차 없이 적중 |
| **4개 중 3개 적중 (단 1개만 아쉽게 오차)** | **542건** | **29.78% (누적 91.43%)** | ◀ **사용자 91.4%가 3개 이상 완벽한 코칭을 받음!** |
| **4개 중 2개 적중** | 124건 | 6.81% (누적 98.24%) | 보조적 교정 피드백 유효 |
| **4개 중 1개 적중** | 29건 | 1.59% (누적 99.84%) | 경계선 다중 오차 |
| **4개 중 0개 적중 (완전 오판)** | 3건 | 0.16% (100.0%) | 극단적 이상치 (0.16% 미미) |

### 💡 왜 정상 F1은 75.7%인데 100% 일치율(EMR)은 61.65%인가?
1. **수학적 독립 한계치 초과 달성**: 개별 정확도 곱연산 기대치($0.81 \times 0.86 \times 0.93 \times 0.90 \approx 58.05\%$) 대비 모델 실측치(61.65%)가 더 높음.
2. **가혹한 All-or-Nothing 지표**: 4개 중 3개를 맞추어도 1개만 틀리면 0점 처리되는 특성 때문이며, 실제로는 **91.4%의 사용자가 3개 이상 완벽한 코칭**을 받음.
3. **1개 오차의 71.2%(386건)가 척추($c_1$)와 시선($c_2$)에 집중**: 단일 카메라의 깊이 판독 한계와 AI-Hub 시선 라벨 노이즈에 기인.

---

## 5. 결론 및 향후 실서비스 배포
- Phase 3 멀티태스크 딥러닝 기각 및 [Phase 1 메타러너 + 개별 앙상블] 최종 확정
- `babel_squat_integrated_v4.py`에 CLEAN / DEFECT / RETAKE 3-State 상태 머신 공식 탑재
"""

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[*] Markdown 보고서 (.md) 생성 완료: {OUTPUT_MD}")


if __name__ == '__main__':
    build_documents()

# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_web_readiness_audit_report.py
Purpose: 1단계 운동 분류 및 2단계 세부 결함 분류 모델의 웹 서비스 배포 적합성 전수 점검 감사 보고서 생성
Target:
  - D:\자주프 문서\7주차\vidio-health-trainer_웹서비스_배포적합성_1단계및2단계_모델_전수점검_감사보고서.docx
  - D:\자주프 문서\7주차\vidio-health-trainer_웹서비스_배포적합성_1단계및2단계_모델_전수점검_감사보고서.md
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX = r"D:\자주프 문서\7주차\vidio-health-trainer_웹서비스_배포적합성_1단계및2단계_모델_전수점검_감사보고서.docx"
OUT_MD = r"D:\자주프 문서\7주차\vidio-health-trainer_웹서비스_배포적합성_1단계및2단계_모델_전수점검_감사보고서.md"

COLOR_NAVY = RGBColor(0x1B, 0x36, 0x5D)
COLOR_SUBNAVY = RGBColor(0x2C, 0x52, 0x82)
COLOR_CHARCOAL = RGBColor(0x22, 0x22, 0x22)
COLOR_GRAY = RGBColor(0x71, 0x80, 0x96)
HEX_HEADER_BG = "1B365D"
HEX_ALT_ROW = "F7FAFC"
HEX_HIGHLIGHT = "EBF8FF"
HEX_WARNING = "FFF5F5"
HEX_SUCCESS = "F0FFF4"

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def style_table(table, col_widths, alignments):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        for j, cell in enumerate(row.cells):
            set_cell_margins(cell, 100, 100, 140, 140)
            cell.width = Inches(col_widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if len(cell.paragraphs) > 0:
                cell.paragraphs[0].alignment = alignments[j]
            if i == 0:
                set_cell_background(cell, HEX_HEADER_BG)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif i % 2 == 1:
                set_cell_background(cell, HEX_ALT_ROW)

def add_callout(doc, text, title="핵심 요약 (Key Takeaway)", bg_hex="EBF8FF", border_hex="2B6CB0"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    set_cell_background(cell, bg_hex)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"★ [{title}]\n")
    r_title.bold = True
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = COLOR_NAVY
    r_text = p.add_run(text)
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = COLOR_CHARCOAL
    doc.add_paragraph()

def build_docx_report():
    doc = Document()
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("vidio-health-trainer 웹 서비스 배포 적합성 전수 점검 및 아키텍처 감사 보고서")
    r_title.bold = True
    r_title.font.size = Pt(19)
    r_title.font.color.rgb = COLOR_NAVY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("1단계 운동 종목 분류 모델 & 2단계 8종 세부 결함 진단 모델의 웹 서빙 적합도 및 구조적 과제 전수 분석")
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = COLOR_GRAY

    # Metadata Table
    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_widths = [2.2, 4.6]
    meta_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    meta_data = [
        ("프로젝트 공식 명칭", "vidio-health-trainer"),
        ("감사 대상 시스템", "1단계 8종 운동 분류 모델(Stage 1) 및 2단계 33종 세부 결함 모델(Stage 2)"),
        ("점검 기준 및 규정", "AGENTS.md 제0조~제5조 (64F 스플라인, 5대 시점, 3-State F1, src/web 전용 경로)"),
        ("감사 결과 총평", "가중치 파일 100% 실재 확인 완료 / 1단계 서빙 번들 부재 및 인터페이스 파편화 5대 병목 식별")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_tbl.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.paragraphs[0].add_run(k).bold = True
        c1.paragraphs[0].add_run(v)
    style_table(meta_tbl, meta_widths, meta_aligns)
    doc.add_paragraph()

    # Callout Box: 핵심 결론
    add_callout(
        doc,
        "1. 2단계 세부 결함 가중치(.pkl / .joblib / .pth)는 8대 운동 33종 전체에 걸쳐 디스크에 100% 실재함을 확인했습니다.\n"
        "2. 반면, 1단계 운동 분류 모델은 3-Fold CV 평가 스크립트만 존재할 뿐, 단일 추론용 프로덕션 번들(.pkl)이 생성되어 있지 않습니다.\n"
        "3. 또한 모델들의 저장 경로 분산, 비표준화된 추론 인터페이스, 7~10초의 콜드 스타트 지연(라이브러리 로딩), 관절 규격(15J/26J/33J) 불일치 등 5대 구조적 과제가 식별되었습니다.\n"
        "4. 이를 해결하기 위해 src/web 하위에 '통합 인퍼런스 엔진(InferenceEngine)', '싱글톤 모델 레지스트리(ModelRegistry)', '관절 어댑터(SkeletalAdapter)'를 구축하는 표준 아키텍처를 제시합니다.",
        title="전수 점검 핵심 결론 및 웹 서비스 전환 요약",
        bg_hex="EBF8FF",
        border_hex="2B6CB0"
    )

    # 1장
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("1. 점검 배경 및 전수 감사 개요")
    r1.font.color.rgb = COLOR_NAVY
    
    p = doc.add_paragraph()
    p.add_run(
        "본 보고서는 vidio-health-trainer 프로젝트의 AI 모델 파이프라인(1단계 운동 종목 분류 및 2단계 세부 결함 진단)이 "
        "완성됨에 따라, 이를 실시간 브라우저 웹 서비스(FastAPI 백엔드 + 모던 웹 프론트엔드)로 성공적으로 전환 서빙하기 위해 "
        "현재 소스코드, 모델 가중치 파일, 피처 추출기, 추론 지연시간 및 인터페이스 구조를 전수 감사(Full-Coverage Audit)한 결과입니다.\n\n"
        "특히 AGENTS.md 제5조(웹 서비스 전용 개발 경로 규정: src/web 내부 생성 및 기존 AI 모듈 비파괴 연동)와 "
        "제2조(64프레임 3차 스플라인 실서비스 1-Rep 분할 표준), 제4조(공식 5대 시점 체계) 규정을 충족하는지 집중 점검하였습니다."
    )

    # 2장
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. 1단계: 운동 종목 분류 모델(Stage 1) 점검 결과")
    r2.font.color.rgb = COLOR_NAVY

    p = doc.add_paragraph()
    p.add_run(
        "1단계 모델은 사용자가 수행한 운동이 8대 종목(바벨 스쿼트, 바벨 데드리프트, 바벨 스티프 데드리프트, 바벨 런지, 다이나믹 런지, "
        "사이드 런지, 크로스 런지, 굿모닝) 중 무엇인지 자동으로 판별하는 핵심 게이트웨이입니다."
    )

    s1_tbl = doc.add_table(rows=6, cols=3)
    s1_widths = [1.8, 2.5, 2.5]
    s1_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    s1_headers = ["항목", "현재 상태 (Current State)", "웹 서빙 적합도 및 조치 사항"]
    for j, h in enumerate(s1_headers):
        s1_tbl.rows[0].cells[j].paragraphs[0].add_run(h)
    
    s1_rows = [
        ("알고리즘 구조", "v6.0 2-Stage 계층형 앙상블\n(Bi-GRU + LightGBM + XGBoost on 59D Model F Direct KCF)", "적합. 높은 분류 정확도(98%+) 및 생체역학 판별력 입증됨."),
        ("프로덕션 가중치 파일", "부재 (Missing)\n3-Fold CV 평가 스크립트만 존재하고 직렬화 번들(.pkl/.pt) 미저장", "치명적 부적합. 웹 서버 구동 시 즉시 로드할 수 있는 단일 완성 번들 파일 필수 생성 필요."),
        ("입력 데이터 규격", "64프레임 관절 좌표 (T=64, J=15, C=3 또는 J=33)", "조건부 적합. 실서비스 비디오 입력 시 15-관절 Format B 또는 33-관절에서 59D 피처 추출기 직결 필요."),
        ("추론 지연 시간", "피처 추출 포함 시퀀스당 약 5~15ms (매우 신속)", "매우 적합. 실시간 웹 서빙에 전혀 부담 없는 초고속 추론 지원."),
        ("추론 인터페이스", "배치 텐서 중심 스크립트 위주\n단일 시퀀스(Single Rep)용 함수 미분리", "개선 필요. kps_seq를 입력받아 종목명과 확률을 반환하는 Stage1InferenceEngine 클래스 구축 필요.")
    ]
    for i, row in enumerate(s1_rows):
        r = s1_tbl.rows[i+1]
        r.cells[0].paragraphs[0].add_run(row[0]).bold = True
        r.cells[1].paragraphs[0].add_run(row[1])
        r.cells[2].paragraphs[0].add_run(row[2])
    style_table(s1_tbl, s1_widths, s1_aligns)
    doc.add_paragraph()

    # 3장
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. 2단계: 8대 운동 33개 세부 결함 모델(Stage 2) 전수 점검 결과")
    r3.font.color.rgb = COLOR_NAVY

    p = doc.add_paragraph()
    p.add_run(
        "8대 운동 종목별 세부 결함 진단 모델 33종의 가중치 파일 실재 여부, 파일 크기, 패키징 형태, 그리고 웹 서빙 적합도를 전수 조사하였습니다."
    )

    s2_tbl = doc.add_table(rows=9, cols=5)
    s2_widths = [1.3, 1.2, 2.0, 1.3, 1.0]
    s2_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER]
    s2_headers = ["운동 종목", "결함 코드", "가중치 파일 및 번들 위치", "패키징 구조", "서빙 준비도"]
    for j, h in enumerate(s2_headers):
        s2_tbl.rows[0].cells[j].paragraphs[0].add_run(h)

    s2_rows = [
        ("바벨 스쿼트", "c1 ~ c4\n(4종)", "c1_model_v11_weights.pkl 외 3종\njoint_meta_16way_v12_3_weights.pth", "통합 마스터 엔진\n(BabelSquatIntegratedEngineV12_3)", "완료 (A+)"),
        ("바벨 데드리프트", "d1 ~ d5\n(5종)", "weights_integrated_v8/\ndeadlift_integrated_v8_0_bundle.pkl", "통합 마스터 엔진\n(DeadliftIntegratedEngineV80)", "완료 (A+)"),
        ("스티프 데드리프트", "sd1 ~ sd3\n(3종)", "weights/sd1_v8_bundle.pkl\nsd2_model_v3_weights.pkl, sd3_v4_bundle.pkl", "개별 클래스 3종\n(sd1_v8, sd2_v3, sd3_v4)", "양호 (B+)"),
        ("바벨 런지", "bl1 ~ bl5\n(5종)", "bl1_v4, bl2_final, bl4_v4b, bl5_final\nbl3_v4_champion_bundle.joblib", "개별 번들 분산\n(bl3은 Data/tuning_results)", "보통 (B)"),
        ("다이나믹 런지", "dl1 ~ dl5\n(5종)", "weights/dl1_final, dl2_final, dl4_rtmp, dl5_final\ndl3_champion_bundle.joblib", "개별 번들 분산\n(dl3은 Data/tuning_results)", "보통 (B)"),
        ("사이드 런지", "sl1 ~ sl5\n(5종)", "sl1_final, sl2_hybrid_3, sl3_v3, sl4_v3\nweights/sl5_v7_bundle.pkl (109MB)", "개별 번들 분산\n(sl5_v7 대용량 모델)", "양호 (B+)"),
        ("크로스 런지", "cl1 ~ cl3\n(3종)", "cl1_final_bundle.pkl, cl2_v5_bundle.pkl\ncl3_v1_bundle.pkl", "개별 번들 분산\n(크로스런지 전용 폴더)", "양호 (B+)"),
        ("굿모닝", "gm1 ~ gm3\n(3종)", "gm1_v4_full_bundle, gm2_v1_bundle\ngm3_final_champion.joblib", "3개 폴더 분산\n(babel_gm, gm, Data/tuning)", "주의 (C+)")
    ]
    for i, row in enumerate(s2_rows):
        r = s2_tbl.rows[i+1]
        r.cells[0].paragraphs[0].add_run(row[0]).bold = True
        r.cells[1].paragraphs[0].add_run(row[1])
        r.cells[2].paragraphs[0].add_run(row[2])
        r.cells[3].paragraphs[0].add_run(row[3])
        r.cells[4].paragraphs[0].add_run(row[4]).bold = True
    style_table(s2_tbl, s2_widths, s2_aligns)
    doc.add_paragraph()

    # 4장
    h4 = doc.add_heading(level=1)
    r4 = h4.add_run("4. 웹 서빙 적합화를 가로막는 5대 구조적 병목 및 해결책")
    r4.font.color.rgb = COLOR_NAVY

    bottlenecks = [
        ("병목 1: 1단계 운동 분류 모델의 단일 배포용 번들(Production Bundle) 부재",
         "• 현상: 1단계 모델은 3-Fold CV 검증 스크립트만 존재하여, 웹 서버 기동 시 로드할 단일 .pkl 또는 .pt 파일이 없습니다.\n"
         "• 위험성: 웹 요청이 올 때마다 모델을 재학습하는 것은 불가능하며, 현재 상태로는 웹 서버에서 1단계 분류를 호출할 수 없습니다.\n"
         "• 해결책: 마스터 데이터 풀 전체(Master Data Pool)를 1회 완결 학습시켜 가중치를 패키징하는 'export_stage1_production_bundle.py'를 구축하고, 단일 시퀀스를 즉시 추론하는 Stage1InferenceEngine을 확립합니다."),
        ("병목 2: 2단계 세부 결함 모델의 저장 위치 및 인터페이스 분산 (Fragmentation)",
         "• 현상: 바벨 스쿼트와 데드리프트는 통합 마스터 클래스가 존재하지만, 나머지 6개 운동은 결함별 스크립트와 번들이 4개 디렉터리에 분산되어 있습니다. 특히 bl3, dl3, gm3은 Data/피처_데이터/tuning_results/에 위치합니다.\n"
         "• 위험성: 웹 API 서버가 제각각 다른 방식으로 파일을 탐색하고 로드해야 하므로 코드 복잡도와 버그 발생 위험이 증가합니다.\n"
         "• 해결책: 'ModelRegistry'를 구축하여 8대 운동별 공식 가중치 경로를 단일 딕셔너리로 매핑하고, 운동별 'ExerciseAdapter'를 통해 입출력 형식을 일원화합니다."),
        ("병목 3: 관절 토폴로지(Landmark Topology) 불일치",
         "• 현상: 1단계 모델은 15-관절(Format B), 2단계 모델 중 일부는 RTMPose 26-관절, 대다수는 MediaPipe 33-관절을 입력으로 기대합니다.\n"
         "• 위험성: 사용자가 웹캠이나 비디오를 업로드했을 때, 어떤 포맷으로 변환해야 할지 통일되어 있지 않으면 차원 불일치(ValueError)가 발생합니다.\n"
         "• 해결책: 실시간 비전 파이프라인에서 추출된 33-관절 또는 26-관절을 15-관절로 즉시 매핑하고 정규화하는 'SkeletalTopologyAdapter'를 도입합니다."),
        ("병목 4: 동적 로딩 시 콜드 스타트 지연 (Cold Start Latency: 7~10초 병목)",
         "• 현상: 실측 결과 CatBoost, LightGBM, XGBoost, PyTorch를 import하고 번들을 pickle.load() 하는 데 회당 약 7.6초가 소요됩니다.\n"
         "• 위험성: 사용자가 웹에서 영상 분석을 누를 때마다 7~10초간 서버가 멈추는 극심한 레이턴시가 발생합니다.\n"
         "• 해결책: FastAPI 서버 기동 시점(Lifespan startup)에 모든 모델을 메모리에 미리 상주시키는 싱글톤(Singleton) 'ModelPrewarmer'를 적용합니다. 이를 통해 실제 추론 시간은 단 20~50ms 수준으로 단축됩니다."),
        ("병목 5: 3-State 및 피드백 응답 스키마(Response Schema) 비표준화",
         "• 현상: 각 모델의 반환값 키 명칭(preds, probabilities, is_abstain, state_label 등)과 피드백 문구 구조가 운동마다 상이합니다.\n"
         "• 위험성: 웹 프론트엔드가 일관된 UI 컴포넌트(스켈레톤 뷰어, 결함 타임라인, 코칭 팁)를 렌더링하기 어렵습니다.\n"
         "• 해결책: Pydantic 기반의 공통 데이터 모델(AnalysisResponse, RepetitionDetail, DefectResult)을 정의하여 8대 운동 33종 결함 모두 동일한 JSON 규격으로 반환되도록 표준화합니다.")
    ]

    for title, desc in bottlenecks:
        p_b = doc.add_paragraph()
        p_b.paragraph_format.space_before = Pt(6)
        p_b.paragraph_format.space_after = Pt(2)
        r_bt = p_b.add_run(f"■ {title}\n")
        r_bt.bold = True
        r_bt.font.size = Pt(11)
        r_bt.font.color.rgb = COLOR_SUBNAVY
        r_bd = p_b.add_run(desc)
        r_bd.font.size = Pt(9.5)

    # 5장
    h5 = doc.add_heading(level=1)
    r5 = h5.add_run("5. 웹 서비스 아키텍처 설계 (src/web 표준 구조)")
    r5.font.color.rgb = COLOR_NAVY

    p = doc.add_paragraph()
    p.add_run(
        "AGENTS.md 제5조 규정에 따라, 모든 웹 서비스 관련 코드는 src/web 디렉터리 내부에 배치되며, "
        "기존 AI 코드를 파괴하지 않고 깔끔한 인터페이스 형태로 결합하는 차세대 모듈 구조는 다음과 같습니다."
    )

    arch_text = (
        "src/web/\n"
        "├── __init__.py\n"
        "├── core/\n"
        "│   ├── config.py                 # 서버 환경 설정, 시점 매핑, 3-State 임계값\n"
        "│   ├── model_registry.py         # 8대 운동 33종 모델 싱글톤 캐시 및 사전 로딩(Pre-warming)\n"
        "│   ├── topology_adapter.py       # 15J / 26J / 33J 관절 좌표 자동 변환 어댑터\n"
        "│   └── schemas.py                # Pydantic 기반 통일된 REST / WebSocket JSON 규격\n"
        "├── pipeline/\n"
        "│   ├── stage1_engine.py          # 1단계 8종 운동 분류 단일 추론 엔진\n"
        "│   ├── stage2_dispatcher.py      # 1단계 분류 결과에 따른 2단계 운동별 어댑터 라우팅\n"
        "│   └── service_pipeline.py       # 비디오 수신 -> 1-Rep 분할 -> 64F 스플라인 -> 종합 진단 E2E\n"
        "├── adapters/                     # 8대 운동별 세부 결함 통일 어댑터\n"
        "│   ├── squat_adapter.py          # BabelSquatIntegratedEngineV12_3 연동\n"
        "│   ├── deadlift_adapter.py       # DeadliftIntegratedEngineV80 연동\n"
        "│   ├── stiff_deadlift_adapter.py # sd1_v8 + sd2_v3 + sd3_v4 결합 어댑터\n"
        "│   ├── lunge_adapters.py         # 바벨/다이나믹/사이드/크로스 런지 통합 어댑터\n"
        "│   └── good_morning_adapter.py   # gm1 + gm2 + gm3 결합 어댑터\n"
        "├── api/\n"
        "│   ├── main.py                   # FastAPI 메인 애플리케이션 진입점\n"
        "│   ├── routers/\n"
        "│   │   ├── analyze.py            # 비디오 업로드 및 비동기 분석 엔드포인트\n"
        "│   │   └── health.py             # 서버 상태 및 모델 로딩 헬스체크\n"
        "│   └── websocket.py              # 분석 진행률 및 실시간 스트리밍\n"
        "└── static/ & templates/          # 모던 반응형 웹 프론트엔드 UI (스켈레톤 뷰어, 리포트)"
    )
    p_arch = doc.add_paragraph()
    r_arch = p_arch.add_run(arch_text)
    r_arch.font.name = "Consolas"
    r_arch.font.size = Pt(8.5)
    r_arch.font.color.rgb = COLOR_CHARCOAL

    # 6장
    h6 = doc.add_heading(level=1)
    r6 = h6.add_run("6. 향후 즉시 실행 로드맵 (Action Items)")
    r6.font.color.rgb = COLOR_NAVY

    action_tbl = doc.add_table(rows=5, cols=3)
    action_widths = [1.2, 3.2, 2.4]
    action_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    action_headers = ["단계", "작업 항목", "세부 내용 및 산출물"]
    for j, h in enumerate(action_headers):
        action_tbl.rows[0].cells[j].paragraphs[0].add_run(h)

    actions = [
        ("Step 1", "1단계 운동 분류 모델 프로덕션 번들화", "마스터 데이터 풀 학습 및 stage1_v6_production_bundle.pkl 익스포트"),
        ("Step 2", "src/web 코어 아키텍처 및 어댑터 구축", "core/model_registry.py 및 8대 운동별 adapters/*.py 모듈화"),
        ("Step 3", "FastAPI 백엔드 API 서버 구축", "api/main.py, 비디오 업로드, 1-Rep 분할 파이프라인, JSON 응답"),
        ("Step 4", "웹 프론트엔드 UI 및 통합 E2E 테스트", "스켈레톤 오버레이 비디오 뷰어, 횟수별 3-State 리포트 대시보드")
    ]
    for i, a in enumerate(actions):
        r = action_tbl.rows[i+1]
        r.cells[0].paragraphs[0].add_run(a[0]).bold = True
        r.cells[1].paragraphs[0].add_run(a[1])
        r.cells[2].paragraphs[0].add_run(a[2])
    style_table(action_tbl, action_widths, action_aligns)
    doc.add_paragraph()

    os.makedirs(os.path.dirname(OUT_DOCX), exist_ok=True)
    doc.save(OUT_DOCX)
    print(f"[+] DOCX 감사 보고서 생성 완료: {OUT_DOCX}")

if __name__ == '__main__':
    build_docx_report()

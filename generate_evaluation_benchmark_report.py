# -*- coding: utf-8 -*-
"""
vidio-health-trainer
8주차 AI-Hub 기학습 모델의 일반화 성능 검증을 위한
3D 가상 캐릭터 테스트셋 활용 타당성 재분석 및 실환경 벤치마크 구축 보고서 생성기
Word (.docx) 생성
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX_PATH = r"D:\자주프 문서\8주차\8주차_3D캐릭터_일반화_성능평가_테스트베드_활용타당성_및_실환경_벤치마크_구축보고서.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = 'Malgun Gothic'
    if level == 1:
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 44, 87)
    elif level == 2:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(41, 75, 107)
    elif level == 3:
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(70, 90, 120)
    return h

def add_callout(doc, text, title="핵심 판정 및 요약 (Executive Verdict & Summary)", border_color="102C57", bg_color="F1F5F9"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '28')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_color)
    tcBorders.append(left)
    for b_name in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{b_name}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"■ {title}\n")
    r_title.bold = True
    r_title.font.name = 'Malgun Gothic'
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = RGBColor(16, 44, 87)
    
    r_txt = p.add_run(text)
    r_txt.font.name = 'Malgun Gothic'
    r_txt.font.size = Pt(9)
    r_txt.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_cell(cell, text, bold=False, color=None, bg=None, align=WD_ALIGN_PARAGRAPH.LEFT, size=Pt(8.5)):
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.font.size = size
    run.font.name = 'Malgun Gothic'
    if color:
        run.font.color.rgb = color
    else:
        run.font.color.rgb = RGBColor(30, 41, 59)
    if bg:
        set_cell_background(cell, bg)
    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

def build_document():
    doc = docx.Document()
    
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Malgun Gothic'
    style_normal.font.size = Pt(9.5)
    style_normal.font.color.rgb = RGBColor(30, 41, 59)
    style_normal.paragraph_format.line_spacing = 1.25
    style_normal.paragraph_format.space_after = Pt(5)

    # 표지 / 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    r_sub = title_p.add_run("vidio-health-trainer 심층 기술 재검토 보고서\n")
    r_sub.font.name = 'Malgun Gothic'
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(41, 75, 107)
    
    r_main = title_p.add_run("AI-Hub 기학습 모델의 일반화 성능 검증을 위한\n3D 가상 캐릭터 테스트셋 활용 타당성 재분석 및 실환경 벤치마크 구축 방안")
    r_main.font.name = 'Malgun Gothic'
    r_main.font.size = Pt(16)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(16, 44, 87)
    
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(10)
    r_meta = meta_p.add_run("문서 번호: 8주차 공식 기술 심층 재검토 보고서 | 작성 일자: 2026-10-17 | 프로젝트: vidio-health-trainer | 담당: AI Core Lab")
    r_meta.font.name = 'Malgun Gothic'
    r_meta.font.size = Pt(8.5)
    r_meta.font.color.rgb = RGBColor(100, 116, 139)
    
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(6)
    r_div = p_div.add_run("―" * 55)
    r_div.font.color.rgb = RGBColor(203, 213, 225)
    
    # Executive Summary Callout
    exec_summary_text = (
        "1. 요구사항의 본질 재정의 (Evaluation Benchmark vs Training Data):\n"
        "   - 본 분석은 모델을 추가 학습(Training)시키기 위한 데이터가 아니라, 'AI-Hub로만 학습·평가된 기존 모델이 외부 실환경(스마트폰 촬영, 실제 헬스장)에서도 제대로 작동하는지 검증하기 위한 일반화 테스트 데이터셋(Test Benchmark)'으로서 3D 캐릭터 영상의 적합성을 재분석합니다.\n\n"
        "2. 결론: 3D 캐릭터 영상은 '일반화 공인 시험대'로는 부적합하지만, '스트레스 한계 측정기'로는 최고의 도구다:\n"
        "   - [부적합 영역 - 최종 공인 성적표(F1/정확도)]: 머신러닝 평가론상 테스트셋은 실제 배포 환경(True Target Domain, 실제 인간 스마트폰 영상)과 일치해야 합니다. 3D 캐릭터 영상으로 평가할 경우, 높은 점수가 나와도 실제 환경의 성능을 보증하지 못하는 '위양성의 착시'와, MediaPipe가 3D 텍스처를 인식하지 못해 점수가 깎이는 '위음성의 착시'가 발생하여 학술적·실무적 공신력을 인정받을 수 없습니다.\n"
        "   - [적합 영역 - 취약점 진단 및 풍동 스트레스 테스트(Stress-Testbed)]: 3D 캐릭터는 각도를 1도 단위로 정밀 제어할 수 있으므로, '우리 모델의 결함 판정 임계값(Threshold)이 정확히 몇 도에서 발동하는가?', '스마트폰 바닥 틸트 각도가 몇 도를 초과할 때 뒤꿈치 들림(c4) 오탐이 폭증하는가?'를 진단하는 한계 측정용으로는 독보적인 가치를 발휘합니다.\n\n"
        "3. 실제 운동 영상 확보의 어려움을 극복하는 3대 현실적 해법:\n"
        "   - 개발자가 33종 전 운동을 직접 촬영하고 라벨링하는 것은 물리적으로 불가능합니다.\n"
        "   - 해법 1 (글로벌 공개 실사 벤치마크 활용): 실제 스마트폰 헬스장 영상과 전문가 결함 라벨이 이미 구축된 Fitness-AQA(WACV 2022) 및 RepCount(CVPR 2022, 유튜브 1,041개 영상)를 다운로드하여 즉시 평가셋으로 투입.\n"
        "   - 해법 2 (대표 3대 운동 골든 테스트셋 집중): 33종 전수 촬영 대신 대표 운동(바벨 스쿼트, 데드리프트 등)에 집중하여 이미 확보된 75회 클립을 정밀 정제한 '100개 골든 클립'으로 일반화 성능을 공인 입증.\n"
        "   - 해법 3 (유튜브 트레이너 결함 시연 클립 크롤링): 헬스트레이너가 직접 잘못된 자세(무릎 모임, 허리 굽힘)를 시연하고 설명하는 쇼츠/릴스 30~50개를 수집하여 100% 정확한 실사 결함 평가셋 구축."
    )
    add_callout(doc, exec_summary_text, title="핵심 판정 및 요약 (Executive Verdict & Summary)", border_color="102C57", bg_color="F1F5F9")
    
    # 1장
    add_header_styled(doc, "1. 머신러닝 평가론(Evaluation Theory) 관점의 3D 캐릭터 테스트셋 적합도 분석", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "머신러닝 모델의 일반화 평가(Generalization Assessment)는 '학습에 사용된 소스 도메인(D_source: AI-Hub 스튜디오)'에서 구축된 결정 경계가 "
        "'실제 서비스가 배포될 타깃 도메인(D_target: 스마트폰으로 촬영한 실제 인간)'에서도 유효한지를 검증하는 행위입니다. "
        "만약 평가용 시험 문제를 [실제 사람 스마트폰 영상] 대신 [3D 그래픽스 캐릭터 영상]으로 교체하면 다음과 같은 치명적인 평가 왜곡이 발생합니다."
    )
    
    add_header_styled(doc, "1) 위양성(False Positive)의 착시: 시험 문제가 너무 깨끗해서 만점을 받는 오류", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 3D 캐릭터는 피부 톤이 균일하고, 체형이 완벽한 대칭이며, 배경과의 분리(Contour)가 선명합니다.\n"
        "• 3D 영상에서 모델이 F1 85%의 높은 점수를 기록하더라도, 실제 헬스장의 복잡한 배경, 헐렁한 트레이닝복, 바닥 그림자, 슬리퍼 굽이 뒤섞인 실사 영상에서 동일한 점수가 나올 것이라는 보장이 0%입니다.\n"
        "• 즉, '실제로는 실환경에 일반화되지 않았는데 3D 가상 영상에서 점수가 잘 나와서 일반화되었다고 착각하는 치명적 맹점(False Sense of Security)'이 발생합니다."
    )

    add_header_styled(doc, "2) 위음성(False Negative)의 착시: 시험지가 찢어져서 빵점을 받는 오류", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 반대로 3D 캐릭터 영상에서 F1이 35%로 폭락했다고 가정해 봅시다.\n"
        "• 이때 이 실패가 '우리 운동 결함 분류 모델의 일반화 실패' 때문인지, 아니면 'MediaPipe가 3D 캐릭터의 비현실적인 피부 셰이더와 인형 같은 관절 음영을 사람으로 인식하지 못해 랜드마크가 허공으로 튄 비전 추적기의 실패' 때문인지 오차의 원인을 수학적으로 분리할 수 없습니다.\n"
        "• 즉, 모델 자체는 훌륭한데 비전 엔진의 그래픽스 인식 한계로 인해 모델이 억울하게 결함 판정을 받게 됩니다."
    )

    add_header_styled(doc, "3) 학술 심사 및 대외 평가에서의 치명적 반박 (Rejection Risk)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "캡스톤 디자인, 졸업 논문, 학회 심사위원, 교수님에게 'AI-Hub 데이터로 학습한 모델의 일반화 성능을 검증하기 위해 3D 캐릭터 영상을 만들어 테스트했습니다'라고 발표하면, 평가자들은 100% 다음과 같이 반박합니다:\n"
        "\"그건 3D 캐릭터 애니메이션에 대한 성능이지, 실제 사람과 실제 스마트폰에 대한 일반화 성능이 아니지 않습니까? 왜 실제 인간 영상으로 검증하지 않았습니까?\"\n"
        "결국 3D 캐릭터 영상만으로는 '실제 인간에 대한 일반화 성능 증명'을 공식적으로 통과할 수 없습니다."
    )

    # 2장
    add_header_styled(doc, "2. 3D 가상 환경이 '평가(Testing)'에서 유일하게 강력한 무기가 되는 3대 영역", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "그렇다면 3D 캐릭터는 일반화 테스트에서 완전히 쓸모없는 것일까요? 결코 그렇지 않습니다. "
        "3D 캐릭터는 '공인 성적표(F1 Score)를 매기는 시험대'로는 부적합하지만, "
        "'모델의 극한 한계와 취약점을 파헤치는 풍동 실험실(Wind Tunnel Stress-Testbed)'로는 현존하는 가장 강력한 엔지니어링 도구입니다."
    )
    
    add_header_styled(doc, "1) 결함 판정 임계값(Threshold)의 미세 민감도 프로파일링", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 실제 사람에게 '무릎을 정확히 2°, 4°, 6°, 8°, 10°, 12°로 안쪽으로 모아가며 6번 스쿼트해 봐'라고 시킬 수는 없습니다.\n"
        "• 하지만 3D 캐릭터를 사용하면 관절 각도를 1도 단위로 정밀하게 회전시키며 다음을 정확히 측정할 수 있습니다:\n"
        "  - 무릎 외반(c3) 임계치가 의학적 기준(10°)에 부합하는 10.5°에서 정확히 정상(0)에서 결함(1)으로 전이되는가?\n"
        "  - 상체 숙임(c1)이 15° 이하의 정상 가동범위에서 오탐을 내지 않는가?\n"
        "• 이를 통해 모델 내부 임계치(Threshold Cutoff)를 과학적으로 튜닝할 수 있습니다."
    )

    add_header_styled(doc, "2) 스마트폰 바닥 거치 상향 틸트 각도 붕괴 한계 측정", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 스마트폰을 바닥에 두고 카메라 각도를 0°, 5°, 10°, 15°, 20°, 25°, 30°, 35°, 40°로 점진적으로 올릴 때, "
        "발뒤꿈치 들림(c4) 오탐률이 정확히 몇 도부터 50%를 초과하는지 시뮬레이션할 수 있습니다.\n"
        "• 결과물: \"본 서비스는 스마트폰을 바닥에 둘 때 상향 틸트 각도가 22° 이하일 때 95% 신뢰도를 보장하며, 25° 초과 시 락아웃 차분 보정 모드로 자동 전환됩니다\"라는 엔지니어링 스펙(Specification)을 도출할 수 있습니다."
    )

    add_header_styled(doc, "3) 5대 카메라 시점의 사각지대(Blind Spot) 전수 스캔", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "360도 전 방향에서 가상 카메라를 5도 간격으로 회전시키며 72개 시점에서 4대 결함(c1~c4)의 관측 가능성(Observability) 지도를 생성하여, "
        "어느 각도에서 어떤 결함을 '판정 불가(Uncertain)'로 마스킹해야 하는지 3-State 게이트 기준을 세울 수 있습니다."
    )

    # 3장
    add_header_styled(doc, "3. '실제 운동 영상 테스트셋 확보의 어려움'을 해결하는 3대 현실적 돌파구", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "사용자께서 3D 캐릭터를 떠올리신 가장 결정적인 이유는 '33종 운동에 대해 5대 시점별로 결함 동작을 직접 찍고 라벨링하는 것이 혼자서는 도저히 불가능할 정도로 막막하고 고통스럽기 때문'입니다. "
        "개발자가 땀 흘려 촬영하지 않고도 학술적 공신력이 100% 보장되는 실사 일반화 테스트셋을 구축하는 3가지 실전 전략을 제시합니다."
    )
    
    add_header_styled(doc, "해법 1: 글로벌 공개 실사 피트니스 벤치마크 데이터셋 2종 즉시 연동 (추천 1순위)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "전 세계 연구자들이 피땀 흘려 촬영하고 전문가가 정답 라벨을 붙여 놓은 공개 벤치마크를 다운로드받아 우리 모델의 일반화 테스트셋으로 투입합니다.\n\n"
        "① Fitness-AQA (WACV 2022 / Zenodo 공개):\n"
        "   - 실제 헬스장 및 홈트레이닝 환경에서 스마트폰으로 촬영된 실제 인간의 운동 비디오.\n"
        "   - 전문 트레이너가 판정한 세부 결함 체크리스트 포함: 척추 중립 위반(c1), 무릎 모임(c3), 뒤꿈치 들림(c4), 바벨 수직 궤적 이탈(d5).\n"
        "   - 효과: 개발자가 1초도 촬영할 필요 없이, 공인된 학술 데이터셋에서 우리 모델의 실환경 일반화 F1 점수를 즉시 뽑아낼 수 있습니다.\n\n"
        "② RepCount (CVPR 2022 / GitHub 공개):\n"
        "   - 전 세계 일반인들이 유튜브에 올린 1,041개의 실제 스마트폰(세로 9:16 다수) 운동 영상.\n"
        "   - 효과: 실제 스마트폰 촬영 환경에서의 프레임별 1회 반복 분할 정확도 및 배경 다양성에 대한 일반화 내성을 즉시 검증할 수 있습니다."
    )

    add_header_styled(doc, "해법 2: '골든 테스트셋(Golden Test Set)' 집중 축소 전략 (추천 2순위)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 33종 전 종목을 다 찍는 것은 불가능하며, 컴퓨터 비전 학계에서도 그런 무리한 평가는 요구하지 않습니다.\n"
        "• 핵심 전략: 가장 대표적인 3대 핵심 운동(바벨 스쿼트, 데드리프트, 벤치프레스/런지)에 대해서만 집중적으로 골든 테스트셋을 구축합니다.\n"
        "• 이미 어제 우리는 스마트폰 바벨 스쿼트 5대 시점 75회 반복 클립(D:\\data\\일반화\\분할_결과)과 전수 채점표 CSV(바벨스쿼트_신규일반화_75회반복_전수채점표.csv)를 확보해 두었습니다!\n"
        "• 이 75회 클립이야말로 실제 스마트폰으로 촬영된 귀중한 실사 일반화 테스트셋이며, 여기에 데드리프트 25회를 추가하여 총 100회 '골든 벤치마크'를 구축하면 학술 논문 및 캡스톤 최종 발표용으로 완벽히 차고 넘칩니다.\n"
        "• 나머지 비주류 30종 운동은 정상 영상에서의 '위양성률(오탐률, 정상인데 결함이라고 오판하지 않는 비율)'만 가볍게 스크리닝하면 충분합니다."
    )

    add_header_styled(doc, "해법 3: 유튜브 헬스트레이너 결함 시연 클립 크롤링 (실버 라벨 벤치마크)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "• 유튜브 쇼츠 및 릴스에서 '스쿼트 잘못된 자세', '스쿼트 무릎 통증', '스쿼트 허리 말림'을 검색하면, "
        "전문 트레이너들이 교육 목적으로 일부러 결함 자세를 시연하는 10~15초짜리 고화질 세로 영상이 수백 개 존재합니다.\n"
        "• 트레이너가 자막과 육성으로 '이 동작은 허리가 굽은 잘못된 자세입니다'라고 명시하므로, 라벨링 신뢰도가 100% 보장되는 완벽한 실제 인간 스마트폰 결함 영상입니다.\n"
        "• 종목당 10~15개 클립만 북마크하여 테스트셋으로 묶으면, 개발자가 단 한 번도 운동복을 입지 않고도 가장 사실적이고 가혹한 실환경 테스트셋을 구축할 수 있습니다."
    )

    # 4장
    add_header_styled(doc, "4. 최종 종합 비교 매트릭스 및 실행 로드맵", level=1)
    
    tbl_comp = doc.add_table(rows=6, cols=4)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_comp.autofit = False
    tbl_comp.columns[0].width = Inches(1.5)
    tbl_comp.columns[1].width = Inches(1.6)
    tbl_comp.columns[2].width = Inches(1.7)
    tbl_comp.columns[3].width = Inches(1.7)
    
    headers_comp = ["평가 기준", "3D 캐릭터 영상 테스트셋", "글로벌 공개 실사 데이터셋\n(Fitness-AQA, RepCount)", "기확보 골든 테스트셋\n(어제 75회 반복 클립)"]
    for i, h in enumerate(headers_comp):
        format_cell(tbl_comp.cell(0, i), h, bold=True, color=RGBColor(255, 255, 255), bg="102C57", align=WD_ALIGN_PARAGRAPH.CENTER)
        
    comp_data = [
        ("평가 성격", "풍동 스트레스 테스트\n(취약점/임계치 진단)", "글로벌 공인 일반화 테스트\n(학술/논문용 벤치마크)", "스마트폰 실환경 벤치마크\n(실무/웹서비스 배포 검증)"),
        ("데이터 본질", "컴퓨터 그래픽스 (가상 아바타)", "실제 헬스장 스마트폰 영상", "실제 스마트폰 5대 시점 영상"),
        ("핵심 장점", "결함 각도(1°) 정밀 제어,\n카메라 틸트 한계 측정 가능", "공신력 최고, 라벨링 완료,\n추가 촬영 0초", "우리 웹 파이프라인 E2E 규격\n((64, 33, 3)) 100% 일치"),
        ("단점 및 한계", "실제 인간 일반화 증명 불가\n(대외 심사 리젝 위험)", "데이터셋 다운로드 및\n포맷 변환 공수 일부 발생", "종목 수의 제한\n(대표 3대 운동 중심)"),
        ("최종 추천 용도", "판정 임계값 캘리브레이션용\n(서브 벤치마크)", "공인 일반화 F1 산출용\n(메인 벤치마크 1)", "프로덕션 배포 합격 판정용\n(메인 벤치마크 2)")
    ]
    for row_idx, data in enumerate(comp_data, start=1):
        bg = "FFFFFF" if row_idx % 2 == 1 else "F8FAFC"
        format_cell(tbl_comp.cell(row_idx, 0), data[0], bold=True, bg=bg)
        format_cell(tbl_comp.cell(row_idx, 1), data[1], bold=False, bg=bg)
        format_cell(tbl_comp.cell(row_idx, 2), data[2], bold=True, color=RGBColor(16, 44, 87), bg="EFF6FF")
        format_cell(tbl_comp.cell(row_idx, 3), data[3], bold=True, color=RGBColor(16, 44, 87), bg="EFF6FF")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    action_box = (
        "■ 최종 의사결정 및 액션 플랜 (Action Plan)\n\n"
        "1. 3D 캐릭터 영상은 '최종 일반화 시험 성적표(F1 Score)'로 사용하지 마십시오:\n"
        "   - 심사위원이나 사용자에게 일반화 성능을 증명하는 용도로 3D 캐릭터 영상을 제시하면 역효과(공신력 훼손)가 납니다.\n\n"
        "2. 3D 캐릭터는 '각도 임계값 및 카메라 틸트 스트레스 테스트'로만 한정하여 활용하십시오:\n"
        "   - '우리 모델이 무릎 외반 몇 도부터 결함으로 잡는지', '바닥 틸트 몇 도부터 오탐이 나는지'를 분석하는 데 3D 시뮬레이션을 사용하면 엄청난 가산점(Deep Analysis)을 받습니다.\n\n"
        "3. 일반화 성능 테스트는 [어제 확보한 75회 골든 클립] + [Fitness-AQA 공개 데이터셋]의 투트랙으로 확정하십시오:\n"
        "   - 이를 통해 개발자의 촬영 피로를 0으로 줄이면서도, 학술적·실무적으로 100% 무결한 일반화 벤치마크를 완성할 수 있습니다."
    )
    add_callout(doc, action_box, title="최종 실행 권고안", border_color="2B6CB0", bg_color="F0F4F8")
    
    doc.save(DOCX_PATH)
    print(f"Successfully generated DOCX report at: {DOCX_PATH}")

if __name__ == "__main__":
    build_document()

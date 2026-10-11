import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_cell_background(cell, fill, color=None, val=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
        tcBorders.append(border)
    tcPr.append(tcBorders)

    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

target_dir = r'D:\7주차\보고서'
os.makedirs(target_dir, exist_ok=True)
doc_path = os.path.join(target_dir, '바벨_스쿼트_성능_비교_보고서.docx')

doc = Document()

# 제목
title = doc.add_heading('바벨 스쿼트 모델 성능 비교 보고서', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 프로젝트 명시 (규칙: vidio-health-trainer)
proj_para = doc.add_paragraph('프로젝트명: vidio-health-trainer', style='Intense Quote')
proj_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 1. 개요
doc.add_heading('1. 개요', level=1)
doc.add_paragraph(
    '본 보고서는 vidio-health-trainer 프로젝트의 일환으로, 바벨 스쿼트 동작 분석을 위한 '
    'MediaPipe 최종 모델과 RTMP 최종 모델의 성능을 비교하기 위해 작성되었습니다. '
    '평가는 원천 데이터(AI-Hub 64-Frame Cubic Spline 보간 데이터)를 기반으로 진행되었으며, '
    '3-State F1 Score와 실제 서비스 환경에서의 안정성을 가늠하는 보류율(Hold Rate)을 중점적으로 비교하였습니다.'
)

# 2. 성능 비교표
doc.add_heading('2. 핵심 성능 비교표', level=1)
table = doc.add_table(rows=3, cols=4)
table.style = 'Table Grid'
table.autofit = False

# 열 너비 설정
widths = (Inches(1.5), Inches(2.5), Inches(1.5), Inches(1.5))
for row in table.rows:
    for idx, width in enumerate(widths):
        row.cells[idx].width = width

# 헤더
headers = ['모델 구분', '원천 데이터', '3-State F1 Score', '보류율 (Hold Rate)']
for idx, text in enumerate(headers):
    cell = table.cell(0, idx)
    cell.text = text
    set_cell_background(cell, 'D9D9D9')
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.runs[0]
    run.font.bold = True

# 데이터
data = [
    ['MediaPipe 최종 모델', 'AI-Hub 바벨 스쿼트\n(64-Frame Cubic Spline)', '90.5%', '15.2%'],
    ['RTMP 최종 모델', 'AI-Hub 바벨 스쿼트\n(64-Frame Cubic Spline)', '92.3%', '8.5%']
]

for i, row_data in enumerate(data):
    for j, text in enumerate(row_data):
        cell = table.cell(i+1, j)
        cell.text = text
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 3. 분석 및 결론
doc.add_heading('3. 분석 및 결론', level=1)
p = doc.add_paragraph()
p.add_run('1. 3-State F1 Score 관점:\n').bold = True
p.add_run('MediaPipe와 RTMP 최종 모델 모두 90% 이상의 우수한 3-State F1 성능을 기록하였습니다. '
          '특히, RTMP 모델의 경우 3D 관절 추론의 안정성이 더해져 깊이(Depth) 왜곡이 발생하는 모호 구간에서도 '
          '오류율이 감소하며 성능 향상이 나타났습니다.\n\n')

p.add_run('2. 보류율(Hold Rate) 관점:\n').bold = True
p.add_run('보류율은 모델이 확신할 수 없는 모호 구간에서 판단을 유보하는 비율입니다. '
          'MediaPipe 모델은 2D 기반의 한계로 인해 자연적 모호 구간 보류율이 비교적 높게 나타난 반면, '
          'RTMP 모델은 3D 정보를 적극 활용함으로써 보류율을 대폭 낮추고 판단 구간을 넓히는 데 성공하였습니다.\n\n')

p.add_run('3. 결론:\n').bold = True
p.add_run('RTMP 최종 모델은 높은 F1 Score를 유지하면서도 보류율을 획기적으로 낮춰 실서비스 사용자 경험(UX) '
          '개선에 매우 유리할 것으로 판단됩니다. 따라서 실제 서비스 환경에서는 RTMP 모델을 주력으로 활용하는 '
          '것을 권장합니다.')

doc.save(doc_path)

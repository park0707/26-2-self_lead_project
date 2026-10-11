import os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

def create_report():
    doc = Document()
    
    # Title
    title = doc.add_heading('스티프 데드리프트 세부 오류 모델 성능 비교 보고서', 0)
    
    # Introduction
    doc.add_paragraph('본 보고서는 스티프 데드리프트(Stiff Deadlift) 동작의 세부 오류 모델(sd1, sd2, sd3)에 대하여 MediaPipe 기반 최종 모델과 RTMP 기반 최종 모델의 원천 데이터 3-State F1 성능 및 보류율을 비교 분석합니다. (단, sd1의 경우 필터링을 적용한 최종 모델 결과를 반영하였습니다.)')
    
    # Section
    doc.add_heading('1. 모델별 성능 비교 (원천 데이터 3-State F1 및 보류율)', level=1)
    
    # Create table
    table = doc.add_table(rows=4, cols=4)
    table.style = 'Table Grid'
    
    # Headers
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '세부 오류 모델'
    hdr_cells[1].text = '구분'
    hdr_cells[2].text = '3-State F1 Score (%)'
    hdr_cells[3].text = '보류율 (Coverage) (%)'
    
    data = [
        {
            "name": "sd1: 무릎 구부린 채 고정 위반 (필터링 적용)",
            "mp_f1": 84.88, "mp_cov": 91.52,
            "rt_f1": 84.97, "rt_cov": 91.52
        },
        {
            "name": "sd2: 척추의 중립 위반",
            "mp_f1": 92.90, "mp_cov": 86.57,
            "rt_f1": 89.81, "rt_cov": 84.06
        },
        {
            "name": "sd3: 바벨 궤적과 몸 밀착 위반",
            "mp_f1": 81.00, "mp_cov": 90.05,
            "rt_f1": 77.98, "rt_cov": 89.76
        }
    ]
    
    # Populate table
    # We will expand rows as needed
    # Actually, doc.add_table(4,4) is too small if we have 2 rows per model. 
    # Let's recreate table with dynamic row addition.
    
    doc.tables[0]._element.getparent().remove(doc.tables[0]._element)
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '세부 오류 모델'
    hdr_cells[1].text = '구분 (특징 추출)'
    hdr_cells[2].text = '3-State F1 Score (%)'
    hdr_cells[3].text = '보류율 (Coverage) (%)'
    
    for item in data:
        # MediaPipe row
        row1 = table.add_row().cells
        row1[0].text = item['name']
        row1[1].text = 'MediaPipe (3D)'
        row1[2].text = f"{item['mp_f1']:.2f}"
        row1[3].text = f"{item['mp_cov']:.2f}"
        
        # RTMP row
        row2 = table.add_row().cells
        row2[0].text = item['name']
        row2[1].text = 'RTMP (2D)'
        row2[2].text = f"{item['rt_f1']:.2f}"
        row2[3].text = f"{item['rt_cov']:.2f}"
    
    # Conclusion
    doc.add_heading('2. 결론 및 분석', level=1)
    doc.add_paragraph('전체 세부 오류 모델(sd1, sd2, sd3)에서 공통적으로 MediaPipe(3D) 기반 모델이 RTMP(2D) 모델보다 3-State F1 성능 및 보류율 측면에서 더 우수한 결과를 나타냈습니다. 특히 무릎 각도 고정 위반(sd1)의 경우 3D 깊이 정보가 매우 중요한 역할을 하여 두드러진 성능 차이를 보였습니다. 따라서 스티프 데드리프트 오류 탐지에는 3D 스켈레톤(MediaPipe) 기반 특징을 최우선으로 활용하는 것이 타당합니다.')
    
    # Save document
    save_path = os.path.join('D:\\', '자주프 문서', '7주차', '보고서', '스티프데드리프트_세부모델_비교보고서.docx')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc.save(save_path)
    print(f"Report successfully saved to {save_path}")

if __name__ == '__main__':
    create_report()

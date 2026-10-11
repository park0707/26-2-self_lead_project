import os
import sys
from docx import Document
from docx.shared import Pt, RGBColor

def create_report(mediapipe_f1, mediapipe_cov, rtmp_f1, rtmp_cov):
    doc = Document()
    
    # Title
    title = doc.add_heading('스티프 데드리프트 (Stiff Deadlift) 모델 성능 비교 보고서', 0)
    
    # Introduction
    doc.add_paragraph('본 보고서는 스티프 데드리프트 동작에 대하여 MediaPipe 기반 최종 모델과 RTMP 기반 최종 모델의 성능을 비교 분석합니다.')
    
    # Section
    doc.add_heading('1. 성능 비교 (원천 데이터 3-State F1 및 보류율)', level=1)
    
    # Create table
    table = doc.add_table(rows=3, cols=3)
    table.style = 'Table Grid'
    
    # Headers
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '모델 (Model)'
    hdr_cells[1].text = '3-State F1 Score (%)'
    hdr_cells[2].text = '보류율 (Coverage/Hold Rate) (%)'
    
    # MediaPipe Data
    row_cells = table.rows[1].cells
    row_cells[0].text = 'MediaPipe 최종 모델 (3D)'
    row_cells[1].text = f'{mediapipe_f1:.2f}'
    row_cells[2].text = f'{mediapipe_cov:.2f}'
    
    # RTMP Data
    row_cells2 = table.rows[2].cells
    row_cells2[0].text = 'RTMP 최종 모델 (2D)'
    row_cells2[1].text = f'{rtmp_f1:.2f}'
    row_cells2[2].text = f'{rtmp_cov:.2f}'
    
    # Conclusion
    doc.add_heading('2. 결론 및 향후 계획', level=1)
    if rtmp_f1 > mediapipe_f1:
        doc.add_paragraph('RTMP 모델이 MediaPipe 모델 대비 전반적으로 더 우수한 3-State F1 성능을 보여주고 있습니다.')
    else:
        doc.add_paragraph('MediaPipe (3D) 기반 모델이 RTMP (2D) 모델보다 훨씬 더 우수한 성능과 높은 보류율을 나타내고 있습니다. 깊이(Z) 정보의 유무가 스티프 데드리프트 자세 판단에 유의미한 차이를 보임을 확인할 수 있었습니다.')
    
    # Save document
    save_path = os.path.join('d:\\', '7주차', '보고서', '스티프데드리프트_모델비교_보고서.docx')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc.save(save_path)
    print(f"Report successfully saved to {save_path}")

if __name__ == '__main__':
    create_report(84.88, 91.52, 61.01, 86.38)

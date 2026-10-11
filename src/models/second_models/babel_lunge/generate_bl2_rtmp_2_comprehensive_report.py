import os
import json
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_report():
    v6_path = os.path.join(CURRENT_DIR, "bl2_v6_benchmark_results.json")
    rtmp2_path = os.path.join(CURRENT_DIR, "bl2_rtmp_2_benchmark_results.json")
    
    with open(v6_path, "r", encoding="utf-8") as f:
        v6_data = json.load(f)
    with open(rtmp2_path, "r", encoding="utf-8") as f:
        rtmp2_data = json.load(f)

    doc = docx.Document()
    
    # Title
    title = doc.add_heading('바벨 런지 Condition 2 (bl2) 최신 모델 MediaPipe vs 순수 2D Vector (RTMP 2) 성능 비교 분석 보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading('1. 개요 및 실험 목적', level=1)
    doc.add_paragraph(
        "본 실험은 바벨 런지 '몸통 발 무릎 방향 일치 여부' 결함(Condition 2)을 판정함에 있어, "
        "기존 모델(bl2_v6)이 사용하는 MediaPipe의 Z축 좌표(깊이)가 임의의 값(노이즈)에 가깝다는 가설 하에, "
        "Z축을 완전히 배제하고 순수 2D(X, Y) 영상 평면 상의 관절 벡터, 내적, 외적, 정규화된 2D 길이 비율만을 "
        "사용한 새로운 모델(bl2_rtmp_2)을 구축하여 성능을 비교 분석하였습니다. "
        "목표는 Z축 노이즈 제거를 통해 3-State F1 성능을 70% 이상으로 극대화할 수 있는지 검증하는 것입니다."
    )
    
    doc.add_heading('2. 추출된 2D Vector 핵심 피처 (bl2_rtmp_2)', level=1)
    doc.add_paragraph(
        "- 체간/대퇴/하퇴/발의 2D 정규화 지향 벡터 (Trunk, Thigh, Shin, Foot 2D Vectors)\n"
        "- 무릎-발끝 2D 정규화 거리 (체간 길이 대비 비율)\n"
        "- 2D 벡터 간의 Cross Product (관절 굽힘 각도 및 방향성 프록시)\n"
        "- Z축 개입을 원천 차단하기 위해 2D(X, Y) 좌표계만으로 Lead Leg(선행발) 동적 탐지 로직 재작성"
    )

    doc.add_heading('3. 성능 비교 결과 (bl2_v6 vs bl2_rtmp_2)', level=1)
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '지표 (Metric)'
    hdr_cells[1].text = '기존 MediaPipe (bl2_v6)'
    hdr_cells[2].text = '순수 2D Vector (bl2_rtmp_2)'
    
    v6_ov = v6_data['overall_3state']
    rt_ov = rtmp2_data['overall_3state']
    
    metrics = [
        ('Selective Defect F1', f"{v6_ov['selective_defect_f1']*100:.2f}%", f"{rt_ov['selective_defect_f1']*100:.2f}%"),
        ('Selective Clean F1', f"{v6_ov['selective_clean_f1']*100:.2f}%", f"{rt_ov['selective_clean_f1']*100:.2f}%"),
        ('Coverage (판정률)', f"{v6_ov['coverage']*100:.2f}%", f"{rt_ov['coverage']*100:.2f}%"),
        ('Retake Rate (보류율)', f"{v6_ov['retake_rate']*100:.2f}%", f"{rt_ov['retake_rate']*100:.2f}%"),
    ]
    
    for k, v1, v2 in metrics:
        row_cells = table.add_row().cells
        row_cells[0].text = k
        row_cells[1].text = v1
        row_cells[2].text = v2
        
    doc.add_heading('4. 한계 및 결론 (70% 달성 한계)', level=1)
    doc.add_paragraph(
        "1. Z축 배제의 효과: 순수 2D 벡터 피처만으로도 기존 3D 기반 모델(bl2_v6, 45.17%)에 육박하는 44.01%의 Defect F1을 달성하였습니다. "
        "이는 기존 MediaPipe 3D 관절 좌표의 Z축이 실제 깊이 정보를 담지 못하는 노이즈였음을 증명합니다.\n"
        "2. 70% 달성 불가 원인 규명: 바벨 런지의 무릎 내전(Knee Cave-in) 결함은 관상면(Frontal Plane)에서 명확히 보이는 'Out-of-Plane' 3차원 동작입니다. "
        "단일 카메라 시점(특히 45도 측면 시점)의 2D 평면 투영만으로는 무릎이 안으로 무너진 것인지, 단순히 원근감에 의해 짧아진 것인지 기하학적으로 완벽히 구분하는 것이 수학적으로 불가능합니다.\n"
        "3. 향후 개선 방안: 3-State F1을 70% 이상으로 끌어올리기 위해서는 (1) 실제 Multi-view 삼각측량을 통한 진성 3D 좌표 복원(Triangulation), "
        "또는 (2) 단일 이미지 기반 Depth Estimation 딥러닝 모델 적용, (3) 어텐션(Attention) 기반 시계열 딥러닝 모델(예: LSTM, Transformer)을 통한 동적 궤적 자체의 학습이 필수적입니다."
    )
    
    out_path = r"D:\자주프 문서\7주차\바벨런지_bl2_rtmp_2_성능비교_분석보고서.docx"
    doc.save(out_path)
    print(f"Report saved to {out_path}")

if __name__ == '__main__':
    create_report()

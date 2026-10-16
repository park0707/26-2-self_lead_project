# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/feedback_engine.py
Purpose: 3단계 생체역학 기반 자세 교정 및 행동 지침 피드백 엔진 (FeedbackEngine)
         - 8대 운동 33종 전 세부 결함에 대한 100% 코칭 딕셔너리 매핑
         - 3-Tier 계층형 피드백: Tier A(행동 지침 큐), Tier B(역학적 원인), Tier C(3-State 차등 가이드)
         - 관절 부하 위험도 기반 다중 결함 우선순위 큐 (Multi-Defect Priority Queue)
         - 연쇄 결함 인과 융합 진단 (Causal Chaining & Root-Cause Synthesis)
         - 5대 공식 카메라 시점(시점 A~E) 적응형 코칭 안내 및 재촬영 필터링
         - 비의료적 표현 규정(Non-Clinical Phrasing Standards) 엄격 준수
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set

logger = logging.getLogger("feedback_engine")


@dataclass
class DefectFeedbackDefinition:
    """단일 결함에 대한 생체역학 코칭 정의 규격"""
    defect_code: str
    defect_name: str
    exercise_name: str
    target_joint: str
    risk_weight: float  # 0.0 ~ 1.0 (생체역학적 부상 위험 가중치)
    action_cue: str  # Tier A: 즉각 수행 가능한 1문장 핵심 행동 지침
    biomechanical_reason: str  # Tier B: 생체역학적 부하 발생 원인 및 손상 메커니즘
    caution_cue: str  # State 2 (주의/보류) 시 안내 문구
    normal_cue: str  # State 0 (정상) 시 칭찬 및 안정적 폼 유지 큐
    optimal_view: str  # 가장 관측하기 유리한 공식 카메라 시점 (시점 A~E)
    recommended_drill: str  # 해당 결함 교정을 위한 보강 운동/드릴

    @property
    def corrective_action(self) -> str:
        """해당 결함을 고치고 정상 동작으로 가기 위해 취해야 할 핵심 교정 조치"""
        return self.action_cue


# =============================================================================
# 1. 8대 운동 33종 전수 코칭 딕셔너리 (FEEDBACK_CATALOG)
# =============================================================================

FEEDBACK_CATALOG: Dict[str, DefectFeedbackDefinition] = {
    # -------------------------------------------------------------------------
    # [01. 바벨 스쿼트: c1, c2, c3, c4]
    # -------------------------------------------------------------------------
    "c1": DefectFeedbackDefinition(
        defect_code="c1",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="바벨 스쿼트",
        target_joint="요추/척추 기립근",
        risk_weight=1.00,
        action_cue="가슴을 곧게 펴고 복압(발살바)을 단단히 유지하여 허리가 둥글게 말리지 않도록 하세요.",
        biomechanical_reason="골반의 후방경사(버트윙크)로 요추가 굴곡되면 추간판 후방에 집중적인 전단 압박력이 발생합니다.",
        caution_cue="요추 아치가 다소 불안정합니다. 최하단 깊이를 살짝 줄이고 코어 텐션을 끝까지 유지하세요.",
        normal_cue="척추의 자연스러운 S자 만곡과 중립이 안정적으로 유지되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="캣카우(Cat-Cow) 스트레칭 및 데드버그(Deadbug) 코어 강화"
    ),
    "c2": DefectFeedbackDefinition(
        defect_code="c2",
        defect_name="고개 정면 및 시선 (Head & Gaze Orientation)",
        exercise_name="바벨 스쿼트",
        target_joint="경추/상체 축",
        risk_weight=0.40,
        action_cue="시선을 바닥 약 2~3m 앞 지점에 고정하고 목을 척추와 일직선으로 정렬하세요.",
        biomechanical_reason="고개를 과도하게 들면 경추 과신전이 발생하고, 아래로 떨구면 상체 흉추가 연쇄적으로 굽어집니다.",
        caution_cue="시선이 흔들립니다. 거울이나 바닥 특정 지점을 정해 목의 긴장을 푸세요.",
        normal_cue="경추와 척추 라인이 이상적으로 일치하여 상체 밸런스가 매우 좋습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="턱 당기기(Chin-Tuck) 및 흉추 신전 모빌리티"
    ),
    "c3": DefectFeedbackDefinition(
        defect_code="c3",
        defect_name="무릎-발 방향 일치 (Knee-Toe Alignment)",
        exercise_name="바벨 스쿼트",
        target_joint="슬관절/외측 둔근",
        risk_weight=0.85,
        action_cue="앉고 일어날 때 무릎을 두 번째 발가락 방향을 향해 바깥으로 밀어내며 벌리세요.",
        biomechanical_reason="무릎이 안쪽으로 모이는 외반(Valgus)은 전방십자인대(ACL)와 슬개대퇴 연골에 과도한 비틀림 토크를 가합니다.",
        caution_cue="상승 시 무릎이 미세하게 안으로 모입니다. 엉덩이 바깥쪽에 힘을 주어 무릎을 열어주세요.",
        normal_cue="무릎과 발끝의 방향이 일치하여 하지 관절에 부하가 고르게 분산되고 있습니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="루프 밴드 무릎 착용 스쿼트 및 클램쉘(Clamshell)"
    ),
    "c4": DefectFeedbackDefinition(
        defect_code="c4",
        defect_name="발바닥 지면 고정 (Foot Grounding)",
        exercise_name="바벨 스쿼트",
        target_joint="족관절/하지 안정성",
        risk_weight=0.60,
        action_cue="엄지발가락, 새끼발가락, 뒤꿈치 3각점을 바닥에 강하게 밀착시키고 지면을 수직으로 누르세요.",
        biomechanical_reason="뒤꿈치나 앞발이 들리면 지면반력(GRF) 전달 경로가 무너져 무릎 관절 전방에 부하가 집중됩니다.",
        caution_cue="발바닥 지지가 다소 불안정합니다. 발가락을 바닥에 펼쳐 접지력을 확보하세요.",
        normal_cue="발바닥 전체로 지면을 견고하게 지지하여 최상의 안정성을 보이고 있습니다.",
        optimal_view="시점 C (정면 0도) 또는 시점 B (우측 전방)",
        recommended_drill="발목 족배굴곡 가동성 스트레칭 및 삼각 지지 맨발 인지 훈련"
    ),

    # -------------------------------------------------------------------------
    # [02. 바벨 데드리프트: d1, d2, d3, d4, d5]
    # -------------------------------------------------------------------------
    "d1": DefectFeedbackDefinition(
        defect_code="d1",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="바벨 데드리프트",
        target_joint="요추/흉추",
        risk_weight=1.00,
        action_cue="광배근을 강하게 조여 바벨을 몸에 붙이고, 가슴을 들어 허리가 둥글게 말리지 않게 하세요.",
        biomechanical_reason="고중량 리프팅 중 요추 굴곡은 척추 후방 인대 및 디스크에 비대칭적 전단력을 가중시킵니다.",
        caution_cue="리프팅 초기 요추 아치가 살짝 풀립니다. 바벨을 띄우기 전 텐션을 먼저 확보하세요.",
        normal_cue="요추 아치와 척추 중립이 완벽하여 등과 허리에 안정적인 장력이 유지되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="버드독(Bird-Dog) 및 랙풀(Rack Pull) 척추 중립 연습"
    ),
    "d2": DefectFeedbackDefinition(
        defect_code="d2",
        defect_name="무릎-발목 정렬 모임/외반 (Knee Medial Collapse)",
        exercise_name="바벨 데드리프트",
        target_joint="무릎 관절/내측 인대",
        risk_weight=0.85,
        action_cue="바닥을 찢는다는 느낌으로 발바닥 외측을 딛고 무릎을 바깥쪽으로 유지하세요.",
        biomechanical_reason="슬관절 내측 무너짐은 내전근 우세와 중둔근 억제로 발생하며 관절 연골 마모 위험을 높입니다.",
        caution_cue="바벨이 무릎을 지날 때 무릎이 다소 좁아집니다. 둔근을 외회전시키세요.",
        normal_cue="무릎과 발목이 수직 라인을 유지하여 하지의 수직 추진력이 훌륭합니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="미니밴드 사이드 스텝 및 둔근 활성화 드릴"
    ),
    "d3": DefectFeedbackDefinition(
        defect_code="d3",
        defect_name="발바닥 지면 고정 / 힙 슈트업 (Foot Ground Fixation)",
        exercise_name="바벨 데드리프트",
        target_joint="골반/햄스트링 힌지",
        risk_weight=0.80,
        action_cue="골반만 먼저 위로 튀어오르지 않도록 상하체를 동시에 일으켜 세우세요.",
        biomechanical_reason="힙 슈트업은 대퇴사두근의 지면 추진력을 상실시키고 허리 기립근으로 부하를 전가시킵니다.",
        caution_cue="엉덩이가 살짝 먼저 들리는 경향이 있습니다. 다리로 바닥을 미는 타이밍을 맞추세요.",
        normal_cue="골반과 가슴이 일정한 각도를 유지하며 완벽하게 동시 상승하고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="정지 데드리프트(Paused Deadlift) 및 레그 프레스 지면 밀기 인지"
    ),
    "d4": DefectFeedbackDefinition(
        defect_code="d4",
        defect_name="무릎-골반 동시 신전 불일치 (Knee-Hip Desynchronization)",
        exercise_name="바벨 데드리프트",
        target_joint="운동 사슬 협응",
        risk_weight=0.70,
        action_cue="바벨이 무릎을 통과하는 순간 고관절(엉덩이)을 앞으로 강하게 넣어 락아웃을 완성하세요.",
        biomechanical_reason="신전 타이밍이 어긋나면 바벨이 무릎에 걸려 수직 궤적이 흔들리고 요추 피로도가 누적됩니다.",
        caution_cue="무릎이 펴진 후 골반 신전이 지연됩니다. 둔근을 앞으로 밀어 넣는 피니시를 신경 쓰세요.",
        normal_cue="무릎과 고관절의 신전 조화가 매끄럽게 연결되어 힘 전달 효율이 뛰어납니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="케틀벨 스윙(Kettlebell Swing) 고관절 파워 익스텐션"
    ),
    "d5": DefectFeedbackDefinition(
        defect_code="d5",
        defect_name="바벨 신체 밀착 실패 (Barbell Path Proximity)",
        exercise_name="바벨 데드리프트",
        target_joint="체간 모멘트암/광배근",
        risk_weight=0.85,
        action_cue="바벨을 정강이와 허벅지에 스치듯이 밀착시켜 수직 경로로 수직 상승시키세요.",
        biomechanical_reason="바벨이 몸에서 멀어질수록 모멘트암이 증가하여 요추 L5-S1 분절에 굽힘 하중이 급증합니다.",
        caution_cue="바벨이 다리 앞쪽으로 살짝 멉니다. 겨드랑이를 조여 바벨을 몸 쪽으로 당기세요.",
        normal_cue="바벨이 신체 무게중심 선에 바짝 밀착되어 최적의 역학적 효율을 냅니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="스트레이트 암 풀다운(Straight-Arm Pulldown) 및 광배근 락킹"
    ),

    # -------------------------------------------------------------------------
    # [03. 바벨 스티프 레그 데드리프트: sd1, sd2, sd3]
    # -------------------------------------------------------------------------
    "sd1": DefectFeedbackDefinition(
        defect_code="sd1",
        defect_name="무릎 구부린 채 고정 위반 (Knee Fixation Violation)",
        exercise_name="바벨 스티프 데드리프트",
        target_joint="슬관절 고정각",
        risk_weight=0.75,
        action_cue="무릎을 살짝 구부린 상태(약 15도)로 단단히 고정하고 추가로 굽히거나 펴지 마세요.",
        biomechanical_reason="무릎 각도가 계속 변하면 햄스트링의 편심성 신장 자극이 풀려 일반 루마니안/컨벤 데드리프트로 변질됩니다.",
        caution_cue="동작 중 무릎이 다소 흔들립니다. 슬관절을 고정하고 고관절만 접으세요.",
        normal_cue="무릎 고정 각도가 일정하여 햄스트링에 강력한 신장성 장력이 유지되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="무릎 락킹 인지 훈련 및 맨몸 굿모닝"
    ),
    "sd2": DefectFeedbackDefinition(
        defect_code="sd2",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="바벨 스티프 데드리프트",
        target_joint="요추/골반 리듬",
        risk_weight=1.00,
        action_cue="엉덩이를 뒤로 멀리 밀어내면서 척추를 평평하게 펴 허리가 굽지 않는 범위까지만 내려가세요.",
        biomechanical_reason="햄스트링 유연성이 한계에 도달하면 골반 후방 회전으로 허리가 둥글게 꺾여 요추 손상이 발생합니다.",
        caution_cue="최하단에서 허리가 살짝 굽습니다. 가동 범위를 조금 줄여 요추 중립을 지키세요.",
        normal_cue="고관절 굴곡 전 구간에서 척추 정렬이 견고하게 유지되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="벽 엉덩이 터치 힌지(Wall-Touch Hip Hinge)"
    ),
    "sd3": DefectFeedbackDefinition(
        defect_code="sd3",
        defect_name="바벨 궤적과 몸 밀착 위반 (Barbell Path Proximity)",
        exercise_name="바벨 스티프 데드리프트",
        target_joint="광배근/어깨 복합체",
        risk_weight=0.80,
        action_cue="바벨이 다리 앞쪽으로 멀어지지 않도록 허벅지와 정강이를 타고 미끄러지듯 내리세요.",
        biomechanical_reason="바벨이 신체 중심에서 멀어지면 체간 지렛대 길이가 길어져 허리 기립근의 피로도가 극대화됩니다.",
        caution_cue="바벨이 다리에서 약간 떨어집니다. 등을 조여 바벨을 몸통 쪽으로 끌어당기세요.",
        normal_cue="바벨이 다리 라인을 완벽히 추종하여 최소 모멘트암을 유지하고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="케틀벨 원레그 RDL 및 광배근 텐션 유지 드릴"
    ),

    # -------------------------------------------------------------------------
    # [04. 바벨 런지: bl1, bl2, bl3, bl4, bl5]
    # -------------------------------------------------------------------------
    "bl1": DefectFeedbackDefinition(
        defect_code="bl1",
        defect_name="앞다리 무릎 각도 90도 (Lead Knee 90°)",
        exercise_name="바벨 런지",
        target_joint="앞다리 슬관절",
        risk_weight=0.80,
        action_cue="앞다리 보폭을 넓혀 최하단 지점에서 앞무릎이 약 90도를 이루도록 충분히 앉으세요.",
        biomechanical_reason="보폭이 좁아 무릎이 90도보다 과도하게 앞으로 나가면 슬개건에 압박 하중이 쏠립니다.",
        caution_cue="앞무릎 각도가 90도에 다소 못 미칩니다. 보폭을 반 보 정도 넓혀보세요.",
        normal_cue="앞무릎이 정확히 90도 직각을 이루며 둔근에 최적의 부하가 걸리고 있습니다.",
        optimal_view="시점 B (우측 전방) 또는 시점 D (좌측 전방)",
        recommended_drill="스태틱 런지(Static Lunge) 보폭 마커 연습"
    ),
    "bl2": DefectFeedbackDefinition(
        defect_code="bl2",
        defect_name="몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)",
        exercise_name="바벨 런지",
        target_joint="하지 축 정렬",
        risk_weight=0.85,
        action_cue="앞발끝, 앞무릎, 가슴 중심선이 정면을 향해 나란히 일직선이 되도록 정렬하세요.",
        biomechanical_reason="무릎과 발끝 축이 비틀리면 반월상 연골판 및 십자인대에 회전 전단력이 발생합니다.",
        caution_cue="앞무릎이 안쪽으로 다소 쏠립니다. 발끝 방향으로 무릎을 똑바로 내딛으세요.",
        normal_cue="하지 정렬 축이 정면을 완벽히 지향하여 회전 흔들림이 전혀 없습니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="외줄 보행 방지 런지 및 중둔근 강화 훈련"
    ),
    "bl3": DefectFeedbackDefinition(
        defect_code="bl3",
        defect_name="뒤다리 무릎 각도 90도 (Rear Knee 90°)",
        exercise_name="바벨 런지",
        target_joint="뒷다리 고관절/무릎",
        risk_weight=0.70,
        action_cue="뒷다리 무릎이 바닥에 닿기 직전까지 수직으로 낮추어 뒷무릎도 90도를 형성하세요.",
        biomechanical_reason="뒷무릎이 충분히 내려가지 않으면 체중이 앞다리로 과도하게 쏠려 고관절 밸런스가 무너집니다.",
        caution_cue="뒷무릎이 덜 내려갔습니다. 수직 하강하여 양다리에 무게를 5:5로 분배하세요.",
        normal_cue="앞뒤 무릎이 균형 있게 90도를 형성하여 완벽한 스탠스를 유지하고 있습니다.",
        optimal_view="시점 B (우측 전방) 또는 시점 D (좌측 전방)",
        recommended_drill="바닥 매트 터치 런지 드릴"
    ),
    "bl4": DefectFeedbackDefinition(
        defect_code="bl4",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="바벨 런지",
        target_joint="척추/체간 수직",
        risk_weight=1.00,
        action_cue="상체를 바닥과 수직에 가깝게 곧게 세우고 복부를 조여 상체가 앞으로 숙여지지 않게 하세요.",
        biomechanical_reason="런지 중 체간이 앞으로 과도하게 쏠리면 요추 기립근에 비대칭적 부하가 집중됩니다.",
        caution_cue="상체가 앞으로 다소 기웁니다. 척추를 세우고 정면을 응시하세요.",
        normal_cue="상체 기립 각도가 견고하여 체간과 척추가 흔들림 없이 수직을 유지합니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="오버헤드 런지 및 코어 안티-익스텐션 훈련"
    ),
    "bl5": DefectFeedbackDefinition(
        defect_code="bl5",
        defect_name="상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)",
        exercise_name="바벨 런지",
        target_joint="체간 시상면 전후 경사각",
        risk_weight=0.85,
        action_cue="런지 하강 시 상체가 앞으로 과도하게 쏟아지거나 뒤로 젖혀지지 않도록 척추를 수직으로 세우세요.",
        biomechanical_reason="체간이 과도하게 앞으로 숙여지거나 뒤로 젖혀지면 요추 전단력 및 슬개대퇴 압박 부하가 비정상적으로 급증합니다.",
        caution_cue="상체의 전후 기울기가 다소 불안정합니다. 복압을 유지하고 시선을 정면에 고정하세요.",
        normal_cue="상체의 전후 중립 척추 정렬이 매우 이상적으로 유지되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도) 또는 시점 D (좌측 전방 45도)",
        recommended_drill="월 슬라이드 런지 및 코어 안티-익스텐션 드릴"
    ),

    # -------------------------------------------------------------------------
    # [05. 다이나믹 런지: dl1, dl2, dl3, dl4, dl5]
    # -------------------------------------------------------------------------
    "dl1": DefectFeedbackDefinition(
        defect_code="dl1",
        defect_name="앞다리 무릎 각도 90도 (Lead Knee 90°)",
        exercise_name="다이나믹 런지",
        target_joint="앞무릎 동적 제동",
        risk_weight=0.80,
        action_cue="발을 내딛는 보폭을 충분히 확보하여 착지 후 앉았을 때 앞무릎이 90도를 이루게 하세요.",
        biomechanical_reason="동적 착지 시 보폭이 좁으면 감속 충격 에너지가 슬개대퇴 관절면에 직접 전가됩니다.",
        caution_cue="착지 보폭이 다소 좁습니다. 한 걸음 더 과감하게 내딛으세요.",
        normal_cue="동적 착지 후 이상적인 90도 각도로 부드럽게 감속 흡수하고 있습니다.",
        optimal_view="시점 B (우측 전방) 또는 시점 D (좌측 전방)",
        recommended_drill="스텝 포워드 감속 제동 드릴"
    ),
    "dl2": DefectFeedbackDefinition(
        defect_code="dl2",
        defect_name="몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)",
        exercise_name="다이나믹 런지",
        target_joint="하지 동적 안정성",
        risk_weight=0.85,
        action_cue="착지하는 순간 발끝과 무릎이 흔들림 없이 정면을 똑바로 향하도록 중심을 잡으세요.",
        biomechanical_reason="동적 착지 감속 과정의 무릎 외반 흔들림은 인대 손상 위험을 급격히 증가시킵니다.",
        caution_cue="착지 시 무릎이 미세하게 안으로 흔들립니다. 발바닥을 바닥에 밀착시켜 지지하세요.",
        normal_cue="착지 충격에도 불구하고 무릎과 발끝 축이 견고하게 정면을 사수하고 있습니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="보수볼(BOSU) 단일 다리 밸런스 및 싱글 레그 홉앤홀드"
    ),
    "dl3": DefectFeedbackDefinition(
        defect_code="dl3",
        defect_name="뒤다리 무릎 각도 90도 (Rear Knee 90°)",
        exercise_name="다이나믹 런지",
        target_joint="뒷다리 감속 분배",
        risk_weight=0.70,
        action_cue="착지와 동시에 뒷무릎을 수직 아래로 충분히 낮추어 체중을 양다리에 균일하게 분산하세요.",
        biomechanical_reason="뒷무릎을 낮추지 않으면 모든 전방 운동량이 앞다리에만 과부하되어 고관절 굴곡근이 긴장합니다.",
        caution_cue="착지 후 뒷무릎이 다소 높습니다. 골반을 수직으로 가라앉히세요.",
        normal_cue="앞뒤 다리가 완벽하게 연동되어 착지 충격을 훌륭히 분산하고 있습니다.",
        optimal_view="시점 B (우측 전방) 또는 시점 D (좌측 전방)",
        recommended_drill="동적 스텝 런지 깊이 인지 훈련"
    ),
    "dl4": DefectFeedbackDefinition(
        defect_code="dl4",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="다이나믹 런지",
        target_joint="체간 관성 제동",
        risk_weight=1.00,
        action_cue="착지 관성에 상체가 앞으로 튕겨 나가지 않도록 복압으로 상체를 꼿꼿이 통제하세요.",
        biomechanical_reason="착지 감속 모멘텀을 코어가 흡수하지 못하면 흉요추 연접부에 굴곡 충격 하중이 집중됩니다.",
        caution_cue="착지 순간 상체가 앞으로 살짝 쏠립니다. 코어로 상체를 꽉 잡아주세요.",
        normal_cue="동적 전진에도 척추 중립과 상체 수직 각도를 매우 견고하게 방어하고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="팔로프 프레스(Pallof Press) 및 동적 코어 브레이싱"
    ),
    "dl5": DefectFeedbackDefinition(
        defect_code="dl5",
        defect_name="상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)",
        exercise_name="다이나믹 런지",
        target_joint="동적 착지 체간 시상면 경사",
        risk_weight=0.85,
        action_cue="동적 착지 시 전방 관성에 의해 상체가 앞으로 쏠리거나 젖혀지지 않도록 코어로 상체를 꼿꼿이 제동하세요.",
        biomechanical_reason="착지 순간 상체 굴곡/신전 통제에 실패하면 요추 및 무릎 관절에 충격 모멘텀이 직접적으로 전가됩니다.",
        caution_cue="착지 순간 상체의 전후 흔들림이 감지됩니다. 착지와 동시에 복압으로 상체를 단단히 잡아주세요.",
        normal_cue="동적 스텝 이동 및 착지 전 과정에서 상체 전후 밸런스를 완벽하게 통제하고 있습니다.",
        optimal_view="시점 B (우측 전방 45도) 또는 시점 D (좌측 전방 45도)",
        recommended_drill="스텝 착지 정지 드릴 및 팔로프 프레스"
    ),

    # -------------------------------------------------------------------------
    # [06. 사이드 런지: sl1, sl2, sl3, sl4, sl5]
    # -------------------------------------------------------------------------
    "sl1": DefectFeedbackDefinition(
        defect_code="sl1",
        defect_name="앞다리 무릎 각도 90도 (Lead Knee 90°)",
        exercise_name="사이드 런지",
        target_joint="외측 지지 슬관절",
        risk_weight=0.80,
        action_cue="체중을 실어 굽히는 다리의 엉덩이를 뒤로 빼면서 무릎이 90도가 되도록 깊숙이 앉으세요.",
        biomechanical_reason="외측 굴곡이 충분하지 않으면 내전근 신장 효과가 줄고 대퇴사두근에만 피로가 쌓입니다.",
        caution_cue="앉는 깊이가 다소 얕습니다. 엉덩이를 뒤로 더 보내며 깊이를 확보하세요.",
        normal_cue="지지 다리가 완벽한 90도를 형성하며 둔근과 햄스트링에 올바른 자극이 걸립니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="사이드 스쿼트 박스 딥스 드릴"
    ),
    "sl2": DefectFeedbackDefinition(
        defect_code="sl2",
        defect_name="몸통 발 무릎 방향 일치 여부 (Torso-Foot-Knee Alignment)",
        exercise_name="사이드 런지",
        target_joint="관상면 하지 정렬",
        risk_weight=0.85,
        action_cue="굽히는 무릎이 발끝 방향과 정확히 일치하도록 바깥으로 벌려주며 체중을 실으세요.",
        biomechanical_reason="외측 이동 시 무릎이 안쪽으로 꺾이면 내측 측부인대(MCL)에 강한 인장 스트레스가 발생합니다.",
        caution_cue="무릎이 발끝보다 안으로 모입니다. 엉덩이 바깥쪽을 써서 무릎을 열어주세요.",
        normal_cue="무릎과 발끝 축이 일직선으로 완벽히 정렬되어 외측 무릎 안정성이 탁월합니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="밴드 착용 사이드 워크 및 둔근 외회전 강화"
    ),
    "sl3": DefectFeedbackDefinition(
        defect_code="sl3",
        defect_name="안 굽히는 다리 구부정 / 신전 결함 (Trail Leg Knee Extension)",
        exercise_name="사이드 런지",
        target_joint="반대쪽 하지 신전",
        risk_weight=0.70,
        action_cue="반대쪽 뻗은 다리의 무릎을 완전히 쭉 펴고 발바닥 안쪽이 뜨지 않도록 바닥에 밀착시키세요.",
        biomechanical_reason="뻗는 다리가 굽혀지면 내전근(Adductors)의 정상적인 신장이 저해되고 양다리 하중 배분이 무너집니다.",
        caution_cue="뻗은 다리의 무릎이 살짝 굽습니다. 다리를 완전히 펴서 내전근을 늘려주세요.",
        normal_cue="뻗은 다리가 곧게 펴져 있어 뛰어난 유연성과 양측 대칭성을 보여줍니다.",
        optimal_view="시점 C (정면 0도)",
        recommended_drill="코삭 스쿼트(Cossack Squat) 및 내전근 동적 스트레칭"
    ),
    "sl4": DefectFeedbackDefinition(
        defect_code="sl4",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="사이드 런지",
        target_joint="척추/체간 굴곡",
        risk_weight=1.00,
        action_cue="체중을 옆으로 이동할 때 허리가 말리지 않도록 가슴을 펴고 척추 중립을 유지하세요.",
        biomechanical_reason="외측 힌지 중 요추 굴곡은 비대칭 회전 전단 하중을 유발하여 디스크에 손상을 줄 수 있습니다.",
        caution_cue="상체를 내릴 때 등이 다소 둥글어집니다. 가슴을 들고 코어에 힘을 유지하세요.",
        normal_cue="외측 체중 이동 중에도 척추 라인이 곧게 뻗어 매우 안정적입니다.",
        optimal_view="시점 B (우측 전방) 또는 시점 C (정면)",
        recommended_drill="굿모닝 힌지 패턴 훈련"
    ),
    "sl5": DefectFeedbackDefinition(
        defect_code="sl5",
        defect_name="상체의 과도한 숙임/젖힘 여부 (Torso Pitch/Lean)",
        exercise_name="사이드 런지",
        target_joint="체간 전방 경사각",
        risk_weight=0.75,
        action_cue="상체를 바닥으로 과도하게 숙이지 말고, 지지 정강이 각도와 나란한 경사도를 유지하세요.",
        biomechanical_reason="상체가 무릎 아래로 지나치게 쏠리면 고관절 모멘트암이 붕괴되어 허리 부담이 폭증합니다.",
        caution_cue="상체가 바닥 쪽으로 다소 숙여집니다. 가슴을 펴고 시선을 정면으로 유지하세요.",
        normal_cue="상체 경사도와 정강이 각도가 완벽한 평행을 이루고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="고블릿 사이드 런지(Goblet Side Lunge) 상체 세우기 인지"
    ),

    # -------------------------------------------------------------------------
    # [07. 크로스 런지: cl1, cl2, cl3]
    # -------------------------------------------------------------------------
    "cl1": DefectFeedbackDefinition(
        defect_code="cl1",
        defect_name="앞다리 무릎 각도 90도 (Lead Knee 90°)",
        exercise_name="크로스 런지",
        target_joint="앞다리 슬관절",
        risk_weight=0.80,
        action_cue="뒤로 교차해 내딛는 발의 거리를 충분히 두어 앞무릎이 90도까지 안정적으로 수직 하강하게 하세요.",
        biomechanical_reason="교차 보폭이 얕아 90도 굴곡이 안 나오면 타깃인 중둔근 편심성 수축 자극이 소실됩니다.",
        caution_cue="앞무릎 굴곡이 얕습니다. 뒤쪽으로 조금 더 멀리 내딛고 깊게 앉으세요.",
        normal_cue="앞무릎이 정확히 90도를 형성하여 둔근에 깊은 긴장감이 전달되고 있습니다.",
        optimal_view="시점 C (정면 0도) 또는 시점 B",
        recommended_drill="커트시 런지(Curtsy Lunge) 보폭 인지 연습"
    ),
    "cl2": DefectFeedbackDefinition(
        defect_code="cl2",
        defect_name="앞다리 무릎 안쪽 무너짐 (Lead Knee Valgus Collapse)",
        exercise_name="크로스 런지",
        target_joint="앞무릎 내측 안정성",
        risk_weight=0.90,
        action_cue="교차 동작 중 앞무릎이 안쪽으로 꺾이지 않도록 바깥쪽으로 단단히 버텨주세요.",
        biomechanical_reason="크로스 런지는 구조상 대퇴 내전 토크가 크므로, 무릎 외반 붕괴 시 전방십자인대 및 연골 손상 위험이 극대화됩니다.",
        caution_cue="하강 시 앞무릎이 안쪽으로 흔들립니다. 발바닥 바깥쪽에 힘을 주며 버티세요.",
        normal_cue="교차 부하 속에서도 앞무릎이 정면을 완벽히 지키며 굳건한 안정성을 보입니다.",
        optimal_view="시점 C (정면 0도 - 필수)",
        recommended_drill="밴드 착용 커트시 런지 및 중둔근 등척성 강화"
    ),
    "cl3": DefectFeedbackDefinition(
        defect_code="cl3",
        defect_name="교차 다리 궤적 이탈 / 과도한 교차 (Cross Alignment Deficit)",
        exercise_name="크로스 런지",
        target_joint="골반/교차 궤적",
        risk_weight=0.75,
        action_cue="뒤로 뻗는 다리를 과도하게 멀리 교차시키지 말고, 지지 발 대각선 뒤쪽 15~20도 정도로만 넘기세요.",
        biomechanical_reason="과도한 교차는 골반을 극단적으로 회전시켜 고관절 비구순 충돌과 천장관절 비틀림을 초래합니다.",
        caution_cue="다리가 너무 과하게 교차되었습니다. 교차 각도를 조금 줄여 골반 정면을 유지하세요.",
        normal_cue="교차 각도가 15~20도로 이상적이어서 골반의 비틀림 없이 둔근만 고립되고 있습니다.",
        optimal_view="시점 C (정면 0도) 또는 시점 A/E (후방)",
        recommended_drill="매트 대각선 기준선 스텝 훈련"
    ),

    # -------------------------------------------------------------------------
    # [08. 굿모닝: gm1, gm2, gm3]
    # -------------------------------------------------------------------------
    "gm1": DefectFeedbackDefinition(
        defect_code="gm1",
        defect_name="무릎 구부린 채 고정 위반 (Knee Fixed Angle Violation)",
        exercise_name="굿모닝",
        target_joint="슬관절 락킹",
        risk_weight=0.75,
        action_cue="무릎을 살짝만 굽힌 상태(약 15도)로 단단히 고정하고 스쿼트처럼 무릎을 더 굽히지 마세요.",
        biomechanical_reason="무릎이 계속 굽혀지면 순수 고관절 힌지 궤적이 무너져 햄스트링/둔근 자극이 분산됩니다.",
        caution_cue="상체를 숙이면서 무릎이 따라 굽혀집니다. 무릎을 단단히 고정하세요.",
        normal_cue="무릎 고정 각도가 완벽하게 유지되어 햄스트링에 순수한 장력이 집중되고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="벽 엉덩이 닿기 굿모닝 훈련"
    ),
    "gm2": DefectFeedbackDefinition(
        defect_code="gm2",
        defect_name="척추의 중립 위반 (Spine Neutrality)",
        exercise_name="굿모닝",
        target_joint="요추/흉추 기립근",
        risk_weight=1.00,
        action_cue="상체를 숙일 때 등이 굽지 않도록 가슴을 펴고 견갑골을 모아 척추 전체를 일직선으로 유지하세요.",
        biomechanical_reason="경추/흉추에 바벨을 얹은 상태에서 요추 굴곡은 지렛대 원리에 의해 척추 기립근에 치명적 과부하를 가합니다.",
        caution_cue="허리가 둥글게 말리기 시작합니다. 상체를 내려놓는 깊이를 줄이고 척추를 펴세요.",
        normal_cue="척추 중립선이 완벽하게 일직선을 이루며 코어가 상체 하중을 안전하게 지지합니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="PVC 파이프 등 밀착 3포인트(후두부-흉추-천골) 힌지 드릴"
    ),
    "gm3": DefectFeedbackDefinition(
        defect_code="gm3",
        defect_name="고관절 힌지 깊이 미흡 (Hip Hinge Depth Deficit)",
        exercise_name="굿모닝",
        target_joint="고관절 힌지 가동성",
        risk_weight=0.75,
        action_cue="엉덩이를 뒤 벽을 향해 충분히 밀어내어 상체가 지면과 거의 수평(약 15~25도 내외)이 될 때까지 내려가세요.",
        biomechanical_reason="힌지 가동범위가 부족하면 햄스트링의 완전 신장이 일어나지 않아 운동 유효 자극이 크게 감소합니다.",
        caution_cue="상체를 숙이는 깊이가 다소 얕습니다. 엉덩이를 조금 더 뒤로 멀리 보내세요.",
        normal_cue="충분한 고관절 힌지 깊이로 햄스트링과 둔근에 최대 신장 자극을 이끌어내고 있습니다.",
        optimal_view="시점 B (우측 전방 45도)",
        recommended_drill="밴드 저항 힙 힌지(Banded Hip Hinge) 가동성 훈련"
    )
}


# =============================================================================
# 2. 복합 결함 연쇄 인과관계 융합 규칙 (CAUSAL_CHAIN_RULES)
# =============================================================================

@dataclass
class CausalChainRule:
    """복합 결함 연쇄 진단 규칙"""
    exercise_id: int
    trigger_codes: Set[str]
    root_cause_title: str
    synthesized_cue: str
    biomechanical_context: str
    priority_boost: float = 0.20


CAUSAL_CHAIN_RULES: List[CausalChainRule] = [
    # 스쿼트: c4(발바닥 뜸) + c3(무릎 모임) -> 하지 연쇄 붕괴
    CausalChainRule(
        exercise_id=0,
        trigger_codes={"c3", "c4"},
        root_cause_title="지면 접지력 붕괴로 인한 무릎 외반 연쇄 (Kinetic Chain Collapse)",
        synthesized_cue="발바닥 삼각점(엄지/새끼/뒤꿈치)을 바닥에 강하게 밀착시키고, 무릎을 바깥쪽으로 활짝 열며 일어나세요.",
        biomechanical_context="발바닥 지지가 흔들려 족궁(Arch)이 무너지면서 정강이가 내회전되고, 결과적으로 무릎이 안쪽으로 함몰되는 전형적인 상행성 운동 사슬 결함입니다."
    ),
    # 스쿼트: c1(척추 중립 위반) + c2(고개 숙임/들림) -> 상체 연쇄 붕괴
    CausalChainRule(
        exercise_id=0,
        trigger_codes={"c1", "c2"},
        root_cause_title="시선 이탈로 인한 흉요추 연쇄 굴곡 (Cervico-Thoracic Chain)",
        synthesized_cue="시선을 정면 2~3m 앞 바닥에 고정하고 가슴을 들어 척추 전체를 곧게 펴세요.",
        biomechanical_context="고개가 아래로 떨어지면서 상부 흉추와 요추가 연쇄적으로 둥글게 말려 상체 안정성이 소실되었습니다."
    ),
    # 데드리프트: d3(힙 슈트업) + d1(척추 중립 위반) -> 레버리지 악화
    CausalChainRule(
        exercise_id=1,
        trigger_codes={"d1", "d3"},
        root_cause_title="골반 선행 들림으로 인한 요추 부하 전가 (Hip-Shoot Up Leverage Deficit)",
        synthesized_cue="엉덩이만 먼저 들리지 않도록 발바닥으로 바닥을 밀며 상하체를 한 덩어리로 동시에 일으키세요.",
        biomechanical_context="골반이 먼저 상승하면서 무릎 각도가 펴져 대퇴사두근 힘을 잃고, 모든 하중이 허리 기립근으로 집중되었습니다."
    ),
    # 데드리프트: d4(신전 불일치) + d5(바벨 이격) -> 궤적 이탈
    CausalChainRule(
        exercise_id=1,
        trigger_codes={"d4", "d5"},
        root_cause_title="바벨 이격으로 인한 신전 타이밍 지연 (Barbell Distance Desync)",
        synthesized_cue="바벨을 정강이에 바짝 밀착시키고, 무릎을 스치는 순간 고관절을 앞으로 강하게 넣어 잠그세요.",
        biomechanical_context="바벨이 몸에서 멀어지면서 무게중심이 앞으로 쏠려 골반의 완전 신장이 지연되고 허리에 불필요한 모멘트가 발생했습니다."
    ),
    # 스티프 데드리프트: sd1(무릎 가변) + sd2(허리 굽힘) -> 힌지 실패
    CausalChainRule(
        exercise_id=2,
        trigger_codes={"sd1", "sd2"},
        root_cause_title="무릎 락킹 해제로 인한 요추 굴곡 보상 (Compensatory Lumbar Flexion)",
        synthesized_cue="무릎을 15도 살짝 굽힌 채 락(Lock)을 걸고, 엉덩이만 뒤로 멀리 보내며 척추를 평평하게 펴세요.",
        biomechanical_context="무릎이 흔들리며 햄스트링 장력이 풀리자 가동범위를 채우기 위해 허리를 둥글게 구부리는 보상작용이 일어났습니다."
    ),
    # 바벨 런지: bl1(앞무릎 각도) + bl2(방향 불일치) -> 슬관절 축 비틀림
    CausalChainRule(
        exercise_id=3,
        trigger_codes={"bl1", "bl2"},
        root_cause_title="보폭 부족으로 인한 슬관절 회전 비틀림 (Short Stride Knee Torsion)",
        synthesized_cue="앞발을 반 보 더 멀리 내딛고, 무릎이 두 번째 발가락을 똑바로 향하도록 정면 축을 유지하세요.",
        biomechanical_context="보폭이 좁아 앞무릎에 하중이 쏠리며 중심을 잃고 무릎이 안쪽으로 꺾이는 복합 결함입니다."
    ),
    # 바벨 런지: bl4(척추 중립) + bl5(골반 비대칭) -> 코어-골반 연동 붕괴
    CausalChainRule(
        exercise_id=3,
        trigger_codes={"bl4", "bl5"},
        root_cause_title="골반 수평 붕괴로 인한 체간 측면 기울임 (Pelvic Tilt Core Collapse)",
        synthesized_cue="양 골반 높이를 수평으로 맞추고 복압을 단단히 잡아 상체를 수직으로 곧게 세우세요.",
        biomechanical_context="중둔근 지지력이 약해 골반이 한쪽으로 기울어지면서 체간 척추가 측면으로 보상 굴곡되었습니다."
    ),
    # 다이나믹 런지: dl1(앞무릎 각도) + dl2(무릎 방향) -> 동적 착지 불안
    CausalChainRule(
        exercise_id=4,
        trigger_codes={"dl1", "dl2"},
        root_cause_title="동적 착지 감속 실패로 인한 슬관절 붕괴 (Deceleration Failure)",
        synthesized_cue="보폭을 충분히 확보하여 발바닥 전체로 부드럽게 착지하며 무릎이 안으로 흔들리지 않게 버티세요.",
        biomechanical_context="내딛는 보폭이 좁고 감속 조절이 안 되어 착지 충격력이 무릎 내측으로 직격했습니다."
    ),
    # 사이드 런지: sl1(앞무릎 90도) + sl3(뻗는다리 굽힘) -> 중심 이동 미흡
    CausalChainRule(
        exercise_id=5,
        trigger_codes={"sl1", "sl3"},
        root_cause_title="신전 다리 굽힘으로 인한 체중 이동 실패 (Incomplete Lateral Shift)",
        synthesized_cue="뻗는 다리의 무릎을 완전히 펴서 지면을 지지하고, 굽히는 다리의 엉덩이를 뒤로 깊게 접으세요.",
        biomechanical_context="반대쪽 다리가 굽혀져 체중이 지지 다리로 충분히 실리지 못하고 어정쩡한 깊이에서 멈췄습니다."
    ),
    # 크로스 런지: cl1(앞무릎 90도) + cl2(무릎 모임) -> 교차 외반 위험
    CausalChainRule(
        exercise_id=6,
        trigger_codes={"cl1", "cl2"},
        root_cause_title="과도한 하강으로 인한 앞무릎 외반 급증 (Severe Valgus on Depth)",
        synthesized_cue="앞무릎이 안으로 꺾이지 않도록 바깥쪽으로 강하게 지탱하고 안정된 깊이까지만 하강하세요.",
        biomechanical_context="교차 밸런스를 잃은 상태에서 깊이 내려가려다 앞무릎 인대에 가장 위험한 외반 스트레스가 걸렸습니다."
    ),
    # 굿모닝: gm1(무릎 굽힘) + gm2(척추 중립 위반) -> 힌지 패턴 상실
    CausalChainRule(
        exercise_id=7,
        trigger_codes={"gm1", "gm2"},
        root_cause_title="무릎 굴곡으로 인한 힙 힌지 해제 및 요추 부하 (Hinge Breakdown to Squat)",
        synthesized_cue="무릎을 15도로 단단히 고정하고 가슴을 펴 엉덩이만 뒤로 밀어내세요. 스쿼트처럼 무릎을 굽히지 마세요.",
        biomechanical_context="무릎이 따라 굽혀지며 둔근 힌지가 풀리고, 상체 바벨 하중이 허리 요추로 완전히 쏟아졌습니다."
    )
]


# =============================================================================
# 3. 3단계 피드백 엔진 코어 클래스 (FeedbackEngine)
# =============================================================================

class FeedbackEngine:
    """
    3단계 자세 교정 및 결함 피드백 싱글톤 엔진
    - 33종 결함 전수 매핑
    - 우선순위 큐 기반 1~2개 핵심 큐 선별
    - 복합 결함 연쇄 추론
    - 5대 카메라 시점 적응형 안내
    - 세트 종합 맞춤 처방전 생성
    """

    _instance = None

    def __init__(self):
        self.catalog = FEEDBACK_CATALOG
        self.causal_rules = CAUSAL_CHAIN_RULES

    @classmethod
    def get_instance(cls) -> "FeedbackEngine":
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_feedback_def(self, defect_code: str) -> Optional[DefectFeedbackDefinition]:
        """결함 코드 기반 피드백 정의 반환"""
        return self.catalog.get(defect_code.lower())

    # -------------------------------------------------------------------------
    # 3.1 단일 결함 텍스트 포맷팅 (3-State 차등 가이드)
    # -------------------------------------------------------------------------
    def format_single_feedback(
        self,
        defect_code: str,
        state: int,
        prob: float = 0.0,
        view_code: str = "view3"
    ) -> str:
        """
        단일 결함에 대한 3-State 대응 문구 생성
        - State 0 (정상): 칭찬 및 폼 유지
        - State 1 (결함): 즉각적 핵심 행동 지침 큐
        - State 2 (주의/보류): 미세 이탈 경고 및 재촬영 권장
        """
        item = self.get_feedback_def(defect_code)
        if not item:
            return "올바른 자세를 유지하며 천천히 수행하세요."

        if state == 1:
            # 결함 발생 시: 해당 결함을 고치고 정상 동작으로 가기 위한 필수 조치만 명확히 반환
            return f"[교정 조치] {item.corrective_action}"
        elif state == 2:
            # 주의/보류: 가이드 안내
            return f"[주의] {item.caution_cue}"
        else:
            # 정상: 정상 동작 확인
            return "정상 동작입니다."

    # -------------------------------------------------------------------------
    # 3.2 단일 횟수(1-Rep) 다중 결함 우선순위 해결 및 큐잉
    # -------------------------------------------------------------------------
    def prioritize_rep_defects(
        self,
        exercise_id: int,
        defect_predictions: List[Dict[str, Any]],
        view_code: str = "view3"
    ) -> Dict[str, Any]:
        """
        단일 반복(1-Rep)에서 검출된 결함들을 생체역학 위험도 기준으로 정렬하고
        사용자 인지 한계에 맞춰 최상위 1~2개 행동 큐 및 연쇄 인과관계를 합성
        """
        detected_defects = []
        cautions = []
        clean_count = 0
        defect_codes_present = set()

        for d in defect_predictions:
            code = d.get('defect_code', '').lower()
            state = int(d.get('state', 0))
            prob = float(d.get('probability', 0.0))
            item = self.get_feedback_def(code)

            if not item:
                continue

            if state == 1:
                # 결함 발생: 위험 점수 산정 = risk_weight * prob
                score = round(item.risk_weight * max(0.5, prob), 4)
                detected_defects.append({
                    'code': code,
                    'name': item.defect_name,
                    'item': item,
                    'score': score,
                    'prob': prob,
                    'state': state
                })
                defect_codes_present.add(code)
            elif state == 2:
                cautions.append({
                    'code': code,
                    'name': item.defect_name,
                    'item': item,
                    'score': round(item.risk_weight * 0.4, 4),
                    'prob': prob,
                    'state': state
                })
            else:
                clean_count += 1

        # 1. 완전 정상 반복인 경우
        if not detected_defects and not cautions:
            return {
                'overall_status': '정상',
                'is_clean': True,
                'primary_action_cue': "완벽한 자세입니다! 대칭성과 관절 정렬이 이상적입니다.",
                'primary_detail': "신체 중심선(COM)이 지면 지지면 내에 안정적으로 유지되고 있으며 관절 부하 분산이 훌륭합니다.",
                'secondary_cues': [],
                'causal_chain_detected': None,
                'view_tip': None
            }

        # 2. 연쇄 결함 인과관계 검출 (Causal Chaining)
        matched_chain = None
        for rule in self.causal_rules:
            if rule.exercise_id == exercise_id and rule.trigger_codes.issubset(defect_codes_present):
                matched_chain = rule
                break

        # 3. 위험 점수 내림차순 정렬 (Priority Queue)
        detected_defects.sort(key=lambda x: x['score'], reverse=True)

        primary_action_cue = ""
        primary_detail = ""
        secondary_cues: List[str] = []

        if matched_chain:
            # 복합 연쇄 결함이 검출된 경우: 합성된 큐를 최우선 적용
            primary_action_cue = f"[핵심 원인 교정] {matched_chain.synthesized_cue}"
            primary_detail = f"{matched_chain.root_cause_title}: {matched_chain.biomechanical_context}"
            # 보조 큐는 연쇄에 포함되지 않은 다른 상위 결함이 있으면 1개 추가
            for d in detected_defects:
                if d['code'] not in matched_chain.trigger_codes:
                    secondary_cues.append(f"[{d['item'].target_joint}] {d['item'].action_cue}")
                    break
        elif detected_defects:
            # 단독 결함 기반 최상위 1개 선정
            top_1 = detected_defects[0]
            primary_action_cue = f"[{top_1['item'].target_joint}] {top_1['item'].action_cue}"
            primary_detail = f"{top_1['name']}: {top_1['item'].biomechanical_reason}"

            # 2순위 결함이 있으면 보조 큐로 1개만 추가
            if len(detected_defects) > 1:
                top_2 = detected_defects[1]
                secondary_cues.append(f"[{top_2['item'].target_joint}] {top_2['item'].action_cue}")
        elif cautions:
            # 결함은 없고 주의/보류만 있는 경우
            c_top = cautions[0]
            primary_action_cue = f"[자세 주의] {c_top['item'].caution_cue}"
            primary_detail = f"미세한 자세 불안정이 감지되었습니다. 다음 반복에서는 {c_top['item'].target_joint} 부위의 긴장을 유지하세요."

        # 4. 시점 권장 팁 (현재 시점과 최적 시점 비교)
        view_tip = None
        if detected_defects:
            best_v = detected_defects[0]['item'].optimal_view
            # 시점 C인데 시점 B 권장이거나 반대인 경우 안내
            if "시점 C" in best_v and view_code != "view3":
                view_tip = f"정밀한 좌우 대칭 및 관절 정렬 진단을 위해 '시점 C (정면)'에서 재촬영을 권장합니다."
            elif "시점 B" in best_v and view_code == "view3":
                view_tip = f"척추 만곡 및 전후 모멘트암의 정밀 진단을 위해 '시점 B (우측 전방 45도)' 촬영을 권장합니다."

        return {
            'overall_status': '결함 검출' if detected_defects else '주의/보류',
            'is_clean': False,
            'primary_action_cue': primary_action_cue,
            'primary_detail': primary_detail,
            'secondary_cues': secondary_cues,
            'causal_chain_detected': matched_chain.root_cause_title if matched_chain else None,
            'view_tip': view_tip
        }

    # -------------------------------------------------------------------------
    # 3.3 다중 세트(Multi-Rep) 종합 맞춤 처방전 생성
    # -------------------------------------------------------------------------
    def generate_set_summary(
        self,
        exercise_id: int,
        exercise_name: str,
        rep_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        운동 세트 전체(예: 5회 반복)를 종합 분석하여
        가장 빈번하게 발생한 취약 관절과 맞춤형 개선 처방전을 도출
        """
        total_reps = len(rep_results)
        if total_reps == 0:
            return {
                'total_reps': 0,
                'clean_reps': 0,
                'clean_ratio_percent': 0.0,
                'summary_title': "운동 분석 데이터 없음",
                'most_frequent_defect': None,
                'overall_prescription': "분석된 유효 반복 구간이 없습니다.",
                'recommended_drills': []
            }

        defect_frequency: Dict[str, int] = {}
        clean_reps = 0

        for r in rep_results:
            is_clean = r.get('is_clean', False)
            if is_clean:
                clean_reps += 1
            defects = r.get('defects', [])
            for d in defects:
                if d.get('state') == 1:
                    code = d.get('defect_code', '').lower()
                    defect_frequency[code] = defect_frequency.get(code, 0) + 1

        clean_ratio = round((clean_reps / total_reps) * 100.0, 1)

        # 가장 빈번한 결함 도출
        most_freq_code = None
        most_freq_count = 0
        if defect_frequency:
            most_freq_code = max(defect_frequency, key=defect_frequency.get)
            most_freq_count = defect_frequency[most_freq_code]

        drills: List[str] = []
        if most_freq_code:
            top_item = self.get_feedback_def(most_freq_code)
            if top_item:
                drills.append(top_item.recommended_drill)
                summary_title = f"{exercise_name} 세트 진단: '{top_item.target_joint}' 집중 교정 필요"
                prescription = (
                    f"총 {total_reps}회 중 {most_freq_count}회에서 '{top_item.defect_name}'이(가) 지속적으로 관찰되었습니다. "
                    f"피로도가 누적될수록 {top_item.target_joint}에 부담이 가중되므로, "
                    f"다음 세트에서는 {top_item.action_cue} "
                    f"보강 운동으로 '{top_item.recommended_drill}'을 권장합니다."
                )
            else:
                summary_title = f"{exercise_name} 세트 진단 완료"
                prescription = "전반적으로 양호하나 일부 결함이 관찰되었습니다."
        else:
            summary_title = f"{exercise_name} 완벽한 세트 수행 달성!"
            prescription = f"총 {total_reps}회 반복 모두에서 치명적 결함 없이 완벽한 생체역학적 폼을 유지하셨습니다. 현재의 템포와 척추 중립을 유지하세요."
            drills.append("현재 중량 유지 및 안정적 점진적 과부하")

        return {
            'total_reps': total_reps,
            'clean_reps': clean_reps,
            'clean_ratio_percent': clean_ratio,
            'summary_title': summary_title,
            'most_frequent_defect': {
                'code': most_freq_code,
                'count': most_freq_count,
                'name': self.get_feedback_def(most_freq_code).defect_name if most_freq_code else None
            } if most_freq_code else None,
            'overall_prescription': prescription,
            'recommended_drills': drills
        }


# 모듈 레벨 헬퍼 함수
def get_feedback_engine() -> FeedbackEngine:
    """FeedbackEngine 싱글톤 헬퍼"""
    return FeedbackEngine.get_instance()

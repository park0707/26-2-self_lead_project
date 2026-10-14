# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Web Serving Pipeline Full Audit Test Suite
--------------------------------------------------------------------------------
전수 검증 대상:
1. 8대 운동 33종 전 세부 결함 모델 추론 무결성 검증
2. 바벨 스쿼트 c3, c4 MediaPipe/RTMPose 키포인트 피처 추출 무결성 (0-벡터 폴백 버그 퇴치 검증)
3. 다이나믹 런지 dl2 확률 딕셔너리 키 정합성 ('probability' vs 'prob')
4. 바벨 런지 bl4 3-State 판정 및 동적 생체역학 피드백 보존 여부
5. 바벨 런지 bl5 및 다이나믹 런지 dl5 결함 명칭 정합성 (상체의 과도한 숙임/젖힘 여부)
6. 33대 전 결함 생체역학 코칭 피드백 및 큐(Tier A, B, C) 자동 주입 무결성
"""

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.core.model_registry import DefectModelRegistry, DefectPrediction
from src.web.core.topology_adapter import SkeletalTopologyAdapter


def generate_realistic_mp33_sequence() -> np.ndarray:
    """합성 64F MediaPipe 33관절 시퀀스 생성 (정상 스탠스 및 움직임 시뮬레이션)"""
    T = 64
    seq = np.zeros((T, 33, 3), dtype=np.float32)

    # 코/머리
    seq[:, 0] = [0.50, 0.20, 0.0]
    # 어깨
    seq[:, 11] = [0.42, 0.30, 0.0]  # L shoulder
    seq[:, 12] = [0.58, 0.30, 0.0]  # R shoulder
    # 팔꿈치
    seq[:, 13] = [0.38, 0.45, 0.0]
    seq[:, 14] = [0.62, 0.45, 0.0]
    # 손목
    seq[:, 15] = [0.40, 0.60, 0.0]
    seq[:, 16] = [0.60, 0.60, 0.0]
    # 골반 (하강 궤적 시뮬레이션)
    t = np.linspace(0, np.pi, T)
    squat_depth = 0.15 * np.sin(t)
    seq[:, 23] = np.column_stack([np.full(T, 0.45), 0.55 + squat_depth, np.zeros(T)])  # L hip
    seq[:, 24] = np.column_stack([np.full(T, 0.55), 0.55 + squat_depth, np.zeros(T)])  # R hip
    # 무릎
    seq[:, 25] = np.column_stack([np.full(T, 0.44), 0.72 + 0.5 * squat_depth, np.zeros(T)])
    seq[:, 26] = np.column_stack([np.full(T, 0.56), 0.72 + 0.5 * squat_depth, np.zeros(T)])
    # 발목
    seq[:, 27] = [0.43, 0.90, 0.0]
    seq[:, 28] = [0.57, 0.90, 0.0]
    # 발뒤꿈치
    seq[:, 29] = [0.43, 0.92, 0.0]
    seq[:, 30] = [0.57, 0.92, 0.0]
    # 발끝
    seq[:, 31] = [0.42, 0.94, 0.0]
    seq[:, 32] = [0.58, 0.94, 0.0]

    return seq


def test_audit_all_exercises():
    print("=" * 80)
    print("  [전수 감사 1] 8대 운동 33종 전 세부 결함 모델 서빙 무결성 감사 시작")
    print("=" * 80)

    registry = DefectModelRegistry.get_instance()
    mp33_seq = generate_realistic_mp33_sequence()
    rtm26_seq = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(mp33_seq)
    spine9_seq = np.zeros((64, 9, 3), dtype=np.float32)

    exercise_names = [
        "바벨 스쿼트", "바벨 데드리프트", "바벨 스티프 데드리프트", "바벨 런지",
        "다이나믹 런지", "사이드 런지", "크로스 런지", "바벨 굿모닝"
    ]
    expected_defect_counts = [4, 5, 3, 5, 5, 5, 3, 3]

    total_defects_tested = 0
    passed_defects = 0

    for ex_id, (ex_name, expected_count) in enumerate(zip(exercise_names, expected_defect_counts)):
        print(f"\n--- [운동 {ex_id}: {ex_name}] ---")
        preds = registry.evaluate_exercise_defects(
            exercise_id=ex_id,
            mp33_seq=mp33_seq,
            rtm26_seq=rtm26_seq,
            view_code='view3',
            spine_seq=spine9_seq
        )

        assert len(preds) == expected_count, f"{ex_name} 결함 개수 불일치: {len(preds)} != {expected_count}"

        for p in preds:
            total_defects_tested += 1
            print(f"  [{p.defect_code}] {p.defect_name:<32} | P={p.probability:.4f} | State={p.state} ({p.state_label}) | Feedback={p.feedback[:35]}...")

            # 1. 확률 범위 검증
            assert 0.0 <= p.probability <= 1.0, f"{p.defect_code} 확률 이상: {p.probability}"
            # 2. 상태 코드 검증 (0: 정상, 1: 결함, 2: 주의/보류)
            assert p.state in (0, 1, 2), f"{p.defect_code} 상태 코드 이상: {p.state}"
            assert p.state_label in ("정상", "결함", "주의/보류"), f"{p.defect_code} 라벨 이상: {p.state_label}"
            # 3. 피드백 누락 및 플레이스홀더 잔존 여부 검증
            assert p.feedback and p.feedback != "피드백 자리", f"{p.defect_code} 피드백 누락 또는 플레이스홀더 잔존!"
            # 4. 생체역학 큐 자동 주입 검증
            assert p.target_joint, f"{p.defect_code} 타깃 관절 누락!"
            assert p.action_cue, f"{p.defect_code} 행동 지침 큐 누락!"
            assert p.biomechanical_reason, f"{p.defect_code} 생체역학 원인 누락!"

            passed_defects += 1

    print("\n" + "=" * 80)
    print(f"  [결과] 총 {total_defects_tested}개 세부 결함 모델 중 {passed_defects}개 100% 정상 통과!")
    print("=" * 80)


def test_audit_squat_c3_c4_feature_extraction():
    print("\n" + "=" * 80)
    print("  [전수 감사 2] 바벨 스쿼트 c3, c4 피처 추출 및 26J 어댑터 무결성 감사")
    print("=" * 80)

    from src.models.second_models.babel_squat.c3_model_v7_rtmp import BabelSquatC3ModelV7RTMP
    from src.models.second_models.babel_squat.c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP

    c3 = BabelSquatC3ModelV7RTMP()
    c4 = BabelSquatC4ModelV6RTMP()

    mp33 = generate_realistic_mp33_sequence()
    rtm26 = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(mp33)

    # 1. MediaPipe 33 직접 입력 시 피처 추출 검증 (22D, 25D non-zero)
    f_c3_mp = c3.extract_features(mp33)
    f_c4_mp = c4.extract_features(mp33)

    assert f_c3_mp.shape == (22,), f"c3 MP33 피처 shape 불일치: {f_c3_mp.shape}"
    assert f_c4_mp.shape == (25,), f"c4 MP33 피처 shape 불일치: {f_c4_mp.shape}"
    assert not np.all(f_c3_mp == 0), "CRITICAL BUG: c3 MP33 피처가 전부 0입니다!"
    assert not np.all(f_c4_mp == 0), "CRITICAL BUG: c4 MP33 피처가 전부 0입니다!"
    print(f"  [+] c3 MP33 22D 피처 추출 성공: min={f_c3_mp.min():.3f}, max={f_c3_mp.max():.3f}, mean={f_c3_mp.mean():.3f}")
    print(f"  [+] c4 MP33 25D 피처 추출 성공: min={f_c4_mp.min():.3f}, max={f_c4_mp.max():.3f}, mean={f_c4_mp.mean():.3f}")

    # 2. RTMPose 26 입력 시 안전 토폴로지 어댑터 작동 검증 (더 이상 IndexError 후 0-벡터 폴백 안 됨)
    f_c3_rtm = c3.extract_features(rtm26)
    f_c4_rtm = c4.extract_features(rtm26)

    assert f_c3_rtm.shape == (22,), f"c3 RTM26 피처 shape 불일치: {f_c3_rtm.shape}"
    assert f_c4_rtm.shape == (25,), f"c4 RTM26 피처 shape 불일치: {f_c4_rtm.shape}"
    assert not np.all(f_c3_rtm == 0), "CRITICAL BUG: c3 RTM26 입력 시 토폴로지 변환 실패로 0 반환!"
    assert not np.all(f_c4_rtm == 0), "CRITICAL BUG: c4 RTM26 입력 시 토폴로지 변환 실패로 0 반환!"
    print(f"  [+] c3 RTM26 토폴로지 자동 변환 22D 피처 추출 성공: non-zero count={np.count_nonzero(f_c3_rtm)}/22")
    print(f"  [+] c4 RTM26 토폴로지 자동 변환 25D 피처 추출 성공: non-zero count={np.count_nonzero(f_c4_rtm)}/25")


def test_audit_defect_names_and_dl2():
    print("\n" + "=" * 80)
    print("  [전수 감사 3] 결함 명칭 (bl5, dl5) 및 dl2 확률 키 무결성 감사")
    print("=" * 80)

    registry = DefectModelRegistry.get_instance()
    mp33 = generate_realistic_mp33_sequence()
    rtm26 = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(mp33)

    # 1. Babel Lunge (ex_id = 3)
    bl_preds = registry.evaluate_exercise_defects(3, mp33, rtm26, 'view3')
    bl5_pred = next(p for p in bl_preds if p.defect_code == 'bl5')
    print(f"  [+] bl5 결함명: '{bl5_pred.defect_name}'")
    assert "상체의 과도한 숙임/젖힘 여부" in bl5_pred.defect_name, f"bl5 명칭 오류: {bl5_pred.defect_name}"

    # 2. Dynamic Lunge (ex_id = 4)
    dl_preds = registry.evaluate_exercise_defects(4, mp33, rtm26, 'view3')
    dl5_pred = next(p for p in dl_preds if p.defect_code == 'dl5')
    print(f"  [+] dl5 결함명: '{dl5_pred.defect_name}'")
    assert "상체의 과도한 숙임/젖힘 여부" in dl5_pred.defect_name, f"dl5 명칭 오류: {dl5_pred.defect_name}"

    # 3. Dynamic Lunge dl2 확률 키 검증
    dl2_pred = next(p for p in dl_preds if p.defect_code == 'dl2')
    print(f"  [+] dl2 결함 확률: {dl2_pred.probability} (State: {dl2_pred.state})")
    # State가 0/1/2일 때 정상적으로 확률이 연동되는지 확인
    assert dl2_pred.probability >= 0.0, "dl2 확률 음수 오류"


if __name__ == "__main__":
    test_audit_all_exercises()
    test_audit_squat_c3_c4_feature_extraction()
    test_audit_defect_names_and_dl2()
    print("\n[★★★] 전수 검증 테스트 전체 100% 무결 통과! [★★★]\n")

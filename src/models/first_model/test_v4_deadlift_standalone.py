# =========================================================================================================
# [스크립트명]: test_v4_deadlift_standalone.py
# [위치]: src/models/first_model/test_v4_deadlift_standalone.py
# [목적]: 
#   - Stage 1(6대 상위 그룹 앙상블)을 건너뛰고, 데드리프트 패밀리(02. 바벨 vs 03. 스티프)만 추출하여
#     Stage 2-A 독립 2중 캐스케이드 모델의 순수 판별 성능을 1~2초 만에 초고속으로 검증하는 경량 실험기
# [실행 방법]:
#   - 루트 디렉토리(d:\)에서: python -m src.models.first_model.test_v4_deadlift_standalone
#   - 또는 직접 실행: python src/models/first_model/test_v4_deadlift_standalone.py
# =========================================================================================================

import os
import sys
import time
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# 한글 출력 인코딩 보정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 루트 경로 등록
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.models.first_model.train_hierarchical_kfold import HierarchicalKFoldPipelineTrainer
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.deadlift_sub_classifier import DeadliftSubClassifier

def evaluate_level_subset(
    X_dead_all: np.ndarray,
    y_dead_all: np.ndarray,
    subjs_dead_all: np.ndarray,
    views_dead_all: np.ndarray,
    levels_all: np.ndarray,
    max_level: int,
    level_title: str,
    tau_threshold: float = 0.50,
    conf_high: float = 0.60,
    conf_low: float = 0.40,
    folds_config: list = None,
    sids_dead_all: list = None
):
    print("\n" + "=" * 105)
    print(f"  ★ [실험 섹션] {level_title}")
    print("=" * 105)

    lvl_mask = (levels_all <= max_level)
    X_sub = X_dead_all[lvl_mask]
    y_sub = y_dead_all[lvl_mask]
    s_sub = subjs_dead_all[lvl_mask]
    v_sub = views_dead_all[lvl_mask]
    sids_sub = np.array(sids_dead_all)[lvl_mask] if sids_dead_all is not None else None

    tot_b = int(np.sum(y_sub == 0))
    tot_s = int(np.sum(y_sub == 1))
    print(f"  • 유효 데이터 규모 : 총 {len(y_sub):,}개 (바벨 데드: {tot_b:,}개, 스티프 데드: {tot_s:,}개)")
    print(f"  • 결정 임계값      : tau={tau_threshold:.2f} | 1차 확정 신뢰도: High={conf_high:.2f}, Low={conf_low:.2f}")
    print("-" * 105)

    t_sec = time.time()
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fc in folds_config:
        t_f = time.time()
        tr_m = np.isin(s_sub, fc['train_subjs']) & np.isin(v_sub, fc['train_views'])
        te_m = np.isin(s_sub, fc['test_subjs']) & np.isin(v_sub, fc['test_views'])

        X_tr, y_tr, s_tr = X_sub[tr_m], y_sub[tr_m], s_sub[tr_m]
        X_te, y_te, s_te = X_sub[te_m], y_sub[te_m], s_sub[te_m]
        sid_tr = sids_sub[tr_m] if sids_sub is not None else None
        sid_te = sids_sub[te_m] if sids_sub is not None else None

        if len(y_te) == 0:
            continue

        clf = DualCascadeDeadliftClassifier(
            conf_thresh_high=conf_high,
            conf_thresh_low=conf_low,
            stage2_decision_threshold=tau_threshold
        )
        clf.fit(X_tr, y_tr, subjects=s_tr, sequence_ids=sid_tr)

        probs = clf.predict_proba(X_te, sequence_ids=sid_te)[:, 1]
        preds = (probs >= tau_threshold).astype(int)

        f_rec = recall_score(y_te, preds, pos_label=1, zero_division=0) * 100
        f_prec = precision_score(y_te, preds, pos_label=1, zero_division=0) * 100
        f_f1 = f1_score(y_te, preds, pos_label=1, zero_division=0) * 100
        rt = clf.routing_stats

        all_y_true.extend(y_te)
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)

        print(f"    ✔ {fc['name']} (소요: {time.time()-t_f:.2f}초) -> 스티프 Prec: {f_prec:5.2f}%, Rec: {f_rec:5.2f}%, F1: {f_f1:5.2f}% | 1차 확정: {rt['barbell_fast']+rt['stiff_fast']}개, 2차 위임: {rt['routed_to_stage2']}개")

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    stiff_rec = recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0) * 100
    stiff_prec = precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0) * 100
    stiff_f1 = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0) * 100

    barbell_rec = recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0) * 100
    barbell_prec = precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0) * 100
    barbell_f1 = f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0) * 100

    try:
        auc = roc_auc_score(y_true_arr, y_prob_arr)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    elapsed = time.time() - t_sec

    print("-" * 105)
    print(f"  • {level_title} 최종 성능 (소요: {elapsed:.2f}초, ROC-AUC: {auc:.4f}):")
    print(f"    - 02. 바벨 데드리프트 (샘플 {np.sum(y_true_arr==0):>4d}개) : 정밀도 {barbell_prec:6.2f}% | 재현율 {barbell_rec:6.2f}% | ★ F1 {barbell_f1:6.2f}%")
    print(f"    - 03. 스티프 데드리프트 (샘플 {np.sum(y_true_arr==1):>4d}개) : 정밀도 {stiff_prec:6.2f}% | 재현율 {stiff_rec:6.2f}% | ★ F1 {stiff_f1:6.2f}%")
    print(f"    - 2x2 혼동 행렬: [실제 바벨 -> 바벨:{cm[0,0]:>4d}, 스티프:{cm[0,1]:>4d}] | [실제 스티프 -> 바벨:{cm[1,0]:>4d}, 스티프:{cm[1,1]:>4d}]")

    return {
        'title': level_title,
        'samples': len(y_true_arr),
        'barbell_cnt': int(np.sum(y_true_arr==0)),
        'stiff_cnt': int(np.sum(y_true_arr==1)),
        'barbell_prec': barbell_prec,
        'barbell_rec': barbell_rec,
        'barbell_f1': barbell_f1,
        'stiff_prec': stiff_prec,
        'stiff_rec': stiff_rec,
        'stiff_f1': stiff_f1,
        'auc': auc,
        'elapsed': elapsed,
        'cm': cm
    }

def main():
    print("\n" + "=" * 105)
    print("  ★ [NovelFlow] Stage 2-A 데드리프트 초고속 비교 검증기 (Level 1 전용 vs Level 1~3 전용)")
    print("  ★ 3-Fold Out-of-Domain (피험자 3그룹 x 시점 3그룹) 전수 평가")
    print("=" * 105)

    t_total = time.time()

    # 1. 데이터 로드
    print("[*] 3D 생체역학 마스터 데이터셋 로드 중...", flush=True)
    trainer = HierarchicalKFoldPipelineTrainer()
    y_action = trainer.master_data['y_action']
    subjs = trainer.master_data['subjects']
    views = trainer.master_data['views']
    sids = trainer.master_data['sequence_ids']
    X_b = trainer.master_data['X_Format_B']

    dead_mask = (y_action == 1) | (y_action == 2)
    X_dead_all = X_b[dead_mask]
    y_dead_all = (y_action[dead_mask] == 2).astype(int)
    subjs_dead_all = subjs[dead_mask]
    views_dead_all = views[dead_mask]
    sids_dead_all = [str(s) for s in sids[dead_mask]]

    # 생체역학 14D 피처 추출 및 Level 산정
    feats_14d = DualCascadeDeadliftClassifier.slice_features(X_dead_all)
    levels_all = DualCascadeDeadliftClassifier.determine_levels(feats_14d, y_dead_all)

    folds_config = [
        {
            'fold_id': 1,
            'name': 'Fold 1 [Test: 피험자 1~3 x 전방(0°,45°)]',
            'test_subjs': [1, 2, 3],
            'test_views': ['view1', 'view2'],
            'train_subjs': [4, 5, 6, 7, 8, 9],
            'train_views': ['view3', 'view4', 'view5']
        },
        {
            'fold_id': 2,
            'name': 'Fold 2 [Test: 피험자 4~6 x 측면(90°)]',
            'test_subjs': [4, 5, 6],
            'test_views': ['view3'],
            'train_subjs': [1, 2, 3, 7, 8, 9],
            'train_views': ['view1', 'view2', 'view4', 'view5']
        },
        {
            'fold_id': 3,
            'name': 'Fold 3 [Test: 피험자 7~9 x 후방(135°,180°)]',
            'test_subjs': [7, 8, 9],
            'test_views': ['view4', 'view5'],
            'train_subjs': [1, 2, 3, 4, 5, 6],
            'train_views': ['view1', 'view2', 'view3']
        }
    ]

    # 실행 1: Level 1만 사용 (완전 순수 정자세)
    res_lvl1 = evaluate_level_subset(
        X_dead_all, y_dead_all, subjs_dead_all, views_dead_all, levels_all,
        max_level=1,
        level_title="1. Level 1만 사용 (순수 완전 정자세 데이터셋)",
        tau_threshold=0.50,
        folds_config=folds_config,
        sids_dead_all=sids_dead_all
    )

    # 실행 2: Level 1~3 사용 (Level 4 완전 물리역전만 배제한 현실 데이터셋)
    res_lvl13 = evaluate_level_subset(
        X_dead_all, y_dead_all, subjs_dead_all, views_dead_all, levels_all,
        max_level=3,
        level_title="2. Level 1~3 사용 (v5.0 18D 발 관절 파생피처 결합 현실 데이터셋)",
        tau_threshold=0.50,
        folds_config=folds_config,
        sids_dead_all=sids_dead_all
    )

    # 종합 대조 성적표 출력
    print("\n" + "■" * 105)
    print("★ [종합 비교 성적표] Level 1 (순수 정자세) vs Level 1~3 (현실 데이터)")
    print("■" * 105)
    header = f"{'평가 조건 (Dataset Filter)':<35} | {'평가 샘플':^9} | {'스티프 Prec':^12} | {'스티프 Rec':^12} | {'★ 스티프 F1':^13} | {'바벨 F1':^11} | {'ROC-AUC':^9}"
    print(header)
    print("-" * 105)

    for r in [res_lvl1, res_lvl13]:
        print(f"{r['title'][:33]:<35} | {r['samples']:^9d} | {r['stiff_prec']:^11.2f}% | {r['stiff_rec']:^11.2f}% | {r['stiff_f1']:^12.2f}% | {r['barbell_f1']:^10.2f}% | {r['auc']:^9.4f}")

    print("-" * 105)
    print(f"★ 전체 2회 전수 3-Fold 벤치마크 총 소요 시간: {time.time()-t_total:.2f}초")
    print("=" * 105 + "\n")

if __name__ == '__main__':
    main()

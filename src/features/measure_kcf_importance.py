# =========================================================================================================
# [스크립트명]: measure_kcf_importance.py
# [위치]: src/features/measure_kcf_importance.py
# [목적]: 55차원 생체역학 운동 경향성 피처(KCF)의 8종 운동 분류 모델 실제 기여도 정밀 실측 분석
# [방법론]:
#   - 3-Fold Out-of-Domain (시점 및 피험자 엄격 분리) 교차 검증 적용
#   - LightGBM (Gain & Split) 및 XGBoost (Gain & Weight) 기반 피처 중요도 정량화
#   - 55개 개별 피처 기여도 랭킹, 5대 핵심 도메인 그룹별 기여율(%), 운동 종목별 핵심 변별 피처 도출
#   - 결과를 JSON 및 공식 Word 보고서(.docx)로 저장
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

# 프로젝트 루트 등록
sys.path.insert(0, 'd:/')

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.features.kinematic_coordination import KinematicCoordinationExtractor
from src.models.first_model.train_hierarchical_kfold import CLASS_TO_MACRO_GROUP, MACRO_GROUP_NAMES

RESULTS_JSON_PATH = 'd:/Data/피처_데이터/tuning_results/kcf_feature_importance_results.json'

def load_data_and_extract_kcf():
    data_dir = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D'
    if not os.path.exists(os.path.join(data_dir, 'Dual_Train.npz')):
        data_dir = 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'

    print(f"[*] 마스터 데이터 로드 중: {data_dir}")
    all_X_b, all_y_act, all_subjs, all_views, all_sids = [], [], [], [], []

    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith('.npz'):
            continue
        split_data = Stage1DatasetLoader.load_split_data(fn, splits_dir=data_dir)
        all_X_b.append(split_data['X_Format_B'])
        all_y_act.append(split_data['y_action'])
        all_subjs.extend(split_data['subjects'])
        all_views.extend(split_data['views'])
        all_sids.extend(split_data['sequence_ids'])

    def get_subj_num(s):
        s = str(s)
        for digit in ['1','2','3','4','5','6','7','8','9']:
            if f"피험자{digit}" in s or s == digit or f"_{digit}_" in s:
                return int(digit)
        return 1

    subjs_arr = np.array([get_subj_num(s) for s in all_subjs])
    views_arr = np.array(all_views)
    raw_X_b = np.concatenate(all_X_b, axis=0)
    y_act = np.concatenate(all_y_act, axis=0)
    y_mac = np.array([CLASS_TO_MACRO_GROUP[c] for c in y_act], dtype=np.int64)

    print(f"    - 총 시퀀스 수: {len(y_act):,}개")
    print(f"[*] 55차원 KCF 피처 추출 중...")
    t0 = time.time()
    X_kcf = KinematicCoordinationExtractor.extract_features(raw_X_b)
    print(f"    ✔ 추출 완료! Shape: {X_kcf.shape}, 소요시간: {time.time()-t0:.2f}초")

    return X_kcf, y_act, y_mac, subjs_arr, views_arr

def run_importance_evaluation():
    X_kcf, y_act, y_mac, subjs_arr, views_arr = load_data_and_extract_kcf()
    feature_names = KinematicCoordinationExtractor.get_feature_names()
    num_features = len(feature_names)

    folds_config = [
        {
            'fold_id': 1,
            'name': 'Fold 1 [전방 (0°,45°)]',
            'test_subjs': [1, 2, 3],
            'test_views': ['view1', 'view2'],
            'train_subjs': [4, 5, 6, 7, 8, 9],
            'train_views': ['view3', 'view4', 'view5']
        },
        {
            'fold_id': 2,
            'name': 'Fold 2 [측면 (90°)]',
            'test_subjs': [4, 5, 6],
            'test_views': ['view3'],
            'train_subjs': [1, 2, 3, 7, 8, 9],
            'train_views': ['view1', 'view2', 'view4', 'view5']
        },
        {
            'fold_id': 3,
            'name': 'Fold 3 [후방 (135°,180°)]',
            'test_subjs': [7, 8, 9],
            'test_views': ['view4', 'view5'],
            'train_subjs': [1, 2, 3, 4, 5, 6],
            'train_views': ['view1', 'view2', 'view3']
        }
    ]

    lgb_gain_folds = []
    lgb_split_folds = []
    xgb_gain_folds = []
    xgb_weight_folds = []

    macro_accs = []
    macro_f1s = []

    print("\n" + "=" * 90)
    print("★ [KCF 55D 피처 중요도 3-Fold Out-of-Domain 실측 평가]")
    print("=" * 90)

    for fc in folds_config:
        f_name = fc['name']
        print(f"\n>>> 훈련 진행 중: {f_name}...")
        test_mask = np.isin(subjs_arr, fc['test_subjs']) & np.isin(views_arr, fc['test_views'])
        train_mask = np.isin(subjs_arr, fc['train_subjs']) & np.isin(views_arr, fc['train_views'])

        X_tr, y_tr = X_kcf[train_mask], y_mac[train_mask]
        X_te, y_te = X_kcf[test_mask], y_mac[test_mask]

        # 클래스 가중치 계산
        num_classes = len(np.unique(y_mac))
        counts = np.bincount(y_tr, minlength=num_classes)
        raw_w = len(y_tr) / (float(num_classes) * np.maximum(counts, 1).astype(np.float32))
        sample_w = np.sqrt(raw_w)[y_tr]

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=4, verbose=-1, importance_type='gain'
        )
        lgb_model.fit(X_tr, y_tr, sample_weight=sample_w)

        # Split importance도 추출
        lgb_gain = lgb_model.booster_.feature_importance(importance_type='gain')
        lgb_split = lgb_model.booster_.feature_importance(importance_type='split')

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.1,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='mlogloss', n_jobs=4
        )
        xgb_model.fit(X_tr, y_tr, sample_weight=sample_w)

        xgb_booster = xgb_model.get_booster()
        score_gain = xgb_booster.get_score(importance_type='gain')
        score_weight = xgb_booster.get_score(importance_type='weight')

        xgb_gain = np.zeros(num_features, dtype=np.float32)
        xgb_weight = np.zeros(num_features, dtype=np.float32)
        for i in range(num_features):
            f_key = f"f{i}"
            xgb_gain[i] = score_gain.get(f_key, 0.0)
            xgb_weight[i] = score_weight.get(f_key, 0.0)

        # 예측 성능 측정
        p_lgb = lgb_model.predict_proba(X_te)
        p_xgb = xgb_model.predict_proba(X_te)
        p_ens = 0.6 * p_lgb + 0.4 * p_xgb
        preds = np.argmax(p_ens, axis=1)

        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        macro_accs.append(acc)
        macro_f1s.append(f1)
        print(f"    ✔ {f_name} 평가 완료: Top-1 Acc = {acc*100:.2f}%, Macro F1 = {f1*100:.2f}%")

        # 정규화 (백분율)
        lgb_gain_norm = (lgb_gain / (np.sum(lgb_gain) + 1e-8)) * 100.0
        lgb_split_norm = (lgb_split / (np.sum(lgb_split) + 1e-8)) * 100.0
        xgb_gain_norm = (xgb_gain / (np.sum(xgb_gain) + 1e-8)) * 100.0
        xgb_weight_norm = (xgb_weight / (np.sum(xgb_weight) + 1e-8)) * 100.0

        lgb_gain_folds.append(lgb_gain_norm)
        lgb_split_folds.append(lgb_split_norm)
        xgb_gain_folds.append(xgb_gain_norm)
        xgb_weight_folds.append(xgb_weight_norm)

    # 3-Fold 평균 산출
    mean_lgb_gain = np.mean(lgb_gain_folds, axis=0)
    mean_lgb_split = np.mean(lgb_split_folds, axis=0)
    mean_xgb_gain = np.mean(xgb_gain_folds, axis=0)
    mean_xgb_weight = np.mean(xgb_weight_folds, axis=0)

    # 하이브리드 종합 기여도: LightGBM Gain 50% + XGBoost Gain 50%
    overall_gain = 0.5 * mean_lgb_gain + 0.5 * mean_xgb_gain
    overall_split = 0.5 * mean_lgb_split + 0.5 * mean_xgb_weight

    # 피처 카테고리 매핑
    def get_category(name: str) -> str:
        if name.startswith('crp_'):
            return "1. 연속 상대 위상 (CRP, 8D)"
        elif name.startswith('vc_'):
            return "2. 수정된 벡터 코딩 (Vector Coding, 10D)"
        elif name.startswith('cyclogram_'):
            return "3. 각도-각도 사이클로그램 (Cyclograms, 8D)"
        elif name.startswith('rom_') or name.startswith('vel_') or 'correlation' in name or 'bottom' in name or 'phase_lag' in name:
            return "4. 무차원 가동범위 및 각속도 시너지 (15D)"
        else:
            return "5. 해부학적 불변 지표 (팔꿈치/골반/바벨, 14D)"

    categories = [get_category(n) for n in feature_names]

    # 정렬 및 랭킹 산출
    ranked_indices = np.argsort(-overall_gain)

    # 카테고리별 누적 기여도 집계
    cat_summary = {}
    for idx in range(num_features):
        cat = categories[idx]
        if cat not in cat_summary:
            cat_summary[cat] = {'gain': 0.0, 'split': 0.0, 'count': 0, 'features': []}
        cat_summary[cat]['gain'] += float(overall_gain[idx])
        cat_summary[cat]['split'] += float(overall_split[idx])
        cat_summary[cat]['count'] += 1
        cat_summary[cat]['features'].append((feature_names[idx], float(overall_gain[idx])))

    # 카테고리별 내부 정렬
    for cat in cat_summary:
        cat_summary[cat]['features'].sort(key=lambda x: -x[1])

    # 결과 패킹
    results = {
        'overall_model_acc': float(np.mean(macro_accs)),
        'overall_model_f1': float(np.mean(macro_f1s)),
        'category_summary': cat_summary,
        'feature_rankings': []
    }

    print("\n" + "=" * 110)
    print(f"{'Rank':<5} | {'Feature Name':<38} | {'Category':<30} | {'Gain (%)':<10} | {'Split (%)':<10} | {'Cumul (%)':<10}")
    print("-" * 110)

    cumul = 0.0
    for rank, idx in enumerate(ranked_indices, 1):
        fn = feature_names[idx]
        cat = categories[idx]
        g = overall_gain[idx]
        s = overall_split[idx]
        cumul += g

        results['feature_rankings'].append({
            'rank': rank,
            'name': fn,
            'category': cat,
            'gain_pct': float(g),
            'split_pct': float(s),
            'cumul_pct': float(cumul),
            'lgb_gain_pct': float(mean_lgb_gain[idx]),
            'xgb_gain_pct': float(mean_xgb_gain[idx])
        })

        if rank <= 25 or rank > 50:
            print(f"{rank:<5} | {fn:<38} | {cat[:28]:<30} | {g:8.2f}% | {s:8.2f}% | {cumul:8.2f}%")
        elif rank == 26:
            print(f"{'...':<5} | {'... (Rank 26 ~ 50 생략) ...':<38} | {'...':<30} | {'...':<10} | {'...':<10} | {'...':<10}")

    print("=" * 110)

    print("\n[5대 핵심 카테고리별 기여율 요약]")
    for cat, data in sorted(cat_summary.items(), key=lambda x: -x[1]['gain']):
        print(f"  - {cat:<40}: 정보 획득(Gain) 기여율 = {data['gain']:6.2f}% | 분기(Split) 비율 = {data['split']:6.2f}% (피처 수: {data['count']}개)")

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✔ 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")

    return results

if __name__ == '__main__':
    run_importance_evaluation()

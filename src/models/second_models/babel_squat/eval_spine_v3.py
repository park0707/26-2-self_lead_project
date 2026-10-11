# -*- coding: utf-8 -*-
# =========================================================================================================
# [스크립트]: 바벨 스쿼트 척추 중립(c1) 피처셋 v3 하이브리드 확장 평가
# [전략 변경]: 경추/시선 피처 제거 시 -3.87%p 하락 확인 → 제거 대신 유지 후 신규 피처 추가
#
# [1차 실험 교훈]:
#   - 경추/시선 피처(cervical_min, gaze_pitch_*): 제거 시 성능 -3.87%p 하락
#     → 이들은 c2 전담이지만 c1 결함과 실질적 공변량 관계 존재 (잘못된 자세는 목+척추 동시 불량)
#     → 제거하면 이 공변량 정보까지 사라져 성능 저하
#   - 신규 선행연구 기반 4D: 단독으로는 기존 정보를 대체하지 못함
#   - 최적 전략: 경추/시선 피처 유지 + 신규 피처 추가 = 20D (baseline 16D + 4D 신규)
#
# [v3 피처셋 구성]:
#   A. 기존 베이스라인 16D (경추/시선 포함, 기존 챔피언 F1 84.34%)
#   B. v3_hybrid = 16D + 4D 신규 선행연구 = 20D (목표: F1 >84.34%)
#   C. v3_select = 16D + 상위 2D 신규 선별 = 18D (경량화 목표)
# =========================================================================================================
import os, sys, time, json, warnings
sys.path.insert(0, 'd:/')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
warnings.filterwarnings('ignore')

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score

from src.models.second_models.extract_squat_spine_v2 import extract_squat_spine_v2, SPINE_TREND_V2_FEATURE_NAMES_10
from src.models.second_models.extract_squat_spine_trend import extract_squat_spine_trend_features, SPINE_TREND_FEATURE_NAMES_9
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics
from src.models.second_models.unified_stage2_dataset import UnifiedStage2DatasetLoader

OUTPUT_JSON = 'd:/src/models/second_models/spine_v3_eval_results.json'

# ─── 1. 데이터 로딩 ─────────────────────────────────────────────────────────
def load_all_data():
    print("[1/4] 데이터 로딩...", flush=True)
    npz_file = SquatPilotDatasetLoader._find_master_npz()
    d = np.load(npz_file, allow_pickle=True)
    sq_mask = (d['y_action'] == 0)
    X_b = d['X_Format_B'][sq_mask].astype(np.float32)

    data_u = UnifiedStage2DatasetLoader.load_exercise_data(0)
    subjects = data_u['subjects']
    y_c1     = data_u['Y_defects'][:, 0]

    print(f"    총 샘플: {len(y_c1)}, 결함 비율: {y_c1.mean()*100:.1f}%", flush=True)

    curv_d     = np.load('d:/src/models/second_models/squat_spine_curvature_feats.npz', allow_pickle=True)
    curv_feats = curv_d['curvature_feats']
    curv_names = list(curv_d['feature_names'])

    fine_d     = np.load('d:/src/models/second_models/squat_fine_feats_12d.npz', allow_pickle=True)
    fine_feats = fine_d['fine_feats']
    fine_names = list(fine_d['feature_names'])

    return X_b, subjects, y_c1, curv_feats, curv_names, fine_feats, fine_names


def build_feature_sets(X_b, curv_feats, curv_names, fine_feats, fine_names):
    print("[2/4] 피처셋 빌드...", flush=True)

    # ── 기존 베이스라인 (16D = 9D 동적 + 4D 자세기하 + 3D 귀-어깨-고관절) ──
    feats_9d = extract_squat_spine_trend_features(X_b)
    X_bot    = extract_squat_bottom_biomechanics(X_b)
    torso_shank_err = X_bot[:, 1:2]

    gaze_pitch_mean = fine_feats[:, 0:1]
    gaze_pitch_min  = fine_feats[:, 1:2]
    cervical_min    = fine_feats[:, 3:4]
    curv_3d = curv_feats[:, :3]  # ear_sh_hip: min, mean, drop

    baseline_X = np.concatenate([
        feats_9d, torso_shank_err, cervical_min, gaze_pitch_mean, gaze_pitch_min, curv_3d
    ], axis=1)
    baseline_names = (
        SPINE_TREND_FEATURE_NAMES_9
        + ['torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min']
        + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']
    )

    # ── 신규 4D 피처 (v2 추출기에서 새로 추가된 4개만) ──
    feats_10d = extract_squat_spine_v2(X_b)
    new_4d = feats_10d[:, 6:10]  # indices 6-9: 신규 4D만
    new_4d_names = SPINE_TREND_V2_FEATURE_NAMES_10[6:10]

    # ── v3 hybrid: baseline 16D + 신규 4D = 20D ──
    v3_hybrid_X = np.concatenate([baseline_X, new_4d], axis=1)
    v3_hybrid_names = baseline_names + new_4d_names

    # ── v3 select: baseline 16D + 상위 2D (c1_descent_spine_drift_rate + c1_pelvis_tilt_at_bottom) ──
    drift_rate   = feats_10d[:, 8:9]   # c1_descent_spine_drift_rate
    pelvis_tilt  = feats_10d[:, 7:8]   # c1_pelvis_tilt_at_bottom
    v3_select_X  = np.concatenate([baseline_X, pelvis_tilt, drift_rate], axis=1)
    v3_select_names = baseline_names + ['c1_pelvis_tilt_at_bottom', 'c1_descent_spine_drift_rate']

    # ── v3 plus_all_curv: baseline 16D + 나머지 곡률 3D (neck-back-waist) ──
    neck_curv_3d = curv_feats[:, 3:6]  # fine_neck_back_waist_angle_min, _drop, fine_head_forward_protrusion
    v3_all_curv_X = np.concatenate([baseline_X, neck_curv_3d], axis=1)
    v3_all_curv_names = baseline_names + curv_names[3:6]

    print(f"    베이스라인: {baseline_X.shape[1]}D, v3_hybrid: {v3_hybrid_X.shape[1]}D, "
          f"v3_select: {v3_select_X.shape[1]}D, v3_all_curv: {v3_all_curv_X.shape[1]}D", flush=True)

    return {
        'baseline':    (baseline_X,     baseline_names),
        'v3_hybrid':   (v3_hybrid_X,    v3_hybrid_names),
        'v3_select':   (v3_select_X,    v3_select_names),
        'v3_all_curv': (v3_all_curv_X,  v3_all_curv_names),
    }


# ─── 모델 평가 함수 (더 넓은 파라미터 탐색 포함) ────────────────────────────
def get_model_candidates():
    return {
        'LightGBM': [
            lgb.LGBMClassifier(n_estimators=100,  max_depth=5, learning_rate=0.05, num_leaves=31,  colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=150,  max_depth=7, learning_rate=0.03, num_leaves=63,  colsample_bytree=0.7, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=200,  max_depth=6, learning_rate=0.03, num_leaves=31,  colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=200,  max_depth=5, learning_rate=0.02, num_leaves=31,  colsample_bytree=0.8, reg_lambda=3.0, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=300,  max_depth=6, learning_rate=0.01, num_leaves=31,  colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1),
        ],
        'XGBoost': [
            xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.5, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
        ],
        'CatBoost': [
            CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=150, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=200, depth=6, learning_rate=0.02, l2_leaf_reg=7.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=200, depth=7, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, thread_count=-1, verbose=False),
        ],
        'RandomForest': [
            RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=4, max_features=0.5, random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1),
        ],
        'ExtraTrees': [
            ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_split=4, max_features=0.5, random_state=42, n_jobs=-1),
        ],
    }


def run_tournament(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, tag: str, level4_mask: np.ndarray = None):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    if level4_mask is not None:
        X = X[level4_mask]; y = y[level4_mask]; subjects = subjects[level4_mask]

    candidates = get_model_candidates()
    champion_name  = None
    champion_score = -1.0
    champion_metrics = {}
    all_results = {}

    for model_name, model_list in candidates.items():
        for trial_i, model in enumerate(model_list):
            fold_f1s, fold_baccs, fold_aucs, fold_pres, fold_recs, lats = [], [], [], [], [], []

            for tr_idx, val_idx in sgkf.split(X, y, groups=subjects):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                t0 = time.time()
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1]
                lat = ((time.time() - t0) / max(1, len(X_val))) * 1000.0
                lats.append(lat)

                fold_f1s.append(f1_score(y_val, preds, zero_division=0))
                fold_baccs.append(balanced_accuracy_score(y_val, preds))
                try:
                    fold_aucs.append(roc_auc_score(y_val, probs))
                except Exception:
                    fold_aucs.append(0.5)
                fold_pres.append(precision_score(y_val, preds, zero_division=0))
                fold_recs.append(recall_score(y_val, preds, zero_division=0))

            mean_f1   = float(np.mean(fold_f1s))
            mean_bacc = float(np.mean(fold_baccs))
            mean_auc  = float(np.mean(fold_aucs))
            mean_pre  = float(np.mean(fold_pres))
            mean_rec  = float(np.mean(fold_recs))
            mean_lat  = float(np.mean(lats))
            score     = 0.40 * mean_f1 + 0.35 * mean_bacc + 0.25 * mean_auc

            key = f"{model_name}_t{trial_i}"
            all_results[key] = {
                'f1': round(mean_f1 * 100, 2),
                'precision': round(mean_pre * 100, 2),
                'recall': round(mean_rec * 100, 2),
                'bacc': round(mean_bacc * 100, 2),
                'roc_auc': round(mean_auc * 100, 2),
                'score': round(score * 100, 2),
                'latency_ms': round(mean_lat, 3),
            }

            star = ' ★' if score > champion_score else ''
            print(f"    [{tag}] {model_name}[{trial_i}] F1={mean_f1*100:.2f}% BAcc={mean_bacc*100:.2f}% AUC={mean_auc*100:.2f}%{star}", flush=True)

            if score > champion_score:
                champion_score   = score
                champion_name    = key
                champion_metrics = all_results[key]

    print(f"  >> [{tag}] 챔피언: {champion_name} | F1={champion_metrics['f1']}% AUC={champion_metrics['roc_auc']}%", flush=True)
    return champion_name, champion_metrics, all_results


def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 스쿼트 척추 중립(c1) 피처 v3 하이브리드 성능 향상 검증", flush=True)
    print("  [전략]: 경추/시선 피처 유지 + 선행 연구 기반 신규 4D 추가 → 20D 하이브리드", flush=True)
    print("=" * 100, flush=True)

    X_b, subjects, y_c1, curv_feats, curv_names, fine_feats, fine_names = load_all_data()
    feat_sets = build_feature_sets(X_b, curv_feats, curv_names, fine_feats, fine_names)

    print("\n[3/4] Level 4 마스크 로드...", flush=True)
    amb_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    lvl4_mask = ~amb_d['l4_mask']
    print(f"    Level 1~3 보존: {lvl4_mask.sum()}개", flush=True)

    print("\n[4/4] 피처셋별 토너먼트 시작...", flush=True)
    results = {}

    for feat_tag, (X_feat, feat_names) in feat_sets.items():
        print(f"\n{'='*80}", flush=True)
        print(f"  === 피처셋: {feat_tag} ({X_feat.shape[1]}D) ===", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  [A] Level 1~3 데이터 ({lvl4_mask.sum()}개) 평가:", flush=True)

        champ_name, champ_metrics, all_res = run_tournament(
            X_feat, y_c1, subjects, f"{feat_tag}", level4_mask=lvl4_mask
        )

        results[feat_tag] = {
            'feature_count': X_feat.shape[1],
            'feature_names': list(feat_names),
            'level123': {'champion': champ_name, 'metrics': champ_metrics, 'all_results': all_res}
        }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장: {OUTPUT_JSON}", flush=True)

    baseline_f1 = 83.23
    print("\n" + "=" * 100, flush=True)
    print("  [최종 F1-Score 비교 요약 (v3 하이브리드)]", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'피처셋':<22} {'차원':>4} {'F1-Score':>10} {'향상':>10} {'ROC-AUC':>10} {'BAcc':>8} {'챔피언':<30}", flush=True)
    print(f"  {'-'*90}", flush=True)
    print(f"  {'기존 공식 베이스라인':<22} {'16D':>4} {baseline_f1:>9.2f}% {'─':>10} {'92.27%':>10} {'82.95%':>8} {'LightGBM cascade':<30}", flush=True)

    for feat_tag, res in results.items():
        m = res['level123']['metrics']
        f1 = m['f1']; auc = m['roc_auc']; bacc = m['bacc']
        delta = f1 - baseline_f1
        sign = '+' if delta >= 0 else ''
        delta_str = f"{sign}{delta:.2f}%p"
        print(f"  {feat_tag:<22} {res['feature_count']:>4}D {f1:>9.2f}% {delta_str:>10} {auc:>9.2f}% {bacc:>7.2f}% {res['level123']['champion']:<30}", flush=True)

    print("=" * 100, flush=True)
    return results


if __name__ == '__main__':
    main()

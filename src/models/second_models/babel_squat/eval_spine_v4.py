# -*- coding: utf-8 -*-
# =========================================================================================================
# [스크립트]: 바벨 스쿼트 척추 중립(c1) 피처셋 v4 최종 확장 평가
# [전략]: v3_all_curv (19D, F1 85.31%) 챔피언에 선행 연구 기반 문헌 6D를 추가
#
# [v3 결과 요약]:
#   - baseline (16D): F1 84.61%, AUC 92.45%
#   - v3_all_curv (19D = baseline + neck-back-waist 곡률 3D): F1 85.31%, AUC 92.89% [NEW BEST!]
#   - v3_select (18D): F1 84.42%
#   - v3_hybrid (20D): F1 84.29%
#
# [v4 피처셋 구성]:
#   A. v3_all_curv (19D) = 기존 챔피언 (베이스라인)
#   B. v4_lit_full (25D) = v3_all_curv 19D + 문헌 6D
#   C. v4_lit_best (21D) = v3_all_curv 19D + 문헌 6D 중 상위 2D (phase_offset + jerk_peak)
#   D. v4_lit_4d  (23D) = v3_all_curv 19D + 문헌 6D 중 상위 4D
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

from src.models.second_models.extract_squat_spine_trend import extract_squat_spine_trend_features, SPINE_TREND_FEATURE_NAMES_9
from src.models.second_models.extract_squat_spine_literature import extract_squat_spine_literature_features, SPINE_LIT_FEATURE_NAMES_6
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics
from src.models.second_models.unified_stage2_dataset import UnifiedStage2DatasetLoader

OUTPUT_JSON = 'd:/src/models/second_models/spine_v4_eval_results.json'

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

    return X_b, subjects, y_c1, curv_feats, curv_names, fine_feats


def build_feature_sets(X_b, curv_feats, curv_names, fine_feats):
    print("[2/4] 피처셋 빌드...", flush=True)

    # ── v3_all_curv 재현 (19D = baseline 16D + neck 곡률 3D) ──
    feats_9d        = extract_squat_spine_trend_features(X_b)
    X_bot           = extract_squat_bottom_biomechanics(X_b)
    torso_shank_err = X_bot[:, 1:2]
    gaze_pitch_mean = fine_feats[:, 0:1]
    gaze_pitch_min  = fine_feats[:, 1:2]
    cervical_min    = fine_feats[:, 3:4]
    curv_3d         = curv_feats[:, :3]   # ear_sh_hip: min, mean, drop
    neck_curv_3d    = curv_feats[:, 3:6]  # fine_neck_back_waist + fine_head_forward_protrusion

    v3_all_curv_X = np.concatenate([
        feats_9d, torso_shank_err, cervical_min, gaze_pitch_mean, gaze_pitch_min,
        curv_3d, neck_curv_3d
    ], axis=1)
    v3_all_curv_names = (
        SPINE_TREND_FEATURE_NAMES_9
        + ['torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min']
        + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']
        + curv_names[3:6]
    )

    # ── 선행 연구 6D 추출 ──
    lit_6d = extract_squat_spine_literature_features(X_b)

    # ── v4_lit_full (25D = 19D + 6D) ──
    v4_lit_full_X     = np.concatenate([v3_all_curv_X, lit_6d], axis=1)
    v4_lit_full_names = v3_all_curv_names + SPINE_LIT_FEATURE_NAMES_6

    # ── v4_lit_best (21D = 19D + 상위 2D: phase_offset + jerk_peak) ──
    # [2] c1_max_trunk_lean_phase_offset, [3] c1_trunk_jerk_peak_at_nadir
    lit_2d_best = lit_6d[:, [2, 3]]
    v4_lit_best_X     = np.concatenate([v3_all_curv_X, lit_2d_best], axis=1)
    v4_lit_best_names = v3_all_curv_names + [SPINE_LIT_FEATURE_NAMES_6[2], SPINE_LIT_FEATURE_NAMES_6[3]]

    # ── v4_lit_4d (23D = 19D + 4D: sh_hip_knee + trunk_thigh_ratio + phase_offset + jerk_peak) ──
    lit_4d_best = lit_6d[:, [0, 1, 2, 3]]
    v4_lit_4d_X     = np.concatenate([v3_all_curv_X, lit_4d_best], axis=1)
    v4_lit_4d_names = v3_all_curv_names + SPINE_LIT_FEATURE_NAMES_6[:4]

    print(f"    v3_all_curv: {v3_all_curv_X.shape[1]}D, v4_lit_full: {v4_lit_full_X.shape[1]}D, "
          f"v4_lit_best: {v4_lit_best_X.shape[1]}D, v4_lit_4d: {v4_lit_4d_X.shape[1]}D", flush=True)

    return {
        'v3_all_curv': (v3_all_curv_X,  v3_all_curv_names),
        'v4_lit_full': (v4_lit_full_X,  v4_lit_full_names),
        'v4_lit_best': (v4_lit_best_X,  v4_lit_best_names),
        'v4_lit_4d':   (v4_lit_4d_X,    v4_lit_4d_names),
    }


def get_model_candidates():
    return {
        'LightGBM': [
            lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, num_leaves=31,  colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.03, num_leaves=63,  colsample_bytree=0.7, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31,  colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.02, num_leaves=31,  colsample_bytree=0.8, reg_lambda=3.0, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.01, num_leaves=31,  colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1),
        ],
        'XGBoost': [
            xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,  colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, subsample=0.9,  colsample_bytree=0.7, reg_alpha=0.5, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.02, subsample=0.8,  colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
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
    }


def run_tournament(X, y, subjects, tag, level4_mask=None):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    if level4_mask is not None:
        X, y, subjects = X[level4_mask], y[level4_mask], subjects[level4_mask]

    candidates = get_model_candidates()
    champion_name, champion_score, champion_metrics, all_results = None, -1.0, {}, {}

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

            mean_f1, mean_bacc, mean_auc = float(np.mean(fold_f1s)), float(np.mean(fold_baccs)), float(np.mean(fold_aucs))
            mean_pre, mean_rec, mean_lat = float(np.mean(fold_pres)), float(np.mean(fold_recs)), float(np.mean(lats))
            score = 0.40 * mean_f1 + 0.35 * mean_bacc + 0.25 * mean_auc
            key = f"{model_name}_t{trial_i}"
            all_results[key] = {'f1': round(mean_f1*100, 2), 'precision': round(mean_pre*100, 2),
                                 'recall': round(mean_rec*100, 2), 'bacc': round(mean_bacc*100, 2),
                                 'roc_auc': round(mean_auc*100, 2), 'score': round(score*100, 2),
                                 'latency_ms': round(mean_lat, 3)}
            star = ' ★' if score > champion_score else ''
            print(f"    [{tag}] {model_name}[{trial_i}] F1={mean_f1*100:.2f}% BAcc={mean_bacc*100:.2f}% AUC={mean_auc*100:.2f}%{star}", flush=True)
            if score > champion_score:
                champion_score, champion_name, champion_metrics = score, key, all_results[key]

    print(f"  >> [{tag}] 챔피언: {champion_name} | F1={champion_metrics['f1']}% AUC={champion_metrics['roc_auc']}%", flush=True)
    return champion_name, champion_metrics, all_results


def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 스쿼트 척추 중립(c1) 피처 v4 최종 선행연구 추가 검증", flush=True)
    print("  [목표]: v3_all_curv (F1 85.31%) 기반 + 문헌 6D 추가로 추가 개선", flush=True)
    print("=" * 100, flush=True)

    X_b, subjects, y_c1, curv_feats, curv_names, fine_feats = load_all_data()
    feat_sets = build_feature_sets(X_b, curv_feats, curv_names, fine_feats)

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

        champ_name, champ_metrics, all_res = run_tournament(X_feat, y_c1, subjects, feat_tag, level4_mask=lvl4_mask)
        results[feat_tag] = {
            'feature_count': X_feat.shape[1],
            'feature_names': list(feat_names),
            'level123': {'champion': champ_name, 'metrics': champ_metrics, 'all_results': all_res}
        }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장: {OUTPUT_JSON}", flush=True)

    prev_best = 85.31
    baseline_official = 83.23
    print("\n" + "=" * 100, flush=True)
    print("  [최종 F1-Score 비교 요약 (전체 실험)]", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'피처셋':<24} {'차원':>4} {'F1-Score':>10} {'vs 공식기준':>12} {'vs v3_best':>10} {'ROC-AUC':>10} {'챔피언'}", flush=True)
    print(f"  {'-'*95}", flush=True)
    print(f"  {'기존 공식 베이스라인':<24} {'16D':>4} {baseline_official:>9.2f}% {'─':>12} {'─':>10} {'92.27%':>10} LightGBM cascade", flush=True)
    print(f"  {'v3_all_curv (이전 베스트)':<24} {'19D':>4} {prev_best:>9.2f}% {'+2.08%p':>12} {'─':>10} {'92.89%':>10} CatBoost", flush=True)

    for feat_tag, res in results.items():
        m = res['level123']['metrics']
        f1, auc = m['f1'], m['roc_auc']
        vs_official = f"+{f1-baseline_official:.2f}%p" if f1 > baseline_official else f"{f1-baseline_official:.2f}%p"
        vs_v3       = f"+{f1-prev_best:.2f}%p" if f1 > prev_best else f"{f1-prev_best:.2f}%p"
        print(f"  {feat_tag:<24} {res['feature_count']:>4}D {f1:>9.2f}% {vs_official:>12} {vs_v3:>10} {auc:>9.2f}% {res['level123']['champion']}", flush=True)

    print("=" * 100, flush=True)
    return results


if __name__ == '__main__':
    main()

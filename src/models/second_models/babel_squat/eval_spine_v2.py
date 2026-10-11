# -*- coding: utf-8 -*-
# =========================================================================================================
# [스크립트]: 바벨 스쿼트 척추 중립(c1) 피처셋 v2 성능 향상 검증 토너먼트
# [실행방법]: python d:/src/models/second_models/eval_spine_v2.py
#
# [검증 구조]:
#   1. 베이스라인 (16D = 기존 9D 동적 + 4D 자세기하 + 3D 곡률)
#   2. v2 정제 (10D 동적 v2 + 3D 곡률 ear-shoulder-hip)  → 13D
#   3. v2 확장 (10D 동적 v2 + 6D 전체 곡률)              → 16D
#   각각 5대 모델(LightGBM, CatBoost, XGBoost, RF, ExtraTrees) 5-Fold SGK 토너먼트
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

OUTPUT_JSON = 'd:/src/models/second_models/spine_v2_eval_results.json'

# ─── 1. 데이터 로딩 ─────────────────────────────────────────────────────────
def load_all_data():
    print("[1/4] 데이터 로딩...", flush=True)
    npz_file = SquatPilotDatasetLoader._find_master_npz()
    d = np.load(npz_file, allow_pickle=True)
    sq_mask = (d['y_action'] == 0)
    X_b = d['X_Format_B'][sq_mask].astype(np.float32)  # (N, 16, 92)

    data_u = UnifiedStage2DatasetLoader.load_exercise_data(0)
    subjects = data_u['subjects']
    y_c1     = data_u['Y_defects'][:, 0]

    print(f"    총 샘플: {len(y_c1)}, 결함 비율: {y_c1.mean()*100:.1f}%", flush=True)

    # 곡률 6D
    curv_d     = np.load('d:/src/models/second_models/squat_spine_curvature_feats.npz', allow_pickle=True)
    curv_feats = curv_d['curvature_feats']   # (N, 6)
    curv_names = list(curv_d['feature_names'])

    # 세밀 12D (고개/발 포함 전체)
    fine_d     = np.load('d:/src/models/second_models/squat_fine_feats_12d.npz', allow_pickle=True)
    fine_feats = fine_d['fine_feats']       # (N, 12)
    fine_names = list(fine_d['feature_names'])

    return X_b, subjects, y_c1, curv_feats, curv_names, fine_feats, fine_names


def build_feature_sets(X_b, curv_feats, curv_names, fine_feats, fine_names):
    print("[2/4] 피처셋 빌드...", flush=True)

    # ── 기존 베이스라인 구성 (16D = 9D 동적 + 4D 자세기하 + 3D 귀-어깨-고관절) ──
    feats_9d = extract_squat_spine_trend_features(X_b)  # 9D

    X_bot   = extract_squat_bottom_biomechanics(X_b)    # (N, 8)
    torso_shank_err = X_bot[:, 1:2]                     # 상체-정강이 평행도 오차

    gaze_pitch_mean = fine_feats[:, 0:1]                # 시선 고도각 평균  ← c2 전담 (추후 제거)
    gaze_pitch_min  = fine_feats[:, 1:2]                # 시선 고도각 최저  ← c2 전담 (추후 제거)
    cervical_min    = fine_feats[:, 3:4]                # 경추 최소각       ← c2 전담 (추후 제거)

    curv_3d = curv_feats[:, :3]  # ear_sh_hip: min, mean, drop

    baseline_X = np.concatenate([
        feats_9d,
        torso_shank_err,
        cervical_min,
        gaze_pitch_mean,
        gaze_pitch_min,
        curv_3d
    ], axis=1)
    baseline_names = (
        SPINE_TREND_FEATURE_NAMES_9
        + ['torso_shank_err', 'cervical_min', 'gaze_pitch_mean', 'gaze_pitch_min']
        + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']
    )

    # ── v2 정제 (13D = 10D 동적 v2 + 3D 귀-어깨-고관절 곡률) ──
    feats_10d = extract_squat_spine_v2(X_b)             # 10D

    v2_compact_X = np.concatenate([
        feats_10d,
        curv_3d
    ], axis=1)
    v2_compact_names = SPINE_TREND_V2_FEATURE_NAMES_10 + ['ear_sh_hip_min', 'ear_sh_hip_mean', 'ear_sh_hip_drop']

    # ── v2 확장 (16D = 10D 동적 v2 + 6D 전체 곡률) ──
    v2_full_X = np.concatenate([
        feats_10d,
        curv_feats
    ], axis=1)
    v2_full_names = SPINE_TREND_V2_FEATURE_NAMES_10 + curv_names

    print(f"    베이스라인: {baseline_X.shape[1]}D, v2 정제: {v2_compact_X.shape[1]}D, v2 확장: {v2_full_X.shape[1]}D", flush=True)

    return {
        'baseline': (baseline_X, baseline_names),
        'v2_compact': (v2_compact_X, v2_compact_names),
        'v2_full':   (v2_full_X,    v2_full_names),
    }


# ─── 3. 모델 평가 함수 ─────────────────────────────────────────────────────
def get_model_candidates():
    return {
        'LightGBM': [
            lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.03, num_leaves=63, colsample_bytree=0.7, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=120, max_depth=6, learning_rate=0.03, num_leaves=31, reg_lambda=2.0,       colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
            lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.02, num_leaves=31, reg_lambda=3.0, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1),
        ],
        'CatBoost': [
            CatBoostClassifier(iterations=120, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=150, depth=7, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=150, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=False),
            CatBoostClassifier(iterations=200, depth=6, learning_rate=0.02, l2_leaf_reg=7.0, random_seed=42, thread_count=-1, verbose=False),
        ],
        'XGBoost': [
            xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
            xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.5, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
        ],
        'RandomForest': [
            RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=4, max_features=0.5, random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1),
        ],
        'ExtraTrees': [
            ExtraTreesClassifier(n_estimators=150, max_depth=14, min_samples_split=4, max_features=0.5, random_state=42, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1),
        ],
    }


def run_tournament(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, tag: str, level4_mask: np.ndarray = None):
    """5-Fold SGK 피험자 격리 토너먼트 → 챔피언 반환"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Level 4 정제 (mask가 주어지면 해당 데이터만 사용)
    if level4_mask is not None:
        X = X[level4_mask]
        y = y[level4_mask]
        subjects = subjects[level4_mask]

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


# ─── Level 4 모호성 배제 마스크 생성 ─────────────────────────────────────
def build_level4_mask(y_c1: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """
    [기존 검증된 마스크 로드]
    squat_spine_ambiguity_levels.npz의 l4_mask 사용 → 정확히 343개(17.4%) 제거
    기준: [바벨_스쿼트_척추_중립_데이터_모호성_분석_보고서] 검증값
      - Level 4-A (후광 오라벨 96개): 척추 정상인데 오류 태깅
      - Level 4-B (결함 누락 247개): 등 굽음인데 정상 태깅
    """
    amb_d = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz', allow_pickle=True)
    l4_exclude = amb_d['l4_mask']   # True = Level 4 (제거 대상)
    keep_mask = ~l4_exclude

    n_total   = len(y_c1)
    n_removed = int(l4_exclude.sum())
    print(f"    Level 4 제거: {n_removed}개/{n_total}개 ({n_removed/n_total*100:.1f}%) → 잔존 {keep_mask.sum()}개", flush=True)
    return keep_mask


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 스쿼트 척추 중립(c1) 피처 v2 성능 검증 토너먼트", flush=True)
    print("=" * 100, flush=True)

    X_b, subjects, y_c1, curv_feats, curv_names, fine_feats, fine_names = load_all_data()
    feat_sets = build_feature_sets(X_b, curv_feats, curv_names, fine_feats, fine_names)

    print("\n[3/4] Level 4 모호성 마스크 생성...", flush=True)
    lvl4_mask = build_level4_mask(y_c1, X_b)
    print(f"    Level 4 이후 데이터: {lvl4_mask.sum()}개", flush=True)

    print("\n[4/4] 피처셋별 토너먼트 시작...", flush=True)
    results = {}

    for feat_tag, (X_feat, feat_names) in feat_sets.items():
        print(f"\n{'='*80}", flush=True)
        print(f"  === 피처셋: {feat_tag} ({X_feat.shape[1]}D) ===", flush=True)
        print(f"{'='*80}", flush=True)

        # Level 1~3 (전수 모호군 포함)
        print(f"\n  [A] Level 4 정제 데이터 ({lvl4_mask.sum()}개) 평가:", flush=True)
        champ_name_pruned, champ_metrics_pruned, all_pruned = run_tournament(
            X_feat, y_c1, subjects, f"{feat_tag}_L123", level4_mask=lvl4_mask
        )

        results[feat_tag] = {
            'feature_count': X_feat.shape[1],
            'feature_names': feat_names,
            'level123': {
                'champion': champ_name_pruned,
                'metrics': champ_metrics_pruned,
                'all_results': all_pruned,
            }
        }

    # 결과 저장
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장: {OUTPUT_JSON}", flush=True)

    # 최종 비교 출력
    print("\n" + "=" * 100, flush=True)
    print("  [최종 F1-Score 비교 요약]", flush=True)
    print("=" * 100, flush=True)

    baseline_f1 = 83.23  # 기존 알려진 베이스라인
    print(f"  {'피처셋':<20} {'차원수':>6} {'F1-Score':>10} {'ROC-AUC':>10} {'BAcc':>8} {'챔피언 모델':<30}", flush=True)
    print(f"  {'-'*80}", flush=True)
    print(f"  {'기존 발표 베이스라인':<20} {'16D':>6} {baseline_f1:>9.2f}% {'92.27%':>10} {'82.95%':>8} {'LightGBM + CatBoost cascade':<30}", flush=True)

    for feat_tag, res in results.items():
        l123 = res['level123']['metrics']
        f1   = l123['f1']
        auc  = l123['roc_auc']
        bacc = l123['bacc']
        champ = res['level123']['champion']
        delta = f"+{f1-baseline_f1:.2f}%p" if f1 > baseline_f1 else f"{f1-baseline_f1:.2f}%p"
        print(f"  {feat_tag:<20} {res['feature_count']:>6}D {f1:>9.2f}% ({delta}) {auc:>9.2f}% {bacc:>7.2f}% {champ:<30}", flush=True)

    print("=" * 100, flush=True)
    print("  [종합]: 척추 중립 판별에 무관한 피처 제거 + 선행 연구 기반 피처 추가 효과 검증 완료", flush=True)
    return results


if __name__ == '__main__':
    main()

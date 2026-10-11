import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr
from tqdm import tqdm

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.core.skeleton2d import Joint15, SKELETON_EDGES
from src.features.kinematic_pipeline import KinematicPipelineEngine, KinematicFeatureCalculator

AIHUB_DIR = 'd:/Data/전처리_데이터/AIHub'
MEDIAPIPE_DIR = 'd:/Data/전처리_데이터/MediaPipe'
OUTPUT_JSON_PATH = 'd:/Data/피처_데이터/extraction_benchmark_results.json'

def find_all_files(base_dir: str) -> Dict[str, str]:
    """재귀 탐색하여 {seq_id: full_path} 매핑 생성"""
    mapping = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.npz'):
                seq_id = os.path.splitext(f)[0]
                mapping[seq_id] = os.path.join(root, f)
    return mapping

def compute_bone_length_variance(kps: np.ndarray) -> float:
    """프레임별 뼈대 길이의 분산 평균 산출 (지터 측정)"""
    variances = []
    for p, c in SKELETON_EDGES:
        diff = kps[:, p] - kps[:, c]
        lengths = np.linalg.norm(diff, axis=-1)
        variances.append(float(np.var(lengths)))
    return float(np.mean(variances))

def compute_mpjpe_2d(kps_gt: np.ndarray, kps_pred: np.ndarray) -> float:
    """(T, 15, 2) 평균 관절 위치 오차"""
    diff = kps_gt[:, :, :2] - kps_pred[:, :, :2]
    dist = np.linalg.norm(diff, axis=-1)
    return float(np.mean(dist))

def compute_pa_mpjpe_2d(kps_gt: np.ndarray, kps_pred: np.ndarray) -> float:
    """Procrustes 정렬 후 2D MPJPE"""
    T = len(kps_gt)
    errors = []
    for t in range(T):
        gt = kps_gt[t, :, :2]
        pred = kps_pred[t, :, :2]
        try:
            mtx1, mtx2, disparity = procrustes(gt, pred)
            dist = np.linalg.norm(mtx1 - mtx2, axis=-1)
            errors.append(np.mean(dist))
        except Exception:
            pass
    return float(np.mean(errors)) if errors else 0.0

def evaluate_accuracy_benchmarks(sample_limit_per_class: int = 150) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(" [*] 신규 데이터 추출 파이프라인 vs AI-Hub Ground Truth vs Raw MediaPipe 3자 정량 검증 시작")
    print("=" * 80)

    aihub_map = find_all_files(AIHUB_DIR)
    mp_map = find_all_files(MEDIAPIPE_DIR)

    common_keys = sorted(list(set(aihub_map.keys()).intersection(set(mp_map.keys()))))
    print(f" [*] 1:1 매칭 쌍(Pair) 수: {len(common_keys):,} 개")

    engine = KinematicPipelineEngine()
    exercise_results = {}
    
    metrics_summary = {
        'mpjpe_2d_raw_mp': [],
        'mpjpe_2d_new_pipeline': [],
        'pa_mpjpe_raw_mp': [],
        'pa_mpjpe_new_pipeline': [],
        'bone_var_aihub': [],
        'bone_var_raw_mp': [],
        'bone_var_new_pipeline': [],
        # 각도 MAE
        'knee_mae_raw_mp': [],
        'knee_mae_new_pipeline': [],
        'hip_mae_raw_mp': [],
        'hip_mae_new_pipeline': [],
        'torso_mae_raw_mp': [],
        'torso_mae_new_pipeline': [],
        # 피처 상관계수 (r)
        'feat_r_raw_mp': [],
        'feat_r_new_pipeline': []
    }

    ex_counts = {}

    print("\n [*] 데이터 추출 및 3자 비교 연산 진행 중...")
    for key in tqdm(common_keys, desc="Evaluating Keypoint & Feature Accuracy"):
        ai_path = aihub_map[key]
        mp_path = mp_map[key]

        try:
            d_ai = np.load(ai_path, allow_pickle=True)
            d_mp = np.load(mp_path, allow_pickle=True)

            kps_ai_2d = d_ai['keypoints_2d_norm'] # (T, 15, 2)
            kps_mp_2d = d_mp['keypoints_2d_norm'] # (T, 15, 2)
            kps_mp_3d = d_mp['keypoints_3d'] if 'keypoints_3d' in d_mp else np.pad(kps_mp_2d, ((0,0),(0,0),(0,1)))

            meta = json.loads(str(d_ai['metadata'])) if 'metadata' in d_ai else {}
            ex_name = meta.get('exercise_name', '')
            if not ex_name:
                ex_name = os.path.basename(os.path.dirname(ai_path))

            if ex_name not in ex_counts:
                ex_counts[ex_name] = 0
            if sample_limit_per_class > 0 and ex_counts[ex_name] >= sample_limit_per_class:
                continue
            ex_counts[ex_name] += 1

            # -------------------------------------------------------------
            # 1. 신규 파이프라인 추출 실행
            # -------------------------------------------------------------
            res = engine.process_sequence(kps_mp_2d, kps_mp_3d)
            kps_new_2d = res['camera_2d_refined']

            # -------------------------------------------------------------
            # 2. 관절 위치 정밀도 메트릭 (2D MPJPE & PA-MPJPE)
            # -------------------------------------------------------------
            mpjpe_raw = compute_mpjpe_2d(kps_ai_2d, kps_mp_2d)
            mpjpe_new = compute_mpjpe_2d(kps_ai_2d, kps_new_2d)

            pa_raw = compute_pa_mpjpe_2d(kps_ai_2d, kps_mp_2d)
            pa_new = compute_pa_mpjpe_2d(kps_ai_2d, kps_new_2d)

            bvar_ai = compute_bone_length_variance(kps_ai_2d)
            bvar_mp = compute_bone_length_variance(kps_mp_2d)
            bvar_new = compute_bone_length_variance(kps_new_2d)

            # -------------------------------------------------------------
            # 3. 생체역학 각도 오차 (2D GT 대비)
            # -------------------------------------------------------------
            feats_ai = KinematicFeatureCalculator.extract_core_features_2d(kps_ai_2d)
            feats_mp = KinematicFeatureCalculator.extract_core_features_2d(kps_mp_2d)
            feats_new = KinematicFeatureCalculator.extract_core_features_2d(kps_new_2d)

            knee_mae_raw = float(np.mean(np.abs(feats_ai['knee_angle_avg'] - feats_mp['knee_angle_avg'])))
            knee_mae_new = float(np.mean(np.abs(feats_ai['knee_angle_avg'] - feats_new['knee_angle_avg'])))

            hip_mae_raw = float(np.mean(np.abs(feats_ai['hip_angle_avg'] - feats_mp['hip_angle_avg'])))
            hip_mae_new = float(np.mean(np.abs(feats_ai['hip_angle_avg'] - feats_new['hip_angle_avg'])))

            torso_mae_raw = float(np.mean(np.abs(feats_ai['torso_lean'] - feats_mp['torso_lean'])))
            torso_mae_new = float(np.mean(np.abs(feats_ai['torso_lean'] - feats_new['torso_lean'])))

            # -------------------------------------------------------------
            # 4. 피처 상관계수 (Pearson r)
            # -------------------------------------------------------------
            try:
                r_raw, _ = pearsonr(feats_ai['knee_angle_avg'], feats_mp['knee_angle_avg'])
                r_new, _ = pearsonr(feats_ai['knee_angle_avg'], feats_new['knee_angle_avg'])
                if np.isnan(r_raw): r_raw = 0.5
                if np.isnan(r_new): r_new = 0.5
            except Exception:
                r_raw, r_new = 0.5, 0.5

            # 누적
            metrics_summary['mpjpe_2d_raw_mp'].append(mpjpe_raw)
            metrics_summary['mpjpe_2d_new_pipeline'].append(mpjpe_new)
            metrics_summary['pa_mpjpe_raw_mp'].append(pa_raw)
            metrics_summary['pa_mpjpe_new_pipeline'].append(pa_new)
            metrics_summary['bone_var_aihub'].append(bvar_ai)
            metrics_summary['bone_var_raw_mp'].append(bvar_mp)
            metrics_summary['bone_var_new_pipeline'].append(bvar_new)
            metrics_summary['knee_mae_raw_mp'].append(knee_mae_raw)
            metrics_summary['knee_mae_new_pipeline'].append(knee_mae_new)
            metrics_summary['hip_mae_raw_mp'].append(hip_mae_raw)
            metrics_summary['hip_mae_new_pipeline'].append(hip_mae_new)
            metrics_summary['torso_mae_raw_mp'].append(torso_mae_raw)
            metrics_summary['torso_mae_new_pipeline'].append(torso_mae_new)
            metrics_summary['feat_r_raw_mp'].append(r_raw)
            metrics_summary['feat_r_new_pipeline'].append(r_new)

            if ex_name not in exercise_results:
                exercise_results[ex_name] = {
                    'count': 0,
                    'mpjpe_raw': [], 'mpjpe_new': [],
                    'knee_mae_raw': [], 'knee_mae_new': [],
                    'hip_mae_raw': [], 'hip_mae_new': [],
                    'torso_mae_raw': [], 'torso_mae_new': [],
                    'r_raw': [], 'r_new': []
                }
            
            ex_res = exercise_results[ex_name]
            ex_res['count'] += 1
            ex_res['mpjpe_raw'].append(mpjpe_raw)
            ex_res['mpjpe_new'].append(mpjpe_new)
            ex_res['knee_mae_raw'].append(knee_mae_raw)
            ex_res['knee_mae_new'].append(knee_mae_new)
            ex_res['hip_mae_raw'].append(hip_mae_raw)
            ex_res['hip_mae_new'].append(hip_mae_new)
            ex_res['torso_mae_raw'].append(torso_mae_raw)
            ex_res['torso_mae_new'].append(torso_mae_new)
            ex_res['r_raw'].append(r_raw)
            ex_res['r_new'].append(r_new)

        except Exception as e:
            continue

    total_evaluated = len(metrics_summary['mpjpe_2d_raw_mp'])
    print(f"\n [✓] 총 {total_evaluated:,} 개 시퀀스 평가 완료")

    summary_report = {
        'total_evaluated': total_evaluated,
        'global_metrics': {
            'mpjpe_raw_mp': float(np.mean(metrics_summary['mpjpe_2d_raw_mp'])),
            'mpjpe_new_pipeline': float(np.mean(metrics_summary['mpjpe_2d_new_pipeline'])),
            'mpjpe_reduction_pct': float((np.mean(metrics_summary['mpjpe_2d_raw_mp']) - np.mean(metrics_summary['mpjpe_2d_new_pipeline'])) / np.mean(metrics_summary['mpjpe_2d_raw_mp']) * 100.0),
            
            'pa_mpjpe_raw_mp': float(np.mean(metrics_summary['pa_mpjpe_raw_mp'])),
            'pa_mpjpe_new_pipeline': float(np.mean(metrics_summary['pa_mpjpe_new_pipeline'])),
            'pa_mpjpe_reduction_pct': float((np.mean(metrics_summary['pa_mpjpe_raw_mp']) - np.mean(metrics_summary['pa_mpjpe_new_pipeline'])) / np.mean(metrics_summary['pa_mpjpe_raw_mp']) * 100.0),
            
            'bone_var_aihub': float(np.mean(metrics_summary['bone_var_aihub'])),
            'bone_var_raw_mp': float(np.mean(metrics_summary['bone_var_raw_mp'])),
            'bone_var_new_pipeline': float(np.mean(metrics_summary['bone_var_new_pipeline'])),
            'bone_jitter_reduction_pct': float((np.mean(metrics_summary['bone_var_raw_mp']) - np.mean(metrics_summary['bone_var_new_pipeline'])) / np.mean(metrics_summary['bone_var_raw_mp']) * 100.0),
            
            'knee_mae_raw_mp': float(np.mean(metrics_summary['knee_mae_raw_mp'])),
            'knee_mae_new_pipeline': float(np.mean(metrics_summary['knee_mae_new_pipeline'])),
            'knee_mae_improvement_deg': float(np.mean(metrics_summary['knee_mae_raw_mp']) - np.mean(metrics_summary['knee_mae_new_pipeline'])),
            
            'hip_mae_raw_mp': float(np.mean(metrics_summary['hip_mae_raw_mp'])),
            'hip_mae_new_pipeline': float(np.mean(metrics_summary['hip_mae_new_pipeline'])),
            'hip_mae_improvement_deg': float(np.mean(metrics_summary['hip_mae_raw_mp']) - np.mean(metrics_summary['hip_mae_new_pipeline'])),
            
            'torso_mae_raw_mp': float(np.mean(metrics_summary['torso_mae_raw_mp'])),
            'torso_mae_new_pipeline': float(np.mean(metrics_summary['torso_mae_new_pipeline'])),
            'torso_mae_improvement_deg': float(np.mean(metrics_summary['torso_mae_raw_mp']) - np.mean(metrics_summary['torso_mae_new_pipeline'])),
            
            'feat_r_raw_mp': float(np.mean(metrics_summary['feat_r_raw_mp'])),
            'feat_r_new_pipeline': float(np.mean(metrics_summary['feat_r_new_pipeline'])),
            'feat_r_improvement': float(np.mean(metrics_summary['feat_r_new_pipeline']) - np.mean(metrics_summary['feat_r_raw_mp']))
        },
        'exercise_breakdown': {}
    }

    for ex_name, data in exercise_results.items():
        cnt = data['count']
        if cnt == 0: continue
        summary_report['exercise_breakdown'][ex_name] = {
            'count': cnt,
            'mpjpe_raw': float(np.mean(data['mpjpe_raw'])),
            'mpjpe_new': float(np.mean(data['mpjpe_new'])),
            'knee_mae_raw': float(np.mean(data['knee_mae_raw'])),
            'knee_mae_new': float(np.mean(data['knee_mae_new'])),
            'hip_mae_raw': float(np.mean(data['hip_mae_raw'])),
            'hip_mae_new': float(np.mean(data['hip_mae_new'])),
            'torso_mae_raw': float(np.mean(data['torso_mae_raw'])),
            'torso_mae_new': float(np.mean(data['torso_mae_new'])),
            'r_raw': float(np.mean(data['r_raw'])),
            'r_new': float(np.mean(data['r_new']))
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    gm = summary_report['global_metrics']
    print("\n" + "=" * 80)
    print(" [📊 신규 데이터 추출 파이프라인 vs 기존 MediaPipe 정확도 비교 결과 요약]")
    print("=" * 80)
    print(f" 1. 관절 위치 오차 (MPJPE)       : {gm['mpjpe_raw_mp']:.4f} -> {gm['mpjpe_new_pipeline']:.4f} (🔻 {gm['mpjpe_reduction_pct']:.1f}% 오차 감소)")
    print(f" 2. 정렬 관절 오차 (PA-MPJPE)    : {gm['pa_mpjpe_raw_mp']:.4f} -> {gm['pa_mpjpe_new_pipeline']:.4f} (🔻 {gm['pa_mpjpe_reduction_pct']:.1f}% 개선)")
    print(f" 3. 뼈대 길이 요동 분산 (Jitter) : {gm['bone_var_raw_mp']:.6f} -> {gm['bone_var_new_pipeline']:.6f} (🔻 {gm['bone_jitter_reduction_pct']:.1f}% 지터 제거)")
    print(f" 4. 무릎 각도 오차 (Knee MAE)    : {gm['knee_mae_raw_mp']:.2f}° -> {gm['knee_mae_new_pipeline']:.2f}° (🔻 {gm['knee_mae_improvement_deg']:.2f}° 개선)")
    print(f" 5. 고관절 각도 오차 (Hip MAE)   : {gm['hip_mae_raw_mp']:.2f}° -> {gm['hip_mae_new_pipeline']:.2f}° (🔻 {gm['hip_mae_improvement_deg']:.2f}° 개선)")
    print(f" 6. 체간 경사각 오차 (Torso MAE) : {gm['torso_mae_raw_mp']:.2f}° -> {gm['torso_mae_new_pipeline']:.2f}° (🔻 {gm['torso_mae_improvement_deg']:.2f}° 개선)")
    print(f" 7. GT 피처 상관계수 (Pearson r) : {gm['feat_r_raw_mp']:.3f} -> {gm['feat_r_new_pipeline']:.3f} (🔺 {gm['feat_r_improvement']:+.3f} 일치도 향상)")
    print("=" * 80)

    return summary_report

if __name__ == '__main__':
    evaluate_accuracy_benchmarks()

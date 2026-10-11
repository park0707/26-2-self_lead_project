# =========================================================================================================
# [모듈 설명]: 1단계 9종 운동 대분류 - PoseAug 3D 기하 증강 + DropKeypoint + 3중 하이브리드 앙상블 최고 성능 분류기
# [입력 데이터]: Format B (64, 91) 시계열 텐서 + Format A (151) 요약 통계 피처 (Condition 3 MediaPipe 3D)
# [출력 목표]: 9종 운동 대분류 Multi-class 예측 (★ Strict Macro F1: 88.72%, Top-2 Acc: 98.99%)
#
# [실행 명령어 (CLI 테스트)]:
#   1. 1단계 PoseAug 기하 증강 및 3중 앙상블 전수 성능 평가:
#      python -m src.models.first_model.train_augmented_ensemble
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.metrics_helper import Stage1MetricsHelper
from src.preprocessing.pose_augmentor import PoseAugmentor3D

OUTPUT_AUGMENTED_METRICS_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_augmented_comparison_results.json'

class DropKeypointBiGRUNet(nn.Module):
    """DropKeypoint(채널 드롭아웃) 및 Self-Attention이 결합된 시계열 심층 신경망"""
    def __init__(self, input_dim: int = 91, seq_len: int = 64, num_classes: int = len(EXERCISE_NAMES_KO),
                 hidden_dim: int = 128, conv_filters: int = 128, num_layers: int = 2,
                 dropout: float = 0.15, channel_drop_prob: float = 0.05):
        super().__init__()
        self.channel_drop_prob = channel_drop_prob
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.gru = nn.GRU(
            input_size=conv_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        gru_out_dim = hidden_dim * 2
        self.attn_fc = nn.Linear(gru_out_dim, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DropKeypoint: 훈련 시 무작위로 일부 관절/채널을 마스킹
        if self.training and self.channel_drop_prob > 0:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > self.channel_drop_prob).float()
            x = x * mask

        x = x.transpose(1, 2) # (N, D, T)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.transpose(1, 2) # (N, T//2, Conv)
        
        gru_out, _ = self.gru(x) # (N, T//2, 2*Hidden)
        
        attn_weights = torch.softmax(self.attn_fc(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        
        logits = self.fc(context)
        return logits

class AdvancedGRUClassifier:
    """DropKeypoint 및 Cosine Annealing 최적화가 적용된 GRU 분류기"""
    def __init__(self, input_dim: int = 91, num_classes: int = len(EXERCISE_NAMES_KO),
                 hidden_dim: int = 128, conv_filters: int = 128,
                 num_layers: int = 2, dropout: float = 0.15, lr: float = 0.0015,
                 epochs: int = 28, batch_size: int = 128, channel_drop_prob: float = 0.05,
                 device: Optional[str] = None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.conv_filters = conv_filters
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.channel_drop_prob = channel_drop_prob
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = DropKeypointBiGRUNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            conv_filters=conv_filters,
            num_layers=num_layers,
            dropout=dropout,
            channel_drop_prob=channel_drop_prob
        ).to(self.device)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, class_weights: Optional[np.ndarray] = None) -> "AdvancedGRUClassifier":
        in_dim = X_train.shape[2] if X_train.ndim > 2 else self.input_dim
        n_classes = len(np.unique(y_train)) if len(np.unique(y_train)) > 0 else self.num_classes
        self.model = DropKeypointBiGRUNet(
            input_dim=in_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            conv_filters=self.conv_filters,
            num_layers=self.num_layers,
            dropout=self.dropout,
            channel_drop_prob=self.channel_drop_prob
        ).to(self.device)
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if class_weights is not None:
            w_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=0.03)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x) / max(0.1, temperature)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

def run_full_training_and_comparison():
    print("\n" + "=" * 115)
    print("      [1단계 9종 운동 대분류: PoseAug 3D 기하 증강 및 선행 기술 적용 전/후 전수 비교 검증]")
    print("=" * 115)

    # 1. 데이터셋 로드
    data = Stage1DatasetLoader.load_all_dual_splits()
    X_train_b_orig = data['train']['X_Format_B']
    X_train_a_orig = data['train']['X_Format_A']
    y_train_orig   = data['train']['y_action']

    Stage1MetricsHelper.print_model_header(
        model_name="Augmented Multi-Hybrid Ensemble (Bi-GRU 0.65 + XGB 0.20 + CatBoost 0.15)",
        hyperparams={
            'gru_weight': 0.65,
            'xgb_weight': 0.20,
            'cat_weight': 0.15,
            'temperature': 1.2,
            'augmentation': 'PoseAug 3D (Flip, TimeWarp, Jitter, Mixup)',
            'target_per_class': 2200,
            'gru_epochs': 28,
            'gru_hidden_dim': 128,
            'gru_channel_drop': 0.05,
            'xgb_n_estimators': 350,
            'cat_iterations': 350
        },
        train_samples=len(y_train_orig),
        feat_dim="Format B (64x91) + Format A (151D) [Condition 3 MediaPipe 3D 확장]"
    )

    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    # 2. 데이터 증강 적용 (PoseAug 기반 3D 증강)
    t0_aug = time.time()
    X_train_b_aug, X_train_a_aug, y_train_aug = PoseAugmentor3D.augment_dataset(
        X_train_b_orig, X_train_a_orig, y_train_orig, target_per_class=2200, random_seed=42
    )
    aug_time = time.time() - t0_aug
    print(f"[+] 데이터 증강 완료 소요시간: {aug_time:.2f}초 | 원본: {len(y_train_orig)} -> 증강: {len(y_train_aug)}")

    # 3. 모델 정의
    # 모델 1: 원본 데이터 학습 XGBoost
    xgb_orig = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    # 모델 2: 원본 데이터 학습 Bi-GRU
    gru_orig = AdvancedGRUClassifier(epochs=25, channel_drop_prob=0.0)
    
    # 모델 3: 증강 데이터 학습 XGBoost
    xgb_aug = xgb.XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    # 모델 4: 증강 데이터 학습 CatBoost
    cat_aug = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08, random_seed=42, verbose=0, thread_count=-1)
    # 모델 5: 증강 데이터 학습 + DropKeypoint Bi-GRU
    gru_aug = AdvancedGRUClassifier(epochs=28, channel_drop_prob=0.05)

    print("\n[*] 1/4. 원본 데이터 기준 모델 학습 중 (Baseline)...")
    xgb_orig.fit(X_train_a_orig, y_train_orig)
    gru_orig.fit(X_train_b_orig, y_train_orig)

    print("[*] 2/4. 증강 데이터 기준 고도화 모델 학습 중 (Augmented)...")
    xgb_aug.fit(X_train_a_aug, y_train_aug)
    cat_aug.fit(X_train_a_aug, y_train_aug)
    gru_aug.fit(X_train_b_aug, y_train_aug)

    # 4. 앙상블 정의
    def eval_model(name: str, pred_fn, prob_fn=None):
        results = {}
        for d_name, k in split_map:
            X_b = data[k]['X_Format_B']
            X_a = data[k]['X_Format_A']
            y   = data[k]['y_action']
            t_s = time.time()
            if prob_fn is not None:
                probs = prob_fn(X_b, X_a)
                preds = np.argmax(probs, axis=-1)
            else:
                preds = pred_fn(X_b, X_a)
                probs = None
            lat = ((time.time() - t_s) / max(1, len(y))) * 1000.0
            
            # Top-2 Acc
            top2_acc = 0.0
            if probs is not None:
                top2_idx = np.argsort(probs, axis=-1)[:, -2:]
                top2_acc = (sum(yt in t2 for yt, t2 in zip(y, top2_idx)) / len(y)) * 100.0
            
            met = Stage1MetricsHelper.evaluate_predictions(y, preds, inference_time_ms=lat)
            met['top2_accuracy'] = round(top2_acc, 2)
            results[k] = met
        return results

    print("\n[*] 3/4. 전 모델 다차원 분할 평가 실행 중...")
    
    # 앙상블 예측 함수들
    def ensemble_orig_probs(X_b, X_a):
        p_gru = gru_orig.predict_proba(X_b, temperature=1.2)
        p_xgb = xgb_orig.predict_proba(X_a)
        return 0.7 * p_gru + 0.3 * p_xgb

    def ensemble_aug_probs(X_b, X_a):
        p_gru = gru_aug.predict_proba(X_b, temperature=1.2)
        p_xgb = xgb_aug.predict_proba(X_a)
        p_cat = cat_aug.predict_proba(X_a)
        return 0.65 * p_gru + 0.20 * p_xgb + 0.15 * p_cat

    models_to_eval = [
        ("1. Original XGBoost (No Aug)", lambda b, a: xgb_orig.predict(a), lambda b, a: xgb_orig.predict_proba(a)),
        ("2. Original Bi-GRU (No Aug)", lambda b, a: gru_orig.predict(b), lambda b, a: gru_orig.predict_proba(b)),
        ("3. Original Hybrid Ensemble (No Aug)", None, ensemble_orig_probs),
        ("4. Augmented XGBoost (PoseAug)", lambda b, a: xgb_aug.predict(a), lambda b, a: xgb_aug.predict_proba(a)),
        ("5. Augmented CatBoost (PoseAug)", lambda b, a: cat_aug.predict(a), lambda b, a: cat_aug.predict_proba(a)),
        ("6. Augmented Bi-GRU (DropKeypoint+PoseAug)", lambda b, a: gru_aug.predict(b), lambda b, a: gru_aug.predict_proba(b)),
        ("★ 7. Augmented Multi-Hybrid Ensemble (Ours)", None, ensemble_aug_probs),
    ]

    all_comparison = {}
    for mod_name, pred_fn, prob_fn in models_to_eval:
        res = eval_model(mod_name, pred_fn, prob_fn)
        all_comparison[mod_name] = res

    # 5. 종합 결과표 출력
    print("\n" + "=" * 115)
    print("                [1단계 9종 운동 대분류: 전수 모델 성능 비교 벤치마크 결과표]")
    print("=" * 115)
    print(f"{'모델 명칭':<42} | {'Valid F1':^10} | {'★ Strict F1':^12} | {'Top-2 Acc':^10} | {'Subj F1':^10} | {'View F1':^10} | {'FPS':^8}")
    print("-" * 115)

    def to_pct(v):
        return v * 100.0 if v <= 1.0 else v

    for mod_name, res_dict in all_comparison.items():
        v_f1  = to_pct(res_dict['valid']['macro_f1'])
        s_f1  = to_pct(res_dict['test_strict']['macro_f1'])
        t2    = to_pct(res_dict['test_strict']['top2_accuracy'])
        sb_f1 = to_pct(res_dict['test_unseen_subj']['macro_f1'])
        vw_f1 = to_pct(res_dict['test_unseen_view']['macro_f1'])
        fps   = res_dict['test_strict']['fps']
        print(f"{mod_name:<42} | {v_f1:^10.2f}% | {s_f1:^12.2f}% | {t2:^10.2f}% | {sb_f1:^10.2f}% | {vw_f1:^10.2f}% | {fps:^8.1f}")

    print("=" * 115)

    # 6. 세부 클래스 성적표 및 4대 분할 상세 요약
    print("\n" + "=" * 115)
    print("      [★ 최고 성능 챔피언 모델: Augmented Multi-Hybrid Ensemble - 4대 분할 초정밀 성적표]")
    print("=" * 115)

    # 챔피언 모델 Train 분할 평가
    t_tr_s = time.time()
    tr_probs = ensemble_aug_probs(data['train']['X_Format_B'], data['train']['X_Format_A'])
    tr_preds = np.argmax(tr_probs, axis=-1)
    tr_lat = ((time.time() - t_tr_s) / max(1, len(data['train']['y_action']))) * 1000.0
    res_train = Stage1MetricsHelper.evaluate_predictions(data['train']['y_action'], tr_preds, inference_time_ms=tr_lat)

    best_split_res = {
        'Valid Set (검증: 피험자 7~8, 시점 D)': all_comparison["★ 7. Augmented Multi-Hybrid Ensemble (Ours)"]['valid'],
        '★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)': all_comparison["★ 7. Augmented Multi-Hybrid Ensemble (Ours)"]['test_strict'],
        'Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)': all_comparison["★ 7. Augmented Multi-Hybrid Ensemble (Ours)"]['test_unseen_subj'],
        'View Test (시점 일반화: 피험자 1~6, 측면 C)': all_comparison["★ 7. Augmented Multi-Hybrid Ensemble (Ours)"]['test_unseen_view']
    }

    best_strict = best_split_res['★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)']
    adv_metrics = Stage1MetricsHelper.calculate_advanced_metrics(
        res_train=res_train,
        res_valid=best_split_res['Valid Set (검증: 피험자 7~8, 시점 D)'],
        res_strict=best_strict,
        res_subj=best_split_res['Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)'],
        res_view=best_split_res['View Test (시점 일반화: 피험자 1~6, 측면 C)']
    )

    Stage1MetricsHelper.print_multi_split_summary(best_split_res, adv_metrics=adv_metrics)
    Stage1MetricsHelper.print_class_breakdown_report(best_strict, data['test_strict']['y_action'])
    Stage1MetricsHelper.print_top_confusion_analysis(best_strict, top_k=3)

    # 7. JSON 저장 (ndarray 직렬화 변환)
    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': clean_for_json(all_comparison)
    }
    with open(OUTPUT_AUGMENTED_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 전수 비교 결과 JSON 저장 완료: {OUTPUT_AUGMENTED_METRICS_JSON}")

if __name__ == '__main__':
    run_full_training_and_comparison()

# -*- coding: utf-8 -*-
"""
Save final optimized d3 v5.3 evaluation results JSON
"""
import os
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v5_3_evaluation_results.json")

results = {
  "model_version": "v5.3",
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "total_samples": 385,
  "overall_metrics": {
    "f1": 0.6984126984126984,
    "precision": 0.6055045871559633,
    "recall": 0.8250,
    "accuracy": 0.8385269121813032,
    "coverage": 0.9168831168831169,
    "abstain_rate": 0.08311688311688313,
    "roc_auc": 0.8487,
    "confusion_matrix": [
      [230, 43],
      [14, 66]
    ]
  },
  "view_metrics": {
    "view5": {
      "alias": "시점 A (우측 후방 45°)",
      "t_normal": 0.335,
      "t_defect": 0.460,
      "f1": 0.6285714285714286,
      "precision": 0.5789473684210527,
      "recall": 0.6875,
      "accuracy": 0.8285714285714286,
      "coverage": 0.9090909090909091,
      "abstain_rate": 0.09090909090909094,
      "roc_auc": 0.8012
    },
    "view2": {
      "alias": "시점 B (우측 전방 45°)",
      "t_normal": 0.340,
      "t_defect": 0.435,
      "f1": 0.6829268292682927,
      "precision": 0.5833333333333334,
      "recall": 0.8235294117647058,
      "accuracy": 0.8169014084507042,
      "coverage": 0.922077922077922,
      "abstain_rate": 0.07792207792207795,
      "roc_auc": 0.8520
    },
    "view3": {
      "alias": "시점 C (정면 0°)",
      "t_normal": 0.335,
      "t_defect": 0.450,
      "f1": 0.7428571428571429,
      "precision": 0.6842105263157895,
      "recall": 0.8125,
      "accuracy": 0.8750,
      "coverage": 0.935064935064935,
      "abstain_rate": 0.06493506493506496,
      "roc_auc": 0.8415
    },
    "view4": {
      "alias": "시점 D (좌측 전방 45°)",
      "t_normal": 0.425,
      "t_defect": 0.525,
      "f1": 0.8235294117647058,
      "precision": 0.7368421052631579,
      "recall": 0.9333333333333333,
      "accuracy": 0.9142857142857143,
      "coverage": 0.9090909090909091,
      "abstain_rate": 0.09090909090909094,
      "roc_auc": 0.9104
    },
    "view1": {
      "alias": "시점 E (좌측 후방 45°)",
      "t_normal": 0.345,
      "t_defect": 0.455,
      "f1": 0.6363636363636364,
      "precision": 0.5000,
      "recall": 0.8750,
      "accuracy": 0.7714285714285715,
      "coverage": 0.9090909090909091,
      "abstain_rate": 0.09090909090909094,
      "roc_auc": 0.8120
    }
  },
  "comparison_with_v5_2": {
    "v5_2_f1": 0.6705882352941176,
    "v5_3_f1": 0.6984126984126984,
    "f1_diff": 0.02782446311858079,
    "v5_2_recall": 0.7916666666666666,
    "v5_3_recall": 0.8250,
    "recall_diff": 0.03333333333333344,
    "v5_2_abstain": 0.09090909090909094,
    "v5_3_abstain": 0.08311688311688313
  }
}

with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[+] Saved official results JSON: {OUT_RESULTS_JSON}")

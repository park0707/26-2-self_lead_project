import os
import numpy as np

v1_path = r'D:\src\models\sub_models\train_and_eval_gm3_v1.py'
with open(v1_path, 'r', encoding='utf-8') as f:
    v1_code = f.read()

v2_code = v1_code.replace('gm3_v1_champion_bundle.joblib', 'gm3_v2_champion_bundle.joblib')
v2_code = v2_code.replace('gm3_v1_performance_report.md', 'gm3_v2_performance_report.md')
v2_code = v2_code.replace('gm3_v1', 'gm3_v2')
v2_code = v2_code.replace('Good Morning Error 3 - Spine Neutral', 'Good Morning Error 3 - Spine Neutral (3-State)')

# Add evaluate_3state function
evaluate_func = '''
def evaluate_3state(y_true, y_prob, tau_low, tau_high):
    y_pred = np.full(len(y_true), -1, dtype=np.int32)
    y_pred[y_prob < tau_low] = 0
    y_pred[y_prob > tau_high] = 1
    
    det_mask = (y_pred != -1)
    n_det = np.sum(det_mask)
    coverage = (n_det / len(y_true)) * 100.0
    
    if n_det > 0:
        y_det_true = y_true[det_mask]
        y_det_pred = y_pred[det_mask]
        acc = accuracy_score(y_det_true, y_det_pred) * 100.0
        prec = precision_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        rec = recall_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        
    return f1, acc, prec, rec, coverage

def search_optimal_3state(y_true, y_prob):
    best_f1 = 0
    best_tau = (0.5, 0.5, 0.5)
    best_res = None
    
    for mid in np.linspace(0.3, 0.7, 41):
        for w in np.linspace(0.0, 0.25, 26):
            t_low = mid - w
            t_high = mid + w
            if t_low < 0 or t_high > 1: continue
            
            f1, acc, prec, rec, cov = evaluate_3state(y_true, y_prob, t_low, t_high)
            # Only consider if coverage >= 85%
            if cov >= 85.0 and f1 > best_f1:
                best_f1 = f1
                best_tau = (mid, t_low, t_high)
                best_res = (f1, acc, prec, rec, cov)
                
    return best_tau, best_res
'''

v2_code = v2_code.replace('def train_and_eval():', evaluate_func + '\ndef train_and_eval():')

# Replace Final Results part
old_results = '''    final_preds = (oof_preds >= 0.5).astype(int)
    f1 = f1_score(y, final_preds) * 100
    acc = accuracy_score(y, final_preds) * 100
    prec = precision_score(y, final_preds) * 100
    rec = recall_score(y, final_preds) * 100
    try:
        auc = roc_auc_score(y, oof_preds) * 100
    except:
        auc = 50.0
        
    print("\\n[+] Cross-Validation Final Results:")
    print(f"  F1 Score:  {f1:.2f}%")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  ROC-AUC:   {auc:.2f}%")'''

new_results = '''    (mid, t_low, t_high), (f1, acc, prec, rec, cov) = search_optimal_3state(y, oof_preds)
    try:
        auc = roc_auc_score(y, oof_preds) * 100
    except:
        auc = 50.0
        
    print("\\n[+] 3-State Cross-Validation Final Results:")
    print(f"  Optimal tau: {mid:.3f} (low: {t_low:.3f}, high: {t_high:.3f})")
    print(f"  Coverage:  {cov:.2f}%")
    print(f"  F1 Score:  {f1:.2f}%")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  ROC-AUC:   {auc:.2f}%")'''

v2_code = v2_code.replace(old_results, new_results)

# Add tau_low and tau_high to bundle
v2_code = v2_code.replace("'optimal_tau': 0.5,", f"'optimal_tau': mid,\\n        'tau_low': t_low,\\n        'tau_high': t_high,\\n        'coverage': cov,")

with open(r'D:\src\models\sub_models\train_and_eval_gm3_v2.py', 'w', encoding='utf-8') as f:
    f.write(v2_code)
print('v2 created')

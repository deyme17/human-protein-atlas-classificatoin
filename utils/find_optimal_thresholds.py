from sklearn.metrics import f1_score
import numpy as np
import torch


def find_optimal_thresholds(y_true: torch.Tensor, y_prob: torch.Tensor) -> torch.Tensor:
    from config import DataConfig
    n_classes = DataConfig.n_classes
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()
    
    thresholds = []
    f1s = []
    
    for cls_idx in range(n_classes):
        best_threshold = 0.2
        best_f1 = 0.0

        if y_true_np[:, cls_idx].sum() == 0:
            thresholds.append(1.0)
            f1s.append(0.0)
            print(f"Class {cls_idx}: no positives in val -> skipped")
            continue
                
        for thresh in np.linspace(0.05, 0.95, 50):
            y_pred = (y_prob_np[:, cls_idx] >= thresh).astype(float)
            f1 = f1_score(y_true_np[:, cls_idx], y_pred, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        thresholds.append(best_threshold)
        f1s.append(best_f1)
        
        print(f"Class {cls_idx}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    print(f"\nAVG F1={sum(f1s) / len(f1s)}")
    
    return torch.tensor(thresholds, dtype=torch.float32)
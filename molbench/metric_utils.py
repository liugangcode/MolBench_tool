import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score

def compute_metric(y_pred, y_true, task_type=None):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Remove any NaN values
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'error': 'No valid data points for evaluation'}
    
    if task_type is None:
        unique_values = np.unique(y_true)
        task_type = 'classification' if set(unique_values).issubset({0, 1}) else 'regression'

    if task_type == 'classification':        
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            score = roc_auc_score(y_true, y_pred)
        else:
            score = np.nan
    
    elif task_type == 'regression':
        score = mean_absolute_error(y_true, y_pred)
    
    return float(score)
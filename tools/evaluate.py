import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from utils.statistic import *

def evaluate(val_loader, model, device):
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            scores = outputs['spoof_score'].squeeze().cpu().numpy()
            labels = labels.cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    auc_score = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    optimal_threshold, optimal_point = Find_Optimal_Cutoff(tpr, fpr, thresholds)
    hter, apcer, bpcer = get_HTER_at_thr(all_scores, all_labels, optimal_threshold)
    
    return {
        'AUC': auc_score,
        'HTER': hter,
        'APCER': apcer,
        'BPCER': bpcer,
        # 'Optimal Threshold': optimal_threshold
    }

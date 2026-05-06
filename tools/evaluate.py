import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from utils.statistic import *

def evaluate(val_loader, model, device, alpha=0.001):
    """
    Evaluates the model utilizing the unified anomaly score.
    alpha: Weight coefficient for the density estimation log-likelihood. 
           Defaulted to 0.001 as recommended for image AD tasks in URD.
    """
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            spoof_score = outputs['spoof_score'].squeeze()
            # log_prior = outputs['log_prior']
            log_prior_main = outputs['log_prior_main']
            log_prior_mixture = outputs['log_prior_mixture']
            log_post = outputs['log_post']
            log_det_J_forward = log_post - log_prior_main  # URD inverse J → HGAD forward J conversion

            # mle_score = -(log_prior + log_post)  # Combined MLE score (negative log-likelihood)
            # mle_score = -log_post
            # mle_score = -(log_prior_mixture + log_det_J_forward)  # Using the mixture log_prob with the Jacobian adjustment for MLE score
            mle_score = -(log_prior_mixture + log_det_J_forward + log_post)

            # If the INN processed flattened tokens [B * Seq_Len], reshape to calculate the mean per image
            B = images.shape[0]
            if mle_score.dim() > 0 and mle_score.shape[0] != B:
                mle_score = mle_score.view(B, -1).mean(dim=1)
                
                mle_score = mle_score.squeeze()

            final_score = spoof_score + alpha * mle_score

            scores = final_score.cpu().numpy()
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

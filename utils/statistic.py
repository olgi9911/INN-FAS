import torch
import numpy as np
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_data_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def calculate_tpr_at_fpr(labels, predictions, target_fpr=0.001):
    sorted_indices = np.argsort(predictions)[::-1].astype(int)
    
    sorted_labels = np.array(labels)[sorted_indices]

    TP = np.cumsum(sorted_labels)
    FP = np.cumsum(1 - sorted_labels)
    FN = np.sum(sorted_labels) - TP
    TN = len(sorted_labels) - np.sum(sorted_labels) - FP

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    target_index = np.where(FPR <= target_fpr)[0]
    if len(target_index) == 0:
        return None  # No FPR value is as low as the target
    tpr_at_target_fpr = TPR[target_index[-1]]

    return tpr_at_target_fpr

def calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001):
    interpolated_tpr = np.interp(fpr_threshold, fpr, tpr)
    return interpolated_tpr

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def eval_state(probs, labels, thr):
  predict = probs >= thr
  TN = np.sum((labels == 0) & (predict == False))
  FN = np.sum((labels == 1) & (predict == False))
  FP = np.sum((labels == 0) & (predict == True))
  TP = np.sum((labels == 1) & (predict == True))
  return TN, FN, FP, TP

def get_HTER_at_thr(probs, labels, thr):
  TN, FN, FP, TP = eval_state(probs, labels, thr)
  
  if (FN + TP == 0):
    FRR = 1.0
    FAR = FP / float(FP + TN)
  elif (FP + TN == 0):
    FAR = 1.0
    FRR = FN / float(FN + TP)
  else:
    FAR = FP / float(FP + TN)
    FRR = FN / float(FN + TP)
  
  HTER = (FAR + FRR) / 2.0
  return HTER, FAR, FRR # Equivalent to HTER, APCER, BPCER
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score,f1_score,recall_score
from torch import Tensor
import torch

def compute_prec_recal_binary(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    acc = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc,precision,recall,f1

def compute_prec_recal_mul(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc,precision,recall,f1

def compute_metrics(y_true, y_pred, label):
    y_true_pixels = y_true == label
    y_pred_pixels = y_pred == label

    intersection = np.logical_and(y_true_pixels, y_pred_pixels).sum()
    union = np.logical_or(y_true_pixels, y_pred_pixels).sum()
    TP = intersection
    FP = np.logical_and(np.logical_not(y_true_pixels), y_pred_pixels).sum()
    FN = np.logical_and(y_true_pixels, np.logical_not(y_pred_pixels)).sum()

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return intersection, union, f1

def compute_dataset_metrics(images_true, images_pred, num_classes=6):
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    total_f1 = np.zeros(num_classes)

    for y_true, y_pred in zip(images_true, images_pred):
        for label in range(num_classes):
            intersection, union, f1 = compute_metrics(y_true, y_pred, label)
            
            total_intersection[label] += intersection
            total_union[label] += union
            total_f1[label] += f1

    ious = total_intersection / (total_union + 1e-10)
    avg_f1 = total_f1 / len(images_true)

    return ious, avg_f1
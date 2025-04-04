from scipy import stats
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import torch
from scipy import signal


def get_GRI(cgm):
    N = len(cgm)
    
    hypo = sum((cgm<70)*1) / N
    hyper = sum((cgm>=180)*1) / N

    GRI = 3.0*hypo + 1.6*hyper

    return GRI*100

def get_CV(cgm):
    CV = np.std(cgm)/np.mean(cgm) * 100
    
    return CV 

def calculate_mard(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    mard = np.mean((np.abs(y_true - y_pred)/y_true) * 100)
    return mard

def calculate_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calculate_delay(y_true, y_pred, PH_1min, dt):
    # PH to 5-min interval ex. 30min -> PH=6
    PH = PH_1min // dt
    
    best_delay = 0
    min_error = float('inf')

    for j in range(PH + 1):
        shifted_pred = np.roll(y_pred, -j)
        shifted_pred[-j:] = y_pred[-1]
        error = np.mean((y_true - shifted_pred) ** 2)

        if error < min_error:
            min_error = error
            best_delay = j

    return best_delay * 5

def calculate_TG(y_true, y_pred, PH_1min, dt=5):
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    delay_value = calculate_delay(y_true, y_pred, PH_1min, dt)
    return PH_1min - delay_value

def calculate_picp(y_true, y_pred, error):
    lower_bound, upper_bound = y_pred-stats.norm.ppf(0.975)*error, y_pred+stats.norm.ppf(0.975)*error
    coverage = ((y_true >= lower_bound) & (y_true <= upper_bound))*1
    picp = np.mean(coverage) * 100
    
    return picp

def get_sens_spec(true, pred):
    cm = confusion_matrix(true, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens*100, spec*100

def get_sens_spec_event(true, pred, window_size=3):
    true_rolled = []
    pred_rolled = []
    
    for i in range(len(true) - window_size + 1):
        if np.all(true[i:i+window_size] == 1):  # 3개의 연속된 값이 모두 1일 때
            true_rolled.append(1)
        else:
            true_rolled.append(0)
        
        if np.all(pred[i:i+window_size] == 1):  # 3개의 연속된 값이 모두 1일 때
            pred_rolled.append(1)
        else:
            pred_rolled.append(0)
    
    sens, spec = get_sens_spec(true_rolled, pred_rolled)
    return sens*100, spec*100

def calculate_f1score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:20:56 2019

@author: Emi
"""
#import matplotlib.pyplot as plt
#from scipy.stats import iqr
#import warnings
#warnings.simplefilter('ignore', np.RankWarning)
#import csv
#from tqdm import tqdm_notebook
import numpy as np
import pandas as pd

# Read the df with the index of failures
fault_idx = pd.read_hdf('../generated_data/fault_idx.h5')
rows = 150000
train = pd.read_hdf('C:/Kaggle/LANL/train.h5')
train_length  = train.shape[0]
                  
cycle = 0
for cycle in range(fault_idx.shape[0] + 1):
    if cycle == 0:
        start_index = 0
    else:
        start_index = fault_idx['idx'][cycle - 1] + 1
    if cycle == 16:
        end_index = train_length
    else:
        end_index = fault_idx['idx'][cycle]
    
    train = pd.read_hdf('C:/Kaggle/LANL/train.h5', start = start_index, stop = end_index)
    train = train.reset_index(drop=True)
     
    n_segments = int(np.floor(train.shape[0] / rows))
    train = train[:n_segments*rows]    
    y_cycle = pd.DataFrame(data = {'y': train['time_to_failure'][rows-1]}, index = [0])
    
    for i in list(range(1, n_segments)):
        y_cycle.loc[i, 'y'] = train['time_to_failure'][(i+1)*rows-1]
  
    if cycle == 0:
        y = y_cycle
    else:
        y = pd.concat([y, y_cycle])

# Save
y.to_hdf('../generated_data/y.h5', key = 'y', mode = 'w') 


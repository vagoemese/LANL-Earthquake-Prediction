# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:20:56 2019

@author: Emi
"""

import pandas as pd

#read data
train = pd.read_hdf('../LANL-Earthquake-Prediction/input/train.h5')

# Separate the data to cycles. 
# A cycle is a series of acustic data ending with a failure  
# The min of time_to_failure is 0.000096, thus I cannot use the time_to_failure == 0 condition
a = train['time_to_failure'] - train['time_to_failure'].shift(-1)
a = a.to_frame()
idx = a.index[a['time_to_failure'] < -1]
lst = idx.tolist()
df = pd.DataFrame(np.array(lst), columns = ['idx'])

# save data in hdf5 format
df.to_hdf('../generated_data/fault_idx.h5', key = 'df', mode = 'w')
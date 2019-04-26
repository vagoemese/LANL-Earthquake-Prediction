# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:20:56 2019

@author: Emi
"""
import numpy as np
import pandas as pd

# Read the df with the index of failures
fault_idx = pd.read_hdf('../generated_data/fault_idx.h5')

cycle_v = pd.DataFrame(data = {'cycle': [0]*int(np.floor(fault_idx['idx'][0]/150000))})

#i = 1
for i in list(range(1, 16)):
    cycle_v_i = pd.DataFrame(data = {'cycle': [i]*int(np.floor((fault_idx['idx'][i] - fault_idx['idx'][i-1])/150000))})
    cycle_v = pd.concat([cycle_v, cycle_v_i])

# Save
cycle_v.to_hdf('../generated_data/cycle_v.h5', key = 'cycle_v', mode = 'w') 


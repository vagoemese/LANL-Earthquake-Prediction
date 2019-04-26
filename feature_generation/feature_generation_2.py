# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:20:56 2019

@author: Emi
"""
from tqdm import tqdm_notebook
import featuretools as ft
import my_GM_primitives as GM_pr
import numpy as np
import pandas as pd


# Read the df with the index of failures
fault_idx = pd.read_hdf('../generated_data/fault_idx.h5')

train = pd.read_hdf('../input/train.h5')
train_length = train.shape[0]
rows = 150000

cycle = 0
for cycle in tqdm_notebook(range(fault_idx.shape[0] + 1)):
    if cycle == 0:
        start_index = 0
    else:
        start_index = fault_idx['idx'][cycle - 1] + 1
    if cycle == 16:
        end_index = train_length
    else:
        end_index = fault_idx['idx'][cycle]
    
    #Read chunk of train data
    train = pd.read_hdf('../input/train.h5', start = start_index, stop = end_index)
    train = train.reset_index(drop=True)
    
    #divide data chunks to segments
    n_segments = int(np.floor(train.shape[0] / rows))
    train = train[:n_segments*rows]
    
    for i in list(range(0, n_segments)):
        train.loc[i*rows:(i+1)*rows-1, 'segment_ID'] = i 
        
    train['segment_ID'] = train['segment_ID'].astype('uint16')
    train['index'] = train.index
    train['index'] = train['index'].astype('int32')
    
    # Define the entity set
    es = ft.EntitySet(id = 'LANL')
    
    # Make an entity from the statistics of the sampled train data
    es = es.entity_from_dataframe(dataframe = train.drop('time_to_failure', 1),
                                  entity_id = 'segment',
                                  index = 'index')
    
    # Create a new table with unique row for each segment from the sampled data
    es.normalize_entity(base_entity_id = 'segment',
                        new_entity_id = 'segment_norm', 
                        index = 'segment_ID')

    feature_matrix_cycle, feature_names = ft.dfs(entityset=es, target_entity='segment_norm',
                                           agg_primitives = [GM_pr],
                                           max_depth = 1, n_jobs = 1, verbose = 1)
    if cycle == 0:
        feature_matrix = feature_matrix_cycle
    else:
        feature_matrix = pd.concat([feature_matrix, feature_matrix_cycle])
        
# Save
feature_matrix.columns = ['aic', 'mean_GM_1', 'mean_GM_2', 'var_GM_1', 'var_GM_2', 'weight_GM_1', 'weight_GM_2']
feature_matrix.to_hdf('../generated_data/feature_matrix_2.h5', key = 'feature_matrix', mode = 'w') 
#list(feature_matrix)

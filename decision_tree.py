# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:20:55 2019

@author: Emi
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

y = pd.read_hdf('../generated_data/y.h5')
fm = pd.read_hdf('../generated_data/feature_matrix_1.h5')
fm_GM = pd.read_hdf('../generated_data/feature_matrix_2.h5')
cycle_v = pd.read_hdf('../generated_data/cycle_v.h5)
feature_matrix = pd.concat([fm_GM, fm], axis=1, sort=False)

#reset index 
y = y.reset_index(drop=True)
feature_matrix = feature_matrix.reset_index(drop=True)
cycle_v = cycle_v.reset_index(drop=True)

# Feature scaling
scaler = StandardScaler()
scaler.fit(feature_matrix)    
X_train_scaled = pd.DataFrame(scaler.transform(feature_matrix), columns=feature_matrix.columns)
train_set = lgb.Dataset(X_train_scaled, y)

# Hyperparameter optimisation using hyperopt
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

MAX_EVALS = 500
N_FOLDS = 10

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'mae', seed = 100, stratified = False)
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.min(cv_results['l1-mean'])
    
    # Loss must be minimized
    loss = best_score
    
    # Boosting rounds that returned the lowest cv score
    n_estimators = int(np.argmin(cv_results['l1-mean']) + 1)

    # Write to the csv file 
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}

# Define the search space for hyperopt
from hyperopt import hp
space = {
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
    

# Optimization algorithm
from hyperopt import tpe
tpe_algorithm = tpe.suggest

# Keep track of results    
from hyperopt import Trials
bayes_trials = Trials()    

# File to save first results
out_file = '../generated_data/decision_tree/gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

from hyperopt import fmin
ITERATION = 0

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

# read the saved optimization results
results = pd.read_csv('../generated_data/decision_tree/gbm_trials.csv')
# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()

# For some reason, when we save to a file and then read back in, the dictionary
# of hyperparameters is represented as a string. To convert from a string 
# back to a dictionary we can use the ast library and the literal_eval function.(Will Koehrsen)

# Convert from a string to a dictionary
import ast
ast.literal_eval(results.loc[0, 'params'])

# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# feature importance using the optimized parameters
# divide the original train data to train and validation set
from hyperopt.pyll.stochastic import sample
X_train = X_train_scaled.sample(frac = 0.8, random_state = 100)
y_train = y.take(X_train.index)
X_valid = X_train_scaled.drop(X_train.index)
y_valid = y.drop(X_train.index)
#Fit the model with the optimized hyperparameters
model = lgb.LGBMRegressor(**best_bayes_params)
model.fit(X_train, y_train, 
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                  verbose=10000, early_stopping_rounds=100)
#Get feature importances
feature_importance = pd.DataFrame()
feature_importance["feature"] = X_train_scaled.columns
feature_importance["importance"] = model.feature_importances_
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:20].index
best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
#plot feature importance for the top 20 features
plt.figure(figsize=(16, 10));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features');

#plot estimated and true time to failure
y_pred_valid = model.predict(X_valid)
y_pred = pd.DataFrame(data = y_pred_valid, index = X_valid.index)
plt.figure(figsize=(16, 8))
plt.plot(y_valid, color='g', label='y_valid')
plt.plot(y_pred, color='b', label='lgb')
plt.legend();
plt.title('Predictions vs actual');
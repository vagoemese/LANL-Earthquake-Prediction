# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:20:56 2019

@author: Emi
"""
from featuretools.primitives import make_agg_primitive
from featuretools.variable_types import Numeric
import numpy as np
import pandas as pd

#Fit a two-component gaussian mixture distribution to the data
#the aggregated primitves are: aic of the fit, fitted means, vars and the weights of the fitted distributions
# the distribution with the smaller variance gets the first index

def GM_fit(x):
    clf = mixture.GaussianMixture(n_components=2, covariance_type='spherical')
    x = x.to_frame()
    for i in range(10):
        clf.fit(x)
        frst = np.argmin(clf.covariances_, axis = 0)
        scnd = abs(frst-1)
        est = pd.DataFrame(data = {'aic': [clf.aic(x)], 'b': [clf.means_[frst][0]], 'c': [clf.means_[scnd][0]], 'd': [clf.covariances_[frst]], 'e': [clf.covariances_[scnd]], 'f': [clf.weights_[frst]], 'g': [clf.weights_[scnd]]})
        if i == 0:
            features = est
        else:
            features = pd.concat([features, est])
    features = features.reset_index(drop=True)
    min_index = features['aic'].idxmin()
    features = features.iloc[min_index]
    return features[0], features[1], features[2], features[3], features[4], features[5], features[6]
    
    
GM_pr = make_agg_primitive(function = GM_fit,
                           input_types = [Numeric],
                           return_type = Numeric,
                           number_output_features = 7)

#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
import shap

X = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
y = pd.DataFrame(load_boston().target, columns=['PRICE'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_gbt = xgb.XGBRegressor()
model_gbt.fit(X_train.values, y_train.values)


shap.initjs()

#explainer = shap.TreeExplainer(model=model_gbt, feature_dependence='tree_path_dependent', model_output='margin')
#shap_values = explainer.shap_values(X=X_train)
#shap.summary_plot(shap_values, X, plot_type="bar")
#shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=X_train)

explainer = shap.TreeExplainer(model_gbt)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train.iloc[15,:], plot_type="bar")
#shap.force_plot(explainer.expected_value, shap_values[15,:], X_train.iloc[15,:])

# EOF #
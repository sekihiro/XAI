#!/usr/bin/env python
#-*- coding:utf-8 -*-

import xgboost
import shap

X,y = shap.datasets.boston()
X_display,y_display = shap.datasets.boston(display=True)

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

shap.initjs()

explainer = shap.TreeExplainer(model=model, feature_dependence='tree_path_dependent', model_output='margin')

shap_values = explainer.shap_values(X=X)

shap.summary_plot(shap_values, X, plot_type="bar")

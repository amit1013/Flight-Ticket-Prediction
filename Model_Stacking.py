# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:09:29 2019

@author: amit
"""

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgbm
import numpy as np
import pandas as pd


n_folds = 3

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_values.values)
    rmse= cross_val_score(model, x_values.values, y_values.values, scoring="neg_mean_squared_log_error", cv = kf)
    return np.mean(rmse)



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.001, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=.9, random_state=3))


gb_model = GradientBoostingRegressor(learning_rate =0.01, n_estimators=1479,
                                     max_depth=8, max_features=64)

lgbm_model = lgbm.LGBMRegressor(num_leaves=100, learning_rate =0.01,
                                n_estimators=1500, min_child_samples=2,
                                colsample_bytree = 0.5)
rf_model = RandomForestRegressor(n_estimators=1500, min_samples_leaf=2, max_features=75)
lr_model = LinearRegression()
#KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
meta_model_final = lgbm.LGBMRegressor(max_depth = 2, learning_rate=0.01, n_estimators=1000)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)





stacked_averaged_models = StackingAveragedModels(base_models = (lr_model, lgbm_model, rf_model, gb_model),
                                                 meta_model = meta_model_final)

score = rmsle_cv(stacked_averaged_models)

test_data = pd.get_dummies(test_data)
test_data.fillna(0, inplace=True)

stacked_averaged_models.fit(x_values.values, y_values.values)
predictions = pd.DataFrame({'Price': stacked_averaged_models.predict(test_data.values)})
predictions.to_csv("E:/Kaggle_Problem/Flight Ticket Prediction/23032019_satcked.csv", index=False)
    



















































# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:04:28 2019

@author: User
"""

import numpy as np
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import os
import matplotlib
import graphviz

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def report_best_params(xgb_model):
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1 
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.6, 0.4)
    }
    
    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)
    
    search.fit(X, y)
    
    report_best_scores(search.cv_results_, 1)
    
'''------------------------------------------------------------------'''

wine = load_wine()

X = wine.data
y = wine.target

kfold = KFold(n_splits=4, shuffle=True, random_state=42)

scores = []

'''-----------------------------------------------------------------'''

xgb_model = xgb.XGBClassifier(objective="multi:softprob")
report_best_params(xgb_model)

'''-----------------------------------------------------------------'''

for train_index, test_index in kfold.split(X):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42, eval_metric=["merror"])
    xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

    y_pred = xgb_model.predict(X_test)
    
    accuracy_score(y_test, y_pred)

print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))

'''for train_index, test_index in kfold.split(X):   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", colsample_bytree= 0.7409, gamma=0.007272 , 
    random_state=42, max_depth=3, n_estimators = 117, subsample=0.75689)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    scores.append(mean_squared_error(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("\n")
display_scores(np.sqrt(scores))'''

xgb.plot_importance(xgb_model)
xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)#fig = matplotlib.pyplot.gcf()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
# converts the target tree to a graphviz instance
xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
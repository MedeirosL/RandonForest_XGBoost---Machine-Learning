# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:04:28 2019

@author: User
"""
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from numpy.linalg import inv
import scikitplot as skplt

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
y_testm = []
correctS=0
correctD=0
wrongS=0
wrongD=0
predictions=0
simplepc=0
doublepc=0
j=0
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

X = pd.read_csv('dataset3_HDW.csv', index_col=False)

plt.figure(figsize=(12,5))
sns.countplot(x=X.FTR, color='mediumseagreen')
plt.title('Emotional sentiment class distribution', fontsize=16)
plt.ylabel('Class Counts', fontsize=16)
plt.xlabel('Class Label', fontsize=16)
plt.xticks(rotation='vertical');


y = X['FTR']
X.drop('FTR', axis = 1, inplace=True)

'''wine = load_wine()

X = wine.data
y = wine.target'''

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = []

'''-----------------------------------------------------------------'''

'''xgb_model = xgb.XGBClassifier(objective="multi:softprob")
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

report_best_scores(search.cv_results_, 1)'''

'''-----------------------------------------------------------------'''

for train_index, test_index in kfold.split(X):   
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    xgb_model = xgb.XGBClassifier(objective="multi:softprob",  colsample_bytree= 0.7779849, gamma=0.2266204, learning_rate = 0.0396947,random_state=42, max_depth=2, n_estimators = 132, subsample=0.7644826)
    xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

    y_pred = xgb_model.predict(X_test)
    
    accuracy_score(y_test, y_pred)

print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))

for train_index, test_index in kfold.split(X):   
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
    
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", colsample_bytree= 0.7779849, gamma=0.2266204, learning_rate = 0.0396947,
    random_state=42, max_depth=2, n_estimators = 132, subsample=0.7644826)
    #xgb_model = xgb.XGBClassifier(objective="multi:softprob",random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_testm=y_test.as_matrix()
    scores.append(mean_squared_error(y_testm, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    display_scores(np.sqrt(scores))
    
    #print("Test: ",y_testm,"\n")
    #print("Pred: ",y_pred,"\n")

    
    for i in range (0,len(y_testm)):
        if y_testm[i] == y_pred[i]:
            correctS+=1
            correctD+=1
        elif y_testm[i] != y_pred[i] and (y_pred[i]==1 or y_testm[i]==1):
            wrongS+=1
            correctD+=1
        else:
            wrongS+=1
            wrongD+=1
        predictions+=1
    j+=1

#print("Acertos em ipótese simples:{}/{} - {0:.2f} %".format(correctS,predictions,correctS/predictions*100))
simplepc=round(correctS/predictions*100,2)
print("Acertos em ipótese simples:{}/{} - {}%".format(correctS,predictions,simplepc,simplepc))
doublepc=round(correctD/predictions*100,2)
print("Acertos em ipótese simples:{}/{} - {}%".format(correctD,predictions,doublepc,doublepc))
'''xgb.plot_importance(xgb_model)
xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)#fig = matplotlib.pyplot.gcf()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
# converts the target tree to a graphviz instance
xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)'''
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12,12))
plot_importance(xgb_model, ax = ax)
plt.show()


    
''' cm = confusion_matrix(y_testm, y_pred) 

 # Transform to df for easier plotting
 cm_df = pd.DataFrame(cm,
                      index = ['Home Win','Draw','Away Win'], 
                      columns = ['Home Win','Draw','Away Win'])
 
 plt.figure(figsize=(5,5))
 sns.heatmap(cm_df, annot=True)
 plt.title('XGBoost - Cross Validation {}'.format(j))
 plt.ylabel('True label')
 plt.xlabel('Predicted label')
 plt.show()'''
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

# FILE_PATH = '/root/sampleTest/train/'
# file_name = FILE_PATH + '000001.SZ.csv'
# df = pd.read_csv(file_name, parse_dates=True)

# for file in os.listdir('/root/sampleTest/test/'):
#     #读文件
#     filepath = '/root/sampleTest/test/' + file
#     df = pd.read_csv(filepath, index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')]
#     X, y = df.iloc[:, 3:-8].values, df.iloc[:, -1].values
#     print(file,X,y)

df = pd.read_csv('/root/sampleTest/train/000002.SZ.csv', index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')]
X, y = df.iloc[:, 3:-8].values, df.iloc[:, -1].values
print(file,y)

a = []
for i in range(0,len(y)): 
    if y[i] <= 0:
        a.append(0)
    elif int(round(y[i]*100)) >= 30:
        a.append(30)
        continue
    elif 30 > int(round(y[i]*100)) >= 20:
        a.append(25)
    elif 20 > int(round(y[i]*100)) > 10:
        a.append(15)
    else:
        a.append(5)

print(a)
from collections import Counter
Counter(a)
# Random Forest
forest_params = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], "max_depth": list(range(2,4,1)), 'criterion' :['gini', 'entropy']}
grid_forest = GridSearchCV(RandomForestClassifier(), forest_params, cv=2)
grid_forest.fit(X, a)

# Random Forest best estimator
random_forest = grid_forest.best_estimator_
grid_forest.best_estimator_# Logistic Regression 

forest_score = cross_val_score(random_forest, X, a, cv=2)
print('Random Forest Classifier Cross Validation Score', round(forest_score.mean() * 100, 2).astype(str) + '%')


# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 1, 5, 10], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'decision_function_shape':['ovo','ovr']}
grid_svc = GridSearchCV(SVC(), svc_params, cv=2)
grid_svc.fit(X, a)

# SVC best estimator
svc = grid_svc.best_estimator_
grid_svc.best_estimator_

svc_score = cross_val_score(svc, X, a, cv=2)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

# clf = SVC(kernel='linear', C=10, gamma=0.5, decision_function_shape='ovo')
# clf.fit(X, a)
# Z = clf.predict(XY).reshape(XX.shape)




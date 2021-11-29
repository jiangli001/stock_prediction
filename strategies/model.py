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
from sklearn.impute import SimpleImputer
# FILE_PATH = '/root/sampleTest/train/'
# file_name = FILE_PATH + '000001.SZ.csv'
# df = pd.read_csv(file_name, parse_dates=True)
d3= []
y = []
for file in os.listdir('/root/sampleTest/test/'):
    #读文件
    filepath = '/root/sampleTest/test/' + file
    d1 = pd.read_csv(filepath, index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')].iloc[:,3:]
    d1 = d1.fillna(method='ffill')
    d2_x = []
    for i in range(0,len(d1.index)-5):
        d2_x.append(pd.concat([d1.iloc[i,0:36],d1.iloc[i+1,0:36],d1.iloc[i+2,0:36],d1.iloc[i+3,0:36],d1.iloc[i+4,0:36]]))
        y.append(d1.iloc[i+1,-1])
    d3.append(pd.concat(d2_x,axis = 1).T)
x = pd.concat(d3,axis = 0).values
x
# x
# df
# X, Y = df.iloc[:, 3:-8].values, df.iloc[:, -1].values
# type(X)
# type(x)
# type(Y)
# type(y)
# len(y)
# Y
# y
# d2 = []
# df = pd.read_csv('/root/sampleTest/train/000728.SZ.csv', index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')].iloc[:,2:]
# type(df)
# df.dropna(subset=['Ticker','Index','Day'])
# imp = SimpleImputer(missing_values = np.nan, strategy='mean')
# imp.fit(df)
# df = imp.transform(df)
# print(df.isnull())
# len(df)
# for i in range(0,len(df.index)-5):
#     d2.append(df.iloc[i+1,-1]])
# d2
# d = pd.concat(d2,axis = 1).T
# d
# df.iloc[:,3:-1]
# # X, y = df.iloc[:, 3:-8].values, df.iloc[:, -1].values
# pd.concat([df.iloc[1],df.iloc[2]])

# for i in range(0,len(y)): 
#     if y[i] <= 0:
#         a.append(0)
#     elif int(round(y[i]*100)) >= 30:
#         a.append(30)
#         continue
#     elif 30 > int(round(y[i]*100)) >= 20:
#         a.append(25)
#     elif 20 > int(round(y[i]*100)) > 10:
#         a.append(15)

#     else:
#         a.append(5)
a = []
for i in range(0,len(y)): 
    if y[i] <= 0:
        a.append(0)
    elif int(round(y[i]*100)) >= 22:
        a.append(23)
    elif 22 > int(round(y[i]*100)) >= 20:
        a.append(21)
    elif 18 > int(round(y[i]*100)) >= 16:
        a.append(17)
    elif 16 > int(round(y[i]*100)) > 14:
        a.append(15)
    elif 14 > int(round(y[i]*100)) >= 12:
        a.append(13)
    elif 12 > int(round(y[i]*100)) >= 10:
        a.append(11)
    elif 10 > int(round(y[i]*100)) > 8:
        a.append(9)
    elif 8 > int(round(y[i]*100)) >= 6:
        a.append(7)
    elif 6 > int(round(y[i]*100)) >= 4:
        a.append(5)
    elif 4 > int(round(y[i]*100)) > 2:
        a.append(3)
    else:
        a.append(1)


print(a)
from collections import Counter
Counter(a)
# Random Forest
forest_params = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], "max_depth": list(range(2,4,1)), 'criterion' :['gini', 'entropy']}
grid_forest = GridSearchCV(RandomForestClassifier(), forest_params, cv=5)
grid_forest.fit(x, a)

# Random Forest best estimator
random_forest = grid_forest.best_estimator_
grid_forest.best_estimator_# Logistic Regression 
forest_score = cross_val_score(random_forest, x, a, cv=5)
print('Random Forest Classifier Cross Validation Score', round(forest_score.mean() * 100, 2).astype(str) + '%')


# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 1, 5, 10], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'decision_function_shape':['ovo','ovr']}
grid_svc = GridSearchCV(SVC(), svc_params, cv=5)
grid_svc.fit(x, a)

# SVC best estimator
svc = grid_svc.best_estimator_
grid_svc.best_estimator_
svc_score = cross_val_score(svc, x, a, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

# clf = SVC(kernel='linear', C=10, gamma=0.5, decision_function_shape='ovo')
# clf.fit(X, a)
# Z = clf.predict(XY).reshape(XX.shape)


a = 0
for file in os.listdir('/root/sampleTest/test/'):
    #读文件
    filepath = '/root/sampleTest/test/' + file
    d1 = pd.read_csv(filepath, index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')]
    print(len(d1.columns))
    a += 1
print('a',a)
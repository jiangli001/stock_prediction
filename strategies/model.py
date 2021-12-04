import numpy as np
import pandas as pd 
import datetime
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

d3= []
a = []
for file in os.listdir('/root/sampleTest/train/'):
    #读文件
    filepath = '/root/sampleTest/train/' + file
    d1 = pd.read_csv(filepath, index_col= 0)['2016/12/31':datetime.datetime.now().strftime('%y/%m/%d')].iloc[:,3:]
    d1 = d1.fillna(method='ffill')
    d2_x = []
    for i in range(0,len(d1.index)-5):
        a.append(d1.iloc[i+5,-1])
        d1_s = StandardScaler().fit_transform(d1)
        d1_t= pd.DataFrame(d1_s)
        d2_x.append(pd.concat([d1_t.iloc[i,0:33],d1_t.iloc[i+1,0:33],d1_t.iloc[i+2,0:33],d1_t.iloc[i+3,0:33],d1_t.iloc[i+4,0:33]]))
        # d2_x.append(np.hstack([d1_t[i,0:33],d1_t[i+1,0:33],d1_t[i+2,0:33],d1_t[i+3,0:33],d1_t[i+4,0:33]]))
    d3.append(pd.concat(d2_x,axis = 1).T)
x = pd.concat(d3,axis = 0).values
x

y = []
for i in range(0,len(a)): 
    if a[i] <= 0:
        y.append(0)
    elif int(round(a[i]*100)) >= 22:
        y.append(23)
    elif 22 > int(round(a[i]*100)) >= 20:
        y.append(21)
    elif 18 > int(round(a[i]*100)) >= 16:
        y.append(17)
    elif 16 > int(round(a[i]*100)) > 14:
        y.append(15)
    elif 14 > int(round(a[i]*100)) >= 12:
        y.append(13)
    elif 12 > int(round(a[i]*100)) >= 10:
        y.append(11)
    elif 10 > int(round(a[i]*100)) > 8:
        y.append(9)
    elif 8 > int(round(a[i]*100)) >= 6:
        y.append(7)
    elif 6 > int(round(a[i]*100)) >= 4:
        y.append(5)
    elif 4 > int(round(a[i]*100)) > 2:
        y.append(3)
    else:
        y.append(1)

Counter(y)

# Random Forest
forest_params = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], "max_depth": list(range(2,4,1)), 'criterion' :['gini', 'entropy']}
grid_forest = GridSearchCV(RandomForestClassifier(), forest_params, cv=5)
grid_forest.fit(x, y)
# Random Forest best estimator
random_forest = grid_forest.best_estimator_
grid_forest.best_estimator_# Logistic Regression 
forest_score = cross_val_score(random_forest, x, y, cv=5)
print('Random Forest Classifier Cross Validation Score', round(forest_score.mean() * 100, 2).astype(str) + '%')
rf=random_forest.fit(x, y)
joblib.dump(rf,'random_forest.model')
# RandomForestClassifier(max_depth=3, max_features='sqrt', n_estimators=500) Random Forest Classifier Cross Validation Score 54.42%

# # Support Vector Classifier
# svc_params = {'C': [0.5, 0.7, 1, 5, 10], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'decision_function_shape':['ovo','ovr']}
# grid_svc = GridSearchCV(SVC(), svc_params, cv=5)
# grid_svc.fit(x, y)
# # SVC best estimator
# svc = grid_svc.best_estimator_
# grid_svc.best_estimator_
# svc_score = cross_val_score(svc, x, y, cv=5)
# print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(x, y)
# DecisionTree best estimator
tree_clf = grid_tree.best_estimator_
grid_tree.best_estimator_
tree_score = cross_val_score(tree_clf, x, y, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
#DecisionTreeClassifier(max_depth=2, min_samples_leaf=5) DecisionTree Classifier Cross Validation Score 52.85%

# KNearest
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(x, y)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_
grid_knears.best_estimator_
knears_score = cross_val_score(knears_neighbors, x, y, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')
# KNeighborsClassifier(n_neighbors=4)


def random_forest(predict_day, dt):
    from sklearn.model_selection import cross_val_predict
    random_forest = RandomForestClassifier(max_depth=3, max_features='sqrt', n_estimators=500)
    d1 = dt.fillna(method='ffill')
    d2_x = []
    for i in range(0,len(d1.index)-5):
        a.append(d1.iloc[i+5,-1])
        d1_t = StandardScaler().fit(d1)
        d2_x.append(pd.concat([d1_t.iloc[i,0:33],d1_t.iloc[i+1,0:33],d1_t.iloc[i+2,0:33],d1_t.iloc[i+3,0:33],d1_t.iloc[i+4,0:33]]))
        x = pd.concat(d2,axis = 0).values
    forest_pred = cross_val_predict(random_forest, X_train_std, ysm_train, cv=5)    
########################

#创建文件夹test_Random
shutil.rmtree(os.path.join('/root','sampleTest','test_Random_Forest'), ignore_errors=True)
os.makedirs(os.path.join('/root','sampleTest','test_Random_Forest'))

#test数据生成随机的一列(用5天平均)，仅用来测试，正常时候不用
for file in os.listdir('/root/sampleTest/test/'):
        #读文件
        filepath = '/root/sampleTest/test/' + file
        dt = pd.read_csv(filepath, index_col= 0)
        dt['GrowthRatePredict'] = random_forest(5, dt)
        dt.to_csv(os.path.join('/root','sampleTest','test_Random_Forest', file))
        
        
        
        
        
#  1 from sklearn import svm
#  2 from sklearn.externals import joblib
#  3 
#  4 #训练模型
#  5 clf = svc = svm.SVC(kernel='linear')
#  6 rf=clf.fit(array(trainMat), array(listClasses))
#  7 
#  8 ＃保存模型
#  9 joblib.dump(rf,'rf.model')
# 10 
# 11 ＃加载模型
# 12 RF=joblib.load('rf.model')
# 13 
# 14 ＃应用模型进行预测
# 15 result=RF.predict(thsDoc)
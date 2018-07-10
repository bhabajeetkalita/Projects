'''
Author: Bhabajeet Kalita
Date: 28 - 5 - 2018
Description: Demonstration of the use of RandomizedSearchCV
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)

feature_cols=['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales
lm = LinearRegression()
scores=cross_val_score(lm,x,y,cv=10,scoring='mean_squared_error')
print scores

mse_scores = -scores
print mse_scores

rmse_scores=np.sqrt(mse_scores)
print rmse_scores
print rmse_scores.mean()

feature_cols = ['TV','Radio']
x=data[feature_cols]
print np.sqrt(-cross_val_score(lm,x,y,cv=10,scoring='mean_squared_error')).mean()

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline
iris = load_iris()
x=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print scores
print scores.mean()
k_range = range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

from sklearn.grid_search import GridSearchCV
k_range = range(1,31)
print k_range
param_grid = dict(n_neighbors=k_range)
print param_grid

grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(x,y)
grid.grid_scores_
print grid.grid_scores_[0].parameters
print grid.grid_scores_[0].cv_validation_scores
print grid.grid_scores_[0].mean_validation_score
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print grid_mean_scores

plt.plot(k_range,grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

print grid.best_score_
print grid.best_params_
print grid.best_estimator_

k_range = range(1,31)
weight_options = ['uniform','distance']

param_grid = dict(n_neighbors=k_range,weights=weight_options)
print param_grid

grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(x,y)
grid.grid_scores_
print grid.best_score_
print grid.best_params_

print grid.best_score_
print grid.best_params_

knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')
knn.fit(x,y)

knn.predict([3,5,4,2])

grid.predict([3,5,4,2])

from sklearn.grid_search import RandomizedSearchCV
param_dist = dict(n_neighbors = k_range,weights=weight_options)

rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10,random_state=5)
rand.fit(x,y)
rand.grid_scores_

print rand.best_score_
print rand.best_params_

best_scores=[]
for _ in range(20):
    rand = RandomizedSearchCV(knn,param_dist,cv=10,scoring='accuracy',n_iter=10)
    rand.fit(x,y)
    best_scores.append(round(rand.best_score_,3))
print best_scores

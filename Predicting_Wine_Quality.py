'''
Author: Bhabajeet Kalita
Description: Task 1 to predict the wine quality for a wine id
Date: 15/11/17
'''
#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

#Importing csv files to dataframe using pandas and setting indexes
df_1 = pd.read_csv('train_stud.csv')
df_1.drop(df_1.columns[[0]], axis=1, inplace=True)
df_1.set_index('wine_id')
print(df_1.head())

df_2 = pd.read_csv('test_stud.csv')
df_2.drop(df_2.columns[[0]], axis=1, inplace=True)
df_2.set_index('wine_id')
print(df_2.head())

df_3 = pd.read_csv('sample_submission.csv')

#Exploratory Data Analysis
print(df_1.info())
print(df_1.describe())
print(df_1.columns)
sns.heatmap(df_1.corr())
#Training the model

x_train = df_1[['wine_id', 'fixed acidity', 'volatile acidity', 'citric acid','residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol','style']]
y_train = df_1['quality']
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(x_train,y_train)

#Mean absolute error calculation using cross validation
scores = cross_val_score(rfc,x_train,y_train,cv=10,scoring='mean_absolute_error')
print(scores)

#Predicting for the unknown dataset using the model
predict = rfc.predict(df_2)
print(predict)

#Saving the dataframe to a csv file
df_3['predicted_quality'] = predict
df_3.to_csv("/Users/Gourhari/Documents/Studies/sample_submission.csv", index = False, encoding = "utf-8")

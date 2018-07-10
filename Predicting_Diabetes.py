'''
Author: Bhabajeet Kalita
Date: 12 - 06 - 2018
Description: Demonstration of Predicting diabetes
'''

import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv(url,header=None,names=col_names)
pima.head()

feature_cols=['pregnant','insulin','bmi','age']
x=pima[feature_cols]
y=pima.label
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred_class=logreg.predict(x_test)

from sklearn import metrics
print metrics.accuracy_score(y_test,y_pred_class)

y_test.value_counts()
y_test.mean()

1-y_test.mean()
max(y_test.mean(),1-y_test.mean())
y_test.value_counts().head(1)/len(y_test)

print 'True:',y_test.values[0:25]
print 'Pred:',y_pred_class[0:25]

print metrics.confusion_matrix(y_test,y_pred_class)
![Small confusion matrix](images/09_confusion_matrix_1.png)

print 'True:',y_test.values[0:25]
print 'Pred:',y_pred_class[0:25]

confusion=metrics.confusion_matrix(y_test,y_pred_class)
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[1,1]
FN=confusion[1,0]

![Large confusion matrix])images/09/confusion_matrix_2.png)

print(TP+TN)/float(TP+TN+FP+FN)
print metrics.accuracy_score(y_test,y_pred_class)

print (FP+TN)/float(TP+TN+FP+FN)
print 1-metrics.accuracy_score(y_test,y_pred_class)

print TP/float(TP+FN)
print metrics.recall_score(y_test,y_pred_class)

print TN/float(TN+FP)
print FP/float(TN+FP)

print TP/float(TP+FP)
print metrics.precision_score(y_test,y_pred_class)

logreg.predict(x_test)[0:10]
logreg.predict_proba(x_test)[0:10,:]

logreg.predict_proba(x_test)[0:10,1]
y_pred_prob = logreg.predict_proba(x_test)[:,1]

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14


plt.hist(y_pred_prob,bins=8)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')

from sklearn.preprocessing import binarize
y_pred_class=binarize(y_pred_prob,0.3)[0]
y_pred_prob[0:10]
y_pred_class[0:10]

print confusion
print metrics.confusion_matrix(y_test,y_pred_class)

print 46/float(46+16)
print 80/float(80+50)

fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.title('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

def evaluate_threshold(threshold):
    print 'Sensitivity',tpr[thresholds>threshold][-1]
    print 'Specificity',1-fpr[thresholds>threshold][-1]

evaluate_threshold(0.5)
evaluate_threshold(0.3)

print metrics.roc_auc_score(y_test,y_pred_prob)

from sklearn.cross_validation import cross_val_score
cross_val_score(logreg,x,y,cv=10,scoring='roc_auc').mean()

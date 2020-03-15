# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:36:38 2020

@author: TokaAshraf
"""


# importing required libraries 
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:36:38 2020

@author: TokaAshraf
"""


#import necessary liberaries
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load the data set from datasets of sklearn called "load_breast_cancer"
cancer = datasets.load_breast_cancer()
# divid data into train and test data "30% test and 70% train"
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

# build the classifier "Decision Tree" model
clf = svm.SVC(kernel='linear')
#fit the model with train data
clf.fit(X_train, y_train)

# test the model using X test 
y_pred = clf.predict(X_test)

# show the result of testing comparing the out put of the model "Y_pred" with the actual output "Y_test"
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

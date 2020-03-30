# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:23:10 2020

@author: TokaAshraf
"""
#import necessary liberaries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# names of all columns of the data set
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load the data set from a file in my PC 
pima = pd.read_csv(r"D:\3_term2_CE\MobileComputing\Tasks\data\pima-indians-diabetes.csv", header = None, names = col_names)

#names of features columns
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

# divid the data into 2 parts X for features and Y for labelling
X = pima[feature_cols] # Features
y = pima.label # Target variable "labels"

# divid data into train and test data "30% test and 70% train"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# build the classifier "Decision Tree" model
classifier = DecisionTreeClassifier()
#fit the model with train data
classifier = classifier.fit(X_train,y_train)

# test the model using X test 
y_pred = classifier.predict(X_test)

# show the result of testing comparing the out put of the model "Y_pred" with the actual output "Y_test"
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

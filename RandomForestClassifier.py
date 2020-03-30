# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:09:11 2020

@author: TokaAshraf
"""
#import necessary liberaries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# the path of the dataset 
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#column names of dataset
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#read the data from dataset and name columns with header names
dataset = pd.read_csv(path, names = headernames)

# divid the data into 2 parts X for features and Y for labelling
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values

# divid data into train and test data "30% test and 70% train"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# build the classifier "Random Forest" model
classifier = RandomForestClassifier(n_estimators = 50)
#fit the model with train data
classifier.fit(X_train, y_train)

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

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:15:05 2020

@author: TokaAshraf
"""

import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r"D:\3_term2_CE\MobileComputing\Tasks\data\pima-indians-diabetes.csv", header = None, names = col_names)
pima.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_test = mlp.predict(X_test)

result = confusion_matrix(y_test,predict_test)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test,predict_test)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,predict_test)
print("Accuracy:",result2)
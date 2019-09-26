# Write a program to implement k-Nearest Neighbour algorithm to classify the iris
# data set. Print both correct and wrong predictions. Java/Python ML library
# classes can be used for this problem. 

import numpy as np
import pandas as pd
import matplotlib as plt

### Step 2 : Load the inbuilt data or the csv/excel file into pandas dataframe and clean the data # ln[66]

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Class'] = data.target_names[data.target]
df.head()
x = df.iloc[:,:-1].values
y = df.Class.values

print(x[:5])
print(y[:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(x_train, y_train)
prediction = knn_classifier.predict(x_test)
print(prediction)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Training accuracy Score is : \n", accuracy_score(y_train, knn_classifier.predict(x_train)))
print("Testing accuracy Score is : \n", accuracy_score(y_test, knn_classifier.predict(x_test)))

print("Training Confusion Matrix is : \n", confusion_matrix(y_train, knn_classifier.predict(x_train)))
print("Testing Confusion Matrix is : \n", confusion_matrix(y_test, knn_classifier.predict(x_test)))


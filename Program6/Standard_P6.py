import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

c1,c2,c3,c4 = np.loadtxt('data.csv',unpack=True,delimiter = ',')
x= np.column_stack((c1,c3))
y= c4
#Create NaiveBayes Classifier
clf = GaussianNB()
#fit the mode
clf.fit(x,y)
#make predictions
predictions = clf.predict(x)

print('accuracy metrics')

#calculate accuracy
print('accuracy of classifer is',metrics.accuracy_score(y,predictions))
print('confusion matrix')
print(confusion_matrix(y,predictions))
print('recall and precision')
print(metrics.recall_score(y,predictions))
print(metrics.precision_score(y,predictions))
from matplotlib import pyplot as plt

plt.scatter(c1, c3, c=c4)
plt.colorbar(ticks=[1, 2])
plt.xlabel("Age of the patient")
plt.ylabel("No of positive axillary nodes")
plt.show()


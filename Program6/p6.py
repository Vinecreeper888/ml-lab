import pandas as pd
msg=pd.read_csv('pg6.csv',names=['message','label'])
print('the dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum
print(X)
print(Y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)
print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape)
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
xtrain_dtm=count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print(count_vect.get_feature_names())
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df)
print(xtrain_dtm)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)
from sklearn import metrics
print('accuracy metrics')
print('accuracy of classifer is',metrics.accuracy_score(ytest,predicted))
print('confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('recall and precision')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))

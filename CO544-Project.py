#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import matplotlib.pyplot as plt

#read the dataset
dataset = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/data.csv',sep= ',')
print(dataset.head())


# In[58]:


print ("Dataset Length: ", len(dataset)) 
print ("Dataset Shape: ", dataset.shape)


# In[59]:


print(dataset['A16'].unique())


# In[60]:


print(dataset.groupby('A16').size())


# In[61]:


import seaborn as sns
sns.countplot(dataset['A16'],label="Count")
plt.show()


# In[62]:


print(dataset.__eq__('?').sum())


# In[63]:


print(dataset.dtypes)


# In[64]:


#Since A2,A5,A7,A10,A12,A14 has to be numeric
#replace all the missing data with 0

dataset[['A2','A5','A7','A10','A12','A14']] = dataset[['A2','A5','A7','A10','A12','A14']].replace('?',0)


# In[65]:


#change A2,A5,A7,A10,A12,A14 data type to float
dataset['A14'] = dataset.A14.astype(float)
dataset['A2'] = dataset.A2.astype(float)
dataset['A5'] = dataset.A5.astype(float)
dataset['A7'] = dataset.A7.astype(float)
dataset['A10'] = dataset.A10.astype(float)
dataset['A12'] = dataset.A12.astype(float)


# In[66]:


print(dataset.dtypes)


# In[67]:


# summary statistics of character column
print (dataset.describe(include='all'))


# In[68]:


import numpy as np
from sklearn.preprocessing import LabelEncoder

#Since python machine learning algorithm do not accept string values
le = LabelEncoder()

dataset['A1'] = le.fit_transform(dataset['A1'])
dataset['A3'] = le.fit_transform(dataset['A3'])
dataset['A4'] = le.fit_transform(dataset['A4'])
dataset['A6'] = le.fit_transform(dataset['A6'])
dataset['A8'] = le.fit_transform(dataset['A8'])
dataset['A9'] = le.fit_transform(dataset['A9'])
dataset['A11'] = le.fit_transform(dataset['A11'])
dataset['A13'] = le.fit_transform(dataset['A13'])
dataset['A15'] = le.fit_transform(dataset['A15'])


# In[69]:


print(dataset)


# In[70]:


#In above it has encoded the unique values of each column with a unique number


# In[71]:


print(dataset.describe())


# In[72]:


from sklearn.model_selection import train_test_split

X = dataset.values[:,0:15]
Y = dataset.values[:,15]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[73]:


#Let's train the machine learning algorithms with the dataset
#Then find the model with highest accuracy


# In[74]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, Y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, Y_test)))


# In[75]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, Y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, Y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, Y_test)))


# In[76]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, Y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, Y_test)))


# In[77]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, Y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, Y_test)))


# In[78]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, Y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, Y_test)))


# In[79]:


#Support Vector Machine algorithm
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, Y_test)))


# In[80]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = logreg.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[81]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[82]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[83]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = lda.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[84]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = gnb.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[85]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = svm.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[86]:


model = ['logreg','clf','knn','lda','gnb','svm']
scores = []

scores.append(logreg.score(X_test, Y_test)) 
scores.append(clf.score(X_test, Y_test)) 
scores.append(knn.score(X_test, Y_test)) 
scores.append(lda.score(X_test, Y_test)) 
scores.append(gnb.score(X_test, Y_test)) 
scores.append(svm.score(X_test, Y_test))

plt.figure() 
plt.xlabel('Classification Model') 
plt.ylabel('Accuracy') 
plt.scatter(model, scores)


# In[87]:


testset = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testdata.csv')
print(testset.head())


# In[88]:


print ("Dataset Length: ", len(testset)) 
print ("Dataset Shape: ", testset.shape)


# In[89]:


tf = pd.DataFrame(testset)


# In[90]:


testset[['A2','A5','A7','A10','A12','A14']] = testset[['A2','A5','A7','A10','A12','A14']].replace('?',0)
# to change use .astype() 
testset['A2'] = testset.A2.astype(float)
testset['A5'] = testset.A5.astype(float)
testset['A7'] = testset.A7.astype(float)
testset['A10'] = testset.A10.astype(float)
testset['A12'] = testset.A12.astype(float)
testset['A14'] = testset.A14.astype(float)


# In[91]:


tf['A4'] = le.fit_transform(tf['A4'])
tf['A1'] = le.fit_transform(tf['A1'])
tf['A3'] = le.fit_transform(tf['A3'])
tf['A6'] = le.fit_transform(tf['A6'])
tf['A8'] = le.fit_transform(tf['A8'])
tf['A9'] = le.fit_transform(tf['A9'])
tf['A11'] = le.fit_transform(tf['A11'])
tf['A13'] = le.fit_transform(tf['A13'])
tf['A15'] = le.fit_transform(tf['A15'])


# In[92]:


test_scaler = scaler.transform(tf.values[:,0:15])


# In[93]:


test_pred = svm.predict(test_scaler) 
print("Predicted values:") 
print(test_pred)


# In[94]:


tf['A16'] = test_pred
#Final prediction on the test data set
print(tf)


# In[95]:


#This is to show the final test without any encodings
test_final = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testdata.csv')
test_final['A16'] = test_pred


# In[96]:


test_final_frame = pd.DataFrame(test_final)


# In[97]:


#Final prediction on the test set
print(test_final_frame)


# In[98]:


test_final_frame.to_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testresultsnew.csv',sep=',')


# In[ ]:





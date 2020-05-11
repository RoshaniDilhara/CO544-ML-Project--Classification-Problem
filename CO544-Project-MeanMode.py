#!/usr/bin/env python
# coding: utf-8

# In[1]:
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#load the dataset
dataset = pd.read_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/data.csv')
#print the rows of data
dataset.head()


# In[3]:


# checking for null values
print("Sum of NULL values in each column. ")
print(dataset.__eq__('?').sum())


# In[4]:


# No of values and features in the dataset
print(dataset.shape)


# In[5]:


print(dataset.info())


# In[6]:


# mark missing values as NaN
dataset[["A1","A2","A3","A4","A6","A9","A14"]] = dataset[["A1","A2","A3","A4","A6","A9","A14"]].replace("?", np.NaN)


# In[7]:


# count the number of NaN values in each column
print(dataset.isnull().sum())


# In[8]:


# print the first 80 rows of data
print(dataset.head(332))


# In[9]:


#change object type to float of some columns which the data type has been identified as object,but they are of type float
dataset["A2"]= dataset["A2"].astype(float)
dataset["A14"]= dataset["A14"].astype(float)


# In[10]:


#change data type to category

dataset["A1"]= dataset["A1"].astype('category')
dataset["A3"]= dataset["A3"].astype('category')
dataset["A4"]= dataset["A4"].astype('category')
dataset["A6"]= dataset["A6"].astype('category')
dataset["A8"]= dataset["A8"].astype('category')
dataset["A9"]= dataset["A9"].astype('category')
dataset["A11"]= dataset["A11"].astype('category')
dataset["A13"]= dataset["A13"].astype('category')
dataset["A15"]= dataset["A15"].astype('category')
dataset["A16"]= dataset["A16"].astype('category')

dataset.dtypes


# In[11]:


# count the number of NaN values in each column
print(dataset.isnull().sum())


# In[15]:


from statistics import mode
from statistics import mean


# In[17]:


dataset["A2"].fillna(dataset["A2"].mean(),inplace = True)
dataset["A14"].fillna(dataset["A14"].mean(),inplace = True)


# In[18]:


dataset.head(80)


# In[19]:


print(statistics.mode(dataset['A1']))
print(statistics.mode(dataset['A3']))
print(statistics.mode(dataset['A4']))
print(statistics.mode(dataset['A6']))
print(statistics.mode(dataset['A9']))


# In[20]:


#fill missing values of nominal attributes by mode
dataset['A1'].fillna('b',inplace=True)
dataset['A3'].fillna('u',inplace=True)
dataset['A4'].fillna('g',inplace=True)
dataset['A6'].fillna('c',inplace=True)
dataset['A9'].fillna('v',inplace=True)


# In[21]:


dataset.head(332)


# In[22]:


#Label Encoding
dataset["A1"]= dataset["A1"].cat.codes
dataset["A3"]= dataset["A3"].cat.codes
dataset["A4"]= dataset["A4"].cat.codes
dataset["A6"]= dataset["A6"].cat.codes
dataset["A8"]= dataset["A8"].cat.codes
dataset["A9"]= dataset["A9"].cat.codes
dataset["A11"]= dataset["A11"].cat.codes
dataset["A13"]= dataset["A13"].cat.codes
dataset["A15"]= dataset["A15"].cat.codes
dataset["A16"]= dataset["A16"].cat.codes

dataset.head(332)


# In[23]:


# count the number of NaN values in each column
print(dataset.isnull().sum())


# In[24]:


newer_dataset = dataset.copy()
newer_dataset.head(80)


# In[25]:


newfeature_names =["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15"]
X = newer_dataset[newfeature_names]
Y = newer_dataset['A16']


# In[26]:


#split the data set as training set and test set randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# In[27]:


#apply scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[28]:


#use model Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#use model Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
#use model Decision Tree Classifier with maximum depth of 3
clf1 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
#use model Decision Tree Classifier with maximum depth of 4
clf2 = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
#use model k-neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
#use model Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
#use model Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
#use model support vector machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

#use model Random forest
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_test)

from sklearn import model_selection
rfr = RandomForestClassifier(n_estimators=100, random_state=0)
rfr.fit(X_train, y_train)


# In[29]:


print('Accuracy of Logistic regression classifier on training set: {:.2f}'
.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
.format(logreg.score(X_test, y_test)))

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
.format(clf.score(X_test, y_test)))

print('Accuracy of Decision Tree classifier on training set with maximum depth of 3 : {:.2f}'
.format(clf1.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set with maximum depth of 3 : {:.2f}'
.format(clf1.score(X_test, y_test)))

print('Accuracy of Decision Tree classifier on training set with maximum depth of 4 : {:.2f}'
.format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set with maximum depth of 4 : {:.2f}'
.format(clf2.score(X_test, y_test)))

print('Accuracy of K-NN classifier on training set: {:.2f}'
.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
.format(knn.score(X_test, y_test)))

print('Accuracy of LDA classifier on training set: {:.2f}'
.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
.format(lda.score(X_test, y_test)))

print('Accuracy of GNB classifier on training set: {:.2f}'
.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
.format(gnb.score(X_test, y_test)))

print('Accuracy of SVM classifier on training set: {:.2f}'
.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
.format(svm.score(X_test, y_test)))

print('Accuracy of Random forest classifier on training set: {:.2f}'
      .format(rfr.score(X_train, y_train)))
print('Accuracy of Random forest classifier on test set: {:.2f}'
      .format(rfr.score(X_test, y_test)))

# In[30]:
Test_names = ['Logistic regression', 'Decision Tree', 'Decision Tree-max depth=3', 'Decision Tree-max depth=4',
              'k-neighbors', 'Linear Discriminant', 'Gaussian Naive Bayes', 'Support vector machine', 'Randomforest']
Test_name_repres = ['LR', 'DT', 'DTdep3', 'DTdeP4', 'k-NN', 'LD', 'GNB', 'SVM', 'RF']

training = []
test = []

training.append(logreg.score(X_train, y_train))
training.append(clf.score(X_train, y_train))
training.append(clf1.score(X_train, y_train))
training.append(clf2.score(X_train, y_train))
training.append(knn.score(X_train, y_train))
training.append(lda.score(X_train, y_train))
training.append(gnb.score(X_train, y_train))
training.append(svm.score(X_train, y_train))
training.append(rfr.score(X_train, y_train))

test.append(logreg.score(X_test, y_test))
test.append(clf.score(X_test, y_test))
test.append(clf1.score(X_test, y_test))
test.append(clf2.score(X_test, y_test))
test.append(knn.score(X_test, y_test))
test.append(lda.score(X_test, y_test))
test.append(gnb.score(X_test, y_test))
test.append(svm.score(X_test, y_test))
test.append(rfr.score(X_test, y_test))


# In[193]:


plt.scatter(Test_name_repres,training,label='training')
plt.scatter(Test_name_repres,test,label='test')
plt.xlabel('Name of the test')
plt.ylabel('Accuracy for each test')
plt.legend()
plt.show()




#now lets try for the test data set


# In[35]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#load the test dataset
test_dataset = pd.read_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/kaggle/testdata.csv')
test_dataset_copy = test_dataset.copy()
#print the rows of data
test_dataset_copy.head(80)


# In[36]:


# checking for null values
print("Sum of NULL values in each column. ")
test_dataset_copy.__eq__('?').sum()


# In[37]:


# mark missing values as NaN
test_dataset_copy[["A1","A2","A3","A4","A6","A9","A14"]] = test_dataset_copy[["A1","A2","A3","A4","A6","A9","A14"]].replace("?", np.NaN)


# In[38]:


# count the number of NaN values in each column
print(test_dataset_copy.isnull().sum())


# In[39]:


# print the first 80 rows of data
print(test_dataset_copy.head(14))


# In[40]:


test_dataset_copy.info(14)


# In[44]:


#change object type to float of some columns which the data type has been identified as object,but they are of type float
test_dataset_copy["A2"]= test_dataset_copy["A2"].astype(float)
test_dataset_copy["A14"]= test_dataset_copy["A14"].astype(float)


# In[47]:


#change data type to category

test_dataset_copy["A1"]= test_dataset_copy["A1"].astype('category')
test_dataset_copy["A3"]= test_dataset_copy["A3"].astype('category')
test_dataset_copy["A4"]= test_dataset_copy["A4"].astype('category')
test_dataset_copy["A6"]= test_dataset_copy["A6"].astype('category')
test_dataset_copy["A8"]= test_dataset_copy["A8"].astype('category')
test_dataset_copy["A9"]= test_dataset_copy["A9"].astype('category')
test_dataset_copy["A11"]= test_dataset_copy["A11"].astype('category')
test_dataset_copy["A13"]= test_dataset_copy["A13"].astype('category')
test_dataset_copy["A15"]= test_dataset_copy["A15"].astype('category')

test_dataset_copy.dtypes


# In[48]:


# count the number of NaN values in each column
print(test_dataset_copy.isnull().sum())


# In[51]:


from statistics import mode
from statistics import mean

test_dataset_copy["A2"].fillna(test_dataset_copy["A2"].mean(),inplace = True)
test_dataset_copy["A14"].fillna(test_dataset_copy["A14"].mean(),inplace = True)

test_dataset_copy.head(14)


# In[52]:


print(statistics.mode(test_dataset_copy['A1']))
print(statistics.mode(test_dataset_copy['A3']))
print(statistics.mode(test_dataset_copy['A4']))
print(statistics.mode(test_dataset_copy['A6']))
print(statistics.mode(test_dataset_copy['A9']))


# In[53]:


#fill missing values of nominal attributes by mode
test_dataset_copy['A1'].fillna('b',inplace=True)
test_dataset_copy['A3'].fillna('u',inplace=True)
test_dataset_copy['A4'].fillna('g',inplace=True)
test_dataset_copy['A6'].fillna('c',inplace=True)
test_dataset_copy['A9'].fillna('v',inplace=True)

test_dataset_copy.head(14)


# In[54]:


# checking for null values
print("Sum of NULL values in each column. ")
print(test_dataset_copy.__eq__('Nan').sum())


# In[55]:


#Label Encoding
test_dataset_copy["A1"]= test_dataset_copy["A1"].cat.codes
test_dataset_copy["A3"]= test_dataset_copy["A3"].cat.codes
test_dataset_copy["A4"]= test_dataset_copy["A4"].cat.codes
test_dataset_copy["A6"]= test_dataset_copy["A6"].cat.codes
test_dataset_copy["A8"]= test_dataset_copy["A8"].cat.codes
test_dataset_copy["A9"]= test_dataset_copy["A9"].cat.codes
test_dataset_copy["A11"]= test_dataset_copy["A11"].cat.codes
test_dataset_copy["A13"]= test_dataset_copy["A13"].cat.codes
test_dataset_copy["A15"]= test_dataset_copy["A15"].cat.codes

test_dataset_copy.head(14)


# In[56]:


X_predict = scaler.transform(test_dataset_copy[newfeature_names])


# In[57]:


#using support vector machine
# y_predict=svm.predict(X_predict)
# print(y_predict)

# use model Decision Tree Classifier
# y_predict = clf.predict(X_predict)
# print(y_predict)

# use model Decision Tree Classifier with maximum depth of 3
# y_predict = clf1.predict(X_predict)
# print(y_predict)

# use model Decision Tree Classifier with maximum depth of 4
# y_predict = clf2.predict(X_predict)
# print(y_predict)

# use model K-nearest neighbour
# y_predict = knn.predict(X_predict)
# print(y_predict)

# use model linear discriminant analysis
# y_predict = lda.predict(X_predict)
# print(y_predict)

# use model gaussian naive bayes
# y_predict = gnb.predict(X_predict)
# print(y_predict)


# using Logistic regression
# y_predict = logreg.predict(X_predict)
# print(y_predict)

# using random forest
y_predict = rfr.predict(X_predict)
print(y_predict)


# In[58]:


test_dataset.insert(15,"A16",y_predict,True)


# In[59]:


cleanup_predict = {"A16":     {1: "Success", 0 : "Failure"}}
# print(cleanup_predict)


# In[60]:


test_dataset.replace(cleanup_predict, inplace=True)


# In[61]:


test_dataset.head(20)

# In[ ]:
pd.options.display.max_rows
pd.set_option('display.max_rows', None)
# test_dataset.to_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/kaggle/results1-RF-mn1.csv' , index=False)






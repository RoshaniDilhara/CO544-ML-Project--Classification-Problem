#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load data file
data = pd.read_csv('D:\Semester 6 - 3rd year\Machine Learning -CO544\data.csv')


# In[3]:


data.head()


# In[4]:


#find number of missing data in each column
data.__eq__('?').sum()


# In[5]:


#replace missing data in each column with nan
data['A1'].replace('?',np.nan, inplace=True)
data['A2'].replace('?',np.nan, inplace=True)
data['A3'].replace('?',np.nan, inplace=True)
data['A4'].replace('?',np.nan, inplace=True)
data['A6'].replace('?',np.nan, inplace=True)
data['A9'].replace('?',np.nan, inplace=True)
data['A14'].replace('?',np.nan, inplace=True)


# In[6]:


data.describe()


# In[7]:


#drop rows with missing data
data.dropna()


# In[8]:


print(data.info())


# In[9]:


data = data.dropna()


# In[10]:


print(data.info())


# In[11]:


#change object type to float type of some columns
data["A2"]= data["A2"].astype(float) 
data["A12"]= data["A12"].astype(float)
data["A7"]= data["A7"].astype(float)
data["A14"]= data["A14"].astype(float)


# In[12]:


print(data.info())


# In[13]:


import seaborn as sns


# In[14]:


sns.countplot(data['A16'],label="count")
plt.show()


# In[15]:


data.boxplot('A2','A16',rot=30, figsize=(10,10))
data.boxplot('A7','A16',rot=30, figsize=(10,10))
data.boxplot('A12','A16',rot=30, figsize=(10,10))
data.boxplot('A14','A16',rot=30, figsize=(10,10))


# In[16]:


newer_data = data.copy()


# In[17]:


print(newer_data['A1'].value_counts())
print(newer_data['A3'].value_counts())
print(newer_data['A4'].value_counts())
print(newer_data['A6'].value_counts())
print(newer_data['A9'].value_counts())
print(newer_data['A15'].value_counts())


# In[18]:


#one-hot encoding to object type columns
onehote_data = newer_data.copy()
onehote_data = pd.get_dummies(onehote_data, columns=['A3'], prefix=['A3'])
onehote_data = pd.get_dummies(onehote_data, columns=['A4'], prefix=['A4'])
onehote_data = pd.get_dummies(onehote_data, columns=['A6'], prefix=['A6'])
onehote_data = pd.get_dummies(onehote_data, columns=['A9'], prefix=['A9'])
onehote_data = pd.get_dummies(onehote_data, columns=['A15'], prefix=['A15'])


# In[19]:


onehote_data.head()


# In[20]:


#label encoding to column A1
onehote_data["A1"]= onehote_data["A1"].astype('category')
onehote_data['A1'] = onehote_data['A1'].cat.codes
onehote_data.head()


# In[21]:


print(onehote_data.info())


# In[22]:


feature_names = ['A1', 'A2', 'A5', 'A7','A8','A10','A11','A12','A13','A14','A3_l','A3_u','A3_y','A4_g','A4_gg','A4_p','A6_aa','A6_c','A6_cc','A6_d','A6_e','A6_ff','A6_i','A6_j','A6_k','A6_m','A6_q','A6_r','A6_w','A6_x','A9_bb','A9_dd','A9_ff','A9_h','A9_j','A9_n','A9_o','A9_v','A9_z','A15_g','A15_p','A15_s']
X = onehote_data[feature_names]
Y = onehote_data['A16']


# In[23]:


#split the data set as training set and test set randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# In[24]:


#apply scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[25]:


#use model Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# In[26]:


#use model Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[27]:


#use model Decision Tree Classifier with maximum depth of 3
clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))


# In[28]:


#use model Decision Tree Classifier with maximum depth of 4
clf2 = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))


# In[29]:


#use model k-neighbours
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[30]:


#use model Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


# In[31]:


#use model Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))


# In[32]:


#use model support vector machine
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


# In[51]:


Test_names = ['Logistic regression', 'Decision Tree', 'Decision Tree-max depth=3', 'Decision Tree-max depth=4', 'k-neighbors','Linear Discriminant','Gaussian Naive Bayes', 'Support vector machine']
Test_name_codes= ['LR', 'DT', 'DTdep3', 'DTdeP4', 'k-neig','LD','GNB', 'SVM']
training_accuracy = [0.85,1,0.82,0.87,0.85,0.87,0.61,0.86]
test_accuracy = [0.94,0.86,0.93,0.90,0.89,0.95,0.59,0.94]


# In[52]:


plt.scatter(Test_name_codes,training_accuracy,label='training')
plt.scatter(Test_name_codes,test_accuracy,label='test')
plt.xlabel('Name of the test')
plt.ylabel('Accuracy')
plt.annotate('Highest test accuracy', xy=('LD', 0.95), xytext=(1.5, 1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.legend()
plt.show()


# In[35]:


#use linear discriminant since it has the highest accuracy in test set and has second highest accuracy when considering the training test
#then select the 'lda' model


# In[36]:


original_test_data = pd.read_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/testdata_10%.csv')
test_data = original_test_data.copy()
test_data.head()


# In[37]:


test_data.__eq__('?').sum()


# In[38]:


#then in the test data set only A1 has missing values
print(test_data['A1'].value_counts())


# In[39]:


#replace the missing value with the mode.. here 'b'
test_data['A1'].replace('?','b', inplace=True)
print(test_data['A1'].value_counts())


# In[40]:


#data types of attributes
print(test_data.info())


# In[41]:


#convert data types int to float of some attributes
test_data["A12"]=test_data["A12"].astype(float)
test_data["A7"]=test_data["A7"].astype(float)
test_data["A14"]=test_data["A14"].astype(float)


# In[42]:


#label encoding of A1 as b=1 and a=0
test_data["A1"]= test_data["A1"].astype('category')
test_data['A1'] = test_data['A1'].cat.codes
test_data.head()


# In[43]:


#one-hot encoding for objects
test_data = pd.get_dummies(test_data, columns=['A3'], prefix=['A3'])
test_data = pd.get_dummies(test_data, columns=['A4'], prefix=['A4'])
test_data = pd.get_dummies(test_data, columns=['A6'], prefix=['A6'])
test_data = pd.get_dummies(test_data, columns=['A9'], prefix=['A9'])
test_data = pd.get_dummies(test_data, columns=['A15'], prefix=['A15'])


# In[44]:


test_data.head()


# In[45]:


print(test_data.info())


# In[46]:


#there are missing data columns fro the trained set
# Get missing columns in the training test
missing_cols = set( onehote_data.columns ) - set( test_data.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test_data[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_data = test_data[ onehote_data.columns]


# In[47]:


print(test_data.info())


# In[48]:


X_predict = scaler.transform(test_data[feature_names])


# In[49]:


#using linear discriminant analysis 'lda'
y_predict = lda.predict(X_predict)
print(y_predict)


# In[50]:


original_test_data.insert(15,"A16",y_predict,True)
original_test_data.head(14)


# In[ ]:





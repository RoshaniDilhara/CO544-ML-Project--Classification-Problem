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


data.replace('?',np.nan, inplace=True)


# In[4]:


print(data.info())


# In[5]:


data.__eq__(np.nan).sum()


# In[6]:


data.head(60)


# In[7]:


data.mean()


# In[8]:


data["A12"]=data["A12"].astype(float)
data["A7"]=data["A7"].astype(float)
data["A14"]=data["A14"].astype(float)
data["A2"]=data["A2"].astype(float)


# In[9]:


data.mean()


# In[10]:


print(data['A1'].value_counts())


# In[11]:


data['A1'].replace(np.nan,'b', inplace=True)
print(data['A1'].value_counts())


# In[12]:


data = data.fillna(data.mean())


# In[13]:


print(data['A3'].value_counts())


# In[14]:


data['A3'].replace(np.nan,'u', inplace=True)
print(data['A3'].value_counts())


# In[15]:


print(data['A6'].value_counts())


# In[16]:


data['A6'].replace(np.nan,'c', inplace=True)
print(data['A6'].value_counts())


# In[17]:


print(data['A9'].value_counts())


# In[18]:


data['A9'].replace(np.nan,'v', inplace=True)
print(data['A9'].value_counts())


# In[19]:


import seaborn as sns
sns.countplot(data['A16'],label="count")
plt.show()


# In[20]:


newer_data = data.copy()


# In[21]:


#one-hot encoding to object type columns
onehote_data = newer_data.copy()
onehote_data = pd.get_dummies(onehote_data, columns=['A3'], prefix=['A3'])
onehote_data = pd.get_dummies(onehote_data, columns=['A4'], prefix=['A4'])
onehote_data = pd.get_dummies(onehote_data, columns=['A6'], prefix=['A6'])
onehote_data = pd.get_dummies(onehote_data, columns=['A9'], prefix=['A9'])
onehote_data = pd.get_dummies(onehote_data, columns=['A15'], prefix=['A15'])


# In[22]:


onehote_data.head()


# In[23]:


#label encoding to column A1
onehote_data["A1"]= onehote_data["A1"].astype('category')
onehote_data['A1'] = onehote_data['A1'].cat.codes
onehote_data.head()


# In[24]:


print(onehote_data.info())


# In[25]:


feature_names = ['A1', 'A2', 'A5', 'A7','A8','A10','A11','A12','A13','A14','A3_l','A3_u','A3_y','A4_g','A4_gg','A4_p','A6_aa','A6_c','A6_cc','A6_d','A6_e','A6_ff','A6_i','A6_j','A6_k','A6_m','A6_q','A6_r','A6_w','A6_x','A9_bb','A9_dd','A9_ff','A9_h','A9_j','A9_n','A9_o','A9_v','A9_z','A15_g','A15_p','A15_s']
X = onehote_data[feature_names]
Y = onehote_data['A16']


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

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# In[29]:


#use model Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[30]:


#use model Decision Tree Classifier with maximum depth of 3
clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))


# In[31]:


#use model Decision Tree Classifier with maximum depth of 4
clf3 = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))


# In[32]:


#use model k-neighbours
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[33]:


#use model Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


# In[34]:


#use model Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))


# In[35]:


#use model support vector machine
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


# In[36]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rfc.fit(X_train, y_train)
print('Accuracy of RFC classifier on training set: {:.2f}'
     .format(rfc.score(X_train, y_train)))
print('Accuracy of RFC classifier on test set: {:.2f}'
     .format(rfc.score(X_test, y_test)))


# In[37]:


ABC = AdaBoostClassifier(n_estimators=100)
ABC.fit(X_train, y_train)
print('Accuracy of ABC classifier on training set: {:.2f}'
     .format(ABC.score(X_train, y_train)))
print('Accuracy of ABC classifier on test set: {:.2f}'
     .format(ABC.score(X_test, y_test)))


# In[38]:




from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Qda = QuadraticDiscriminantAnalysis()
Qda.fit(X_train, y_train)
print('Accuracy of QDA classifier on training set: {:.2f}'
     .format(Qda.score(X_train, y_train)))
print('Accuracy of QDA classifier on test set: {:.2f}'
     .format(Qda.score(X_test, y_test)))


# In[39]:


from sklearn.gaussian_process import GaussianProcessClassifier
GPC = GaussianProcessClassifier()
GPC.fit(X_train, y_train)
print('Accuracy of GPC classifier on training set: {:.2f}'
     .format(GPC.score(X_train, y_train)))
print('Accuracy of GPC classifier on test set: {:.2f}'
     .format(GPC.score(X_test, y_test)))


# In[40]:


svm2 = SVC(gamma=2, C=1)
svm2.fit(X_train, y_train)
print('Accuracy of svm2 classifier on training set: {:.2f}'
     .format(svm2.score(X_train, y_train)))
print('Accuracy of svm2 classifier on test set: {:.2f}'
     .format(svm2.score(X_test, y_test)))


# In[41]:


from sklearn.neural_network import MLPClassifier 
mlp = MLPClassifier(alpha=1, max_iter=1000)
mlp.fit(X_train, y_train)
print('Accuracy of svm2 classifier on training set: {:.2f}'
     .format(mlp.score(X_train, y_train)))
print('Accuracy of svm2 classifier on test set: {:.2f}'
     .format(mlp.score(X_test, y_test)))


# In[42]:


from sklearn.gaussian_process.kernels import RBF
gpc_rbf = GaussianProcessClassifier(1.0 * RBF(1.0))
gpc_rbf.fit(X_train, y_train)
print('Accuracy of svm2 classifier on training set: {:.2f}'
     .format(gpc_rbf.score(X_train, y_train)))
print('Accuracy of svm2 classifier on test set: {:.2f}'
     .format(gpc_rbf.score(X_test, y_test)))


# In[43]:


svm3 = SVC(kernel="linear", C=0.007)
svm3.fit(X_train, y_train)
print('Accuracy of svm3 classifier on training set: {:.2f}'
     .format(svm3.score(X_train, y_train)))
print('Accuracy of svm3 classifier on test set: {:.2f}'
     .format(svm3.score(X_test, y_test)))


# In[44]:


#use support vector machine since it has the highest accuracy in test set and has second highest accuracy when considering the training test
#then select the 'svm' model


# In[45]:


original_test_data = pd.read_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/testdata_10%.csv')
test_data = original_test_data.copy()
test_data.head()


# In[46]:


test_data.__eq__('?').sum()


# In[47]:


#then in the test data set only A1 has missing values
print(test_data['A1'].value_counts())


# In[48]:


#replace the missing value with the mode.. here 'b'
test_data['A1'].replace('?','b', inplace=True)
print(test_data['A1'].value_counts())


# In[49]:


#data types of attributes
print(test_data.info())


# In[50]:


#convert data types int to float of some attributes
test_data["A12"]=test_data["A12"].astype(float)
test_data["A7"]=test_data["A7"].astype(float)
test_data["A14"]=test_data["A14"].astype(float)


# In[51]:


#label encoding of A1 as b=1 and a=0
test_data["A1"]= test_data["A1"].astype('category')
test_data['A1'] = test_data['A1'].cat.codes
test_data.head()


# In[52]:


#one-hot encoding for objects
test_data = pd.get_dummies(test_data, columns=['A3'], prefix=['A3'])
test_data = pd.get_dummies(test_data, columns=['A4'], prefix=['A4'])
test_data = pd.get_dummies(test_data, columns=['A6'], prefix=['A6'])
test_data = pd.get_dummies(test_data, columns=['A9'], prefix=['A9'])
test_data = pd.get_dummies(test_data, columns=['A15'], prefix=['A15'])


# In[53]:


test_data.head()


# In[54]:


print(test_data.info())


# In[55]:


#there are missing data columns fro the trained set
# Get missing columns in the training test
missing_cols = set( onehote_data.columns ) - set( test_data.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test_data[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_data = test_data[ onehote_data.columns]


# In[56]:


print(test_data.info())


# In[57]:


X_predict = scaler.transform(test_data[feature_names])


# In[58]:


#using linear discriminant analysis 'svm'
y_predict = svm.predict(X_predict)
print(y_predict)


# In[59]:


y_pre_gpc = GPC.predict(X_predict)
print(y_pre_gpc)


# In[60]:


y_pre_gpc_RBF = gpc_rbf.predict(X_predict)
print(y_pre_gpc_RBF)


# In[61]:


y_pre_SVM3 = svm3.predict(X_predict)
print(y_pre_SVM3)


# In[62]:


tdata_ori = pd.read_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/testdata.csv');
tdata = tdata_ori.copy()
tdata.head()


# In[63]:


tdata.__eq__('?').sum()


# In[64]:


print(tdata.info())


# In[65]:


print(tdata['A1'].value_counts())


# In[66]:


tdata['A1'].replace('?','b', inplace=True)
print(tdata['A1'].value_counts())


# In[67]:


tdata.replace('?',np.nan, inplace=True)


# In[68]:


tdata["A2"]=tdata["A2"].astype(float)


# In[69]:


tdata["A12"]=tdata["A12"].astype(float)
tdata["A7"]=tdata["A7"].astype(float)
tdata["A14"]=tdata["A14"].astype(float)


# In[70]:


tdata.mean()


# In[71]:


tdata = tdata.fillna(tdata.mean())


# In[72]:


print(tdata['A3'].value_counts())


# In[73]:


tdata['A3'].replace(np.nan,'u', inplace=True)


# In[74]:


print(tdata['A4'].value_counts())


# In[75]:


tdata['A4'].replace(np.nan,'g', inplace=True)


# In[76]:


print(tdata['A6'].value_counts())


# In[77]:


tdata['A6'].replace(np.nan,'c', inplace=True)


# In[78]:


print(tdata['A9'].value_counts())


# In[79]:


tdata['A9'].replace(np.nan,'v', inplace=True)


# In[80]:


print(tdata.info())


# In[81]:


tdata["A1"]= tdata["A1"].astype('category')
tdata['A1'] = tdata['A1'].cat.codes
tdata.head()


# In[82]:


tdata = pd.get_dummies(tdata, columns=['A3'], prefix=['A3'])
tdata = pd.get_dummies(tdata, columns=['A4'], prefix=['A4'])
tdata = pd.get_dummies(tdata, columns=['A6'], prefix=['A6'])
tdata = pd.get_dummies(tdata, columns=['A9'], prefix=['A9'])
tdata = pd.get_dummies(tdata, columns=['A15'], prefix=['A15'])


# In[83]:


tdata.head()


# In[84]:


#there are missing data columns fro the trained set
# Get missing columns in the training test
missing_cols = set( onehote_data.columns ) - set( tdata.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    tdata[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
tdata = tdata[ onehote_data.columns]


# In[85]:


tdata.head()


# In[86]:


X_predict_tdata = scaler.transform(tdata[feature_names])


# In[87]:


y_predict_tData = svm.predict(X_predict_tdata)
print(y_predict_tData)


# In[88]:


dataset = pd.DataFrame({ 'Category': y_predict_tData})


# In[89]:


dataset.to_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/resultTestdata2.csv')


# In[90]:


print(dataset['Category'].value_counts())


# In[91]:


y_predictgpc_tData = GPC.predict(X_predict_tdata)
print(y_predictgpc_tData)


# In[92]:


dset = pd.DataFrame({ 'Category': y_predictgpc_tData})
dset.to_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/resultTestdatagpc2.csv')
print(dset['Category'].value_counts())


# In[93]:


y_predictgpcr_tData = gpc_rbf.predict(X_predict_tdata)
print(y_predictgpcr_tData)


# In[94]:


dset2 = pd.DataFrame({ 'Category': y_predictgpcr_tData})
dset2.to_csv('D:/Semester 6 - 3rd year/Machine Learning -CO544/resultTestdatagpcr2.csv')
print(dset2['Category'].value_counts())


# In[ ]:





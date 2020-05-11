#!/usr/bin/env python
# coding: utf-8

# In[1]:
import inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

# In[3]:


# load the dataset
dataset = pd.read_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/data.csv')
# print the rows of data
dataset.head()

# In[4]:


# checking for null values
print("Sum of NULL values in each column. ")
print(dataset.__eq__('?').sum())

# In[5]:


# No of values and features in the dataset
print(dataset.shape)

# In[6]:


print(dataset.info())

# In[7]:


# mark missing values as NaN
dataset[["A1", "A2", "A3", "A4", "A6", "A9", "A14"]] = dataset[["A1", "A2", "A3", "A4", "A6", "A9", "A14"]].replace("?",
                                                                                                                    np.NaN)

# In[8]:


# count the number of NaN values in each column
print(dataset.isnull().sum())

# In[9]:


# print the first 80 rows of data
print(dataset.head(80))

# In[10]:


from sklearn import linear_model
import missingno as mno

#realizing this missingness by visualizing the same using missingno package by drawing a nullity matrix
# mno.matrix(dataset, figsize = (20, 6))
# plt.show()

# In[11]:


missing_columns = ["A1", "A2", "A3", "A4", "A6", "A9", "A14"]

# In[12]:


bools_to_int = {"A11": {True: 1, False: 0},
                "A13": {True: 1, False: 0},
                "A8": {True: 1, False: 0}}

# In[13]:


dataset.replace(bools_to_int, inplace=True)
dataset.head()

# In[14]:


print(dataset.info())

# In[15]:


# change object type to float of some columns which the data type has been identified as object,but they are of type float
dataset["A2"] = dataset["A2"].astype(float)
dataset["A14"] = dataset["A14"].astype(float)

# In[16]:


# change object type to category
dataset["A1"] = dataset["A1"].astype('category')
# dataset["A2"]= dataset["A2"].astype('category')
dataset["A3"] = dataset["A3"].astype('category')
dataset["A4"] = dataset["A4"].astype('category')
dataset["A6"] = dataset["A6"].astype('category')
dataset["A9"] = dataset["A9"].astype('category')
# dataset["A14"]= dataset["A14"].astype('category')
dataset["A15"] = dataset["A15"].astype('category')
dataset["A16"] = dataset["A16"].astype('category')

dataset.dtypes

# In[17]:


# Label Encoding
dataset["A1"] = dataset["A1"].cat.codes
dataset["A3"] = dataset["A3"].cat.codes
dataset["A4"] = dataset["A4"].cat.codes
dataset["A6"] = dataset["A6"].cat.codes
dataset["A9"] = dataset["A9"].cat.codes
dataset["A15"] = dataset["A15"].cat.codes
dataset["A16"] = dataset["A16"].cat.codes

dataset.head(80)

# In[18]:


missing_columns = ["A1", "A2", "A3", "A4", "A6", "A9", "A14"]


# In[19]:


def random_imputation(dataset, feature):
    number_missing = dataset[feature].isnull().sum()
    observed_values = dataset.loc[dataset[feature].notnull(), feature]
    dataset.loc[dataset[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing,
                                                                                replace=True)

    return dataset


# In[20]:


for feature in missing_columns:
    dataset[feature + '_imp'] = dataset[feature]
    dataset = random_imputation(dataset, feature)

# In[21]:


# checking for null values
print("Sum of NULL values in each column. ")
print(dataset.__eq__('?').sum())

# In[22]:


# In Deterministic Regression Imputation, we replace the missing data with the values predicted in our regression model and repeat this process for each variable.

deter_data = pd.DataFrame(columns=["Det" + name for name in missing_columns])

for feature in missing_columns:
    deter_data["Det" + feature] = dataset[feature + "_imp"]
    parameters = list(set(dataset.columns) - set(missing_columns) - {feature + '_imp'})

    X = dataset[parameters]
    y = dataset[feature + '_imp']

    # Create a Linear Regression model to estimate the missing data
    model = linear_model.LinearRegression()
    model.fit(X, y)

    # observe that I preserve the index of the missing data from the original dataframe
    deter_data.loc[dataset[feature].isnull(), "Det" + feature] = model.predict(dataset[parameters])[
        dataset[feature].isnull()]

# In[23]:
# mno.matrix(deter_data, figsize = (20,5))
# plt.show()

dataset.head(80)

# In[24]:


for feature in missing_columns:
    dataset[feature + '_imp'] = deter_data['Det' + feature]

# In[25]:


dataset.head(80)

#see the balance of the dataset
# import seaborn as sns
# sns.countplot(dataset['A16'],label="Count")
# plt.show()

#Box plot for each numeric variable will give us a clearer idea of the distribution of the input variables:
# dataset.boxplot('A2_imp','A16',rot=30, figsize=(5,5))
# dataset.boxplot('A7','A16',rot=30, figsize=(5,5))
# dataset.boxplot('A12','A16',rot=30, figsize=(5,5))
# dataset.boxplot('A14_imp','A16',rot=30, figsize=(5,5))
# plt.show()

#Scatter-matrix for some input variable
# from pandas.plotting import scatter_matrix
# from matplotlib import cm
#
# feature_names = ["A2_imp","A7","A12","A14_imp"]
# X = dataset[feature_names]
# y = dataset['A16']
# cmap = cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':20}, figsize=(12,12), cmap = cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('scatter_matrix')
# plt.show()

# In[26]:


for feature in missing_columns:
    dataset[feature] = dataset[feature + '_imp']

# In[27]:


dataset.head(80)

# In[28]:


newer_dataset = dataset.copy()
newer_dataset.head(80)

# In[29]:


newfeature_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15"]
X = newer_dataset[newfeature_names]
Y = newer_dataset['A16']

# In[30]:


# split the data set as training set and test set randomly
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# In[31]:


# apply scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# In[32]:


# use model Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# use model Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)
# use model Decision Tree Classifier with maximum depth of 3
clf1 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
# use model Decision Tree Classifier with maximum depth of 4
clf2 = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
# use model k-neighbours
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# use model Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
# use model Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
# use model support vector machine
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

# use model Random forest

from sklearn import model_selection
rfr = RandomForestClassifier(n_estimators=100, random_state=0)
rfr.fit(X_train, y_train)



# In[33]:


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

# In[34]:


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

#################use our models to predict the resuls for new dataset#############


# In[36]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load the test dataset
test_dataset = pd.read_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/kaggle/testdata.csv')
test_dataset_copy = test_dataset.copy()
# print the rows of data
test_dataset_copy.head(80)

# In[37]:


test_dataset_copy.__eq__('?').sum()

# In[38]:


# mark missing values as NaN
test_dataset_copy[["A1", "A2", "A3", "A4", "A6", "A9", "A14"]] = test_dataset_copy[
    ["A1", "A2", "A3", "A4", "A6", "A9", "A14"]].replace("?", np.NaN)

# In[39]:


# count the number of NaN values in each column
print(test_dataset_copy.isnull().sum())

# In[40]:


# print the first 80 rows of data
print(test_dataset_copy.head(14))

# In[41]:


test_dataset_copy.info(14)

# In[42]:


# change bool type to numeric
bools_to_int_test = {"A11": {True: 1, False: 0},
                     "A13": {True: 1, False: 0},
                     "A8": {True: 1, False: 0}}
# change object type to float of some columns which the data type has been identified as object,but they are of type float
test_dataset_copy["A2"] = test_dataset_copy["A2"].astype(float)
test_dataset_copy["A14"] = test_dataset_copy["A14"].astype(float)

# In[43]:


test_dataset_copy.replace(bools_to_int_test, inplace=True)
test_dataset_copy.head()

# In[44]:


test_dataset_copy.info(14)

# In[45]:


# change object type to category
test_dataset_copy["A1"] = test_dataset_copy["A1"].astype('category')
test_dataset_copy["A3"] = test_dataset_copy["A3"].astype('category')
test_dataset_copy["A4"] = test_dataset_copy["A4"].astype('category')
test_dataset_copy["A6"] = test_dataset_copy["A6"].astype('category')
test_dataset_copy["A9"] = test_dataset_copy["A9"].astype('category')
test_dataset_copy["A15"] = test_dataset_copy["A15"].astype('category')
test_dataset_copy.dtypes

# In[46]:


# label encoding
test_dataset_copy["A1"] = test_dataset_copy["A1"].cat.codes
test_dataset_copy["A3"] = test_dataset_copy["A3"].cat.codes
test_dataset_copy["A4"] = test_dataset_copy["A4"].cat.codes
test_dataset_copy["A6"] = test_dataset_copy["A6"].cat.codes
test_dataset_copy["A9"] = test_dataset_copy["A9"].cat.codes
test_dataset_copy["A15"] = test_dataset_copy["A15"].cat.codes

# In[47]:


missing_columns_test = ["A1", "A2", "A3", "A4", "A6", "A9", "A14"]


# In[48]:


def random_imputation_test(test_dataset_copy, feature_test):
    number_missing_test = test_dataset_copy[feature_test].isnull().sum()
    observed_values_test = test_dataset_copy.loc[test_dataset_copy[feature_test].notnull(), feature_test]
    test_dataset_copy.loc[test_dataset_copy[feature_test].isnull(), feature_test + '_impTest'] = np.random.choice(
        observed_values_test, number_missing_test, replace=True)

    return test_dataset_copy


# In[49]:


for feature_test in missing_columns_test:
    test_dataset_copy[feature_test + '_impTest'] = test_dataset_copy[feature_test]
    test_dataset_copy = random_imputation_test(test_dataset_copy, feature_test)

# In[50]:


# checking for null values
print("Sum of NULL values in each column. ")
print(test_dataset_copy.__eq__('Nan').sum())

# In[51]:


# In Deterministic Regression Imputation, we replace the missing data with the values predicted in our regression model and repeat this process for each variable.

deter_data_test = pd.DataFrame(columns=["DetTest" + name for name in missing_columns_test])

for feature_test in missing_columns_test:
    deter_data_test["DetTest" + feature_test] = test_dataset_copy[feature_test + "_impTest"]
    parameters_test = list(set(test_dataset_copy.columns) - set(missing_columns_test) - {feature_test + '_impTest'})

    XT = test_dataset_copy[parameters_test]
    yT = test_dataset_copy[feature_test + '_impTest']

    # Create a Linear Regression model to estimate the missing data
    modelT = linear_model.LinearRegression()
    modelT.fit(XT, yT)

    # observe that I preserve the index of the missing data from the original dataframe
    deter_data_test.loc[test_dataset_copy[feature_test].isnull(), "DetTest" + feature_test] = \
    modelT.predict(test_dataset_copy[parameters_test])[test_dataset_copy[feature_test].isnull()]

# In[52]:


test_dataset_copy.head(14)

# In[53]:


for feature_test in missing_columns_test:
    test_dataset_copy[feature_test + '_impTest'] = deter_data_test['DetTest' + feature_test]

# In[54]:


for feature_test in missing_columns_test:
    test_dataset_copy[feature_test] = test_dataset_copy[feature_test + '_impTest']

# In[55]:


test_dataset_copy.head(14)

# In[56]:


X_predict = scaler.transform(test_dataset_copy[newfeature_names])

# In[57]:


# using support vector machine
y_predict = svm.predict(X_predict)
print(y_predict)

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
# y_predict = rfr.predict(X_predict)
# print(y_predict)


# In[58]:


# test_dataset.insert(15, "A16", y_predict, True) #for SVM
test_dataset.insert(15, "A16", y_predict, True)

# In[59]:


cleanup_predict = {"A16": {1: "Success", 0: "Failure"}}
# print(cleanup_predict)


# In[60]:


test_dataset.replace(cleanup_predict, inplace=True)

# In[62]:


test_dataset.head(20)

# In[ ]:
pd.options.display.max_rows
pd.set_option('display.max_rows', None)
# test_dataset.to_csv('G:/6TH SEM/CO544-Machine Learning and Data Mining/Project/kaggle/results1-RF-project.csv', index=False)



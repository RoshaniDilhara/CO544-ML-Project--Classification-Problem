#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#read the dataset 
dataset = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/data.csv',sep= ',') 


# In[3]:


print(dataset)


# In[4]:


print(dataset.__eq__('?').sum())


# In[5]:


#replace missing values with np.nan
dataset['A1'].replace('?',np.nan,inplace=True)
dataset['A2'].replace('?',np.nan,inplace=True)#numeric
dataset['A3'].replace('?',np.nan,inplace=True)
dataset['A4'].replace('?',np.nan,inplace=True)
dataset['A6'].replace('?',np.nan,inplace=True)
dataset['A9'].replace('?',np.nan,inplace=True)
dataset['A14'].replace('?',np.nan,inplace=True)#numeric

#change A2,A5,A7,A10,A12,A14 data type to float 
dataset['A14'] = dataset.A14.astype(float)
dataset['A2'] = dataset.A2.astype(float) 


# In[6]:


df = pd.DataFrame(dataset)


# In[7]:


df = pd.get_dummies(df,columns=['A1'],prefix=['A1'])
df= pd.get_dummies(df,columns=['A3'],prefix=['A3'])
df = pd.get_dummies(df,columns=['A4'],prefix=['A4'])
df = pd.get_dummies(df,columns=['A6'],prefix=['A6'])
df = pd.get_dummies(df,columns=['A9'],prefix=['A9'])
df = pd.get_dummies(df,columns=['A15'],prefix=['A15'])


# In[8]:


print(df)


# In[9]:


print(df.dtypes)


# In[10]:


# split data into X and y
feature_names = ['A2','A5','A7','A8','A10','A11','A12','A13','A14','A1_a','A1_b','A3_l','A3_u','A3_y','A4_g','A4_gg','A4_p','A6_aa','A6_c','A6_cc','A6_d','A6_e','A6_ff','A6_i','A6_j','A6_k','A6_m','A6_q','A6_r','A6_w','A6_x','A9_bb','A9_dd','A9_ff','A9_j','A9_n','A9_o','A9_v','A9_z','A15_g','A15_p','A15_s']
X = df[feature_names]
Y = df['A16']


# In[11]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size =test_size ,random_state=seed)


# In[12]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# In[13]:


print(model)


# In[14]:


# make predictions for test data
y_pred = model.predict(X_test)


# In[16]:


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[17]:


testset = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testdata.csv') 
print(testset.head())


# In[18]:


print ("Dataset Length: ", len(testset)) 
print ("Dataset Shape: ", testset.shape)


# In[19]:


tf = pd.DataFrame(testset)


# In[20]:


print(tf)


# In[21]:


#replace missing values with np.nan
tf['A1'].replace('?',np.nan,inplace=True)
tf['A2'].replace('?',np.nan,inplace=True)#numeric
tf['A3'].replace('?',np.nan,inplace=True)
tf['A4'].replace('?',np.nan,inplace=True)
tf['A5'].replace('?',np.nan,inplace=True)#numeric
tf['A6'].replace('?',np.nan,inplace=True)
tf['A7'].replace('?',np.nan,inplace=True)#numeric
tf['A8'].replace('?',np.nan,inplace=True)
tf['A9'].replace('?',np.nan,inplace=True)
tf['A10'].replace('?',np.nan,inplace=True)#numeric
tf['A11'].replace('?',np.nan,inplace=True)
tf['A12'].replace('?',np.nan,inplace=True)#numeric
tf['A13'].replace('?',np.nan,inplace=True)
tf['A14'].replace('?',np.nan,inplace=True)#numeric
tf['A15'].replace('?',np.nan,inplace=True)

#change A2,A5,A7,A10,A12,A14 data type to float 
# to change use .astype() 
tf['A2'] = tf.A2.astype(float) 
tf['A5'] = tf.A5.astype(float)
tf['A7'] = tf.A7.astype(float) 
tf['A10'] = tf.A10.astype(float) 
tf['A12'] = tf.A12.astype(float) 
tf['A14'] = tf.A14.astype(float)


# In[22]:


tf = pd.get_dummies(tf,columns=['A1'],prefix=['A1'])
tf= pd.get_dummies(tf,columns=['A3'],prefix=['A3'])
tf = pd.get_dummies(tf,columns=['A4'],prefix=['A4'])
tf = pd.get_dummies(tf,columns=['A6'],prefix=['A6'])
tf = pd.get_dummies(tf,columns=['A9'],prefix=['A9'])
tf = pd.get_dummies(tf,columns=['A15'],prefix=['A15'])


# In[23]:


print(tf.info())


# In[24]:


#handling missing columns
missing_cols = set(df.columns) - set(tf.columns)
print(missing_cols)


# In[25]:


for c in missing_cols:
    tf[c] = 0
    
tf = tf[df.columns]


# In[26]:


print(tf.info())


# In[27]:


test_pred = model.predict(tf[feature_names]) 


# In[28]:


print("Predicted values:") 
print(test_pred)


# In[29]:


tf['A16'] = test_pred #Final prediction on the test data set
print(tf)


# In[30]:


test_final = pd.read_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testdata.csv') 
test_final['A16'] = test_pred


# In[31]:


test_final_frame = pd.DataFrame(test_final)


# In[32]:


#Final prediction on the test set 
print(test_final_frame)


# In[33]:


test_final_frame.to_csv('E:/University Works/3rd Year/Semester 6/CO 544 - Machine Learning and Data Mining/Project/new/testresults8.csv',sep=',')


# In[ ]:





# In[ ]:





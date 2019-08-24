#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


# In[2]:


dataset=pd.read_csv('C:/Users/Vishnu/Desktop/data2.csv',header = None)


# In[3]:


target=pd.read_csv('C:/Users/Vishnu/Desktop/target1.csv',header = None)


# In[4]:


features=pd.read_csv('C:/Users/Vishnu/Desktop/headers.csv',header = None)


# In[5]:


data=np.array(dataset)


# In[6]:


data


# In[7]:


target=np.array(target)


# In[8]:


features=np.array(features)


# In[9]:


import sklearn.datasets


# In[10]:


datasets=sklearn.datasets.base.Bunch(data=data,target=target,feature_names=features)


# In[11]:


features1 = pd.DataFrame(datasets.data, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque',
         'i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding',
         'profile_id'])


# In[12]:


target1=pd.DataFrame(datasets.target)


# In[13]:


x=features1


# In[14]:


y=target1


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(X_test), len(y_test))


# In[17]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[18]:


rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)


# In[19]:


rr100 = Ridge(alpha=100)
rr100.fit(X_train, y_train)


# In[20]:


rr100 = Ridge(alpha=10000)
rr100.fit(X_train, y_train)


# In[21]:


rr100 = Ridge(alpha=0.00000001)
rr100.fit(X_train, y_train)


# In[22]:


train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)


# In[23]:


Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)


# In[24]:


Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)


# In[25]:


Ridge_train_score10000 = rr100.score(X_train,y_train)
Ridge_test_score10000 = rr100.score(X_test, y_test)


# In[26]:


Ridge_train_score001 = rr100.score(X_train,y_train)
Ridge_test_score001 = rr100.score(X_test, y_test)


# In[29]:


print ("linear regression train score:", train_score)
print ("linear regression test score:", test_score)
print()
print ("ridge regression train score low alpha:", Ridge_train_score)
print ("ridge regression test score low alpha:", Ridge_test_score)
print()
print ("ridge regression train score high alpha:", Ridge_train_score100)
print ("ridge regression test score high alpha:", Ridge_test_score100)
print()
print ("ridge regression train score alpha10000:", Ridge_train_score10000)
print ("ridge regression test score alpha10000:", Ridge_test_score10000)
print()
print ("ridge regression train score alpha001:", Ridge_train_score001)
print ("ridge regression test score alpha001:", Ridge_test_score001)


# In[ ]:





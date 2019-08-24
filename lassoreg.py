#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import sklearn.datasets


# In[2]:


dataset=pd.read_csv('C:/Users/Vishnu/Desktop/data2.csv',header = None)


# In[3]:


target=pd.read_csv('C:/Users/Vishnu/Desktop/target1.csv',header = None)


# In[4]:


features=pd.read_csv('C:/Users/Vishnu/Desktop/headers.csv',header = None)


# In[5]:


data=np.array(dataset)


# In[6]:


target=np.array(target)


# In[7]:


features=np.array(features)


# In[8]:


import sklearn.datasets


# In[9]:


datasets=sklearn.datasets.base.Bunch(data=data,target=target,feature_names=features)


# In[10]:


features1 = pd.DataFrame(datasets.data, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque',
         'i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding',
         'profile_id'])


# In[11]:


target1=pd.DataFrame(datasets.target)


# In[12]:


x=features1


# In[13]:


y=target1


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(X_test), len(y_test))


# In[15]:


lr = Lasso(alpha=0.01)
lr.fit(X_train, y_train)


# In[16]:


lr100=Lasso(alpha=0.09)
lr100.fit(X_train,y_train)


# In[17]:


Lasso_train_score = lr.score(X_train,y_train)
Lasso_test_score =lr.score(X_test, y_test)
Lasso_train_score100 = lr100.score(X_train,y_train)
Lasso_test_score100 = lr100.score(X_test, y_test)


# In[18]:


print ("Lasso regression train score low alpha:", Lasso_train_score)
print ("ridge regression test score low alpha:", Lasso_test_score)
print ("ridge regression train score high alpha:", Lasso_train_score100)
print ("ridge regression test score high alpha:", Lasso_test_score100)


# In[ ]:




